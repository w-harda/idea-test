"""G60/H10完整只读恢复；CPU门禁禁止前向、优化和CUDA。"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
from .contracts import require
from .state import clone_tree, tree_sha, snapshot, capture_rng, restore_rng
from .native_sources import native_session, bindings
from tbps.artifacts import checked, FileRef, read_json, digest, ArtifactStore

@contextmanager
def no_optimizer_steps():
    import torch
    def stop(*args,**kwargs):
        raise ValueError("optimizer update forbidden in numerical probe/CPU restore")
    with ExitStack() as stack:
        seen=set()
        for value in vars(torch.optim).values():
            if isinstance(value,type) and issubclass(value,torch.optim.Optimizer) and value not in seen:
                seen.add(value);stack.enter_context(patch.object(value,"step",stop))
        yield

@contextmanager
def cpu_restore_only(paths):
    import torch
    allowed={str(Path(p).resolve()) for p in paths}
    original=torch.load;events={"loads":[],"forwards":0,"optimizer_steps":0,"CUDA":False}
    require(not torch.cuda.is_initialized(),"CPU restore must precede CUDA")
    def load(path,*args,**kwargs):
        require(isinstance(path,(str,Path)) and str(Path(path).resolve()) in allowed,
                "CPU restore attempted non-parent weight load")
        require(kwargs.get("map_location")=="cpu","CPU-only parent deserialization")
        events["loads"].append(str(Path(path).resolve()))
        return original(path,*args,**kwargs)
    def stop(*a,**k):
        raise ValueError("CPU parent restore forbids forward/derivative/CUDA")
    with no_optimizer_steps(), ExitStack() as stack:
        stack.enter_context(patch.object(torch,"load",load))
        for owner,name in ((torch.nn.Module,"_call_impl"),(torch.cuda,"init"),(torch.cuda,"_lazy_init"),
                           (torch.Tensor,"backward"),(torch.autograd,"grad"),(torch.autograd,"backward")):
            stack.enter_context(patch.object(owner,name,stop))
        yield events
    require(not torch.cuda.is_initialized(),"CPU restore initialized CUDA")

def normalized_rng(g,h):
    require(set(g)=={"python","numpy","torch","cuda"},"complete original G RNG")
    require(set(h)=={"python","numpy","torch_cpu","sample_generator","torch_cuda"},"complete original H RNG")
    require(len(g["cuda"])==len(h["torch_cuda"])==1,"fixed one-device parent RNG")
    return clone_tree(dict(python=g["python"],numpy=g["numpy"],torch_cpu=g["torch"],
                           cuda=list(g["cuda"]),sampler=h["sample_generator"])), clone_tree(
        dict(python=h["python"],numpy=h["numpy"],torch_cpu=h["torch_cpu"],
             cuda=list(h["torch_cuda"]),sampler=h["sample_generator"]))

@dataclass
class ParentBundle:
    g: object
    h: object
    g_adam: object
    h_adam: object
    sampler: object
    state: dict
    bindings: object
    modules: dict
    report: dict
    probe: object
    frontend: object

def restore_cpu_parents(c):
    """不加载IRRA/reader，不调用G/H；仅在新CPU对象读回完整状态。"""
    import torch
    from tbps.ablations.gv2_sweep import generator_state
    from tbps.ablations.gv2_epoch_state import finite_state,adam_steps
    p=c["prototype"]
    gref,href=c["sources"]["G60"],c["sources"]["H10"]
    for ref in (gref,href):
        checked(FileRef(**ref))
    caller_sampler=torch.Generator()
    caller_rng=capture_rng(caller_sampler)
    try:
        with native_session(c) as session:
            with cpu_restore_only(()):
                b=bindings(session)
            with cpu_restore_only((gref["path"],href["path"])) as events:
                gp=Path(gref["path"])
                store=ArtifactStore(gp.parent.parent,new_base=gp.parents[3])
                store.verify("epoch_0060")
                gvalue=torch.load(gp,map_location="cpu",weights_only=False)
                original=read_json(c["sources"]["original_G_contract"]["path"])
                weights=generator_state(gvalue,60,original,gvalue["contract"]["parent_checkpoint"])
                require(gvalue["completed_epoch"]==60 and gvalue["next_epoch"]==61
                        and adam_steps(gvalue["optimizer"])==[484080],"G60 complete Adam/epoch")
                architecture=original["config"]["architecture"]
                require(architecture=={"width":32,"blocks":4,"epsilon":8/255},"original G(I) architecture")
                g=b.get("image_only_generator").ImageOnlyGenerator(**architecture)
                g.load_state_dict(weights,strict=True)
                ga=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
                ga.load_state_dict(gvalue["optimizer"])
                require(tree_sha(g.state_dict())==tree_sha(weights)
                        and tree_sha(ga.state_dict())==tree_sha(gvalue["optimizer"]),"G/Adam restore exact")
                hp=Path(href["path"]);md=read_json(hp.with_name("checkpoint.json"))
                protocol=read_json(Path(p["history_root"])/"training/protocol.json")
                ck=b.get("formal_training.checkpoint")
                hvalue,_=ck.verify(hp.parent,protocol,md["parent_checkpoint_sha256"],
                                  md["start_cursor"],md["start_updates"])
                require(hvalue["cursor"]==20480 and hvalue["updates"]==20370
                        and hvalue["reason"]=="epoch_complete","original complete H10 required")
                native_tokenizer=b.get("utils.simple_tokenizer").SimpleTokenizer(
                    bpe_path=c["sources"]["bpe"]["path"])
                probe=b.get("tokenizer_probe").TokenizerProbe(native_tokenizer)
                h=b.get("policy").CharacterPolicy(probe,**protocol["config"]["architecture"])
                h.load_state_dict(hvalue["policy"],strict=True)
                ha=b.get("phase4_training").optimizer_for(h,protocol["config"])
                ha.load_state_dict(hvalue["optimizer"])
                require(tree_sha(h.state_dict())==tree_sha(hvalue["policy"])
                        and tree_sha(ha.state_dict())==tree_sha(hvalue["optimizer"]),"H/Adam restore exact")
                finite_state(ga.state_dict());finite_state(ha.state_dict())
                sampler=torch.Generator();sampler.set_state(hvalue["rng"]["sample_generator"])
                gr,hr=normalized_rng(gvalue["rng"],hvalue["rng"])
                for rng in (gr,hr):
                    cpu={**rng,"cuda":None}
                    restore_rng(cpu,sampler)
                    require(tree_sha(capture_rng(sampler))==tree_sha(cpu),"parent CPU RNG restore exact")
                sampler.set_state(hr["sampler"])
                parents={"G":gref["sha256"],"H":href["sha256"],"protocol":digest(p)}
                state=snapshot(g,h,ga,ha,g_rng=gr,h_rng=hr,
                    cursor={"round":0,"g_updates":484080,"h_visits":20480,"h_updates":20370},parents=parents)
                frontend=b.get("text_frontend").CaptionFrontend()
                report={"status":"GH_NATIVE_PARENT_CPU_RESTORE_PASS","G_state":tree_sha(g.state_dict()),
                    "H_state":tree_sha(h.state_dict()),"G_Adam":tree_sha(ga.state_dict()),
                    "H_Adam":tree_sha(ha.state_dict()),"G_Adam_steps":adam_steps(ga),
                    "H_Adam_step_min":min(adam_steps(ha)),"H_Adam_step_max":max(adam_steps(ha)),
                    "G_RNG":tree_sha(gr),"H_RNG":tree_sha(hr),"CPU_RNG_exact":True,
                    "CUDA_RNG_preserved":True,"CUDA_RNG_restored":False,
                    "parent_protocol":parents,"guards":events,"real_model_forwards":0,"updates":0}
            bundle=ParentBundle(g,h,ga,ha,sampler,state,b,session.modules,report,probe,frontend)
        # session.modules在退出上下文时才完成收集。
        bundle.modules=session.modules
        require(len(events["loads"])==2,"only G60/H10 loads permitted")
        return bundle
    finally:
        restore_rng(caller_rng,caller_sampler)


def bind_g_device(bundle,device):
    """先移动模型，再绑定新Adam对象并读回完整原状态，避免Parameter换对象导致悬空引用。"""
    import torch
    g=bundle.g.to(device)
    optimizer=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
    optimizer.load_state_dict(clone_tree(bundle.state["G_Adam"]))
    require([id(x) for group in optimizer.param_groups for x in group["params"]]
            ==[id(x) for x in g.parameters()],"G Adam ownership after device move")
    require(tree_sha(optimizer.state_dict())==tree_sha(bundle.state["G_Adam"]),"full G Adam changed after device bind")
    require(tree_sha(g.state_dict())==tree_sha(bundle.state["G"]),"G changed after device bind")
    return g,optimizer
