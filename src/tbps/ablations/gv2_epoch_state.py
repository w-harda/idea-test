"""扩展训练的来源、完整状态与事务保存；不包含攻击算法。"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import copy,fcntl,hashlib,io,json,os,random,sys,time
import numpy as np
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest,sha_file

POINTS=(10,20,30,40,50,60)
PARENT_SHA="f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e"
CONFIG_SHA="7e87bd272abfee440afc13109d439507a34b140506a19a132a8f740afe093bc4"
STEPS_PER_EPOCH=8068

def need(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    return FileRef(str(Path(path).resolve()),sha_file(path))

def tree_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def verify_parent_reference(value):
    r=FileRef(**value);need(r.sha256==PARENT_SHA,"only original Epoch10 SHA is allowed");checked(r)

def config(path):
    c=read_json(path,"tbps-gv2-epochs-1");c["path"]=str(Path(path).resolve())
    need(c["points"]==list(POINTS) and c["target_epochs"]==60 and c["seed"]==42,"fixed six checkpoints/seed42")
    verify_parent_reference(c["parent_checkpoint"])
    need(c["base_config_ref"]["sha256"]==CONFIG_SHA,"original epochs10 config SHA")
    for r in c["legacy_refs"].values():checked(FileRef(**r))
    root=Path(__file__).resolve().parents[3]
    required={str(root/p) for p in ("src/tbps/ablations/__init__.py","src/tbps/ablations/gv2_epochs.py",
        "src/tbps/ablations/gv2_epoch_state.py","src/tbps/artifacts.py","src/tbps/contracts.py",
        "src/tbps/models/irra_source.py","src/tbps/engines/replay_control.py")}
    need(set(c["producers"])==required,"complete extension producer set")
    for p,h in c["producers"].items():checked(FileRef(p,h))
    base=Path(c["output_base"]).resolve()
    need(base.is_relative_to("/home/lzf/ldx/outputs") and base.name=="tbps-framework-gv2-epoch60-ablation","isolated extension root")
    need(Path(c["formal_output"]).resolve()==base/"train" and Path(c["smoke_output"]).resolve()==base/"resume_smoke_v1","separate formal/smoke outputs")
    for name in ("formal_output","smoke_output"):
        p=Path(c[name]).resolve()
        need(not any(p.is_relative_to(Path(x).resolve()) for x in c["protected_roots"]),"output overlaps historical assets")
    return c

def legacy(c):
    root=Path(c["legacy_root"]).resolve()
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    import epoch_training as et,epoch_checkpoint as ck,cache_io as oldio
    for module,name in ((et,"epoch_training.py"),(ck,"epoch_checkpoint.py"),(oldio,"cache_io.py")):
        need(Path(module.__file__).resolve()==root/name,"legacy module collision")
    return et,ck,oldio

def extension_contract(original,c):
    effective=copy.deepcopy(original["config"]);effective["epochs"]=60
    return {"schema":"gv2-epoch-extension-contract-1","original_training_contract":copy.deepcopy(original),
        "original_contract_sha256":digest(original),"parent_checkpoint":copy.deepcopy(c["parent_checkpoint"]),
        "effective_training_configuration":effective,"original_epochs":10,"target_epochs":60,"start_epoch":11,
        "evaluation_points":list(POINTS),"formal_output":str(Path(c["formal_output"]).resolve()),
        "smoke_output":str(Path(c["smoke_output"]).resolve()),
        "allowed_configuration_changes":{"epochs":{"from":10,"to":60}},
        "extension_configuration":asdict(ref(c["path"])),
        "trajectory_claim":"same algorithm and restored epoch-boundary state; no historical CUDA bitwise trajectory claim"}

def validate_contract(contract):
    need(contract["schema"]=="gv2-epoch-extension-contract-1","explicit extension schema")
    old=contract["original_training_contract"]
    need(digest(old)==contract["original_contract_sha256"] and old["config"]["epochs"]==10,"immutable original contract")
    effective=copy.deepcopy(contract["effective_training_configuration"])
    need(effective["epochs"]==60,"fixed target60")
    effective["epochs"]=10
    need(effective==old["config"] and contract["allowed_configuration_changes"]=={"epochs":{"from":10,"to":60}},"only epoch horizon may differ")
    need(contract["original_epochs"]==10 and contract["start_epoch"]==11 and contract["target_epochs"]==60
        and contract["evaluation_points"]==list(POINTS),"absolute epoch/points")
    need(contract["parent_checkpoint"]["sha256"]==PARENT_SHA,"original parent SHA")
    need(Path(contract["formal_output"]).resolve()!=Path(contract["smoke_output"]).resolve(),"smoke cannot publish formal epochs")

def base_state(c,*,check_cache=True):
    et,ck,oldio=legacy(c);verify_parent_reference(c["parent_checkpoint"])
    for r in c["legacy_refs"].values():checked(FileRef(**r))
    old=read_json(c["base_contract"])
    need(old["config"]==read_json(c["base_config_ref"]["path"]),"original config/contract mismatch")
    need(old["config"]["epochs"]==10 and old["config"]["algorithm"]=="A","formal algorithm A")
    need(old["config"]["architecture"]=={"width":32,"blocks":4,"epsilon":8/255}
        and old["config"]["optimizer"]=={"lr":2e-4,"betas":[.5,.999]}
        and old["config"]["weights"]=={"lambda_global":1.,"lambda_local":1.,"lambda_retain":1.}
        and old["config"]["budget_epsilon"]==8/255,"frozen architecture/loss/Adam/budget")
    need(old["runtime"]=={"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__},"original runtime required")
    oldio.verify(old["training_source_sha256"]);oldio.verify(old["config"]["entry_source_sha256"])
    value=ck.verify_version(Path(c["parent_checkpoint"]["path"]).parent,old)
    need(value["completed_epoch"]==10 and value["next_epoch"]==11,"Epoch10 -> absolute Epoch11")
    need(adam_steps(value["optimizer"])==[80680],"complete original Adam step80680")
    need(set(value["rng"])=={"python","numpy","torch","cuda"} and len(value["rng"]["cuda"])==1,"all four RNG domains")
    finite_state(value["optimizer"])
    m=None
    if check_cache:
        m=et.checked_cache(c["cache"],require_full=True)
        need(et.contract_for(m,c["cache"],old["config"])==old,"cache/data/source contract unchanged")
    need(not torch.cuda.is_initialized(),"CPU preflight initialized CUDA")
    contract=extension_contract(old,c);validate_contract(contract)
    return contract,value,m

def adam_steps(state):
    if hasattr(state,"state_dict"):state=state.state_dict()
    return sorted({int(s["step"]) for s in state["state"].values()})

def finite_state(value):
    if torch.is_tensor(value):need(bool(torch.isfinite(value).all()),"nonfinite model/Adam state")
    elif isinstance(value,dict):
        for v in value.values():finite_state(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite_state(v)

def restore(value,g,opt,ck):
    g.load_state_dict(value["generator"],strict=True);opt.load_state_dict(value["optimizer"])
    finite_state(opt.state_dict());ck.restore_rng(value["rng"])
    checks={"G_exact":tree_hash(g.state_dict())==tree_hash(value["generator"]),
        "Adam_exact":tree_hash(opt.state_dict())==tree_hash(value["optimizer"]),
        "RNG_exact":tree_hash(ck.rng_state())==tree_hash(value["rng"])}
    need(all(checks.values()),"G/Adam/RNG restore mismatch")
    return {**checks,"Adam_steps":adam_steps(opt),"G_sha256":tree_hash(g.state_dict()),
        "Adam_sha256":tree_hash(opt.state_dict()),"RNG_sha256":tree_hash(ck.rng_state()),
        "RNG_domains":{k:tree_hash(v) for k,v in ck.rng_state().items()},
        "optimizer_groups":[{k:v for k,v in group.items() if k!="params"} for group in opt.state_dict()["param_groups"]]}

def validate_progress(p,*,formal=True):
    rows=p["sample_logs"]
    need(len(rows)==p["samples"] and [r["row_id"] for r in rows]==p["sample_order"],"sample identity/order")
    need(sum(r["optimizer_steps"] for r in rows)==p["optimizer_steps"],"step total")
    need(len(set(p["sample_order"]))==len(rows),"duplicate training sample")
    for pos,r in enumerate(rows,1):
        need(r["sample_position"]==pos and r["role"]=="train" and r["epoch"]==p["epoch"]
            and len(r["rounds"])==r["K"]+1==r["optimizer_steps"],"K+1 schedule/epoch")
        need([t["phase"] for t in r["rounds"]]==["attribute"]*r["K"]+["joint"],"attribute order and last Joint")
        for t in r["rounds"]:
            need(t["finite_gradients"] and 0<=t["cumulative_linf"]<=8/255,"finite gradient/strict budget")
            need(all(np.isfinite(t[k]) for k in ("global","local","retain","total","G_gradient_l2")),"finite loss")
    need(all(r["optimizer_steps"]==0 and r["cumulative_linf"]<=8/255 for r in p["validation"]),"validation separation")
    if formal:need(p["samples"]==2048 and len(p["validation"])==128 and p["optimizer_steps"]==8068,"formal 2048/128/8068")

def tensor_bytes(value):
    b=io.BytesIO();torch.save(value,b);return b.getvalue()

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]

def pack(g,opt,epoch,contract,progress,ck):
    validate_contract(contract)
    return {"schema":"gv2-epoch-extension-checkpoint-1","completed_epoch":epoch,"next_epoch":epoch+1,
        "contract":contract,"contract_sha256":digest(contract),"generator":ck.cpu_tree(g.state_dict()),
        "optimizer":ck.cpu_tree(opt.state_dict()),"rng":ck.rng_state(),"progress":progress}

def checked_log(value,store):
    p=value["progress"];path=Path(p["sample_log_path"]).resolve()
    need(path.is_relative_to(store.root.parent.resolve()),"log outside independent training root")
    need(sha_file(path)==p["sample_log_sha256"] and read_rows(path)==p["sample_logs"],"associated sample log SHA/content")
    return ref(path)

def save_checkpoint(store,value,*,formal=True,fault=None):
    validate_contract(value["contract"]);validate_progress(value["progress"],formal=formal)
    epoch=value["completed_epoch"]
    need(value["schema"]=="gv2-epoch-extension-checkpoint-1" and 11<=epoch<=60
        and value["next_epoch"]==epoch+1 and value["progress"]["epoch"]==epoch,"completed boundary only")
    need(value["contract_sha256"]==digest(value["contract"]),"extension digest")
    if formal:need(adam_steps(value["optimizer"])==[epoch*8068],"cumulative Adam")
    finite_state(value["generator"]);finite_state(value["optimizer"]);log=checked_log(value,store)
    key=f"epoch_{epoch:04d}"
    store.publish(key,{"checkpoint.pt":tensor_bytes(value),"checkpoint.json":json_bytes({
        "epoch":epoch,"contract_sha256":value["contract_sha256"],"parent_checkpoint":value["contract"]["parent_checkpoint"]})},
        {"kind":"gv2-continuous-epoch","epoch":epoch,"contract_sha256":value["contract_sha256"]},
        sources=(log,FileRef(**value["contract"]["parent_checkpoint"]),FileRef(**value["contract"]["extension_configuration"])),fault=fault)
    return ref(store.root/key/"checkpoint.pt")

def load_checkpoint(store,epoch,contract,*,formal=True):
    validate_contract(contract);key=f"epoch_{epoch:04d}"
    store.verify(key,expected_provenance={"kind":"gv2-continuous-epoch","epoch":epoch,"contract_sha256":digest(contract)})
    value=torch.load(store.root/key/"checkpoint.pt",map_location="cpu",weights_only=False)
    need(value["schema"]=="gv2-epoch-extension-checkpoint-1" and value["contract"]==contract
        and value["contract_sha256"]==digest(contract) and value["completed_epoch"]==epoch
        and value["next_epoch"]==epoch+1 and value["progress"]["epoch"]==epoch,"checkpoint identity/source")
    validate_progress(value["progress"],formal=formal);checked_log(value,store)
    if formal:need(adam_steps(value["optimizer"])==[epoch*8068],"cumulative saved Adam")
    finite_state(value["generator"]);finite_state(value["optimizer"]);return value

def restore_boundary(root,contract,*,resume=False,formal=True):
    root=Path(root)
    if not root.exists():need(not resume,"no previous extension to resume");return None
    need(resume,"existing training directory requires explicit resume")
    store=ArtifactStore(root,new_base=root.parent)
    need(not store.pending() and not (root/"completed").exists(),"pending/completed session; no implicit restart")
    store.verify("protocol")
    need(read_json(root/"protocol/protocol.json")==contract,"resume protocol mismatch")
    cs=ArtifactStore(root/"checkpoints",new_base=root.parent)
    need(not cs.pending(),"uncommitted checkpoint preserved; inspect failure")
    published=sorted(int(p.name[6:]) for p in cs.root.glob("epoch_[0-9][0-9][0-9][0-9]"))
    need(bool(published) and published==list(range(11,published[-1]+1)),"only contiguous fully committed Epochs")
    need(not (root/f"claim_{published[-1]+1:04d}").exists(),"incomplete Epoch updates: preserve; no automatic replay")
    for e in published:
        store.verify(f"claim_{e:04d}")
        v=load_checkpoint(cs,e,contract,formal=formal)
    return v

@contextmanager
def session_lock(base):
    base=Path(base);base.mkdir(parents=True,exist_ok=True)
    with (base/".training.lock").open("a") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError("another extension process holds training lock")
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def snapshot(c):
    states={}
    for root in c["protected_roots"]:
        for p in Path(root).rglob("*"):
            if p.is_file():
                s=p.stat();states[str(p)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
    sha={p:sha_file(p) for p in c["legacy_refs_paths"]}
    return {"stats":states,"sha256":sha}

def check_snapshot(value):
    for p,expected in value["stats"].items():
        s=Path(p).stat()
        need([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==expected,"historical file changed: "+p)
    for p,h in value["sha256"].items():checked(FileRef(p,h))
    return {"files_stat_unchanged":len(value["stats"]),"files_sha256_unchanged":len(value["sha256"])}

def summarize(p,seconds,lr):
    rows=[r for s in p["sample_logs"] for r in s["rounds"]]
    def terms(rs):
        return {k:{"mean":sum(r[k] for r in rs)/len(rs),"min":min(r[k] for r in rs),"max":max(r[k] for r in rs)}
            for k in ("global","local","retain","total","G_gradient_l2","cumulative_linf","budget_saturation_ratio","strict_projection_clipping_ratio")} if rs else {}
    val=p["validation"]
    return {"absolute_epoch":p["epoch"],"samples":p["samples"],"steps":p["optimizer_steps"],
        "train":terms(rows),"attribute":terms([r for r in rows if r["phase"]=="attribute"]),
        "joint":terms([r for r in rows if r["phase"]=="joint"]),"validation":{
            "global":sum(r["global"] for r in val)/len(val),"local":sum(r["local"] for r in val)/len(val),"retain":0.,
            "total":sum(r["global"]+r["local"] for r in val)/len(val)},
        "zero_gradient_rounds":sum(r["G_gradient_l2"]==0 for r in rows),"lr":lr,"seconds":seconds,
        "global_is_paired_image_caption_dot":True,"Test_feedback":False}
