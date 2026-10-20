"""六节点只读审计与G图像变体；只接线冻结实现，不包含攻击/训练算法。"""
from pathlib import Path
from dataclasses import asdict,replace
from contextlib import contextmanager
from unittest.mock import patch
import argparse,csv,io,json,os,subprocess,time,traceback,resource
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest,sha_file
from tbps.materialization.store import image_checks
from tbps.materialization.bulk_session import BulkSession
from tbps.engines.materialized_inputs import load_inputs
from tbps.baselines.comparison import FullReference,four
from tbps.attacks.legacy_runtime import legacy_imports
from .gv2_epoch_state import tree_hash

EPOCHS=(10,20,30,40,50,60)
ARCH={"width":32,"blocks":4,"epsilon":8/255}
KEYS=("R@1","R@5","R@10","mAP")
def need(ok,message):
    if not ok:raise ValueError(message)
def ref(p):return FileRef(str(Path(p).resolve()),sha_file(p))
def config(path):
    c=read_json(path,"gv2-six-epoch-evaluation-1");c["path"]=str(Path(path).resolve())
    need(list(c["points"])==[str(e) for e in EPOCHS],"six fixed ordered Epochs")
    need(c["models"]==["irra","rasa","aptm"] and c["precision"]=="fp32/math/no_tf32" and c["seed"]==42
         and c["image_batch"]==1 and c["query_chunk"]==128 and c["metrics"]==list(KEYS),"frozen evaluation protocol")
    for d in c["points"].values():checked(FileRef(**d))
    for d in c["sources"].values():checked(FileRef(**d))
    for p,h in c["producers"].items():checked(FileRef(p,h))
    evidence=read_json(c["sources"]["training_evidence"]["path"],"gv2-epoch60-completed-results-1")
    need(c["points"]==evidence["points"] and evidence["audit"]["status"]=="PASS"
         and evidence["audit"]["epochs"]==50 and evidence["audit"]["Adam_step"]==484080
         and evidence["completed"]["status"]=="PASS","published six-node training evidence")
    root=Path(__file__).resolve().parents[3]
    expected={str(p) for p in (root/"src/tbps").rglob("*.py")}
    need(set(c["producers"])==expected,"complete fixed framework runtime sources")
    base=Path(c["output_base"]).resolve()
    need(base.name=="tbps-framework-gv2-epoch-sweep" and base.parent==Path("/home/lzf/ldx/outputs"),"independent output root")
    return c

def select(manifest,c,scope):
    if scope=="full":return manifest
    need(scope=="smoke","explicit scope")
    gids=c["smoke"]["gallery_ids"];qids=c["smoke"]["query_ids"]
    g=tuple(x for x in manifest.gallery if x.image_id in gids)
    q=tuple(x for x in manifest.queries if x.query_id in qids)
    need([x.image_id for x in g]==gids and [x.query_id for x in q]==qids and len(g)==len(q)==10,"fixed official ordered smoke members")
    need(all(x.paired_image_id in gids for x in q),"smoke positive coverage")
    return replace(manifest,gallery=g,queries=q)

def generator_state(value,epoch,original,parent):
    need(epoch in EPOCHS,"not an evaluation node")
    need(value["completed_epoch"]==epoch and value["next_epoch"]==epoch+1,"checkpoint Epoch identity")
    contract=value["contract"]
    need(value["contract_sha256"]==digest(contract),"checkpoint contract digest")
    if epoch==10:
        need(value["schema"]=="gv2-epoch-checkpoint-1" and contract==original,"original Epoch10 contract")
    else:
        need(value["schema"]=="gv2-epoch-extension-checkpoint-1"
             and contract["original_training_contract"]==original and contract["parent_checkpoint"]==parent
             and contract["original_contract_sha256"]==digest(original)
             and contract["start_epoch"]==11 and contract["target_epochs"]==60,"extension parent/source")
        expected=dict(original["config"]);expected["epochs"]=60
        need(contract["effective_training_configuration"]==expected,"only epoch horizon changed")
    need(original["config"]["architecture"]==ARCH,"original image-only architecture")
    # 整个已验SHA容器由torch.load读入；只取G，不构造/恢复任何Optimizer。
    return value["generator"]

def inspect_weights(c,legacy,epochs=EPOCHS):
    original=read_json(c["sources"]["original_contract"]["path"]);report={}
    def forbid(*a,**k):raise ValueError("CPU weight inspection forbids CUDA/forward")
    with legacy_imports(legacy,allow_import=True) as old,patch.object(torch.cuda,"_lazy_init",forbid),patch.object(torch.nn.Module,"_call_impl",forbid):
        klass=old("image_only_generator").ImageOnlyGenerator
        old("inference_once") # 实际传递导入闭包；旧load_generator不支持扩展schema，保持其原样。
        for epoch in epochs:
            point=c["points"][str(epoch)];checked(FileRef(**point))
            value=torch.load(point["path"],map_location="cpu",weights_only=False)
            weights=generator_state(value,epoch,original,c["points"]["10"])
            model=klass(**ARCH);model.load_state_dict(weights,strict=True)
            report[str(epoch)]={"checkpoint":point,"G_state_sha256":tree_hash(weights),
                "architecture":ARCH,"strict_state_dict_load":True,"only_generator_used":True}
            del model,weights,value
    need(not torch.cuda.is_initialized(),"CPU preflight CUDA side effect")
    return report

def native_cpu(c,model=None):
    def forbid(*a,**k):raise ValueError("native preflight cannot deserialize victim/init CUDA")
    with patch.object(torch,"load",forbid),patch.object(torch.cuda,"_lazy_init",forbid):
        for name in ((model,) if model else c["models"]):
            if name=="irra":
                from tbps.models.irra_source import IRRASource
                IRRASource.from_dict(read_json(c["inputs"]["dataset_sources"]["path"])["irra"]).verify()
            else:
                mod=__import__("tbps.models."+name,fromlist=["native_modules"])
                source=(mod.RasaSource if name=="rasa" else mod.APTMSource)(c["victim_sources"][name]["path"])
                checked(FileRef(**c["victim_sources"][name]))
                with mod.native_modules(source) as (_,tok,utils):
                    tokenizer=mod.official_tokenizer(tok,source);limit=50 if name=="rasa" else 56
                    ids=tokenizer([utils.pre_caption("A man in a blue shirt.",limit)],padding="max_length",
                        truncation=True,max_length=limit,return_tensors="pt").input_ids
                    need(ids.shape==(1,limit),"native tokenizer boundary")

def prepare(c,scope,*,weights=False,model=None):
    torch.set_num_threads(2)
    native_cpu(c,model)
    ctx=load_inputs(c["inputs"],"full")
    ctx["manifest"]=select(ctx["plan"].manifest,c,scope);ctx["c"]=c;ctx["report"]["scope"]=scope
    points=inspect_weights(c,ctx["legacy"]) if weights else {}
    references={name:FullReference(c,ctx,name) for name in ((model,) if model else c["models"])}
    for name,reference in references.items():
        need(reference.protocol["model_space"]==c["history_spaces"][name],"historical native ModelSpace")
        for mode in ("Clean","Text-only","Image-only","Joint"):four(reference.summary["groups"][mode])
        audit=c["history_audits"][name];checked(FileRef(**audit["log"]))
        need(audit["marker"] in Path(audit["log"]["path"]).read_text(),"existing historical CPU audit PASS")
    need(ctx["plan"].g.checkpoint==FileRef(**c["points"]["10"])
         and ctx["plan"].h.checkpoint.sha256==c["H_checkpoint_sha256"],"frozen G10/H7 provenance")
    ctx.update(references=references)
    report={"status":"GV2_SWEEP_CPU_PREFLIGHT_PASS","scope":scope,"Gallery":len(ctx["manifest"].gallery),
        "Query":len(ctx["manifest"].queries),"full_Gallery":3074,"full_Query":6156,"points":points,
        "epoch10_all_four_modes_reusable":True,"clean_and_H_text_features_reusable":True,
        "history_models_verified":list(references),"history_plan":ctx["plan"].id,
        "H_regeneration":False,"optimizer_restored":False,"CUDA":False,
        "input_bridge":"sealed FP32 RGB and Raw Caption; no PIL requantization",
        "fixed_smoke":c["smoke"]}
    need(not torch.cuda.is_initialized(),"preflight initialized CUDA")
    ctx["preflight"]=report
    return ctx

def csv_metrics(groups,rounded=False):
    s=io.StringIO(newline="");w=csv.writer(s,lineterminator="\n");w.writerow(("Attack Method",*KEYS))
    for mode,values in groups.items():w.writerow((mode,*(format(values[k],".2f") if rounded else values[k] for k in KEYS)))
    return s.getvalue().encode()

def paths(c,scope,epoch):
    need(scope in ("smoke","full") and epoch in EPOCHS,"scope/node")
    base=Path(c["output_base"]);root=base/("smoke_v1" if scope=="smoke" else "full")/f"epoch_{epoch:04d}"
    return root,base

class VariantStore:
    """旧Plan只读引用，新Epoch独立身份；不修改MaterializationPlan/旧MaterializedStore。"""
    def __init__(self,root,base,protocol):
        self.store=ArtifactStore(root,new_base=base);self.protocol=json.loads(json_bytes(protocol))
        need(protocol["epoch"] in EPOCHS and protocol["no_target_feedback"] is True,"variant identity")
        ids=protocol["ordered_gallery"]
        need(len(ids)==len(set(ids)) and ids,"ordered unique Gallery")
        self.ids=tuple(ids)
    def publish_protocol(self,sources):
        self.store.publish("protocol",{"protocol.json":json_bytes(self.protocol)},{"kind":"gv2-epoch-pixels"},sources=sources)
    @property
    def protocol_ref(self):return ref(self.store.root/"protocol/manifest.json")
    def verify(self):
        self.store.verify("protocol")
        need(read_json(self.store.root/"protocol/protocol.json")==self.protocol and not self.store.pending(),"variant protocol/pending")
    def key(self,index):
        need(type(index)is int and 0<=index<len(self.ids),"Gallery offset")
        return f"image-{index:06d}"
    def claim(self,index):
        key=self.key(index);claim=f"claim-{index:06d}"
        need(not (self.store.root/key).exists() and not (self.store.root/claim).exists()
             and not self.store.pending(),"existing/unfinished image Claim; no automatic replay")
        self.store.publish(claim,{"claim.json":json_bytes({"index":index,"image_id":self.ids[index],
            "epoch":self.protocol["epoch"],"one_G_forward":True})},{"kind":"pre-G-claim"},sources=(self.protocol_ref,))
    def publish(self,index,arrays,base_unit,*,fault=None):
        key=self.key(index);claim=f"claim-{index:06d}";self.store.verify(claim)
        need(type(arrays)is dict and set(arrays)=={"original","adversarial","effective_delta"},"actual RGB required")
        arrays={k:x.detach().cpu().clone() for k,x in arrays.items()}
        stats=image_checks(arrays["original"],arrays["adversarial"],arrays["effective_delta"],(384,128))
        data=io.BytesIO();torch.save(arrays,data)
        md={"image_id":self.ids[index],"index":index,"epoch":self.protocol["epoch"],"pixels":stats,
            "base_unit":asdict(base_unit),"G_forwards":1,"G_checkpoint":self.protocol["checkpoint"],
            "target_feedback":False}
        self.store.publish(key,{"pixels.pt":data.getvalue(),"metadata.json":json_bytes(md)},
            {"kind":"gv2-epoch-image","epoch":self.protocol["epoch"],"image_id":self.ids[index]},
            sources=(self.protocol_ref,ref(self.store.root/claim/"manifest.json"),base_unit),fault=fault)
        loaded,_=self.load(index)
        need(all(torch.equal(loaded[k],v) for k,v in arrays.items()),"saved RGB readback not bitwise exact")
        return stats
    def load(self,index):
        key=self.key(index);self.store.verify(key,expected_provenance={"kind":"gv2-epoch-image",
            "epoch":self.protocol["epoch"],"image_id":self.ids[index]})
        self.store.verify(f"claim-{index:06d}")
        md=read_json(self.store.root/key/"metadata.json")
        need(md["G_forwards"]==1 and md["G_checkpoint"]==self.protocol["checkpoint"]
             and md["index"]==index and md["image_id"]==self.ids[index],"image source/proof")
        raw=torch.load(self.store.root/key/"pixels.pt",map_location="cpu",weights_only=True)
        need(type(raw)is dict and set(raw)=={"original","adversarial","effective_delta"},"pixel payload keys")
        arrays={k:v.detach().clone() for k,v in raw.items()};del raw
        need(image_checks(arrays["original"],arrays["adversarial"],arrays["effective_delta"],(384,128))==md["pixels"],"pixel content SHA/budget")
        return arrays,md

def variant(ctx,scope,epoch):
    c=ctx["c"];root,base=paths(c,scope,epoch)
    protocol={"schema":"gv2-epoch-rgb-variant-1","epoch":epoch,"checkpoint":c["points"][str(epoch)],
        "historical_plan":ctx["plan"].id,"base_collection":asdict(ctx["core"]),
        "manifest_sha256":digest(asdict(ctx["manifest"])),"ordered_gallery":[g.image_id for g in ctx["manifest"].gallery],
        "H_checkpoint":asdict(ctx["plan"].h.checkpoint),"H_text_reuse":True,"image_size":[384,128],
        "budget":8/255,"original_input":"sealed original FP32 RGB (no PIL roundtrip)","no_target_feedback":True,
        "config":asdict(ref(c["path"]))}
    return VariantStore(root/"images",base,protocol)

def resources(c,scope="smoke"):
    from tbps.engines.materialized_irra import gpu_available
    gpu=gpu_available({"physical_gpu":0})
    need(gpu["free_mib"]>=c["minimum_free_mib"],"GPU memory insufficient; no retry")
    need(gpu["utilization_percent"]<=c["maximum_initial_utilization_percent"],
         "GPU active task utilization exceeds launch policy; stop without waiting/interference")
    mem=next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
    disk=os.statvfs("/home/lzf/ldx/outputs")
    need(mem>=c["minimum_ram_bytes"] and disk.f_bavail*disk.f_frsize>=c["minimum_disk_bytes"][scope],"RAM/disk insufficient")
    return dict(GPU=gpu,RAM_available_bytes=mem,disk_free_bytes=disk.f_bavail*disk.f_frsize,
        free_inodes=disk.f_favail,policy="shared_observe_only; no waiting/interference")

def generate(c,scope,epoch,*,resume=False):
    need(scope=="smoke" or epoch!=10,"Full Epoch10 must reuse historical result")
    ctx=prepare(c,scope);v=variant(ctx,scope,epoch)
    root=v.store.root
    need(resume or not root.exists(),"existing generation; explicit resume required")
    need(not (root/"failure").exists() and not (root/"completed").exists() and not v.store.pending(),"failure/completed/pending: preserve")
    if root.exists():v.verify()
    else:v.publish_protocol((ref(c["path"]),FileRef(**c["points"][str(epoch)]),ctx["core"]))
    # 未完成Claim在模型加载前拒绝；完整已提交单元才可按显式resume跳过。
    for i in range(len(v.ids)):
        if (root/f"claim-{i:06d}").exists():need((root/v.key(i)).exists(),"unfinished Claim; cannot replay forward")
    gpu=resources(c,scope);torch.manual_seed(42);ctx["gate"].initialize_cuda()
    counts={"G_loads":0,"G_forwards":0,"H_loads":0,"H_forwards":0,"victim_loads":0,"backward":0,"optimizer":0,"JVP":0}
    begin=time.monotonic();report={"status":"RUNNING","resources":gpu,"scope":scope,"epoch":epoch,"counts":counts}
    try:
        from tbps.models.irra import fp32_runtime
        with fp32_runtime(),legacy_imports(ctx["legacy"],allow_import=True) as old:
            original=read_json(c["sources"]["original_contract"]["path"])
            point=FileRef(**c["points"][str(epoch)]);checked(point)
            value=torch.load(point.path,map_location="cpu",weights_only=False)
            weights=generator_state(value,epoch,original,c["points"]["10"])
            with torch.inference_mode(False):
                model=old("image_only_generator").ImageOnlyGenerator(**ARCH).to("cuda:0")
                model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False)
            del value,weights
            fn=old("inference_once").generate_once;counts["G_loads"]=1;before=tree_hash(model.state_dict())
            def observe(module,args):
                need(len(args)==1 and len(args[0])==1 and not torch.is_grad_enabled(),"image-only single frozen forward")
                counts["G_forwards"]+=1
            hook=model.register_forward_pre_hook(observe);maximum=0.;errors=[];skipped=0
            processing=time.monotonic()
            with BulkSession(ctx["store"]),torch.inference_mode():
                for i,row in enumerate(ctx["manifest"].gallery):
                    if (root/v.key(i)).exists():
                        _,md=v.load(i);maximum=max(maximum,md["pixels"]["linf"]);skipped+=1;continue
                    arrays,_=ctx["store"].load(ctx["images"][row.image_id])
                    original_rgb=arrays["original"].detach().clone()
                    v.claim(i);before_count=counts["G_forwards"]
                    result=fn(model,original_rgb.to("cuda:0"))
                    need(counts["G_forwards"]==before_count+1,"one G per image")
                    saved={"original":original_rgb,"adversarial":result["adversarial"].cpu().clone(),
                           "effective_delta":result["effective_delta"].cpu().clone()}
                    if epoch==10:
                        err=float((saved["adversarial"]-arrays["adversarial"]).abs().max())
                        need(err<=1e-6,"Epoch10 differs from historical RGB; stop");errors.append(err)
                    stats=v.publish(i,saved,ctx["images"][row.image_id].manifest)
                    maximum=max(maximum,stats["linf"])
                    del arrays,saved,original_rgb,result
                    if (i+1)%32==0 or i+1==len(v.ids):print("GV2_VARIANT_PROGRESS",epoch,i+1,"/",len(v.ids),flush=True)
            hook.remove()
            need(tree_hash(model.state_dict())==before and all(p.grad is None and not p.requires_grad for p in model.parameters()),"G frozen")
            report.update(status="PASS",images=len(v.ids),skipped=skipped,max_linf=maximum,
                processing_seconds=time.monotonic()-processing,model_before=before,model_after=before,
                epoch10_historical_max_abs_error=max(errors) if errors else None,
                H_checkpoint=asdict(ctx["plan"].h.checkpoint))
            need(counts["G_forwards"]+skipped==len(v.ids),"actual G count")
        v.store.publish("completed",{"report.json":json_bytes(report)},{"kind":"complete-epoch-Gallery"},
            sources=(v.protocol_ref,*(ref(root/v.key(i)/"manifest.json") for i in range(len(v.ids)))))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
        v.store.publish("failure",{"report.json":json_bytes(report)},{"kind":"generation-failure"},sources=(v.protocol_ref,))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        v.store.publish("execution",{"report.json":json_bytes(report)},{"kind":"variant-generation-execution"},sources=(v.protocol_ref,))
    print("GV2_VARIANT_GENERATION_PASS",scope,epoch,json.dumps(report),flush=True)

def audit_variant(ctx,scope,epoch):
    v=variant(ctx,scope,epoch);v.verify()
    need(not (v.store.root/"failure").exists(),"failed pixels")
    v.store.verify("completed");v.store.verify("execution")
    report=read_json(v.store.root/"execution/report.json")
    need(report["status"]=="PASS" and report["counts"]["G_loads"]==1
         and report["counts"]["G_forwards"]+report["skipped"]==len(v.ids),"generation actual count")
    for i in range(len(v.ids)):v.load(i)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--action",choices=("preflight","generate","target","audit","smoke"),default="preflight")
    p.add_argument("--scope",choices=("smoke","full"),default="smoke");p.add_argument("--epoch",type=int,choices=EPOCHS)
    p.add_argument("--model",choices=("irra","rasa","aptm"));p.add_argument("--allow-smoke",action="store_true")
    p.add_argument("--allow-full-materialization",action="store_true");p.add_argument("--allow-full-evaluation",action="store_true")
    p.add_argument("--resume",action="store_true");a=p.parse_args();c=config(a.config)
    if a.action=="preflight":
        ctx=prepare(c,a.scope,weights=True);print(json.dumps(ctx["preflight"],ensure_ascii=False),flush=True);return
    if a.action=="smoke":
        need(a.scope=="smoke" and a.allow_smoke and not a.resume,"one explicit Smoke")
        from .gv2_sweep_target import smoke
        return smoke(c)
    if a.action=="generate":
        need(a.epoch is not None and (a.allow_smoke if a.scope=="smoke" else a.allow_full_materialization),"scope-specific generation authorization")
        return generate(c,a.scope,a.epoch,resume=a.resume)
    from .gv2_sweep_target import target,audit
    need(a.epoch is not None and a.model is not None,"explicit epoch/model")
    if a.action=="audit":return audit(c,a.scope,a.epoch,a.model)
    need(a.allow_smoke if a.scope=="smoke" else a.allow_full_evaluation,"scope-specific evaluation authorization")
    return target(c,a.scope,a.epoch,a.model,resume=a.resume)
