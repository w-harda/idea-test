"""B3三受害者接线：复用原生Adapter/缓存/Engine；只导出四项指标。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager,ExitStack
import argparse,csv,gc,io,json,os,time,traceback
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest
from tbps.models.base import ModelSpace,InputBinding,RepresentationBatch
from tbps.engines.provenance import BoundRepresentation,AttackOrigin,payload_digest
from tbps.engines.evaluation import EvaluationEngine
from tbps.engines.materialized_cache import FeatureCache,CachedIRRA,ref
from tbps.engines.native_cache import NativeFeatureCache,compact
from tbps.engines.materialized_inputs import load_inputs
from tbps.engines.materialized_irra import bindings,gpu_available
from tbps.materialization.bulk_session import BulkSession
from tbps.contracts import AttackBudget,ImageBudget
from tbps.smoke_irra import model_snapshot
from tbps.baselines.ap_b3 import HistoricalB3,B3Pixels,need,SHA

KEYS=("R@1","R@5","R@10","mAP")
METHODS=("Clean","AP-Attack B3","G-v2 Image-only")
BUDGET=AttackBudget(image=ImageBudget("linf",8/255,"rgb_0_1"))

def four(values):
    need(all(k in values and type(values[k]) in (int,float) and 0<=values[k]<=100 for k in KEYS),"four percent metrics")
    return {k:values[k] for k in KEYS}

def table_bytes(groups,rounded=False):
    out=io.StringIO(newline="");writer=csv.writer(out);writer.writerow(("Attack Method",*KEYS))
    need(tuple(groups)==METHODS,"fixed B3 comparison order")
    for method in METHODS:writer.writerow((method,*(format(groups[method][k],".2f") if rounded else groups[method][k] for k in KEYS)))
    return out.getvalue().encode()

def configuration(path):
    c=read_json(path,"tbps-b3-comparison-1")
    need(c["models"]==["irra","rasa","aptm"] and c["image_batch"]==c["fusion_batch"]==1
         and c["query_chunk"]==128 and c["seed"]==42 and c["precision"]=="fp32/math/no_tf32"
         and c["metrics"]==list(KEYS),"fixed native B3 comparison")
    need(c.get("gpu_policy")=="shared_observe_only" and c.get("minimum_free_mib")==3072,"fixed shared GPU policy")
    need(ref(Path(__file__).with_name("run_b3_full.sh")).sha256==c.get("launcher_sha256"),"bound B3 pipeline source changed")
    root=Path(__file__).resolve().parents[3]
    expected={str(p) for d in ("models","engines","baselines","materialization","attacks","datasets") for p in (root/"src/tbps"/d).rglob("*.py")}
    expected|={str(root/"src/tbps"/n) for n in ("b3_eval.py","artifacts.py","contracts.py","config.py","runtime_closure.py","smoke_irra.py")}
    need(set(c["producers"])==expected,"complete explicit new runtime sources")
    for p,h in c["producers"].items():checked(FileRef(p,h))
    return c

def payload_select(x,indices):
    if isinstance(x,dict):return {k:payload_select(v,indices) for k,v in x.items()}
    return x[indices].detach().clone()

def payload_cat(parts):
    if isinstance(parts[0],dict):return {k:payload_cat([x[k] for x in parts]) for k in parts[0]}
    return torch.cat(parts).detach().clone()

class FullReference:
    """同一受害者的原Full只读桥；不允许跨空间或在子集冒用Full指标。"""
    def __init__(self,c,ctx,model):
        self.root=Path(c["references"][model]["root"])
        for key,d in c["references"][model]["files"].items():checked(FileRef(**d))
        self.store=ArtifactStore(self.root,new_base=self.root.parent)
        for key in ("protocol","completed","result","execution-000001"):self.store.verify(key)
        need(not self.store.pending(),"pending historical victim output")
        self.protocol=read_json(self.root/"protocol/protocol.json")
        self.summary=read_json(self.root/"completed/summary.json")
        need(self.summary==read_json(self.root/"result/metrics.json"),"historical summary/metrics conflict")
        need(self.protocol["manifest"]==json.loads(json_bytes(asdict(ctx["plan"].manifest)))
             and self.protocol["collection"]==asdict(ctx["core"]) and self.protocol["scope"]=="full","same sealed Full inputs and official order")
        self.space=ModelSpace(**self.protocol["model_space"])
        need(self.space.model_id==model,"cross-victim cache forbidden")
        report=read_json(self.root/"execution-000001/report.json")
        need(report["status"]=="PASS" and report["model_before"]==report["model_after"],"historical frozen execution")
        expected,_=bindings({**ctx,"manifest":ctx["plan"].manifest},self.space)
        need(self.protocol["inputs"]==json.loads(json_bytes({k:asdict(v) for k,v in expected.items()})),"old input SHA/order/Plan differ")
        self.bound=expected;self.model=model
        self.protocol_ref=ref(self.root/"protocol/manifest.json")
        if model=="irra":self.cache=FeatureCache(self.store,self.protocol,self.space,chunk=64)
        else:
            validate=__import__("tbps.models."+model,fromlist=["validate_payload"]).validate_payload
            self.cache=NativeFeatureCache(self.store,self.protocol,self.space,validate,chunk=16)

    def read(self,family,ctx,*,origin=None):
        old=self.bound[family];selected=ctx["manifest"].gallery if family.endswith("gallery") else ctx["manifest"].queries
        ids=tuple(x.image_id if family.endswith("gallery") else x.query_id for x in selected)
        pos={key:i for i,key in enumerate(old.ordered_ids)}
        need(all(key in pos for key in ids),"missing historical selected member")
        # 按所选顺序只加载需要的批次；每个返回Tensor都clone，不保留完整底层Storage。
        parts=[];sources=[];current=None;loaded=None
        for key in ids:
            i=pos[key];start=i//self.cache.chunk*self.cache.chunk
            if current!=start:
                loaded=self.cache.load(family,old,start,min(start+self.cache.chunk,len(old.ordered_ids)))
                sources.append(ref(self.root/self.cache.key(family,start)/"manifest.json"));current=start
            parts.append(payload_select(loaded,[i-start]))
        value=payload_cat(parts);del parts,loaded
        hashes=tuple(old.ordered_sha256[pos[key]] for key in ids)
        b=InputBinding(self.space.id,digest(asdict(ctx["manifest"])),hashes,old.modality,ids)
        originals=tuple(g.image_sha256 for g in ctx["manifest"].gallery) if origin else ()
        return BoundRepresentation(RepresentationBatch(value,self.space.id,old.modality),b,payload_digest(value),tuple(dict.fromkeys(sources)),origin,originals)

def preflight(path,scope,model=None):
    need(scope in ("smoke","full"),"fixed scope")
    c=configuration(path)
    # 完整导入/Tokenizer CPU检查先于CPUStartupGate快照，禁止任何真实权重反序列化。
    def forbidden(*a,**k):raise RuntimeError("B3 preflight forbids model deserialization/CUDA")
    with patch.object(torch,"load",forbidden),patch.object(torch.cuda,"_lazy_init",forbidden):
        for name in ([model] if model else c["models"]):
            if name in ("rasa","aptm"):
                mod=__import__("tbps.models."+name,fromlist=["native_modules"])
                source=(mod.RasaSource if name=="rasa" else mod.APTMSource)(c["victim_sources"][name]["path"])
                checked(FileRef(**c["victim_sources"][name]));source.verify()
                with mod.native_modules(source) as (_,tok,utils):
                    tokenizer=mod.official_tokenizer(tok,source)
                    limit=50 if name=="rasa" else 56
                    need(tokenizer([utils.pre_caption("A man in a blue shirt.",limit)],padding="max_length",
                        truncation=True,max_length=limit,return_tensors="pt").input_ids.shape==(1,limit),"official text tokenizer")
    ctx=load_inputs(c["inputs"],scope);ctx["c"]=c
    hist=HistoricalB3(c["b3"],ctx["plan"].manifest);ctx["history"]=hist
    references={name:FullReference(c,ctx,name) for name in ([model] if model else c["models"])}
    ctx["references"]=references
    need(not torch.cuda.is_initialized(),"CPU preflight CUDA side effect")
    print("B3_CPU_PREFLIGHT_PASS",scope,json.dumps({"Gallery":len(ctx["manifest"].gallery),"Query":len(ctx["manifest"].queries),
        "B3_checkpoint_sha256":SHA,"historical_chunks_verified":len(hist.refs),"G_H_loads":0,"CUDA":False}),flush=True)
    return ctx

def pixels(ctx,scope):
    c=ctx["c"];root=Path(c["output_base"])/scope/"pixels"
    protocol={"schema":"tbps-B3-pixels-1","manifest":asdict(ctx["manifest"]),"generation_v3":asdict(ctx["core"]),
        "B3":c["b3"],"config":c["producers"],"mapping":"original transport_delta(native_adv-native_clean), unchanged",
        "budget":asdict(BUDGET),"target_feedback":False,"new_G_H_forwards":0}
    protocol=json.loads(json_bytes(protocol))
    store=B3Pixels(root,protocol);store.publish_protocol((FileRef(**c["b3"]["gallery"]),FileRef(**c["b3"]["metadata"])))
    need(not store.store.pending(),"pending B3 pixels; preserve and inspect")
    indices={g.image_id:i for i,g in enumerate(ctx["plan"].manifest.gallery)}
    maximum=0.
    with BulkSession(ctx["store"]):
        for count,row in enumerate(ctx["manifest"].gallery,1):
            if (root/store.key(row.image_id)).exists():arrays,md=store.load(row)
            else:
                original,_=ctx["store"].load(ctx["images"][row.image_id])
                arrays,stats,source=ctx["history"].pixels(indices[row.image_id],original["original"])
                del original
                arrays,md=store.publish(row,arrays,source)
            maximum=max(maximum,md["pixels"]["linf"]);del arrays
            if count%100==0 or count==len(ctx["manifest"].gallery):print("B3_PIXEL_PROGRESS",count,flush=True)
    collection={"status":"B3_PIXELS_READY","count":len(ctx["manifest"].gallery),"ordered_ids":[g.image_id for g in ctx["manifest"].gallery],
        "max_linf":maximum,"new_G_H_forwards":0,"native_cache_reused":True}
    if not (root/"collection").exists():store.store.publish("collection",{"collection.json":json_bytes(collection)},
        {"kind":"B3_pixels_collection"},sources=(store.protocol_ref,))
    store.store.verify("collection");need(read_json(root/"collection/collection.json")==collection,"B3 collection changed")
    return store

@contextmanager
def load_model(c,model,scope):
    if model=="irra":
        from tbps.models.irra import IRRAAdapter
        from tbps.models.irra_source import IRRASource
        source=IRRASource.from_dict(read_json(c["inputs"]["dataset_sources"]["path"])["irra"])
        a=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
        counts={"image_samples":0,"text_samples":0,"fusion_pairs":0}
        def observed(fn,kind):
            def call(x):counts[kind]+=len(x);return fn(x)
            return call
        with patch.object(a._backend,"encode_image",observed(a._backend.encode_image,"image_samples")),patch.object(a._backend,"encode_text",observed(a._backend.encode_text,"text_samples")):
            yield a,a._backend,source.checkpoint,counts
    else:
        mod=__import__("tbps.models."+model,fromlist=["load_"+model])
        source=(mod.RasaSource if model=="rasa" else mod.APTMSource)(c["victim_sources"][model]["path"])
        loader=mod.load_rasa if model=="rasa" else mod.load_aptm
        with loader(source,smoke=scope=="smoke") as a:yield a,a.backend,source.checkpoint,a.counts

def new_gallery(ctx,model,adapter,store,b3pixels,origin,protocol):
    m=ctx["manifest"]
    hashes=tuple(b3pixels.load(row)[1]["pixels"]["adversarial_pixels_sha256"] for row in m.gallery)
    b=InputBinding(adapter.space.id,digest(asdict(m)),hashes,"image",tuple(g.image_id for g in m.gallery))
    if model=="irra":cache=FeatureCache(store,protocol,adapter.space,chunk=16)
    else:
        validate=__import__("tbps.models."+model,fromlist=["validate_payload"]).validate_payload
        cache=NativeFeatureCache(store,protocol,adapter.space,validate,chunk=16)
    for start in range(0,len(m.gallery),16):
        end=min(start+16,len(m.gallery))
        if (store.root/cache.key("adv_gallery",start)).exists():value=cache.load("adv_gallery",b,start,end)
        else:
            rows=[]
            for row in m.gallery[start:end]:
                x,_=b3pixels.load(row)
                if model=="irra":value=adapter.inference().encode_images(x["adversarial"].to("cuda:0")).payload.cpu()
                else:value=adapter.encode_images(x["adversarial"])
                rows.append(compact(value))
            value=payload_cat(rows);del rows
            cache.publish("adv_gallery",b,start,end,value)
        del value
        print("B3_FEATURE_PROGRESS",model,end,"/",len(m.gallery),flush=True)
    return cache.gather("adv_gallery",b,origin=origin,originals=tuple(g.image_sha256 for g in m.gallery))

def evaluate_one(ctx,adapter,checkpoint,model,inputs,store,reference,method):
    from tbps.engines.native_score_cache import ChunkCachedScorer
    from tbps.engines.aptm_score_cache import APTMSetScorer
    protocol_ref=ref(store.root/"protocol/manifest.json")
    if model=="irra":
        scorer=CachedIRRA(adapter.space,checkpoint,cache_protocol=reference.protocol_ref)
        execution="readonly_cached"
    else:
        scorer=(ChunkCachedScorer if model=="rasa" else APTMSetScorer)(adapter,inputs,store,protocol_ref)
        execution="materialized_native"
    mode="Clean" if method=="Clean" else "Image-only"
    engine=EvaluationEngine(ctx["manifest"],scorer,direction="higher_better",execution=execution,model_sources=(checkpoint,))
    p,r,scores=engine.run(inputs,modes=(mode,),query_chunk=128)
    # 复用既有Ranking，发布严格四指标；不发布通用Engine内部的额外指标/下降率。
    value=four(r["groups"][mode]);stream=io.BytesIO();torch.save(scores[mode].detach().clone(),stream)
    store.publish("result",{"protocol.json":json_bytes(p),"metrics.json":json_bytes(value),"score.pt":stream.getvalue()},
        {"kind":"B3_four_metrics","method":method},sources=(protocol_ref,))
    del scores
    return value

def other_b3_processes():
    """只检查B3执行者，不把Ollama或其他计算进程本身当作冲突。"""
    import subprocess
    own={os.getpid(),os.getppid()};found=[]
    rows=subprocess.check_output(["ps","-eo","pid=,args="],text=True,timeout=10)
    for line in rows.splitlines():
        fields=line.strip().split(None,1)
        if len(fields)!=2 or not fields[0].isdigit():continue
        pid,command=int(fields[0]),fields[1]
        if pid in own:continue
        if ("run_b3_full.sh" in command or
            ("tbps.baselines.launch" in command and "--action run" in command) or
            ("tbps.b3_eval" in command and "--action run" in command) or
            "run_ap_cuhk_irra_surrogate_baseline.py" in command):
            found.append({"pid":pid,"command":command})
    return found

def resource_snapshot(path,c):
    """一次nvidia-smi资源快照；利用率和外部进程仅记录，不等待。"""
    from datetime import datetime,timezone
    gpu=gpu_available({"physical_gpu":0});others=other_b3_processes()
    reason=("existing B3 execution" if others else
            "insufficient free GPU memory: need at least 3072 MiB" if gpu["free_mib"]<c["minimum_free_mib"] else None)
    return {"schema":"tbps-b3-shared-resource-1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "config_sha256":ref(path).sha256,"policy":"shared_observe_only","minimum_free_mib":c["minimum_free_mib"],
        "GPU":gpu,"other_B3_processes":others,"ready":reason is None,"reason":reason}

def startup_gpu(path,c):
    """后台链复用启动快照；后续实际资源变化由CUDA错误处理，不轮询。"""
    snapshot_path=os.environ.get("TBPS_B3_GPU_SNAPSHOT")
    if snapshot_path:
        snapshot=read_json(snapshot_path,"tbps-b3-shared-resource-1")
        need(snapshot["config_sha256"]==ref(path).sha256 and snapshot["policy"]=="shared_observe_only"
             and snapshot["minimum_free_mib"]==c["minimum_free_mib"],"startup GPU snapshot/config mismatch")
    else:snapshot=resource_snapshot(path,c)
    need(snapshot["ready"] is True and not snapshot["other_B3_processes"] and
         snapshot["GPU"]["free_mib"]>=c["minimum_free_mib"],"shared GPU startup rejected: "+str(snapshot.get("reason")))
    return {**snapshot["GPU"],"snapshot_created_at_utc":snapshot["created_at_utc"],
        "snapshot_policy":"single startup observation; no polling or utilization gate"}

def run(path,scope,model,*,authorized=False,resume=False):
    need(authorized,"explicit --allow-smoke / --allow-full-evaluation required")
    need(model in ("irra","rasa","aptm"),"explicit native target")
    ctx=preflight(path,scope,model);c=ctx["c"];root=Path(c["output_base"])/scope/model
    store=ArtifactStore(root,new_base=c["output_base"])
    if (root/"completed").exists():return audit(path,scope,model)
    need(resume or not root.exists(),"existing B3 evaluation requires explicit --resume")
    need(not store.pending() and not (root/"failure").exists(),"pending/failed Attempt: preserve; no automatic retry")
    b3pixels=pixels(ctx,scope);reference=ctx["references"][model]
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit GPU0")
    gpu=startup_gpu(path,c);print("B3_GPU_STATE",json.dumps(gpu),flush=True)
    store.publish("attempt",{"preflight.json":json_bytes(ctx["report"])},{"scope":scope,"model":model},
        sources=(ref(path),reference.protocol_ref,b3pixels.protocol_ref))
    counts={};report={"status":"RUNNING","GPU_before":gpu,"scope":scope,"model":model,"G_H_loads":0,
        "G_H_forwards":0,"backward":0,"optimizer":0,"JVP":0,"victim_loads":0}
    start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        ctx["gate"].initialize_cuda()
        with load_model(c,model,scope) as (adapter,backend,checkpoint,counts):
            report["victim_loads"]=1;before=model_snapshot(backend)
            need(adapter.space==reference.space,"native victim ModelSpace differs from historical Full")
            origin=AttackOrigin("ap_attack_b3",FileRef(**c["b3"]["checkpoint"]),b3pixels.protocol_ref,BUDGET,digest(c["b3"]))
            cleantext=reference.read("clean_text",ctx)
            protocol={"schema":"tbps-b3-native-session-1","scope":scope,"model_space":asdict(adapter.space),
                "manifest":asdict(ctx["manifest"]),"source_protocol":asdict(reference.protocol_ref),
                "pixels_protocol":asdict(b3pixels.protocol_ref),"config":asdict(ref(path)),"target_feedback":False}
            protocol=json.loads(json_bytes(protocol))
            if not (root/"protocol").exists():store.publish("protocol",{"protocol.json":json_bytes(protocol)},
                {"kind":"B3_native_session"},sources=(ref(path),reference.protocol_ref,b3pixels.protocol_ref,checkpoint))
            else:store.verify("protocol");need(read_json(root/"protocol/protocol.json")==protocol,"resume protocol changed")
            adv=new_gallery(ctx,model,adapter,store,b3pixels,origin,protocol)
            def score(method,query,gallery):
                local=ArtifactStore(root/("comparison-"+str(METHODS.index(method))),new_base=c["output_base"])
                p={"session":asdict(ref(root/"protocol/manifest.json")),"method":method}
                if not (local.root/"protocol").exists():local.publish("protocol",{"protocol.json":json_bytes(p)},p,
                    sources=(ref(root/"protocol/manifest.json"),))
                else:local.verify("protocol");need(read_json(local.root/"protocol/protocol.json")==p,"comparison session changed")
                if (local.root/"result").exists():local.verify("result");return four(read_json(local.root/"result/metrics.json"))
                family="clean_gallery" if method=="Clean" else "adv_gallery"
                return evaluate_one(ctx,adapter,checkpoint,model,{"clean_text":query,family:gallery},local,reference,method)
            b3metrics=score("AP-Attack B3",cleantext,adv);del adv
            if scope=="full":
                clean=four(reference.summary["groups"]["Clean"]);gv2=four(reference.summary["groups"]["Image-only"])
                report["control_origin"]="existing independently audited same-space Full; no recomputation"
            else:
                clean=score("Clean",cleantext,reference.read("clean_gallery",ctx))
                from tbps.attacks.gv2 import BUDGET as GB
                p=ctx["plan"].g
                gorigin=AttackOrigin(p.attack_id,p.checkpoint,ctx["core"],GB,p.sources_sha256)
                gv2=score("G-v2 Image-only",cleantext,reference.read("adv_gallery",ctx,origin=gorigin))
                report["control_origin"]="Full feature rows selected BEFORE scoring; new 10x10 scores"
            groups=dict(zip(METHODS,(clean,b3metrics,gv2)))
            need(model_snapshot(backend)==before and all(not p.requires_grad and p.grad is None for p in backend.parameters()),"victim freeze changed")
            report.update(status="PASS",model_before=before,model_after=before,counts=dict(counts),model_frozen=True,
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if model!="irra":need(all(adapter.counts[k]==adapter.neural_counts[k] for k in adapter.neural_counts),"native actual hook counts")
            summary={"schema":"tbps-b3-four-metrics-1","groups":groups,"metric_unit":"percent",
                "gallery_count":len(ctx["manifest"].gallery),"query_count":len(ctx["manifest"].queries),"scope":scope,
                "victim_checkpoint":asdict(checkpoint),"B3_checkpoint":c["b3"]["checkpoint"],
                "original_text":True,"source_reference":asdict(reference.protocol_ref),"reporting_metrics":list(KEYS)}
            store.publish("completed",{"summary.json":json_bytes(summary),"metrics.csv":table_bytes(groups),
                "metrics_2dp.csv":table_bytes(groups,True)},{"kind":"B3_comparison_complete"},sources=(ref(root/"protocol/manifest.json"),))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc(),counts=dict(counts))
        print(report["traceback"],flush=True)
    finally:
        report["elapsed_seconds"]=time.monotonic()-start
        from tbps.engines.replay_control import rss_bytes
        report["peak_RSS_bytes"]=rss_bytes()
        if torch.cuda.is_initialized():
            report["GPU_peak_allocated_bytes"]=torch.cuda.max_memory_allocated();report["GPU_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"B3_execution"},sources=(ref(root/"attempt/manifest.json"),))
    print("B3_EVALUATION_"+report["status"],scope,model,json.dumps(report),flush=True)
    return int(report["status"]!="PASS")

def audit(path,scope,model):
    c=configuration(path);root=Path(c["output_base"])/scope/model;store=ArtifactStore(root,new_base=c["output_base"])
    need(not store.pending(),"uncommitted B3 evaluation")
    for key in ("attempt","protocol","execution","completed"):store.verify(key)
    for p in root.rglob("manifest.json"):
        sub=ArtifactStore(p.parent.parent,new_base=c["output_base"]);sub.verify(p.parent.name)
    s=read_json(root/"completed/summary.json");groups=s["groups"]
    need(set(groups)==set(METHODS) and all(set(v)==set(KEYS) for v in groups.values()),"four metrics only")
    ordered={k:groups[k] for k in METHODS}
    need((root/"completed/metrics.csv").read_bytes()==table_bytes(ordered)
         and (root/"completed/metrics_2dp.csv").read_bytes()==table_bytes(ordered,True),"summary/CSV consistency")
    report=read_json(root/"execution/report.json")
    need(report["status"]=="PASS" and report["model_before"]==report["model_after"] and report["model_frozen"]
         and report["counts"]["image_samples"]==(3074 if scope=="full" else 10) and report["counts"]["text_samples"]==0
         and all(report[k]==0 for k in ("G_H_loads","G_H_forwards","backward","optimizer","JVP")),"successful bounded frozen execution")
    need((s["gallery_count"],s["query_count"])==((3074,6156) if scope=="full" else (10,10)),"B3 member counts")
    print("B3_CPU_AUDIT_PASS",scope,model,flush=True);return 0

def main(argv=None):
    p=argparse.ArgumentParser(description="B3 only; default CPU; no GPU at import")
    p.add_argument("--config",required=True);p.add_argument("--scope",choices=("smoke","full"),default="smoke")
    p.add_argument("--action",choices=("preflight","resources","pixels","run","audit"),default="preflight")
    p.add_argument("--model",choices=("irra","rasa","aptm"))
    p.add_argument("--allow-smoke",action="store_true");p.add_argument("--allow-full-evaluation",action="store_true")
    p.add_argument("--allow-full-pixel-transport",action="store_true");p.add_argument("--resume",action="store_true")
    a=p.parse_args(argv)
    if a.action=="resources":
        need(not(a.allow_smoke or a.allow_full_evaluation or a.allow_full_pixel_transport or a.resume),"resource snapshot has no model execution flags")
        snapshot=resource_snapshot(a.config,configuration(a.config))
        print(json.dumps(snapshot),flush=True)
        if snapshot["ready"] is not True:
            print("B3_RESOURCE_REJECTED: "+str(snapshot["reason"]),file=__import__("sys").stderr,flush=True)
            return 1
        return 0
    if a.action=="preflight":
        need(not(a.allow_smoke or a.allow_full_evaluation or a.allow_full_pixel_transport or a.resume),"CPU preflight no execution flags")
        preflight(a.config,a.scope,a.model);return 0
    if a.action=="pixels":
        need(a.scope=="smoke" or a.allow_full_pixel_transport,"Full CPU transport needs explicit confirmation")
        pixels(preflight(a.config,a.scope),a.scope);print("B3_PIXELS_PASS",a.scope);return 0
    need(a.model is not None,"explicit target model")
    if a.action=="audit":return audit(a.config,a.scope,a.model)
    need(a.allow_smoke!=a.allow_full_evaluation and (a.allow_smoke if a.scope=="smoke" else a.allow_full_evaluation),"scope-specific authorization")
    return run(a.config,a.scope,a.model,authorized=True,resume=a.resume)
