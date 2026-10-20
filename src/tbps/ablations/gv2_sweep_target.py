"""Epoch图像变体的原生评分；复用历史Raw文本特征，不改既有Adapter/Engine。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import ExitStack
import gc,json,os,subprocess,sys,time,traceback,resource
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest
from tbps.engines.materialized_cache import FeatureCache,CachedIRRA
from tbps.engines.native_cache import NativeFeatureCache,compact
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.engines.aptm_score_cache import APTMSetScorer
from tbps.engines.provenance import BoundRepresentation,AttackOrigin,payload_digest
from tbps.engines.evaluation import EvaluationEngine,publish_result,load_result
from tbps.models.base import InputBinding
from tbps.baselines.comparison import load_model,payload_cat,four
from tbps.attacks.gv2 import BUDGET as G_BUDGET
from tbps.attacks.hv1 import BUDGET as H_BUDGET
from tbps.smoke_irra import model_snapshot
from .gv2_sweep import need,ref,prepare,variant,paths,resources,audit_variant,csv_metrics

METHODS=("G-v2 Image-only","G-v2 + H-v1 Joint")

def texts(reference,ctx):
    clean=reference.read("clean_text",ctx)
    raw=reference.read("adv_text",ctx) # 旧桥只提供数据；按文本模态单独绑定原文SHA。
    h=ctx["plan"].h
    origin=AttackOrigin("hv1",h.checkpoint,ctx["core"],H_BUDGET,h.sources_sha256)
    import hashlib
    originals=tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in ctx["manifest"].queries)
    adv=BoundRepresentation(raw.batch,raw.inputs,raw.payload_sha256,raw.cache_sources,origin,originals)
    return clean,adv

def encode_gallery(ctx,adapter,model,v,store,protocol):
    manifest=ctx["manifest"];md=[read_json(v.store.root/v.key(i)/"metadata.json") for i in range(len(manifest.gallery))]
    hashes=tuple(x["pixels"]["adversarial_pixels_sha256"] for x in md)
    binding=InputBinding(adapter.space.id,digest(asdict(manifest)),hashes,"image",tuple(g.image_id for g in manifest.gallery))
    if model=="irra":cache=FeatureCache(store,protocol,adapter.space,chunk=16)
    else:
        validator=__import__("tbps.models."+model,fromlist=["validate_payload"]).validate_payload
        cache=NativeFeatureCache(store,protocol,adapter.space,validator,chunk=16)
    encoded=0
    for start in range(0,len(hashes),16):
        end=min(start+16,len(hashes));key=cache.key("adv_gallery",start)
        if (store.root/key).exists():cache.load("adv_gallery",binding,start,end)
        else:
            parts=[]
            for index in range(start,end):
                arrays,_=v.load(index)
                value=adapter.inference().encode_images(arrays["adversarial"].to("cuda:0")).payload.cpu() if model=="irra" else adapter.encode_images(arrays["adversarial"])
                parts.append(compact(value));encoded+=1
                del arrays,value
            payload=payload_cat(parts);cache.publish("adv_gallery",binding,start,end,payload);del payload,parts
        print("GV2_SWEEP_FEATURE_PROGRESS",model,end,"/",len(hashes),flush=True)
    origin=AttackOrigin("gv2",FileRef(**v.protocol["checkpoint"]),v.protocol_ref,G_BUDGET,digest(v.protocol))
    return cache.gather("adv_gallery",binding,origin=origin,originals=tuple(g.image_sha256 for g in manifest.gallery)),encoded

def target(c,scope,epoch,model,*,resume=False):
    ctx=prepare(c,scope,model=model);reference=ctx["references"][model]
    eroot,base=paths(c,scope,epoch);root=eroot/model;store=ArtifactStore(root,new_base=base)
    need(resume or not root.exists(),"existing target; explicit resume required")
    need(not store.pending() and not (root/"execution").exists() and not (root/"failure").exists(),"failed/completed target cannot retry automatically")
    if epoch==10 and scope=="full":
        # 原Full已按相同Raw输入和空间审计通过，不重新生成或评分。
        groups=dict(zip(METHODS,(four(reference.summary["groups"][k]) for k in ("Image-only","Joint"))))
        store.publish("completed",{"summary.json":json_bytes({"epoch":10,"scope":"full","model":model,
            "groups":groups,"historical_result_reused":True,"source":asdict(reference.protocol_ref)}),
            "metrics.csv":csv_metrics(groups),"metrics_2dp.csv":csv_metrics(groups,True)},
            {"kind":"read-only-Epoch10-result-reference"},sources=(reference.protocol_ref,ref(reference.root/"completed/manifest.json")))
        print("GV2_EPOCH10_FULL_REUSED",model,flush=True);return
    generation=audit_variant(ctx,scope,epoch);v=variant(ctx,scope,epoch)
    clean_text,adv_text=texts(reference,ctx)
    gpu=resources(c,scope);torch.manual_seed(42);ctx["gate"].initialize_cuda()
    from tbps.models.irra import fp32_runtime
    started=time.monotonic();report={"status":"RUNNING","scope":scope,"epoch":epoch,"model":model,
        "G_H_loads":0,"G_H_forwards":0,"backward":0,"optimizer":0,"JVP":0,"victim_loads":0,"GPU_before":gpu}
    try:
        with fp32_runtime(),load_model(c,model,scope) as (adapter,backend,checkpoint,counts):
            report["victim_loads"]=1;report["counts"]=counts;before=model_snapshot(backend)
            need(adapter.space==reference.space,"victim preprocessing/tokenizer/scoring space differs from historical Full")
            protocol={"schema":"gv2-epoch-native-evaluation-1","epoch":epoch,"scope":scope,
                "model_space":asdict(adapter.space),"manifest":asdict(ctx["manifest"]),
                "variant_protocol":asdict(v.protocol_ref),"old_native_protocol":asdict(reference.protocol_ref),
                "config":asdict(ref(c["path"])),"image_batch":1,"query_chunk":128,
                "effective_k_test":10 if scope=="smoke" else 128,"no_attack_feedback":True,
                "only_new_adv_gallery_encoded":True,"raw_caption_preserved":True}
            protocol=json.loads(json_bytes(protocol))
            if not (root/"protocol").exists():
                store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"kind":"epoch-native-session"},
                    sources=(ref(c["path"]),v.protocol_ref,reference.protocol_ref,checkpoint))
            else:
                store.verify("protocol");need(read_json(root/"protocol/protocol.json")==protocol,"resume protocol conflict")
            gallery,encoded=encode_gallery(ctx,adapter,model,v,store,protocol)
            inputs={"clean_text":clean_text,"adv_text":adv_text,"adv_gallery":gallery}
            protocol_ref=ref(root/"protocol/manifest.json")
            if model=="irra":scorer=CachedIRRA(adapter.space,checkpoint,cache_protocol=reference.protocol_ref)
            else:scorer=(ChunkCachedScorer if model=="rasa" else APTMSetScorer)(adapter,inputs,store,protocol_ref)
            engine=EvaluationEngine(ctx["manifest"],scorer,direction="higher_better",
                execution="readonly_cached" if model=="irra" else "materialized_native",model_sources=(checkpoint,))
            if (root/"result").exists():
                p=read_json(root/"result/protocol.json");_,result,scores=load_result(store,"result",expected_protocol_sha256=digest(p));del scores
            else:
                p,result,scores=engine.run(inputs,modes=("Image-only","Joint"),query_chunk=128)
                publish_result(store,"result",p,result,scores,sources=(protocol_ref,));del scores
            groups={METHODS[i]:four(result["groups"][mode]) for i,mode in enumerate(("Image-only","Joint"))}
            need(model_snapshot(backend)==before and not backend.training and all(not p.requires_grad and p.grad is None for p in backend.parameters()),"victim freeze")
            need(counts["image_samples"]==encoded and counts["text_samples"]==0,"no hidden re-encoding")
            if model!="irra":
                need(all(counts[k]==adapter.neural_counts[k] for k in adapter.neural_counts),"native actual hooks")
            summary={"epoch":epoch,"scope":scope,"model":model,"groups":groups,"metric_unit":"percent",
                "gallery_count":len(ctx["manifest"].gallery),"query_count":len(ctx["manifest"].queries),
                "G_checkpoint":c["points"][str(epoch)],"H_checkpoint":asdict(ctx["plan"].h.checkpoint),
                "variant_protocol":asdict(v.protocol_ref),"history_native_protocol":asdict(reference.protocol_ref),
                "raw_clean_controls_reused":{k:four(reference.summary["groups"][k]) for k in ("Clean","Text-only")} if scope=="full" else None,
                "smoke_not_formal_result":scope=="smoke"}
            store.publish("completed",{"summary.json":json_bytes(summary),"metrics.csv":csv_metrics(groups),
                "metrics_2dp.csv":csv_metrics(groups,True)},{"kind":"G-epoch-two-mode-result"},
                sources=(ref(root/"result/manifest.json"),protocol_ref))
            report.update(status="PASS",counts=dict(counts),model_before=before,model_after=before,encoded_images=encoded,
                new_text_encoding=0,model_frozen=True)
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc());print(report["traceback"],flush=True)
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"epoch-target-execution"},
            sources=(ref(c["path"]),v.protocol_ref))
    print("GV2_SWEEP_TARGET_"+report["status"],scope,epoch,model,json.dumps(report),flush=True)
    need(report["status"]=="PASS","native target failed; preserve; no retry")

def audit(c,scope,epoch,model):
    need(not torch.cuda.is_initialized(),"CPU audit only")
    ctx=prepare(c,scope,model=model);root,base=paths(c,scope,epoch);store=ArtifactStore(root/model,new_base=base)
    if epoch==10 and scope=="full":
        store.verify("completed")
        need(read_json(store.root/"completed/summary.json")["historical_result_reused"] is True,"Epoch10 explicit historical reuse")
        store.publish("cpu_audit",{"report.json":json_bytes({"status":"PASS","epoch":10,"model":model,
            "scope":"full","historical_audit_referenced":c["history_audits"][model],"score_recomputed":False,"CUDA":False})},
            {"kind":"read-only-Epoch10-audit-reference"},sources=(ref(store.root/"completed/manifest.json"),))
        print("GV2_SWEEP_CPU_AUDIT_PASS",scope,epoch,model,flush=True);return
    audit_variant(ctx,scope,epoch)
    for p in store.root.rglob("manifest.json"):
        ArtifactStore(p.parent.parent,new_base=base).verify(p.parent.name)
    s=read_json(store.root/"completed/summary.json");groups={k:s["groups"][k] for k in METHODS}
    need((store.root/"completed/metrics.csv").read_bytes()==csv_metrics(groups)
         and (store.root/"completed/metrics_2dp.csv").read_bytes()==csv_metrics(groups,True),"CSV/summary")
    native_protocol=read_json(store.root/"result/protocol.json")
    _,native,scores=load_result(store,"result",expected_protocol_sha256=digest(native_protocol))
    del scores # 只读shape/FP32/有限性/内容SHA，不调用Ranking。
    need(groups=={METHODS[i]:four(native["groups"][m]) for i,m in enumerate(("Image-only","Joint"))},"native result/public four metrics")
    e=read_json(store.root/"execution/report.json")
    need(e["status"]=="PASS" and e["model_before"]==e["model_after"] and e["counts"]["text_samples"]==0
         and all(e[k]==0 for k in ("G_H_loads","G_H_forwards","backward","optimizer","JVP")),"frozen bounded target")
    store.publish("cpu_audit",{"report.json":json_bytes({"status":"PASS","epoch":epoch,"model":model,
        "scope":scope,"CUDA":False,"score_recomputed":False,"gallery_count":s["gallery_count"],"query_count":s["query_count"]})},
        {"kind":"epoch-target-CPU-audit"},sources=(ref(store.root/"completed/manifest.json"),ref(store.root/"execution/manifest.json")))
    print("GV2_SWEEP_CPU_AUDIT_PASS",scope,epoch,model,flush=True)

def smoke(c):
    ctx=prepare(c,"smoke",weights=True);gpu=resources(c)
    base=Path(c["output_base"]);root=base/"smoke_v1";store=ArtifactStore(root,new_base=base)
    need(not root.exists(),"only one Smoke Attempt; preserve, no retry")
    store.publish("attempt",{"preflight.json":json_bytes(ctx["preflight"]),"resources.json":json_bytes(gpu)},
        {"kind":"one-controlled-six-node-smoke"},sources=(ref(c["path"]),))
    logs=root/"logs";logs.mkdir();started=time.monotonic()
    steps=[("G10",["--action","generate","--epoch","10"]),("G60",["--action","generate","--epoch","60"])]
    steps += [(m,["--action","target","--epoch","60","--model",m]) for m in c["models"]]
    steps += [(m+"-audit",["--action","audit","--epoch","60","--model",m]) for m in c["models"]]
    try:
        for name,args in steps:
            cmd=[c["python"],"-B","-u","-m","tbps.gv2_epoch_eval","--config",c["path"],"--scope","smoke",*args]
            if not name.endswith("-audit"):cmd.append("--allow-smoke")
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(Path(__file__).resolve().parents[2]),
                 "CUDA_VISIBLE_DEVICES":"" if name.endswith("-audit") else "0"}
            with (logs/(name+".log")).open("x") as f:
                value=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
            need(value.returncode==0,f"Smoke stage {name} failed, exit{value.returncode}; see {logs/(name+'.log')}")
            print("GV2_SWEEP_SMOKE_STAGE_PASS",name,flush=True)
        # 总完成回执仅在原始E10对照和三个目标独立CPU审计均通过后发布。
        g10=read_json(root/"epoch_0010/images/execution/report.json")
        g60=read_json(root/"epoch_0060/images/execution/report.json")
        report={"status":"PASS","scope":"smoke","epochs_generated":[10,60],"G_loads":2,"G_forwards":20,
            "H_loads":0,"H_forwards":0,"backward":0,"optimizer":0,"JVP":0,
            "epoch10_historical_max_abs_error":g10["epoch10_historical_max_abs_error"],
            "new_epoch60_generation":g60,"targets":{m:read_json(root/f"epoch_0060/{m}/execution/report.json") for m in c["models"]},
            "seconds":time.monotonic()-started,"full_started":False}
        need(g10["counts"]["G_forwards"]==g60["counts"]["G_forwards"]==10,"bounded real G sample counts")
        store.publish("completed",{"report.json":json_bytes(report)},{"kind":"six-node-evaluation-preparation-smoke"},
            sources=(ref(c["path"]),*(ref(root/f"epoch_0060/{m}/cpu_audit/manifest.json") for m in c["models"])))
        print("GV2_SIX_NODE_SMOKE_PASS",json.dumps(report),flush=True)
    except BaseException as exc:
        store.publish("failure",{"report.json":json_bytes({"status":"FAIL","error":str(exc),"traceback":traceback.format_exc()})},
            {"kind":"single-Smoke-failure"},sources=(ref(c["path"]),))
        raise
