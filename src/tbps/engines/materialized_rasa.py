"""RaSa跨模型迁移评估：只读既有攻击样本；默认CPU，Full由用户显式启动。"""
from pathlib import Path
from dataclasses import asdict
import argparse,io,json,os,time,traceback,sys
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,digest,checked
from tbps.models.rasa import RasaSource,native_modules,load_rasa,validate_payload,PREPROCESS,SCORING,official_tokenizer
from tbps.engines.materialized_inputs import load_inputs
from tbps.engines.materialized_irra import bindings,csv_bytes,gpu_available
from tbps.engines.materialized_cache import ref
from tbps.engines.native_cache import NativeFeatureCache
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.materialization.bulk_session import BulkSession
from tbps.engines.provenance import AttackOrigin
from tbps.engines.evaluation import EvaluationEngine,publish_result,load_result
from tbps.smoke_irra import model_snapshot
from tbps.runtime_closure import cpu_only
from tbps.engines.replay_control import rss_bytes

def need(ok,message):
    if not ok:raise ValueError(message)

def config(path):
    c=read_json(path,"tbps-rasa-materialized-evaluation-1")
    need(c["image_batch"]==c["text_batch"]==c["fusion_batch"]==1 and c["query_chunk"]==128 and
         c["cache_chunk"]==16 and c["seed"]==42 and c["precision"]=="fp32/math/no_tf32","frozen execution configuration")
    for p,sha in c["producer_sources"].items():checked(FileRef(p,sha))
    checked(FileRef(**c["rasa_source"]))
    return c

def preflight(path,scope):
    c=config(path);source=RasaSource(c["rasa_source"]["path"]);source.verify()
    def forbidden(*a,**k):raise RuntimeError("CPU native import/tokenizer preflight forbids weights/CUDA")
    with patch.object(torch,"load",forbidden),patch.object(torch.cuda,"init",forbidden),patch.object(torch.cuda,"_lazy_init",forbidden),native_modules(source) as (_,tokenizer,utils):
        tok=official_tokenizer(tokenizer,source)
        test=tok([utils.pre_caption("A man in a blυe shirt.",50)],padding="max_length",
                 truncation=True,max_length=50,return_tensors="pt")
        need(test.input_ids.shape==(1,50),"native BERT tokenizer closure")
    # 先完成真实导入闭包，再由既有CPUStartupGate固定快照。
    ctx=load_inputs(c,scope);ctx["source"]=source
    print("RASA_CPU_PREFLIGHT_PASS",json.dumps(ctx["report"]),flush=True)
    return ctx

def encode(ctx,adapter,family,ids):
    rows=[]
    for key in ids:
        if family.endswith("gallery"):
            arrays,_=ctx["store"].load(ctx["images"][key])
            row=adapter.encode_images(arrays["original" if family=="clean_gallery" else "adversarial"])
            del arrays
        else:row=adapter.encode_texts((ctx["texts"][key][0 if family=="clean_text" else 1],))
        rows.append(row)
    return {k:torch.cat([r[k] for r in rows],0) for k in rows[0]}

def bound_inputs(ctx,cache):
    bound,original=bindings(ctx,cache.space);inputs={}
    for family,b in bound.items():
        kind=b.modality;origin=None
        if family.startswith("adv"):
            component=ctx["plan"].g if kind=="image" else ctx["plan"].h
            from tbps.attacks.gv2 import BUDGET as IMAGE
            from tbps.attacks.hv1 import BUDGET as TEXT
            origin=AttackOrigin(component.attack_id,component.checkpoint,ctx["core"],IMAGE if kind=="image" else TEXT,
                                component.sources_sha256)
        inputs[family]=cache.gather(family,b,origin=origin,originals=original[kind] if origin else ())
    return inputs

def audit(path,scope):
    ctx=preflight(path,scope);root=Path(ctx["c"]["output_base"])/scope
    store=ArtifactStore(root,new_base=ctx["c"]["output_base"])
    store.verify("completed");p=read_json(root/"result/protocol.json")
    _,r,scores=load_result(store,"result",expected_protocol_sha256=digest(p));del scores
    need(read_json(root/"completed/summary.json")==r and (root/"completed/metrics.csv").read_bytes()==csv_bytes(r),"summary/CSV mismatch")
    # 全部特征及融合Score批次的manifest/文件SHA；不重算排名或模型前向。
    for p in root.iterdir():
        if p.is_dir() and (p.name.startswith(("clean_","adv_","score-","attempt-","execution-"))):store.verify(p.name)
    need(not store.pending(),"pending output")
    executions=[read_json(p/"report.json") for p in root.glob("execution-*")]
    need(executions and executions[-1]["status"]=="PASS" and executions[-1]["model_frozen"],"successful frozen execution required")
    print("RASA_MATERIALIZED_CPU_AUDIT_PASS",scope,flush=True);return 0

def run(path,scope,*,authorized=False,resume=False):
    need(authorized,"explicit GPU authorization required")
    ctx=preflight(path,scope);c=ctx["c"];source=ctx["source"]
    root=Path(c["output_base"])/scope;store=ArtifactStore(root,new_base=c["output_base"])
    if (root/"completed").exists():return audit(path,scope)
    need(resume or not root.exists(),"existing output requires --resume")
    need(not store.pending(),"pending output; inspect without overwriting")
    old=list(root.glob("attempt-*"))
    need(all((root/("execution-"+p.name.split("-")[-1])).exists() for p in old),"unclosed Attempt; inspect before resume")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit GPU0 required")
    n=len(old)+1;tag=str(n).zfill(6)
    gpu=gpu_available({"physical_gpu":0});print("RASA_GPU_BEFORE",json.dumps(gpu),flush=True)
    store.publish("attempt-"+tag,{"preflight.json":json_bytes(ctx["report"])},{"scope":scope},
        sources=(ref(path),source.ref,source.checkpoint,ctx["core"]))
    report={"status":"RUNNING","GPU_before":gpu,"IRRA_G_H_loads":0,"IRRA_G_H_forwards":0,
            "backward":0,"optimizer":0,"JVP":0,"RaSa_load_attempts":0,"RaSa_loads":0}
    start=time.monotonic()
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        ctx["gate"].initialize_cuda()
        report["RaSa_load_attempts"]=1
        with load_rasa(source,smoke=scope=="smoke") as adapter:
            report["RaSa_loads"]=1;before=model_snapshot(adapter.backend)
            report["counts"]=adapter.counts;report["neural_hook_counts"]=adapter.neural_counts
            if scope=="smoke":
                tokens=[]
                for q in ctx["manifest"].queries:
                    original,adv=ctx["texts"][q.query_id]
                    a,b=(adapter.tokenize((t,)).input_ids[0].tolist() for t in (original,adv))
                    tokens.append({"query_id":q.query_id,"token_ids_changed":a!=b,
                                   "original_ids":a,"adversarial_ids":b})
                report["tokenizer_audit"]=tokens
            bound,_=bindings(ctx,adapter.space)
            protocol={"schema":"tbps-native-materialized-cache-1","config":asdict(ref(path)),
                "source":asdict(source.ref),"model_space":asdict(adapter.space),"scope":scope,
                "manifest":asdict(ctx["manifest"]),"inputs":{k:asdict(v) for k,v in bound.items()},
                "collection":asdict(ctx["core"]),"preprocess":PREPROCESS,"scoring":SCORING,
                "effective_k_test":10 if scope=="smoke" else 128,"attack_feedback":False,"new_attack_generation":False}
            if (root/"protocol").exists():
                store.verify("protocol");need(read_json(root/"protocol/protocol.json")==json.loads(json_bytes(protocol)),"resume source/protocol mismatch")
            else:store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"kind":"native_encoding_session"},
                        sources=(ref(path),source.ref,source.checkpoint,ctx["core"]))
            cache=NativeFeatureCache(store,protocol,adapter.space,validate_payload,chunk=16)
            with BulkSession(ctx["store"]):
                for family,b in bound.items():
                    for offset in range(0,len(b.ordered_ids),16):
                        end=min(offset+16,len(b.ordered_ids));key=cache.key(family,offset)
                        if (root/key).exists():cache.load(family,b,offset,end);state="REUSED"
                        else:
                            x=encode(ctx,adapter,family,b.ordered_ids[offset:end]);cache.publish(family,b,offset,end,x);del x;state="ENCODED"
                        print("RASA_FEATURE_PROGRESS",family,end,"/",len(b.ordered_ids),state,flush=True)
            inputs=bound_inputs(ctx,cache)
            scorer=ChunkCachedScorer(adapter,inputs,store,cache.protocol_ref)
            engine=EvaluationEngine(ctx["manifest"],scorer,direction="higher_better",execution="materialized_native",
                                    model_sources=(source.checkpoint,source.ref))
            if (root/"result").exists():
                store.verify("result");p=read_json(root/"result/protocol.json")
                need(p["model_space"]==asdict(adapter.space) and p["manifest_sha256"]==digest(asdict(ctx["manifest"])),"existing result identity")
                _,result,scores=load_result(store,"result",expected_protocol_sha256=digest(p))
            else:
                p,result,scores=engine.run(inputs,query_chunk=128)
                publish_result(store,"result",p,result,scores,sources=(cache.protocol_ref,source.ref))
            del scores,inputs
            after=model_snapshot(adapter.backend);need(before==after,"RaSa parameters/buffers changed")
            need(all(adapter.counts[k]==adapter.neural_counts[k] for k in adapter.neural_counts),
                 "actual neural hooks differ from reported samples/pairs")
            report.update(model_before=before,model_after=after,model_frozen=True,
                          score_cache_hits=scorer.hits,score_cache_new=scorer.new)
            store.publish("completed",{"summary.json":json_bytes(result),"metrics.csv":csv_bytes(result)},
                {"kind":"native_materialized_evaluation_complete","scope":scope},sources=(ref(root/"result/manifest.json"),cache.protocol_ref))
            report["status"]="PASS"
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),exception_type=type(exc).__name__,traceback=traceback.format_exc())
        print(report["traceback"],file=sys.stderr,flush=True)
    finally:
        report["elapsed_seconds"]=time.monotonic()-start;report["peak_RSS_bytes"]=rss_bytes()
        if torch.cuda.is_initialized():
            report["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        try:report["GPU_after"]=gpu_available({"physical_gpu":0})
        except Exception as exc:report["GPU_after_error"]=str(exc)
        store.publish("execution-"+tag,{"report.json":json_bytes(report)},{"kind":"RaSa_execution"},sources=(ref(root/("attempt-"+tag)/"manifest.json"),))
    print("RASA_EXECUTION_REPORT",json.dumps(report),flush=True)
    print("RASA_MATERIALIZED_EVALUATION_"+report["status"],scope,flush=True)
    return int(report["status"]!="PASS")

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--action",choices=["preflight","run","audit"],default="preflight")
    p.add_argument("--scope",choices=["smoke","full"],default="smoke")
    p.add_argument("--allow-smoke",action="store_true");p.add_argument("--allow-full-evaluation",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args(argv)
    if a.action=="preflight":preflight(a.config,a.scope);return 0
    if a.action=="audit":return audit(a.config,a.scope)
    return run(a.config,a.scope,authorized=a.allow_smoke if a.scope=="smoke" else a.allow_full_evaluation,resume=a.resume)
if __name__=="__main__":raise SystemExit(main())
