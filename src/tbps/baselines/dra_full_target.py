"""DRA冻结产物的三模型原生Smoke评分；复用现有缓存/Engine验证与官方rank。"""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import io
import json
import time
import traceback
import torch
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,digest,sha_file
from tbps.baselines.dra_official import COMMIT,MODES,need,projection_check,verify_official
from tbps.baselines.dra_full import source_inputs,METRICS
from tbps.baselines.sga_official_io import load_file
from tbps.baselines.comparison import load_model,payload_cat,payload_select
from tbps.engines.materialized_cache import ref,FeatureCache,CachedIRRA
from tbps.engines.native_cache import NativeFeatureCache,compact
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.engines.aptm_score_cache import APTMSetScorer
from tbps.engines.provenance import payload_digest,AttackOrigin
from tbps.engines.evaluation import EvaluationEngine,MODES as MODE_INPUTS
from tbps.models.base import InputBinding
from tbps.models.irra_source import IRRASource
from tbps.models.irra import fp32_runtime
from tbps.contracts import AttackBudget,ImageBudget,TextBudget
from tbps.smoke_irra import model_snapshot
from tbps.engines.replay_control import rss_bytes

def rank_function(c):
    s=IRRASource.from_dict(c["irra"])
    return load_file(s.ref("utils/metrics.py"),"_dra_official_rank").rank

def evaluate(c,model):
    source,col,manifest=source_inputs(c)
    root=Path(c["output_base"])/"full"/model
    need(not root.exists(),"existing Target Attempt; no retry/overwrite")
    from tbps.baselines.dra_full import resources
    snapshot=resources(c)
    store=ArtifactStore(root,new_base=c["output_base"])
    loader={"inputs":{"dataset_sources":c["dataset_sources"]},"victim_sources":c["victim_sources"]}
    report={"status":"RUNNING","model":model,"target_loads":0,"source_generation":0,"backward":0,"optimizer":0,
            "resources_before":snapshot}
    start=time.monotonic()
    try:
        torch.set_num_threads(2)
        with fp32_runtime(),load_model(loader,model,"full") as (adapter,backend,checkpoint,counts):
            report["target_loads"]=1;before=model_snapshot(backend)
            protocol={"schema":"tbps-materialized-irra-cache-1" if model=="irra" else "tbps-DRA-native-target-1",
                "model_space":asdict(adapter.space),"manifest":asdict(manifest),
                "source_collection":asdict(ref(source.root/"completed/manifest.json")),"target_feedback":False,
                "G_H_generation":False,"ranking_feedback_to_attack":False}
            store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"kind":"DRA_native_target"},
                sources=(checkpoint,ref(c["path"]),ref(source.root/"completed/manifest.json")))
            if model=="irra":cache=FeatureCache(store,protocol,adapter.space,chunk=16)
            else:
                mod=__import__("tbps.models."+model,fromlist=["validate_payload"])
                cache=NativeFeatureCache(store,protocol,adapter.space,mod.validate_payload,chunk=16)
            inputs={}
            for family in ("clean_gallery","adv_gallery","clean_text","adv_text"):
                is_image=family.endswith("gallery");adv=family.startswith("adv")
                ids=tuple(g.image_id for g in manifest.gallery) if is_image else tuple(q.query_id for q in manifest.queries)
                originals=tuple(g.image_sha256 for g in manifest.gallery) if is_image else tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in manifest.queries)
                hashes=[]
                if is_image:
                    for key in col["units"]:
                        raw=torch.load(source.root/key/"pixels.pt",map_location="cpu",weights_only=True)
                        xs=raw["adversarial" if adv else "original"].detach().clone();del raw
                        for x in xs:hashes.append(payload_digest(x.unsqueeze(0).clone()) if adv else originals[len(hashes)])
                        del xs
                else:
                    texts=[col["adv_texts"][q.query_id] if adv else q.caption for q in manifest.queries]
                    hashes=[hashlib.sha256(x.encode()).hexdigest() for x in texts]
                binding=InputBinding(adapter.space.id,digest(asdict(manifest)),tuple(hashes),"image" if is_image else "text",ids)
                if is_image:
                    parts=[];offset=0
                    for key in col["units"]:
                        source.verify(key)
                        raw=torch.load(source.root/key/"pixels.pt",map_location="cpu",weights_only=True)
                        xs=raw["adversarial" if adv else "original"].detach().clone();del raw
                        for x in xs:
                            value=adapter.inference().encode_images(x.unsqueeze(0).to("cuda:0")).payload.cpu() if model=="irra" else adapter.encode_images(x.unsqueeze(0))
                            parts.append(compact(value))
                            if len(parts)==16:
                                cache.publish(family,binding,offset,offset+len(parts),payload_cat(parts))
                                offset+=len(parts);parts=[]
                                if offset%128==0:print("DRA_TARGET_PROGRESS",model,family,offset,"/",len(ids),flush=True)
                        del xs
                    if parts:cache.publish(family,binding,offset,len(ids),payload_cat(parts))
                    del parts,value
                else:
                    for i in range(0,len(texts),16):
                        end=min(i+16,len(texts))
                        value=adapter.inference().encode_texts(texts[i:end])[0].payload.cpu() if model=="irra" else adapter.encode_texts(texts[i:end])
                        cache.publish(family,binding,i,end,value)
                        if end%128==0 or end==len(ids):print("DRA_TARGET_PROGRESS",model,family,end,"/",len(ids),flush=True)
                    del value
                budget=AttackBudget(image=ImageBudget("linf",8/255,"rgb_0_1")) if is_image else AttackBudget(text=TextBudget("word",1,None,"DRA_official_"+COMMIT))
                origin=AttackOrigin("vlptattack_dra_"+("image" if is_image else "text"),FileRef(**c["irra"]["checkpoint"]),
                                   ref(source.root/"protocol/manifest.json"),budget,sha_file(c["path"])) if adv else None
                inputs[family]=cache.gather(family,binding,origin=origin,originals=originals if adv else ())
                print("DRA_TARGET_ENCODING",model,family,len(ids),flush=True)
            scorer=CachedIRRA(adapter.space,checkpoint,cache_protocol=ref(root/"protocol/manifest.json")) if model=="irra" else (
                ChunkCachedScorer if model=="rasa" else APTMSetScorer)(adapter,inputs,store,ref(root/"protocol/manifest.json"))
            engine=EvaluationEngine(manifest,scorer,direction="higher_better",
                execution="readonly_cached" if model=="irra" else "materialized_native",model_sources=(checkpoint,))
            engine_protocol,engine_result,scores=engine.run(inputs,modes=MODES,query_chunk=128)
            groups={mode:{k:engine_result["groups"][mode][k] for k in METRICS} for mode in MODES}
            need(model_snapshot(backend)==before and not backend.training and all(not p.requires_grad and p.grad is None for p in backend.parameters()),"Target weights/buffers/freeze")
            def csv_data(rounded):
                b=io.StringIO();writer=csv.writer(b,lineterminator="\n");writer.writerow(("Mode",*METRICS))
                for mode in MODES:writer.writerow((mode,*(format(groups[mode][k],".2f") if rounded else groups[mode][k] for k in METRICS)))
                return b.getvalue().encode()
            data=io.BytesIO();torch.save(compact(scores),data)
            summary={"scope":"full","Gallery":3074,"Query":6156,"groups":groups,
                "matched_clean":True,"model":model,"source":"irra","metric_unit":"percent",
                "source_collection":asdict(ref(source.root/"completed/manifest.json"))}
            feature_refs=tuple(ref(store.root/cache.key(family,i)/"manifest.json") for family in inputs
                for i in range(0,len(inputs[family].inputs.ordered_ids),16))
            store.publish("completed",{"summary.json":json_bytes(summary),"metrics.csv":csv_data(False),
                "metrics_2dp.csv":csv_data(True),"scores.pt":data.getvalue(),
                "evaluation_protocol.json":json_bytes(engine_protocol)},
                {"kind":"DRA_native_full_result"},sources=(ref(root/"protocol/manifest.json"),*feature_refs))
            report.update(status="PASS",model_before=before,model_after=model_snapshot(backend),counts=dict(counts),
                          neural_counts=dict(counts if model=="irra" else adapter.neural_counts))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc());print(report["traceback"],flush=True)
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss_bytes())
        if torch.cuda.is_initialized():report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if not (root/"protocol").exists():store.publish("protocol",{"protocol.json":json_bytes({"model":model})},{"kind":"failed_target"},sources=(ref(c["path"]),))
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"DRA_target_execution"},sources=(ref(root/"protocol/manifest.json"),))
    print("DRA_NATIVE_TARGET_"+report["status"],"full",model,flush=True)
    return int(report["status"]!="PASS")

def full_metrics(matrix,manifest,rank):
    need(matrix.shape==(len(manifest.queries),len(manifest.gallery)) and matrix.dtype==torch.float32
         and matrix.device.type=="cpu" and torch.isfinite(matrix).all() and not matrix.requires_grad,
         "Full score shape/FP32/finite")
    identity=manifest.evaluation_identity()
    labels={pid:i for i,pid in enumerate(dict.fromkeys(identity.gallery_pids))}
    need(all(pid in labels for pid in identity.query_pids),"all Query positives")
    gids=torch.tensor([labels[p] for p in identity.gallery_pids])
    qids=torch.tensor([labels[p] for p in identity.query_pids])
    cmc,ap,_,order=rank(matrix,qids,gids,max_rank=10,get_mAP=True)
    return {"R@1":float(cmc[0]),"R@5":float(cmc[4]),"R@10":float(cmc[9]),"mAP":float(ap)},order

def audit(c):
    need(not torch.cuda.is_initialized(),"independent CPU audit")
    torch.set_num_threads(2)
    verify_official(c);source,col,m=source_inputs(c)
    from tbps.baselines.dra_full import validate_source
    validate_source(source,col,m)
    bridge_count=0
    with (source.root/"bridge_trace.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line)
            need(row["wordpiece_roundtrip_exact"] and not row["irra_truncated"],"candidate bridge mismatch")
            bridge_count+=1
    refs=[];rank=rank_function(c)
    for name in ("irra","rasa","aptm"):
        root=Path(c["output_base"])/"full"/name
        store=ArtifactStore(root,new_base=c["output_base"])
        need(not store.pending(),"pending target")
        for file in store.root.glob("*/manifest.json"):store.verify(file.parent.name)
        report=read_json(root/"execution/report.json")
        need(report["status"]=="PASS" and report["target_loads"]==1 and report["source_generation"]==0
             and report["model_before"]==report["model_after"] and report["backward"]==report["optimizer"]==0,
             "Target complete/frozen/no source")
        need(report["counts"]["image_samples"]==6148 and report["counts"]["text_samples"]==12312,"all native target inputs")
        summary=read_json(root/"completed/summary.json");protocol=read_json(root/"protocol/protocol.json")
        need(summary["scope"]=="full" and (summary["Gallery"],summary["Query"])==(3074,6156)
             and summary["matched_clean"] and summary["model"]==name and summary["source"]=="irra","Full summary")
        need(protocol["source_collection"]==asdict(ref(source.root/"completed/manifest.json"))
             and protocol["manifest"]==json.loads(json_bytes(asdict(m))) and protocol["target_feedback"] is False
             and protocol["ranking_feedback_to_attack"] is False,"same frozen artifacts")
        scores=torch.load(root/"completed/scores.pt",map_location="cpu",weights_only=True)
        need(set(scores)==set(MODES),"complete four native scores")
        for mode in MODES:
            actual,order=full_metrics(scores[mode].detach().clone(),m,rank)
            need(actual==summary["groups"][mode],"official CPU score/summary exact mismatch")
            del order
        for filename,rounded in (("metrics.csv",False),("metrics_2dp.csv",True)):
            with (root/"completed"/filename).open() as f:rows=list(csv.DictReader(f))
            need([v["Mode"] for v in rows]==list(MODES),"four-mode CSV order")
            for row in rows:
                for k in METRICS:
                    need(row[k]==format(summary["groups"][row["Mode"]][k],".2f") if rounded else
                         float(row[k])==summary["groups"][row["Mode"]][k],"CSV/summary")
        del scores
        refs.append(ref(root/"completed/manifest.json"))
        print("DRA_FULL_CPU_TARGET_AUDIT_PASS",name,flush=True)
    store=ArtifactStore(Path(c["output_base"])/"full/audit",new_base=c["output_base"])
    store.publish("completed",{"audit.json":json_bytes({"status":"PASS","scope":"full","Gallery":3074,"Query":6156,
        "batches":385,"source":"irra","target_feedback":False,"GPU":False,"bridge_rows":bridge_count,
        "official_core_sha256":c["official"]["attacker"]["sha256"],"no_resume":True})},
        {"kind":"DRA_independent_CPU_full_audit"},sources=(ref(c["path"]),ref(source.root/"completed/manifest.json"),*refs))
    print("DRA_INDEPENDENT_CPU_AUDIT_PASS full",flush=True)
    return 0
