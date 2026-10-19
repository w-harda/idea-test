"""DRA冻结产物的三模型原生Smoke评分；复用现有缓存/Engine验证与官方rank。"""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import io
import time
import traceback
import torch
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,digest,sha_file
from tbps.baselines.dra_official import COMMIT,MODES,need,projection_check,verify_official
from tbps.baselines.dra_irra import source_inputs
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

def smoke_metrics(matrix,manifest,rank):
    need(matrix.shape==(16,8) and matrix.dtype==torch.float32 and matrix.device.type=="cpu"
         and torch.isfinite(matrix).all() and not matrix.requires_grad,"fixed eight-image score matrix")
    identity=manifest.evaluation_identity()
    labels={pid:i for i,pid in enumerate(dict.fromkeys(identity.gallery_pids))}
    need(all(pid in labels for pid in identity.query_pids),"all smoke Query have positive Gallery")
    gids=torch.tensor([labels[p] for p in identity.gallery_pids])
    qids=torch.tensor([labels[p] for p in identity.query_pids])
    cmc,ap,_,order=rank(matrix,qids,gids,max_rank=8,get_mAP=True)
    return {"R@1":float(cmc[0]),"R@5":float(cmc[4]),"mAP":float(ap)},order

class EightGalleryEngine(EvaluationEngine):
    """仅固定8/16 Smoke的官方max_rank8桥；不放宽通用Full门禁、不伪造R@10。"""
    def run_smoke(self,inputs,rank):
        self._validate(inputs,MODES)
        need((len(self.manifest.gallery),len(self.manifest.queries))==(8,16),"small Smoke only")
        groups={};scores={}
        with torch.inference_mode():
            for mode in MODES:
                q,g=MODE_INPUTS[mode]
                query=self.adapter.slice_query(inputs[q].batch,0,16)
                value=self.adapter.score(query,inputs[g].batch)
                need(value.space_id==self.space.id and value.scoring_protocol_sha256==self.space.score_id,"native score source")
                x=value.payload.detach().cpu().clone()
                groups[mode],_=smoke_metrics(x,self.manifest,rank);scores[mode]=x
        self._validate(inputs,MODES)
        return groups,scores

def evaluate(c,model):
    source,col,manifest=source_inputs(c)
    root=Path(c["output_base"])/"smoke"/c.get("target_output_names",{}).get(model,model)
    need(not root.exists(),"existing Target Attempt; no retry/overwrite")
    from tbps.baselines.dra_irra import resources
    snapshot=resources(c)
    store=ArtifactStore(root,new_base=c["output_base"])
    loader={"inputs":{"dataset_sources":c["dataset_sources"]},"victim_sources":c["victim_sources"]}
    report={"status":"RUNNING","model":model,"target_loads":0,"source_generation":0,"backward":0,"optimizer":0,
            "resources_before":snapshot}
    start=time.monotonic()
    try:
        torch.set_num_threads(2)
        with fp32_runtime(),load_model(loader,model,"smoke") as (adapter,backend,checkpoint,counts):
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
                hashes=[];parts=[]
                if is_image:
                    for key in col["units"]:
                        source.verify(key)
                        raw=torch.load(source.root/key/"pixels.pt",map_location="cpu",weights_only=True)
                        xs=raw["adversarial" if adv else "original"].detach().clone();del raw
                        for x in xs:
                            hashes.append(payload_digest(x.unsqueeze(0).clone()) if adv else originals[len(hashes)])
                            value=adapter.inference().encode_images(x.unsqueeze(0).to("cuda:0")).payload.cpu() if model=="irra" else adapter.encode_images(x.unsqueeze(0))
                            parts.append(compact(value))
                else:
                    texts=[col["adv_texts"][q.query_id] if adv else q.caption for q in manifest.queries]
                    hashes=[hashlib.sha256(x.encode()).hexdigest() for x in texts]
                    for i in range(0,len(texts),16):
                        value=adapter.inference().encode_texts(texts[i:i+16])[0].payload.cpu() if model=="irra" else adapter.encode_texts(texts[i:i+16])
                        parts.append(compact(value))
                all_values=payload_cat(parts);del parts
                binding=InputBinding(adapter.space.id,digest(asdict(manifest)),tuple(hashes),"image" if is_image else "text",ids)
                for i in range(0,len(ids),16):
                    end=min(i+16,len(ids));cache.publish(family,binding,i,end,payload_select(all_values,list(range(i,end))))
                budget=AttackBudget(image=ImageBudget("linf",8/255,"rgb_0_1")) if is_image else AttackBudget(text=TextBudget("word",1,None,"DRA_official_"+COMMIT))
                origin=AttackOrigin("vlptattack_dra_"+("image" if is_image else "text"),FileRef(**c["irra"]["checkpoint"]),
                                   ref(source.root/"protocol/manifest.json"),budget,sha_file(c["path"])) if adv else None
                inputs[family]=cache.gather(family,binding,origin=origin,originals=originals if adv else ())
                print("DRA_TARGET_ENCODING",model,family,len(ids),flush=True)
            scorer=CachedIRRA(adapter.space,checkpoint,cache_protocol=ref(root/"protocol/manifest.json")) if model=="irra" else (
                ChunkCachedScorer if model=="rasa" else APTMSetScorer)(adapter,inputs,store,ref(root/"protocol/manifest.json"))
            engine=EightGalleryEngine(manifest,scorer,direction="higher_better",
                execution="readonly_cached" if model=="irra" else "materialized_native",model_sources=(checkpoint,))
            groups,scores=engine.run_smoke(inputs,rank_function(c))
            need(model_snapshot(backend)==before and not backend.training and all(not p.requires_grad and p.grad is None for p in backend.parameters()),"Target weights/buffers/freeze")
            b=io.StringIO();writer=csv.writer(b,lineterminator="\n");writer.writerow(("Mode","R@1","R@5","mAP"))
            for mode in MODES:writer.writerow((mode,*(format(groups[mode][k],".2f") for k in ("R@1","R@5","mAP"))))
            data=io.BytesIO();torch.save(compact(scores),data)
            summary={"scope":"smoke","Gallery":8,"Query":16,"groups":groups,"R@10":None,
                "R@10_reason":"8 Gallery cannot measure R@10; no padding or fabricated members",
                "matched_clean":True,"model":model,"source":"irra","metric_unit":"percent"}
            store.publish("completed",{"summary.json":json_bytes(summary),"metrics_2dp.csv":b.getvalue().encode(),"scores.pt":data.getvalue()},
                {"kind":"DRA_native_smoke_result"},sources=(ref(root/"protocol/manifest.json"),))
            report.update(status="PASS",model_before=before,model_after=model_snapshot(backend),counts=dict(counts),
                          neural_counts=dict(counts if model=="irra" else adapter.neural_counts))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc());print(report["traceback"],flush=True)
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss_bytes())
        if torch.cuda.is_initialized():report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if not (root/"protocol").exists():store.publish("protocol",{"protocol.json":json_bytes({"model":model})},{"kind":"failed_target"},sources=(ref(c["path"]),))
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"DRA_target_execution"},sources=(ref(root/"protocol/manifest.json"),))
    print("DRA_NATIVE_TARGET_"+report["status"],model,flush=True)
    return int(report["status"]!="PASS")

def audit(c):
    need(not torch.cuda.is_initialized(),"CPU audit only")
    verify_official(c);source,col,m=source_inputs(c)
    base=Path(c["output_base"])/"smoke"
    for name in ("source","irra","rasa","aptm"):
        store=ArtifactStore(base/(name if name=="source" else c.get("target_output_names",{}).get(name,name)),new_base=c["output_base"])
        need(not store.pending(),"pending publication")
        for file in store.root.glob("*/manifest.json"):store.verify(file.parent.name)
        report=read_json(store.root/"execution/report.json")
        need(report["status"]=="PASS" and report["model_before"]==report["model_after"],"stage incomplete or model changed")
    report=read_json(source.root/"execution/report.json")
    need(report["official_flow"]["counts"]["backward"]==100 and report["optimizer"]==0 and
         report["MLM_before"]==report["MLM_after"],"source official backward and fixed weights")
    qids=[];gids=[]
    for key in col["units"]:
        raw=torch.load(source.root/key/"pixels.pt",map_location="cpu",weights_only=True)
        projection_check(raw["original"],raw["adversarial"])
        need(torch.equal(raw["adversarial"]-raw["original"],raw["effective_delta"]),"effective delta")
        md=read_json(source.root/key/"metadata.json")
        claim=read_json(source.root/("claim-"+key)/"claim.json")
        need(claim["gallery_ids"]==md["gallery_ids"] and claim["query_ids"]==md["query_ids"],"Claim identity")
        gids.extend(md["gallery_ids"]);qids.extend(md["query_ids"])
        need(md["flow"]["events"]==["text1","image","text2"],"official three stages")
        for q,t in zip(md["query_ids"],md["adv_texts"]):need(col["adv_texts"][q]==t,"final text")
    need(gids==[g.image_id for g in m.gallery] and qids==[q.query_id for q in m.queries],"official order")
    bridge_count=0
    with (source.root/"bridge_trace.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);need(row["wordpiece_roundtrip_exact"] and not row["irra_truncated"],"candidate bridge mismatch")
            bridge_count+=1
    source_refs=[]
    for name in ("irra","rasa","aptm"):
        root=base/c.get("target_output_names",{}).get(name,name);s=read_json(root/"completed/summary.json")
        protocol=read_json(root/"protocol/protocol.json")
        need(protocol["source_collection"]==asdict(ref(source.root/"completed/manifest.json")) and
             protocol["manifest"]==json.loads(json_bytes(asdict(m))) and protocol["target_feedback"] is False,"same frozen input source")
        scores=torch.load(root/"completed/scores.pt",map_location="cpu",weights_only=True)
        for mode in MODES:
            val,_=smoke_metrics(scores[mode].detach().clone(),m,rank_function(c))
            need(val==s["groups"][mode],"CPU score/summary mismatch")
        csvrows=list(csv.DictReader((root/"completed/metrics_2dp.csv").open()))
        need([r["Mode"] for r in csvrows]==list(MODES),"four-mode order")
        for row in csvrows:
            need(all(row[k]==format(s["groups"][row["Mode"]][k],".2f") for k in ("R@1","R@5","mAP")),"CSV mismatch")
        source_refs.append(ref(root/"completed/manifest.json"))
    store=ArtifactStore(base/"audit",new_base=c["output_base"])
    store.publish("completed",{"audit.json":json_bytes({"status":"PASS","Gallery":8,"Query":16,"GPU":False,
        "bridge_rows":bridge_count,"R@10":"not applicable","source":"irra","target_feedback":False,
        "official_core_sha256":c["official"]["attacker"]["sha256"]})},{"kind":"DRA_independent_CPU_audit"},
        sources=(ref(c["path"]),ref(source.root/"completed/manifest.json"),*source_refs))
    print("DRA_INDEPENDENT_CPU_AUDIT_PASS smoke",flush=True)
    return 0
