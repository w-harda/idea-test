"""固定2图4文Smoke评分；保留原排名门禁，Full仍使用原EvaluationEngine。"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import argparse, io, json, os, subprocess, sys
import torch
from tbps.artifacts import ArtifactStore, FileRef, checked, read_json, json_bytes, digest
from tbps.baselines import sga_irra as runner
from tbps.baselines.sga_official_io import load_file
from tbps.engines.evaluation import EvaluationEngine, MODES
from tbps.engines.materialized_cache import FeatureCache, CachedIRRA, ref
from tbps.contracts import AttackBudget,ImageBudget,TextBudget
from tbps.engines.provenance import AttackOrigin
from tbps.models.base import InputBinding
import hashlib

def config(path):
    x=read_json(path,"tbps-sga-two-gallery-smoke-1")
    checked(FileRef(**x["source_config"]))
    for p,h in x["producers"].items():checked(FileRef(p,h))
    c=runner.configuration(x["source_config"]["path"])
    c["path"]=str(Path(path).resolve());c["smoke_overlay"]=x
    return c

def official_rank(c):
    s=runner.IRRASource.from_dict(c["irra"])
    return load_file(s.ref("utils/metrics.py"),"_sga_smoke_official_metrics").rank

def small_ranking(matrix,identity,rank):
    runner.require(matrix.dtype==torch.float32 and matrix.device.type=="cpu" and matrix.shape==(4,2)
                   and torch.isfinite(matrix).all() and not matrix.requires_grad,"fixed two-gallery finite score")
    labels={pid:i for i,pid in enumerate(dict.fromkeys(identity.gallery_pids))}
    gids=torch.tensor([labels[p] for p in identity.gallery_pids]);qids=torch.tensor([labels[p] for p in identity.query_pids])
    cmc,ap,_,order=rank(matrix,qids,gids,max_rank=2,get_mAP=True)
    return {"R@1":float(cmc[0]),"mAP":float(ap)},order

class SmokeEngine(EvaluationEngine):
    rank=None
    def run(self,inputs,**unused):
        modes=tuple(MODES);self._validate(inputs,modes)
        runner.require(len(self.identity.gallery_ids)==2 and len(self.identity.query_ids)==4,"no Full through smoke bridge")
        groups={};scores={}
        with torch.inference_mode():
            for mode,(q,g) in MODES.items():
                query=inputs[q].batch;gallery=inputs[g].batch
                sliced=self.adapter.slice_query(query,0,4)
                value=self.adapter.score(sliced,gallery)
                runner.require(value.space_id==self.space.id and value.scoring_protocol_sha256==self.space.score_id,"native score binding")
                matrix=value.payload.detach().cpu().clone()
                groups[mode],_=small_ranking(matrix,self.identity,self.rank)
                scores[mode]=matrix
        self._validate(inputs,modes)
        return {},{"groups":groups},scores

@contextmanager
def smoke_engine(c):
    class NativeSmoke(SmokeEngine):pass
    NativeSmoke.rank=staticmethod(official_rank(c))
    with patch.object(runner,"EvaluationEngine",NativeSmoke),patch.object(runner,"METRICS",("R@1","mAP")):
        yield

def recover_irra_scores(c):
    # 只读取失败评分前已完整提交的特征，不重新加载/编码IRRA。
    base=Path(c["output_base"])/"smoke";old=ArtifactStore(base/"irra",new_base=c["output_base"])
    old.verify("execution");failure=read_json(old.root/"execution/report.json")
    runner.require(failure["status"]=="FAIL" and failure["error"]=="R@10 requires at least 10 Gallery members","exact ranking-only failure")
    old.verify("protocol");protocol=read_json(old.root/"protocol/protocol.json")
    src,col,manifest=runner.source_inputs(c,"smoke")
    from tbps.models.base import ModelSpace
    space=ModelSpace(**protocol["model_space"])
    checkpoint=FileRef(**c["irra"]["checkpoint"])
    cache=FeatureCache(old,protocol,space,chunk=16);inputs={}
    for family in ("clean_gallery","adv_gallery","clean_text","adv_text"):
        file=old.root/cache.key(family,0)/"manifest.json";md=read_json(file)
        binding_data=md["provenance"]["binding"]
        binding=InputBinding(**{**binding_data,"ordered_ids":tuple(binding_data["ordered_ids"]),
                                "ordered_sha256":tuple(binding_data["ordered_sha256"])})
        image=family.endswith("gallery");adv=family.startswith("adv")
        originals=tuple(g.image_sha256 for g in manifest.gallery) if image else tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in manifest.queries)
        budget=AttackBudget(image=ImageBudget("linf",2/255,"rgb_0_1")) if image else AttackBudget(text=TextBudget("word",1,None,"SGA_official_"+runner.OFFICIAL_COMMIT))
        origin=AttackOrigin("sga_official_irra_"+("image" if image else "text"),checkpoint,
                           ref(src.root/"protocol/manifest.json"),budget,FileRef(**c["smoke_overlay"]["source_config"]).sha256) if adv else None
        inputs[family]=cache.gather(family,binding,origin=origin,originals=originals if adv else ())
    adapter=CachedIRRA(space,checkpoint,cache_protocol=ref(old.root/"protocol/manifest.json"))
    with smoke_engine(c):
        e=runner.EvaluationEngine(manifest,adapter,direction="higher_better",execution="readonly_cached",model_sources=(checkpoint,))
        _,result,scores=e.run(inputs)
    store=ArtifactStore(base/"irra_smoke_scores",new_base=c["output_base"])
    b=io.BytesIO();torch.save(scores,b)
    store.publish("completed",{"summary.json":json_bytes({"status":"PASS","Gallery":2,"Query":4,"groups":result["groups"],
        "R@5":"N/A: only 2 Gallery","R@10":"N/A: only 2 Gallery","model_loads_this_recovery":0,
        "image_forward_this_recovery":0,"text_forward_this_recovery":0,"original_target_attempt_retained":"FAIL"}),
        "scores.pt":b.getvalue()},{"kind":"SGA_two_gallery_CPU_score_recovery"},
        sources=(ref(c["path"]),ref(old.root/"execution/manifest.json"),*(ref(old.root/cache.key(k,0)/"manifest.json") for k in inputs)))
    print("SGA_TWO_GALLERY_IRRA_CPU_SCORES_PASS",flush=True)

def audit(c):
    base=Path(c["output_base"])/"smoke"
    for name in ("source","irra","irra_smoke_scores","rasa","aptm"):
        store=ArtifactStore(base/name,new_base=c["output_base"])
        runner.require(not store.pending(),"pending")
        for f in store.root.glob("*/manifest.json"):store.verify(f.parent.name)
    src,col,m=runner.source_inputs(c,"smoke")
    md=read_json(src.root/col["units"][0]/"metadata.json")
    runner.require(md["flow"]==["text1","image","text2"] and md["gallery_ids"]==[g.image_id for g in m.gallery],"official ordered source")
    trace=[json.loads(x) for x in (src.root/"bridge_trace.jsonl").read_text().splitlines()]
    runner.require(all(x["wordpiece_roundtrip_exact"] and not x["irra_truncated"] for x in trace),"text bridge exact")
    for name in ("irra_smoke_scores","rasa","aptm"):
        s=read_json(base/name/"completed/summary.json")
        runner.require(s["Gallery"]==2 and s["Query"]==4 and set(s["groups"])==set(MODES),"complete fixed scope")
        scores=torch.load(base/name/"completed/scores.pt",map_location="cpu",weights_only=True)
        for mode,x in scores.items():
            val,_=small_ranking(x,m.evaluation_identity(),official_rank(c))
            runner.require(val==s["groups"][mode],"score/metric mismatch")
    store=ArtifactStore(base/"smoke_audit",new_base=c["output_base"])
    store.publish("completed",{"audit.json":json_bytes({"status":"PASS","Gallery":2,"Query":4,
        "original_irra_ranking_attempt":"FAIL retained; complete encoded features CPU reused",
        "R@5_R@10":"NOT_APPLICABLE","additional_source_generation":0,"additional_IRRA_model_loading":0,
        "bridge_rows":len(trace),"GPU_audit":False})},{"kind":"SGA_two_gallery_CPU_audit"},
        sources=(ref(c["path"]),*(ref(base/n/"completed/manifest.json") for n in ("source","irra_smoke_scores","rasa","aptm"))))
    print("SGA_OFFICIAL_IRRA_TWO_GALLERY_CPU_AUDIT_PASS",flush=True)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--action",choices=("recover-irra","evaluate","audit"),required=True)
    p.add_argument("--model",choices=("rasa","aptm"));p.add_argument("--allow-gpu-smoke",action="store_true")
    a=p.parse_args(argv);c=config(a.config)
    if a.action=="recover-irra":runner.require(not torch.cuda.is_initialized(),"CPU recovery");recover_irra_scores(c);return 0
    if a.action=="audit":runner.require(not torch.cuda.is_initialized(),"CPU audit");audit(c);return 0
    runner.require(a.allow_gpu_smoke and a.model is not None,"explicit only fixed smoke")
    with smoke_engine(c):return runner.encode_target(c,"smoke",a.model)
if __name__=="__main__":raise SystemExit(main())
