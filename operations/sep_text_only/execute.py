"""复用 PIDPlan classifier、Source-only SEP 搜索与原生 Text-only 评分。"""
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import csv
import io
import json
import os
import sys
import time

from tbps.artifacts import digest, json_bytes, sha_file
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.runtime import ref,tensor_bytes,unit
from tbps.baselines.cpgc_sep.sep_source import classifier,EphemeralQuery
from tbps.baselines.cpgc_sep.victims import load_victim,frozen,global_features,encode_images,encode_texts,native_pixels
from tbps.baselines.cpgc_sep.cache import FeatureBank,NativeScorer
from tbps.baselines.cpgc_sep.target import cat_payload
from tbps.engines.metrics import ranking
from tbps.models.base import RepresentationBatch
from operations.sep_source_only import native
from . import word_budget as attack
from operations.sep_source_only.common import read,seed,train_manifest,tensor_sha
from operations.sep_source_only.phases import token_receipt
from .runtime import (config,store,metadata,source_inputs,selected_source,target_manifest,source_guard,
                      full_manifest,query_key,frozen_rows,event,DATASETS,VICTIMS)

METRICS=("R@1","R@5","R@10","mAP")


def protocol(c,s,body,**fields):
    s.publish("protocol",{"protocol.json":json_bytes(body)},{"kind":"SEP Text-only","config_sha":sha_file(c["config_path"]),**fields},
              sources=(ref(c["config_path"]),))


def prepare_gallery(c):
    import torch
    if c.get("reference_features"):
        old=Path(c["reference_features"])
        from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
        inherited=VerifiedArtifactStore(old,new_base=old.parents[1])
        inherited.verify("completed")
        report=read(old/"completed/report.json")
        require(report["status"]=="PASS" and report["gallery"]==34054 and report["pids"]==11003,
                "completed fixed Source reference")
        bundle=source_inputs(c)
        require(report["checkpoint"]==bundle["config"]["victims"]["irra"]["datasets"]["cuhk"]["checkpoint"],
                "same Source IRRA checkpoint")
        output=store(c,"source/reference")
        protocol(c,output,{"Source Train":c["source_train_manifest_sha256"],"reused":str(old)})
        output.publish("completed",{"features.pt":(old/"completed/features.pt").read_bytes(),
                       "report.json":json_bytes(report)},{"kind":"reuse query-independent Source Train features"},
                       sources=(ref(old/"completed/manifest.json"),ref(output.root/"protocol/manifest.json")))
        event(c,"SOURCE_REFERENCE_REUSED",source=str(old))
        return
    bundle=source_inputs(c)
    source_guard(c,bundle)
    cfg=bundle["config"]
    manifest=train_manifest(bundle["train_gallery"])
    output=store(c,"source/reference")
    protocol(c,output,{"Source Train":c["source_train_manifest_sha256"],"model":"CUHK IRRA"})
    seed()
    started=time.monotonic()
    with load_victim(cfg,"irra","cuhk",smoke=False) as (adapter,model,src):
        before=native.state_sha(model)
        bank=native.Gallery(adapter,"irra",bundle["train_root"],manifest.gallery,partial(event,c,purpose="Source Train"))
        require(before==native.state_sha(model),"Source model changed")
        output.publish("completed",{"features.pt":tensor_bytes(bank.globals),
                       "report.json":json_bytes({"status":"PASS","gallery":34054,"pids":11003,
                          "checkpoint":asdict(src.checkpoint),"model_state_sha256":before,
                          "feature_sha256":tensor_sha(bank.globals),"seconds":time.monotonic()-started})},
                       {"kind":"fixed Source Train reference features"},
                       sources=(ref(output.root/"protocol/manifest.json"),ref(store(c,"inputs").root/"source/manifest.json")))
    event(c,"SOURCE_REFERENCE_PASS",seconds=time.monotonic()-started)


def generate(c,ds,scope):
    import torch,nltk
    bundle=source_inputs(c)
    items=selected_source(c,ds,scope)
    # Remove all identity-bearing inputs from the process before the access guard.
    manifest=train_manifest(bundle["train_gallery"])
    reference=store(c,"source/reference")
    reference.verify("completed")
    gallery=torch.load(reference.root/"completed/features.pt",map_location="cpu",weights_only=True)
    require(list(gallery.shape)==[34054,512],"complete Source reference")
    cfg=bundle["config"]
    source_guard(c,bundle)
    seed()
    nltk.data.path[:]=[cfg["nltk_root"]]
    vocabulary=attack.resources(cfg)
    output=store(c,"source/"+ds)
    marker="completed-"+scope
    require(not (output.root/marker).exists(), "generation scope cannot restart")
    with load_victim(cfg,"irra","cuhk",smoke=False) as (adapter,model,src):
        before=native.state_sha(model)
        predictor=classifier(adapter,model,"irra",EphemeralQuery(gallery),manifest)
        newly_generated=0
        begin=time.monotonic()
        for index,item in enumerate(items):
            key=item["key"]
            if (output.root/key).exists():
                # Full imports only completed Smoke texts from this same protocol.
                require(scope=="full","Smoke may not retry")
                output.verify(key)
                existing=read(output.root/key/"text.json")
                require(existing["original"]==item["caption"],"Smoke reuse original mismatch")
                continue
            started=time.monotonic()
            log=partial(event,c,dataset=ds,scope=scope,query=index+1,key=key)
            log("QUERY_START",total=len(items))
            # A seed per immutable Caption permits Smoke reuse without changing
            # the RNG trajectory of any other Caption.
            import random,numpy as np
            query_seed=int(key[:8],16)
            random.seed(query_seed);np.random.seed(query_seed);torch.manual_seed(query_seed)
            require(not predictor.cache,"empty per-Caption cache")
            def producer():
                row,logits=attack.attack_caption(cfg,item["caption"],predictor,"cuda:0",vocabulary,log)
                row.update(key=key,seed=query_seed,cache_keys=list(predictor.cache),
                    tokenizer=token_receipt(adapter._tokenizer,[row["matched_clean"],row["selected"]]))
                return {"text.json":json_bytes(row),"clean_pid_logits.pt":tensor_bytes(logits)}
            unit(output,key,producer,{"key":key,"raw_sha":digest(item["caption"]),"protocol":c["protocol"]},
                 sources=(ref(reference.root/"completed/manifest.json"),ref(store(c,"inputs").root/("captions-"+ds)/"manifest.json")))
            row=read(output.root/key/"text.json")
            predictor.cache.clear()
            newly_generated+=1
            elapsed=time.monotonic()-begin
            event(c,"QUERY_COMPLETE",dataset=ds,scope=scope,done=index+1,total=len(items),
                  key=key,seconds=time.monotonic()-started,candidates=len(row["candidates"]),
                  forwards=row["text_forwards"],noop=row["noop"],
                  mean_new_query_seconds=elapsed/newly_generated,
                  estimated_remaining_source_seconds=(len(items)-index-1)*elapsed/newly_generated)
        require(before==native.state_sha(model),"Source model changed")
    index=[{"key":i["key"],"manifest_sha256":sha_file(output.root/i["key"]/"manifest.json")} for i in items]
    output.publish(marker,{"index.json":json_bytes(index),
                    "report.json":json_bytes({"status":"PASS","scope":scope,"dataset":ds,"queries":len(items),
                       "checkpoint":asdict(src.checkpoint),"DPP":{"N":1,"E":1,"weight":[1.]},
                       "seconds":time.monotonic()-begin,"new_queries":newly_generated,
                       "target_PID_access":False,"target_evaluation_feedback":False})},
                   {"kind":"frozen SEP text coverage","scope":scope,"dataset":ds},
                   sources=tuple(ref(output.root/i["key"]/"manifest.json") for i in items))
    event(c,"GENERATION_PASS",dataset=ds,scope=scope,queries=len(items))


def evaluate(c,base,ds,victim,scope):
    import torch
    manifest=target_manifest(c,ds,scope)
    source=store(c,"source/"+ds)
    source.verify("completed-"+scope)
    rows=frozen_rows(c,ds,manifest.queries)
    texts=[r["adversarial"] for r in rows]
    path=f"{scope}/target/{ds}/{victim}"
    output=store(c,path)
    require(not output.root.exists(),"new Target Attempt")
    started=time.monotonic()
    protocol(c,output,{"dataset":ds,"victim":victim,"scope":scope,"manifest":asdict(manifest),
                      "source":asdict(ref(source.root/("completed-"+scope)/"manifest.json")),
                      "mode":"Text-only","native_TopK":128})
    cfg={**base,"sha256":sha_file(c["config_path"]),"evaluation":{**base["evaluation"],
         "fusion_batch":c["fusion_batch"]},"output_base":c["output"]}
    ids_g=tuple(g.image_id for g in manifest.gallery)
    ids_q=tuple(q.query_id for q in manifest.queries)
    with load_victim(cfg,victim,ds,smoke=scope=="smoke") as (adapter,model,model_source):
        before=native.state_sha(model)
        gallery=FeatureBank(cfg,output,"clean_gallery",adapter,manifest,ids_g,
                            tuple(g.image_sha256 for g in manifest.gallery),modality="image",chunk=8)
        for i in range(0,len(ids_g),8):
            end=min(i+8,len(ids_g))
            def produce():
                payloads=[]
                for g in manifest.gallery[i:end]:
                    image=Path(base["datasets"][ds]["root"])/"imgs"/g.image_rel_path
                    require(sha_file(image)==g.image_sha256,"clean image changed")
                    payloads.append(encode_images(adapter,native_pixels(image,victim),victim))
                return cat_payload(payloads)
            gallery.put(i,end,produce)
            if end%128==0 or end==len(ids_g):
                event(c,"TARGET_GALLERY",scope=scope,dataset=ds,victim=victim,done=end,total=len(ids_g))
        gallery.read_globals()
        queries=FeatureBank(cfg,output,"adv_text",adapter,manifest,ids_q,tuple(digest(t) for t in texts),modality="text",chunk=8)
        for i in range(0,len(ids_q),8):
            end=min(i+8,len(ids_q))
            queries.put(i,end,lambda:encode_texts(adapter,texts[i:end],victim))
        queries.read_globals()
        scorer=NativeScorer(adapter,victim,{"query":queries,"gallery":gallery},store=output,config=cfg)
        matrices=[]
        q=RepresentationBatch(queries.globals,adapter.space.id,"text")
        g=RepresentationBatch(gallery.globals,adapter.space.id,"image")
        for i in range(0,len(ids_q),c["query_chunk"]):
            end=min(i+c["query_chunk"],len(ids_q))
            matrix=scorer.score(scorer.slice_query(q,i,end),g).payload
            require(matrix.dtype==torch.float32 and torch.isfinite(matrix).all(),"finite native scores")
            matrices.append(matrix)
            event(c,"TARGET_SCORE",scope=scope,dataset=ds,victim=victim,done=end,total=len(ids_q),
                  elapsed_seconds=time.monotonic()-started)
        scores=torch.cat(matrices)
        values,_=ranking(scores,manifest.evaluation_identity())
        frozen(model)
        require(before==native.state_sha(model),"Target model changed")
        require(victim!="rasa" or scorer.native_pairs==len(ids_q)*128,"native full Top128 fusion coverage")
    metrics={k:values[k] for k in METRICS}
    stream=io.StringIO()
    writer=csv.writer(stream)
    writer.writerow(("Dataset","Victim","Attack",*METRICS))
    writer.writerow((ds,victim,"SEP Text-only",*(metrics[k] for k in METRICS)))
    report={"status":"PASS","scope":scope,"dataset":ds,"victim":victim,"mode":"Text-only",
            "Gallery":len(ids_g),"Query":len(ids_q),"metrics":metrics,"checkpoint":asdict(model_source.checkpoint),
            "query_ids":list(ids_q),"frozen_text_sha256":[digest(x) for x in texts],
            "native_TopK":128 if victim=="rasa" else None,"native_ITM_pairs":scorer.native_pairs,
            "model_state_sha256":before,"elapsed_seconds":time.monotonic()-started}
    output.publish("completed",{"summary.json":json_bytes(report),"metrics.csv":stream.getvalue().encode(),
                   "scores.pt":tensor_bytes(scores)},{"kind":"complete Text-only evaluation","scope":scope},
                   sources=(ref(source.root/("completed-"+scope)/"manifest.json"),ref(output.root/"protocol/manifest.json")))
    event(c,"TARGET_PASS",scope=scope,dataset=ds,victim=victim,queries=len(ids_q),seconds=time.monotonic()-started)


def audit(c,ds,victim,scope):
    import torch,nltk
    require(not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES")=="","independent CPU audit")
    manifest=target_manifest(c,ds,scope)
    s=store(c,f"{scope}/target/{ds}/{victim}")
    s.verify("completed")
    summary=read(s.root/"completed/summary.json")
    require(summary["Query"]==len(manifest.queries) and summary["Gallery"]==len(manifest.gallery)
            and summary["query_ids"]==[q.query_id for q in manifest.queries],"complete ordered evaluation coverage")
    rows=frozen_rows(c,ds,manifest.queries)
    require(summary["frozen_text_sha256"]==[digest(r["adversarial"]) for r in rows],"same frozen text")
    bundle=source_inputs(c)
    cfg=bundle["config"]
    nltk.data.path[:]=[cfg["nltk_root"]]
    vocabulary=attack.resources(cfg)
    official=attack.sep_functions(cfg)["attack"]
    pid_plan=__import__("tbps.baselines.cpgc_sep.sep_bridge",fromlist=["PIDPlan"]).PIDPlan.build(
        tuple(g.pid for g in train_manifest(bundle["train_gallery"]).gallery))
    stats=[]
    for q,row in zip(manifest.queries,rows):
        source=store(c,"source/"+ds)
        logits=torch.load(source.root/query_key(ds,q.query_id)/"clean_pid_logits.pt",map_location="cpu",weights_only=True)
        label=int(logits[0].argmax())
        require(logits.shape==(1,11003) and torch.isfinite(logits).all() and
                row["pseudo_pid_index"]==label and row["pseudo_pid"]==pid_plan.order[label].value,"fixed Source pseudo PID")
        require(row["matched_clean"]==attack.prepare_caption(cfg,q.caption),"original SEP preprocessing")
        chosen=row["candidates"][0]["text"] if row["candidates"] else row["matched_clean"]
        require(row["selected"]==chosen and row["adversarial"]==attack.restore_original(q.caption,chosen),
                "official tau selection/raw NOOP")
        pos=official.__globals__["criteria"].get_pos(row["matched_clean"].split())
        require(row["word_edits"]==attack.verify_edits(row["matched_clean"],chosen,vocabulary,pos),"word budget/synonyms")
        legal=attack.eligible_positions(row["matched_clean"],vocabulary,pos)
        require(row["legal_positions"]==legal,"actual legal synonym coverage")
        if legal:
            require(row["DPP"]=={"N":1,"E":1,"draws":100,"weights":[[1.]]},"S1 DPP explicitly checked")
        else:
            require(row["noop"] and row["noop_reason"]=="NO_LEGAL_SYNONYM"
                    and row["DPP"]=={"N":1,"E":1,"draws":0,"weights":[]},"explicit no-candidate NOOP")
        require(row["word_budget"]==1 and row["budget_violations"]==0
                and row["word_positions"]==attack.budget_check(q.caption,row["adversarial"]),"original word budget")
        for request in row["requests"]:
            for text in request["texts"]:attack.budget_check(q.caption,text)
        for candidate in row["candidates"]:attack.budget_check(q.caption,candidate["text"])
        for timing in row["timings"]:
            if timing["stage"]=="OFFICIAL_OPTIMIZE":
                require(timing["max_change_num"]==timing["att_succ_num"]==1,"search budget evidence")
        stats.append(row["elapsed_seconds"])
    scores=torch.load(s.root/"completed/scores.pt",map_location="cpu",weights_only=True)
    values,_=ranking(scores,manifest.evaluation_identity())
    require(summary["metrics"]=={k:values[k] for k in METRICS},"independent full metrics")
    if scope=="full":
        expected={"cuhk":(3074,6156),"rstp":(1000,2000),"icfg":(19848,19848)}[ds]
        require((len(manifest.gallery),len(manifest.queries))==expected,"official Full size")
    for p in s.root.glob("*.Claim.json"):
        require((s.root/p.name.removesuffix(".Claim.json")/"manifest.json").exists(),"incomplete Target claim")
    report={"status":"PASS","scope":scope,"dataset":ds,"victim":victim,"Query":len(rows),
            "Gallery":len(manifest.gallery),"result":asdict(ref(s.root/"completed/manifest.json")),
            "mean_source_query_seconds":sum(stats)/len(stats),"source_total_seconds":sum(stats),
            "metrics":summary["metrics"],"target_seconds":summary["elapsed_seconds"],
            "checkpoint":summary["checkpoint"],"CUDA":False,
            "word_budget":1,"budget_violations":0,
            "one_word_queries":sum(not r["noop"] for r in rows),
            "noop_queries":sum(r["noop"] for r in rows),
            "noop_reasons":{reason:sum(r["noop_reason"]==reason for r in rows)
                            for reason in sorted({r["noop_reason"] for r in rows if r["noop"]})},
            "source_checkpoint":bundle["config"]["victims"]["irra"]["datasets"]["cuhk"]["checkpoint"],
            "source_forward_count":sum(r["text_forwards"] for r in rows)}
    out=store(c,f"{scope}/audit/{ds}/{victim}")
    out.publish("completed",{"report.json":json_bytes(report)},{"kind":"CPU integrity audit","scope":scope},
                sources=(ref(s.root/"completed/manifest.json"),))
    event(c,"AUDIT_PASS",scope=scope,dataset=ds,victim=victim,queries=len(rows),
          mean_source_query_seconds=report["mean_source_query_seconds"])


def main():
    import torch
    torch.set_num_threads(2)
    c,base=config()
    c["config_path"]=str(Path(__file__).resolve().parents[2]/"configs/framework/sep_text_only_full.server.json")
    action,*args=sys.argv[1:]
    if action=="prepare":
        from .runtime import prepare
        prepare(c,base)
    elif action=="reference":
        prepare_gallery(c)
    elif action=="source":
        generate(c,*args)
    elif action=="target":
        evaluate(c,base,*args)
    elif action=="audit":
        audit(c,*args)
    else:
        raise ValueError(action)


if __name__=="__main__":
    main()
