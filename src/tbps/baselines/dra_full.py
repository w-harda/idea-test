"""官方DRA Full外层调度：单次IRRA/MLM、全局seed42、385批不可变提交；算法不复制。"""
from pathlib import Path
from dataclasses import asdict
from importlib.metadata import version
import argparse
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import time
import traceback
import numpy as np
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,sha_file,digest
from tbps.baselines.dra_official import COMMIT,PARAMETERS,MODES,need,verify_official,modules,ordered_dataset,batch_members,source_pixels,projection_check,PassiveTrace
from tbps.models.dra_irra_source import IRRADRASource,AuditedBert
from tbps.models.irra import IRRAAdapter
from tbps.models.irra_source import IRRASource
from tbps.models.irra_io import MEAN,STD
from tbps.engines.materialized_cache import ref
from tbps.engines.native_cache import compact
from tbps.baselines.sga_irra import manifest_from_dict
from tbps.smoke_irra import model_snapshot
from tbps.engines.replay_control import rss_bytes

from tbps.baselines.dra_irra import save_pixels,configuration as base_configuration,resources

METRICS=("R@1","R@5","R@10","mAP")

def configuration(path):
    c=base_configuration(path)
    need(c["full"]=={"Gallery":3074,"Query":6156,"batches":385,"batch_size":8,
        "seed":42,"query_chunk":128,"resume":False,"target_order":["irra","rasa","aptm"]},
        "fixed Full execution contract")
    need(c["minimum_free_mib"]==20480 and c["minimum_RAM_bytes"]==48*1024**3
         and c["minimum_disk_bytes"]==35*1024**3,"frozen Full resources")
    checked(FileRef(**c["dataset_sources"]))
    for x in c["victim_sources"].values():checked(FileRef(**x))
    for x in c["smoke_evidence"].values():checked(FileRef(**x))
    need(read_json(c["smoke_evidence"]["audit"]["path"])["status"]=="PASS","Smoke CPU audit")
    for name in ("source","irra","rasa","aptm"):
        report=read_json(c["smoke_evidence"][name]["path"])
        need(report["status"]=="PASS" and report["model_before"]==report["model_after"],"Smoke stage/model")
    return c

def schedule(manifest):
    batches=[batch_members(manifest,i,8) for i in range(0,len(manifest.gallery),8)]
    need([g.image_id for gs,_,_ in batches for g in gs]==[g.image_id for g in manifest.gallery],"Gallery order")
    need([q.query_id for _,qs,_ in batches for q in qs]==[q.query_id for q in manifest.queries],"Query order")
    return batches

def preflight(path):
    from unittest.mock import patch
    def forbidden(*a,**k):raise RuntimeError("Full CPU preflight prohibits model load/CUDA")
    with patch.object(torch,"load",forbidden),patch.object(torch.cuda,"_lazy_init",forbidden):
        c=configuration(path)
        IRRASource.from_dict(c["irra"]).verify()
        for model in ("rasa","aptm"):
            checked(FileRef(**c["victim_sources"][model]))
            mod=__import__("tbps.models."+model,fromlist=["native_modules"])
            src=(mod.RasaSource if model=="rasa" else mod.APTMSource)(c["victim_sources"][model]["path"])
            src.verify()
            with mod.native_modules(src):pass
        ds,raw,matched=ordered_dataset(c,"full")
        batches=schedule(matched)
        sizes=[len(batch_members(matched,i,1)[1]) for i in range(3074)]
        need(len(batches)==385 and len(raw.queries)==6156 and sizes.count(10)==1 and sizes.count(2)==3073,
             "official Full identities and ten-Caption member")
        for g in raw.gallery:
            checked(FileRef(str(ds.image_path(g.image_rel_path)),g.image_sha256))
    need(not torch.cuda.is_initialized(),"CPU-only preflight")
    report={"Gallery":3074,"Query":6156,"batches":385,"last_batch_Gallery":len(batches[-1][0]),
            "ten_caption_gallery":raw.gallery[sizes.index(10)].image_id,
            "model_loads":0,"GPU":False,"resume":False}
    print("DRA_FULL_CPU_PREFLIGHT_PASS",json.dumps(report),flush=True)
    return c,ds,raw,matched,report

def rng_state():
    # 只读取状态；不重新设种子、不消耗随机数。
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state().clone(),
            "cuda":[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []}

def rng_bytes(state):
    stream=io.BytesIO();torch.save(state,stream);return stream.getvalue()

def source_run(path):
    c,ds,raw,matched,pf=preflight(path)
    need({k:version(k) for k in c["source_versions"]}==c["source_versions"],"pinned dedicated DRA runtime")
    root=Path(c["output_base"])/"full/source"
    need(not root.exists(),"existing DRA Source Attempt; no retry/overwrite")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit GPU0")
    snapshot=resources(c)
    store=ArtifactStore(root,new_base=c["output_base"])
    store.publish("protocol",{"protocol.json":json_bytes({"source":"irra","official":verify_official(c),
        "parameters":PARAMETERS,"raw_manifest":asdict(raw),"matched_manifest":asdict(matched),
        "config":asdict(ref(path)),"target_feedback":False,"CPU_preflight":pf,
        "source_preprocess":"IRRA native PIL RGB bilinear384x128; official Normalize; pre_caption30",
        "parameter_gradients":"enabled for unmodified official scoring backward; no weight update"})},
        {"kind":"DRA_IRRA_source"},sources=(ref(path),FileRef(**c["irra"]["checkpoint"])))
    report={"status":"RUNNING","scope":"full","source_model_loads":0,"MLM_loads":0,"target_models_in_generation":[],
            "optimizer":0,"G_H_loads":0,"resources_before":snapshot,"official_commit":COMMIT}
    start=time.monotonic()
    try:
        from torchvision import transforms
        from transformers import BertForMaskedLM
        torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);random.seed(42)
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        adapter=IRRAAdapter.from_source(IRRASource.from_dict(c["irra"]),device="cuda:0",allow_model_load=True)
        report["source_model_loads"]=1
        official,tok,_=modules(c)
        tokenizer=AuditedBert(tok.BertTokenizer.from_pretrained(c["bert"],local_files_only=True,clean_up_tokenization_spaces=True))
        mlm=BertForMaskedLM.from_pretrained(c["bert"],local_files_only=True).float().eval().requires_grad_(False).to("cuda:0")
        report["MLM_loads"]=1
        before=model_snapshot(adapter._backend);mlm_before=model_snapshot(mlm)
        report.update(model_before=before,MLM_before=mlm_before)
        tracefile=root/"bridge_trace.jsonl";flowfile=root/"official_flow.jsonl"
        with tracefile.open("x") as bridge,flowfile.open("x") as flow:
            def write_bridge(row):bridge.write(json.dumps(row,ensure_ascii=False)+"\n")
            def write_flow(row):flow.write(json.dumps(row,ensure_ascii=False)+"\n")
            source=IRRADRASource(adapter._backend,adapter._tokenizer,tokenizer,write_trace=write_bridge)
            counts={"calls":0,"samples":0}
            def observe_mlm(module,args):
                counts["calls"]+=1;counts["samples"]+=len(args[0])
            hook=mlm.register_forward_pre_hook(observe_mlm)
            image=official.ImageAttacker(transforms.Normalize(MEAN,STD),eps=8/255,steps=10,step_size=2/255,sample_numbers=5)
            text=official.TextAttacker(mlm,tokenizer,cls=False,max_length=77,number_perturbation=1,topk=10,
                                      threshold_pred_score=.3,text_ratios=[.6,.2,.2])
            attacker=official.Attacker(source,image,text)
            need(attacker.attack.__func__ is official.Attacker.attack and
                 image.txt_guided_attack.__func__ is official.ImageAttacker.txt_guided_attack and
                 text.img_guided_attack.__func__ is official.TextAttacker.img_guided_attack,"unmodified official bound methods")
            units=[];all_texts={};edited=0;noop=0;max_linf=0.;attack_total=0.;image_total=0.
            batch_count=0
            for number,(gallery,queries,mapping) in enumerate(schedule(matched)):
                key="batch-"+str(number).zfill(6)
                print("DRA_SOURCE_BATCH_BEGIN",number+1,"/ 385","Gallery",len(gallery),"Query",len(queries),flush=True)
                store.publish("claim-"+key,{"claim.json":json_bytes({"gallery_ids":[g.image_id for g in gallery],
                    "query_ids":[q.query_id for q in queries]})},{"kind":"DRA_claim"},sources=(ref(root/"protocol/manifest.json"),))
                pixels=torch.stack([source_pixels(ds.image_path(g.image_rel_path)) for g in gallery]).to("cuda:0")
                texts=[q.caption for q in queries]
                trace=PassiveTrace(c["official"]["attacker"]["path"],source,write_flow)
                state_before=rng_state()
                timer=time.monotonic()
                with trace.active():
                    adv,adv_texts,official_image_seconds=attacker.attack(pixels,texts,mapping,device="cuda:0",max_length=77,scales=PARAMETERS["scales"])
                attack_seconds=time.monotonic()-timer
                state_after=rng_state()
                actual=trace.report()
                need(actual["events"]==["text1","image","text2"] and actual["counts"]["call:rand3Num"]==45 and
                     actual["counts"]["backward"]==100 and len(actual["selected_ratios"])==9 and actual["all_image_gradients_finite"],
                     "official full three-stage/region/backward execution")
                check=projection_check(pixels,adv)
                gradients=[p.grad for p in source.backend.parameters() if p.grad is not None]
                need(gradients and all(torch.isfinite(g).all() for g in gradients),"finite official source parameter gradients")
                report["parameter_gradients_present"]=len(gradients)
                del gradients
                cpu={"original":pixels.detach().cpu().clone(),"adversarial":adv.detach().cpu().clone(),
                     "effective_delta":(adv-pixels).detach().cpu().clone()}
                md={"gallery_ids":[g.image_id for g in gallery],"query_ids":[q.query_id for q in queries],"txt2img":mapping,
                    "clean_texts":texts,"adv_texts":adv_texts,"T1":actual["T1"],"flow":actual,"budget":check,
                    "seconds":attack_seconds,"official_image_seconds":official_image_seconds,
                    "edits":[]}
                actions=[r for r in actual["text_actions"] if r["stage"]==2]
                need([r["query_local_index"] for r in actions]==list(range(len(queries))),"official final action order")
                for q,old,new,action in zip(queries,texts,adv_texts,actions):
                    before_slots=action["original_word_slots"];after_slots=action["final_word_slots"]
                    changed=[i for i,(a,b) in enumerate(zip(before_slots,after_slots)) if a!=b]
                    need(" ".join(before_slots)==old and " ".join(after_slots)==new and len(before_slots)==len(after_slots)
                         and len(changed)==action["official_change_count"]<=1,"official original-word-slot budget")
                    md["edits"].append({"query_id":q.query_id,"before":old,"after":new,"changed_positions":changed,
                        "official_change_count":action["official_change_count"],"original_word_slots":before_slots,
                        "final_word_slots":after_slots,"space_word_count_delta":len(new.split(" "))-len(old.split(" ")),"NOOP":old==new})
                rb=rng_bytes(state_before);ra=rng_bytes(state_after)
                store.publish("rng-"+key,{"before.pt":rb,"after.pt":ra},
                    {"kind":"DRA_RNG_evidence","resume_authorized":False},
                    sources=(ref(root/("claim-"+key)/"manifest.json"),))
                md["rng_before_sha256"]=hashlib.sha256(rb).hexdigest()
                md["rng_after_sha256"]=hashlib.sha256(ra).hexdigest()
                md["rng_before_state"]=rng_digest(state_before);md["rng_after_state"]=rng_digest(state_after)
                save_pixels(store,key,cpu,md,(ref(root/"protocol/manifest.json"),
                    ref(root/("claim-"+key)/"manifest.json"),ref(root/("rng-"+key)/"manifest.json")))
                units.append(key);all_texts.update(zip([q.query_id for q in queries],adv_texts))
                edited+=sum(not e["NOOP"] for e in md["edits"]);noop+=sum(e["NOOP"] for e in md["edits"])
                max_linf=max(max_linf,check["real_linf"]);attack_total+=attack_seconds;image_total+=official_image_seconds
                batch_count+=1
                report.update(completed_batches=batch_count,completed_Gallery=min(batch_count*8,3074),completed_Query=len(all_texts))
                source.zero_grad(set_to_none=True)
                bridge.flush();flow.flush();os.fsync(bridge.fileno());os.fsync(flow.fileno())
                print("DRA_SOURCE_PROGRESS",json.dumps({"batches":batch_count,"total_batches":385,
                    "Gallery":sum(min(8,3074-i*8) for i in range(batch_count)),"Query":len(all_texts),
                    "elapsed_seconds":time.monotonic()-start,"peak_RSS_bytes":rss_bytes(),
                    "GPU_allocated_bytes":torch.cuda.memory_allocated(),"GPU_reserved_bytes":torch.cuda.memory_reserved()}),flush=True)
                del pixels,adv,cpu,trace,actual,md,state_before,state_after,rb,ra
            source.zero_grad(set_to_none=True)
            source.backend.requires_grad_(False)
            need(model_snapshot(source.backend)==before and model_snapshot(mlm)==mlm_before,"source weights/buffers changed")
            need(not source.training and not source.backend.training and not mlm.training,"source eval state")
            source.zero_grad(set_to_none=True)
            report.update(status="PASS",model_after=model_snapshot(source.backend),MLM_after=model_snapshot(mlm),
                source_counts=source.counts,MLM_counts=counts,source_weights_fixed=True,parameter_updates=0,parameter_gradients_during_official_attack=True,
                official_attack_calls=batch_count,official_backwards=batch_count*100,region_samples=batch_count*45,
                maximum_real_linf=max_linf,attack_seconds=attack_total,official_image_seconds=image_total,
                edited_queries=edited,NOOP=noop,
                parameter_grads_cleared_after_attack=all(p.grad is None for p in source.parameters()))
            hook.remove()
            bridge.flush();flow.flush()
            need(batch_count==385 and len(all_texts)==6156,"complete Source set")
            collection={"manifest":asdict(matched),"raw_manifest":asdict(raw),"units":units,
                "adv_texts":all_texts,"official_source_calls":batch_count,"Gallery":3074,"Query":6156,
                "rng_policy":"one global seed42 before model loading; no per-batch seed; no resume"}
        validate_source(store,collection,matched)

        verify_official(c)
        store.publish("completed",{"collection.json":json_bytes(collection)},{"kind":"DRA_IRRA_source_complete"},
            sources=(ref(root/"protocol/manifest.json"),*(ref(root/k/"manifest.json") for k in units),ref(tracefile),ref(flowfile)))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss_bytes())
        if torch.cuda.is_initialized():report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"DRA_source_execution"},sources=(ref(root/"protocol/manifest.json"),))
    print("DRA_OFFICIAL_IRRA_SOURCE_"+report["status"],"full",flush=True)
    return int(report["status"]!="PASS")


def rng_digest(state):
    n=state["numpy"]
    return digest({"python":state["python"],"numpy":[n[0],n[1].tolist(),*n[2:]],
                   "torch":state["torch"].tolist(),"cuda":[v.tolist() for v in state["cuda"]]})

def validate_source(store,col,m):
    need(not store.pending(),"pending Source publication")
    expected=["batch-"+str(i).zfill(6) for i in range(len(schedule(m)))]
    need(col["units"]==expected and col["Gallery"]==len(m.gallery) and col["Query"]==len(m.queries),"complete Source units")
    gids=[];qids=[];last_rng=None
    for key,(gs,qs,mapping) in zip(expected,schedule(m)):
        for name in (key,"claim-"+key,"rng-"+key):store.verify(name)
        md=read_json(store.root/key/"metadata.json")
        raw=torch.load(store.root/key/"pixels.pt",map_location="cpu",weights_only=True)
        need(set(raw)=={"original","adversarial","effective_delta"} and raw["original"].shape==(len(gs),3,384,128)
             and all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in raw.values()),"Source pixels")
        need((raw["original"]>=0).all() and (raw["original"]<=1).all(),"original RGB range")
        budget=projection_check(raw["original"],raw["adversarial"])
        need(md["budget"]==budget,"budget evidence")
        need(torch.equal(raw["adversarial"]-raw["original"],raw["effective_delta"]),"effective delta")
        del raw
        claim=read_json(store.root/("claim-"+key)/"claim.json")
        need(claim["gallery_ids"]==md["gallery_ids"]==[g.image_id for g in gs] and
             claim["query_ids"]==md["query_ids"]==[q.query_id for q in qs] and md["txt2img"]==mapping,"batch identity/order")
        need(md["clean_texts"]==[q.caption for q in qs] and len(md["adv_texts"])==len(qs),"Caption identity")
        f=md["flow"]
        need(f["events"]==["text1","image","text2"] and f["counts"]["backward"]==100 and
             f["counts"]["call:rand3Num"]==45 and len(f["selected_ratios"])==9 and
             f["all_image_gradients_finite"] and md["T1"]==f["T1"],"official complete stages")
        actions=[x for x in f["text_actions"] if x["stage"]==2]
        need([x["query_local_index"] for x in actions]==list(range(len(qs))) and len(md["edits"])==len(qs),"text actions/order")
        for q,old,new,e,a in zip(qs,md["clean_texts"],md["adv_texts"],md["edits"],actions):
            before=a["original_word_slots"];after=a["final_word_slots"]
            changed=[i for i,(x,y) in enumerate(zip(before,after)) if x!=y]
            need(" ".join(before)==old and " ".join(after)==new and len(before)==len(after)
                 and len(changed)==a["official_change_count"]<=1 and e["query_id"]==q.query_id
                 and e["before"]==old and e["after"]==new and e["changed_positions"]==changed
                 and col["adv_texts"][q.query_id]==new,"official word-slot actions")
        for stage in ("before","after"):
            checked(FileRef(str(store.root/("rng-"+key)/(stage+".pt")),md["rng_"+stage+"_sha256"]))
            state=torch.load(store.root/("rng-"+key)/(stage+".pt"),map_location="cpu",weights_only=False)
            need(rng_digest(state)==md["rng_"+stage+"_state"],"RNG serialized content")
            del state
        need(last_rng is None or last_rng==md["rng_before_state"],"global RNG continuity")
        last_rng=md["rng_after_state"];gids.extend(md["gallery_ids"]);qids.extend(md["query_ids"])
    need(gids==[g.image_id for g in m.gallery] and qids==[q.query_id for q in m.queries] and
         set(col["adv_texts"])==set(qids),"complete official set")

def source_inputs(c):
    root=Path(c["output_base"])/"full/source";store=ArtifactStore(root,new_base=c["output_base"])
    for key in ("protocol","execution","completed"):store.verify(key)
    report=read_json(root/"execution/report.json")
    need(report["status"]=="PASS" and report["official_attack_calls"]==385 and
         report["model_before"]==report["model_after"] and report["MLM_before"]==report["MLM_after"]
         and report["source_model_loads"]==report["MLM_loads"]==1 and report["optimizer"]==0,"Full Source/freeze")
    x=read_json(root/"completed/collection.json");m=manifest_from_dict(x["manifest"])
    protocol=read_json(root/"protocol/protocol.json")
    need(protocol["matched_manifest"]==x["manifest"] and protocol["raw_manifest"]==x["raw_manifest"]
         and protocol["parameters"]==PARAMETERS and protocol["target_feedback"] is False,"Source protocol/collection")
    need((len(m.gallery),len(m.queries),x["official_source_calls"])==(3074,6156,385),"Full members")
    validate_source(store,x,m)
    return store,x,m

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--action",choices=("preflight","resources","source","evaluate","audit"),default="preflight")
    p.add_argument("--model",choices=("irra","rasa","aptm"))
    p.add_argument("--allow-full",action="store_true")
    a=p.parse_args(argv)
    if a.action=="preflight":preflight(a.config);return 0
    c=configuration(a.config)
    if a.action=="resources":print("DRA_FULL_RESOURCES_PASS",json.dumps(resources(c)),flush=True);return 0
    if a.action=="audit":
        from tbps.baselines.dra_full_target import audit
        need(not torch.cuda.is_initialized(),"CPU audit");return audit(c)
    need(a.allow_full,"explicit Full authorization")
    if a.action=="source":return source_run(a.config)
    need(a.model is not None,"explicit Target")
    from tbps.baselines.dra_full_target import evaluate
    return evaluate(c,a.model)
if __name__=="__main__":raise SystemExit(main())
