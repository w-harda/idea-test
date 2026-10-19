"""DRA官方IRRA Source薄入口；本阶段只开放固定8图Smoke，不提供Full执行。"""
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

def save_pixels(store,key,tensors,metadata,sources):
    buffer=io.BytesIO();torch.save(compact(tensors),buffer)
    store.publish(key,{"pixels.pt":buffer.getvalue(),"metadata.json":json_bytes(metadata)},
                  {"kind":"DRA_source_pixels"},sources=sources)
    store.verify(key)
    got=torch.load(store.root/key/"pixels.pt",map_location="cpu",weights_only=True)
    need(set(got)==set(tensors) and all(torch.equal(got[k],v) for k,v in tensors.items()),"compact pixel bitwise readback")

def configuration(path):
    c=read_json(path,"tbps-dra-irra-source-1")
    need(c["parameters"]==PARAMETERS and c["source_model"]=="irra","fixed official DRA actual settings")
    verify_official(c)
    if "source_config_ref" in c:checked(FileRef(**c["source_config_ref"]))
    for file,h in c["producers"].items():checked(FileRef(file,h))
    for file,h in c["bert_files"].items():checked(FileRef(str(Path(c["bert"])/file),h))
    c["path"]=str(Path(path).resolve())
    return c

def preflight(path):
    c=configuration(path)
    IRRASource.from_dict(c["irra"]).verify()
    ds,raw,matched=ordered_dataset(c,"smoke")
    _,full,_=ordered_dataset(c,"full")
    sizes=[len(batch_members(full,i,1)[1]) for i in range(len(full.gallery))]
    need(len(full.gallery)==3074 and len(full.queries)==6156 and sizes.count(10)==1 and sizes.count(2)==3073,
         "official complete Caption mapping including ten-Caption member")
    need(not torch.cuda.is_initialized(),"preflight must be CPU-only")
    report={"Gallery":8,"Query":len(raw.queries),"full_Gallery":3074,"full_Query":6156,
            "caption_count_distribution":{"2":sizes.count(2),"10":sizes.count(10)},
            "ten_caption_gallery":full.gallery[sizes.index(10)].image_id,"source_models_loaded":0,"GPU":False}
    print("DRA_CPU_PREFLIGHT_PASS",json.dumps(report),flush=True)
    return c,ds,raw,matched,report

def resources(c):
    from tbps.engines.materialized_irra import gpu_available
    gpu=gpu_available({"physical_gpu":0})
    ram=int(next(l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")))*1024
    import shutil
    free=shutil.disk_usage(Path(c["output_base"]).parent).free
    need(gpu["free_mib"]>=c["minimum_free_mib"],"DRA insufficient free GPU memory; no waiting")
    need(ram>=c["minimum_RAM_bytes"] and free>=c["minimum_disk_bytes"],"DRA insufficient RAM/disk")
    return dict(GPU=gpu,available_RAM_bytes=ram,free_disk_bytes=free)

def source_run(path):
    c,ds,raw,matched,pf=preflight(path)
    need({k:version(k) for k in c["source_versions"]}==c["source_versions"],"pinned dedicated DRA runtime")
    root=Path(c["output_base"])/"smoke/source"
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
    report={"status":"RUNNING","scope":"smoke","source_model_loads":0,"MLM_loads":0,"target_models_in_generation":[],
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
            gallery,queries,mapping=batch_members(matched,0)
            store.publish("claim-batch-000000",{"claim.json":json_bytes({"gallery_ids":[g.image_id for g in gallery],
                "query_ids":[q.query_id for q in queries]})},{"kind":"DRA_claim"},sources=(ref(root/"protocol/manifest.json"),))
            pixels=torch.stack([source_pixels(ds.image_path(g.image_rel_path)) for g in gallery]).to("cuda:0")
            texts=[q.caption for q in queries]
            trace=PassiveTrace(c["official"]["attacker"]["path"],source,write_flow)
            timer=time.monotonic()
            with trace.active():
                adv,adv_texts,official_image_seconds=attacker.attack(pixels,texts,mapping,device="cuda:0",max_length=77,scales=PARAMETERS["scales"])
            attack_seconds=time.monotonic()-timer
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
            save_pixels(store,"batch-000000",cpu,md,(ref(root/"protocol/manifest.json"),ref(root/"claim-batch-000000/manifest.json")))
            source.zero_grad(set_to_none=True)
            source.backend.requires_grad_(False)
            need(model_snapshot(source.backend)==before and model_snapshot(mlm)==mlm_before,"source weights/buffers changed")
            need(not source.training and not source.backend.training and not mlm.training,"source eval state")
            source.zero_grad(set_to_none=True)
            report.update(status="PASS",model_after=model_snapshot(source.backend),MLM_after=model_snapshot(mlm),
                source_counts=source.counts,MLM_counts=counts,source_weights_fixed=True,parameter_updates=0,parameter_gradients_during_official_attack=True,
                official_flow=actual,budget=check,attack_seconds=attack_seconds,official_image_seconds=official_image_seconds,
                edited_queries=sum(not e["NOOP"] for e in md["edits"]),NOOP=sum(e["NOOP"] for e in md["edits"]),
                parameter_grads_cleared_after_attack=all(p.grad is None for p in source.parameters()))
            hook.remove()
            bridge.flush();flow.flush()
            collection={"manifest":asdict(matched),"raw_manifest":asdict(raw),"units":["batch-000000"],
                "adv_texts":dict(zip([q.query_id for q in queries],adv_texts)),"official_source_calls":1,"Gallery":8,"Query":len(queries)}
        verify_official(c)
        store.publish("completed",{"collection.json":json_bytes(collection)},{"kind":"DRA_IRRA_source_complete"},
            sources=(ref(root/"protocol/manifest.json"),ref(root/"batch-000000/manifest.json"),ref(tracefile),ref(flowfile)))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss_bytes())
        if torch.cuda.is_initialized():report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"DRA_source_execution"},sources=(ref(root/"protocol/manifest.json"),))
    print("DRA_OFFICIAL_IRRA_SOURCE_"+report["status"],"smoke",flush=True)
    return int(report["status"]!="PASS")

def source_inputs(c):
    root=Path(c["output_base"])/"smoke/source";store=ArtifactStore(root,new_base=c["output_base"])
    for key in ("protocol","execution","completed"):store.verify(key)
    need(read_json(root/"execution/report.json")["status"]=="PASS","incomplete DRA source")
    x=read_json(root/"completed/collection.json")
    return store,x,manifest_from_dict(x["manifest"])

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--action",choices=("preflight","source","evaluate","audit"),default="preflight")
    p.add_argument("--model",choices=("irra","rasa","aptm"))
    p.add_argument("--allow-gpu-smoke",action="store_true")
    a=p.parse_args(argv)
    if a.action=="preflight":preflight(a.config);return 0
    c=configuration(a.config)
    if a.action=="audit":
        from tbps.baselines.dra_target import audit
        need(not torch.cuda.is_initialized(),"CPU audit")
        return audit(c)
    need(a.allow_gpu_smoke,"explicit fixed eight-gallery GPU authorization; Full not implemented")
    if a.action=="source":return source_run(a.config)
    need(a.model is not None,"explicit native Target")
    from tbps.baselines.dra_target import evaluate
    return evaluate(c,a.model)
if __name__=="__main__":raise SystemExit(main())
