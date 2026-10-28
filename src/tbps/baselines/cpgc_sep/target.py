"""独立原生Target评测：Raw Clean、Matched Clean及攻击；不把Smoke当作Full。"""
from pathlib import Path
from dataclasses import asdict,replace
import io,json,time,hashlib,resource
from tbps.artifacts import ArtifactStore,json_bytes,digest
from tbps.contracts import AttackBudget,ImageBudget,TextBudget,DatasetManifest
from tbps.engines.provenance import AttackOrigin,BoundRepresentation,payload_digest
from tbps.engines.native_cache import compact
from .dataset import read_test,scoped_manifest,require
from .runtime import attempt,unit,ref,resources,tensor_bytes
from .victims import load_victim,native_pixels,encode_images,encode_texts,snapshot,frozen
from .cache import FeatureBank,NativeScorer,TBPSEvaluationEngine
METRICS=("R@1","R@5","R@10","mAP")
def source_texts(c,method,dataset,scope,victim=None):
 root=Path(c["output_base"])/scope/(method+"_source")/dataset
 if method=="sep":root=root/victim
 store=ArtifactStore(root,new_base=c["output_base"]);store.verify("completed")
 report=json.loads((root/"completed/report.json").read_text())
 require(report["status"]=="PASS" and report["scope"]==scope and report["dataset"]==dataset,"source completion identity")
 rows=[]
 for p in sorted(root.glob("text-*/text.json")):
  store.verify(p.parent.name);rows.append(json.loads(p.read_text()))
 require(len(rows)==report["queries"] and len({r["query_id"] for r in rows})==len(rows),"all source Query identities")
 return store,{r["query_id"]:r for r in rows}
def cat_payload(parts):
 import torch
 return {k:torch.cat([p[k] for p in parts]) for k in parts[0]} if isinstance(parts[0],dict) else torch.cat(parts)
def rebind_gallery(bound,original,matched,proof):
 require(original.gallery==matched.gallery,"only Caption bridge may change; Gallery exact")
 binding=replace(bound.inputs,dataset_manifest_sha256=digest(asdict(matched)))
 return BoundRepresentation(bound.batch,binding,bound.payload_sha256,bound.cache_sources+(proof,),bound.origin,bound.original_sha256)
def evaluate(c,method,dataset,victim,scope,*,resume=False):
 import torch
 import torch.nn.functional as F
 require(not torch.cuda.is_initialized(),"CPU prerequisites before CUDA")
 source,rows=source_texts(c,method,dataset,scope,victim)
 manifest=scoped_manifest(read_test(c["datasets"][dataset]),scope)
 require(set(rows)=={q.query_id for q in manifest.queries},"source Query coverage")
 require(all(rows[q.query_id]["original_caption"]==q.caption for q in manifest.queries),"original Caption identity")
 matched=DatasetManifest(manifest.dataset_id,"test",manifest.annotation,manifest.gallery,
   tuple(replace(q,caption=rows[q.query_id]["matched_clean"]) for q in manifest.queries))
 clean_text=[q.caption for q in manifest.queries]
 matched_text=[rows[q.query_id]["matched_clean"] for q in manifest.queries]
 adv_text=[rows[q.query_id]["adversarial"] for q in manifest.queries]
 require(all(t.strip() for t in adv_text),"all Query failures/NOOP retained")
 snapshot_resources=resources(c)
 path=Path(c["output_base"])/scope/method/dataset/victim
 start=time.monotonic();torch.set_num_threads(2)
 with attempt(c,path,scope,resume=resume) as store:
  with load_victim(c,victim,dataset,"cuda:0",smoke=scope=="smoke") as (adapter,model,model_source):
   before=snapshot(model)
   protocol={"method":method,"dataset":dataset,"victim":victim,"scope":scope,"config_sha256":c["sha256"],
    "model_space":asdict(adapter.space),"manifest":asdict(manifest),"matched_manifest":asdict(matched),
    "source_completion":asdict(ref(source.root/"completed/manifest.json")),
    "native_input":"original PIL -> native Resize/ToTensor -> RGB UAP -> native Normalize; no extra JPEG/PIL quantization",
    "Raw_Clean":"original Caption","Matched_Clean":"official attack preprocessing only; no replacement",
    "text_attack_permissions":"Source only" if method=="cpgc" else "IRRA+RaSa ensemble includes evaluated Target",
    "final_metric_feedback_to_attack":False}
   if not (path/"protocol").exists():
    store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"config_sha":c["sha256"]},sources=(ref(c["path"]),ref(source.root/"completed/manifest.json")))
   else:store.verify("protocol",expected_provenance={"config_sha":c["sha256"]})
   store.publish("matched-projection",{"mapping.json":json_bytes({"raw_manifest":digest(asdict(manifest)),"matched_manifest":digest(asdict(matched)),"gallery_exact":True})},{"kind":"explicit_Caption_only_manifest_projection"},sources=(ref(path/"protocol/manifest.json"),)) if not (path/"matched-projection").exists() else store.verify("matched-projection")
   ids_g=tuple(g.image_id for g in manifest.gallery);ids_q=tuple(q.query_id for q in manifest.queries)
   orig_g=tuple(g.image_sha256 for g in manifest.gallery)
   orig_t=tuple(hashlib.sha256(t.encode()).hexdigest() for t in clean_text)
   banks={};counts={"image_samples":0,"text_samples":0,"backward":0,"optimizer":0}
   noise=None;noise_report=None;pixel_hashes=[]
   if method=="cpgc":
    raw_noise=torch.load(c["resources"]["cpgc_noise"]["path"],map_location="cpu",weights_only=True)
    require(raw_noise.dtype==torch.float32 and tuple(raw_noise.shape)==(3,384,384),"original CPGC UAP shape/dtype")
    target_size=(384,128) if victim=="irra" else (384,384)
    noise=F.interpolate(raw_noise.unsqueeze(0),size=target_size,mode="bilinear",align_corners=False) if target_size!=(384,384) else raw_noise.unsqueeze(0).clone()
    noise_report={"source_sha256":c["resources"]["cpgc_noise"]["sha256"],"source_tensor_sha256":payload_digest(raw_noise),
      "source_max_abs":float(raw_noise.abs().max()),"adaptation":"bilinear align_corners=False; no amplitude renormalization" if victim=="irra" else "identity",
      "adapted_shape":list(noise.shape),"adapted_max_abs":float(noise.abs().max()),"adapted_tensor_sha256":payload_digest(noise),"actual_max_RGB_linf":0.0,
      "FP32_budget_semantics":"original float32 12/255; actual maximum reported, not strict real-number upper-bound claim"}
   # 每图原/Adv/Delta的不可变FP32单元；图像侧无任何模型梯度。
   if method=="cpgc":
    for i,g in enumerate(manifest.gallery):
     provenance={"kind":"CPGC_RGB","image_id":g.image_id,"source_image_sha":g.image_sha256,"noise_sha":noise_report["adapted_tensor_sha256"],"config_sha":c["sha256"]}
     def produce_pixels():
      x=native_pixels(Path(c["datasets"][dataset]["root"])/"imgs"/g.image_rel_path,victim)
      y=(x+noise).clamp(0,1);delta=y-x
      return {"pixels.pt":tensor_bytes({"original":x.clone(),"adversarial":y.clone(),"delta":delta.clone()}),
        "details.json":json_bytes({"linf":float(delta.abs().max()),"adversarial_sha":payload_digest(y)})}
     unit(store,f"pixels-{i:06d}",produce_pixels,provenance,sources=(ref(path/"protocol/manifest.json"),),resume=resume)
     meta=json.loads((path/f"pixels-{i:06d}/details.json").read_text())
     pixel_hashes.append(meta["adversarial_sha"]);noise_report["actual_max_RGB_linf"]=max(noise_report["actual_max_RGB_linf"],meta["linf"])
   image_families=["clean_gallery"]+(["adv_gallery"] if method=="cpgc" else [])
   for family in image_families:
    bank=FeatureBank(c,store,family,adapter,manifest,ids_g,pixel_hashes if family=="adv_gallery" else orig_g,modality="image")
    for i in range(0,len(ids_g),8):
     end=min(i+8,len(ids_g))
     def produce_features():
      parts=[]
      for j in range(i,end):
       if method=="cpgc":
        pixelstore=store;pixelstore.verify(f"pixels-{j:06d}")
        data=torch.load(path/f"pixels-{j:06d}/pixels.pt",map_location="cpu",weights_only=True)
        x=data["adversarial" if family=="adv_gallery" else "original"]
       else:x=native_pixels(Path(c["datasets"][dataset]["root"])/"imgs"/manifest.gallery[j].image_rel_path,victim)
       parts.append(encode_images(adapter,x,victim));counts["image_samples"]+=len(x)
      return cat_payload(parts)
     bank.put(i,end,produce_features,resume=resume)
     if end%128==0 or end==len(ids_g):print("TARGET_IMAGE_PROGRESS",method,dataset,victim,scope,family,end,len(ids_g),flush=True)
    bank.read_globals();banks[family]=bank
   for family,texts,m in [("clean_text",clean_text,manifest),("adv_text",adv_text,manifest),("matched_text",matched_text,matched)]:
    hashes=tuple(hashlib.sha256(t.encode()).hexdigest() for t in texts)
    bank=FeatureBank(c,store,family,adapter,m,ids_q,hashes,modality="text")
    for i in range(0,len(ids_q),8):
     end=min(i+8,len(ids_q))
     def produce_text():
      x=encode_texts(adapter,texts[i:end],victim);counts["text_samples"]+=end-i
      return x
     bank.put(i,end,produce_text,resume=resume)
     if end%128==0 or end==len(ids_q):print("TARGET_TEXT_PROGRESS",method,dataset,victim,scope,family,end,len(ids_q),flush=True)
    bank.read_globals();banks[family]=bank
   source_proto=ref(source.root/"protocol/manifest.json")
   text_budget=TextBudget("word",1,None,"CPGC_official_pre_caption30_then_replace1") if method=="cpgc" else TextBudget("word",None,"floor15pct_min1_original_code","SEP_official_candidate_selection")
   text_origin=AttackOrigin(method+"_text",ref(c["resources"]["cpgc_embedding" if method=="cpgc" else "sep_neighbors"]["path"]),source_proto,AttackBudget(text=text_budget),c["sha256"])
   inputs={"clean_gallery":banks["clean_gallery"].bound(),"clean_text":banks["clean_text"].bound(),
     "adv_text":banks["adv_text"].bound(origin=text_origin,originals=orig_t)}
   modes=("Clean","Joint") if method=="cpgc" else ("Clean","Text-only")
   if method=="cpgc":
    origin=AttackOrigin("cpgc_image",ref(c["resources"]["cpgc_noise"]["path"]),source_proto,AttackBudget(image=ImageBudget("linf",12/255,"rgb_0_1")),c["sha256"])
    inputs["adv_gallery"]=banks["adv_gallery"].bound(origin=origin,originals=orig_g)
   scorer=NativeScorer(adapter,victim,banks,store=store,config=c,resume=resume)
   engine=TBPSEvaluationEngine(manifest,scorer,model_source.checkpoint)
   ep,result,scores=engine.run(inputs,modes=modes,query_chunk=128)
   matched_inputs={"clean_text":banks["matched_text"].bound(),
    "clean_gallery":rebind_gallery(inputs["clean_gallery"],manifest,matched,ref(path/"matched-projection/manifest.json"))}
   mp,mr,ms=TBPSEvaluationEngine(matched,scorer,model_source.checkpoint).run(matched_inputs,modes=("Clean",),query_chunk=128)
   groups={mode:{k:result["groups"][mode][k] for k in METRICS} for mode in modes}
   groups["Matched Clean"]={k:mr["groups"]["Clean"][k] for k in METRICS}
   for mode,matrix in scores.items():
    unit(store,"scores-"+mode.lower().replace("-",""),lambda:{"scores.pt":tensor_bytes(matrix)},{"kind":"exact_native_full_scores","mode":mode},sources=(ref(path/"protocol/manifest.json"),),resume=resume)
   unit(store,"scores-matched",lambda:{"scores.pt":tensor_bytes(ms["Clean"])},{"kind":"matched_clean_native_scores"},sources=(ref(path/"protocol/manifest.json"),),resume=resume)
   frozen(model);after=snapshot(model);require(before==after,"Victim parameters/buffers changed")
   def csv_data(rounded):
    import csv
    b=io.StringIO();w=csv.writer(b,lineterminator="\n");w.writerow(("Mode",*METRICS))
    for mode,v in groups.items():w.writerow((mode,*(format(v[k],".2f") if rounded else v[k] for k in METRICS)))
    return b.getvalue().encode()
   report={"status":"PASS","scope":scope,"method":method,"dataset":dataset,"victim":victim,"Gallery":len(manifest.gallery),"Query":len(manifest.queries),
     "groups":groups,"unit":"percent","counts":counts,"native_counts":dict(getattr(adapter,"neural_counts",{})),
     "native_TopK":128 if scope=="full" and victim=="rasa" else min(128,len(manifest.gallery)) if victim=="rasa" else None,
     "native_scored_ITM_pairs":scorer.native_pairs,"model_before":before,"model_after":after,"noise":noise_report,
     "resources_before":snapshot_resources,"peak_GPU_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_GPU_reserved_bytes":torch.cuda.max_memory_reserved(),
     "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"elapsed_seconds":time.monotonic()-start,
     "native_complete_ranking":True,"attack_mode":"Joint" if method=="cpgc" else "Text-only"}
   refs=tuple(ref(p) for p in sorted(path.glob("*/manifest.json")) if p.parent.name!="completed")
   store.publish("completed",{"summary.json":json_bytes(report),"metrics.csv":csv_data(False),"metrics_2dp.csv":csv_data(True)},{"kind":"official_TBPS_completed"},sources=refs)
 print("CPGC_SEP_TARGET_PASS",method,dataset,victim,scope,flush=True)
 return 0
