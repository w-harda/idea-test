"""C-PGC多数据集两项单模态补充；只读复用冻结Full特征，显式Matched输入桥。"""
from pathlib import Path
from dataclasses import asdict,replace
from types import SimpleNamespace
import json,hashlib,io,csv,os,time
from tbps.artifacts import FileRef,checked,sha_file,digest,json_bytes,ArtifactError
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from tbps.baselines.cpgc_sep.runtime import config as parent_config,ref,unit,load_tensor,tensor_bytes,attempt
from tbps.baselines.cpgc_sep.dataset import require,read_test
from tbps.baselines.cpgc_sep.target import METRICS,source_texts,rebind_gallery
from tbps.baselines.cpgc_sep.cache import FeatureBank,BlockCache,NativeScorer,TBPSEvaluationEngine
from tbps.baselines.cpgc_sep.victims import VictimSource,load_victim,snapshot,frozen
from tbps.engines.provenance import AttackOrigin,BoundRepresentation,payload_digest
from tbps.engines.metrics import ranking
from tbps.contracts import AttackBudget,ImageBudget,TextBudget,DatasetManifest
MODES=("Matched Clean","Image-only","Text-only","Joint")
NEW_MODES=("Image-only","Text-only")
FAMILIES={"Matched Clean":("matched_text","clean_gallery"),"Image-only":("matched_text","adv_gallery"),
          "Text-only":("adv_text","clean_gallery"),"Joint":("adv_text","adv_gallery")}

SIZES={"rstp":(1000,2000),"icfg":(19848,19848)}
def dimensions(c):
 require(c.get("dataset") in SIZES,"only authorized RSTP/ICFG extension")
 require(tuple(c.get("expected_size",()))==SIZES[c["dataset"]],"fixed official Full dimensions")
 return SIZES[c["dataset"]]
def result_root(c,victim,*,audit=False):
 require(victim in c["targets"],"configured Victim only")
 root=Path(c["output_base"])
 return root/"audit"/c["dataset"]/victim if audit else root/c["dataset"]/victim

def settings(path):
 p=Path(path).resolve();c=json.loads(p.read_text())
 require(c["schema"]=="tbps-cpgc-modal-ablation-2","ablation config schema");dimensions(c)
 require(c["modes"]=={k:list(v) for k,v in FAMILIES.items()},"explicit matched family map")
 checked(FileRef(**c["parent_config"]))
 for name,h in c["producers"].items():checked(FileRef(name,h))
 c.update(path=str(p),sha256=sha_file(p))
 old=parent_config(c["parent_config"]["path"])
 require(old["sha256"]==c["parent_config"]["sha256"],"parent config SHA")
 require(old["evaluation"]["ITM_TopK"]==128 and old["evaluation"]["query_chunk"]==32
   and old["evaluation"]["fusion_batch"]==16,"unchanged native numerical batches")
 return c,old

def unchanged(snapshot_):
 for p,v in snapshot_.items():
  s=Path(p).stat()
  require([s.st_size,s.st_mtime_ns,s.st_ino,s.st_ctime_ns]==v,"historical file metadata changed: "+p)

def snapshot_files(roots):
 out={}
 for root in roots:
  for p in sorted(Path(root).rglob("*")):
   if p.is_file():
    s=p.stat();out[str(p)]=[s.st_size,s.st_mtime_ns,s.st_ino,s.st_ctime_ns]
 return out

def check_store(store):
 require(not store.pending() and not (store.root/"FAIL.json").exists(),"historical pending/FAIL")
 manifests=sorted(store.root.glob("*/manifest.json"))
 require(manifests,"missing committed units")
 for p in manifests:store.verify(p.parent.name)
 for p in store.root.glob("*.Claim.json"):
  require((store.root/p.name.removesuffix(".Claim.json")/"completed.json").is_file(),"uncommitted historical Claim")
 return len(manifests)

def history(c,old,victim):
 import torch
 require(not torch.cuda.is_initialized(),"CPU history check before CUDA")
 root=Path(c["targets"][victim]["path"])
 require(root==Path(old["output_base"])/"full/cpgc"/c["dataset"]/victim,"exact historical Target path")
 store=VerifiedArtifactStore(root,new_base=old["output_base"])
 checked(FileRef(**c["targets"][victim]["manifest"]))
 n=check_store(store)
 proto=json.loads((root/"protocol/protocol.json").read_text())
 report=json.loads((root/"completed/summary.json").read_text())
 require(proto["config_sha256"]==old["sha256"],"original config identity")
 require(report["status"]=="PASS" and report["victim"]==victim and report["method"]=="cpgc"
  and report["scope"]=="full" and (report["Gallery"],report["Query"])==dimensions(c) and report["dataset"]==c["dataset"],"real completed Full only")
 require(report["model_before"]==report["model_after"] and report["counts"]["backward"]==report["counts"]["optimizer"]==0,"frozen historical model")
 require(proto["engineering_execution"]==old["evaluation"],"original evaluation configuration")
 manifest=read_test(old["datasets"][c["dataset"]])
 require(digest(asdict(manifest))==digest(proto["manifest"]),"full ordered Dataset Manifest")
 source,rows=source_texts(old,"cpgc",c["dataset"],"full",victim)
 check_store(source)
 matched=DatasetManifest(manifest.dataset_id,"test",manifest.annotation,manifest.gallery,
  tuple(replace(q,caption=rows[q.query_id]["matched_clean"]) for q in manifest.queries))
 require(digest(asdict(matched))==digest(proto["matched_manifest"]),"exact matched Caption identity")
 audit_root=Path(c["audits"][victim]["path"])
 checked(FileRef(**c["audits"][victim]["manifest"]))
 aud=VerifiedArtifactStore(audit_root,new_base=old["output_base"]);check_store(aud)
 audit=json.loads((audit_root/"completed/audit.json").read_text())
 require(audit["status"]=="PASS" and audit["CUDA"] is False and audit["neural_forward"]==0
   and audit["result"]==asdict(ref(root/"completed/manifest.json")) and audit["groups"]==report["groups"],"original independent CPU audit binding")
 controls={}
 for mode,key in (("Matched Clean","scores-matched"),("Joint","scores-joint")):
  score=load_tensor(store,key,"scores.pt")
  vals,_=ranking(score,manifest.evaluation_identity())
  controls[mode]={k:vals[k] for k in METRICS}
  require(controls[mode]==report["groups"][mode],"historical score vs summary mismatch")
 for name,rounded in (("metrics.csv",False),("metrics_2dp.csv",True)):
  for row in csv.DictReader((root/"completed"/name).open()):
   for k in METRICS:require(row[k]==format(report["groups"][row["Mode"]][k],".2f") if rounded else float(row[k])==report["groups"][row["Mode"]][k],"historical CSV mismatch")
 require(not torch.cuda.is_initialized(),"history verification initialized CUDA")
 return store,proto,manifest,matched,source,rows,controls,n

def banks_for(old,store,adapter,manifest,matched,rows):
 root=store.root;ids_g=tuple(g.image_id for g in manifest.gallery);ids_q=tuple(q.query_id for q in manifest.queries)
 pixel_sha=tuple(json.loads((root/f"pixels-{i:06d}/details.json").read_text())["adversarial_sha"] for i in range(len(ids_g)))
 hashes={"clean_gallery":tuple(g.image_sha256 for g in manifest.gallery),"adv_gallery":pixel_sha,
  "matched_text":tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in matched.queries),
  "adv_text":tuple(hashlib.sha256(rows[q.query_id]["adversarial"].encode()).hexdigest() for q in manifest.queries)}
 out={}
 for family in hashes:
  text=family.endswith("text")
  bank=FeatureBank(old,store,family,adapter,matched if family=="matched_text" else manifest,
     ids_q if text else ids_g,hashes[family],modality="text" if text else "image",
     chunk=old["evaluation"]["text_batch"] if text else old["evaluation"]["cache_chunk"])
  for start in range(0,len(bank.ids),bank.chunk):
   x=bank.block(start)
   detail=json.loads((root/bank.key(start)/"details.json").read_text())
   require(payload_digest(x)==detail["payload_sha256"],"native feature payload digest")
  bank.read_globals();out[family]=bank
 return out

def matched_inputs(old,banks,manifest,matched,source,proof):
 source_proto=ref(source.root/"protocol/manifest.json")
 gi=tuple(g.image_sha256 for g in manifest.gallery)
 tx=tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in matched.queries)
 io_=AttackOrigin("cpgc_image",FileRef(**{k:old["resources"]["cpgc_noise"][k] for k in ("path","sha256")}),source_proto,AttackBudget(image=ImageBudget("linf",12/255,"rgb_0_1")),old["sha256"])
 to=AttackOrigin("cpgc_text",FileRef(**{k:old["resources"]["cpgc_embedding"][k] for k in ("path","sha256")}),source_proto,AttackBudget(text=TextBudget("word",1,None,"CPGC_official_pre_caption30_then_replace1")),old["sha256"])
 def rebound(bound):
  return replace(bound,inputs=replace(bound.inputs,dataset_manifest_sha256=digest(asdict(matched))),cache_sources=bound.cache_sources+(proof,))
 return {"clean_text":banks["matched_text"].bound(),
  "clean_gallery":rebound(banks["clean_gallery"].bound()),
  "adv_gallery":rebound(banks["adv_gallery"].bound(origin=io_,originals=gi)),
  "adv_text":rebound(banks["adv_text"].bound(origin=to,originals=tx))}


class CandidateRecordingScorer(NativeScorer):
 """仅记录原生评分选中的完整Top128；不修改NativeScorer或NativeRasa的计算。"""
 def score(self,query,gallery):
  value=super().score(query,gallery)
  if self.model=="rasa":
   import torch
   qb,start,end=self.cursor;gb=self.refs[id(gallery.payload)]
   with torch.inference_mode():
    selected=(query.payload.to(self.adapter.device)@gallery.payload.to(self.adapter.device).T).topk(k=128,dim=1).indices.cpu()
   mask=torch.zeros_like(value.payload,dtype=torch.bool);mask.scatter_(1,selected,True)
   require(torch.equal(mask,value.payload!=-100),"recorded global Top128 vs actual fused members")
   unit(self.score_store,f"candidates-{qb.family}--{gb.family}-{start:06d}",
    lambda:{"indices.pt":tensor_bytes(selected)},{"kind":"post_score_global_top128_observation","start":start,"end":end},
    sources=(ref(self.score_store.root/f"scorechunk-{qb.family}--{gb.family}-{start:06d}"/"manifest.json"),))
  return value

def csv_bytes(victim,groups,rounded):
 f=io.StringIO();w=csv.writer(f,lineterminator="\n");w.writerow(("Victim","Mode",*METRICS))
 for mode in MODES:w.writerow((victim,mode,*(format(groups[mode][k],".2f") if rounded else groups[mode][k] for k in METRICS)))
 return f.getvalue().encode()

def preflight(path,output):
 import torch,importlib.metadata
 c,old=settings(path);torch.set_num_threads(2)
 require(all(importlib.metadata.version(k)==v for k,v in old["runtime_versions"]["victim"].items()),"native victim runtime")
 roots=[v["path"] for key in ("targets","audits") for v in c[key].values()]
 protection=snapshot_files(roots);evidence={}
 for victim in c["targets"]:
  VictimSource(old,victim,c["dataset"]).verify()
  s,p,m,mm,source,rows,controls,n=history(c,old,victim)
  protection.update(snapshot_files([source.root]))
  evidence[victim]={"units":n,"groups":controls,"Target":asdict(ref(s.root/"completed/manifest.json")),
   "Audit":c["audits"][victim]["manifest"],"model_space":p["model_space"],"matched_manifest":digest(asdict(mm))}
 unchanged(protection)
 require(not torch.cuda.is_initialized(),"CPU preflight CUDA forbidden")
 out=Path(output);require(not out.exists(),"preflight receipt exists")
 out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(json_bytes({"status":"PASS","CUDA":False,
  "config":asdict(ref(path)),"evidence":evidence,"protection":protection}))
 print("CPGC_MODAL_CPU_PREFLIGHT_PASS",flush=True)

def run(path,victim,preflight_path):
 import torch,resource
 c,old=settings(path);torch.set_num_threads(2);start=time.monotonic()
 pre=json.loads(Path(preflight_path).read_text());require(pre["status"]=="PASS" and pre["config"]==asdict(ref(path)),"preflight binding")
 unchanged(pre["protection"])
 store,proto,manifest,matched,source,rows,controls,n=history(c,old,victim)
 from tbps.baselines.cpgc_sep.runtime import resources
 resource_before=resources(old)
 outpath=result_root(c,victim)
 with attempt(c,outpath,"full") as out:
  with load_victim(old,victim,c["dataset"],"cuda:0",smoke=False) as (adapter,model,model_source):
   require(asdict(adapter.space)==proto["model_space"],"same native model space")
   before=snapshot(model)
   adapter.token_block_cache=BlockCache(c["token_cache_bytes"])
   banks=banks_for(old,store,adapter,manifest,matched,rows)
   out.publish("protocol",{"protocol.json":json_bytes({"parent":asdict(ref(store.root/"protocol/manifest.json")),
     "MatchedManifest":asdict(matched),"modes":FAMILIES,"Gallery":dimensions(c)[0],"Query":dimensions(c)[1],"dataset":c["dataset"],
     "config_sha256":c["sha256"],"victim":victim,"evaluation":old["evaluation"],
     "no_generation":True,"no_encoding":True})},{"kind":"explicit_matched_caption_bridge"},sources=(ref(c["path"]),ref(store.root/"completed/manifest.json")))
   inputs=matched_inputs(old,banks,manifest,matched,source,ref(outpath/"protocol/manifest.json"))
   scorer=CandidateRecordingScorer(adapter,victim,banks,store=out,config=old)
   # 小范围重放历史两组评分，只用于验证复用接线；不重跑历史Full。
   parity={}
   for mode,key in (("Matched Clean","scores-matched"),("Joint","scores-joint")):
    qf,gf=FAMILIES[mode];qb,gb=banks[qf],banks[gf]
    probe=NativeScorer(adapter,victim,banks)
    q=probe.slice_query(qb.bound().batch,0,old["evaluation"]["query_chunk"])
    actual=probe.score(q,gb.bound().batch).payload
    expected=load_tensor(store,key,"scores.pt")[:len(actual)]
    require(torch.equal(actual,expected),"native score replay not exact")
    parity[mode]={"queries":len(actual),"exact":True,"max_abs":float((actual-expected).abs().max())}
   engine=TBPSEvaluationEngine(matched,scorer,model_source.checkpoint)
   ep,result,scores=engine.run(inputs,modes=NEW_MODES,query_chunk=old["evaluation"]["query_chunk"])
   groups={**controls,**{mode:{k:result["groups"][mode][k] for k in METRICS} for mode in NEW_MODES}}
   for mode,matrix in scores.items():
    unit(out,"scores-"+mode.lower().replace("-",""),lambda matrix=matrix:{"scores.pt":tensor_bytes(matrix)},
      {"mode":mode,"families":FAMILIES[mode]},sources=(ref(outpath/"protocol/manifest.json"),))
   frozen(model);after=snapshot(model);require(before==after,"model changed")
   counts=dict(getattr(adapter,"neural_counts",{"image_samples":0,"text_samples":0}))
   require(counts.get("image_samples",0)==counts.get("text_samples",0)==0,"encoding forbidden")
   unchanged(pre["protection"])
   summary={"status":"PASS","scope":"full","victim":victim,"Gallery":dimensions(c)[0],"Query":dimensions(c)[1],"dataset":c["dataset"],"groups":groups,
    "unit":"percent","modes":FAMILIES,"parent_target":c["targets"][victim]["manifest"],"config_sha256":c["sha256"],
    "model_before":before,"model_after":after,"neural_counts":counts,"generation":0,"backward":0,"optimizer":0,
    "native_TopK":128 if victim=="rasa" else None,"native_pairs":scorer.native_pairs,"historical_score_parity":parity,
    "resources_before":resource_before,"peak_GPU_allocated_bytes":torch.cuda.max_memory_allocated(),
    "peak_GPU_reserved_bytes":torch.cuda.max_memory_reserved(),"peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    "elapsed_seconds":time.monotonic()-start,"history_unchanged":True}
   refs=tuple(ref(p) for p in sorted(outpath.glob("*/manifest.json")))
   out.publish("completed",{"summary.json":json_bytes(summary),"engine_protocol.json":json_bytes(ep),"engine_result.json":json_bytes(result),
      "metrics.csv":csv_bytes(victim,groups,False),"metrics_2dp.csv":csv_bytes(victim,groups,True)},{"kind":"CPGC_matched_single_modality_full"},sources=refs)
 print("CPGC_MODAL_FULL_PASS",c["dataset"],victim,flush=True)

def audit(path,victim,preflight_path):
 import torch
 c,old=settings(path);torch.set_num_threads(2)
 require(not torch.cuda.is_initialized(),"independent CPU audit")
 pre=json.loads(Path(preflight_path).read_text());require(pre["config"]==asdict(ref(path)),"audit preflight config")
 unchanged(pre["protection"])
 original,proto,m,mm,source,rows,controls,n=history(c,old,victim)
 store=VerifiedArtifactStore(result_root(c,victim),new_base=c["output_base"])
 count=check_store(store);r=json.loads((store.root/"completed/summary.json").read_text())
 require(r["status"]=="PASS" and r["config_sha256"]==c["sha256"] and r["modes"]=={k:list(v) for k,v in FAMILIES.items()},"matched mode/source identity")
 require((r["Gallery"],r["Query"])==dimensions(c) and r["dataset"]==c["dataset"] and r["model_before"]==r["model_after"],"full frozen model")
 require(r["generation"]==r["backward"]==r["optimizer"]==0 and r["neural_counts"]["image_samples"]==r["neural_counts"]["text_samples"]==0,"no generation/encoding/training")
 ep=json.loads((store.root/"completed/engine_protocol.json").read_text());er=json.loads((store.root/"completed/engine_result.json").read_text())
 require(digest(ep["dataset_manifest"])==digest(asdict(mm)) and ep["modes"]==list(NEW_MODES) and er["protocol_sha256"]==digest(ep),"engine matched protocol")
 matched_hashes=tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in mm.queries)
 require(tuple(ep["inputs"]["clean_text"]["inputs"]["ordered_sha256"])==matched_hashes,"matched text feature content binding")
 require(tuple(ep["inputs"]["adv_text"]["original_sha256"])==matched_hashes,"adv text anchored to matched clean")
 require(all("/matched_text-" in s["path"] for s in ep["inputs"]["clean_text"]["cache_sources"]),"no raw text cache")
 groups=dict(controls)
 for mode in NEW_MODES:
  qf,gf=FAMILIES[mode];score=load_tensor(store,"scores-"+mode.lower().replace("-",""),"scores.pt")
  require(tuple(score.shape)==(dimensions(c)[1],dimensions(c)[0]) and score.dtype==torch.float32,"full FP32 matrix")
  chunks=[]
  for start in range(0,dimensions(c)[1],old["evaluation"]["query_chunk"]):
   part=load_tensor(store,f"scorechunk-{qf}--{gf}-{start:06d}","scores.pt");chunks.append(part)
   if victim=="rasa":
    selected=load_tensor(store,f"candidates-{qf}--{gf}-{start:06d}","indices.pt")
    require(selected.shape==(len(part),128) and selected.dtype==torch.int64,"Top128 index shape")
    mask=torch.zeros_like(part,dtype=torch.bool);mask.scatter_(1,selected,True)
    require(torch.equal(mask,part!=-100),"all original global Top128 members fused")
  require(torch.equal(torch.cat(chunks),score),"score chunks exact")
  vals,order=ranking(score,m.evaluation_identity());groups[mode]={k:vals[k] for k in METRICS}
  require(groups[mode]==r["groups"][mode] and payload_digest(score)==er["scores_sha256"][mode],"independent ranking and score SHA")
  require({k:er["groups"][mode][k] for k in METRICS}==groups[mode],"engine summary metrics")
 for mode in controls:require(groups[mode]==r["groups"][mode],"read-only controls")
 for rounded,name in ((False,"metrics.csv"),(True,"metrics_2dp.csv")):
  require((store.root/"completed"/name).read_bytes()==csv_bytes(victim,groups,rounded),"CSV exact")
 require(not torch.cuda.is_initialized(),"audit CUDA")
 unchanged(pre["protection"])
 outpath=result_root(c,victim,audit=True)
 with attempt(c,outpath,"full") as out:
  out.publish("completed",{"audit.json":json_bytes({"status":"PASS","CUDA":False,"neural_forward":0,
    "victim":victim,"Gallery":dimensions(c)[0],"Query":dimensions(c)[1],"dataset":c["dataset"],"groups":groups,"units_verified":count,"history_unchanged":True,
    "result":asdict(ref(store.root/"completed/manifest.json")),"config":asdict(ref(path))})},
    {"kind":"independent_CPGC_matched_modality_CPU_audit"},sources=(ref(store.root/"completed/manifest.json"),ref(path)))
 print("CPGC_MODAL_INDEPENDENT_CPU_AUDIT_PASS",c["dataset"],victim,flush=True)

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--action",choices=("preflight","evaluate","audit"),default="preflight")
 p.add_argument("--victim",choices=("irra","rasa"));p.add_argument("--preflight",required=True);p.add_argument("--allow-gpu-evaluation",action="store_true")
 a=p.parse_args()
 if a.action=="preflight":preflight(a.config,a.preflight)
 elif a.action=="evaluate":
  if not a.allow_gpu_evaluation:p.error("explicit GPU evaluation required")
  run(a.config,a.victim,a.preflight)
 else:audit(a.config,a.victim,a.preflight)
if __name__=="__main__":main()
