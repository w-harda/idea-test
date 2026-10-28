"""独立CPU审计既有像素、特征Manifest、完整分数和来源；不加载模型或CUDA。"""
from pathlib import Path
import json,csv,math
from dataclasses import asdict
from tbps.artifacts import ArtifactStore,json_bytes,digest
from tbps.engines.metrics import ranking
from .dataset import read_test,scoped_manifest,require
from .runtime import load_tensor,ref,attempt
from .target import METRICS
def audit(c,method,dataset,victim,scope):
 import torch
 require(not torch.cuda.is_initialized(),"independent CPU audit only")
 torch.set_num_threads(2)
 root=Path(c["output_base"])/scope/method/dataset/victim
 store=ArtifactStore(root,new_base=c["output_base"]);store.verify("completed")
 summary=json.loads((root/"completed/summary.json").read_text())
 manifest=scoped_manifest(read_test(c["datasets"][dataset]),scope)
 require(summary["status"]=="PASS" and summary["scope"]==scope and summary["dataset"]==dataset and summary["victim"]==victim,"scope/model/source")
 require((summary["Gallery"],summary["Query"])==(len(manifest.gallery),len(manifest.queries)),"complete official members")
 require(summary["model_before"]==summary["model_after"] and summary["counts"]["backward"]==summary["counts"]["optimizer"]==0,"frozen inference")
 require(not store.pending(),"pending transaction")
 for claim in root.glob("*.Claim.json"):
  require((root/claim.name.removesuffix(".Claim.json")/"manifest.json").exists(),"orphan unfinished Claim")
 source,source_rows=__import__("tbps.baselines.cpgc_sep.target",fromlist=["source_texts"]).source_texts(c,method,dataset,scope,victim)
 require(set(source_rows)=={q.query_id for q in manifest.queries},"all source captions")
 source_report=json.loads((source.root/"completed/report.json").read_text())
 require(source_report["status"]=="PASS","source completion")
 if method=="sep":require(source_report["source_frozen"] and source_report["backward"]==source_report["optimizer"]==0,"SEP immutable surrogate inference")
 for p in source.root.glob("text-*/manifest.json"):source.verify(p.parent.name)
 for claim in source.root.glob("*.Claim.json"):
  require((source.root/claim.name.removesuffix(".Claim.json")/"manifest.json").exists(),"source unfinished Claim")
 units=0
 for p in root.glob("*/manifest.json"):
  store.verify(p.parent.name);units+=1
 if method=="cpgc":
  import torch.nn.functional as F
  noise=torch.load(c["resources"]["cpgc_noise"]["path"],map_location="cpu",weights_only=True).unsqueeze(0)
  if victim=="irra":noise=F.interpolate(noise,size=(384,128),mode="bilinear",align_corners=False)
  maximum=0.
  for i,g in enumerate(manifest.gallery):
   x=load_tensor(store,f"pixels-{i:06d}","pixels.pt")
   require(x["original"].dtype==torch.float32 and x["adversarial"].dtype==torch.float32,"FP32 pixels")
   require(torch.equal(x["adversarial"],(x["original"]+noise).clamp(0,1)),"same fixed official UAP application")
   require(torch.equal(x["delta"],x["adversarial"]-x["original"]),"exact delta record")
   maximum=max(maximum,float(x["delta"].abs().max()))
  require(maximum==summary["noise"]["actual_max_RGB_linf"],"actual RGB budget report")
  require(summary["noise"]["source_sha256"]==c["resources"]["cpgc_noise"]["sha256"],"same source UAP")
 groups={}
 for mode,key in [("Clean","scores-clean"),("Joint" if method=="cpgc" else "Text-only","scores-joint" if method=="cpgc" else "scores-textonly"),("Matched Clean","scores-matched")]:
  scores=load_tensor(store,key,"scores.pt")
  require(tuple(scores.shape)==(len(manifest.queries),len(manifest.gallery)) and scores.dtype==torch.float32,"full exact score matrix")
  values,order=ranking(scores,manifest.evaluation_identity(),direction="higher_better")
  groups[mode]={k:values[k] for k in METRICS}
  require(groups[mode]==summary["groups"][mode],"independent complete-ranking metrics")
  del scores,order
 for name,rounded in [("metrics.csv",False),("metrics_2dp.csv",True)]:
  rows=list(csv.DictReader((root/"completed"/name).open()))
  require([r["Mode"] for r in rows]==["Clean","Joint" if method=="cpgc" else "Text-only","Matched Clean"],"CSV mode order")
  for r in rows:
   for k in METRICS:
    require(r[k]==format(groups[r["Mode"]][k],".2f") if rounded else float(r[k])==groups[r["Mode"]][k],"CSV summary agreement")
 if scope=="full" and victim=="rasa":require(summary["native_TopK"]==128,"native Full Top128")
 path=Path(c["output_base"])/scope/"audit"/method/dataset/victim
 with attempt(c,path,scope) as out:
  report={"status":"PASS","scope":scope,"method":method,"dataset":dataset,"victim":victim,
    "Gallery":len(manifest.gallery),"Query":len(manifest.queries),"units_verified":units,"CUDA":False,"neural_forward":0,
    "groups":groups,"config_sha256":c["sha256"],"result":asdict(ref(root/"completed/manifest.json"))}
  out.publish("completed",{"audit.json":json_bytes(report)},{"kind":"independent_CPU_audit"},sources=(ref(root/"completed/manifest.json"),ref(c["path"])))
 print("CPGC_SEP_INDEPENDENT_CPU_AUDIT_PASS",method,dataset,victim,scope,flush=True);return 0
def smoke_ready(c,method):
 for ds in ("cuhk","rstp","icfg"):
  for victim in ("irra","rasa"):
   p=Path(c["output_base"])/"smoke/audit"/method/ds/victim
   s=ArtifactStore(p,new_base=c["output_base"]);s.verify("completed")
   r=json.loads((p/"completed/audit.json").read_text())
   require(r["status"]=="PASS" and r["scope"]=="smoke" and r["method"]==method and r["dataset"]==ds and r["victim"]==victim and r["config_sha256"]==c["sha256"],"all real Smoke and CPU audit required")
 return True
