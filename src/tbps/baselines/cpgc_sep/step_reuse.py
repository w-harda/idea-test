"""运行策略变更后的显式只读复用；不改写旧配置、Manifest或成功回执。"""
from pathlib import Path
import json
from .dataset import require
from .verified_store import VerifiedArtifactStore,DigestSession
from tbps.artifacts import FileRef,checked
SEMANTICS=("schema","official","datasets","victims","resources","bert_source","cpgc_protocol","sep_protocol","evaluation","nltk_root","runtime_versions","victim_git","protected_files","smoke")
POLICY_FILES={"baselines/cpgc_sep/runtime.py","baselines/cpgc_sep/audit.py","baselines/cpgc_sep/source_reuse.py","baselines/cpgc_sep/step_reuse.py"}
def step_key(method,dataset,victim,scope,kind):
 return "/".join((scope,method,dataset,victim or "-",kind))
def path_for(base,method,dataset,victim,scope,kind):
 root=Path(base)/scope
 if kind=="audit":return root/"audit"/method/dataset/victim
 if kind=="source":
  root=root/(method+"_source")/dataset
  return root/victim if method=="sep"else root
 return root/method/dataset/victim
def compatible(c,parent):
 for field in SEMANTICS:require(c[field]==parent[field],"reuse changes numerical/attack protocol: "+field)
 for key in("RAM_bytes","disk_bytes","inodes"):require(c["resource_gate"][key]==parent["resource_gate"][key],"host resource gate changed")
 for path,h in parent["producers"].items():
  checked(FileRef(path,h))
  if "/src/tbps/" not in path:continue
  rel=path.split("/src/tbps/",1)[1]
  if rel in POLICY_FILES:continue
  matches=[value for key,value in c["producers"].items()if key.endswith("/src/tbps/"+rel)]
  require(matches==[h],"non-policy producer differs: "+rel)
def verify_completed(base,root,scope,method,dataset,victim,kind,config_sha,*,deep=False,session=None):
 store=VerifiedArtifactStore(root,new_base=base,session=session)
 require(root.exists()and(root/"completed/manifest.json").is_file(),"partial or missing Attempt; never retry: "+str(root))
 require(not (root/"FAIL.json").exists()and not store.pending(),"FAIL/pending preserved: "+str(root))
 require(json.loads((root/"started.json").read_text())["config_sha256"]==config_sha,"original Attempt SHA")
 for claim in root.glob("*.Claim.json"):
  require((root/claim.name.removesuffix(".Claim.json")/"manifest.json").is_file(),"orphan Claim: "+str(claim))
 store.verify("completed")
 if deep:
  for p in root.glob("*/manifest.json"):store.verify(p.parent.name)
 name="audit.json"if kind=="audit"else"report.json"if kind=="source"else"summary.json"
 report=json.loads((root/"completed"/name).read_text())
 require(report["status"]=="PASS"and report["scope"]==scope and report["dataset"]==dataset,"true completion scope")
 if victim:require(report["victim"]==victim,"Victim mismatch")
 if kind=="audit":
  require(report["config_sha256"]==config_sha and report["CUDA"]is False and report["neural_forward"]==0,"original independent CPU audit identity")
  target=path_for(base,method,dataset,victim,scope,"target")
  from tbps.artifacts import sha_file
  require(report["result"]=={"path":str(target/"completed/manifest.json"),"sha256":sha_file(target/"completed/manifest.json")},"audit result SHA")
  require(report["groups"]==json.loads((target/"completed/summary.json").read_text())["groups"],"audit metrics match")
 else:
  require(json.loads((root/"protocol/protocol.json").read_text())["config_sha256"]==config_sha,"original protocol SHA")
  ng,nq=(128,2)if scope=="smoke"else{"cuhk":(3074,6156),"rstp":(1000,2000),"icfg":(19848,19848)}[dataset]
  require((report["gallery"],report["queries"])==(ng,nq)if kind=="source"else(report["Gallery"],report["Query"])==(ng,nq),"complete official members")
  if kind=="target":
   import csv
   require(report["model_before"]==report["model_after"]and report["counts"]["backward"]==report["counts"]["optimizer"]==0,"frozen native inference")
   for filename,rounded in(("metrics.csv",False),("metrics_2dp.csv",True)):
    rows=list(csv.DictReader((root/"completed"/filename).open()))
    require(set(v["Mode"]for v in rows)==set(report["groups"]),"complete CSV modes")
    for row in rows:
     for metric in("R@1","R@5","R@10","mAP"):
      require(row[metric]==format(report["groups"][row["Mode"]][metric],".2f")if rounded else float(row[metric])==report["groups"][row["Mode"]][metric],"CSV/summary exact")
 return store,report
def resolve_step(c,method,dataset,victim,scope,kind):
 key=step_key(method,dataset,victim,scope,kind);binding=c.get("successful_step_reuse",{}).get(key)
 active=path_for(c["output_base"],method,dataset,victim,scope,kind)
 if binding is None:
  if active.exists():
   return verify_completed(c["output_base"],active,scope,method,dataset,victim,kind,c["sha256"])[0]
  return VerifiedArtifactStore(active,new_base=c["output_base"])
 require(not active.exists(),"ambiguous old/new Attempt identity")
 checked(FileRef(**binding["config"]))
 parent=json.loads(Path(binding["config"]["path"]).read_text());compatible(c,parent)
 root=Path(binding["root"])
 require(root==path_for(parent["output_base"],method,dataset,victim,scope,kind),"explicit original artifact path")
 checked(FileRef(str(root/"completed/manifest.json"),binding["manifest_sha256"]))
 store,_=verify_completed(parent["output_base"],root,scope,method,dataset,victim,kind,binding["config"]["sha256"])
 return store
