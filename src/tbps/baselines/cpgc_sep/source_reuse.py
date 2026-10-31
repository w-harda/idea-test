"""跨执行配置的只读Source复用；显式绑定旧配置与Manifest，算法和输入必须一致。"""
from pathlib import Path
import json
from tbps.artifacts import ArtifactStore,FileRef,checked
from .verified_store import VerifiedArtifactStore as ArtifactStore
from .dataset import require
SEMANTIC_FIELDS=("schema","official","datasets","victims","resources","bert_source","cpgc_protocol","sep_protocol","nltk_root","runtime_versions","victim_git","protected_files","smoke")
ENGINEERING_FIELDS={"query_chunk","cache_chunk","image_batch","text_batch","fusion_batch","token_cache_bytes","sep_fusion_batch"}
def resolve(c,method,dataset,scope,victim=None):
 key="/".join((scope,method,dataset))
 binding=c.get("source_reuse",{}).get(key)
 if binding is None:
  root=Path(c["output_base"])/scope/(method+"_source")/dataset
  if method=="sep":root=root/victim
  return ArtifactStore(root,new_base=c["output_base"])
 require(method=="cpgc","only already completed CPGC sources explicitly reusable")
 checked(FileRef(**binding["config"]))
 parent=json.loads(Path(binding["config"]["path"]).read_text())
 for k in SEMANTIC_FIELDS:require(c[k]==parent[k],"source reuse semantic mismatch: "+k)
 left={k:v for k,v in c["evaluation"].items() if k not in ENGINEERING_FIELDS}
 right={k:v for k,v in parent["evaluation"].items() if k not in ENGINEERING_FIELDS}
 require(left==right,"non-engineering evaluation change")
 root=Path(binding["root"])
 require(root==Path(parent["output_base"])/scope/"cpgc_source"/dataset,"explicit original Source path")
 checked(FileRef(str(root/"completed/manifest.json"),binding["manifest_sha256"]))
 store=ArtifactStore(root,new_base=parent["output_base"]);store.verify("completed")
 require(not store.pending() and not (root/"FAIL.json").exists(),"source pending/FAIL cannot reuse")
 report=json.loads((root/"completed/report.json").read_text())
 proto=json.loads((root/"protocol/protocol.json").read_text())
 require(proto["config_sha256"]==binding["config"]["sha256"],"parent Source config")
 require(report["status"]=="PASS" and report["scope"]==scope and report["dataset"]==dataset,"source completion identity")
 return store
