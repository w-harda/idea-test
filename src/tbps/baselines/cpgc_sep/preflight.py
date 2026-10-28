"""只读CPU来源检查；不导入模型、不创建实验Attempt。"""
from pathlib import Path
from dataclasses import asdict
import json,subprocess,hashlib,importlib.metadata
from tbps.artifacts import FileRef,checked,digest,json_bytes
from .dataset import require,read_test
from .runtime import config,ref
def run(path,*,role="victim"):
 c=config(path)
 versions={k:importlib.metadata.version(k) for k in c["runtime_versions"][role]}
 require(versions==c["runtime_versions"][role],"runtime dependency mismatch")
 manifests={}
 for ds in ("cuhk","rstp","icfg"):
  m=read_test(c["datasets"][ds])
  manifests[ds]={"Gallery":len(m.gallery),"Query":len(m.queries),"PIDs":len(set(q.pid for q in m.queries)),"manifest_sha256":digest(asdict(m))}
 from .victims import VictimSource
 for model in ("irra","rasa"):
  for ds in ("cuhk","rstp","icfg"):VictimSource(c,model,ds).verify()
 for p,h in c.get("protected_files",{}).items():checked(FileRef(p,h))
 assert not any((Path(c["output_base"])/x).exists() for x in ["training","train"]), "no training namespace"
 report={"status":"PASS","role":role,"config":asdict(ref(path)),"datasets":manifests,"runtime":versions,
  "CUDA_initialization":False,"model_load":False,"neural_forward":0,"training":False,
  "CPGC_official_commit":c["official"]["cpgc"]["commit"],"SEP_official_commit":c["official"]["sep"]["commit"],
  "victim_count":6,"official_UAP_only":True}
 print("CPGC_SEP_CPU_PREFLIGHT_PASS",json.dumps(report,ensure_ascii=False),flush=True)
 return c,report
