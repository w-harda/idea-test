"""独立prototype IO；Windows不保存实验源码，所有新写入限定本prototype输出。"""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
IMAGE_ROOT=Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def safe(p):
 p=Path(p).resolve()
 if not p.is_relative_to(ROOT.resolve()):raise ValueError("write must remain inside this prototype")
 return p
def save(p,x):
 p=safe(p);p.parent.mkdir(parents=True,exist_ok=True);t=safe(p.with_suffix(p.suffix+".tmp"))
 t.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8");os.replace(t,p)
def verify(locks):
 for p,h in locks.items():
  if sha(p)!=h:raise ValueError("SHA mismatch: "+p)
def own_hashes():return {str(p):sha(p) for p in [*sorted(ROOT.glob("*.py")),ROOT/"config.json"]}
def tensor_save(p,x):
 import torch
 p=safe(p);p.parent.mkdir(parents=True,exist_ok=True);t=safe(p.with_suffix(p.suffix+".tmp"))
 torch.save(x,t);os.replace(t,p)
def read_condition(entry):
 import torch
 verify({entry["tensor_path"]:entry["tensor_sha256"],entry["metadata_path"]:entry["metadata_sha256"]})
 value=torch.load(entry["tensor_path"],map_location="cpu",weights_only=True)
 for key in ("z_t","c_attr","v_local"):
  x=value[key]
  if x.shape!=(512,) or x.dtype!=torch.float32 or not torch.isfinite(x).all() or x.requires_grad:raise ValueError("invalid frozen condition "+key)
 if abs(float(value["z_t"].norm())-1)>2e-4:raise ValueError("non-unit caption feature")
 if not value["has_local"] and (value["c_attr"].any() or value["v_local"].any()):raise ValueError("K0 condition must be zero")
 return value
def checked_cache(path):
 p=Path(path);m=load(p/"manifest.json");done=load(p/"completed.json")
 if not done["method_audit_passed"] or done["manifest_sha256"]!=sha(p/"manifest.json"):raise ValueError("incomplete or modified cache")
 for role in ("train","validation"):
  if len(m[role])!=m["counts"][role] or len({e["row_id"] for e in m[role]})!=len(m[role]):raise ValueError("cache coverage")
 if set(e["pid"] for e in m["train"])&set(e["pid"] for e in m["validation"]):raise ValueError("PID leakage")
 return m
