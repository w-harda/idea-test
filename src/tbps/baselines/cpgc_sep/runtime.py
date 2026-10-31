"""独立资源、不可变单元和CPU来源门禁。不创建训练器或Optimizer。"""
from pathlib import Path
from dataclasses import asdict
import json,os,time,shutil,subprocess,io,contextlib,fcntl
from tbps.artifacts import ArtifactStore,FileRef,checked,sha_file,digest,json_bytes,ArtifactError
from .verified_store import VerifiedArtifactStore as ArtifactStore
from .dataset import require
def config(path):
 p=Path(path).resolve();c=json.loads(p.read_text());c["path"]=str(p);c["sha256"]=sha_file(p)
 for f,h in c["producers"].items():checked(FileRef(f,h))
 from .official import checked_tree
 for d in c["official"].values():checked_tree(d["root"],d["commit"],d["files"])
 for d in c["datasets"].values():checked(FileRef(**d["annotation"]))
 for r in c["resources"].values():checked(FileRef(r["path"],r["sha256"]))
 return c
def ref(path):return FileRef(str(Path(path).resolve()),sha_file(path))
def resources(c):
 s={}
 for field in ["memory.free","memory.total","utilization.gpu"]:
  r=subprocess.run(["nvidia-smi","--id=0","--query-gpu="+field,"--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
  require(r.returncode==0 and r.stdout.strip().isdigit(),"GPU state unavailable")
  s[field]=int(r.stdout.strip())
 ps=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
 require(ps.returncode==0,"GPU process snapshot failed");s["compute_processes"]=ps.stdout.strip().splitlines()
 mem={l.split(":")[0]:int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith(("MemAvailable:","MemTotal:"))}
 disk=shutil.disk_usage(c["output_parent"]);fs=os.statvfs(c["output_parent"])
 s.update(RAM_available=mem["MemAvailable"],disk_available=disk.free,inode_available=fs.f_favail,time_unix=time.time(),policy="shared_observe_only; never signal external processes")
 lim=c["resource_gate"]
 require(s["memory.free"]>=lim["minimum_free_mib"] and s["RAM_available"]>=lim["RAM_bytes"] and s["disk_available"]>=lim["disk_bytes"] and s["inode_available"]>=lim["inodes"],"RESOURCE_BUSY")
 return s
def tensor_bytes(x):
 import torch
 b=io.BytesIO();torch.save(x,b);return b.getvalue()
def load_tensor(store,key,name="features.pt"):
 import torch
 store.verify(key)
 return torch.load(store.root/key/name,map_location="cpu",weights_only=True)
def unit(store,key,producer,provenance,*,sources=(),resume=False):
 """仅显式恢复已完整提交单元；未完成Claim/pending永久拒绝自动重做。"""
 final=store.root/key
 if final.exists():
  require(resume,"existing unit requires explicit resume")
  return store.verify(key,expected_provenance=provenance)
 claim=store.root/(key+".Claim.json");store.root.mkdir(parents=True,exist_ok=True)
 require(not claim.exists() and not store.pending(),"uncommitted Claim/pending; no automatic retry")
 with claim.open("x") as f:f.write(json_bytes(provenance).decode());f.flush();os.fsync(f.fileno())
 files=producer()
 return store.publish(key,files,provenance,sources=sources)
@contextlib.contextmanager
def attempt(c,path,scope,*,resume=False):
 p=Path(path);store=ArtifactStore(p,new_base=c["output_base"])
 require(not p.exists() or resume,"Attempt exists; no overwrite")
 p.mkdir(parents=True,exist_ok=True)
 with (p/".process.lock").open("a") as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise ArtifactError("duplicate experiment process")
  require(not (p/"FAIL.json").exists() and not (p/"completed").exists(),"FAIL/completed cannot restart")
  if not (p/"started.json").exists():
   with (p/"started.json").open("x") as f:json.dump({"config_sha256":c["sha256"],"scope":scope,"pid":os.getpid(),"time":time.time()},f)
  else:
   r=json.loads((p/"started.json").read_text());require(resume and r["config_sha256"]==c["sha256"],"resume identity")
  try:yield store
  except BaseException as e:
   import traceback
   with (p/"FAIL.json").open("x") as f:json.dump({"error":repr(e),"traceback":traceback.format_exc(),"time":time.time()},f,indent=2)
   raise
