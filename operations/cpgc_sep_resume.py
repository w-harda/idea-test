"""只读复用已提交产物；仅对尚未创建Attempt的RESOURCE_BUSY作有界等待。"""
from pathlib import Path
from dataclasses import dataclass
import argparse,json,subprocess,sys,os,time,hashlib,fcntl,shutil,datetime,traceback
NATIVE=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-uap-detach-20261029")
CONFIG=NATIVE/"configs/framework/cpgc_sep_tbps.uap_detach.server.json"
NATIVE_COMMIT="36cf91f97ea3c483883eb6d93456c51ef7d662f0"
CONFIG_SHA="4b56eb60a3f2c928ef084fad64f1648bee17129e450b707a4d166169bc8a370a"
CONTROL=Path("/home/lzf/ldx/tmp/cpgc_sep_resume_v1_20261029")
GLOBAL_LOCK=Path("/home/lzf/ldx/tmp/cpgc_sep_v2_resume.lock")
class Stop(RuntimeError):pass
class WaitExpired(Stop):pass
def require(ok,msg):
 if not ok:raise Stop(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now().astimezone().isoformat()
def atomic(p,data):
 t=Path(str(p)+".tmp")
 with t.open("w") as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(t,p)
@dataclass(frozen=True)
class Step:
 method:str
 scope:str
 dataset:str
 victim:str
 kind:str
 root:Path
 @property
 def label(self):return "_".join(filter(None,(self.method,self.scope,self.dataset,self.victim,self.kind)))
 @property
 def gpu(self):return self.kind in ("source","target")
def steps(c):
 o=Path(c["output_base"])
 for method in ("cpgc","sep"):
  for scope in ("smoke","full"):
   for ds in ("cuhk","rstp","icfg"):
    if method=="cpgc":yield Step(method,scope,ds,"","source",o/scope/"cpgc_source"/ds)
    for v in ("irra","rasa"):
     if method=="sep":yield Step(method,scope,ds,v,"source",o/scope/"sep_source"/ds/v)
     yield Step(method,scope,ds,v,"target",o/scope/method/ds/v)
     yield Step(method,scope,ds,v,"audit",o/scope/"audit"/method/ds/v)
def report_for(s):
 return s.root/"completed"/("audit.json" if s.kind=="audit" else "summary.json" if s.kind=="target" else "report.json")
def verify_step(c,s,deep=True):
 """仅SHA/JSON；不运行原audit、排名、Tensor反序列化或模型。"""
 from tbps.artifacts import ArtifactStore
 if not s.root.exists():return None
 require(not any(s.root.rglob("FAIL*")),"existing FAIL; no retry: "+str(s.root))
 store=ArtifactStore(s.root,new_base=c["output_base"])
 require(not store.pending(),"pending; isolated recovery required: "+str(s.root))
 require((s.root/"completed/manifest.json").is_file(),"partial Attempt; no automatic replay: "+str(s.root))
 start=read(s.root/"started.json")
 require(start["config_sha256"]==c["sha256"] and start["scope"]==s.scope,"completed Attempt config identity mismatch")
 for claim in s.root.glob("*.Claim.json"):
  require((s.root/claim.name.removesuffix(".Claim.json")/"manifest.json").is_file(),"orphan Claim: "+str(claim))
 manifest=store.verify("completed")
 if deep:
  for p in sorted(s.root.glob("*/manifest.json")):
   if p.parent.name!="completed":store.verify(p.parent.name)
 r=read(report_for(s))
 require(r["status"]=="PASS" and r["scope"]==s.scope and r["dataset"]==s.dataset,"completed report identity")
 if s.victim:require(r["victim"]==s.victim,"Victim mismatch")
 if s.kind=="audit":
  require(r["config_sha256"]==c["sha256"] and r["method"]==s.method and r["CUDA"] is False and r["neural_forward"]==0,"CPU audit identity")
  target=Path(c["output_base"])/s.scope/s.method/s.dataset/s.victim/"completed"
  require(r["result"]["path"]==str(target/"manifest.json") and r["result"]["sha256"]==sha(target/"manifest.json"),"audit result SHA binding")
  require(r["groups"]==read(target/"summary.json")["groups"],"audit and summary metric mismatch")
 else:
  require(read(s.root/"protocol/protocol.json")["config_sha256"]==c["sha256"],"source protocol SHA identity")
  ng,nq=(128,2) if s.scope=="smoke" else {"cuhk":(3074,6156),"rstp":(1000,2000),"icfg":(19848,19848)}[s.dataset]
  require((r["Gallery"],r["Query"])==(ng,nq) if s.kind=="target" else (r["gallery"],r["queries"])==(ng,nq),"complete Gallery/Query")
  if s.kind=="target":
   require(r["method"]==s.method and r["model_before"]==r["model_after"],"frozen model evidence")
   require(r["counts"]["backward"]==r["counts"]["optimizer"]==0,"inference-only target")
   import csv
   for name,rounded in (("metrics.csv",False),("metrics_2dp.csv",True)):
    rows=list(csv.DictReader((s.root/"completed"/name).open()))
    require(set(x["Mode"] for x in rows)==set(r["groups"]),"all metric groups")
    for row in rows:
     for k in ("R@1","R@5","R@10","mAP"):
      require(row[k]==format(r["groups"][row["Mode"]][k],".2f") if rounded else float(row[k])==r["groups"][row["Mode"]][k],"CSV/summary mismatch")
 return {"label":s.label,"manifest":str(s.root/"completed/manifest.json"),"sha256":sha(s.root/"completed/manifest.json"),"deep_payload_SHA":deep}
def resources(c):
 """不import torch；失败项逐个记录，利用率只观测。"""
 def smi(fields):
  p=subprocess.run(["nvidia-smi","--id=0","--query-gpu="+fields,"--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
  require(p.returncode==0,"GPU status unavailable: "+p.stderr)
  parts=[x.strip() for x in p.stdout.strip().split(",")]
  require(len(parts)==3 and all(x.isdigit() for x in parts),"invalid GPU resource output")
  return list(map(int,parts))
 free,total,util=smi("memory.free,memory.total,utilization.gpu")
 p=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15)
 require(p.returncode==0,"GPU process state unavailable")
 mem={l.split(":")[0]:int(l.split()[1])*1024 for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:")}
 actual={"minimum_free_mib":free,"RAM_bytes":mem["MemAvailable"],"disk_bytes":shutil.disk_usage(c["output_parent"]).free,"inodes":os.statvfs(c["output_parent"]).f_favail}
 gate=c["resource_gate"];blocked={k:{"actual":v,"required":gate[k],"deficit":gate[k]-v} for k,v in actual.items() if v<gate[k]}
 return {"time":now(),"status":"RESOURCE_BUSY" if blocked else "READY","actual":actual,"required":{k:gate[k] for k in actual},"blocked":blocked,"GPU_total_MiB":total,"GPU_utilization":util,"compute_processes":p.stdout.strip().splitlines()}
class Budget:
 def __init__(self,maximum=43200,interval=300):self.used=0.;self.maximum=maximum;self.interval=interval
 def sleep(self,sleeper=time.sleep,clock=time.monotonic):
  remaining=self.maximum-self.used
  if remaining<=0:raise WaitExpired("RESOURCE_WAIT_TIMEOUT: total resource wait reached 12h")
  before=clock();sleeper(min(self.interval,remaining));self.used+=max(0.,clock()-before)
def wait_ready(c,budget,emit,probe=resources,sleeper=time.sleep,clock=time.monotonic):
 while True:
  r=probe(c);emit(r)
  if r["status"]=="READY":return r
  require(r["status"]=="RESOURCE_BUSY","non-resource error must not wait")
  budget.sleep(sleeper,clock)
def safe_prelaunch_busy(s,log):
 if not s.gpu or s.root.exists():return False
 lines=log.strip().splitlines()
 return bool(lines and lines[-1]=="tbps.artifacts.ArtifactError: RESOURCE_BUSY" and
  'in resources' in log and '/tbps/baselines/cpgc_sep/runtime.py' in log)
def frozen():
 require(subprocess.check_output(["git","-C",str(NATIVE),"rev-parse","HEAD"],text=True).strip()==NATIVE_COMMIT,"native HEAD changed")
 require(not subprocess.check_output(["git","-C",str(NATIVE),"status","--porcelain"],text=True).strip(),"native execution tree changed")
 require(sha(CONFIG)==CONFIG_SHA,"native configuration SHA changed")
def native_command(c,s):
 py=c["source_python"] if s.method=="cpgc" and s.kind=="source" else c["victim_python"]
 action=s.method+"-source" if s.kind=="source" else s.kind
 cmd=[py,"-u","-B","-m","tbps.baselines.cpgc_sep.cli","--config",str(CONFIG),"--action",action,"--method",s.method,"--dataset",s.dataset,"--scope",s.scope]
 if s.victim:cmd+=["--victim",s.victim]
 if s.gpu:cmd+=["--allow-full" if s.scope=="full" else "--allow-gpu-smoke"]
 return cmd
def environment():
 return dict(os.environ,PYTHONPATH=str(NATIVE/"src"),PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",CUDA_VISIBLE_DEVICES="0")
def main():
 p=argparse.ArgumentParser();p.add_argument("--inspect",action="store_true");p.add_argument("--allow-resume",action="store_true");a=p.parse_args()
 sys.path.insert(0,str(NATIVE/"src"));frozen()
 c=read(CONFIG);c["sha256"]=CONFIG_SHA
 if a.inspect:
  print(json.dumps(resources(c),ensure_ascii=False),flush=True)
  for s in steps(c):
   r=verify_step(c,s)
   if r:print("VERIFIED_REUSE",json.dumps(r),flush=True)
  return
 require(a.allow_resume,"explicit resume authorization required")
 CONTROL.mkdir(parents=True,exist_ok=True)
 with GLOBAL_LOCK.open("a") as global_lock,(CONTROL/"process.lock").open("a") as lock:
  try:
   fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise Stop("duplicate recovery supervisor")
  require(not (CONTROL/"started").exists(),"recovery identity consumed; no automatic restart")
  other=subprocess.run(["pgrep","-af","tbps.baselines.cpgc_sep.cli|run_cpgc_sep_full.sh"],capture_output=True,text=True)
  require(not other.stdout.strip(),"existing native experiment; no parallel launch")
  with (CONTROL/"started").open("x") as f:f.write(now()+"\n");f.flush();os.fsync(f.fileno())
  for name,v in [("supervisor.pid",os.getpid()),("supervisor.pgid",os.getpgrp()),("supervisor.sid",os.getsid(0))]:(CONTROL/name).write_text(str(v)+"\n")
  budget=Budget();seq=0;skips=[];current=None
  def state(phase,**kw):
   record={"time":now(),"phase":phase,"step":current.label if current else None,"resource_wait_used_seconds":budget.used,"resource_wait_limit_seconds":budget.maximum,**kw}
   atomic(CONTROL/"status.json",record);(CONTROL/"phase").write_text(phase+"\n")
   with (CONTROL/"status.log").open("a") as f:f.write(json.dumps(record,ensure_ascii=False)+"\n")
   print(phase,record["step"],flush=True)
  def probe_event(r):
   with (CONTROL/"resources.jsonl").open("a") as f:f.write(json.dumps(r,ensure_ascii=False)+"\n")
   atomic(CONTROL/"latest_resources.json",r)
   state("WAITING_RESOURCE" if r["status"]=="RESOURCE_BUSY" else "RESOURCE_READY",resources=r)
  try:
   state("VERIFYING_REUSE")
   # 在任何新GPU任务之前核验所有已存在步骤；真实部分Attempt/FAIL绝不跳过。
   for current in steps(c):
    r=verify_step(c,current)
    if r:skips.append(r);print("VERIFIED_REUSE",current.label,flush=True)
   atomic(CONTROL/"verified_reuse.json",skips)
   current=None
   # 真实CPU预检，不加载模型或初始化CUDA。
   for role,py in (("source",c["source_python"]),("victim",c["victim_python"])):
    state("CPU_PREFLIGHT_"+role)
    with (CONTROL/("preflight_"+role+".log")).open("x") as f:
     r=subprocess.run([py,"-u","-B","-c","import sys;from tbps.baselines.cpgc_sep.preflight import run;run(sys.argv[1],role=sys.argv[2])",str(CONFIG),role],cwd=NATIVE,env=environment(),stdout=f,stderr=subprocess.STDOUT)
    require(r.returncode==0,"real CPU preflight failed: "+role)
   reused={x["label"]:x for x in skips}
   for current in steps(c):
    if current.label in reused:
     state("SKIP_VERIFIED_COMPLETE",evidence=reused[current.label]);continue
    while True:
     frozen();require(verify_step(c,current) is None,"new conflicting output appeared")
     if current.gpu:wait_ready(c,budget,probe_event)
     frozen();require(not current.root.exists(),"Attempt consumed while waiting")
     seq+=1;log=CONTROL/(str(seq).zfill(4)+"_"+current.label+".log");state("RUNNING_STEP",log=str(log),gpu_allowed=current.gpu)
     with log.open("x") as f:r=subprocess.run(native_command(c,current),cwd=NATIVE,env=environment(),stdout=f,stderr=subprocess.STDOUT)
     (CONTROL/(log.name+".exit")).write_text(str(r.returncode)+"\n")
     if r.returncode==0:
      verify_step(c,current,deep=False);state("STEP_PASS",log=str(log));break
     text=log.read_text()
     if safe_prelaunch_busy(current,text):
      state("PRELAUNCH_RESOURCE_RACE",log=str(log))
      # 只允许CPU准入阶段、输出目录仍未创建的资源回落；绝不重试GPU Attempt。
      probe_event(resources(c));budget.sleep();continue
     raise Stop("native step failed; preserve evidence, never retry: "+str(log))
   current=None;state("SUMMARIZING")
   with (CONTROL/"summary.log").open("x") as f:r=subprocess.run([c["victim_python"],"-u","-B","-m","tbps.baselines.cpgc_sep.cli","--config",str(CONFIG),"--action","report"],cwd=NATIVE,env=environment(),stdout=f,stderr=subprocess.STDOUT)
   require(r.returncode==0,"result summary failed")
   m=read(Path(c["output_base"])/"derived_report/RUN_MANIFEST.json");require(m["Full_pass_count"]==12,"not twelve audited Full results")
   state("PUBLISHING_RESULTS")
   with (CONTROL/"publish.log").open("x") as f:r=subprocess.run([c["victim_python"],"-u","-B",str(NATIVE/"operations/cpgc_sep_publish_results.py")],cwd=NATIVE,env=environment(),stdout=f,stderr=subprocess.STDOUT)
   require(r.returncode==0,"Full complete but Git publication failed; do not rerun experiments")
   state("SUCCESS",marker="CPGC_SEP_TWELVE_FULL_AND_CPU_AUDIT");(CONTROL/"exit").write_text("0\n")
  except BaseException as exc:
   state("RESOURCE_WAIT_TIMEOUT" if isinstance(exc,WaitExpired) else "FAIL",error=repr(exc))
   with (CONTROL/"FAIL.json").open("x") as f:json.dump({"time":now(),"error":repr(exc),"traceback":traceback.format_exc(),"step":current.label if current else None},f,ensure_ascii=False,indent=2)
   (CONTROL/"exit").write_text("1\n");raise
if __name__=="__main__":main()
