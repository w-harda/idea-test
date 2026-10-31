"""无固定GPU显存门槛的SEP恢复调度；已成功产物显式只读复用。"""
from pathlib import Path
import os,sys,json,time,subprocess,fcntl,importlib.util,hashlib,traceback,datetime,threading
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json"
CONTROL=Path("/home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101")
PREP=Path("/home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_prepare_20261101")
OLD=Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2")
sys.path.insert(0,str(ROOT/"src"))
from tbps.artifacts import sha_file
from tbps.baselines.cpgc_sep.runtime import config,resource_snapshot
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from tbps.baselines.cpgc_sep.step_reuse import resolve_step,step_key,verify_completed
from tbps.baselines.cpgc_sep.source_reuse import resolve
check_helper=hashlib.sha256(Path('/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-resume-20261029/operations/cpgc_sep_resume.py').read_bytes()).hexdigest()
if check_helper!='f60fd59333a671f33ce99595cbf0c298364c07db21264c981ee18f6f87b1e8f9':raise RuntimeError("frozen recovery helper SHA mismatch")
spec=importlib.util.spec_from_file_location("resume_reference","/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-resume-20261029/operations/cpgc_sep_resume.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now().astimezone().isoformat()
def check(ok,msg):
 if not ok:raise RuntimeError(msg)
def env():return dict(os.environ,PYTHONPATH=str(ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",CUDA_VISIBLE_DEVICES="0")
def schedule(c):
 o=Path(c["output_base"])
 for scope in("smoke","full"):
  for ds in("cuhk","rstp","icfg"):
   for v in("irra","rasa"):
    yield m.Step("sep",scope,ds,v,"source",o/scope/"sep_source"/ds/v)
    yield m.Step("sep",scope,ds,v,"target",o/scope/"sep"/ds/v)
    yield m.Step("sep",scope,ds,v,"audit",o/scope/"audit/sep"/ds/v)
def command(c,s):
 action="sep-source" if s.kind=="source" else s.kind
 cmd=[c["victim_python"],"-u","-B","-m","tbps.baselines.cpgc_sep.cli","--config",str(CONFIG),"--action",action,"--method",s.method,"--scope",s.scope,"--dataset",s.dataset,"--victim",s.victim]
 if s.gpu:cmd+=["--allow-full" if s.scope=="full" else "--allow-gpu-smoke"]
 return cmd
def completed_count(c):
 return 6+sum((Path(c["output_base"])/"full/audit/sep"/d/v/"completed/audit.json").is_file()for d in("cuhk","rstp","icfg")for v in("irra","rasa"))
def protected():
 n=0
 snapshots=(PREP/"v3_metadata.jsonl",Path("/home/lzf/ldx/tmp/cpgc_sep_throughput_prepare_v3_20261031/v2_metadata.jsonl"))
 for snapshot in snapshots:
  for line in snapshot.open():
   p,size,mtime,inode=json.loads(line);stat=Path(p).stat()
   check((stat.st_size,stat.st_mtime_ns,stat.st_ino)==(size,mtime,inode),"prior protected file changed: "+p);n+=1
 return n
def final_report(c):
 import csv,io
 out=Path(c["output_base"])/"derived_report";out.mkdir(exist_ok=False)
 rows=[];evidence=[]
 for method in("cpgc","sep"):
  for ds in("cuhk","rstp","icfg"):
   for v in("irra","rasa"):
    if method=="cpgc":
     item=next(x for x in c["historical_full_results"]if x["dataset"]==ds and x["victim"]==v)
     base=Path(item["base"]);target=base/"full/cpgc"/ds/v;audit=base/"full/audit/cpgc"/ds/v
     version="v3"if(ds,v)==("icfg","rasa")else"v2"
    else:
     target=resolve_step(c,method,ds,v,"full","target").root
     audit=resolve_step(c,method,ds,v,"full","audit").root;base=Path(c["output_base"]);version="v4"
    s=read(target/"completed/summary.json");a=read(audit/"completed/audit.json")
    check(s["status"]==a["status"]=="PASS"and s["groups"]==a["groups"],"final complete audit")
    check(a["result"]["sha256"]==sha_file(target/"completed/manifest.json"),"final audit SHA")
    for mode,metric in s["groups"].items():rows.append({"Method":method,"Dataset":ds,"Victim":v,"Execution":version,"Mode":mode,**metric})
    evidence.append({"method":method,"dataset":ds,"victim":v,"version":version,"summary":str(target/"completed/summary.json"),"sha256":sha_file(target/"completed/summary.json"),"audit":str(audit/"completed/audit.json"),"audit_sha256":sha_file(audit/"completed/audit.json")})
 for rounded in(False,True):
  with(out/("RESULTS_2DP.csv"if rounded else"RESULTS.csv")).open("x")as f:
   w=csv.DictWriter(f,lineterminator="\n",fieldnames=["Method","Dataset","Victim","Execution","Mode","R@1","R@5","R@10","mAP"]);w.writeheader()
   for row in rows:w.writerow({k:(format(v,".2f")if rounded and k in("R@1","R@5","R@10","mAP")else v)for k,v in row.items()})
 (out/"SOURCE_EVIDENCE.json").write_text(json.dumps(evidence,indent=2)+"\n")
 (out/"REPORT.md").write_text("# C-PGC / SEP 十二组正式评测\n\n全部十二组Full及独立CPU审计PASS。CPGC六组只读引用v2/v3，SEP六组来自无显存门槛v4；分别保留真实来源身份。\n\n"+(ROOT/"docs/framework/results/cpgc_sep_no_gpu_floor/REPORT.md").read_text()+"\n\n完整指标见RESULTS.csv与RESULTS_2DP.csv。\n")
 # Publication in an independent checkout never alters the running source tree.
 work=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-v4-results")
 check(not work.exists(),"publication checkout already exists")
 subprocess.run(["git","-C",str(ROOT),"worktree","add","-b","codex/cpgc-sep-v4-results",str(work),"HEAD"],check=True)
 import shutil
 dest=work/"docs/framework/results/cpgc_sep_v4_final";dest.mkdir()
 for p in out.iterdir():shutil.copy2(p,dest/p.name)
 subprocess.run(["git","-C",str(work),"add",str(dest)],check=True)
 subprocess.run(["git","-C",str(work),"diff","--cached","--check"],check=True)
 subprocess.run(["git","-C",str(work),"-c","user.name=w-harda","-c","user.email=wharda@163.com","commit","-m","Report twelve audited C-PGC SEP Full results with explicit v2 v3 identities"],check=True)
 subprocess.run(["git","-C",str(work),"push","-u","origin","codex/cpgc-sep-v4-results"],check=True)
def main():
 check("--allow-no-gpu-floor" in sys.argv,"explicit no-floor continuation authorization")
 CONTROL.mkdir(exist_ok=True)
 with open("/home/lzf/ldx/tmp/cpgc_sep_v2_resume.lock","a")as global_lock,(CONTROL/"process.lock").open("a")as lock:
  fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  check(not(CONTROL/"started").exists(),"no-floor launcher identity consumed")
  check(not subprocess.run(["pgrep","-af","tbps.baselines.cpgc_sep.cli|cpgc_sep_resume.py"],capture_output=True,text=True).stdout.strip(),"other native chain running")
  c=config(CONFIG);head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip();cfgsha=sha_file(CONFIG)
  check(not subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True).strip(),"dirty execution tree")
  with(CONTROL/"started").open("x")as f:f.write(json.dumps({"time":now(),"commit":head,"config_sha":cfgsha})+"\n");f.flush();os.fsync(f.fileno())
  for name,value in(("pid",os.getpid()),("pgid",os.getpgrp()),("sid",os.getsid(0))):(CONTROL/("supervisor."+name)).write_text(str(value)+"\n")
  budget=m.Budget();current=None;seq=0
  # Independent telemetry only; it never gates or signals a GPU process.
  monitor_stop=threading.Event()
  def telemetry():
   while not monitor_stop.wait(15):
    try:
     result=subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=10)
     with(CONTROL/"GPU_monitor.jsonl").open("a")as f:f.write(json.dumps({"time":now(),"step":current.label if current else None,"resource":result.stdout.strip(),"returncode":result.returncode})+"\n")
    except Exception:pass
  threading.Thread(target=telemetry,daemon=True).start()
  def state(phase,**kw):
   r={"time":now(),"phase":phase,"step":current.label if current else None,"completed_Full":completed_count(c),"wait_seconds":budget.used,**kw}
   m.atomic(CONTROL/"status.json",r);(CONTROL/"phase").write_text(phase+"\n")
   with(CONTROL/"status.log").open("a")as f:f.write(json.dumps(r)+"\n")
   print(phase,r["step"],flush=True)
  def resource_event(r):
   with(CONTROL/"resources.jsonl").open("a")as f:f.write(json.dumps(r)+"\n")
   state("WAITING_HOST_RESOURCES"if r["blocked"]else"RESOURCE_READY",resources=r)
  def identity():
   check(subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()==head,"execution HEAD changed")
   check(not subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain"],text=True).strip(),"execution tree changed")
   check(sha_file(CONFIG)==cfgsha,"configuration changed")
  try:
   state("VERIFYING_HISTORY")
   evidence=read(PREP/"verified_reuse.json")
   check(evidence["status"]=="PASS"and evidence["Full_audits"]==6,"six Full and completed SEP SHA/audit evidence required")
   state("HISTORY_PROTECTED",files=protected())
   # Every stage uses this committed configuration and existing native CLI gates.
   for role,py in(("source",c["source_python"]),("victim",c["victim_python"])):
    state("CPU_PREFLIGHT",role=role)
    with(CONTROL/("preflight_"+role+".log")).open("x")as f:
     p=subprocess.run([py,"-u","-B","-c","import sys;from tbps.baselines.cpgc_sep.preflight import run;run(sys.argv[1],role=sys.argv[2])",str(CONFIG),role],cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT)
    check(p.returncode==0,"CPU preflight failed")
   for current in schedule(c):
    identity()
    prior=resolve_step(c,current.method,current.dataset,current.victim,current.scope,current.kind)
    if (prior.root/"completed/manifest.json").exists():
     state("SKIP_VERIFIED_COMPLETE",original_root=str(prior.root));continue
    while True:
     if current.gpu:
      # No GPU memory or utilization veto. Only genuine host resource shortage can wait.
      while True:
       snap=resource_snapshot(c);resource_event(snap)
       if not snap["blocked"]:break
       budget.sleep()
     identity();check(not current.root.exists(),"conflicting consumed Attempt")
     seq+=1;log=CONTROL/(f"{seq:04d}_"+current.label+".log");state("RUNNING_STEP",log=str(log),gpu=current.gpu)
     with log.open("x")as f:p=subprocess.run(command(c,current),cwd=ROOT,env=env(),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
     (CONTROL/(log.name+".exit")).write_text(str(p.returncode)+"\n")
     if p.returncode:
      if m.safe_prelaunch_busy(current,log.read_text()):state("PRELAUNCH_RESOURCE_BUSY");budget.sleep();continue
      raise RuntimeError("native execution failed; preserve and stop: "+str(log))
     resolve_step(c,current.method,current.dataset,current.victim,current.scope,current.kind).verify("completed");state("STEP_PASS")
     break
   current=None;check(completed_count(c)==12,"not 12 audited Full")
   state("FINAL_PROTECTION",protected_files=protected())
   state("PUBLISHING_RESULTS");final_report(c)
   state("SUCCESS",marker="CPGC_SEP_TWELVE_FULL_AND_CPU_AUDIT");(CONTROL/"exit").write_text("0\n")
  except BaseException as e:
   state("FAIL",error=repr(e))
   with(CONTROL/"FAIL.json").open("x")as f:json.dump({"time":now(),"error":repr(e),"traceback":traceback.format_exc()},f,indent=2)
   (CONTROL/"exit").write_text("1\n");raise
if __name__=="__main__":main()
