"""验证通过后在完整Round边界交接原作业；后续阶段独立进程，失败停止。"""
from pathlib import Path
import sys,os,json,time,signal,subprocess,fcntl,datetime
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116")
OLD=Path("/home/lzf/ldx/tmp/gh_attribute_response_recovery_20261119")
OLD_TRAIN=Path("/home/lzf/ldx/tmp/gh_attribute_response_20261116/train")
CTL=Path("/home/lzf/ldx/tmp/gh_performance_v1_20261119")
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116")
PY="/home/lzf/ldx/envs/tbps-image/bin/python"
CFG=SOURCE/"configs/framework/gh_attribute_response.server.json"
sys.path.insert(0,str(SOURCE/"src"))
import tbps
tbps.__path__.append(str(ROOT/"src/tbps"))
from tbps.artifacts import read_json,json_bytes,sha_file,ArtifactStore,checked,FileRef
from tbps.performance.runtime import verify_version,MANIFEST

def require(v,m):
 if not v:raise RuntimeError(m)
def save(path,value,exclusive=False):
 path=Path(path)
 if exclusive:
  with path.open("xb") as f:f.write(json_bytes(value));f.flush();os.fsync(f.fileno())
 else:
  t=path.with_name(path.name+".pending");t.write_bytes(json_bytes(value));os.replace(t,path)
def status(phase,**extra):save(CTL/"status.json",{"phase":phase,"time":time.time(),**extra})
def proc(pid):
 p=Path("/proc")/str(pid)
 if not (p/"stat").exists():return None
 a=(p/"stat").read_text().rsplit(")",1)[1].split()
 return {"pid":pid,"state":a[0],"start_ticks":int(a[19]),
         "cmd":(p/"cmdline").read_bytes().replace(b"\0",b" ").decode()}
def wait_exit(value):
 while True:
  now=proc(value["pid"])
  if now is None or now["state"]=="Z":return
  require(now["start_ticks"]==value["start_ticks"],"PID was reused")
  time.sleep(3)
def env():
 return {**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":str(SOURCE/"src"),
  "MPLCONFIGDIR":str(CTL/"matplotlib"),"XDG_CACHE_HOME":str(CTL/"cache"),
  "CUDA_CACHE_PATH":str(CTL/"cuda_cache")}
def child(label,args,expected=0):
 status(label)
 with (CTL/(label+".log")).open("x") as out:
  p=subprocess.Popen([PY,"-u","-B",str(ROOT/"operations/gh_performance_worker.py"),
   "--config",str(CFG),*args],cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT)
  save(CTL/"child.json",{"pid":p.pid,"label":label,"time":time.time()})
  rc=p.wait()
 save(CTL/(label+".exit.json"),{"returncode":rc,"expected":expected},True)
 require(rc==expected,label+" failed; no retry")

def handoff():
 benchmark=read_json(CTL/"benchmark.json")
 require(benchmark["status"]=="GH_PERFORMANCE_BENCHMARK_PASS"
   and benchmark["manifest_sha256"]==sha_file(MANIFEST),"matching real benchmark required")
 require(benchmark["candidate_KL"]["speedup"]>1 and benchmark["teacher_history"]["speedup"]>1,
   "measured acceleration required")
 tests=read_json(CTL/"cpu_tests.json")
 require(tests["status"]=="PASS" and tests["manifest_sha256"]==sha_file(MANIFEST),"matching CPU regression gate required")
 identity=read_json(OLD/"started.json");oldchild=read_json(OLD/"child.json")
 require(oldchild["label"].startswith("train_to_"),"handoff only while native training is active")
 parent=proc(identity["pid"]);native=proc(oldchild["pid"])
 require(parent and native and "gh_attribute_response_recovery.py" in parent["cmd"]
   and "gh_attribute_response.py" in native["cmd"] and "--action train" in native["cmd"],
   "exact owned supervisor/training processes required")
 snapshot=read_json(OLD_TRAIN/"status.json")
 require(snapshot["pid"]==native["pid"],"fresh training identity required")
 requested={"parent":parent,"training":native,"snapshot":snapshot,
   "reason":"user authorized performance deployment at complete boundary",
   "time":time.time(),"manifest_sha256":sha_file(MANIFEST)}
 save(CTL/"handoff_requested.json",requested,True)
 status("WAITING_COMMITTED_ROUND_BOUNDARY",round=snapshot["round"],old_child=native["pid"])
 # 停止旧调度器，避免它把中间停止边界当作40轮结束；原训练器收到正常边界停止请求。
 os.kill(parent["pid"],signal.SIGTERM);wait_exit(parent)
 now=proc(native["pid"])
 if now and now["state"]!="Z":
  require(now["start_ticks"]==native["start_ticks"],"PID reused before stop request")
  os.kill(native["pid"],signal.SIGTERM)
 wait_exit(native)
 latest=read_json(BASE/"train/boundaries/latest.json");n=latest["round"]
 require(n>=snapshot["cursor"]["round"]+1 and n<=60,"new complete boundary required")
 boundary=ArtifactStore(BASE/"train/boundaries",new_base=BASE).verify(f"round_{n:04d}")
 checked(FileRef(**latest["manifest"]))
 require(latest["manifest"]["path"]==str(BASE/f"train/boundaries/round_{n:04d}/manifest.json")
   and boundary["provenance"]["state_sha"]==latest["state_sha"],"boundary state/manifest binding")
 ArtifactStore(BASE/"train",new_base=BASE).verify("generated" if n==60 else "stopped_"+str(n))
 require(not list((BASE/"train").glob("FAIL*")),"real native failure retained")
 require(not any(int(p.name.split("_")[-1])>n for p in (BASE/"train/boundaries").glob("claim_*")),
         "later claim blocks restore")
 done={"round":n,"latest":latest,"old_supervisor":parent,"old_training":native,
  "manifest_sha256":sha_file(MANIFEST),"time":time.time(),"no_uncommitted_round_replayed":True}
 save(CTL/"handoff_completed.json",done,True)
 return n

def run():
 CTL.mkdir(exist_ok=True)
 with (CTL/"supervisor.lock").open("a") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  verify_version()
  require(not (CTL/"started.json").exists(),"deployment consumed; no retry")
  save(CTL/"started.json",{"pid":os.getpid(),"ppid":os.getppid(),"pgid":os.getpgid(0),
   "sid":os.getsid(0),"time":time.time(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
   "manifest_sha256":sha_file(MANIFEST),"configuration_sha256":sha_file(CFG),
   "benchmark_sha256":sha_file(CTL/"benchmark.json"),"cpu_tests_sha256":sha_file(CTL/"cpu_tests.json")},True)
  child("source_preflight",["--action","preflight","--scope","train"])
  current=handoff()
  for n in (40,50,60):
   if current>n:continue
   if current<n:
    args=["--action","train","--allow-gpu","--resume"]
    if n<60:args+=["--stop-after-round",str(n)]
    child("train_to_"+str(n),args,3 if n<60 else 0)
    current=n
   # 边界恰好已完成的节点也保留完整审计和正式评测。
   if not (BASE/f"train_cpu_audit/round_{n:02d}/completed").exists():
    child("audit_train_"+str(n),["--action","audit","--scope","train","--round",str(n)])
   else:
    require(read_json(BASE/f"train_cpu_audit/round_{n:02d}/completed/report.json")["status"]=="GH_AR_TRAIN_CPU_AUDIT_PASS","existing audit FAIL")
   test=BASE/f"test/round_{n:02d}"
   if not (test/"completed").exists():
    require(not test.exists(),"partial Test must be diagnosed, not replayed")
    child("test_"+str(n),["--action","test","--round",str(n),"--allow-gpu"])
   if not (test/"cpu_audit/completed").exists():
    child("audit_test_"+str(n),["--action","test-audit","--round",str(n)])
   require(read_json(test/"cpu_audit/completed/report.json")["status"]=="GH_AR_JOINT_TEST_CPU_AUDIT_PASS","test audit required")
  child("final_report",["--action","report"])
  report=SOURCE/"docs/framework/results/gh_attribute_response/PERFORMANCE_EXECUTION.json"
  save(report,{"runtime":read_json(MANIFEST),"benchmark":read_json(CTL/"benchmark.json"),
   "handoff":read_json(CTL/"handoff_completed.json"),"not_a_new_training_algorithm":True})
  files=["docs/PROJECT_STATE.md","docs/framework/results/gh_attribute_response"]
  require(not subprocess.check_output(["git","diff","--cached","--name-only"],cwd=SOURCE,text=True).strip(),"foreign staged files")
  for cmd in (["git","add","--",*files],["git","diff","--cached","--check"],
    ["git","-c","user.name=w-harda","-c","user.email=wharda@163.com","commit","-m","docs: 发布工程加速续训后的六节点Joint结果"],
    ["git","push","origin","codex/gh-attribute-response"]):
   subprocess.run(cmd,cwd=SOURCE,check=True)
  head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=SOURCE,text=True).strip()
  remote=subprocess.check_output(["git","ls-remote","origin","refs/heads/codex/gh-attribute-response"],cwd=SOURCE,text=True).split()[0]
  require(head==remote,"published SHA mismatch")
  status("SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED",commit=head)

def show():
 for name in ("status.json","started.json","handoff_completed.json","child.json"):
  p=CTL/name;print(name,read_json(p) if p.exists() else "PENDING")
 p=OLD_TRAIN/"status.json"
 if p.exists():
  t=read_json(p);print("训练快照",{k:t.get(k) for k in ("phase","round","visited","total","cursor","pid")})
 p=CTL/"child.json"
 if p.exists():
  c=read_json(p);alive=proc(c["pid"]);print("child_alive",bool(alive and alive["state"]!="Z"))
  log=CTL/(c["label"]+".log")
  if log.exists():
   with log.open("rb") as f:f.seek(max(0,log.stat().st_size-8000));print("\n".join(f.read().decode(errors="replace").splitlines()[-6:]))
 for n in (10,20,30,40,50,60):
  p=BASE/f"test/round_{n:02d}/cpu_audit/completed/report.json"
  if p.exists():
   v=read_json(p);print(n,v["status"],v["metrics"])

if __name__=="__main__":
 action=sys.argv[1] if len(sys.argv)>1 else "status"
 if action=="status":show()
 elif action=="launch":
  CTL.mkdir(exist_ok=True);verify_version()
  require(not (CTL/"started.json").exists(),"deployment already consumed")
  with (CTL/"supervisor.log").open("x") as out:
   p=subprocess.Popen([PY,"-u","-B",str(Path(__file__).resolve()),"run"],cwd=ROOT,
    env=env(),stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
  save(CTL/"launch.json",{"pid":p.pid,"time":time.time()},True);print("LAUNCHED",p.pid)
 elif action=="run":
  try:run()
  except BaseException as e:status("FAIL_STOPPED_NO_RETRY",error=repr(e));raise
 else:raise ValueError("action must be status/launch/run")
