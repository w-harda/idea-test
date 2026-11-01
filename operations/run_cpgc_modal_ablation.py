#!/usr/bin/env python3
"""C-PGC CUHK消融一次性串行执行；默认等待SEP全链释放锁，不干预既有进程。"""
import argparse,fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROL=Path("/home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_v1_20261101")
CONFIG=ROOT/"configs/framework/cpgc_cuhk_modal_ablation.server.json"
PY="/home/lzf/ldx/envs/tbps-image/bin/python"
def state(phase,**values):
 row={"time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"phase":phase,**values}
 temp=CONTROL/"status.tmp";temp.write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n");temp.replace(CONTROL/"status.json")
 with (CONTROL/"status.log").open("a") as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 print(json.dumps(row,ensure_ascii=False),flush=True)
def gpu_processes():
 p=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=20)
 if p.returncode:raise RuntimeError("GPU process snapshot failed: "+p.stderr)
 return p.stdout.strip()
def task(action,victim=None):
 args=[PY,"-u","-B","-m","tbps.baselines.cpgc_sep.modal_ablation","--config",str(CONFIG),"--action",action,"--preflight",str(CONTROL/"preflight.json")]
 if victim:args+=["--victim",victim]
 if action=="evaluate":args+=["--allow-gpu-evaluation"]
 name=action+("-"+victim if victim else "");state(action.upper(),victim=victim)
 with (CONTROL/(name+".log")).open("x") as log:
  p=subprocess.Popen(args,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"PYTHONDONTWRITEBYTECODE":"1"},stdout=log,stderr=subprocess.STDOUT)
  (CONTROL/(name+".pid")).write_text(str(p.pid));code=p.wait()
 (CONTROL/(name+".exit")).write_text(str(code))
 if code:raise RuntimeError(name+" exit="+str(code)+"; preserve and stop")
def main():
 p=argparse.ArgumentParser();p.add_argument("--allow-ablation",action="store_true")
 a=p.parse_args()
 if not a.allow_ablation:p.error("explicit ablation authorization required")
 CONTROL.mkdir(parents=True,exist_ok=True)
 own=(CONTROL/"process.lock").open("a");fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (CONTROL/"started").exists():raise RuntimeError("started; never restart same attempt")
 (CONTROL/"started").write_text(str(os.getpid()))
 for key,val in (("pid",os.getpid()),("ppid",os.getppid()),("pgid",os.getpgid(0)),("sid",os.getsid(0))):
  (CONTROL/("supervisor."+key)).write_text(str(val))
 try:
  # 真实来源CPU预检无CUDA；回执不可覆盖。
  if not (CONTROL/"preflight.json").exists():task("preflight")
  else:
   import hashlib
   receipt=json.loads((CONTROL/"preflight.json").read_text())
   assert receipt["status"]=="PASS" and receipt["config"]["sha256"]==hashlib.sha256(CONFIG.read_bytes()).hexdigest()
  # 与SEP使用相同互斥锁，只读等待，不发送信号；随后等其他compute进程退出。
  global_lock=Path("/home/lzf/ldx/tmp/cpgc_sep_v2_resume.lock").open("a")
  while True:
   try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:
    state("WAITING_SEP",GPU_processes=gpu_processes());time.sleep(180)
  while gpu_processes():
   state("WAITING_GPU_EXCLUSIVE",GPU_processes=gpu_processes());time.sleep(180)
  state("GPU_READY",GPU_processes=gpu_processes())
  for victim in ("irra","rasa"):task("evaluate",victim);task("audit",victim)
  subprocess.run([PY,"-B",str(ROOT/"operations/cpgc_modal_report.py"),"--config",str(CONFIG),"--publish"],cwd=ROOT,check=True)
  state("SUCCESS",marker="CPGC_CUHK_TWO_VICTIM_MODAL_ABLATION_AND_CPU_AUDIT_PASS")
 except BaseException as e:
  (CONTROL/"FAIL.json").write_text(json.dumps({"error":repr(e),"traceback":traceback.format_exc()},ensure_ascii=False,indent=2))
  state("FAIL",error=repr(e));raise
if __name__=="__main__":main()
