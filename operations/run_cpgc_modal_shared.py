#!/usr/bin/env python3
"""显式共享GPU授权的独立调度包装；冻结数值源码与配置保持不变。"""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-cuhk-modal-ablation-20261101")
CONTROL=Path("/home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_shared_v1_20261101")
PREVIOUS=Path("/home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_v1_20261101")
CONFIG=ROOT/"configs/framework/cpgc_cuhk_modal_ablation.server.json"
PREFLIGHT=PREVIOUS/"preflight.json"
EXPECTED_COMMIT="7973128ec1fc73ab8e2428dfbdea600a565421a9"
CONFIG_SHA="8d6d63e90fa79828de8e5fd45f6bcf76872bc4ae32f48b386aec0973f31fe8e7"
PREFLIGHT_SHA="a2a94f191e2962e74f087d389ee42219ba9d05dfa2f30424e31fe091ff21e63a"
PY="/home/lzf/ldx/envs/tbps-image/bin/python"
def state(phase,**extra):
 row={"time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"phase":phase,**extra}
 p=CONTROL/"status.tmp";p.write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n");p.replace(CONTROL/"status.json")
 with (CONTROL/"status.log").open("a")as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 print(json.dumps(row,ensure_ascii=False),flush=True)
def child_command(action,victim):
 args=[PY,"-u","-B","-m","tbps.baselines.cpgc_sep.modal_ablation","--config",str(CONFIG),"--action",action,
       "--victim",victim,"--preflight",str(PREFLIGHT)]
 return args+["--allow-gpu-evaluation"] if action=="evaluate" else args
def admission(snapshot):
 # 不因SEP/Ollama存在或GPU利用率/空闲显存的固定数值拒绝共享。
 if snapshot["blocked"]:raise RuntimeError("Host RESOURCE_BUSY: "+json.dumps(snapshot["blocked"]))
 return True
def validate_before_start():
 if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=EXPECTED_COMMIT:raise RuntimeError("execution Commit changed")
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():raise RuntimeError("frozen execution tree dirty")
 if hashlib.sha256(CONFIG.read_bytes()).hexdigest()!=CONFIG_SHA:raise RuntimeError("config changed")
 if hashlib.sha256(PREFLIGHT.read_bytes()).hexdigest()!=PREFLIGHT_SHA:raise RuntimeError("preflight receipt changed")
 receipt=json.loads(PREFLIGHT.read_text())
 if receipt["status"]!="PASS" or receipt["CUDA"]is not False or receipt["config"]["sha256"]!=CONFIG_SHA:raise RuntimeError("CPU preflight invalid")
 from tbps.baselines.cpgc_sep.modal_ablation import settings,unchanged
 from tbps.baselines.cpgc_sep.runtime import resource_snapshot
 import torch
 if torch.cuda.is_initialized():raise RuntimeError("preflight initialized CUDA")
 c,old=settings(CONFIG);unchanged(receipt["protection"])
 if Path(c["output_base"]).exists():raise RuntimeError("native Attempt consumed; no retry")
 snap=resource_snapshot(old);admission(snap)
 return snap
def task(action,victim):
 name=action+"-"+victim
 state("RUNNING_"+victim.upper() if action=="evaluate" else "AUDITING_"+victim.upper(),log=str(CONTROL/(name+".log")))
 with (CONTROL/(name+".log")).open("x")as log:
  p=subprocess.Popen(child_command(action,victim),cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"PYTHONDONTWRITEBYTECODE":"1"},stdout=log,stderr=subprocess.STDOUT)
  (CONTROL/(name+".pid")).write_text(str(p.pid));code=p.wait()
 (CONTROL/(name+".exit")).write_text(str(code))
 if code:raise RuntimeError(name+" exit="+str(code)+"; preserve; never retry")
def main():
 p=argparse.ArgumentParser();p.add_argument("--allow-shared-gpu-ablation",action="store_true");a=p.parse_args()
 if not a.allow_shared_gpu_ablation:p.error("explicit shared GPU authorization required")
 CONTROL.mkdir(parents=True,exist_ok=True)
 lock=(PREVIOUS/"process.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 lock2=(CONTROL/"process.lock").open("a");fcntl.flock(lock2,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (CONTROL/"started").exists():raise RuntimeError("started; no restart")
 snap=validate_before_start()
 with (CONTROL/"started").open("x")as f:
  json.dump({"time":time.time(),"pid":os.getpid(),"authorization":"shared GPU permitted; preserve SEP",
   "execution_commit":EXPECTED_COMMIT,"config_sha256":CONFIG_SHA,"preflight_sha256":PREFLIGHT_SHA,
   "wrapper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"resources":snap},f,indent=2);f.flush();os.fsync(f.fileno())
 for k,v in (("pid",os.getpid()),("ppid",os.getppid()),("pgid",os.getpgid(0)),("sid",os.getsid(0))):(CONTROL/("supervisor."+k)).write_text(str(v))
 try:
  for victim in ("irra","rasa"):task("evaluate",victim);task("audit",victim)
  state("PUBLISHING_REPORT")
  subprocess.run([PY,"-B",str(ROOT/"operations/cpgc_modal_report.py"),"--config",str(CONFIG),"--publish"],cwd=ROOT,check=True)
  state("SUCCESS",marker="CPGC_CUHK_TWO_VICTIM_MODAL_ABLATION_AND_CPU_AUDIT_PASS")
 except BaseException as e:
  with (CONTROL/"FAIL.json").open("x")as f:json.dump({"error":repr(e),"traceback":traceback.format_exc()},f,ensure_ascii=False,indent=2)
  state("FAIL",error=repr(e));raise
if __name__=="__main__":main()
