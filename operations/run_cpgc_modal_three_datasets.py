#!/usr/bin/env python3
"""独立共享GPU串行调度：RSTP/ICFG四组Full，逐组CPU审计，成功后发布24行。"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[1]
CONTROL=Path("/home/lzf/ldx/tmp/cpgc_modal_three_datasets_v1_20261102")
PLAN=ROOT/"configs/framework/cpgc_modal_three_datasets.server.json"
PY="/home/lzf/ldx/envs/tbps-image/bin/python"
ORDER=(("rstp","irra"),("rstp","rasa"),("icfg","irra"),("icfg","rasa"))
def native_command(case,action,preflight):
 args=[PY,"-u","-B","-m","tbps.baselines.cpgc_sep.modal_ablation","--config",case["config"]["path"],
       "--action",action,"--victim",case["victim"],"--preflight",str(preflight)]
 return args+["--allow-gpu-evaluation"]if action=="evaluate"else args
def state(phase,**more):
 row={"time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"phase":phase,**more}
 p=CONTROL/"status.tmp";p.write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n");p.replace(CONTROL/"status.json")
 with (CONTROL/"status.log").open("a")as f:f.write(json.dumps(row,ensure_ascii=False)+"\n")
 print(json.dumps(row,ensure_ascii=False),flush=True)
def child(args,name,**fields):
 state(fields.pop("phase"),log=str(CONTROL/(name+".log")),**fields)
 with (CONTROL/(name+".log")).open("x")as log:
  p=subprocess.Popen(args,cwd=ROOT,env={**os.environ,"PYTHONPATH":str(ROOT/"src"),"PYTHONDONTWRITEBYTECODE":"1"},stdout=log,stderr=subprocess.STDOUT)
  (CONTROL/(name+".pid")).write_text(str(p.pid));code=p.wait()
 (CONTROL/(name+".exit")).write_text(str(code))
 if code:raise RuntimeError(name+" exit="+str(code)+"; preserve and stop; no retry")
def main():
 p=argparse.ArgumentParser();p.add_argument("--allow-shared-gpu-full",action="store_true");a=p.parse_args()
 if not a.allow_shared_gpu_full:p.error("explicit four Full shared GPU authorization required")
 CONTROL.mkdir(parents=True,exist_ok=True)
 lock=(CONTROL/"process.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (CONTROL/"started").exists():raise RuntimeError("started; never repeat attempt")
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():raise RuntimeError("execution tree dirty")
 if (CONTROL/"cpu_tests.exit").read_text().strip()!="0":raise RuntimeError("CPU tests not PASS")
 plan=json.loads(PLAN.read_text())
 if tuple((x["dataset"],x["victim"])for x in plan["cases"])!=ORDER:raise RuntimeError("fixed case order")
 if Path(plan["output_base"]).exists():raise RuntimeError("new namespace already exists")
 with (CONTROL/"started").open("x")as f:
  json.dump({"pid":os.getpid(),"time":time.time(),"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
   "authorization":"shared GPU; preserve SEP; only four CPGC modality Full"},f,indent=2);f.flush();os.fsync(f.fileno())
 for k,v in (("pid",os.getpid()),("ppid",os.getppid()),("pgid",os.getpgid(0)),("sid",os.getsid(0))):(CONTROL/("supervisor."+k)).write_text(str(v))
 try:
  child([PY,"-u","-B",str(ROOT/"operations/cpgc_modal_three_dataset_report.py"),"--plan",str(PLAN),"--inventory",str(CONTROL/"inventory.json")],
      "inventory",phase="CPU_INVENTORY",completed=0,total=4)
  completed=0
  for case in plan["cases"]:
   ds,v=case["dataset"],case["victim"];name=ds+"_"+v;pre=CONTROL/("preflight_"+name+".json")
   for action,phase in (("preflight","CPU_PREFLIGHT"),("evaluate","RUNNING_FULL"),("audit","RUNNING_CPU_AUDIT")):
    child(native_command(case,action,pre),name+"_"+action,phase=phase,dataset=ds,victim=v,completed=completed,total=4)
   completed+=1;state("CASE_PASS",dataset=ds,victim=v,completed=completed,total=4)
  child([PY,"-u","-B",str(ROOT/"operations/cpgc_modal_three_dataset_report.py"),"--plan",str(PLAN),"--publish"],
      "report",phase="PUBLISHING_REPORT",completed=4,total=4)
  state("SUCCESS",marker="CPGC_RSTP_ICFG_FOUR_FULL_MODAL_ABLATIONS_AND_CPU_AUDIT_PASS",completed=4,total=4)
 except BaseException as e:
  with (CONTROL/"FAIL.json").open("x")as f:json.dump({"error":repr(e),"traceback":traceback.format_exc()},f,ensure_ascii=False,indent=2)
  state("FAIL",error=repr(e));raise
if __name__=="__main__":main()
