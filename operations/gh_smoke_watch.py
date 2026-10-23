"""单次A1/B资源监督：等待不消费身份，原native_cli启动后永不自动重试。"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OPS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OPS/"operations"))
from gh_cuda_rng_validation import FROZEN_ROOT,CONFIG,CONFIG_SHA,SOURCE_KEY,PROBE_MANIFEST
from gh_smoke_checks import check_sources,rng_receipt,lifecycle
PY="/home/lzf/ldx/envs/tbps-image/bin/python"
def now():return dt.datetime.now().astimezone().isoformat()
def atomic(path,value):
    p=path.with_name(path.name+".tmp")
    with p.open("w") as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(p,path)
def consume(path,value):
    # 必须在全部门禁及独占锁后调用；崩溃也不允许重新启动。
    with path.open("x") as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    fd=os.open(str(path.parent),os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

class Admission:
    def __init__(self):self.first=True;self.streak=0
    def observe(self,eligible):
        immediate=self.first and eligible
        self.first=False
        self.streak=self.streak+1 if eligible else 0
        return immediate or self.streak>=2
    def fallback(self):self.streak=0;self.first=False

class Lock:
    def __init__(self,path):self.f=path.open("a")
    def __enter__(self):
        fcntl.flock(self.f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);return self
    def __exit__(self,*exc):self.f.close()

class Monitor:
    def __init__(self,ctl):
        self.ctl=Path(ctl);self.ctl.mkdir(parents=True,exist_ok=True)
        self.admission=Admission();self.checks=0;self.stop=False
        self.marker=self.ctl/"smoke_started.json"
        self.status_file=self.ctl/"status.json"
        self.source_hashes={str(OPS/"operations"/n):hashlib.sha256((OPS/"operations"/n).read_bytes()).hexdigest()
            for n in ["gh_cuda_rng_validation.py","gh_smoke_checks.py","gh_smoke_watch.py","run_gh_rng_smoke.sh"]}
        self.env=dict(os.environ,PYTHONPATH=str(FROZEN_ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",
            OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",TMPDIR=str(self.ctl),CUDA_VISIBLE_DEVICES="")
        self.current={}
    def event(self,text):
        with (self.ctl/"status.log").open("a") as f:f.write(now()+" | "+text+"\n");f.flush()
    def status(self,phase,**extra):
        self.current.update(extra)
        self.current.update(state=phase,timestamp=now(),PID=os.getpid(),PPID=os.getppid(),
            PGID=os.getpgid(0),SID=os.getsid(0),TTY=os.isatty(0),checks=self.checks,
            Smoke_consumed=self.marker.exists(),interval_seconds=180)
        atomic(self.status_file,self.current);(self.ctl/"phase").write_text(phase+"\n")
    def verify_tools(self):
        for p,h in self.source_hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,"operations changed while supervising"
    def signal(self,*args):
        if self.marker.exists():self.event("STOP_WAITING_REFUSED | Smoke already consumed; keep supervising child");return
        self.stop=True
    def snapshot(self):
        from tbps.alternating.preflight import resource_snapshot
        return resource_snapshot(self.c["prototype"])
    def sleep(self):
        self.status("WATCHING",next_check_at=(dt.datetime.now().astimezone()+dt.timedelta(seconds=180)).isoformat())
        end=time.monotonic()+180
        while not self.stop and time.monotonic()<end:time.sleep(min(1,max(0,end-time.monotonic())))
    def prelaunch(self):
        self.verify_tools()
        self.c=check_sources();rng=rng_receipt(self.c);lifecycle(self.c)
        self.event("CPU_PREFLIGHT | attempt "+str(self.checks))
        p=self.ctl/f"preflight_{self.checks:04d}.log"
        with p.open("x") as f:
            rc=subprocess.run([PY,"-B","-u","-m","tbps.alternating.native_cli","--config",str(CONFIG),
                "--action","preflight","--check-parent-states"],cwd=FROZEN_ROOT,env=self.env,stdout=f,stderr=subprocess.STDOUT).returncode
        assert rc==0 and "GH_NATIVE_CPU_PREFLIGHT_PASS" in p.read_text(),"CPU Preflight failed"
        # ready再次核验历史、来源及即时资源；仅RESOURCE_BUSY可在未消费时继续等待。
        import gh_smoke_checks
        try:ready=gh_smoke_checks.ready()
        except AssertionError as e:
            if str(e)=="RESOURCE_BUSY":self.admission.fallback();return None
            raise
        atomic(self.ctl/f"ready_{self.checks:04d}.json",ready)
        return ready
    def progress(self):
        log=self.ctl/"smoke.log"
        if not log.exists():return {}
        with log.open("rb") as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-8192));tail=f.read().decode("utf-8",errors="replace")
        rows=[s for s in tail.splitlines() if s.startswith("GH_BANK_PROGRESS")]
        return {"latest_bank_progress":rows[-1] if rows else None,
            "committed_rounds":[n for n in ["A1-r1","A1-r2","B-r1","B-r2"]
                if (Path(self.c["smoke_output"])/"rounds"/n/"completed.json").is_file()]}
    def child(self,action,logname,phase):
        self.verify_tools()
        cmd=[PY,"-B","-u","-m","tbps.alternating.native_cli","--config",str(CONFIG),"--action",action]
        if action=="smoke":cmd.append("--allow-gpu-smoke")
        env=dict(self.env,CUDA_VISIBLE_DEVICES="0" if action=="smoke" else "")
        started=time.monotonic()
        with (self.ctl/logname).open("x") as f:
            process=subprocess.Popen(cmd,cwd=FROZEN_ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            atomic(self.ctl/(action+"_process.json"),{"PID":process.pid,"argv":cmd,"started":now()})
            self.status(phase,child_PID=process.pid,next_check_at=None)
            self.event(phase+" | child_PID="+str(process.pid))
            peak=0
            while True:
                try:
                    lines=Path(f"/proc/{process.pid}/status").read_text().splitlines()
                    value=next((int(s.split()[1])*1024 for s in lines if s.startswith("VmHWM:")),0)
                    peak=max(peak,value)
                except (FileNotFoundError,ProcessLookupError):pass
                try:rc=process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.status(phase,child_peak_RSS_bytes=peak,elapsed_seconds=time.monotonic()-started,**self.progress())
                    continue
                break
        (self.ctl/(action+".exit")).write_text(str(rc)+"\n")
        atomic(self.ctl/(action+"_resources.json"),{"wall_seconds":time.monotonic()-started,"sampled_peak_RSS_bytes":peak,"exit":rc})
        assert rc==0,action+" failed exit="+str(rc)+"; no retry"
    def run(self):
        # 再次执行监督器时，不能改写原成功/失败状态或自动恢复。
        with Lock(self.ctl/"monitor.lock"):
            assert not self.marker.exists(),"Smoke launch identity consumed; no retry"
            self.c=check_sources();rng_receipt(self.c);lifecycle(self.c)
            assert "torch" not in sys.modules,"watching must not import Torch/CUDA"
            signal.signal(signal.SIGTERM,self.signal);signal.signal(signal.SIGINT,self.signal)
            atomic(self.ctl/"supervisor_identity.json",{"PID":os.getpid(),"PPID":os.getppid(),
                "PGID":os.getpgid(0),"SID":os.getsid(0),"TTY":os.isatty(0),"started":now(),
                "operations_sha256":self.source_hashes,"source_key":SOURCE_KEY,"config_sha256":CONFIG_SHA})
            for name,value in [("supervisor.pid",os.getpid()),("supervisor.pgid",os.getpgid(0)),("supervisor.sid",os.getsid(0))]:
                (self.ctl/name).write_text(str(value)+"\n")
            self.status("WATCHING");self.event("WATCHING | singleton lock acquired; RNG receipt verified")
            try:
                while not self.stop:
                    self.checks+=1;self.verify_tools();lifecycle(self.c)
                    resource=self.snapshot()
                    self.event("GPU_CHECK | "+json.dumps(resource,ensure_ascii=False,separators=(",",":")))
                    self.status("WATCHING",resources=resource)
                    if self.admission.observe(resource["resource_eligible"]):
                        ready=self.prelaunch()
                        if self.stop:break
                        if ready is not None:
                            self.status("READY",resources=ready["resources"],next_check_at=None)
                            self.event("READY | all gates passed; source and RNG receipts verified")
                            # 最后即时快照；资源回落绝不消费Smoke。
                            final=self.snapshot()
                            if not final["resource_eligible"]:
                                self.admission.fallback();self.status("WATCHING",resources=final);self.sleep();continue
                            lifecycle(self.c)
                            if self.stop:break
                            consume(self.marker,{"timestamp":now(),"source":SOURCE_KEY,"config_sha256":CONFIG_SHA,
                                "probe_manifest":PROBE_MANIFEST,"CUDA_RNG_manifest":ready["CUDA_RNG_manifest"],
                                "resources":final,"operations_sha256":self.source_hashes})
                            self.child("smoke","smoke.log","SMOKE_RUNNING")
                            self.status("SMOKE_PASS",**self.progress());self.event("SMOKE_PASS | begin independent CPU audit")
                            self.child("audit-smoke","cpu_audit.log","RUNNING_CPU_AUDIT")
                            import gh_smoke_checks
                            result=gh_smoke_checks.final()
                            atomic(self.ctl/"final_evidence.json",result)
                            self.status("SMOKE_PASS",CPU_audit="PASS",feedback_effects=result["feedback_effects"])
                            self.event("SUCCESS | GH_TWO_ROUND_SMOKE_AND_CPU_AUDIT")
                            return
                    self.sleep()
                self.status("STOPPED_WAITING",next_check_at=None);self.event("STOPPED_WAITING | no Smoke consumption")
            except BaseException:
                import traceback
                with (self.ctl/"monitor.FAIL").open("x") as f:f.write(traceback.format_exc())
                self.status("SMOKE_FAIL",next_check_at=None)
                self.event("SMOKE_FAIL | stop; no retry")
                raise

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--control",required=True)
    ap.add_argument("--allow-monitor-and-smoke",action="store_true");a=ap.parse_args()
    if not a.allow_monitor_and_smoke:raise SystemExit("explicit monitor/Smoke authorization required")
    Monitor(a.control).run()
