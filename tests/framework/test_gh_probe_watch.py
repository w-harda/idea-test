"""全部为CPU资源/进程Fixture，不加载Torch或真实模型。"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/tbps/operations/gh_probe_watch.py"
spec = importlib.util.spec_from_file_location("watch", SCRIPT)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)

def high(free=12000, util=90):
    return dict(eligible=free >= 10240, free_mib=free, utilization=util,
                checks={"gpu":free >= 10240}, processes=["other unrelated process"])

class Fake:
    def __init__(self, resources, *, fail=False):
        self.resources=iter(resources);self.events=[];self.consumed=False
        self.cancelled=False;self.fail=fail;self.states=[]
    def stopped(self):return self.cancelled
    def guard(self):
        if self.consumed:raise RuntimeError("consumed")
        self.events.append("guard")
    def resource(self):self.events.append("resource");return next(self.resources)
    def status(self,s,**k):self.states.append((s,k))
    def frozen(self):self.events.append("frozen")
    def preflight(self):self.events.append("preflight")
    def consume(self):
        self.guard();self.consumed=True;self.events.append("consume")
    def launch(self):
        self.events.append("launch")
        if self.fail:raise RuntimeError("native failure")

@pytest.mark.parametrize("free",[0,6385,10239])
def test_insufficient_never_preflight_or_launch(free):
    e=Fake([high(free)]);p=w.Poller(e)
    assert not p.tick() and p.streak==0
    assert "preflight" not in e.events and "launch" not in e.events

@pytest.mark.parametrize("util",[0,74,90,100])
def test_two_consecutive_only_then_native_launch(util):
    e=Fake([high(util=util)]*3);p=w.Poller(e)
    assert not p.tick()
    assert "preflight" not in e.events
    assert p.tick()
    assert e.events.index("preflight")<e.events.index("consume")<e.events.index("launch")
    assert e.events.count("launch")==1 and e.states[-1][0]=="PROBE_PASS"

def test_resource_fall_returns_to_waiting_and_resets_count():
    e=Fake([high(),high(),high(6000),high(),high(),high()])
    p=w.Poller(e)
    assert not p.tick()
    assert not p.tick() and p.streak==0 and not e.consumed
    assert not p.tick()
    assert p.tick()
    assert e.events.count("preflight")==2 and e.events.count("launch")==1

def test_nonconsecutive_checks_do_not_launch():
    e=Fake([high(),high(6000),high()])
    p=w.Poller(e)
    assert not p.tick() and not p.tick() and not p.tick()
    assert "preflight" not in e.events

def test_probe_failure_is_not_retried():
    e=Fake([high()]*3,fail=True);p=w.Poller(e)
    assert not p.tick()
    with pytest.raises(RuntimeError,match="native failure"):p.tick()
    with pytest.raises(RuntimeError,match="consumed"):p.tick()
    assert e.events.count("launch")==1 and e.consumed

def test_source_failure_prevents_consumption():
    e=Fake([high(),high()])
    def reject():raise RuntimeError("source mismatch")
    e.frozen=reject;p=w.Poller(e);assert not p.tick()
    with pytest.raises(RuntimeError,match="source mismatch"):p.tick()
    assert not e.consumed and "launch" not in e.events

def test_cancel_during_preflight_no_launch():
    e=Fake([high(),high()])
    def cancel():e.cancelled=True
    e.preflight=cancel;p=w.Poller(e)
    assert not p.tick() and p.tick() and not e.consumed

def test_duplicate_guard_before_resource():
    e=Fake([]);e.consumed=True
    with pytest.raises(RuntimeError):w.Poller(e).tick()
    assert not e.events

def test_lock_actual_process(tmp_path):
    lease=w.Lease(tmp_path/"lock")
    code="import runpy; m=runpy.run_path("+repr(str(SCRIPT))+"); m['Lease']("+repr(str(tmp_path/"lock"))+")"
    try:
        p=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True)
        assert p.returncode!=0 and "another watcher" in p.stderr
    finally:lease.close()
    second=w.Lease(tmp_path/"lock");second.close()

@pytest.fixture
def env(tmp_path,monkeypatch):
    root=tmp_path/"frozen";(root/"configs/framework").mkdir(parents=True)
    config={"probe_output":str(tmp_path/"probe"),"prototype":{
        "resource_probe_directory":str(tmp_path),"proposed_gpu_limits":{
            "minimum_ram_bytes":1,"minimum_disk_bytes":1,"minimum_free_inodes":1}}}
    (root/"configs/framework/gh_bidirectional_backend.server.json").write_text(json.dumps(config))
    control=tmp_path/"control";control.mkdir()
    monkeypatch.setattr(w,"FROZEN",root)
    monkeypatch.setattr(w,"other_probe_processes",lambda:[])
    return w.Environment(control,{},0)

@pytest.mark.parametrize("name",["claim","pending","FAIL","completed"])
def test_any_existing_attempt_rejected(env,name):
    env.probe.mkdir();(env.probe/name).touch()
    with pytest.raises(RuntimeError,match="already exists"):env.guard()

def test_durable_marker_before_child_and_after_crash(env):
    env.consume()
    assert (env.control/"probe.started.json").exists() and not env.probe.exists()
    with pytest.raises(RuntimeError,match="consumed"):env.guard()
    with pytest.raises(FileExistsError):
        w.write(env.control/"probe.started.json",{},exclusive=True)

def test_other_probe_process_rejected(env,monkeypatch):
    monkeypatch.setattr(w,"other_probe_processes",lambda:[999])
    with pytest.raises(RuntimeError,match="another native probe"):env.guard()

def test_cancel_cannot_affect_running_probe(env):
    env.consume()
    with pytest.raises(RuntimeError,match="only allowed while waiting"):w.cancel(env.control)
    assert not (env.control/"cancel.request.json").exists()

def test_safe_waiting_cancel_prevents_consume(env):
    env.status("WATCHING")
    w.cancel(env.control)
    assert env.stopped()
    with pytest.raises(RuntimeError,match="cancelled"):env.consume()
    assert not (env.control/"probe.started.json").exists()

def test_exact_native_command_and_environment(env):
    args=env.argv("--action","probe","--allow-gpu-probe")
    assert args[-3:]==["--action","probe","--allow-gpu-probe"]
    assert "smoke" not in args and "training" not in args
    assert env.env()["CUDA_VISIBLE_DEVICES"]==""
    assert env.env(True)["CUDA_VISIBLE_DEVICES"]=="0"

@pytest.mark.parametrize("component",["ram","disk","inodes"])
def test_other_resources_remain_hard_gates(env,monkeypatch,component):
    mapping={"ram":"minimum_ram_bytes","disk":"minimum_disk_bytes","inodes":"minimum_free_inodes"}
    env.c["prototype"]["proposed_gpu_limits"][mapping[component]]=2**80
    monkeypatch.setattr(w,"gpu_snapshot",lambda:high())
    assert not env.resource()["eligible"]

def test_gpu_state_read_error_rejects(env,monkeypatch):
    def fail():raise subprocess.CalledProcessError(1,"nvidia-smi")
    monkeypatch.setattr(w,"gpu_snapshot",fail)
    with pytest.raises(subprocess.CalledProcessError):env.resource()

def test_bad_gpu_query_rejected(monkeypatch):
    monkeypatch.setattr(w,"command",lambda args:"bad")
    with pytest.raises(RuntimeError,match="malformed GPU"):w.gpu_snapshot()

def test_gpu_processes_are_only_observed(monkeypatch):
    calls=[]
    def cmd(args):
        calls.append(args)
        if "--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu" in args:
            return "u, RTX 4090, 18433, 24564, 74"
        return "u, 123, /python, 5000\nother, 456, /other, 2000"
    monkeypatch.setattr(w,"command",cmd)
    r=w.gpu_snapshot()
    assert r["free_mib"]==18433 and r["utilization"]==74
    assert len(r["processes"])==1
    assert all(x[0]=="nvidia-smi" for x in calls)

def test_cpu_preflight_failure_stops_before_probe(env,monkeypatch):
    class Done:returncode=2
    monkeypatch.setattr(w.subprocess,"run",lambda *a,**k:Done())
    with pytest.raises(RuntimeError,match="CPU Preflight failed"):env.preflight()
    assert not (env.control/"probe.started.json").exists()

def test_real_launch_failure_persists_exit_no_retry(env,monkeypatch):
    class Child:
        pid=99999
        def wait(self):return 1
    monkeypatch.setattr(w.subprocess,"Popen",lambda *a,**k:Child())
    env.consume()
    with pytest.raises(RuntimeError,match="native probe failed"):env.launch()
    assert w.read(env.control/"probe.exit.json")["exit_code"]==1
    with pytest.raises(RuntimeError,match="consumed"):env.guard()

def test_history_loop_and_tamper_rejection(env,monkeypatch,tmp_path):
    old=tmp_path/"old";old.write_text("keep");s=old.stat()
    snap=tmp_path/"snapshot";snap.write_text(json.dumps({str(old):[s.st_size,s.st_mtime_ns,s.st_ino]}))
    rel=w.FROZEN/"docs/framework/results/gh_bidirectional_backend/SOURCE_EVIDENCE.json"
    rel.parent.mkdir(parents=True)
    rel.write_text(json.dumps({"history_protection":{"snapshot":{"path":str(snap),"sha256":w.sha(snap)}}}))
    env.history("before")
    assert w.read(env.control/"before.json")["files"]==1
    old.write_text("tampered")
    with pytest.raises(RuntimeError,match="metadata changed"):env.history("after")
    assert len(w.read(env.control/"after.json")["mismatches"])==1

def test_atomic_status_and_exclusive_immutability(tmp_path):
    p=tmp_path/"state"
    w.write(p,{"v":1});w.write(p,{"v":2})
    assert w.read(p)=={"v":2} and not p.with_name("state.tmp").exists()
    q=tmp_path/"marker";w.write(q,{},exclusive=True)
    with pytest.raises(FileExistsError):w.write(q,{},exclusive=True)

def test_script_import_has_no_torch():
    code="import runpy,sys; runpy.run_path("+repr(str(SCRIPT))+"); assert 'torch' not in sys.modules"
    p=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True)
    assert p.returncode==0,p.stderr

def test_no_gpu_process_intervention_or_new_algorithm():
    text=SCRIPT.read_text()
    assert "os.kill(" not in text and "killall" not in text and "pkill" not in text
    import ast
    tree=ast.parse(text)
    assert not any(isinstance(n,ast.Import) and any(x.name.startswith("torch") for x in n.names)
                   or isinstance(n,ast.ImportFrom) and (n.module or "").startswith("torch") for n in ast.walk(tree))
    assert w.INTERVAL==180 and w.MIN_GPU==10240

def test_frozen_source_sha_mismatch_rejected(env,monkeypatch):
    monkeypatch.setattr(w,"command",lambda args,**kw:w.COMMIT if "rev-parse" in args else "")
    env.deployment={"config_sha256":"wrong","script_sha256":w.sha(SCRIPT)}
    with pytest.raises(RuntimeError,match="config SHA"):env.frozen()

def test_cpu_preflight_uses_original_cli_and_history(env,monkeypatch,tmp_path):
    old=tmp_path/"history";old.write_text("immutable");s=old.stat()
    snap=tmp_path/"snapshot";snap.write_text(json.dumps({str(old):[s.st_size,s.st_mtime_ns,s.st_ino]}))
    rel=w.FROZEN/"docs/framework/results/gh_bidirectional_backend/SOURCE_EVIDENCE.json"
    rel.parent.mkdir(parents=True)
    rel.write_text(json.dumps({"history_protection":{"snapshot":{"path":str(snap),"sha256":w.sha(snap)}}}))
    calls=[]
    def run(args,**kw):
        calls.append(args)
        kw["stdout"].write("GH_NATIVE_CPU_PREFLIGHT_PASS\n")
        return type("Done",(),{"returncode":0})()
    monkeypatch.setattr(w.subprocess,"run",run)
    monkeypatch.setattr(env,"frozen",lambda:None)
    env.preflight()
    assert calls[0][-3:]==["--action","preflight","--check-parent-states"]
    assert env.env()["CUDA_VISIBLE_DEVICES"]==""
    assert w.read(env.control/"status.json")["history_files"]==1
    assert not (env.control/"probe.started.json").exists()

def test_native_success_receipt_and_no_other_experiment(env,monkeypatch):
    class Child:
        pid=99999
        def wait(self):return 0
    calls=[]
    def popen(args,**kw):
        calls.append(args)
        return Child()
    monkeypatch.setattr(w.subprocess,"Popen",popen)
    def run(args,**kw):
        assert "verify_probe" in args[3]
        return type("Done",(),{"returncode":0})()
    monkeypatch.setattr(w.subprocess,"run",run)
    monkeypatch.setattr(env,"frozen",lambda:None)
    monkeypatch.setattr(env,"history",lambda label:None)
    done=env.probe/"completed";done.mkdir(parents=True)
    (done/"manifest.json").write_text("{}")
    (done/"report.json").write_text(json.dumps({"status":"GH_NATIVE_NUMERIC_PROBE_PASS","source_sha256":"fixture"}))
    env.launch()
    assert len(calls)==1 and calls[0][-3:]==["--action","probe","--allow-gpu-probe"]
    assert w.read(env.control/"probe.pass.json")["CUDA_RNG_restore_evidence"]=="NOT_REPORTED"

def test_status_keeps_check_count_during_cpu_gate(env):
    env.status("WATCHING",check_count=12,consecutive=2)
    env.status("WATCHING",detail="CPU_PREFLIGHT_PASS")
    assert w.read(env.control/"status.json")["check_count"]==12
