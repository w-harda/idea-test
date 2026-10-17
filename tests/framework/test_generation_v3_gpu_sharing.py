"""GPU共享仅CPU Fixture；不查询真实GPU、不加载真实权重、不初始化CUDA。"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock,patch
import hashlib
import json
import os
import time
import pytest
import torch
from tbps.artifacts import ArtifactError
from tbps.engines.replay_control import ResourceLimit
from tbps.materialization import gpu_safety as safety
from tbps.materialization.full_gpu import control
import test_stage4d2r_startup as startup

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/framework/stage4d2r2_generation.server.json"

@pytest.fixture
def world(tmp_path):
    c=json.loads(CONFIG.read_text())
    proc=tmp_path/"proc"
    boot=proc/"sys/kernel/random/boot_id";boot.parent.mkdir(parents=True)
    boot.write_text(c["gpu_sharing"]["boot_id"]+"\n")
    for key in ("process","parent"):
        expected=c["gpu_sharing"][key]
        p=proc/str(expected["pid"]);p.mkdir()
        cmd=(expected["argv0"]+"\0--fixture\0").encode()
        expected["cmdline_sha256"]=hashlib.sha256(cmd).hexdigest()
        (p/"cmdline").write_bytes(cmd)
        (p/"comm").write_text(expected["comm"]+"\n")
        (p/"status").write_text(
            "Uid:\t"+" ".join(map(str,expected["uid"]))+"\n"+
            "Gid:\t"+" ".join(map(str,expected["gid"]))+"\n"+
            "PPid:\t"+str(expected["ppid"])+"\n")
        (p/"cgroup").write_text(expected["cgroup"])
        tail=["S",str(expected["ppid"])]+["0"]*17+[str(expected["start_ticks"])]
        (p/"stat").write_text(str(expected["pid"])+" ("+expected["comm"]+") "+" ".join(tail))
    return c,proc

def query(free=4000,util=0,extra="",ollama=True):
    apps=("2436556, /usr/local/lib/ollama/llama-server, 20178\n" if ollama else "")+extra
    def run(command,**kwargs):
        assert command[0]=="nvidia-smi" and kwargs["timeout"]==10
        value="NVIDIA GeForce RTX 4090, "+str(free)+", 20226, "+str(util)+"\n" if "--query-gpu=" in " ".join(command) else apps
        return Mock(stdout=value)
    return run

def check(world,**kwargs):
    c,proc=world
    native=safety.verify_shared_process
    with patch("tbps.artifacts.checked",return_value={}), \
         patch.object(safety,"verify_shared_process",side_effect=lambda value:native(value,proc_root=proc)), \
         patch.object(safety.subprocess,"run",side_effect=query(**{k:v for k,v in kwargs.items() if k!="runtime"})):
        return safety.gpu_available(c,runtime=kwargs.get("runtime",False))

def test_fixed_configuration_and_original_recovery_unchanged():
    c=control(CONFIG)
    assert c["minimum_free_mib"]==3584 and c["gpu_sharing"]["runtime_minimum_free_mib"]==1024
    assert c["gpu_sharing"]["process"]["pid"]==2436556
    assert c["output"].endswith("/generation_v3") and c["gallery"]==3074
    assert c["recovery"]["sha256"]=="429209225c9a947b1dd254b51addbf41c9ef5a5e8dc0532ea7dfbc3396604bd6"
    assert c["max_seconds"]==7200
    original=ROOT/"src/tbps/materialization/claim_recovery.py"
    assert hashlib.sha256(original.read_bytes()).hexdigest()=="2c97f50c93abcc8152d10eb06fbc05f6d230687edfc6dfe3b6a1a5533639f334"

def test_verified_ollama_at_startup_limit(world):
    status=check(world,free=3584,util=10)
    assert status["free_mib"]==3584 and status["compute_processes"][0]["pid"]==2436556
    assert status["minimum_required_free_mib"]==3584

@pytest.mark.parametrize("field",["uid","gid","comm","ppid","start_ticks","cmdline_sha256","cgroup","argv0"])
@pytest.mark.parametrize("key",["process","parent"])
def test_every_identity_field_mismatch_rejected(world,field,key):
    c,_=world
    # Actual proc Fixture stays fixed; mutate the expected identity.
    value=c["gpu_sharing"][key][field]
    c["gpu_sharing"][key][field]=(value+1 if type(value) is int else
        [998]*4 if type(value) is list else "changed")
    with pytest.raises(ArtifactError):check(world)

def test_boot_or_pid_reuse_rejected(world):
    c,proc=world
    p=proc/str(c["gpu_sharing"]["process"]["pid"])/"stat"
    p.write_text(p.read_text().replace(str(c["gpu_sharing"]["process"]["start_ticks"]), "999999"))
    with pytest.raises(ArtifactError,match="identity changed"):check(world)

def test_different_boot_rejected(world):
    c,proc=world
    (proc/"sys/kernel/random/boot_id").write_text("different")
    with pytest.raises(ArtifactError,match="boot identity"):check(world)

@pytest.mark.parametrize("target",["process","parent"])
def test_exited_or_unreadable_process_rejected(world,target):
    c,proc=world
    (proc/str(c["gpu_sharing"][target]["pid"])/"cmdline").unlink()
    with pytest.raises(ArtifactError,match="cannot verify"):check(world)

def test_executable_SHA_mismatch_rejected(world):
    c,proc=world
    with patch("tbps.artifacts.checked",side_effect=ArtifactError("SHA mismatch")):
        with pytest.raises(ArtifactError,match="SHA mismatch"):
            safety.verify_shared_process(c,proc_root=proc)

@pytest.mark.parametrize("extra",[
    "888, /usr/local/lib/ollama/llama-server, 300\n",
    "888, python, 100\n",
    str(os.getpid())+", python, 100\n"])
def test_other_process_rejected_at_startup(world,extra):
    with pytest.raises(ArtifactError,match="unknown GPU"):check(world,extra=extra)

def test_only_self_added_at_runtime(world):
    status=check(world,runtime=True,free=1024,util=99,
        extra=str(os.getpid())+", python, 512\n")
    assert {p["pid"] for p in status["compute_processes"]}=={2436556,os.getpid()}
    assert status["minimum_required_free_mib"]==1024

def test_runtime_unknown_process_stops(world):
    with pytest.raises(ArtifactError,match="unknown GPU"):
        check(world,runtime=True,extra=str(os.getpid())+", python, 512\n888, python, 100\n")

@pytest.mark.parametrize("runtime,free",[(False,3583),(True,1023)])
def test_free_memory_safety_limits(world,runtime,free):
    with pytest.raises(ArtifactError,match="free memory below"):check(world,runtime=runtime,free=free)

def test_startup_utilization_limit(world):
    with pytest.raises(ArtifactError,match="startup utilization"):check(world,util=11)

def test_no_shared_ollama_or_wrong_executable_rejected(world):
    with pytest.raises(ArtifactError,match="absent"):check(world,ollama=False)
    with pytest.raises(ArtifactError,match="executable identity"):
        check(world,ollama=False,extra="2436556, python, 20178\n")

def test_fixed_sharing_policy_cannot_expand_PID_or_shrink_reserve(world):
    c,_=world
    for field,value in [("runtime_minimum_free_mib",512),("runtime_minimum_free_mib",True)]:
        changed=deepcopy(c);changed["gpu_sharing"][field]=value
        with pytest.raises(ArtifactError):safety.sharing_policy(changed)
    c["gpu_sharing"]["process"]["pid"]=999
    with pytest.raises(ArtifactError):safety.sharing_policy(c)

def test_identity_rechecked_after_nvidia_snapshot(world):
    c,_=world
    with patch.object(safety,"verify_shared_process",side_effect=["a","b"]), \
         patch.object(safety.subprocess,"run",side_effect=query()):
        with pytest.raises(ArtifactError,match="during resource query"):safety.gpu_available(c)

def test_external_query_error_and_timeout_are_not_retried(world):
    c,_=world
    with patch.object(safety,"verify_shared_process",return_value="verified"), \
         patch.object(safety.subprocess,"run",side_effect=TimeoutError("timeout")) as run:
        with pytest.raises(TimeoutError):safety.gpu_available(c)
        assert run.call_count==1

def test_watchdog_samples_each_second_and_stops_on_error(tmp_path):
    c={"max_seconds":7200,"system_ram_reserve_bytes":1,"runtime_disk_reserve_bytes":1}
    ticks=[0.]
    poll=Mock(side_effect=[{"free_mib":2000},{"free_mib":1500},ArtifactError("unknown GPU")])
    guard=safety.FullGuard(c,tmp_path,memory=lambda:{"MemAvailable":2**40},
        sample=lambda:128,clock=lambda:ticks[0],gpu_check=poll)
    guard.check();ticks[0]=.9;guard.check()
    assert poll.call_count==1
    ticks[0]=1.;guard.check()
    assert guard.snapshot()["minimum_gpu_free_mib"]==1500
    ticks[0]=2.
    with pytest.raises(ResourceLimit,match="unknown GPU"):guard.check()
    assert poll.call_count==3
    ticks[0]=4.
    with pytest.raises(ResourceLimit):guard.check()
    assert poll.call_count==3 # 一旦拒绝，不再次查询或尝试继续。

def test_watchdog_IO_failure_is_fail_closed(tmp_path):
    c={"max_seconds":7200,"system_ram_reserve_bytes":1,"runtime_disk_reserve_bytes":1}
    guard=safety.FullGuard(c,tmp_path,memory=lambda:{"MemAvailable":2**40},sample=lambda:128,
        gpu_check=Mock(side_effect=OSError("cannot inspect GPU")))
    with pytest.raises(ResourceLimit,match="cannot inspect GPU"):guard.check()

def test_real_child_control_flow_uses_runtime_guard_before_G():
    flow=startup.ChildFlowTests();flow.setUp()
    try:
        flow.c["gpu_sharing"]={"fixture":True} # Production control already separately verified.
        def runtime_check():
            flow.events.append("shared_GPU_checked")
            return {"free_mib":2000,"compute_processes":[{"pid":2436556}]}
        flow.runtime_gpu_check=runtime_check
        code,report=flow.run_child()
        assert code==1 and report["error"]=="exact 3074 generation / one restoration"
        assert flow.events.index("shared_GPU_checked")<flow.events.index("G_load_entry")
        assert report["actual_G_forwards"]["completed"]==2
        assert report["counts"]["G_loads"]==1
        assert report["resources"]["minimum_gpu_free_mib"]==2000
        assert report["model_before"]==report["model_after"]
    finally:flow.tearDown()

def test_real_child_runtime_refusal_prevents_G_load_and_forward():
    flow=startup.ChildFlowTests();flow.setUp()
    try:
        flow.c["gpu_sharing"]={"fixture":True}
        def runtime_check():raise ArtifactError("unknown GPU compute process")
        flow.runtime_gpu_check=runtime_check
        code,report=flow.run_child()
        assert code==1 and "unknown GPU" in report["error"]
        assert report["counts"]["G_loads"]==0
        assert report["actual_G_forwards"]["started"]==0
        assert "G_load_entry" not in flow.events
    finally:flow.tearDown()

def test_CPU_resource_probe_has_no_Torch_GPU_side_effects(world):
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda,"init",side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")), \
         patch.object(torch,"load",side_effect=AssertionError("weights forbidden")), \
         patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("forward forbidden")):
        assert check(world)["free_mib"]==4000
    assert not torch.cuda.is_initialized()
