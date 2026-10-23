"""仅CPU：资源隔离、当前生命周期、等待/消费/互斥/不重试。"""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
OPS=Path(__file__).parents[2]/"operations"
sys.path.insert(0,str(OPS))
import gh_cuda_rng_validation as rng
import gh_smoke_checks as checks
import gh_smoke_watch as watch

def test_rng_resource_independent():
    c={"prototype":{"proposed_gpu_limits":{"minimum_free_mib":10240,"minimum_ram_bytes":34359738368,
        "minimum_disk_bytes":26843545600,"minimum_free_inodes":80000},"other":[1,2]}}
    import copy
    before=copy.deepcopy(c)
    p=rng.rng_resource_protocol(c)
    assert p["proposed_gpu_limits"]["minimum_free_mib"]==2048
    assert c==before and c["prototype"]["proposed_gpu_limits"]["minimum_free_mib"]==10240
    for k in ["minimum_ram_bytes","minimum_disk_bytes","minimum_free_inodes"]:
        assert p["proposed_gpu_limits"][k]==c["prototype"]["proposed_gpu_limits"][k]

def test_existing_probe_current_lifecycle(tmp_path,monkeypatch):
    p=tmp_path/"probe_v2";p.mkdir();(p/"completed").mkdir()
    c={"probe_output":str(p),"smoke_output":str(tmp_path/"smoke_v2")}
    from tbps.alternating import native_gate
    calls=[]
    monkeypatch.setattr(native_gate,"verify_probe",lambda value:calls.append(value) or rng.PROBE_MANIFEST)
    assert checks.lifecycle(c)==rng.PROBE_MANIFEST
    assert calls==[c] and p.exists()

@pytest.mark.parametrize("member",["claim","FAIL","completed",".pending-any"])
def test_any_smoke_member_rejects(tmp_path,monkeypatch,member):
    root=tmp_path/"smoke_v2";root.mkdir();(root/member).touch()
    from tbps.alternating import native_gate
    monkeypatch.setattr(native_gate,"verify_probe",lambda c:rng.PROBE_MANIFEST)
    with pytest.raises(AssertionError,match="consumed"):checks.lifecycle({"smoke_output":str(root)})

def test_foreign_probe_receipt_rejected(tmp_path,monkeypatch):
    from tbps.alternating import native_gate
    monkeypatch.setattr(native_gate,"verify_probe",lambda c:"0"*64)
    with pytest.raises(AssertionError):checks.lifecycle({"smoke_output":str(tmp_path/"absent")})

def test_busy_then_two_consecutive_and_fallback():
    a=watch.Admission()
    assert not a.observe(False)
    assert not a.observe(True)
    assert not a.observe(False)
    assert not a.observe(True)
    assert a.observe(True)
    a.fallback()
    assert not a.observe(True)
    assert a.observe(True)

def test_first_eligible_immediate_authorized():
    a=watch.Admission();assert a.observe(True)

def test_consumption_durable_no_overwrite(tmp_path):
    marker=tmp_path/"started.json"
    watch.consume(marker,{"once":1})
    with pytest.raises(FileExistsError):watch.consume(marker,{"once":2})
    assert json.loads(marker.read_text())=={"once":1}

def test_singleton_lock(tmp_path):
    with watch.Lock(tmp_path/"lock"):
        with pytest.raises(BlockingIOError):
            with watch.Lock(tmp_path/"lock"):pass

def setup_monitor(tmp_path,monkeypatch,observations):
    m=watch.Monitor(tmp_path/"ctl")
    c={"prototype":{},"smoke_output":str(tmp_path/"absent")}
    monkeypatch.setattr(watch,"check_sources",lambda:c)
    monkeypatch.setattr(watch,"rng_receipt",lambda c:"rng")
    monkeypatch.setattr(watch,"lifecycle",lambda c:"probe")
    monkeypatch.setattr(m,"verify_tools",lambda:None)
    monkeypatch.setattr(watch.signal,"signal",lambda *a:None)
    # pytest自身导入Torch，但真实监督器的检查仍保持，测试只模拟空闲进程模块视图。
    monkeypatch.setattr(watch,"sys",SimpleNamespace(modules={}))
    seq=iter(observations)
    monkeypatch.setattr(m,"snapshot",lambda:{"resource_eligible":next(seq)})
    monkeypatch.setattr(m,"prelaunch",lambda:{"resources":{},"CUDA_RNG_manifest":"rng"})
    monkeypatch.setattr(m,"sleep",lambda:setattr(m,"stop",True))
    return m

def test_waiting_never_consumes(tmp_path,monkeypatch):
    m=setup_monitor(tmp_path,monkeypatch,[False])
    monkeypatch.setattr(m,"child",lambda *a:pytest.fail("must not start"))
    m.run()
    assert not m.marker.exists()

def test_resource_falls_before_spawn_no_consumption(tmp_path,monkeypatch):
    m=setup_monitor(tmp_path,monkeypatch,[True,False])
    monkeypatch.setattr(m,"child",lambda *a:pytest.fail("resource fell"))
    m.run()
    assert not m.marker.exists()

def test_started_failure_does_not_retry(tmp_path,monkeypatch):
    m=setup_monitor(tmp_path,monkeypatch,[True,True])
    calls=[]
    def fail(*a):calls.append(a);raise RuntimeError("fixture child failed")
    monkeypatch.setattr(m,"child",fail)
    with pytest.raises(RuntimeError):m.run()
    assert len(calls)==1 and m.marker.exists()
    with pytest.raises(AssertionError,match="consumed"):m.run()
    assert len(calls)==1
