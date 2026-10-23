"""Probe专用显存覆盖；无CUDA、无神经前向。"""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from tbps.alternating.native_sources import resource_protocol
from tbps.alternating import native_cli, native_gate, preflight

def cfg(tmp_path, override=True):
    root=tmp_path/"tbps-framework-probe"
    c={"prototype":{"proposed_gpu_limits":{"minimum_free_mib":10240,
        "minimum_ram_bytes":32*1024**3,"minimum_disk_bytes":25*1024**3,"minimum_free_inodes":80000}},
       "probe_output":str(root/"probe_v2"),"smoke_output":str(root/"smoke_v2"),"output_base":str(root)}
    if override:c["probe_resource_limits"]={"minimum_free_mib":6144}
    return c

@pytest.mark.parametrize("action,minimum",[("probe",6144),("smoke",10240)])
def test_explicit_probe_override_is_scoped_and_nonmutating(tmp_path,action,minimum):
    c=cfg(tmp_path);before=copy.deepcopy(c);p=resource_protocol(c,action)
    assert p["proposed_gpu_limits"]["minimum_free_mib"]==minimum
    assert c==before
    for field in ["minimum_ram_bytes","minimum_disk_bytes","minimum_free_inodes"]:
        assert p["proposed_gpu_limits"][field]==c["prototype"]["proposed_gpu_limits"][field]

@pytest.mark.parametrize("action",["probe","smoke"])
def test_existing_config_without_override_keeps_10GiB(tmp_path,action):
    c=cfg(tmp_path,False)
    assert resource_protocol(c,action)["proposed_gpu_limits"]["minimum_free_mib"]==10240

@pytest.mark.parametrize("invalid",[{"minimum_free_mib":0},{"minimum_free_mib":6143},
    {"minimum_free_mib":6144.0},{"minimum_free_mib":6144,"minimum_ram_bytes":1},{}])
def test_invalid_or_unapproved_overrides_rejected(tmp_path,invalid):
    c=cfg(tmp_path);c["probe_resource_limits"]=invalid
    with pytest.raises(ValueError,match="probe-only"):resource_protocol(c,"probe")

@pytest.mark.parametrize("action",["train","resources"])
def test_no_implicit_training_resource_override(tmp_path,action):
    with pytest.raises(ValueError):resource_protocol(cfg(tmp_path),action)

@pytest.mark.parametrize("action,minimum",[("probe",6144),("smoke",10240)])
def test_actual_gpu_entry_uses_action_resource_protocol_before_cuda(tmp_path,monkeypatch,action,minimum):
    c=cfg(tmp_path);seen=[]
    monkeypatch.setattr(native_cli,"verify_config",lambda path:c)
    monkeypatch.setattr(native_gate,"authorize",lambda *a,**k:object())
    gate=SimpleNamespace(initialize_cuda=lambda:pytest.fail("CUDA cannot initialize"))
    monkeypatch.setattr(native_cli,"prepare_cpu_gate",lambda path:(c,object(),gate,{}))
    def snapshot(p):
        seen.append(p["proposed_gpu_limits"]["minimum_free_mib"])
        return {"resource_eligible":False}
    monkeypatch.setattr(preflight,"resource_snapshot",snapshot)
    with pytest.raises(ValueError,match="RESOURCE_BUSY"):
        native_cli.run_gpu("fixture",action,allow_probe=action=="probe",allow_smoke=action=="smoke")
    assert seen==[minimum]
    assert not Path(c["output_base"]).exists()
