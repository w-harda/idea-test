"""真实loaded_backend设备接线的CPU控制流检查；禁止真实CUDA/权重/前向。"""
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from tbps.artifacts import ArtifactError, ArtifactStore, digest, json_bytes, sha_file
from tbps.alternating import native_backend as native
from tbps.alternating.native_sources import verify_config, source_key
from tbps.alternating.native_gate import verify_probe
from tbps.models.irra import IRRAAdapter
from tbps.models.irra_source import IRRASource, load_official_model

@pytest.fixture
def loader_fixture(monkeypatch):
    assert not torch.cuda.is_initialized()
    events=[]
    def forbidden(*a,**k):raise AssertionError("no CUDA/weights/forward authorized")
    monkeypatch.setattr(torch.cuda,"init",forbidden)
    monkeypatch.setattr(torch.cuda,"_lazy_init",forbidden)
    monkeypatch.setattr(torch,"load",forbidden)
    monkeypatch.setattr(torch.nn.Module,"_call_impl",forbidden)
    # 只模拟门禁已完成标志，不调用CUDA。
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:True)
    source=object();adapter=object()
    monkeypatch.setattr(native,"read_json",lambda p:{"irra":{"fixture":True}})
    def from_dict(value):
        assert value=={"fixture":True};events.append("source")
        return source
    monkeypatch.setattr(IRRASource,"from_dict",from_dict)
    def load(value,*,device,allow_model_load):
        events.append(("loader",device,allow_model_load))
        assert value is source and device=="cuda:0" and allow_model_load is True
        return adapter
    monkeypatch.setattr(IRRAAdapter,"from_source",load)
    @contextmanager
    def session(c,*,reuse):
        assert reuse=={"same_module":True};events.append("legacy_enter")
        try:yield
        finally:events.append("legacy_exit")
    monkeypatch.setattr(native,"native_session",session)
    class Backend:
        def __init__(self,c,b,g,a):
            assert a is adapter;events.append("backend")
        def close(self):events.append("close")
    monkeypatch.setattr(native,"NativeBackend",Backend)
    grant=SimpleNamespace(action="probe",validate=lambda c,a:events.append("grant"))
    c={"sources":{"dataset_config":{"path":"fixture.json"}}}
    return c,SimpleNamespace(modules={"same_module":True}),grant,events

def test_loaded_backend_passes_real_explicit_device_and_closes(loader_fixture):
    c,b,g,events=loader_fixture
    with native.loaded_backend(c,b,g) as backend:
        assert type(backend).__name__=="Backend"
        events.append("body")
    assert events==["grant","source",("loader","cuda:0",True),
                    "legacy_enter","backend","body","close","legacy_exit"]

def test_loaded_backend_loader_failure_precedes_native_session(loader_fixture,monkeypatch):
    c,b,g,events=loader_fixture
    def reject(source,*,device,allow_model_load):
        assert device=="cuda:0";events.append("reject")
        raise ArtifactError("controlled source rejection")
    monkeypatch.setattr(IRRAAdapter,"from_source",reject)
    with pytest.raises(ArtifactError,match="controlled source"):
        with native.loaded_backend(c,b,g):raise AssertionError("unreachable")
    assert events==["grant","source","reject"]

@pytest.mark.parametrize("device",["cuda","cuda:","cuda:x",torch.device("cuda:0")])
def test_official_rejects_invalid_device_before_source_or_weights(device,monkeypatch):
    source=SimpleNamespace(verify=Mock(side_effect=AssertionError("source must not be read")))
    load=Mock(side_effect=AssertionError("no weight load"))
    monkeypatch.setattr(torch,"load",load)
    with pytest.raises(ArtifactError,match="explicit cpu or cuda:index required"):
        load_official_model(source,device=device,allow_model_load=True)
    source.verify.assert_not_called();load.assert_not_called()
    assert not torch.cuda.is_initialized()

def test_official_accepts_cuda0_contract_without_loading(monkeypatch):
    verify=Mock(side_effect=RuntimeError("STOP_BEFORE_MODEL_LOAD"))
    source=SimpleNamespace(verify=verify)
    load=Mock(side_effect=AssertionError("no weight load"))
    monkeypatch.setattr(torch,"load",load)
    with pytest.raises(RuntimeError,match="STOP_BEFORE_MODEL_LOAD"):
        load_official_model(source,device="cuda:0",allow_model_load=True)
    verify.assert_called_once();load.assert_not_called()
    assert not torch.cuda.is_initialized()

def test_v2_refuses_v1_fixture_receipt_even_if_copied(tmp_path):
    base=tmp_path/"tbps-framework-fixture"
    old={"output_base":str(base),"probe_output":str(base/"probe_v1"),
         "smoke_output":str(base/"smoke_v1"),"prototype":{"old":True},
         "probe":{"noop_gradient_atol":1e-6,"max_IRRA_texts":6}}
    result={"status":"GH_NATIVE_NUMERIC_PROBE_PASS","optimizer_steps":0,
            "full_gallery_generations":0,"frozen_sources":True,
            "nonnoop_gradient_max_abs":.1,"noop_gradient_max_abs":0,
            "counts":{"G":4,"IRRA_image":4,"JVP":1,"reader":1,"H":1,"IRRA_text":6},
            "original_text_cache_max_abs":0}
    ArtifactStore(old["probe_output"],new_base=base).publish("completed",
        {"report.json":json_bytes(result)},{"source":source_key(old),"protocol":digest(old["prototype"])})
    new=copy.deepcopy(old);new["probe_output"]=str(base/"probe_v2")
    new["smoke_output"]=str(base/"smoke_v2");new["prototype"]={"v2":True}
    with pytest.raises(ValueError):verify_probe(new)
    # 只复制合成Fixture，证明不能把旧回执冒充新来源。
    shutil.copytree(old["probe_output"],new["probe_output"])
    with pytest.raises(ValueError):verify_probe(new)

@pytest.fixture
def server_config():
    path=os.environ.get("TBPS_NATIVE_TEST_CONFIG")
    if path is None:pytest.skip("设置显式服务器CPU配置才运行来源检查")
    return Path(path),json.loads(Path(path).read_text())

def test_v2_identity_budget_and_complete_condition_sources(server_config):
    path,c=server_config
    old=json.loads(Path(c["sources"]["parent_backend_configuration"]["path"]).read_text())
    assert c["probe"]==old["probe"] and c["smoke"]==old["smoke"]
    assert c["runtime"]==old["runtime"] and c["runtime_pins"]==old["runtime_pins"]
    assert c["prototype"]["frozen_rules"]==old["prototype"]["frozen_rules"]
    assert c["prototype"]["parent_sha256"]==old["prototype"]["parent_sha256"]
    assert c["probe_output"].endswith("/probe_v2") and c["smoke_output"].endswith("/smoke_v2")
    assert c["prototype"]["output_base"]==c["smoke_output"]
    assert c["prototype"]["proposed_gpu_limits"]["minimum_free_mib"]==10240
    root=Path(native.__file__).resolve().parents[3]
    assert set(c["producers"])=={str(p) for p in (root/"src/tbps").rglob("*.py")}
    assert c["prototype"]["new_producers"]==c["producers"]
    expected=digest({"IRRA":c["prototype"]["source_space_sha256"],"reader":c["sources"]["reader"],
        "basis":c["sources"]["basis"],"byte_mapping":c["byte_alignment_policy"],
        "producer_sources":{p:h for p,h in c["producers"].items() if "/alternating/" in p}})
    assert c["condition_space_sha256"]==c["prototype"]["condition_space_sha256"]==expected
    assert sha_file(c["sources"]["parent_backend_configuration"]["path"])==c["sources"]["parent_backend_configuration"]["sha256"]
    assert not Path(c["probe_output"]).exists() and not Path(c["smoke_output"]).exists()
    assert not torch.cuda.is_initialized()

def test_v2_missing_producer_rejected(server_config,tmp_path):
    path,c=server_config;c["producers"].pop(next(iter(c["producers"])))
    p=tmp_path/"missing.json";p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="complete backend producer set"):verify_config(p)

def test_v2_false_condition_identity_rejected(server_config,tmp_path):
    path,c=server_config;c["condition_space_sha256"]="0"*64
    p=tmp_path/"false_condition.json";p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="condition reader/basis/producer identity"):verify_config(p)


@pytest.mark.parametrize("free,eligible",[(10239,False),(10240,True)])
def test_v2_intrinsic_10GiB_gate(server_config,monkeypatch,free,eligible):
    from tbps.alternating import preflight
    path,c=server_config;config=copy.deepcopy(c["prototype"])
    config["proposed_gpu_limits"].update(minimum_ram_bytes=1,minimum_disk_bytes=1,minimum_free_inodes=1)
    def query(args,**kwargs):
        return f"RTX 4090, {free}, 24564, 90" if "--id=0" in args else "other process"
    monkeypatch.setattr(preflight.subprocess,"check_output",query)
    assert preflight.resource_snapshot(config)["resource_eligible"] is eligible
    assert not torch.cuda.is_initialized()
