"""服务器只读CPU接线检查；必须显式传入配置，CUDA/所有模型前向持续禁止。"""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import pytest
import torch
from tbps.alternating.native_sources import verify_config,strict_import_preflight,native_session
from tbps.alternating.parents import restore_cpu_parents
from tbps.alternating.native_gate import ExecutionGrant
from tbps.alternating.native_backend import NativeBackend
from tbps.alternating.state import tree_sha
from tbps.alternating.token_bridge import ByteSpanBridge,token_audit
from tbps.artifacts import digest
from tbps.alternating.native_sources import source_key

@pytest.fixture(scope="module")
def real():
    path=os.environ.get("TBPS_NATIVE_TEST_CONFIG")
    if path is None:
        pytest.skip("显式服务器CPU只读集成检查；设置TBPS_NATIVE_TEST_CONFIG后执行")
    c=verify_config(path)
    assert strict_import_preflight(c)["status"]=="GH_NATIVE_IMPORT_CLOSURE_PASS"
    bundle=restore_cpu_parents(c)
    yield c,bundle
    assert not torch.cuda.is_initialized()

def test_real_full_parent_model_adam_rng(real):
    c,b=real
    assert b.report["G_Adam_steps"]==[484080]
    assert b.report["H_Adam_step_min"]==20124 and b.report["H_Adam_step_max"]==20370
    assert b.report["CPU_RNG_exact"] and b.report["CUDA_RNG_preserved"]
    assert b.report["real_model_forwards"]==b.report["updates"]==0

@pytest.mark.parametrize("caption",["blue shirt","a bluе shirt","blаck pants","a person carrying a bаg"])
def test_real_bpe_byte_bridge_preserves_ids(real,caption):
    c,b=real
    with native_session(c,reuse=b.modules):
        native=b.bindings.get("attributes.irra_token_alignment")
        bridge=ByteSpanBridge(native,b.probe.tokenizer)
        a=bridge.align(caption,b.probe.tokenizer)
        assert [x["token_id"] for x in a["all_content_tokens"]]==b.probe.tokenizer.encode(caption)
        assert a["status"]=="valid"
        report=token_audit(b.probe,caption)
        assert report["eot_index"]==len(b.probe.tokenizer.encode(caption))+1
        for row in a["all_content_tokens"]:
            assert caption[row["start"]:row["end"]]==row["raw"]

def test_real_truncation_is_rejected(real):
    c,b=real
    with pytest.raises(ValueError):token_audit(b.probe,"shirt "*100)
    with native_session(c,reuse=b.modules):
        with pytest.raises(ValueError):
            ByteSpanBridge(b.bindings.get("attributes.irra_token_alignment"),b.probe.tokenizer).align(
                "shirt "*100,b.probe.tokenizer)

def test_real_caption_frontend_matches_locked_qualification(real):
    c,b=real
    import json
    entries=json.loads((Path(c["prototype"]["history_root"])/"qualification/train.json").read_text())["entries"]
    with native_session(c,reuse=b.modules):
        for e in entries:
            if e["identity"]["row_id"] in c["prototype"]["samples"]:
                example=b.frontend(e["caption"])
                from dataclasses import asdict
                assert [asdict(a) for a in example.attributes]==e["attributes"]

def test_real_backend_constructor_and_g_plans_cpu_mock_device(real):
    """使用真实父权重、reader权重及原K计划；仅设备分配重定向CPU，绝不前向。"""
    c,b=real
    class FakeIRRA(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        def encode_image(self,x):raise AssertionError("no model forward authorized")
        def encode_text(self,x):raise AssertionError("no model forward authorized")
    def stop(*a,**k):raise AssertionError("no model forward/CUDA authorized")
    old_to=torch.Tensor.to
    def cpu_to(t,*args,**kwargs):
        args=list(args)
        if args and isinstance(args[0],(str,torch.device)) and str(args[0]).startswith("cuda"):
            args[0]="cpu"
        if str(kwargs.get("device","")).startswith("cuda"):kwargs["device"]="cpu"
        return old_to(t,*args,**kwargs)
    def cpu_factory(fn):
        def call(*args,**kwargs):
            if str(kwargs.get("device","")).startswith("cuda"):kwargs["device"]="cpu"
            return fn(*args,**kwargs)
        return call
    grant=ExecutionGrant("probe",source_key(c),digest(c["prototype"]))
    with native_session(c,reuse=b.modules),ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor,"to",cpu_to))
        for name in ("tensor","zeros","ones","empty","eye","arange","as_tensor"):
            stack.enter_context(patch.object(torch,name,cpu_factory(getattr(torch,name))))
        stack.enter_context(patch.object(torch.nn.Module,"_call_impl",stop))
        stack.enter_context(patch.object(torch.cuda,"init",stop))
        stack.enter_context(patch.object(torch.cuda,"_lazy_init",stop))
        model=FakeIRRA().eval()
        backend=NativeBackend(c,b,grant,SimpleNamespace(_backend=model))
        try:
            assert len(backend.original)==8
            assert sum(x.payload["K"]+1 for x in backend.original.values())==27
            assert all(len(backend.plans[row])==backend.original[row].payload["K"] for row in backend.original)
            backend.assert_frozen_sources()
            assert backend.counts=={"G":0,"IRRA_image":0,"IRRA_text":0,"reader":0,"H":0,"JVP":0}
            assert backend.h_fingerprint()==b.report["H_state"]
        finally:backend.close()
    assert not torch.cuda.is_initialized()


def test_native_span_boundary_remains_strict(real):
    c,b=real
    with native_session(c,reuse=b.modules):
        bridge=ByteSpanBridge(b.bindings.get("attributes.irra_token_alignment"),b.probe.tokenizer)
        alignment=bridge.align("black shirt",b.probe.tokenizer)
        ids,reasons=bridge.span_tokens("black shirt",{"raw":"bla","start":0,"end":3},alignment["all_content_tokens"])
        assert reasons


def test_original_clean_and_G60_bank_readonly_content(real):
    c,b=real
    backend=NativeBackend.__new__(NativeBackend)
    backend.c=c;backend.p=c["prototype"];backend.native=b.bindings
    with native_session(c,reuse=b.modules):
        # 只读特征和旧协议；不恢复IRRA/G/H、不进行前向或评分。
        NativeBackend.load_reward_inputs(backend)
    assert backend.clean.shape==backend.current_features.shape==(2048,512)
    assert torch.isfinite(backend.clean).all() and torch.isfinite(backend.current_features).all()
    assert [s.row_id for s in backend.catalog.train]==c["prototype"]["train_ids"]
    assert not torch.cuda.is_initialized()
