"""仅CPU检验独立验证脚本，禁止CUDA、模型前向或Optimizer更新。"""
import importlib.util
import io
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from tbps.alternating.state import capture_rng,clone_tree,tree_sha
spec=importlib.util.spec_from_file_location("rng_operation",Path(__file__).parents[2]/"operations/gh_cuda_rng_validation.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def fixture():
    sampler=torch.Generator()
    saved=capture_rng(sampler)
    try:
        random.seed(10);np.random.seed(10);torch.manual_seed(10);sampler.manual_seed(10)
        a=capture_rng(sampler)
        random.seed(20);np.random.seed(20);torch.manual_seed(20);sampler.manual_seed(20)
        b=capture_rng(sampler)
        g=torch.nn.Linear(1,1);h=torch.nn.Linear(1,1)
        return SimpleNamespace(g=g,h=h,sampler=sampler,rng={"G":a,"H":b}),saved
    except:
        from tbps.alternating.state import restore_rng
        restore_rng(saved,sampler)
        raise

def test_full_phase_switch_persist_cpu(tmp_path):
    ctx,caller=fixture()
    before=tree_sha({"G":ctx.g.state_dict(),"H":ctx.h.state_dict()})
    def draw():return {"p":random.random(),"n":np.random.rand(3),"c":torch.rand(3),
        "s":torch.rand(3,generator=ctx.sampler)}
    def persist(value):
        p=tmp_path/"rng.pt";torch.save(value,p)
        return torch.load(p,map_location="cpu",weights_only=False)
    def forbidden(*a,**k):raise AssertionError("GPU/forward/step forbidden")
    from tbps.alternating.parents import no_optimizer_steps
    from tbps.alternating.state import restore_rng
    try:
        with patch.object(torch.cuda,"_lazy_init",forbidden),patch.object(torch.nn.Module,"_call_impl",forbidden),no_optimizer_steps():
            result=module.check_sequences(ctx,draw,persist)
        assert all(result[k] for k in ["same_sequence_exact","G_H_G_continuation_exact",
            "H_G_H_continuation_exact","phase_state_isolation","persist_read_restore_exact"])
        assert before==tree_sha({"G":ctx.g.state_dict(),"H":ctx.h.state_dict()})
    finally:restore_rng(caller,ctx.sampler)

def test_corrupt_persisted_state_rejected():
    ctx,caller=fixture()
    from tbps.alternating.state import restore_rng
    try:
        with pytest.raises(AssertionError):
            module.check_sequences(ctx,lambda:torch.rand(1),lambda v:{"wrong":True})
    finally:restore_rng(caller,ctx.sampler)

def test_cross_phase_random_advance_detected():
    ctx,caller=fixture()
    from tbps.alternating.state import restore_rng
    calls=[0]
    def draw():
        calls[0]+=1
        if calls[0]==9:torch.rand(1)
        return torch.rand(3)
    try:
        with pytest.raises(AssertionError):
            module.check_sequences(ctx,draw,clone_tree)
    finally:restore_rng(caller,ctx.sampler)

def test_real_production_phase_methods_are_used():
    import inspect
    text=inspect.getsource(module.check_sequences)
    assert "NativeBackend.enter_phase_rng" in text
    assert "NativeBackend.leave_phase_rng" in text
    assert "allow_cuda=True" in text

def test_rng_script_does_not_have_neural_or_smoke_entry():
    import ast
    tree=ast.parse(Path(module.__file__).read_text())
    forbidden={"run_gpu","numerical_probe","loaded_backend","g_update","h_update","refresh_bank"}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id in forbidden]
    assert module.CONFIG_SHA=="77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649"
    assert module.PROBE_MANIFEST=="d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764"
