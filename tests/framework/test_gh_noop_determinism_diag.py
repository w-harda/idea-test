"""独立诊断工具CPU测试；不得初始化CUDA或接触真实产物。"""
import importlib.util
from pathlib import Path
import copy
import torch
import pytest
from tbps.alternating.state import clone_tree,tree_sha
P=Path(__file__).parents[2]/"operations/gh_noop_determinism_diag.py"
spec=importlib.util.spec_from_file_location("noop_diag",P);diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)

def test_import_does_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
def test_fixed_budget_and_no_full_entry():
    assert diag.RUNS==(("A1_repeat_1","A1"),("A1_repeat_2","A1"),("A1_cross","A1"),("B_cross","B"))
    text=P.read_text()
    assert "Prototype(" not in text and "native_cli --action smoke" not in text
    assert "b.g_update(ROW,condition)" in text
    assert 'condition.payload["K"]==0' in text
    assert "gate.initialize_cuda()" in text
def test_grad_vector_difference_is_not_norm_difference():
    a=torch.tensor([3.,4.]);b=torch.tensor([0.,5.])
    result=diag.compare({"g":a},{"g":b})
    assert float(a.norm())==float(b.norm())==5
    assert result["L2"]==pytest.approx(10**.5)
    assert result["max_abs"]==3 and result["changed_elements"]==2
def test_exact_metric():
    x={"a":torch.tensor([]),"b":[torch.arange(5)]}
    assert diag.compare(x,clone_tree(x))["exact"]
    assert diag.compare(x,clone_tree(x))["L2"]==0
@pytest.mark.parametrize("bad",[torch.ones(3,dtype=torch.float64),torch.ones(2)])
def test_shape_dtype_rejected(bad):
    with pytest.raises(ValueError):diag.compare(torch.ones(3),bad)
def test_first_stage_order():
    assert diag.first_divergence({"forward":{"exact":True},"backward":{"exact":False},"adam":{"exact":False}})=="backward"
    assert diag.first_divergence({"forward":{"exact":True}})=="NONE"
def test_no_authorization_no_io():
    with pytest.raises(ValueError,match="authorization"):diag.execute(False)
def test_metadata_tamper_fixture(tmp_path):
    p=tmp_path/"a";p.write_text("original");before=diag.metadata_tree(tmp_path)
    p.write_text("changed-length")
    assert before!=diag.metadata_tree(tmp_path)
def test_read_only_hooks_preserve_forward_grad_adam():
    # 相同CPU参数、Adam动量和输入，记录器不改变原数学。
    torch.manual_seed(15)
    g=torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.SiLU(),torch.nn.Linear(5,4))
    enc=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.Tanh()).requires_grad_(False).eval()
    opt=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
    image=torch.arange(8,dtype=torch.float32).reshape(2,4)/8
    (enc(g(image)).square().mean()).backward();opt.step()
    state=clone_tree(g.state_dict());adam=clone_tree(opt.state_dict())
    def run(observe):
        g.load_state_dict(state);opt.load_state_dict(clone_tree(adam));opt.zero_grad(set_to_none=True)
        if observe:
            with diag.observations(g,enc) as data:
                feature=enc(g(image));loss=feature.square().mean();loss.backward();opt.step()
            assert torch.equal(data["G_input"],image)
            assert "G_forward_gradient" in data and "projected_RGB_gradient" in data
        else:
            feature=enc(g(image));loss=feature.square().mean();loss.backward();opt.step()
        return clone_tree({"g":g.state_dict(),"adam":opt.state_dict(),"grad":{n:p.grad for n,p in g.named_parameters()}})
    assert tree_sha(run(False))==tree_sha(run(True))
    assert all(not m._forward_hooks for m in g.modules())
    assert not enc._forward_hooks
def test_original_freeze_source_is_not_modified():
    assert diag.CONFIG_SHA=="77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649"
    assert diag.OUTPUT.name=="noop_determinism_diag_v1" and diag.SMOKE.name=="smoke_v2"

def test_diagnostic_lifecycle_derivation_only_changes_output():
    parent={"prototype":{"output_base":"/frozen/consumed/smoke_v2","sources":[{"p":"sha"}],
        "frozen_rules":{"G":"original"},"proposed_gpu_limits":{"minimum_free_mib":10240}}}
    before=copy.deepcopy(parent)
    new=diag.diagnostic_preflight_protocol(parent)
    assert parent==before
    assert {k:v for k,v in new.items() if k!="output_base"}=={k:v for k,v in parent["prototype"].items() if k!="output_base"}
    assert new["output_base"]==str(diag.OUTPUT)
