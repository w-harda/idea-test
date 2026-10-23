"""CPU差向量统计与Adam分组测试，不初始化CUDA。"""
import importlib.util
from pathlib import Path
import math
import pytest
import torch
p=Path(__file__).parents[2]/"operations/gh_noop_determinism_review.py"
spec=importlib.util.spec_from_file_location("review",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_no_cuda_on_import():
    assert not torch.cuda.is_initialized()
def test_equal_norm_different_vector():
    s=m.differences(torch.tensor([3.,4.]),torch.tensor([0.,5.]))
    assert s["reference_L2"]==s["candidate_L2"]==5
    assert s["L2"]==pytest.approx(math.sqrt(10)) and s["max_abs"]==3
def test_empty_exact():
    assert m.differences({"a":torch.empty(0)},{"a":torch.empty(0)})["exact"]
@pytest.mark.parametrize("b",[torch.ones(2),torch.ones(3,dtype=torch.float64)])
def test_structure_reject(b):
    with pytest.raises(ValueError):m.differences(torch.ones(3),b)
def test_moment_denominator_excludes_step():
    a={"state":{0:{"step":torch.tensor(484081.),"exp_avg":torch.tensor([1e-8]),"exp_avg_sq":torch.tensor([1e-20])}}}
    b={"state":{0:{"step":torch.tensor(484081.),"exp_avg":torch.tensor([2e-8]),"exp_avg_sq":torch.tensor([1e-20])}}}
    s=m.differences(m.moment_state(a,"exp_avg"),m.moment_state(b,"exp_avg"))
    assert s["relative_L2"]==pytest.approx(1.)
    assert m.differences(m.moment_state(a,"step"),m.moment_state(b,"step"))["exact"]
def test_no_gpu_or_training_entry():
    s=p.read_text()
    assert "native_cli --action" not in s and "loaded_backend(" not in s
    assert "torch.load(DIAG/label/" in s and 'map_location="cpu"' in s
