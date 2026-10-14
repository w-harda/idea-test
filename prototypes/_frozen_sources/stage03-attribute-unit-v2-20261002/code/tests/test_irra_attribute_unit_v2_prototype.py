"""512D子空间预算、真实导数与nested support反例。"""
import pytest
import torch
from attributes.irra_attribute_unit_v2_prototype import uniform_unit_gradient,unit_delta


def test_lift_gradient_matches_autograd_and_first_order_for_different_k():
    torch.manual_seed(10)
    h=torch.randn(7,1,512,dtype=torch.float64)
    coeff=torch.randn_like(h)
    for support in ([1],[1,2],[0,2,4,6]):
        w,g,old,new=uniform_unit_gradient(coeff,support)
        v=torch.zeros(512,dtype=torch.float64,requires_grad=True)
        changed=h+w.double()[:,None,None]*v[None,None,:]
        objective=(changed*coeff).sum()
        actual_g,=torch.autograd.grad(objective,v)
        assert torch.allclose(g.double(),actual_g,atol=5e-7,rtol=1e-5)
        assert w.double().norm().item()==pytest.approx(1,rel=1e-7)
        delta=unit_delta(w,g,.01)
        assert delta.double().norm().item()==pytest.approx(.01,rel=2e-7)
        change=(delta.double()*coeff).sum().item()
        assert change==pytest.approx(.01*new,rel=1e-6)
        assert new<=old*(1+1e-6)
        assert torch.allclose(delta[support[0]],delta[support[-1]])


def test_nested_support_has_no_mandatory_order_in_new_norm():
    # shirt单token；black+shirt两token。旧范数必须增加，新范数可下降。
    grad=torch.zeros(3,1,512)
    grad[1,0,0]=1
    grad[2,0,0]=-1
    _,_,old_type,new_type=uniform_unit_gradient(grad,[1])
    _,_,old_color,new_color=uniform_unit_gradient(grad,[1,2])
    assert old_color>old_type and new_color<new_type
    grad[2,0,0]=1
    _,_,old_color,new_color=uniform_unit_gradient(grad,[1,2])
    assert new_color>new_type


def test_coherent_tokens_can_still_increase_new_score_with_sqrt_k():
    grad=torch.ones(4,1,512)
    _,_,_,single=uniform_unit_gradient(grad,[0])
    _,_,_,many=uniform_unit_gradient(grad,[0,1,2,3])
    assert many==pytest.approx(2*single,rel=1e-7)


def test_single_token_v2_equals_old_score_and_direction():
    grad=torch.randn(3,1,512)
    w,g,old,new=uniform_unit_gradient(grad,[1])
    assert old==new
    delta=unit_delta(w,g,.05)
    assert torch.count_nonzero(delta[[0,2]])==0
    assert torch.allclose(delta[1,0],.05*grad[1,0]/new)


def test_cancellation_does_not_invent_direction_or_fill_old_score():
    grad=torch.zeros(2,1,512); grad[0]=1; grad[1]=-1
    w,g,old,new=uniform_unit_gradient(grad,[0,1])
    assert old>0 and new==0 and unit_delta(w,g,.01) is None


def test_duplicates_do_not_change_k_or_weights():
    grad=torch.randn(3,1,512)
    a=uniform_unit_gradient(grad,[1,2]); b=uniform_unit_gradient(grad,[1,1,2])
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and a[2:]==b[2:]
