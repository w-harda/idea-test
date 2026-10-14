import torch
from projection import LowRankProjection,orthogonal_basis,ce_sum,geometry

def test_qr_orthogonality_and_gradient_boundary():
    torch.manual_seed(42);model=LowRankProjection([3,4],dim=16,rank=4)
    Z=torch.randn(6,2,16);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    for _ in range(3):
        optimizer.zero_grad();_,logits=model(Z);ce_sum(logits,torch.tensor([[0,1],[1,2],[2,3],[0,0],[1,1],[2,2]])).backward()
        assert model.raw_basis.grad.norm()>0 and Z.grad is None
        assert all(p.grad is not None for c in model.classifiers for p in c.parameters())
        optimizer.step();assert geometry(model.basis(),['s','t'])['max_orthogonality_abs_error']<1e-6

def test_project_pool_commutation():
    torch.manual_seed(5);H=torch.randn(9,16);alpha=torch.softmax(torch.randn(9),0);B=orthogonal_basis(torch.randn(16,4))
    assert torch.allclose((alpha@H)@B,alpha@(H@B),atol=1e-6)

def test_mapped_direction_equal_budget_and_first_order():
    torch.manual_seed(2);H=torch.randn(9,16);G=torch.randn_like(H);w=torch.zeros(9);w[1:6]=torch.softmax(torch.randn(5),0);w/=w.norm()
    B=orthogonal_basis(torch.randn(16,4));raw=w@G;sub=B.T@raw;S=sub.norm();d=B@(sub/S);eps=.01
    delta=eps*w[:,None]*d[None,:]
    assert torch.allclose(delta.norm(),torch.tensor(eps),atol=1e-8)
    assert torch.allclose((G*delta).sum(),eps*S,atol=1e-8)
    assert not delta[[0,6,7,8]].count_nonzero()

def test_compare_only_common_space_under_coordinate_rotation():
    torch.manual_seed(3);B=orthogonal_basis(torch.randn(16,4));R=orthogonal_basis(torch.randn(4,4));G=torch.randn(16)
    sub=B.T@G;rot=(B@R).T@G
    mapped=B@(sub/sub.norm());mappedrot=(B@R)@(rot/rot.norm())
    assert torch.allclose(mapped,mappedrot,atol=1e-6)
    assert abs(float((B.T@(B@R)).square().sum()/4)-1)<1e-6
