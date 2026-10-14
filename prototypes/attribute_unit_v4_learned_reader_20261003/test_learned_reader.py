import torch
from learned_reader import SlotReader,ce_sum,pid_split

def test_pid_disjoint_deterministic_and_heldout():
    a,b=pid_split(list(range(50))*2,{3,9,17})
    assert not a&b and not (a|b)&{3,9,17} and a|b==set(range(50))-{3,9,17}
    assert (a,b)==pid_split(list(reversed(range(50))),{17,9,3})

def test_masking_and_training_gradient_boundary():
    torch.manual_seed(42);model=SlotReader([3,4],dim=8)
    H=torch.randn(4,6,8);mask=torch.zeros(4,6,dtype=torch.bool);mask[:,1:4]=True
    alpha,z,logits=model(H,mask);loss=ce_sum(logits,torch.tensor([[0,1],[1,2],[2,3],[0,0]]));loss.backward()
    assert H.grad is None and model.queries.grad.norm()>0
    assert all(p.grad is not None for c in model.classifiers for p in c.parameters())
    assert not alpha[:,:,~mask[0]].count_nonzero()
    assert torch.allclose(alpha.sum(-1),torch.ones(4,2))
    changed=H.clone();changed[:,~mask[0]]=10000
    assert torch.equal(model(changed,mask)[0],alpha)

def test_frozen_reader_equal_budget_and_first_order():
    torch.manual_seed(3);model=SlotReader([3,4],dim=8).eval().requires_grad_(False)
    H=torch.randn(1,7,8);mask=torch.tensor([[False,True,True,True,True,False,False]])
    with torch.no_grad():alpha,z,_=model(H,mask)
    w=alpha[0,1]/alpha[0,1].norm();G=torch.randn(7,8);g=w@G;eps=.01
    delta=eps*w[:,None]*(g/g.norm())[None,:]
    assert torch.allclose(delta.norm(),torch.tensor(eps))
    assert torch.allclose((G*delta).sum(),eps*g.norm())
    assert not delta[~mask[0]].count_nonzero() and all(not p.requires_grad for p in model.parameters())
