import torch
from query_reader import query_text,read_weights,mass_and_top,unit_delta

def test_identity_template_unique_and_no_added_alias():
    assert query_text('hair_length','short')=='hair length short'
    assert query_text('upper_clothing_type','t_shirt_shirt')=='upper clothing type t shirt shirt'
    assert query_text('hair_length','short')!=query_text('upper_clothing_length','short')

def test_reader_masks_special_and_query_conditioning():
    H=torch.tensor([[99.,99.],[1.,0.],[0.,1.],[99.,99.],[99.,99.]],requires_grad=True)
    q=torch.tensor([1.,0.],requires_grad=True)
    w,a,r=read_weights(H,q,[1,2])
    v,b,s=read_weights(H,torch.tensor([0.,1.]),[1,2])
    assert a[1]>a[2] and b[2]>b[1]
    assert torch.equal(w[[0,3,4]],torch.zeros(3))
    assert not w.requires_grad and not a.requires_grad
    assert not torch.equal(w,v)
    torch.testing.assert_close(a.sum(),torch.tensor(1.))
    torch.testing.assert_close(w.norm(),torch.tensor(1.))

def test_baseline_fixed_reader_gradient_and_budget():
    H=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],requires_grad=True)
    w,a,r=read_weights(H,torch.tensor([1.,0.]),[0,1,2])
    G=torch.tensor([[[2.,1.]],[[3.,-1.]],[[0.,4.]]])
    g=(w[:,None]*G[:,0]).sum(0)
    u=torch.zeros(2,requires_grad=True)
    F=(G*(w[:,None,None]*u[None,None,:])).sum()
    actual,=torch.autograd.grad(F,u)
    torch.testing.assert_close(actual,g)
    delta=unit_delta(w,g,.01)
    assert abs(float(delta.double().norm())-.01)<1e-8

def test_mass_partitions_overlap_and_keeps_punctuation():
    alpha=torch.tensor([0.,.2,.3,.5,0.])
    row={'evidence_token_indices':[1,2],'object_token_indices':[2]}
    tokens=[dict(token_index=i,raw=t,bpe_piece=t) for i,t in [(1,'short'),(2,'shirt'),(3,'.')]]
    p=mass_and_top(alpha,row,tokens)
    assert p['top5'][0]['raw']=='.'
    assert abs(p['evidence_mass']+p['object_exclusive_mass']+p['context_mass']-1)<1e-7
    assert p['evidence_object_overlap_mass']>0 and p['object_exclusive_mass']==0