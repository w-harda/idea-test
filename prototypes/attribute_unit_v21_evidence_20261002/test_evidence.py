import torch
from evidence import evidence_unit,unit_delta,ranking

def test_object_excluded_but_contextual_evidence_is_coordinate_gradient():
    G=torch.tensor([[[1.,2.]],[[100.,-80.]],[[3.,0.]]])
    row={'evidence_token_indices':[0,2],'object_token_indices':[1],'support_token_indices':[0,1,2]}
    w,g,old,s=evidence_unit(G,row)
    u=torch.tensor([.2,-.4],requires_grad=True)
    F=(G*(w[:,None,None]*u[None,None,:])).sum()
    actual,=torch.autograd.grad(F,u)
    torch.testing.assert_close(actual,g)
    delta=unit_delta(w,g,.01)
    assert not delta[1].count_nonzero()
    assert abs(float(delta.double().norm())-.01)<1e-8

def test_shared_evidence_has_no_slot_direction_or_ranking_resolution():
    G=torch.tensor([[[2.,-1.]],[[7.,8.]]])
    hair={'slot':'hair_length','evidence_token_indices':[0]}
    upper={'slot':'upper_clothing_length','evidence_token_indices':[0]}
    wh,gh,_,sh=evidence_unit(G,hair); wu,gu,_,su=evidence_unit(G,upper)
    assert torch.equal(gh,gu) and sh==su
    assert torch.equal(unit_delta(wh,gh,.1),unit_delta(wu,gu,.1))
    groups=ranking([dict(hair,S_e=sh),dict(upper,S_e=su)],'S_e')
    assert len(groups)==1 and len(groups[0]['slots'])==2

def test_evidence_object_overlap_not_removed_and_zero_not_invented():
    G=torch.tensor([[[1.,0.]],[[0.,0.]]])
    w,g,_,s=evidence_unit(G,{'evidence_token_indices':[0],'object_token_indices':[0]})
    assert s==1 and w[0]==1
    w,g,_,s=evidence_unit(G,{'evidence_token_indices':[1],'object_token_indices':[0]})
    assert s==0 and unit_delta(w,g,.01) is None