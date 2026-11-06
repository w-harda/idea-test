import random
import pytest
import torch
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan
from tbps.baselines.cpgc_sep.sep_packed_pid import PackedPIDPlan,enable_packed_pid
from tbps.contracts import PersonID

def exact(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

@pytest.mark.parametrize('batch',[1,3])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('pids',[('x','y','x','z','y'),('x','x','y','z','x','x'),(PersonID('test','x'),PersonID('test','y'),PersonID('test','x'))])
def test_group_values_softmax_margin_ranking_rng(batch,dtype,pids):
    random.seed(42);torch.manual_seed(42)
    plan=PIDPlan.build(pids);packed=PackedPIDPlan(plan)
    x=torch.randn(batch,len(pids),dtype=dtype)
    rng=torch.get_rng_state().clone();prng=random.getstate()
    a,oa=plan.aggregate(x);b,ob=packed.aggregate(x)
    assert oa==ob and packed.members==plan.members
    exact(a,b);pa=a.softmax(-1);pb=b.softmax(-1);exact(pa,pb)
    exact(pa[:,0]-pa[:,1:].max(-1).values,pb[:,0]-pb[:,1:].max(-1).values)
    assert torch.equal(pa.argsort(-1,descending=True),pb.argsort(-1,descending=True))
    assert torch.equal(rng,torch.get_rng_state()) and random.getstate()==prng

@pytest.mark.parametrize('values',[
    [-0.,0.,0.,-0.,-1.],
    [-3.,-1.,-3.,-5.,-1.],
    [float('nan'),2.,float('nan'),float('inf'),-float('inf')],
    [1e38,-1e38,1e-38,-1e-38,0.],
])
def test_ties_nonfinite_extremes(values):
    p=PIDPlan.build(('a','a','a','b','b'))
    x=torch.tensor([values],dtype=torch.float32)
    exact(p.aggregate(x)[0],PackedPIDPlan(p).aggregate(x)[0])

@pytest.mark.parametrize('pids',[(),(float('nan'),)])
def test_original_empty_behavior(pids):
    p=PIDPlan.build(pids);q=PackedPIDPlan(p);x=torch.empty(1,len(pids))
    with pytest.raises(Exception) as old:p.aggregate(x)
    with pytest.raises(type(old.value)):q.aggregate(x)

def test_enable_preserves_classifier_cache_and_counters():
    from types import SimpleNamespace
    cache={'text':torch.zeros(2)}
    c=SimpleNamespace(pid_plan=PIDPlan.build(('a','b')),cache=cache,calls=7,hits=3)
    assert enable_packed_pid(c) is c and c.cache is cache and c.calls==7 and c.hits==3
    with pytest.raises(Exception):enable_packed_pid(c)

def test_shape_rejected():
    p=PackedPIDPlan(PIDPlan.build(('a','b')))
    with pytest.raises(Exception):p.aggregate(torch.zeros(1,3))
