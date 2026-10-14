import torch
from value_query import value_span_indices,pooled_value_query,query_text,read_weights,unit_delta

def test_value_span_is_suffix_not_prefix_or_last_word_only():
    slot='upper_clothing_type'; value='t_shirt_shirt'; text=query_text(slot,value)
    start=len(slot.replace('_',' '))+1
    offsets=[(0,5,1),(6,14,2),(15,19,3),(start,start+1,4),(start+2,start+7,5),(start+8,len(text),6)]
    tokens=[dict(token_index=i,start=s,end=e,raw=text[s:e],included=True,span_status='valid',reasons=[]) for s,e,i in offsets]
    span,ids=value_span_indices(slot,value,dict(caption=text,all_content_tokens=tokens,eot_index=7))
    assert ids==[4,5,6] and span['raw']=='t shirt shirt'

def test_multi_piece_span_and_uniform_pool_excludes_eot():
    slot='upper_clothing_type'; value='hoodie_sweater'; text=query_text(slot,value)
    start=len(slot.replace('_',' '))+1
    offsets=[(start,start+3,4),(start+3,start+6,5),(start+7,len(text),6)]
    tokens=[dict(token_index=i,start=s,end=e,raw=text[s:e],included=True,span_status='valid',reasons=[]) for s,e,i in offsets]
    span,ids=value_span_indices(slot,value,dict(caption=text,all_content_tokens=tokens,eot_index=7))
    states=torch.zeros(8,1,2); states[4,0]=torch.tensor([1.,0.]); states[5,0]=torch.tensor([0.,2.]); states[6,0]=torch.tensor([1.,1.]); states[7,0]=torch.tensor([999.,999.])
    raw,unit,w=pooled_value_query(states,ids)
    torch.testing.assert_close(raw,torch.tensor([2.,3.])/(3**.5))
    assert w[7]==0 and not w[:4].count_nonzero()
    torch.testing.assert_close(w.norm(),torch.tensor(1.)); torch.testing.assert_close(unit.norm(),torch.tensor(1.))

def test_single_value_state_and_reused_cosine_normalization():
    states=torch.tensor([[[0.,0.]],[[3.,4.]],[[99.,99.]]],requires_grad=True)
    raw,unit,w=pooled_value_query(states,[1])
    assert torch.equal(raw,states[1,0].detach()) and not raw.requires_grad
    H=torch.tensor([[1.,0.],[0.,1.]])
    wr,ar,rr=read_weights(H,raw,[0,1]); wu,au,ru=read_weights(H,unit,[0,1])
    torch.testing.assert_close(ar,au,atol=1e-7,rtol=1e-6)
    G=torch.tensor([[[2.,1.]],[[3.,-1.]]]); g=(wr[:,None]*G[:,0]).sum(0)
    delta=unit_delta(wr,g,.01)
    assert abs(float(delta.double().norm())-.01)<1e-8