import torch
from diagnose import decomposition,attribute_pairs,cosine,distribution

def test_overlap_union_not_double_counted():
    grad=torch.tensor([[[1.,0.]],[[0.,2.]],[[0.,0.]]])
    row={'evidence_token_indices':[0,1],'object_token_indices':[1],'support_token_indices':[0,1]}
    d=decomposition(grad,row)
    assert d['overlap_indices']==[1]
    assert d['full']['gradient']==d['evidence']['gradient']
    assert d['full']['norm']==d['evidence']['norm']

def test_object_type_anchor_and_non_overlap():
    def row(slot,e,o,g):
        return dict(slot=slot,canonical='value',evidence_token_indices=e,object_token_indices=o,
          support_token_indices=sorted(set(e+o)),diagnostic={'full':{'gradient':g}})
    rows=[row('upper_clothing_color',[0],[1],[1,0]),row('upper_clothing_type',[1],[],[0,1]),row('hair_length',[2],[],[-1,0])]
    words={0:{'word':'short','indices':[0]},1:{'word':'shirt','indices':[1]},2:{'word':'short','indices':[2]}}
    pairs=attribute_pairs(rows,words)
    assert pairs[0]['shared_object_indices']==[1] and pairs[0]['cosine']==0
    assert not pairs[1]['support_overlap'] and pairs[1]['same_surface_evidence_words'][0]['same_occurrence']==False
    assert pairs[1]['cosine']==-1

def test_zero_undefined_and_distribution_no_threshold():
    assert cosine([0,0],[1,1]) is None
    d=distribution([-1.,0.,1.,None])
    assert d['n']==3 and d['undefined']==1 and d['median']==0