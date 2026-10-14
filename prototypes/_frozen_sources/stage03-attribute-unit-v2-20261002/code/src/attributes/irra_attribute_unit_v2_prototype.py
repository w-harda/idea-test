"""统一512D Attribute Unit，独立验证，不接入正式scorer。"""
from __future__ import annotations
import math
import torch
from .irra_attribute_unit_prototype import AttributeUnitProbe, contextual_supports
from .irra_token_alignment import align_irra_tokens
from .text_extractor import extract_with_provenance


def uniform_unit_gradient(gradient, indices):
    if gradient.ndim != 3 or gradient.shape[1] != 1 or not torch.isfinite(gradient).all():
        raise ValueError('finite singleton [L,1,D] gradient required')
    indices=sorted(set(indices))
    if not indices or min(indices)<0 or max(indices)>=gradient.shape[0]:
        raise ValueError('invalid support indices')
    weights=torch.zeros(gradient.shape[0],device=gradient.device,dtype=torch.float32)
    weights[indices]=1/math.sqrt(len(indices))
    g=(weights[:,None]*gradient[:,0].float()).sum(dim=0)
    old=float(gradient[indices].double().norm().item())
    new=float(g.double().norm().item())
    return weights,g,old,new


def unit_delta(weights,g,epsilon):
    if not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('positive finite epsilon required')
    score=float(g.double().norm().item())
    if score==0:
        return None
    return epsilon*weights[:,None,None]*(g/score)[None,None,:]


class AttributeUnitV2Probe(AttributeUnitProbe):
    @torch.enable_grad()
    def run(self,caption,layer=8,epsilons=(.01,.05,.1)):
        self.assert_frozen()
        traced=extract_with_provenance(caption)
        alignment=align_irra_tokens(caption,self.victim.tokenizer,self.victim.text_length)
        rows,pairs=contextual_supports(caption,traced,alignment)
        with torch.no_grad():
            reference=self.victim.encode_texts([caption])
        feature,hidden,zero=self.encode(caption,layer)
        assert hidden.dtype==torch.float32
        assert torch.equal(reference,feature.detach()) and zero.count_nonzero()==0
        with torch.no_grad():
            explicit_zero,_,actual_zero=self.encode(caption,layer,torch.zeros_like(hidden))
        assert torch.equal(reference,explicit_zero) and actual_zero.count_nonzero()==0
        value,cosine=self.objective(feature)
        gradient,=torch.autograd.grad(value,hidden)
        baseline=float(value.item()); cosine0=float(cosine.item())
        tokens={r['token_index']:r for r in alignment['content_tokens']}
        for row in rows:
            row.update(S_old=None,S_new=None,interventions=[])
            row['support_tokens']=[tokens[i] for i in row['support_token_indices']]
            if row['support_status']!='valid': continue
            weights,g,old,new=uniform_unit_gradient(gradient,row['support_token_indices'])
            row.update(S_old=old,S_new=new,k=len(row['support_token_indices']),
                       weight_norm=float(weights.double().norm().item()),
                       uniform_weight=1/math.sqrt(len(row['support_token_indices'])),
                       direction_degrees_of_freedom=g.numel(),
                       attribute_unit=((weights[:,None]*hidden.detach()[:,0]).sum(dim=0)).cpu().tolist(),
                       attribute_gradient=g.detach().cpu().tolist(),
                       gradient_coherence_ratio=new/old if old else None)
            if new==0:
                row['intervention_status']='zero_attribute_gradient_no_direction'
                continue
            row['intervention_status']='valid'
            for epsilon in epsilons:
                delta=unit_delta(weights,g,epsilon)
                with torch.no_grad():
                    plus,_,effective=self.encode(caption,layer,delta)
                    minus,_,negative=self.encode(caption,layer,-delta)
                    fplus,cplus=self.objective(plus)
                    fminus,cminus=self.objective(minus)
                outside=effective.clone(); outside[row['support_token_indices']]=0
                assert outside.count_nonzero()==0
                expected=epsilon*new
                actual=float(fplus.item())-baseline
                actual_minus=float(fminus.item())-baseline
                central=(actual-actual_minus)/2
                norm=float(effective.norm().item())
                row['interventions'].append({
                    'epsilon':epsilon,'constructed_delta_norm':float(delta.double().norm().item()),
                    'constructed_budget_relative_error':abs(float(delta.double().norm().item())-epsilon)/epsilon,
                    'effective_delta_norm':norm,'effective_budget_relative_error':abs(norm-epsilon)/epsilon,
                    'minus_effective_delta_norm':float(negative.norm().item()),
                    'F_before':baseline,'F_after':float(fplus.item()),'F_change':actual,
                    'F_minus_change':actual_minus,'first_order_prediction':expected,
                    'first_order_effective_prediction':float((gradient.double()*effective).sum().item()),
                    'relative_prediction_error':abs(actual-expected)/expected,
                    'central_change':central,'central_relative_error':abs(central-expected)/expected,
                    'paired_cosine_before':cosine0,'paired_cosine_after':float(cplus.item()),
                    'paired_cosine_minus':float(cminus.item()),
                })
        valid=[r for r in rows if r['S_new'] is not None]
        # 每条caption独立排序，保留真实ties，不改变选中项或接线Top-K。
        rankings={name:[r['slot'] for r in sorted(valid,key=lambda r:-r[name])]
                  for name in ('S_old','S_new')}
        nested=[]
        for sub in valid:
            for sup in valid:
                if set(sub['support_token_indices'])<set(sup['support_token_indices']):
                    assert sub['S_old']<=sup['S_old']+1e-8
                    nested.append({'subset':sub['slot'],'superset':sup['slot'],
                        'subset_support':sub['support_token_indices'],'superset_support':sup['support_token_indices'],
                        'S_old_subset':sub['S_old'],'S_old_superset':sup['S_old'],
                        'S_new_subset':sub['S_new'],'S_new_superset':sup['S_new'],
                        'new_superset_smaller':sup['S_new']<sub['S_new']})
        self.assert_frozen()
        return {'caption':caption,'layer':layer,'tau':self.tau,'F':baseline,'paired_cosine':cosine0,
                'attributes':traced['attributes'],'provenance':traced['provenance'],
                'token_alignment':alignment,'attribute_results':rows,'support_pairs':pairs,
                'rankings':rankings,'nested_supports':nested,'diagnostics':{
                    'zero_intervention_exact':True,'parameters_frozen':True,'parameter_gradients_absent':True,
                    'hidden_dtype':str(hidden.dtype),'gradient_dtype':str(gradient.dtype),
                    'after_eot_gradient_max_abs':float(gradient[alignment['eot_index']+1:].abs().max().item())
                    if alignment['eot_index']+1<gradient.shape[0] else 0.}}
