"""Step5.1 独立score候选；冻结v5.5，不接线正式scorer。"""
from __future__ import annotations
import os,sys,math,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent
V55=REPO/'prototypes/attribute_unit_v55_interface_20261003'
sys.path.insert(0,str(V55))
from attribute_interface import AttributeInstance,SemanticRepresentation,AXES,DIRECT,text_candidates,dynamic_topk,valid_span


def baseline_gradient(probe,caption):
    # 整个caption唯一的retrieval backward；所有属性重用这个G。
    feature,H,zero=probe.encode(caption,8)
    F,cosine=probe.objective(feature)
    G,=torch.autograd.grad(F,H)
    if not torch.isfinite(G).all():raise ValueError('nonfinite retrieval gradient')
    return feature.detach(),H.detach(),G.detach(),float(F.detach()),float(cosine.detach()),zero


def projected_score(G,w,Q):
    if G.ndim==3 and G.shape[1]==1:G=G[:,0]
    if G.ndim!=2 or G.shape[-1]!=512 or w.shape!=(len(G),) or Q.ndim!=2 or Q.shape[0]!=512:raise ValueError('G[L,512], w[L], Q[512,r] required')
    if w.requires_grad or Q.requires_grad:raise ValueError('frozen baseline localization and basis required')
    if not all(torch.isfinite(x).all() for x in (G,w,Q)):raise ValueError('finite inputs required')
    if abs(float(w.double().norm())-1)>2e-6:raise ValueError('w must have unit L2 norm')
    if not Q.shape[1]:raise ValueError('rank0 is score_unsupported, never fallback')
    Q=Q.double();I=torch.eye(Q.shape[1],dtype=torch.float64,device=Q.device)
    if not torch.allclose(Q.T@Q,I,atol=1e-8,rtol=1e-8):raise ValueError('frozen basis must be orthonormal')
    g=(w[:,None]*G.float()).sum(0).detach();p=Q.T@g.double();S=float(p.norm())
    d=(Q@p/S).float() if S else None
    return {'g':g,'p':p,'direction':d,'S_raw':S,'S_norm':S/math.sqrt(Q.shape[1]),'raw_gradient_norm':float(g.double().norm()),'retention':S/float(g.double().norm()) if g.norm()>0 else None}


def equal_budget_delta(w,d,epsilon=.01):
    if d is None:return None
    if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('positive epsilon required')
    if abs(float(w.double().norm())-1)>2e-6 or abs(float(d.double().norm())-1)>2e-6:raise ValueError('unit w/d required')
    return (epsilon*w[:,None,None]*d[None,None,:]).float()


def score_instances(instances,caption,G,weights,bases,random_bases=None):
    if len({a.instance_id for a in instances})!=len(instances):raise ValueError('duplicate instance ID')
    eligible={a.instance_id for a in text_candidates(instances,caption,require_representation=True)};rows=[]
    for a in instances:
        row={'instance':a.to_dict(),'instance_id':a.instance_id,'semantic_axis':a.semantic_axis,'region':a.region,'value':a.canonical_value,'S_raw':None,'S_norm':None,'rank':None}
        if a.unresolved:row['status']='unresolved_excluded';rows.append(row);continue
        b=bases[a.semantic_axis];rank=b['rank'];row['rank']=rank
        if not rank:row['status']='score_unsupported';row['reason']='rank0 semantic basis; no fallback';rows.append(row);continue
        if a.instance_id not in eligible:row['status']='representation_or_source_unavailable';rows.append(row);continue
        e=a.semantic_representation
        if e.basis_id!=b['basis_id'] or e.basis_rank!=rank:raise ValueError('AttributeInstance/basis identity mismatch')
        if a.instance_id not in weights:raise ValueError('missing frozen localization for '+a.instance_id)
        w=weights[a.instance_id];Q=torch.tensor(b['Q'],dtype=torch.float64,device=G.device).reshape(512,rank)
        x=projected_score(G,w,Q);row.update({k:v for k,v in x.items() if k not in ('g','p','direction')})
        row.update(status='scored',w=w.cpu().tolist(),g=x['g'].cpu().tolist(),p=x['p'].cpu().tolist(),direction=x['direction'].cpu().tolist() if x['direction'] is not None else None,weight_norm=float(w.double().norm()))
        if random_bases is not None:
            c=projected_score(G,w,random_bases[a.semantic_axis].to(G.device));row['same_rank_random']={k:c[k] for k in ('S_raw','S_norm','raw_gradient_norm','retention')}
        rows.append(row)
    return rows


def topk_report(instances,rows):
    lookup={a.instance_id:a for a in instances};out={}
    for key in ('S_raw','S_norm'):
        scored=[(lookup[x['instance_id']],x[key]) for x in rows if x['status']=='scored']
        selected=dynamic_topk(scored)  # 复用v5.5 opaque-instance entropy adapter。
        order=sorted(scored,key=lambda x:-x[1]);out[key]={k:v for k,v in selected.items() if k!='selected_instances'}
        out[key].update(m=len(scored),order=[a.instance_id for a,s in order],ordered_scores=[s for a,s in order],selected_ids=[a.instance_id for a in selected['selected_instances']])
    return out


def random_controls(bases):
    # 每axis固定seed42，独立同rank QR；不搜索seed，不进入最终score。
    out={}
    for axis in AXES:
        rank=bases[axis]['rank'];gen=torch.Generator().manual_seed(42)
        out[axis]=torch.linalg.qr(torch.randn(512,rank,dtype=torch.float64,generator=gen),mode='reduced')[0] if rank else torch.empty(512,0,dtype=torch.float64)
    return out


def rank_correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
    def tied_ranks(v):
        order=np.argsort(v,kind='stable');out=np.empty(len(v));i=0
        while i<len(v):
            j=i+1
            while j<len(v) and v[order[j]]==v[order[i]]:j+=1
            out[order[i:j]]=(i+j-1)/2;i=j
        return out
    return float(np.corrcoef(tied_ranks(x),tied_ranks(y))[0,1])


def numeric_row_id(rec):return tuple(int(v) for v in rec['row_id'].split(':')[1:])

def select_rows(split,diagnostic,mode):
    vals=sorted((split['records'][i] for i in split['validation_indices']),key=numeric_row_id)[:16]
    if mode=='smoke':return [next(r for r in diagnostic['results'] if r['row_id']=='cuhk:99:1')]+vals[:2]
    if mode=='diagnostic':return diagnostic['results']+vals
    raise ValueError('no full mode; only smoke or fixed 9+16 diagnostic')


class FrozenLocalization:
    """原v5.5定位语义；fresh H只为9+16的小样本retrieval backward，缓存仅做回归。"""
    def __init__(self,reader,split,bases,cached):
        self.reader=reader;self.split=split;self.bases=bases;self.cached=cached
    @torch.no_grad()
    def attach(self,instances,caption,H,cache_index=None):
        al=self.cached.align(caption,self.cached.tokenizer);mask=torch.zeros(1,len(H),dtype=torch.bool,device=H.device)
        for t in al['content_tokens']:mask[0,t['token_index']]=True
        alpha,z,_=self.reader(H.detach().permute(1,0,2),mask)
        weights={};out=[];regression=[]
        for a in instances:
            if a.unresolved:out.append(a);continue
            b=self.bases[a.semantic_axis]
            if a.canonical_value not in b['values']:out.append(replace(a,representation_status='unseen_training_value'));continue
            why=[]
            for span in [a.source_span]+([a.linked_object] if a.linked_object else []):
                ids,reasons=self.cached.span_tokens(caption,span,al['all_content_tokens']);why.extend(reasons)
                if not ids:why.append('missing_token_support')
            if al['status']!='valid':why.extend(al['reasons'])
            if why:out.append(replace(a,representation_status='token_alignment_unresolved:'+','.join(sorted(set(why)))));continue
            if a.semantic_axis in DIRECT or a.semantic_axis=='hair_length':
                s=self.split['slots'].index(a.semantic_axis);aweights=alpha[0,s];pooled=z[0,s];src='frozen_v4_slot_Z:'+a.semantic_axis
            elif a.region in ('upper','lower'):
                slots=[self.split['slots'].index(a.region+'_clothing_'+v) for v in ('color','type','length')]
                aweights=alpha[0,slots].mean(0);pooled=z[0,slots].mean(0);src='frozen_v4_three_slot_region_mean:'+a.region
            elif a.region=='full_body':
                obj=a.source_span if a.semantic_axis=='garment_type' else a.linked_object
                if not valid_span(caption,obj):out.append(replace(a,representation_status='full_body_missing_original_object_span'));continue
                ids,reasons=self.cached.span_tokens(caption,obj,al['all_content_tokens'])
                if reasons or not ids:out.append(replace(a,representation_status='full_body_object_alignment_unresolved'));continue
                aweights=torch.zeros(len(H),dtype=torch.float32,device=H.device);aweights[sorted(ids)]=1/len(ids)
                pooled=H[sorted(ids),0].double().mean(0);src='existing_block8_object_uniform:full_body'
            else:out.append(replace(a,representation_status='unavailable_region_localization'));continue
            if aweights[~mask[0]].any():raise ValueError('localization escaped content tokens')
            w=(aweights/aweights.norm()).detach();weights[a.instance_id]=w
            Q=np.asarray(b['Q'],dtype=np.float64).reshape(512,b['rank']);center=np.asarray(b['center']);zz=pooled.cpu().double().numpy();proj=Q@(Q.T@(zz-center))
            centroid=np.asarray(b['centroids'][b['values'].index(a.canonical_value)])
            e=SemanticRepresentation(tuple(proj.tolist()),tuple(centroid.tolist()),b['rank'],b['basis_id'],src,b['rank']>0)
            fresh=replace(a,semantic_representation=e,representation_status='ready' if b['rank'] else 'ready_single_value_no_contrast')
            if cache_index is not None:
                old=self.cached.attach(a,cache_index,caption)
                if old.semantic_representation is None or old.representation_status!=fresh.representation_status:raise ValueError('fresh/cache representation status disagreement')
                pe=np.asarray(old.semantic_representation.projected_context);error=float(np.max(np.abs(pe-proj)))
                if not np.allclose(pe,proj,atol=3e-4,rtol=3e-5):raise ValueError('fresh/cache projected context regression failed')
                if old.instance_id!=fresh.instance_id or old.semantic_representation.basis_id!=e.basis_id or old.semantic_representation.pooling_source!=src:raise ValueError('frozen interface mismatch')
                regression.append({'instance_id':a.instance_id,'projected_context_max_abs_error':error})
                out.append(old)  # validation直接消费冻结cache上的AttributeInstance e。
            else:out.append(fresh)  # 原9句从PID split/cache留出，按同一冻结规则计算少量fresh e。
        return out,weights,al,regression
