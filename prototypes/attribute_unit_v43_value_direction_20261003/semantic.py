"""v4.3: fixed centered classifier directions; no training or architecture change."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';REPO=ROOT.parent.parent
V42=REPO/'prototypes/attribute_unit_v42_causal_controls_20261003'
sys.path.insert(0,str(V42))
from controls import V4,V41,ControlProjection,digest,setup,geometry

def value_direction(W,B,index):
    if W.ndim!=2 or B.ndim!=2 or W.shape[1]!=B.shape[1]:raise ValueError('linear class weights and basis shapes disagree')
    center=W.mean(dim=0);c=W[index]-center;norm=c.norm()
    if not torch.isfinite(norm) or float(norm)==0:raise ValueError('zero/nonfinite centered classifier direction; no fallback')
    q=c/norm;d=B@q
    return c,q,d

@torch.no_grad()
def extract_directions(model,B,slots,values):
    table={};rows=[]
    for s,slot in enumerate(slots):
        head=model.classifiers[s]
        if type(head) is not nn.Linear or head.in_features!=64 or head.out_features!=len(values[s]):
            raise TypeError('classifier is not the required simple Linear(64,C_s); stop')
        for index,value in enumerate(values[s]):
            if value=='null':continue
            c,q,d=value_direction(head.weight,B[s],index)
            assert d.shape==(512,) and not d.requires_grad
            table[(slot,value)]=d.detach()
            rows.append({'slot':slot,'canonical':value,'class_index':index,'class_count':len(values[s]),'center_includes_null':True,
              'centered_weight_norm':float(c.double().norm()),'q_64':q.cpu().tolist(),'d_512':d.cpu().tolist(),
              'q_norm':float(q.double().norm()),'d_norm':float(d.double().norm())})
    return table,rows

def retrieval_readout(raw,d,S_subspace):
    r=float(torch.dot(raw.double(),d.double()));S=abs(r);sign=1 if r>0 else -1 if r<0 else 0
    if S>S_subspace*(1+1e-6)+1e-12:raise ValueError('value sensitivity exceeds subspace norm')
    return r,S,sign
