"""v5 pure geometry/statistics: no model training, fixed numerical rank rule."""
import torch

def normalize(x):
    n=x.norm(dim=-1,keepdim=True)
    if not torch.isfinite(x).all() or (n==0).any():raise ValueError('zero/nonfinite sphere vector')
    return x/n

def log_map(x,mu):
    x=normalize(x.double());mu=normalize(mu.double())
    dot=(x@mu).clamp(-1,1);v=x-dot[:,None]*mu
    v=v-(v@mu)[:,None]*mu
    sine=v.norm(dim=-1);theta=torch.atan2(sine,dot)
    if ((sine<1e-12)&(dot<0)).any():raise ValueError('antipodal log map is undefined')
    scale=torch.ones_like(sine);valid=sine>1e-12;scale[valid]=theta[valid]/sine[valid]
    return v*scale[:,None],theta

def span(matrix):
    A=matrix.double();dim=A.shape[1]
    if A.shape[0]==0:return torch.empty(dim,0,dtype=torch.float64),[],0.
    _,singular,Vh=torch.linalg.svd(A,full_matrices=False)
    tol=max(A.shape)*torch.finfo(torch.float64).eps*float(singular[0]) if len(singular) else 0.
    rank=int((singular>tol).sum());Q=Vh[:rank].T.contiguous()
    return Q,singular.tolist(),tol

def geometry_from_centroids(centroids):
    C=centroids.double();m,dim=C.shape
    if m==0:return {'euclidean':torch.empty(dim,0,dtype=torch.float64),'tangent':torch.empty(dim,0,dtype=torch.float64),'euclidean_singular_values':[],'tangent_singular_values':[],'euclidean_tolerance':0.,'tangent_tolerance':0.,'reference':None,'angles':[]}
    if torch.equal(C,C[0:1].expand_as(C)):
        mu=normalize(C[0]);zero=torch.empty(dim,0,dtype=torch.float64);spectrum=[0.]*min(m,dim)
        return {'euclidean':zero,'tangent':zero.clone(),'euclidean_singular_values':spectrum,'tangent_singular_values':spectrum,'euclidean_tolerance':0.,'tangent_tolerance':0.,'reference':mu,'angles':[0.]*m}
    E,sE,tE=span(C-C.mean(0));unit=normalize(C);mu=normalize(unit.mean(0));tangent,angles=log_map(unit,mu);T,sT,tT=span(tangent)
    assert E.shape[1]<=max(0,m-1) and T.shape[1]<=max(0,m-1)
    return {'euclidean':E,'tangent':T,'euclidean_singular_values':sE,'tangent_singular_values':sT,'euclidean_tolerance':tE,'tangent_tolerance':tT,'reference':mu,'angles':angles.tolist()}

def random_basis(dim,rank,generator):
    if rank==0:return torch.empty(dim,0,dtype=torch.float64)
    Q,R=torch.linalg.qr(torch.randn(dim,rank,dtype=torch.float64,generator=generator),mode='reduced')
    sign=torch.where(R.diag()<0,-torch.ones(rank,dtype=torch.float64),torch.ones(rank,dtype=torch.float64));return Q*sign

def overlap(A,B):
    a,b=A.shape[1],B.shape[1];k=float((A.double().T@B.double()).square().sum())
    return {'left_rank':a,'right_rank':b,'trace_P_left_P_right':k,'symmetric_overlap':k/(a*b)**.5 if a*b else None,'left_fraction':k/a if a else None,'right_fraction':k/b if b else None}

def nearest(Z,C,Q=None):
    Z=Z.double();C=C.double()
    if C.shape[0]==0:return torch.full((len(Z),),-1,dtype=torch.long)
    if Q is not None:Z=Z@Q.double();C=C@Q.double()
    return (Z.square().sum(1,keepdim=True)+C.square().sum(1)[None,:]-2*Z@C.T).argmin(1)

def projected_direction(g,Q):
    Q=Q.to(device=g.device,dtype=g.dtype);p=Q@(Q.T@g);S=float(p.double().norm())
    return p,S,p/S if S else torch.zeros_like(g)

def smoke_indices(split,key):
    # Fixed first two records per observed nonnull class; no outcome-based selection.
    chosen=set()
    for s,values in enumerate(split['values']):
        for c,value in enumerate(values):
            if value=='null':continue
            found=[i for i in split[key] if split['records'][i]['labels'][s]==c][:2];chosen.update(found)
    indices=sorted(chosen);assert len(indices)<=2*sum(len(v)-1 for v in split['values'])
    return indices
