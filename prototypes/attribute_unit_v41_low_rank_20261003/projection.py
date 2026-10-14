"""v4.1: frozen v4 attention + differentiable orthonormal rank64 projection."""
import sys,os,json,math,time,statistics
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent;OUT=ROOT/'outputs'
V4=REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'
sys.path.insert(0,str(V4))
from learned_reader import SlotReader,digest,setup,ce_sum,load_victim,CODE,RAW,ONTO
RANK=64;SEED=42;EPOCHS=20;BATCH=512;LR=.001

def orthogonal_basis(A):
    Q,R=torch.linalg.qr(A,mode='reduced')
    # 正R对角消除任意QR符号；非零对角附近可微，不是正交penalty。
    diag=R.diagonal(dim1=-2,dim2=-1);sign=torch.where(diag<0,-torch.ones_like(diag),torch.ones_like(diag))
    return Q*sign.unsqueeze(-2)

class LowRankProjection(nn.Module):
    def __init__(self,classes,dim=512,rank=RANK):
        super().__init__();self.rank=rank
        self.raw_basis=nn.Parameter(orthogonal_basis(torch.randn(len(classes),dim,rank)))
        self.classifiers=nn.ModuleList([nn.Linear(rank,c) for c in classes])
    def basis(self):return orthogonal_basis(self.raw_basis)
    def forward(self,Z,basis=None):
        if Z.requires_grad:raise ValueError('frozen reader pooled states must be detached')
        B=self.basis() if basis is None else basis
        low=torch.einsum('bkd,kdr->bkr',Z,B)
        return low,[c(low[:,s]) for s,c in enumerate(self.classifiers)]

@torch.no_grad()
def geometry(B,slots):
    B=B.detach().double().cpu();rank=B.shape[-1];I=torch.eye(rank,dtype=torch.float64)
    grams=B.transpose(-1,-2)@B;errors=grams-I
    overlap=[]
    for s in range(len(slots)):
        for t in range(s+1,len(slots)):
            overlap.append({'left':slots[s],'right':slots[t],'O':float((B[s].T@B[t]).square().sum()/rank)})
    return {'max_orthogonality_abs_error':float(errors.abs().max()),'max_orthogonality_F_error':float(errors.norm(dim=(-1,-2)).max()),
       'slot_orthogonality_abs_error':{slot:float(errors[s].abs().max()) for s,slot in enumerate(slots)},'overlap':overlap}

@torch.no_grad()
def evaluate(model,Z,y,indices,slots,values):
    model.eval();B=model.basis();correct=[0]*len(slots);nc=[0]*len(slots);nnn=[0]*len(slots);loss=0.
    confusion=[torch.zeros(len(v),len(v),dtype=torch.int64) for v in values]
    for start in range(0,len(indices),BATCH):
        ix=indices[start:start+BATCH];_,logits=model(Z[ix],B);truth=y[ix];loss+=float(ce_sum(logits,truth))*len(ix)
        for s,x in enumerate(logits):
            pred=x.argmax(-1);target=truth[:,s];valid=target!=values[s].index('null');hit=pred==target
            correct[s]+=int(hit.sum());nc[s]+=int((hit&valid).sum());nnn[s]+=int(valid.sum())
            n=len(values[s]);confusion[s]+=torch.bincount((target*n+pred).cpu(),minlength=n*n).reshape(n,n)
    return {'CE_sum':loss/len(indices),'slots':[{'slot':slot,'sample_count':len(indices),'accuracy':correct[s]/len(indices),'non_null_count':nnn[s],
      'non_null_accuracy':nc[s]/nnn[s] if nnn[s] else None,'values':values[s],'confusion':confusion[s].tolist()} for s,slot in enumerate(slots)]}

def train():
    setup();assert not (OUT/'protocol.json').exists()
    prev=json.loads((V4/'outputs/training.json').read_text());split=json.loads((V4/'outputs/split.json').read_text())
    assert digest(V4/'outputs/split.json')==prev['split_sha256']=='33ec828fdf45de86d77a3fcaa23918953085fd2257d825150e4600d430687891'
    assert digest(V4/'outputs/reader_final.pt')==prev['checkpoint_sha256']=='c0f4298cf7f9d1581489a2c1f1e9346dc0a8171c1355f5db0d51974d91afd793'
    assert digest(V4/'learned_reader.py')==prev['protocol']['source_sha256']
    assert prev['protocol']['epochs']==EPOCHS and prev['protocol']['batch_size']==BATCH and prev['protocol']['lr']==LR
    assert digest(ONTO)==prev['protocol']['ontology_sha256']
    cache=V4/'cache/block8.fp32.npy';assert digest(cache)==prev['protocol']['feature_cache_sha256']
    protocol={'rank':RANK,'seed':SEED,'epochs':EPOCHS,'batch_size':BATCH,'optimizer':'Adam','lr':LR,'weight_decay':0,
      'checkpoint_rule':'fixed final epoch20; no selection or search','loss':'sum_s plain CE mean over batch','basis_parameterization':'thin QR with positive R diagonal; differentiable; no penalty',
      'initialization':'independent normal matrices thin QR, seed42; new default Linear(64,K_s)',
      'frozen_v4_query':True,'frozen_IRRA':True,'training_uses_images':False,'training_uses_retrieval_objective':False,'training_uses_retrieval_gradient':False,
      'training_uses_provenance':False,'training_uses_linked_object_supervision':False,'cross_slot_orthogonality_loss':False,
      'split_sha256':digest(V4/'outputs/split.json'),'v4_checkpoint_sha256':digest(V4/'outputs/reader_final.pt'),
      'v4_training_sha256':digest(V4/'outputs/training.json'),'feature_cache_sha256':digest(cache),'source_sha256':digest(__file__)}
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    slots=split['slots'];values=split['values'];records=split['records'];N=len(records)
    state=torch.load(V4/'outputs/reader_final.pt',map_location='cuda',weights_only=True)
    reader=SlotReader([len(v) for v in values]).to('cuda');reader.load_state_dict(state['state_dict']);reader.eval().requires_grad_(False)
    reader_state={k:v.clone() for k,v in reader.state_dict().items()}
    H=torch.from_numpy(np.array(np.load(cache,mmap_mode='r'))).to('cuda');assert H.dtype==torch.float32 and H.shape==(N,77,512)
    mask=torch.zeros(N,77,dtype=torch.bool,device='cuda')
    for i,r in enumerate(records):mask[i,1:1+r['content_count']]=True
    Z=torch.empty(N,13,512,device='cuda',dtype=torch.float32)
    with torch.no_grad():
        for start in range(0,N,BATCH):
            _,pooled,_=reader(H[start:start+BATCH],mask[start:start+BATCH]);Z[start:start+len(pooled)]=pooled
    assert not H.requires_grad and not Z.requires_grad
    assert all(torch.equal(v,reader.state_dict()[k]) for k,v in reader_state.items())
    np.save(ROOT/'cache/frozen_v4_z.fp32.npy',Z.cpu().numpy())
    del H,mask;torch.cuda.empty_cache()
    y=torch.tensor([r['labels'] for r in records],device='cuda');tr=torch.tensor(split['train_indices'],device='cuda');va=torch.tensor(split['validation_indices'],device='cuda')
    torch.manual_seed(SEED);model=LowRankProjection([len(v) for v in values]).to('cuda')
    initial_geometry=geometry(model.basis(),slots);initial_validation=evaluate(model,Z,y,va,slots,values)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=0);history=[]
    for epoch in range(1,EPOCHS+1):
        model.train();order=tr[torch.randperm(len(tr),device='cuda')];total=0.;start_time=time.time()
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH];optimizer.zero_grad(set_to_none=True);_,logits=model(Z[ix]);loss=ce_sum(logits,y[ix]);loss.backward();optimizer.step()
            assert Z.grad is None and reader.queries.grad is None
            total+=float(loss.detach())*len(ix)
        validation=evaluate(model,Z,y,va,slots,values);ge=geometry(model.basis(),slots)
        assert ge['max_orthogonality_abs_error']<2e-6
        assert all(not p.requires_grad and p.grad is None for p in reader.parameters())
        assert all(torch.equal(v,reader.state_dict()[k]) for k,v in reader_state.items())
        event={'epoch':epoch,'train_CE_sum':total/len(tr),'validation':validation,'orthogonality':ge,'seconds':time.time()-start_time}
        history.append(event);print('EPOCH',epoch,'trainCE',event['train_CE_sum'],'valCE',validation['CE_sum'],'orthError',ge['max_orthogonality_abs_error'],'seconds',event['seconds'],flush=True)
    model.eval().requires_grad_(False)
    for p in model.parameters():p.grad=None
    with torch.no_grad():B=model.basis().detach().clone()
    torch.save({'state_dict':model.state_dict(),'B':B,'slots':slots,'values':values,'epoch':EPOCHS,'rank':RANK},OUT/'projection_final.pt')
    report={'protocol':protocol,'initial_geometry':initial_geometry,'initial_validation':initial_validation,'history':history,
       'final_validation':history[-1]['validation'],'final_geometry':geometry(B,slots),'checkpoint_sha256':digest(OUT/'projection_final.pt'),
       'projection_parameters':model.raw_basis.numel(),'classifier_parameters':sum(p.numel() for c in model.classifiers for p in c.parameters()),
       'frozen_reader_state_exact_unchanged':True,'training_completed_before_diagnostic':True}
    assert digest(V4/'outputs/reader_final.pt')==protocol['v4_checkpoint_sha256']
    assert digest(V4/'outputs/split.json')==protocol['split_sha256']
    (OUT/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',json.dumps(report['final_validation']),flush=True)

if __name__=='__main__':train()
