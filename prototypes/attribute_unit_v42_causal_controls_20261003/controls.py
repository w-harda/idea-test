"""v4.2 causal controls; reuse frozen v4 pooled states and fixed v4.1 protocol."""
import sys,json,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';REPO=ROOT.parent.parent
V41=REPO/'prototypes/attribute_unit_v41_low_rank_20261003'
sys.path.insert(0,str(V41))
from projection import LowRankProjection,orthogonal_basis,geometry,evaluate,setup,digest,ce_sum,V4,RANK,SEED,EPOCHS,BATCH,LR

class ControlProjection(LowRankProjection):
    def __init__(self,classes,mode,dim=512,rank=RANK):
        super().__init__(classes,dim,rank);assert mode in ('B','C','D');self.mode=mode
        if mode in ('C','D'):
            with torch.no_grad():self.raw_basis.copy_(self.raw_basis[:1].expand_as(self.raw_basis).clone())
        self.raw_basis.requires_grad_(mode=='D')
    def basis(self):
        B=super().basis()
        # C shares one exact basis; no independent slot geometry.
        return B[:1].expand_as(B) if self.mode=='C' else B

@torch.no_grad()
def projector_distance(B,B0,slots):
    B=B.detach().double().cpu();B0=B0.detach().double().cpu();P0=B0@B0.T
    return {s:float((B[i]@B[i].T-P0).norm()) for i,s in enumerate(slots)}

def train():
    setup();assert not (OUT/'protocol.json').exists()
    prev=json.loads((V41/'outputs/training.json').read_text());split=json.loads((V4/'outputs/split.json').read_text())
    assert digest(V4/'outputs/split.json')==prev['protocol']['split_sha256']
    assert digest(V4/'outputs/reader_final.pt')==prev['protocol']['v4_checkpoint_sha256']
    assert digest(V4/'outputs/training.json')==prev['protocol']['v4_training_sha256']
    assert prev['protocol']['seed']==SEED and prev['protocol']['epochs']==EPOCHS and prev['protocol']['rank']==RANK
    baseline_paths=[V41/'projection.py',V41/'diagnose_projection.py',V41/'outputs/projection_final.pt',V41/'outputs/training.json',V41/'outputs/diagnostic.json',V4/'learned_reader.py',V4/'outputs/reader_final.pt',V4/'outputs/split.json',V4/'outputs/training.json',V4/'outputs/diagnostic.json']
    hashes={str(p):digest(p) for p in baseline_paths}
    cache=V41/'cache/frozen_v4_z.fp32.npy'
    protocol=dict(prev['protocol']);protocol.update({'experiment':'v4.2 causal controls','source_sha256':digest(__file__),'initialization':'B exact v4.1 seed42 initialization; C/D common first-slot B0; identical classifier initialization and shuffle for B/C/D',
      'pooled_cache':str(cache),'pooled_cache_sha256':digest(cache),'preserved_baseline_sha256':hashes,'conditions':{'A':'reuse existing v4.1','B':'independent random frozen bases; train heads','C':'shared random frozen basis; train heads','D':'shared init independent learned bases and heads'}})
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    slots=split['slots'];values=split['values'];N=len(split['records']);classes=[len(v) for v in values]
    Z=torch.from_numpy(np.array(np.load(cache,mmap_mode='r'))).to('cuda');assert Z.shape==(N,13,512) and Z.dtype==torch.float32 and not Z.requires_grad
    y=torch.tensor([r['labels'] for r in split['records']],device='cuda');tr=torch.tensor(split['train_indices'],device='cuda');va=torch.tensor(split['validation_indices'],device='cuda')
    reference_heads=None;reference_basis=None;reference_orders=None
    reports={}
    for mode in ('B','C','D'):
        torch.manual_seed(SEED);model=ControlProjection(classes,mode).to('cuda')
        init=model.basis().detach().clone();raw_init=model.raw_basis.detach().clone();heads={k:v.clone() for k,v in model.state_dict().items() if k.startswith('classifiers.')}
        if reference_heads is None:reference_heads=heads;reference_basis=init;reference_orders=[]
        else:assert all(torch.equal(v,reference_heads[k]) for k,v in heads.items());assert torch.equal(init,reference_basis[:1].expand_as(init))
        initial_geometry=geometry(init,slots)
        if mode=='B':assert initial_geometry==prev['initial_geometry']
        if mode in ('C','D'):assert torch.equal(init,init[:1].expand_as(init))
        torch.save({'B':init,'state_dict':model.state_dict(),'mode':mode},OUT/f'{mode}_initial.pt')
        initial_validation=evaluate(model,Z,y,va,slots,values)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR,weight_decay=0);history=[]
        for epoch in range(1,EPOCHS+1):
            start_time=time.time();model.train();order=tr[torch.randperm(len(tr),device='cuda')];total=0.
            if mode=='B':reference_orders.append(order.clone())
            else:assert torch.equal(order,reference_orders[epoch-1])
            for start in range(0,len(order),BATCH):
                ix=order[start:start+BATCH];optimizer.zero_grad(set_to_none=True);_,logits=model(Z[ix]);loss=ce_sum(logits,y[ix]);loss.backward();optimizer.step()
                assert Z.grad is None
                if mode in ('B','C'):assert model.raw_basis.grad is None and torch.equal(raw_init,model.raw_basis)
                total+=float(loss.detach())*len(ix)
            validation=evaluate(model,Z,y,va,slots,values);B=model.basis().detach();ge=geometry(B,slots);assert ge['max_orthogonality_abs_error']<2e-6
            if mode in ('B','C'):assert torch.equal(B,init)
            event={'epoch':epoch,'train_CE_sum':total/len(tr),'validation':validation,'orthogonality':ge,'projector_distance_from_B0':projector_distance(B,init[0],slots),'seconds':time.time()-start_time};history.append(event)
            print(mode,'EPOCH',epoch,'trainCE',event['train_CE_sum'],'valCE',validation['CE_sum'],'seconds',event['seconds'],flush=True)
        model.eval().requires_grad_(False)
        for p in model.parameters():p.grad=None
        B=model.basis().detach().clone();torch.save({'state_dict':model.state_dict(),'B':B,'slots':slots,'values':values,'epoch':EPOCHS,'rank':RANK,'mode':mode},OUT/f'{mode}_final.pt')
        report={'mode':mode,'initial_geometry':initial_geometry,'initial_validation':initial_validation,'history':history,'final_geometry':geometry(B,slots),'final_validation':history[-1]['validation'],
          'projector_distance_from_B0':projector_distance(B,init[0],slots),'checkpoint_sha256':digest(OUT/f'{mode}_final.pt'),'initial_checkpoint_sha256':digest(OUT/f'{mode}_initial.pt'),
          'frozen_basis_exact_unchanged':torch.equal(B,init) if mode in ('B','C') else None,'identical_classifier_initialization':True,'identical_minibatch_order':True,'trainable_basis_parameters':raw_init.numel() if mode=='D' else 0,'classifier_parameters':sum(p.numel() for c in model.classifiers for p in c.parameters())}
        reports[mode]=report;(OUT/f'{mode}_training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(mode,'COMPLETE',flush=True);del model,optimizer
    assert all(digest(Path(p))==h for p,h in hashes.items())
    (OUT/'training.json').write_text(json.dumps({'protocol':protocol,'conditions':reports,'all_training_completed_before_diagnostic':True,'preserved_baselines_exact':True},ensure_ascii=False,indent=2))
    print('ALL_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':train()
