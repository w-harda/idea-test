"""独立v4文本侧监督reader；不使用provenance/image/retrieval训练。"""
import os,sys,json,math,random,time,hashlib,importlib,argparse
from pathlib import Path
from functools import partial
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent
CODE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code')
RAW=Path('/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json')
LABELS=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03/cuhk_all_scores.jsonl')
DIAG=REPO/'prototypes/attribute_unit_v31_value_query_20261003/outputs/value.json'
ONTO=CODE/'ontology/upar_attribute_space.yaml'
OUT=ROOT/'outputs'
SEED=42; EPOCHS=20; BATCH=512; FEATURE_BATCH=128; LR=.001

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def schema():
    slots=yaml.safe_load(ONTO.read_text())['slots']
    return [s['key'] for s in slots],[[str(v['key']) for v in s['values']] for s in slots]

def pid_split(pids,excluded,seed=SEED):
    permitted=sorted(set(pids)-set(excluded));random.Random(seed).shuffle(permitted)
    boundary=int(len(permitted)*.8)
    return set(permitted[:boundary]),set(permitted[boundary:])

class SlotReader(nn.Module):
    def __init__(self,classes,dim=512):
        super().__init__();self.queries=nn.Parameter(torch.empty(len(classes),dim))
        nn.init.normal_(self.queries,std=.02)
        self.classifiers=nn.ModuleList([nn.Linear(dim,c) for c in classes])
    def forward(self,H,content_mask):
        if H.requires_grad:raise ValueError('reader training requires frozen detached H')
        if not content_mask.any(1).all():raise ValueError('empty content')
        scores=torch.einsum('bsd,kd->bks',H,self.queries)/math.sqrt(H.shape[-1])
        scores=scores.masked_fill(~content_mask[:,None,:],float('-inf'))
        alpha=scores.softmax(-1);z=torch.einsum('bks,bsd->bkd',alpha,H)
        logits=[classifier(z[:,s]) for s,classifier in enumerate(self.classifiers)]
        return alpha,z,logits

def ce_sum(logits,y):
    return sum(F.cross_entropy(x,y[:,s]) for s,x in enumerate(logits))

def setup():
    torch.manual_seed(SEED); np.random.seed(SEED);random.seed(SEED)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def load_victim():
    sys.path.insert(0,str(CODE/'src'))
    from attributes.irra_victim import FrozenIRRA
    irra=Path('/home/lzf/Attack/IRRA-main/IRRA-main')
    sys.path.insert(0,str(irra)); official=importlib.import_module('model.build')
    official.build_CLIP_from_openai_pretrained=partial(official.build_CLIP_from_openai_pretrained,download_root='/home/lzf/ldx/cache/clip')
    victim=FrozenIRRA(irra,'/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth',CODE/'configs/irra_cuhk_official.yaml',device='cuda').eval()
    victim.model.float(); return victim

class PrefixCaptured(Exception):pass

def prefix_states(victim,captions):
    captured=[]
    def capture(module,inputs,output):
        captured.append(output.detach().permute(1,0,2).contiguous())
        raise PrefixCaptured()
    handle=victim.model.base_model.transformer.resblocks[7].register_forward_hook(capture)
    try:
        with torch.no_grad():victim.encode_texts(captions)
    except PrefixCaptured:pass
    finally:handle.remove()
    assert len(captured)==1 and captured[0].dtype==torch.float32
    return captured[0]

def prepare(victim):
    slots,values=schema();vmap=[{v:i for i,v in enumerate(vs)} for vs in values]
    raw=json.loads(RAW.read_text());diagnostic=json.loads(DIAG.read_text())
    held_pids=set();held_text=set()
    for row in diagnostic['results']:
        _,ii,cc=row['row_id'].split(':'); item=raw[int(ii)]
        assert item['split']=='train' and item['captions'][int(cc)]==row['caption']
        held_pids.add(item['id']);held_text.add(row['caption'])
    expected={f'cuhk:{i}:{j}':(item['id'],caption,item['file_path']) for i,item in enumerate(raw) if item['split']=='train' for j,caption in enumerate(item['captions'])}
    original_held_pids=set(held_pids)
    duplicate_pids={pid for pid,caption,image in expected.values() if caption in held_text}-original_held_pids
    held_pids|=duplicate_pids
    records=[];seen=set()
    with LABELS.open() as f:
        for line in f:
            r=json.loads(line)
            if r['split']!='train':continue
            rowid=r['row_id'];pid,caption,image=expected[rowid]
            assert rowid not in seen and r['caption']==caption and r['image']==image
            seen.add(rowid)
            # 只读取文本/canonical标签；provenance、旧scores和其他字段不进入训练记录。
            labels=[vmap[s][str(r['attributes'][slot])] for s,slot in enumerate(slots)]
            if pid in held_pids:continue
            assert caption not in held_text,'cross-PID duplicate diagnostic text'
            ids=victim.tokenizer.encode(caption);n=min(len(ids),75)
            assert n>0
            records.append(dict(row_id=rowid,pid=pid,caption=caption,labels=labels,content_count=n,truncated=len(ids)>75))
    assert seen==set(expected)
    train,val=pid_split([r['pid'] for r in records],held_pids)
    trainids=[i for i,r in enumerate(records) if r['pid'] in train];valids=[i for i,r in enumerate(records) if r['pid'] in val]
    assert not train&val and not (train|val)&held_pids
    manifest={'seed':SEED,'ratio':'80/20 PID','held_out_pids':sorted(held_pids),'diagnostic_rows':[r['row_id'] for r in diagnostic['results']],
       'original_diagnostic_pids':sorted(original_held_pids),'duplicate_text_excluded_pids':sorted(duplicate_pids),
       'train_pid_count':len(train),'validation_pid_count':len(val),'train_caption_count':len(trainids),'validation_caption_count':len(valids),
       'source_train_caption_count':len(expected),'excluded_caption_count':len(expected)-len(records),
       'train_pids':sorted(train),'validation_pids':sorted(val),'train_indices':trainids,'validation_indices':valids,
       'slots':slots,'values':values,'records':records,'truncated_caption_count':sum(r['truncated'] for r in records)}
    (OUT/'split.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
    print('SPLIT',{k:v for k,v in manifest.items() if k.endswith('count') or k=='held_out_pids'},flush=True)
    return manifest

def protocol(victim):
    spec=json.loads(Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/prototype.json').read_text())['gallery_spec']
    assert digest('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth')==spec['checkpoint_sha256']
    assert digest(CODE/'configs/irra_cuhk_official.yaml')==spec['config_sha256']
    for name,h in spec['source_sha256'].items():assert digest(Path('/home/lzf/Attack/IRRA-main/IRRA-main')/name)==h
    assert digest(ONTO)=='b22b99b989c19c14c1766b21bb0347995931f6bfc3c4c6308af6b5a92c458fab'
    assert all(not p.requires_grad and p.grad is None for p in victim.parameters())
    return {'seed':SEED,'epochs':EPOCHS,'checkpoint_rule':'fixed final epoch20; no diagnostic selection','batch_size':BATCH,'feature_batch_size':FEATURE_BATCH,
      'optimizer':'Adam','lr':LR,'weight_decay':0,'loss':'sum_s plain CE mean over batch','query_initialization':'normal std=.02','classifier':'one Linear(512,K_s) per slot with default initialization',
      'layer':8,'precision':'fp32','attention':'q_s dot h_j / sqrt(512), content-only softmax; no temperature',
      'H_detached':True,'training_uses_images':False,'training_uses_retrieval_objective':False,'training_uses_retrieval_gradient':False,'training_uses_provenance':False,
      'IRRA_spec':spec,'source_sha256':digest(__file__),'annotation_sha256':digest(RAW),'labels_sha256':digest(LABELS),'ontology_sha256':digest(ONTO),'v31_sha256':digest(DIAG)}

@torch.no_grad()
def evaluate(model,H,mask,y,indices,slots,values,majority):
    model.eval();correct=[0]*len(slots);nc=[0]*len(slots);nnn=[0]*len(slots);counts=[0]*len(slots);mc=[0]*len(slots);loss=0.
    confusion=[torch.zeros(len(v),len(v),dtype=torch.int64) for v in values]
    for start in range(0,len(indices),BATCH):
        ix=indices[start:start+BATCH];_,_,logits=model(H[ix],mask[ix]);target=y[ix]
        loss+=float(ce_sum(logits,target))*len(ix)
        for s,x in enumerate(logits):
            pred=x.argmax(-1);truth=target[:,s];valid=truth!=values[s].index('null');hit=pred==truth
            correct[s]+=int(hit.sum());nc[s]+=int((hit&valid).sum());nnn[s]+=int(valid.sum());counts[s]+=len(ix)
            mc[s]+=int((truth==majority[s]).sum());nv=len(values[s]);confusion[s]+=torch.bincount((truth*nv+pred).cpu(),minlength=nv*nv).reshape(nv,nv)
    return {'CE_sum':loss/len(indices),'slots':[{'slot':slot,'sample_count':counts[s],'accuracy':correct[s]/counts[s],
       'non_null_count':nnn[s],'non_null_accuracy':nc[s]/nnn[s] if nnn[s] else None,
       'majority_train_value':values[s][majority[s]],'majority_baseline_accuracy':mc[s]/counts[s],
       'values':values[s],'confusion':confusion[s].tolist()} for s,slot in enumerate(slots)]}

def train():
    setup();assert not (OUT/'protocol.json').exists(),'separate run requires explicit new directory'
    victim=load_victim();fixed=protocol(victim)
    (OUT/'protocol.json').write_text(json.dumps(fixed,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest=prepare(victim);records=manifest['records'];N=len(records);cache=ROOT/'cache/block8.fp32.npy'
    features=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(N,77,512))
    # prefix hook只截取原IRRA第8层；与完整原encoder hook交叉核对同一caption。
    text=records[0]['caption'];Hsingle=prefix_states(victim,[text]);baseline=[]
    handle=victim.model.base_model.transformer.resblocks[7].register_forward_hook(lambda m,i,o:baseline.append(o.detach().permute(1,0,2)))
    try:
        with torch.no_grad():victim.encode_texts([text])
    finally:handle.remove()
    assert torch.equal(Hsingle,baseline[0]);fixed['prefix_capture_matches_full_encoder_exact']=True
    for start in range(0,N,FEATURE_BATCH):
        batch=records[start:start+FEATURE_BATCH];features[start:start+len(batch)]=prefix_states(victim,[r['caption'] for r in batch]).cpu().numpy()
        if start%(FEATURE_BATCH*20)==0:print('FEATURE',start,N,flush=True)
    features.flush();fixed['feature_cache_sha256']=digest(cache)
    assert all(not p.requires_grad and p.grad is None for p in victim.parameters())
    # training只持有detached text features+canonical labels，不持有victim/image/gallery。
    del victim;torch.cuda.empty_cache()
    H=torch.from_numpy(np.array(features)).to('cuda');assert not H.requires_grad
    mask=torch.zeros(N,77,dtype=torch.bool,device='cuda')
    for i,r in enumerate(records):mask[i,1:1+r['content_count']]=True
    y=torch.tensor([r['labels'] for r in records],device='cuda');slots=manifest['slots'];values=manifest['values']
    tr=torch.tensor(manifest['train_indices'],device='cuda');va=torch.tensor(manifest['validation_indices'],device='cuda')
    model=SlotReader([len(v) for v in values]).to('cuda')
    fixed['query_parameters']=model.queries.numel();fixed['classifier_parameters']=sum(p.numel() for c in model.classifiers for p in c.parameters())
    optimizer=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=0)
    majority=[int(y[tr,s].bincount(minlength=len(v)).argmax()) for s,v in enumerate(values)]
    history=[];initial=evaluate(model,H,mask,y,va,slots,values,majority)
    for epoch in range(1,EPOCHS+1):
        model.train();order=tr[torch.randperm(len(tr),device='cuda')];total=0.;t=time.time()
        for start in range(0,len(order),BATCH):
            ix=order[start:start+BATCH];optimizer.zero_grad(set_to_none=True)
            alpha,z,logits=model(H[ix],mask[ix]);loss=ce_sum(logits,y[ix]);loss.backward();optimizer.step()
            assert H.grad is None
            total+=float(loss.detach())*len(ix)
        validation=evaluate(model,H,mask,y,va,slots,values,majority)
        event={'epoch':epoch,'train_CE_sum':total/len(tr),'validation':validation,'seconds':time.time()-t}
        history.append(event);print('EPOCH',epoch,'trainCE',event['train_CE_sum'],'valCE',validation['CE_sum'], 'seconds',event['seconds'],flush=True)
    model.eval().requires_grad_(False)
    for p in model.parameters():p.grad=None
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    torch.save({'state_dict':model.state_dict(),'slots':slots,'values':values,'epoch':EPOCHS},OUT/'reader_final.pt')
    report={'protocol':fixed,'split_sha256':digest(OUT/'split.json'),'checkpoint_sha256':digest(OUT/'reader_final.pt'),
       'initial_validation':initial,'history':history,'final_validation':history[-1]['validation'],
       'train_count':len(tr),'validation_count':len(va),'training_completed_before_diagnostic':True}
    (OUT/'training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAINING_COMPLETE',json.dumps(report['final_validation']),flush=True)

if __name__=='__main__':train()
