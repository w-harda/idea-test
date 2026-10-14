"""独立v5，无训练；默认smoke，全量须显式 --mode full。"""
import os,sys,json,argparse,time,itertools,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('TORCH_HOME','/home/lzf/ldx/cache/torch')
os.environ.setdefault('HF_HOME','/home/lzf/ldx/cache/huggingface')
os.environ.setdefault('CUDA_CACHE_PATH','/home/lzf/ldx/cache/cuda')
os.environ.setdefault('XDG_CACHE_HOME','/home/lzf/ldx/cache/xdg')
import numpy as np
import torch
from subspace import geometry_from_centroids,random_basis,overlap,nearest,projected_direction,smoke_indices
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent
V4=REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'
V41=REPO/'prototypes/attribute_unit_v41_low_rank_20261003'
V42=REPO/'prototypes/attribute_unit_v42_causal_controls_20261003'
V43=REPO/'prototypes/attribute_unit_v43_value_direction_20261003'
sys.path.insert(0,str(V4));import learned_reader as v4
sys.path.insert(0,str(REPO/'prototypes/attribute_unit_v3_query_20261002'))
from query_reader import AttributeUnitV2Probe
BASE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/prototype.json')
GALLERY=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
HFILE=V4/'cache/block8.fp32.npy';ZFILE=V41/'cache/frozen_v4_z.fp32.npy'
def load(path):return json.loads(Path(path).read_text())
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--features',choices=['auto','pooled','block8','extract'],default='auto')
    return p

def checked(path,expected,audit):
    path=Path(path);actual=v4.digest(path)
    if actual!=expected:raise RuntimeError(f'provenance mismatch: {path}: {actual} != {expected}')
    audit[str(path)]=actual;return actual

def verify(features):
    a={}
    checked(V4/'outputs/training.json','0267316026f4fbecbc2b0d3025d37088a1eac6439be9be3baad28d76c40e95b8',a)
    t=load(V4/'outputs/training.json');p=t['protocol'];spec=p['IRRA_spec']
    checked(V4/'learned_reader.py',p['source_sha256'],a)
    checked(V4/'outputs/reader_final.pt',t['checkpoint_sha256'],a)
    checked(V4/'outputs/split.json',t['split_sha256'],a)
    checked(v4.RAW,p['annotation_sha256'],a);checked(v4.LABELS,p['labels_sha256'],a);checked(v4.ONTO,p['ontology_sha256'],a)
    checked('/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth',spec['checkpoint_sha256'],a)
    checked(v4.CODE/'configs/irra_cuhk_official.yaml',spec['config_sha256'],a)
    for name,h in spec['source_sha256'].items():checked(Path('/home/lzf/Attack/IRRA-main/IRRA-main')/name,h,a)
    checked(v4.CODE/'src/attributes/irra_victim.py',spec['encoder_adapter_sha256'],a)
    checked(V4/'outputs/diagnostic.json','3cfc5f2bdc2bc40f7a9453a517bf06cd02f8ad16c69df6d50cdaf2f4c4c7629e',a)
    checked(V43/'outputs/diagnostic.json','bc4c981d03ec735dc79f90f5f8fbe031b0ec8a405b8ed5b1fc9697151a6b6847',a)
    if HFILE.exists():checked(HFILE,p['feature_cache_sha256'],a)
    chosen=features
    if features=='auto':chosen='pooled' if ZFILE.exists() else ('block8' if HFILE.exists() else 'extract')
    if chosen=='pooled':
        checked(V41/'outputs/training.json','9dca1a141f868b72d400a6a4576b03a58297b91d7176152ac80a7d2f808c75b8',a)
        u=load(V41/'outputs/training.json')['protocol'];checked(V41/'projection.py',u['source_sha256'],a)
        for name,expected in [('split_sha256',t['split_sha256']),('v4_checkpoint_sha256',t['checkpoint_sha256']),('v4_training_sha256',a[str(V4/'outputs/training.json')]),('feature_cache_sha256',p['feature_cache_sha256'])]:assert u[name]==expected,(name,'pool lineage mismatch')
        checked(V42/'outputs/protocol.json','a2bb3735443cfcb0566b3fe75abde29881f31d0b77aceab0660bc0fc7240f1a2',a)
        control=load(V42/'outputs/protocol.json');assert Path(control['pooled_cache'])==ZFILE
        checked(ZFILE,control['pooled_cache_sha256'],a)
    if chosen=='block8' and not HFILE.exists():raise FileNotFoundError(HFILE)
    print('PROVENANCE_OK',chosen,len(a),flush=True);return a,chosen

def mask_for(records,device):
    mask=torch.zeros(len(records),77,dtype=torch.bool,device=device)
    for i,r in enumerate(records):mask[i,1:1+r['content_count']]=True
    return mask

class Features:
    def __init__(self,mode,split,reader,victim):
        self.mode=mode;self.split=split;self.reader=reader;self.victim=victim;self.map=None
        if mode!='extract':
            self.map=np.load(ZFILE if mode=='pooled' else HFILE,mmap_mode='r')
            shape=(len(split['records']),13,512) if mode=='pooled' else (len(split['records']),77,512)
            assert self.map.shape==shape and self.map.dtype==np.float32
    @torch.no_grad()
    def get(self,indices):
        records=[self.split['records'][i] for i in indices]
        if self.mode=='pooled':z=torch.from_numpy(np.array(self.map[indices]))
        else:
            H=torch.from_numpy(np.array(self.map[indices])).cuda() if self.mode=='block8' else v4.prefix_states(self.victim,[r['caption'] for r in records])
            _,z,_=self.reader(H,mask_for(records,'cuda'));z=z.cpu()
        assert z.shape==(len(indices),13,512) and torch.isfinite(z).all();return z

@torch.no_grad()
def extraction_smoke(victim,reader,split,chosen):
    ix=split['train_indices'][:2];records=[split['records'][i] for i in ix];captions=[r['caption'] for r in records]
    prefix=v4.prefix_states(victim,captions);states=[]
    handle=victim.model.base_model.transformer.resblocks[7].register_forward_hook(lambda m,i,o:states.append(o.detach().permute(1,0,2).contiguous()))
    try:victim.encode_texts(captions)
    finally:handle.remove()
    assert len(states)==1 and torch.equal(prefix,states[0]),'prefix vs full graph'
    alpha,z,_=reader(prefix,mask_for(records,'cuda'))
    result={'indices':ix,'prefix_vs_full_exact':True,'H_shape':list(prefix.shape),'z_shape':list(z.shape),'alpha_sums_max_error':float((alpha.sum(-1)-1).abs().max())}
    if HFILE.exists():
        H=torch.from_numpy(np.array(np.load(HFILE,mmap_mode='r')[ix])).cuda()
        result['cache_H_max_abs_difference']=float((H-prefix).abs().max())
        assert torch.allclose(H,prefix,atol=2e-4,rtol=2e-5),'fresh extraction vs cache; fp32 batch-size rounding'
        _,oldz,_=reader(H,mask_for(records,'cuda'))
        if chosen=='pooled':
            cachez=torch.from_numpy(np.array(np.load(ZFILE,mmap_mode='r')[ix])).cuda()
            result['repool_vs_Z_max_abs_difference']=float((oldz-cachez).abs().max())
            assert torch.allclose(oldz,cachez,atol=2e-4,rtol=2e-5),'raw512 pooling vs cached Z'
    return result


def centroids(split,indices,features,batch=128):
    values=split['values'];counts=[torch.zeros(len(v),dtype=torch.int64) for v in values]
    sums=[torch.zeros(len(v),512,dtype=torch.float64) for v in values]
    assert set(indices)<=set(split['train_indices']),'validation leakage'
    for start in range(0,len(indices),batch):
        ix=indices[start:start+batch];z=features.get(ix).double();y=torch.tensor([split['records'][i]['labels'] for i in ix])
        for s,vs in enumerate(values):
            valid=y[:,s]!=vs.index('null');labels=y[valid,s]
            sums[s].index_add_(0,labels,z[valid,s]);counts[s]+=torch.bincount(labels,minlength=len(vs))
        if start%4096==0:print('CENTROIDS',start,len(indices),flush=True)
    result=[]
    for s,vs in enumerate(values):
        ids=[i for i,v in enumerate(vs) if v!='null' and counts[s][i]>0]
        C=torch.stack([sums[s][i]/counts[s][i] for i in ids]) if ids else torch.empty(0,512,dtype=torch.float64)
        result.append({'slot':split['slots'][s],'class_ids':ids,'values':[vs[i] for i in ids],'counts':counts[s].tolist(),'centroids':C})
    return result

def geometries(corpus):
    generator=torch.Generator().manual_seed(42);records=[]
    for item in corpus:
        geo=geometry_from_centroids(item['centroids']);E=geo['euclidean'];T=geo['tangent']
        rand=random_basis(512,max(E.shape[1],T.shape[1]),generator)
        bases={'euclidean':E,'tangent':T,'random':rand[:,:E.shape[1]]}
        if E.shape[1]!=T.shape[1]:bases['random_tangent']=rand[:,:T.shape[1]]
        records.append({'slot':item['slot'],'bases':bases,'meta':{k:v for k,v in geo.items() if k not in ('euclidean','tangent')}})
    return records

def validation(split,indices,features,corpus,geos,batch=128):
    stats=[]
    for s,item in enumerate(corpus):
        names=['full']+list(geos[s]['bases']);n=len(split['values'][s])
        stats.append({'slot':item['slot'],'total_non_null':0,'uncovered_train_values':0,'methods':{m:{'correct':0,'confusion':torch.zeros(n,n,dtype=torch.int64)} for m in names},'full_euclidean_prediction_disagreements':0})
    for start in range(0,len(indices),batch):
        ix=indices[start:start+batch];z=features.get(ix);y=torch.tensor([split['records'][i]['labels'] for i in ix])
        for s,item in enumerate(corpus):
            valid=y[:,s]!=split['values'][s].index('null');truth=y[valid,s];Z=z[valid,s];st=stats[s]
            st['total_non_null']+=len(truth);st['uncovered_train_values']+=sum(int(t) not in item['class_ids'] for t in truth)
            predictions={}
            for method,target in st['methods'].items():
                Q=None if method=='full' else geos[s]['bases'][method];p=nearest(Z,item['centroids'],Q)
                pred=torch.tensor([item['class_ids'][int(i)] if i>=0 else -1 for i in p],dtype=torch.long)
                target['correct']+=int((pred==truth).sum());predictions[method]=pred
                for t,pp in zip(truth,pred):
                    if pp>=0:target['confusion'][t,pp]+=1
            st['full_euclidean_prediction_disagreements']+=int((predictions['full']!=predictions['euclidean']).sum())
        if start%4096==0:print('VALIDATION',start,len(indices),flush=True)
    for s,st in enumerate(stats):
        for entry in st['methods'].values():
            entry['accuracy']=entry['correct']/st['total_non_null'] if st['total_non_null'] else None
            entry['confusion']=entry['confusion'].tolist()
        st['values_including_null']=split['values'][s]
    return stats

def serial(value):
    if torch.is_tensor(value):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,list):return [serial(v) for v in value]
    return value

def overlap_report(geos):
    result=[]
    for i,j in itertools.combinations(range(len(geos)),2):
        methods={m:overlap(geos[i]['bases'][m],geos[j]['bases'][m]) for m in ('euclidean','tangent','random')}
        result.append({'slots':[geos[i]['slot'],geos[j]['slot']], 'methods':methods})
    return result

def diagnostics(victim,reader,split,geos):
    reader_before={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
    victim_versions={k:p._version for k,p in victim.named_parameters()}
    baseline=load(BASE);checked(GALLERY,baseline['gallery_cache_sha256'],{})
    gallery=torch.load(GALLERY,map_location='cpu',weights_only=False)
    assert str(torch.__version__)==gallery['spec']['torch_version']
    assert gallery['spec']==baseline['gallery_spec'];assert len(gallery['features'])==34054
    old=load(V4/'outputs/diagnostic.json');v43=load(V43/'outputs/diagnostic.json')
    assert old['epsilons']==[.01,.05,.1] and old['caption_count']==9 and old['attribute_count']==45
    prior={r['row_id']:r for r in v43['results']};base={r['row_id']:r for r in baseline['results']};results=[];count=0
    for record in old['results']:
        assert record['pid'] in split['held_out_pids'] and record['pid'] not in split['train_pids'] and record['pid'] not in split['validation_pids']
        rid=record['row_id'];assert prior[rid]['caption']==record['caption'];probe=AttributeUnitV2Probe(victim,gallery['features'].cuda(),base[rid]['paired_gallery_index'],tau=.02)
        feature,H,_=probe.encode(record['caption'],8);F,c=probe.objective(feature);G,=torch.autograd.grad(F,H)
        with torch.no_grad():
            reference=victim.encode_texts([record['caption']]);zf,_,_=probe.encode(record['caption'],8,torch.zeros_like(H))
        assert torch.equal(feature,reference) and torch.equal(zf,feature),'zero intervention exact'
        mask=torch.zeros(1,77,dtype=torch.bool,device='cuda')
        for tok in record['token_alignment']['content_tokens']:mask[0,tok['token_index']]=True
        with torch.no_grad():alpha,z,_=reader(H.detach().permute(1,0,2),mask)
        for s,read in enumerate(record['all_slot_reads']):assert torch.equal(alpha[0,s].cpu(),torch.tensor(read['alpha']))
        attrs=[];directions={}
        for a,prev in zip(record['attributes'],prior[rid]['attributes']):
            assert (a['slot'],a['canonical'])==(prev['slot'],prev['canonical']);s=split['slots'].index(a['slot']);w=alpha[0,s]/alpha[0,s].norm();g=(w[:,None]*G[:,0,:]).sum(0)
            assert torch.equal(w.cpu(),torch.tensor(a['w_learned'])) and torch.equal(g.cpu(),torch.tensor(a['g_learned']))
            assert torch.equal(g.cpu(),torch.tensor(prev['g_raw'])) and torch.equal(w.cpu(),torch.tensor(prev['w']))
            entry={'slot':a['slot'],'canonical':a['canonical'],'raw_gradient_norm':float(g.double().norm()),'historical_v43_1D':{'S':prev['S_value'],'ratio':prev['S_value']/float(g.double().norm()),'interventions':prev['interventions']},'methods':{}}
            for method,Q in geos[s]['bases'].items():
                p,S,d=projected_direction(g,Q);assert S<=entry['raw_gradient_norm']*(1+2e-6);interventions=[]
                for eps in old['epsilons']:
                    if S==0:
                        interventions.append({'epsilon':eps,'status':'zero projected gradient; no unit direction','injected_norm':0.});continue
                    delta=(eps*w[:,None]*d[None,:])[:,None,:];assert not delta[~mask[0]].any()
                    with torch.no_grad():
                        changed,_,eff=probe.encode(record['caption'],8,delta);Fp,cp=probe.objective(changed)
                        minus,_,_=probe.encode(record['caption'],8,-delta);Fm,cm=probe.objective(minus)
                    actual=float(eff.double().norm());prediction=eps*S;difference=float(Fp-F)
                    interventions.append({'epsilon':eps,'requested_norm':float(delta.double().norm()),'injected_norm':actual,'budget_relative_error':abs(actual/eps-1),'prediction':prediction,'delta_F':difference,'first_order_relative_error':abs(difference-prediction)/prediction,'central_delta_F':float((Fp-Fm)/2),'central_relative_error':abs(float((Fp-Fm)/2)-prediction)/prediction,'F_before':float(F),'F_after':float(Fp),'cosine_before':float(c),'cosine_after':float(cp),'cosine_minus':float(cm)})
                entry['methods'][method]={'rank':Q.shape[1],'S':S,'ratio':S/entry['raw_gradient_norm'],'direction':d.cpu().tolist(),'interventions':interventions};directions[(a['slot'],a['canonical'],method)]=d
            attrs.append(entry);count+=1
        pairs=[]
        for ai,aj in itertools.combinations(attrs,2):
            cosines={m:float(torch.dot(directions[(ai['slot'],ai['canonical'],m)],directions[(aj['slot'],aj['canonical'],m)])) if ai['methods'][m]['S']>0 and aj['methods'][m]['S']>0 else None for m in ('euclidean','tangent','random')}
            pairs.append({'attributes':[[ai['slot'],ai['canonical']],[aj['slot'],aj['canonical']]],'cosine':cosines})
        results.append({'row_id':rid,'caption':record['caption'],'F':float(F),'paired_cosine':float(c),'attributes':attrs,'pairs':pairs,'zero_intervention_exact':True})
        probe.assert_frozen();print('DIAGNOSTIC',rid,len(attrs),flush=True)
    assert len(results)==9 and count==45
    assert all(not p.requires_grad and p.grad is None for p in list(victim.parameters())+list(reader.parameters()))
    assert all(torch.equal(v.cpu(),reader_before[k]) for k,v in reader.state_dict().items())
    assert all(p._version==victim_versions[k] for k,p in victim.named_parameters())
    summary={}
    for method in ('euclidean','tangent','random'):
        summary[method]={}
        for eps in old['epsilons']:
            entries=[it for r in results for a in r['attributes'] for it in a['methods'][method]['interventions'] if it['epsilon']==eps and 'prediction' in it]
            def stats(k):
                x=np.array([it[k] for it in entries]);return {'count':len(x),'median':float(np.median(x)) if len(x) else None,'maximum':float(x.max()) if len(x) else None}
            summary[method][str(eps)]={k:stats(k) for k in ('budget_relative_error','first_order_relative_error','central_relative_error')}
    return {'caption_count':9,'attribute_count':45,'gallery_count':34054,'epsilons':old['epsilons'],'results':results,'intervention_summary':summary,'zero_exact':True,'IRRA_frozen':True,'reader_frozen':True,'old_G_w_alpha_exact':True,'reader_state_exact_unchanged':True,'victim_parameter_versions_unchanged':True}


def markdown(out,protocol,corpus,geos,val,ov,diag,audit):
    def number(x):return '—' if x is None else f'{x:.6g}'
    lines=['# Stage03 v5 Contextual Semantic Subspace','',f"执行模式：**{protocol['mode']}**。{'仅smoke，centroid和classification不代表全量结论。全量尚未执行。' if protocol['mode']=='smoke' else '固定全量协议，未按结果调整rank/seed/reader。'}",'',f"train={protocol['train_used']}，validation={protocol['validation_used']}；原9句45属性；block8 fp32，frozen v4 reader，34054 gallery，tau=.02。",'', '## A. Corpus / rank / singular spectrum','','每个value先按样本均值取centroid，再对value等权；fp64 SVD，tol=max(m,512)*eps64*sigma_max，无能量截断。完整centroid和Q在corpus.pt；corpus.json保存counts、centroid、Q、完整谱和tol。','', '| slot | values | samples | E rank | tangent rank | E singular values | tangent singular values |','|---|---:|---:|---:|---:|---|---|']
    for c,g in zip(corpus,geos):
        meta=g['meta'];lines.append(f"| {c['slot']} | {len(c['values'])} | {sum(c['counts'])} | {g['bases']['euclidean'].shape[1]} | {g['bases']['tangent'].shape[1]} | {', '.join(number(x) for x in meta['euclidean_singular_values'])} | {', '.join(number(x) for x in meta['tangent_singular_values'])} |")
    lines+=['','## B. Held-out nearest-centroid','','所有方法用原始z和raw centroid的投影后欧氏距离；log map只构造tangent basis。full与完整Euclidean between-value span的nearest-centroid分类理论等价；随机同rank是必要对照。未见train value计错、不删样本。','', '| slot | n | full | E | tangent | random E | random tangent |','|---|---:|---:|---:|---:|---:|---:|']
    for st in val:lines.append('| '+st['slot']+' | '+str(st['total_non_null'])+' | '+' | '.join(number(st['methods'][m]['accuracy']) for m in ('full','euclidean','tangent','random'))+' | '+number(st['methods'].get('random_tangent',st['methods']['random'])['accuracy'])+' |')
    lines+=['','## C. 关键subspace overlap','','K=||QA^T QB||F²，表中为K/sqrt(rA*rB)；完整K/rA、K/rB和78对在geometry.json。','', '| pair | E | tangent | random |','|---|---:|---:|---:|']
    focus=[{'upper_clothing_color','upper_clothing_type'},{'upper_clothing_color','upper_clothing_length'},{'upper_clothing_length','upper_clothing_type'},{'lower_clothing_color','lower_clothing_type'}]
    for row in ov:
        if set(row['slots']) in focus:lines.append('| '+' / '.join(row['slots'])+' | '+' | '.join(number(row['methods'][m]['symmetric_overlap']) for m in ('euclidean','tangent','random'))+' |')
    lines+=['','## D. 原9句检索敏感度','','每属性S、S/||g||、三个epsilon的真实DeltaF/预测/paired cosine完整保存在diagnostic.json。以下逐属性给出epsilon=.01。v4.3只读复用1D旧结果；新旧span不嵌套，无S必然大小关系。','', '| caption | attribute | method | S | ratio | prediction | DeltaF | relative error | v4.3 S |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in diag['results']:
        for a in row['attributes']:
            for m,v in a['methods'].items():
                it=v['interventions'][0];lines.append('| '+row['row_id']+' | '+a['slot']+'='+a['canonical']+' | '+m+' | '+' | '.join(number(x) for x in (v['S'],v['ratio'],it.get('prediction'),it.get('delta_F'),it.get('first_order_relative_error'),a['historical_v43_1D']['S']))+' |')
    lines+=['','预算/一阶误差汇总：','','```json',json.dumps(diag['intervention_summary'],ensure_ascii=False,indent=2),'```','','## E. 99 / 326 / 3828','','原绑定及样本保持不变；326绑定疑点保留。关键direction pair如下，不能只凭cosine推断语义。','', '| caption | pair | E | tangent | random |','|---|---|---:|---:|---:|']
    for row in diag['results']:
        if row['row_id'] not in ('cuhk:99:1','cuhk:326:0','cuhk:3828:0'):continue
        slots={'upper_clothing_color','lower_clothing_color'} if row['row_id']=='cuhk:3828:0' else {'hair_length','upper_clothing_length'}
        for p in row['pairs']:
            if {a[0] for a in p['attributes']}==slots:lines.append('| '+row['row_id']+' | '+str(p['attributes'])+' | '+' | '.join(number(p['cosine'][m]) for m in ('euclidean','tangent','random'))+' |')
    lines+=['','## F. 结论边界','','全量semantic-vs-random质量结论待手动full实验；smoke仅验证代码、原图干预和文件契约，不从少量centroid选择1/2/3。' if protocol['mode']=='smoke' else '请联合B分类、C重叠及D保留敏感度判断；不将低cosine或高ratio单独视为语义纯度。','','## 来源与检查','','[CLAY Eq.3–4](https://arxiv.org/html/2604.11539v1)：本实验仅借鉴球面log map，不使用CLAY固定rank/图像对齐模块。','','```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','','## 手动完整执行','','```bash',f'cd {REPO}',f'PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/home/lzf/ldx/cache/torch HF_HOME=/home/lzf/ldx/cache/huggingface /home/lzf/ldx/envs/tbps-image/bin/python -u {ROOT}/run.py --mode full --features pooled','```','',f'预计输出：`{ROOT}/outputs/full/`。默认smoke；已有同模式目录拒绝覆盖。']
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main(argv=None):
    args=parser().parse_args(argv);out=ROOT/'outputs'/args.mode
    if out.exists():raise FileExistsError(f'refuse overwrite: {out}')
    out.mkdir(parents=True);v4.setup();t0=time.time()
    provenance,chosen=verify(args.features);save(out/'provenance.json',provenance)
    split=load(V4/'outputs/split.json');assert len(split['train_indices'])==54246 and len(split['validation_indices'])==13700
    assert not set(split['train_pids'])&set(split['validation_pids']);assert not set(split['held_out_pids'])&(set(split['train_pids'])|set(split['validation_pids']))
    victim=v4.load_victim();reader=v4.SlotReader([len(v) for v in split['values']]).cuda()
    checkpoint=torch.load(V4/'outputs/reader_final.pt',map_location='cuda',weights_only=False)
    assert checkpoint['slots']==split['slots'] and checkpoint['values']==split['values'] and checkpoint['epoch']==20
    reader.load_state_dict(checkpoint['state_dict']);reader.eval().requires_grad_(False)
    smoke=extraction_smoke(victim,reader,split,chosen);print('EXTRACTION_SMOKE',smoke,flush=True)
    train=split['train_indices'] if args.mode=='full' else smoke_indices(split,'train_indices')
    val=split['validation_indices'] if args.mode=='full' else smoke_indices(split,'validation_indices')
    protocol={'mode':args.mode,'full_experiment':args.mode=='full','features':chosen,'train_used':len(train),'validation_used':len(val),'train_indices':train,'validation_indices':val,'smoke_selection':'first2 per observed nonnull value, union; fixed indices, no result selection','layer':8,'precision':'fp32','tau':.02,'seed':42,'training':False,'rank_rule':'max(m,512)*eps64*sigma_max; retain all numerical nonzero singular values','classification':'raw z and raw train centroids; Euclidean nearest projected centroid for all methods','tangent':'unit value centroid, normalized mean reference, sphere log map, uncentered span; raw512 intervention','code_sha256':{p.name:v4.digest(p) for p in ROOT.glob('*.py')}}
    save(out/'protocol.json',protocol)
    features=Features(chosen,split,reader,victim);corpus=centroids(split,train,features);geos=geometries(corpus)
    torch.save({'corpus':corpus,'geometries':geos,'protocol':protocol},out/'corpus.pt')
    save(out/'corpus.json',serial([dict(c,geometry=g) for c,g in zip(corpus,geos)]))
    values=validation(split,val,features,corpus,geos);save(out/'validation.json',values)
    geometry=overlap_report(geos);save(out/'geometry.json',geometry)
    diag=diagnostics(victim,reader,split,geos);save(out/'diagnostic.json',diag)
    unchanged={p:v4.digest(p)==h for p,h in provenance.items()};assert all(unchanged.values())
    audit={'extraction_smoke':smoke,'verified_provenance_files':len(provenance),'provenance_sha_unchanged':all(unchanged.values()),'no_training':True,'parameters_frozen':True,'zero_intervention_exact':diag['zero_exact'],'old_G_w_alpha_exact':diag['old_G_w_alpha_exact'],'seconds':time.time()-t0,'tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True),'staged_diff':subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True)}
    save(out/'audit.json',audit);markdown(out,protocol,corpus,geos,values,geometry,diag,audit)
    save(out/'completed.json',{'mode':args.mode,'full_experiment':args.mode=='full','seconds':time.time()-t0})
    print('COMPLETE',args.mode,'train',len(train),'val',len(val),'path',out,flush=True)

if __name__=='__main__':main()
