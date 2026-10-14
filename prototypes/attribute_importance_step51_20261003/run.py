"""固定小样本诊断；没有full模式，默认只跑3句smoke。"""
import os,sys,json,argparse,time,subprocess,hashlib,collections,itertools
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode=True
for key,sub in [('TORCH_HOME','torch'),('HF_HOME','huggingface'),('XDG_CACHE_HOME','xdg'),('CUDA_CACHE_PATH','cuda')]:os.environ.setdefault(key,'/home/lzf/ldx/cache/'+sub)
import numpy as np
import torch
from importance import *
from attribute_interface import convert_record,CachedReader,verify_hashes
V4=REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'
sys.path.insert(0,str(V4));import learned_reader as v4
sys.path.insert(0,str(v4.CODE/'src'))
from attributes.irra_attribute_unit_v2_prototype import AttributeUnitV2Probe
BASE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/prototype.json')
GALLERY=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt')
BASIS=V55/'outputs/full/semantic_bases.json'
EPS=.01;TAU=.02

def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()

def parser():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('smoke','diagnostic'),default='smoke');p.add_argument('--allow-diagnostic',action='store_true');return p

def checked_inputs():
 lock=load(ROOT/'provenance.lock.json');verified={}
 for path,h in lock['frozen_sha256'].items():
  print('SHA_VERIFY',path,flush=True)
  if digest(path)!=h:raise ValueError('frozen source/cache SHA mismatch: '+path)
  verified[path]=h
 return lock,verified

def summarize_bias(results):
 rows=[a for r in results for a in r['attributes'] if a['status']=='scored'];out={'scored_instances':len(rows),'global_spearman':{},'within_caption':[],'axes':{},'rank1_vs_high_rank':{}}
 rank=[a['rank'] for a in rows]
 for k in ('S_raw','S_norm'):
  out['global_spearman'][k]=rank_correlation(rank,[a[k] for a in rows])
  out['global_spearman']['random_'+k]=rank_correlation(rank,[a['same_rank_random'][k] for a in rows])
 for r in results:
  xs=[a for a in r['attributes'] if a['status']=='scored'];item={'row_id':r['row_id'],'m':len(xs)}
  for k in ('S_raw','S_norm'):item[k]=rank_correlation([a['rank'] for a in xs],[a[k] for a in xs])
  item['ranking_changed']=r['topk']['S_raw']['order']!=r['topk']['S_norm']['order'];item['k_changed']=r['topk']['S_raw']['k_star']!=r['topk']['S_norm']['k_star'];item['selection_changed']=set(r['topk']['S_raw']['selected_ids'])!=set(r['topk']['S_norm']['selected_ids'])
  out['within_caption'].append(item)
 for axis in AXES:
  xs=[a for a in rows if a['semantic_axis']==axis]
  out['axes'][axis]={'count':len(xs),'rank':xs[0]['rank'] if xs else None}
  for key in ('S_raw','S_norm','retention'):
   vals=[a[key] for a in xs if a[key] is not None];out['axes'][axis][key]=stats(vals)
   vals=[a['same_rank_random'][key] for a in xs if a['same_rank_random'][key] is not None];out['axes'][axis]['random_'+key]=stats(vals)
 comparisons=[]
 for r in results:
  xs=[a for a in r['attributes'] if a['status']=='scored']
  for high in [a for a in xs if a['rank']>1]:
   for low in [a for a in xs if a['rank']==1]:comparisons.append({'row_id':r['row_id'],'high_axis':high['semantic_axis'],'low_axis':low['semantic_axis'],'S_raw_high_wins':high['S_raw']>low['S_raw'],'S_norm_high_wins':high['S_norm']>low['S_norm'],'random_raw_high_wins':high['same_rank_random']['S_raw']>low['same_rank_random']['S_raw'],'random_norm_high_wins':high['same_rank_random']['S_norm']>low['same_rank_random']['S_norm']})
 out['rank1_vs_high_rank']={'comparison_count':len(comparisons),'comparisons':comparisons}
 for key in ('S_raw_high_wins','S_norm_high_wins','random_raw_high_wins','random_norm_high_wins'):out['rank1_vs_high_rank'][key+'_fraction']=sum(a[key] for a in comparisons)/len(comparisons) if comparisons else None
 out['caveat']='Descriptive only: rank tied to axis semantics and location; pooled caption scales/PID repetitions confound correlation. Same-rank single-seed random is a diagnostic, not a score. No significance claim or unreasonable-ranking ground truth.'
 return out

def stats(vals):
 if not vals:return {'count':0,'mean':None,'median':None,'min':None,'max':None}
 x=np.asarray(vals,float);return {'count':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max())}

def markdown(out,results,bias,audit,protocol):
 fmt=lambda v:'—' if v is None else f'{v:.6g}'
 lines=['# Step 5.1 Attribute Importance Score Prototype','',f"模式：{protocol['mode']}；{len(results)}条caption，仅smoke不能冻结正式score。" if protocol['mode']=='smoke' else f"模式：diagnostic；固定原9句+16validation，共{len(results)}句。",'', '冻结v5.5接口/reader/Euclidean basis；block8 fp32，34054 clean train gallery，paired soft rank τ=.02，ε=.01。没有训练、basis重建、full validation或G/attack。','', '每caption唯一retrieval backward后复用G。g=wᵀG，p=Qᵀg，S_raw=||p||，S_norm=S_raw/sqrt(r)。Q正交、w单位L2、weights从baseline detach固定。rank0=score_unsupported，不参与Top-K；unresolved/e unavailable也排除。','', 'S_raw是一阶约束最大敏感度：max_{||u||=1} <G, εw(Qu)ᵀ>/ε=||Qᵀg||。有限ε的ΔF/ε只应接近它。S_norm为rank诊断，未改budget，不能冒充同budget真实敏感度。','', '## 数学链路与冻结检查','','```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','','## Rank bias诊断','', '相关性仅描述，不等于纯rank因果偏置；axis/region/标签信号与rank混杂，global还混合caption gradient scale。','', '| score | pooled Spearman(rank, score) |','|---|---:|']
 for k,v in bias['global_spearman'].items():lines.append(f'| {k} | {fmt(v)} |')
 lines+=['', '| axis | rank | n | raw median | norm median | semantic retention median | random retention median |','|---|---:|---:|---:|---:|---:|---:|']
 for axis,x in bias['axes'].items():lines.append(f"| {axis} | {x['rank']} | {x['count']} | {fmt(x['S_raw']['median'])} | {fmt(x['S_norm']['median'])} | {fmt(x['retention']['median'])} | {fmt(x['random_retention']['median'])} |")
 lines+=['', '完整caption内correlation、高rank对rank1比较、random同rank对照在rank_bias.json；未根据这些结果改变rank/seed/reader。','', '## 每caption排序与Dynamic Top-K','','tie-break遵循v5.5输入顺序；beta=1；m仅含resolved+representation-ready+rank>0实例。','']
 for r in results:
  lookup={a['instance_id']:a for a in r['attributes']}
  name=lambda aid:lookup[aid]['region']+':'+str(lookup[aid]['semantic_axis'])+'='+str(lookup[aid]['value'])+'@'+str(lookup[aid]['instance']['char_start'])
  lines += [f"### {r['row_id']}",'',r['caption'],'',f"m={r['topk']['S_raw']['m']}；未评分原因={dict(collections.Counter(a['status'] for a in r['attributes'] if a['status']!='scored'))}；backward={r['retrieval_backward_count']}。",'', '| score | k* | 排序 | Top-K |','|---|---:|---|---|']
  for key in ('S_raw','S_norm'):
   t=r['topk'][key];order=' > '.join(name(aid)+':'+fmt(s) for aid,s in zip(t['order'],t['ordered_scores']));selected=', '.join(name(aid) for aid in t['selected_ids']);lines.append(f"| {key} | {t['k_star']} | {order} | {selected} |")
  lines+=['', '| instance | rank | S_raw | S_norm | ΔF | prediction | relative error | applied norm |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for a in r['attributes']:
   if a['status']!='scored':continue
   it=a['intervention'];lines.append('| '+name(a['instance_id'])+' | '+str(a['rank'])+' | '+' | '.join(fmt(v) for v in (a['S_raw'],a['S_norm'],it.get('delta_F'),it.get('prediction'),it.get('first_order_relative_error'),it.get('effective_norm')))+' |')
 lines+=['', '## 当前判断','','S_raw严格对应冻结w/Q下的局部一阶等预算敏感度；rank效应同时包含真实可用方向和维数，不能凭相关性断言不合理。S_norm会改变跨rank排序，但扣除sqrt(r)是否适合作为正式重要性尚需固定9+16诊断。Smoke不选择正式S，且不运行attack。' if protocol['mode']=='smoke' else '请联合semantic-vs-random、caption内rank1比较及Top-K变化审计raw/norm；不凭攻击效果选择，正式score尚未冻结。','','完整metadata/source span、g/p/w/d、S及cosine/F before/after在results.json。预算与预测误差全部保留，不按结果删除实例。','']
 (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main(argv=None):
 args=parser().parse_args(argv)
 if args.mode=='diagnostic' and not args.allow_diagnostic:raise SystemExit('fixed diagnostic is user-run only; supply --allow-diagnostic')
 out=ROOT/'outputs'/args.mode
 if out.exists():raise FileExistsError('refuse overwrite: '+str(out))
 lock,verified=checked_inputs();t0=time.time();out.mkdir()
 split=load(V4/'outputs/split.json');old=load(V4/'outputs/diagnostic.json');baseline=load(BASE);bases=load(BASIS)
 assert list(bases)==list(AXES) and len(split['validation_indices'])==13700 and len(old['results'])==9
 assert not set(split['validation_pids'])&set(split['held_out_pids'])
 selected=select_rows(split,old,args.mode);ids=[r['row_id'] for r in selected];assert len(ids)==len(set(ids))
 needed=set(ids);source={}
 with v4.LABELS.open() as f:
  for line in f:
   row=json.loads(line)
   if row['row_id'] in needed:source[row['row_id']]=row
 assert set(source)==needed
 raw=load(v4.RAW);cache_lookup={r['row_id']:i for i,r in enumerate(split['records'])}
 protocol={'mode':args.mode,'row_ids':ids,'caption_count':len(ids),'validation_sample_rule':'first16 held-out validation records sorted by numeric (annotation_index,caption_index), regardless labels/scores','smoke_rule':'99:1 plus first2 of same validation sample','original_diagnostic_rows':split['diagnostic_rows'],'layer':8,'precision':'fp32','tau':TAU,'epsilon':EPS,'gallery_count':34054,'beta':1.,'basis_source':str(BASIS),'ranks':{a:bases[a]['rank'] for a in AXES},'reader_source':str(V4/'outputs/reader_final.pt'),'random_rule':'each axis: torch CPU float64 QR, independent generator seed42, exact semantic rank; never enters Top-K','scoring':'S_raw=||Q^T(w^T G)||2; S_norm=S_raw/sqrt(rank); no other weights','training':False,'basis_rebuilt':False,'full_validation':False,'retrieval_backward':'exactly once per caption; reader weights baseline-detached','frozen_sha256':verified,'new_source_sha256':{str(p):digest(p) for p in ROOT.glob('*.py')}}
 save(out/'protocol.json',protocol);save(out/'selection.json',[{'row_id':r['row_id'],'pid':r['pid'],'caption':r['caption'],'cache_index':cache_lookup.get(r['row_id'])} for r in selected])
 v4.setup();victim=v4.load_victim();reader=v4.SlotReader([len(v) for v in split['values']]).cuda();cp=torch.load(V4/'outputs/reader_final.pt',map_location='cuda',weights_only=False)
 assert cp['epoch']==20 and cp['slots']==split['slots'] and cp['values']==split['values'];reader.load_state_dict(cp['state_dict']);reader.eval().requires_grad_(False)
 before={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()};versions={k:p._version for k,p in victim.named_parameters()}
 gallery=torch.load(GALLERY,map_location='cpu',weights_only=False);assert gallery['spec']==baseline['gallery_spec'] and len(gallery['features'])==34054 and str(torch.__version__)==gallery['spec']['torch_version']
 features=gallery['features'].cuda();paths=gallery['spec']['gallery_paths'];path_index={p:i for i,p in enumerate(paths)};assert len(path_index)==34054
 frozen=CachedReader(split,bases);localizer=FrozenLocalization(reader,split,bases,frozen);controls=random_controls(bases);results=[]
 for rec in selected:
  rid=rec['row_id'];record=source[rid];assert rec['caption']==record['caption']
  anno=raw[int(rid.split(':')[1])];assert anno['split']=='train' and anno['id']==rec['pid'] and anno['file_path']==record['image']
  paired=path_index[record['image']];probe=AttributeUnitV2Probe(victim,features,paired,tau=TAU)
  with patch('torch.autograd.grad',wraps=torch.autograd.grad) as grad_calls:
   feature,H,G,F,c0,zero=baseline_gradient(probe,rec['caption'])
   assert H.dtype==torch.float32 and G.dtype==torch.float32 and not zero.any()
   with torch.no_grad():
    reference=victim.encode_texts([rec['caption']]);explicit,_,effective_zero=probe.encode(rec['caption'],8,torch.zeros_like(H))
   assert torch.equal(feature,reference) and torch.equal(explicit,reference) and not effective_zero.any()
   attrs=convert_record(record);attrs,weights,alignment,regression=localizer.attach(attrs,rec['caption'],H,cache_lookup.get(rid))
   if rid in cache_lookup:
    cached=torch.from_numpy(np.array(frozen.H[cache_lookup[rid]])).to(H.device);h_error=float((cached-H[:,0]).abs().max());assert torch.allclose(cached,H[:,0],atol=2e-4,rtol=2e-5)
   else:h_error=None
   scored=score_instances(attrs,rec['caption'],G,weights,bases,controls)
   for row in scored:
    if row['status']!='scored':continue
    w=weights[row['instance_id']];d=torch.tensor(row['direction'],device=H.device) if row['direction'] is not None else None;delta=equal_budget_delta(w,d,EPS)
    if delta is None:row['intervention']={'status':'zero_score_no_unit_direction','epsilon':EPS};continue
    with torch.no_grad():changed,_,eff=probe.encode(rec['caption'],8,delta);Fp,cpair=probe.objective(changed)
    content={t['token_index'] for t in alignment['content_tokens']};outside=eff.clone();outside[list(content)]=0;assert not outside.any()
    requested=float(delta.double().norm());actual=float(eff.norm());prediction=EPS*row['S_raw'];difference=float(Fp)-F
    budget_error=abs(actual/EPS-1);assert budget_error<1e-4,'fp32 equal-budget failed'
    row['intervention']={'status':'valid','epsilon':EPS,'requested_norm':requested,'effective_norm':actual,'budget_relative_error':budget_error,'F_before':F,'F_after':float(Fp),'delta_F':difference,'delta_F_over_epsilon':difference/EPS,'prediction':prediction,'effective_prediction':float((G.double()*eff).sum()),'first_order_absolute_error':abs(difference-prediction),'first_order_relative_error':abs(difference-prediction)/prediction,'paired_cosine_before':c0,'paired_cosine_after':float(cpair)}
   assert grad_calls.call_count==1
  probe.assert_frozen();top=topk_report(attrs,scored)
  result={'row_id':rid,'pid':rec['pid'],'caption':rec['caption'],'paired_image':record['image'],'paired_gallery_index':paired,'F':F,'paired_cosine':c0,'retrieval_backward_count':grad_calls.call_count,'zero_intervention_exact':True,'cache_H_max_abs_error':h_error,'cache_representation_regression':regression,'attributes':scored,'topk':top}
  results.append(result);save(out/'results.json',results);print('CAPTION_DONE',rid,'m',top['S_raw']['m'],'kraw',top['S_raw']['k_star'],'knorm',top['S_norm']['k_star'],flush=True)
 assert all(not p.requires_grad and p.grad is None for p in list(reader.parameters())+list(victim.parameters()))
 assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in reader.state_dict().items()) and all(p._version==versions[k] for k,p in victim.named_parameters())
 interventions=[a['intervention'] for r in results for a in r['attributes'] if a.get('intervention',{}).get('status')=='valid'];counts=collections.Counter(a['status'] for r in results for a in r['attributes']);unsupported=collections.Counter(a['semantic_axis'] for r in results for a in r['attributes'] if a['status']=='score_unsupported')
 audit={'caption_count':len(results),'instance_status_counts':dict(counts),'rank0_axes':dict(unsupported),'each_caption_one_backward':all(r['retrieval_backward_count']==1 for r in results),'total_retrieval_backwards':sum(r['retrieval_backward_count'] for r in results),'zero_intervention_exact':True,'IRRA_frozen_grad_absent':True,'reader_frozen_grad_absent_state_exact':True,'IRRA_parameter_versions_unchanged':True,'budget_relative_error':stats([a['budget_relative_error'] for a in interventions]),'first_order_relative_error':stats([a['first_order_relative_error'] for a in interventions]),'cache_H_error':stats([r['cache_H_max_abs_error'] for r in results if r['cache_H_max_abs_error'] is not None]),'cache_e_error':stats([a['projected_context_max_abs_error'] for r in results for a in r['cache_representation_regression']]),'unresolved_entered_topk':0,'rank0_entered_topk':0,'source_roundtrip_all_resolved':all(valid_span(r['caption'],a['instance']['source_span']) for r in results for a in r['attributes'] if not a['instance']['unresolved']),'same_rank_random_never_entered_topk':True,'tracked_diff':subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True),'staged_diff':subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True)}
 for r in results:
  allowed={a['instance_id'] for a in r['attributes'] if a['status']=='scored'}
  for t in r['topk'].values():assert set(t['selected_ids'])<=allowed
 print('FINAL_FROZEN_SHA_VERIFY',flush=True);verify_hashes(verified);verify_hashes(protocol['new_source_sha256']);audit['frozen_source_cache_sha_unchanged']=True;audit['verified_files']=len(verified)
 bias=summarize_bias(results);save(out/'rank_bias.json',bias);save(out/'audit.json',audit);markdown(out,results,bias,audit,protocol);save(out/'completed.json',{'mode':args.mode,'caption_count':len(results),'seconds':time.time()-t0,'full_validation':False})
 print('COMPLETE',out,json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
