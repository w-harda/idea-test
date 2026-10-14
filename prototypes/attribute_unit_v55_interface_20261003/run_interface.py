import os,sys,json,argparse,collections,math,hashlib,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','8');os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('XDG_CACHE_HOME','/home/lzf/ldx/cache/xdg')
import numpy as np
from attribute_interface import *

FOCUS=('color','garment_type','sleeve_length','lower_garment_length','hem_extent','hair_length')
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def stat(xs):
 return {'mean':float(np.mean(xs)) if xs else None,'median':float(np.median(xs)) if xs else None,'min':min(xs) if xs else None,'max':max(xs) if xs else None,'histogram':dict(sorted(collections.Counter(xs).items()))}
def preflight():
 p=audit.load(V54/'outputs/protocol.json');expected=dict(p['old_sha256'])
 for n,h in p['source_sha256'].items():expected[str(V54/n)]=h
 for n,h in audit.load(V54/'outputs/classification_frozen.json').items():expected[str(V54/'outputs'/n)]=h
 extra=[V54/'outputs/centroid_rank.json',V54/'outputs/candidate_axes.json',audit.V53/'outputs/corpus.json',audit.old.V5/'outputs/full/corpus.json',REPO/'src/attributes/dynamic_topk.py',REPO/'src/attributes/attribute_scoring.py']
 for n in extra:expected[str(n)]=audit.digest(n)
 lock=ROOT/'provenance.lock.json'
 if lock.exists():
  prior=audit.load(lock)
  if prior['input_sha256']!=expected:raise ValueError('provenance lock input lineage changed; no automatic cache substitution')
 verify_hashes(expected)
 sources={str(ROOT/n):audit.digest(ROOT/n) for n in ('attribute_interface.py','run_interface.py','test_attribute_interface.py')}
 manifest={'input_sha256':expected,'prototype_sha256':sources,'axes':AXES,'reader':'frozen v4 epoch20','IRRA_layer':8,'precision':'fp32 existing H/Z','split':'fixed v4 PID','source_span_policy':'Stage01 provenance only; no substring relocation or raw-only extraction','semantic_method':'existing train centroid sufficient statistics; equal-value centering/SVD; no learned/random projection','formula':'projected_context=Q Q^T(z-center); value_centroid metadata remains separate; rank0 explicitly no contrast','full_execution_policy':'CLI full requires --allow-full-validation; never full by smoke/default'}
 if lock.exists() and audit.load(lock)['prototype_sha256']!=sources:raise ValueError('prototype code changed after provenance lock')
 if not lock.exists():save(lock,manifest)
 return manifest

def smoke_indices(split,limit,diag_ids):
 lookup={x['row_id']:i for i,x in enumerate(split['records'])};valid=set(split['validation_indices'])
 first=sorted(valid,key=lambda i:audit.key(split['records'][i]['row_id']))[:limit]
 cached_diag={lookup[row] for row in diag_ids if row in lookup and lookup[row] in valid}
 return sorted(set(first)|cached_diag,key=lambda i:audit.key(split['records'][i]['row_id']))

def reader_regression(split,indices):
 import torch
 torch.set_num_threads(8)
 state=torch.load(audit.old.V4/'outputs/reader_final.pt',map_location='cpu',weights_only=True)
 q=state['state_dict']['queries'].detach().requires_grad_(False);before=hashlib.sha256(q.numpy().tobytes()).hexdigest()
 H=np.load(audit.old.HFILE,mmap_mode='r');Z=np.load(audit.old.ZFILE,mmap_mode='r');checks=[]
 with torch.no_grad():
  for i in indices[:2]:
   n=split['records'][i]['content_count'];h=torch.from_numpy(np.array(H[i,1:n+1]));alpha=(h@q.T/math.sqrt(512)).T.softmax(-1);z=alpha@h
   err=float(np.max(np.abs(z.numpy()-Z[i])));assert err<2e-5
   checks.append({'row_id':split['records'][i]['row_id'],'max_abs_cache_error':err})
 assert not q.requires_grad and q.grad is None and before==hashlib.sha256(q.numpy().tobytes()).hexdigest()
 return {'checks':checks,'frozen':True,'gradient_created':False,'state_unchanged':True,'IRRA_forward_calls':0}

def run(args):
 if args.mode=='full' and not args.allow_full_validation:raise ValueError('manual full requires --allow-full-validation')
 if args.mode=='smoke' and not 1<=args.limit<=32:raise ValueError('smoke limit must be 1..32')
 dest=(ROOT/'outputs'/args.mode).resolve();assert dest.is_relative_to(Path('/home/lzf/ldx'))
 if dest.exists() and (dest/'completed.json').exists():raise ValueError('completed output exists; do not silently overwrite')
 dest.mkdir(parents=True,exist_ok=True)
 print('frozen provenance verification',flush=True);manifest=preflight();save(dest/'protocol.json',manifest)
 split=audit.load(audit.old.V4/'outputs/split.json');assert not set(split['train_pids'])&set(split['validation_pids'])
 lookup={x['row_id']:i for i,x in enumerate(split['records'])};valid=set(split['validation_indices']);diag_ids={x['row_id'] for x in audit.load(audit.old.V4/'outputs/diagnostic.json')['results']}
 selected=sorted(valid,key=lambda i:audit.key(split['records'][i]['row_id']))
 if args.mode=='smoke':
  selected=smoke_indices(split,args.limit,diag_ids)
 assert set(selected)<=valid
 wanted={split['records'][i]['row_id']:i for i in selected};records={};diag_records={}
 with audit.old.LABELS.open() as f:
  for line in f:
   rec=json.loads(line);i=wanted.get(rec['row_id'])
   if rec['row_id'] in diag_ids:diag_records[rec['row_id']]=rec
   if i is not None:
    assert rec['caption']==split['records'][i]['caption'];records[i]=rec
    if len(records)==len(wanted) and len(diag_records)==len(diag_ids):break
 assert len(records)==len(wanted) and len(diag_records)==len(diag_ids)
 source_only=[]
 for row in sorted(diag_records,key=audit.key):
  rec=diag_records[row];aa=convert_record(rec);cc=text_candidates(aa,rec['caption']);assert all(valid_span(rec['caption'],a.source_span) for a in cc);assert not any(a.unresolved for a in cc)
  source_only.append({'row_id':row,'caption':rec['caption'],'not_in_fixed_feature_cache':row not in lookup,'source_only':True,'semantic_representation_not_computed':True,'attributes':[a.to_dict() for a in aa],'resolved_source_count':len(cc)})
 save(dest/'heldout_source_span_audit.json',source_only)
 print('selected validation captions',len(selected),'mode',args.mode,flush=True)
 bases=build_bases();save(dest/'semantic_bases.json',bases);reader=CachedReader(split,bases);save(dest/'reader_regression.json',reader_regression(split,selected))
 vectors=[];counts=collections.Counter();resolved_n=[];candidate_n=[];axis_counts={a:collections.Counter() for a in AXES};axis_captions={a:set() for a in AXES};axis_unresolved=collections.Counter();reasons=collections.Counter();represent_status=collections.Counter();diagnosis=collections.defaultdict(list);roundtrip={'checked_instances':0,'mismatches':0,'same_value_different_span_groups':0,'unresolved_in_text_candidates':0};examples=[]
 with (dest/'instances.jsonl').open('w') as out:
  for nn,i in enumerate(selected):
   rec=records[i];text=rec['caption'];raw=convert_record(rec);attached=[reader.attach(a,i,text) for a in raw]
   resolved=[a for a in attached if not a.unresolved];candidates=text_candidates(attached,text,True)
   resolved_n.append(len(resolved));candidate_n.append(len(candidates));counts['instances']+=len(attached);counts['resolved']+=len(resolved);counts['attack_candidates']+=len(candidates)
   byvalue=collections.defaultdict(set)
   for a in attached:
    represent_status[a.representation_status]+=1
    if a.semantic_axis:
     axis_counts[a.semantic_axis]['instances']+=1
     if a.unresolved:axis_unresolved[a.semantic_axis]+=1
     else:axis_counts[a.semantic_axis]['resolved']+=1;axis_captions[a.semantic_axis].add(i)
    if a.unresolved:reasons.update(a.unresolved_reasons)
    else:
     roundtrip['checked_instances']+=1;ok=valid_span(text,a.source_span);roundtrip['mismatches']+=int(not ok);assert ok
     byvalue[a.canonical_value].add((a.char_start,a.char_end))
    d=a.to_dict();e=d.pop('semantic_representation');d['representation']=None
    if e:
     feature_index=len(vectors);vectors.append(np.array([e.pop('projected_context'),e.pop('value_centroid')],dtype=np.float32));d['representation']=dict(e,feature_index=feature_index,feature_file='representations.fp32.npy',feature_layout=['projected_context','value_centroid'])
     if a.semantic_axis in FOCUS and a in candidates:
      b=bases[a.semantic_axis];Q=np.asarray(b['Q']).reshape(512,b['rank']);c=np.asarray(b['centroids']);center=np.asarray(b['center']);pcs=(c-center)@Q@Q.T;p=np.array(a.semantic_representation.projected_context)
      pred=b['values'][int(np.argmin(np.sum((pcs-p)**2,axis=1)))];diagnosis[(a.semantic_axis,a.region)].append((pred==a.canonical_value,pred,a.canonical_value,a.row_id))
    out.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
   roundtrip['same_value_different_span_groups']+=sum(len(x)>1 for x in byvalue.values());roundtrip['unresolved_in_text_candidates']+=sum(a.unresolved for a in candidates)
   if nn<8 or rec['row_id'] in diag_ids:
    examples.append({'row_id':rec['row_id'],'caption':text,'attributes':[{'instance_id':a.instance_id,'axis':a.semantic_axis,'region':a.region,'value':a.canonical_value,'source_span':a.source_span,'linked_object':a.linked_object,'unresolved':a.unresolved,'reasons':a.unresolved_reasons,'representation_status':a.representation_status} for a in attached]})
   if nn and nn%1000==0:print('processed',nn,'validation captions',flush=True)
 np.save(dest/'representations.fp32.npy',np.stack(vectors) if vectors else np.empty((0,2,512),dtype=np.float32))
 n=len(selected);coverage={'mode':args.mode,'caption_count':n,'full_validation':args.mode=='full','total_instances':counts['instances'],'resolved_instances':counts['resolved'],'unresolved_ratio':1-counts['resolved']/counts['instances'] if counts['instances'] else None,'resolved_per_caption':stat(resolved_n),'topk_candidate_per_caption':stat(candidate_n),'axis':{a:{**axis_counts[a],'resolved_caption_count':len(axis_captions[a]),'resolved_caption_frequency':len(axis_captions[a])/n,'unresolved_instances':axis_unresolved[a],'unresolved_ratio':axis_unresolved[a]/axis_counts[a]['instances'] if axis_counts[a]['instances'] else None} for a in AXES},'unmapped_or_excluded_axis_instances':sum(reasons[x] for x in reasons if x.startswith('outside_fixed_attack_axes:')),'unresolved_reasons':reasons,'representation_status':represent_status,'candidate_policy':'original-span resolved + attached representation, no S or selected K computed; values do not create semantic axes'}
 save(dest/'coverage.json',coverage);save(dest/'roundtrip.json',roundtrip);save(dest/'source_span_examples.json',examples)
 sem={'method':'nearest train value centroid in existing between-value span, no classifier training; per-instance pseudo-label diagnosis, not causal selectivity','results':[{ 'axis':a,'region':r,'n':len(xs),'correct':sum(x[0] for x in xs),'accuracy':sum(x[0] for x in xs)/len(xs),'rank':bases[a]['rank'],'values':bases[a]['values'],'trivial_single_value':bases[a]['rank']==0,'examples':xs[:5]} for (a,r),xs in sorted(diagnosis.items())],'known_limitations':['color/type overlap','body/sleeve ambiguity excluded','frozen region-mean reader is not fine-axis trained','fullbody original object-uniform has no learned reader','parent/subtype partial labels remain','backpack single nonnull value rank0; bag/hat explicit negative train counts 1/2','no claim from smoke of full coverage or interface freeze']}
 overlaps=[]
 for ia,a in enumerate(FOCUS):
  qa=np.asarray(bases[a]['Q']).reshape(512,bases[a]['rank'])
  for b in FOCUS[ia+1:]:
   qb=np.asarray(bases[b]['Q']).reshape(512,bases[b]['rank']);x=qa.T@qb;t=float(np.sum(x*x));overlaps.append({'left':a,'right':b,'left_rank':qa.shape[1],'right_rank':qb.shape[1],'trace_overlap':t,'left_fraction':t/qa.shape[1] if qa.shape[1] else None,'right_fraction':t/qb.shape[1] if qb.shape[1] else None,'max_principal_cosine':float(np.linalg.svd(x,compute_uv=False)[0]) if x.size else None})
 sem['overlap']=overlaps;save(dest/'semantic_diagnostic.json',sem)
 assert roundtrip['mismatches']==0 and roundtrip['unresolved_in_text_candidates']==0
 # 最终来源回检，包括旧cache/checkpoint，不能偷偷替换cache。
 print('final source/cache SHA verification',flush=True);verify_hashes(manifest['input_sha256']);verify_hashes(manifest['prototype_sha256'])
 assert not subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True);assert not subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True)
 save(dest/'final_audit.json',{'source_cache_hash_count':len(manifest['input_sha256']),'hashes_unchanged':True,'formal_tracked_staged_clean':True,'training':False,'IRRA_forward_calls':0,'new_retrieval_or_intervention':False,'automatic_full_experiment':False,'interface_freeze_claim':False})
 lines=['# Stage03 v5.5 AttributeInstance接口验证',f'模式：{args.mode}；处理{n}条固定validation captions。Smoke只验证执行/格式，不代表完整覆盖或允许冻结正式Stage03。','固定semantic axes：`'+', '.join(AXES)+'`。',f"平均resolved instances={coverage['resolved_per_caption']['mean']:.3f}；unresolved={coverage['unresolved_ratio']:.3%}；候选数分布={coverage['topk_candidate_per_caption']['histogram']}。",f"Roundtrip检查{roundtrip['checked_instances']}，失败{roundtrip['mismatches']}；unresolved进文本候选{roundtrip['unresolved_in_text_candidates']}。",'## 各axis覆盖（当前运行集合）','| axis | resolved captions | 频率 | instances | unresolved比例 | rank |\n|---|---:|---:|---:|---:|---:|']
 for a,d in coverage['axis'].items():lines.append(f"| {a} | {d['resolved_caption_count']} | {d['resolved_caption_frequency']:.3%} | {d.get('instances',0)} | {d['unresolved_ratio']} | {bases[a]['rank']} |")
 lines+=['原9诊断句不在固定train/validation feature cache；本轮仅单列source-span审计，不补提IRRA、不混入validation覆盖/语义指标。见heldout_source_span_audit.json。','## 表示/Top-K/攻击接口','semantic representation为结构化e：512D projected_context=QQᵀ(z-center)，以及512D canonical value_centroid、basis/rank/pooling metadata。Value centroid不是retrieval攻击方向；本轮未选定哪部分用于未来S。rank0明确没有between-value contrast，不用常量centroid伪造非零子空间。','Prototype Top-K收[(AttributeInstance, supplied finite nonnegative S)]，采用旧entropy/effective-count规则；原正式函数限制旧slots和整数，未接线/修改。score()保持NotImplementedError，smoke不评分/选K。图像/文本payload复用同一e对象，文本另外拿原source span；无generator实现或修改。','## 限制','Source-span可回映不等于binding正确；原shared evidence错绑不修。细length reader未训练，fullbody使用原object span uniform H。Nearest-centroid只是pseudo-label相关信号；label层次、color/type overlap及稀少negative均保留。','## 手动full命令','```bash\ncd '+str(ROOT)+'\nPYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python run_interface.py --mode full --allow-full-validation > outputs/full.log 2>&1\n```','预计产物：`'+str(ROOT/'outputs/full')+'`。包含coverage/roundtrip/semantic_bases/semantic_diagnostic/instances.jsonl/representations.fp32.npy/reader_regression/protocol/final_audit。已完成目录不自动覆盖。未自动执行full，待用户手动完成后再审阅并决定冻结。']
 (dest/'summary.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8');save(dest/'completed.json',{'mode':args.mode,'caption_count':n,'interface_smoke_passed':True,'full_executed':args.mode=='full'});print('completed',dest,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','full'),default='smoke');p.add_argument('--limit',type=int,default=16);p.add_argument('--allow-full-validation',action='store_true');run(p.parse_args())
