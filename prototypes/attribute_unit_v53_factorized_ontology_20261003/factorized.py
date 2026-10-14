"""v5.3：固定规则 ontology audit / cached region pooling / shared semantic span。"""
import os,sys,json,re,hashlib,time,collections
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','8');os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('XDG_CACHE_HOME','/home/lzf/ldx/cache/xdg')
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent;OUT=ROOT/'outputs'
V4=REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'
V5=REPO/'prototypes/attribute_unit_v5_contextual_subspace_20261003'
V51=REPO/'prototypes/attribute_unit_v51_selectivity_20261003'
V52=REPO/'prototypes/attribute_unit_v52_leakage_sources_20261003'
CODE=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code')
LABELS=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03/cuhk_all_scores.jsonl')
ZFILE=REPO/'prototypes/attribute_unit_v41_low_rank_20261003/cache/frozen_v4_z.fp32.npy'
HFILE=V4/'cache/block8.fp32.npy'
REGIONS=['upper','lower','full_body'];SEM=['color','clothing_type','clothing_length']
FIELDS=[f'{r}:{s}' for r in REGIONS for s in SEM]
CLOTH=[f'{r}_clothing_{s}' for r in ('upper','lower') for s in ('color','type','length')]
OTHERS=['age','gender','hair_length','backpack','bag','glasses','hat']
TYPEMAP={
 't shirt':('upper','shirt'),'tshirt':('upper','shirt'),'tee':('upper','shirt'),'shirt':('upper','shirt'),'blouse':('upper','shirt'),'polo':('upper','shirt'),'jersey':('upper','shirt'),
 'hoodie':('upper','hoodie_sweater'),'sweatshirt':('upper','hoodie_sweater'),'sweater':('upper','hoodie_sweater'),'jumper':('upper','hoodie_sweater'),'cardigan':('upper','hoodie_sweater'),'pullover':('upper','hoodie_sweater'),
 'jacket':('upper','jacket_coat'),'coat':('upper','jacket_coat'),'blazer':('upper','jacket_coat'),'overcoat':('upper','jacket_coat'),'parka':('upper','jacket_coat'),'windbreaker':('upper','jacket_coat'),'suit':('upper','jacket_coat'),
 'vest':('upper','vest_sleeveless'),'sleeveless top':('upper','vest_sleeveless'),'tank top':('upper','vest_sleeveless'),
 'pants':('lower','trousers'),'trouser':('lower','trousers'),'trousers':('lower','trousers'),'slacks':('lower','trousers'),'khakis':('lower','trousers'),
 'jean':('lower','jeans'),'jeans':('lower','jeans'),'shorts':('lower','shorts'),'leggings':('lower','leggings'),'tights':('lower','tights'),'capri':('lower','capri'),'capris':('lower','capri'),'skirt':('lower','skirt'),
 'dress':('full_body','dress'),'gown':('full_body','gown'),'one piece':('full_body','one_piece'),'one piece dress':('full_body','one_piece'),'full body clothing':('full_body','full_body_clothing'),
 'top':('upper',None),'upper clothes':('upper',None),'upper clothing':('upper',None),'bottoms':('lower',None),'lower clothes':('lower',None),'lower clothing':('lower',None)}
FAMILIES={'dress':r'\bdress\b','gown':r'\bgown\b','one_piece':r'\bone[ -]piece\b','skirt':r'\bskirt\b','shorts':r'\bshorts\b','trousers_pants':r'\b(?:trousers?|pants)\b','jeans':r'\bjeans?\b','shirt':r'\b(?:t[ -]?shirt|shirts?|blouse|polo|jersey)\b','jacket_coat':r'\b(?:jacket|coat|blazer|overcoat|parka|windbreaker)\b','hoodie_sweater':r'\b(?:hoodie|sweatshirt|sweater|jumper|cardigan|pullover)\b'}
BODY=re.compile(r'\b(?:one[ -]piece(?:\s+dress)?|full[ -]body\s+clothing|gown|dress)\b',re.I)
BARRIER=re.compile(r'\b(?:hair|head|face|boobs|boots?|shoes?|socks?|bags?|backpacks?|purses?|waist|neck|tie|pants|trousers?|jeans|skirt|dress|shirt|jacket|coat|sweater|hoodie)\b',re.I)
def norm(x):return re.sub(r'\s+',' ',re.sub(r'[-_]',' ',x.lower())).strip()
def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def key(x):return tuple(map(int,x.split(':')[1:]))
def checked(text,m):return isinstance(m,dict) and isinstance(m.get('start'),int) and isinstance(m.get('end'),int) and text[m['start']:m['end']]==m.get('raw')
def object_info(text,obj):
 if not checked(text,obj):return None,'invalid_object_span'
 raw=norm(obj['raw']);tail=text[obj['end']:obj['end']+18].lower()
 if raw=='one piece' and re.match(r'\s+of\b',tail):return None,'non_garment_one_piece'
 if raw=='dress' and re.match(r'\s+(?:shirts?|shoes?|pants|trousers?)\b',tail):return None,'dress_compound'
 if raw=='top' and re.search(r'\bon\s*$',text[max(0,obj['start']-7):obj['start']],re.I):return None,'spatial_top'
 return (TYPEMAP[raw],None) if raw in TYPEMAP else (None,'unknown_object_alias')
def body_mentions(text):
 result=[]
 for m in BODY.finditer(text):
  obj={'raw':m.group(),'start':m.start(),'end':m.end()};info,reason=object_info(text,obj)
  if reason or info[0]!='full_body':continue
  before=text[max(0,m.start()-35):m.start()]
  risk=bool(re.search(r'\b(?:carrying|holding|carries|holds|without|no|not)\b[^.!?;,]*$',before,re.I))
  result.append(dict(obj,type=info[1],risk=risk))
 return result

def map_record(rec):
 text=rec['caption'];attrs=rec['attributes'];prov=rec.get('provenance') or {}
 events=[];bad=[];body=body_mentions(text)
 def add(region,sem,value,source,mention,obj=None):
  events.append({'region':region,'semantic_type':sem,'value':value,'source':source,'evidence':mention,'linked_object':obj})
 for slot in CLOTH:
  value=attrs.get(slot,'null')
  if value=='null':continue
  p=prov.get(slot) or {};mentions=p.get('mentions') or []
  if not mentions:bad.append({'source':slot,'reason':'missing_provenance'});continue
  sem='clothing_'+slot.split('_')[-1] if not slot.endswith('color') else 'color'
  for m in mentions:
   obj=m if sem=='clothing_type' else m.get('linked_object')
   info,reason=object_info(text,obj)
   if not checked(text,m):reason='invalid_evidence_span'
   if reason:
    bad.append({'source':slot,'reason':reason,'mention':m});continue
   region,typ=info
   if region=='full_body' and any(b['risk'] and b['start']==obj['start'] for b in body):
    bad.append({'source':slot,'reason':'negated_or_carried_body','mention':m});continue
   if sem=='clothing_type':
    if typ is None:bad.append({'source':slot,'reason':'generic_type'});continue
    value2=typ
   else:
    value2=value
    lo,hi=(m['end'],obj['start']) if m['end']<=obj['start'] else (obj['end'],m['start'])
    gap=text[lo:hi] if hi>=lo else ''
    # jean/denim 修饰 shorts 是同一 garment phrase，不视为另一个对象。
    checked_gap=re.sub(r'\b(?:jean|denim)\b','',gap,flags=re.I)
    if len(gap.split())>8 or BARRIER.search(checked_gap) or re.search(r'[.!?;]|\band\b',checked_gap,re.I):
     bad.append({'source':slot,'reason':'cross_object_or_weak_binding','mention':m});continue
    if region=='full_body' and re.search(r'\b(?:on top|waist down|waist up|upper part|lower part)\b',text,re.I):
     bad.append({'source':slot,'reason':'full_body_panel_attribute','mention':m});continue
   add(region,sem,value2,slot,m,obj)
 # Stage01 未识别的明确 full-body type 仅作为 raw-only prototype 标签；不补 color/length。
 for m in body:
  if m['risk']:bad.append({'source':'raw_full_body','reason':'negated_or_carried_body','mention':m});continue
  if not any(e['semantic_type']=='clothing_type' and e['region']=='full_body' and e['evidence']['start']==m['start'] for e in events):
   add('full_body','clothing_type',m['type'],'raw_full_body',m,m)
 labels={f:None for f in FIELDS};sources={f:[] for f in FIELDS}
 for f in FIELDS:
  r,s=f.split(':');es=[e for e in events if e['region']==r and e['semantic_type']==s];vs=sorted(set(e['value'] for e in es))
  if len(vs)==1:labels[f]=vs[0];sources[f]=es
  elif len(vs)>1:bad.append({'source':f,'reason':'multiple_conflicting_values','values':vs})
 # 一件 one-piece dress 的复合 raw anchor 与 Stage01 dress 是同一个对象，统一更明确标签。
 if labels['full_body:clothing_type'] is None and any(m['type']=='one_piece' for m in body) and all(e['value'] in ('dress','one_piece') for e in events if e['region']=='full_body' and e['semantic_type']=='clothing_type'):
  labels['full_body:clothing_type']='one_piece';sources['full_body:clothing_type']=[e for e in events if e['region']=='full_body' and e['semantic_type']=='clothing_type']
 return {'row_id':rec['row_id'],'labels':labels,'sources':sources,'unresolved':bad,'body_mentions':body,'full_body_token_indices':None}

def pair_select(records,mapped,indices,region,sem):
 target=f'{region}:{sem}';siblings=[f'{region}:{x}' for x in SEM if x!=sem]
 # strict：color固定非null type，type固定非null color；length固定两者。剩余控制进入Hamming。
 required=[f'{region}:clothing_type'] if sem=='color' else [f'{region}:color'] if sem=='clothing_type' else siblings
 fields=FIELDS+OTHERS
 def sig(i):return tuple(mapped[i]['labels'].get(f) if f in FIELDS else records[i]['attributes'][f] for f in fields)
 eligible=sorted([i for i in indices if mapped[i]['labels'][target] is not None and (region!='full_body' or mapped[i]['full_body_token_indices'])],key=lambda i:key(records[i]['row_id']))
 signatures={}
 for i in eligible:signatures.setdefault(sig(i),i)
 items=list(signatures.items());si=fields.index(target);cs=[fields.index(f) for f in required];ss=[fields.index(f) for f in siblings]
 # 字符串编码避免 Python pairwise循环；null是未知，不当作真实相等控制。
 Y=np.array([['__null__' if x in (None,'null') else x for x in s] for s,_ in items]);ids=[i for _,i in items];choices={}
 for row,(s,i) in enumerate(items):
  a=Y[row];different=Y[:,si]!=a[si];strict=different.copy()
  for t in cs:strict&=(a[t]!='__null__')&(Y[:,t]==a[t])
  cand=np.flatnonzero(strict);tier='strict'
  if not len(cand):cand=np.flatnonzero(different);tier='relaxed'
  if not len(cand):choices[s]=None;continue
  yy=Y[cand];h=(yy!=a).sum(1)-1
  if tier=='strict':order=np.lexsort((cand,h))
  else:
   rh=(yy[:,ss]!=a[ss]).sum(1);unknown=((yy[:,ss]=='__null__')|(a[ss]=='__null__')).sum(1)
   order=np.lexsort((cand,h,unknown,rh))
  j=ids[int(cand[order[0]])];choices[s]=(j,tier)
 pairs={}
 for i in eligible:
  c=choices[sig(i)]
  if c is None:continue
  j,tier=c;a,b=sorted((i,j),key=lambda t:key(records[t]['row_id']));pkey=(a,b)
  if pkey in pairs:continue
  sa,sb=sig(a),sig(b);diff=[f for f,x,y in zip(fields,sa,sb) if f!=target and x!=y]
  pairs[pkey]={'pair_id':f'{target}|{records[a]["row_id"]}|{records[b]["row_id"]}','region':region,'semantic_type':sem,'target':target,'tier':tier,'a':a,'b':b,'row_a':records[a]['row_id'],'row_b':records[b]['row_id'],'value_a':mapped[a]['labels'][target],'value_b':mapped[b]['labels'][target],'non_target_hamming':len(diff),'different':diff,'controls':required}
 return sorted(pairs.values(),key=lambda p:(key(p['row_a']),key(p['row_b'])))

def basis(centroids,rng=None):
 c=np.asarray(centroids,dtype=np.float64);center=c.mean(0);C=c-center
 _,s,vh=np.linalg.svd(C,full_matrices=False);tol=max(C.shape)*np.finfo(np.float64).eps*(s[0] if len(s) else 0);rank=int((s>tol).sum());Q=vh[:rank].T
 rng=np.random.default_rng(42) if rng is None else rng;R=np.linalg.qr(rng.normal(size=(512,rank)),mode='reduced')[0] if rank else np.zeros((512,0))
 return {'center':center.tolist(),'rank':rank,'singular_values':s.tolist(),'tolerance':float(tol),'Q':Q.tolist(),'Q_random':R.tolist()}
def energies(delta,Qs):
 x=np.asarray(delta,dtype=np.float64);den=np.sum(x*x,axis=-1);valid=den>0
 return {s:np.divide(np.sum((x@np.asarray(q))**2,axis=-1),den,out=np.full_like(den,np.nan),where=valid) for s,q in Qs.items()},valid

def stats(x):
 x=np.asarray(x);return {'mean':float(x.mean()),'median':float(np.median(x)),'min':float(x.min()),'max':float(x.max())} if len(x) else None

def summarize(E,R,valid,target):
 if not np.any(valid):return {'count':0}
 other=[s for s in E if s!=target];et=E[target][valid];leak=np.maximum.reduce([E[s][valid] for s in other]);rt=R[target][valid];rl=np.maximum.reduce([R[s][valid] for s in other])
 return {'count':int(valid.sum()),'target':stats(et),'leak':stats(leak),'margin':stats(et-leak),'hit':float((et>leak).mean()),'sibling_energies':{s:stats(E[s][valid]) for s in other},'sibling_hit':{s:float((et>E[s][valid]).mean()) for s in other},'random_target':stats(rt),'random_leak':stats(rl),'random_hit':float((rt>rl).mean()),'target_gt_random':float((et>rt).mean())}


def prepare():
 OUT.mkdir(exist_ok=True);(ROOT/'cache').mkdir(exist_ok=True)
 baseline=load(V5/'outputs/full/provenance.json');hashes={}
 for p,h in baseline.items():
  assert digest(p)==h,('baseline SHA mismatch',p);hashes[p]=h
 for p in [REPO/'ontology/upar_attribute_space.yaml',REPO/'ontology/alias_map.yaml',REPO/'src/attributes/text_extractor.py',V5/'subspace.py',V5/'outputs/full/corpus.json',V51/'selectivity.py',V51/'outputs/pairs.json',V51/'outputs/energy.json',V52/'outputs/groups.json',V52/'outputs/groups_frozen.json']:
  hashes[str(p)]=digest(p)
 for n in ('ontology/upar_attribute_space.yaml','ontology/alias_map.yaml','src/attributes/text_extractor.py'):
  assert (REPO/n).read_text()==(CODE/n).read_text(),('semantic source mismatch',n)
  hashes[str(CODE/n)]=digest(CODE/n)
 save(OUT/'source_format_audit.json',{'difference':'CRLF/LF only; normalized text exact identical for ontology/alias/extractor','formal_byte_sha256':{n:digest(REPO/n) for n in ('ontology/upar_attribute_space.yaml','ontology/alias_map.yaml','src/attributes/text_extractor.py')},'snapshot_byte_sha256':{n:digest(CODE/n) for n in ('ontology/upar_attribute_space.yaml','ontology/alias_map.yaml','src/attributes/text_extractor.py')}})
 split=load(V4/'outputs/split.json');assert len(split['train_indices'])==54246 and len(split['validation_indices'])==13700
 assert not set(split['train_indices'])&set(split['validation_indices'])
 assert not set(split['train_pids'])&set(split['validation_pids'])
 lookup={r['row_id']:i for i,r in enumerate(split['records'])};train=set(split['train_indices']);val=set(split['validation_indices'])
 records=[None]*len(lookup);freq={s:{'train':collections.Counter(),'validation':collections.Counter()} for s in CLOTH}
 raw_audit={f:{t:{'raw_caption_count':0,'raw_occurrences':0,'provenance_slots':collections.Counter(),'linked_slots':collections.Counter(),'raw_without_provenance_caption_count':0} for t in ('train','validation')} for f in FAMILIES}
 excluded=0
 with LABELS.open() as fh:
  for line in fh:
   a=json.loads(line);i=lookup.get(a['row_id'])
   if i is None:excluded+=1;continue
   assert a['caption']==split['records'][i]['caption'];records[i]=a;part='train' if i in train else 'validation'
   for s in CLOTH:freq[s][part][a['attributes'][s]]+=1
   for family,pat in FAMILIES.items():
    mentions=list(re.finditer(pat,a['caption'],re.I));d=raw_audit[family][part]
    if not mentions:continue
    d['raw_caption_count']+=1;d['raw_occurrences']+=len(mentions);mapped=False
    for s,p in (a.get('provenance') or {}).items():
     for m in (p or {}).get('mentions',[]):
      if re.search(pat,m.get('raw',''),re.I):d['provenance_slots'][s]+=1;mapped=True
      ob=m.get('linked_object') or {}
      if re.search(pat,ob.get('raw',''),re.I):d['linked_slots'][s]+=1;mapped=True
    d['raw_without_provenance_caption_count']+=int(not mapped)
 assert all(r is not None for r in records)
 ontology_freq={s:{'canonical_values':split['values'][split['slots'].index(s)],'train':dict(freq[s]['train']),'validation':dict(freq[s]['validation'])} for s in CLOTH}
 # 明确补所有0频次canonical值。
 for s,d in ontology_freq.items():
  for t in ('train','validation'):d[t]={v:d[t].get(v,0) for v in d['canonical_values']}
 save(OUT/'ontology_audit.json',{'canonical_frequency':ontology_freq,'raw_mapping':raw_audit,'excluded_from_fixed_split':excluded,'raw_mapping_counts':'caption occurrence / provenance mention / linked_object mention separately; dress shirt compounds retained in raw audit'})
 mapped=[map_record(r) for r in records]
 print('mapping complete',len(mapped),flush=True)
 sys.path.insert(0,str(CODE/'src'));sys.path.insert(0,'/home/lzf/Attack/IRRA-main/IRRA-main')
 from attributes.irra_token_alignment import align_irra_tokens,_span_token_indices
 from utils.simple_tokenizer import SimpleTokenizer
 tokenizer=SimpleTokenizer();bodycount=0;unresolved=collections.Counter()
 for i,m in enumerate(mapped):
  if not any(m['labels'][f'full_body:{s}'] is not None for s in SEM):continue
  objs=[b for b in m['body_mentions'] if not b['risk']]
  a=align_irra_tokens(records[i]['caption'],tokenizer);ix=set();reasons=[]
  for b in objs:
   j,rr=_span_token_indices(records[i]['caption'],b,a['all_content_tokens']);ix.update(j);reasons.extend(rr)
  if a['status']!='valid':reasons.extend(a['reasons'])
  if not ix or max(ix,default=0)>split['records'][i]['content_count']:reasons.append('full_body_object_truncated_or_missing')
  if reasons:
   m['unresolved'].append({'source':'full_body_localization','reason':';'.join(sorted(set(reasons)))});unresolved.update(reasons)
   for s in SEM:m['labels'][f'full_body:{s}']=None
  else:m['full_body_token_indices']=sorted(ix);bodycount+=1
 z=np.load(ZFILE,mmap_mode='r');H=np.load(HFILE,mmap_mode='r');assert z.shape==(67946,13,512) and H.shape==(67946,77,512)
 zr=np.lib.format.open_memmap(ROOT/'cache/region_z.fp32.npy',mode='w+',dtype=np.float32,shape=(len(mapped),3,512));zr[:]=0
 for ri,r in enumerate(('upper','lower')):
  ids=[split['slots'].index(f'{r}_clothing_{s}') for s in ('color','type','length')]
  for lo in range(0,len(mapped),1024):zr[lo:lo+1024,ri]=np.mean(z[lo:lo+1024,ids,:],axis=1,dtype=np.float32)
 for i,m in enumerate(mapped):
  ix=m['full_body_token_indices']
  if ix:zr[i,2]=np.mean(H[i,ix,:],axis=0,dtype=np.float32)
 zr.flush()
 # CPU cached-H alpha重算只验证region mean确实是同一组token权重；没有IRRA实例。
 import torch
 torch.set_num_threads(8)
 state=torch.load(V4/'outputs/reader_final.pt',map_location='cpu',weights_only=True);q=state['state_dict']['queries'].detach();q.requires_grad_(False)
 checks=[];examples=[]
 testids=[0,34,100]+[lookup[x] for x in ('cuhk:99:1','cuhk:326:0','cuhk:3828:0') if x in lookup]
 with torch.no_grad():
  for i in testids:
   h=torch.from_numpy(np.array(H[i]));n=split['records'][i]['content_count'];score=h[1:n+1]@q.T/(512**.5);alpha=score.T.softmax(-1)
   for ri,r in enumerate(('upper','lower')):
    ids=[split['slots'].index(f'{r}_clothing_{s}') for s in ('color','type','length')];ar=alpha[ids].mean(0);fresh=(ar@h[1:n+1]).numpy();err=float(np.max(np.abs(fresh-zr[i,ri])))
    assert err<2e-5,('region pooling cache regression',i,r,err)
    checks.append({'row_id':records[i]['row_id'],'region':r,'max_abs_error':err,'alpha_sum':float(ar.sum())})
    top=torch.topk(ar,min(5,n)).indices.tolist();alignment=align_irra_tokens(records[i]['caption'],tokenizer);tokens={t['token_index']:t for t in alignment['content_tokens']}
    examples.append({'row_id':records[i]['row_id'],'region':r,'top_positions':[t+1 for t in top],'top_tokens':[tokens.get(t+1) for t in top],'top_weights':[float(ar[t]) for t in top],'all_alpha':ar.tolist()})
 save(OUT/'localization_audit.json',{'upper_lower':'L1 mean of existing 3 color/type/length v4 alphas; cached z mean exactly represents this pooling modulo fp32 accumulation','full_body':'L1 uniform over aligned explicit full-body object spans, original cached H; no learned full-body reader','full_body_usable_captions':bodycount,'alignment_unresolved':unresolved,'cache_checks':checks,'examples':examples,'IRRA_forward_calls':0,'parameters_trainable':False})
 coverage={};failcounts={}
 for part,indices in [('train',split['train_indices']),('validation',split['validation_indices'])]:
  coverage[part]={f:dict(collections.Counter(mapped[i]['labels'][f] or 'unresolved_or_null' for i in indices)) for f in FIELDS}
  failcounts[part]=dict(collections.Counter(b['reason'] for i in indices for b in mapped[i]['unresolved']))
 save(OUT/'coverage.json',{'by_split':coverage,'unresolved_reasons':failcounts})
 save(OUT/'mapped_labels.json',mapped)
 pairs=[]
 for r in REGIONS:
  for s in SEM:
   pp=pair_select(records,mapped,split['validation_indices'],r,s);pairs.extend(pp);print('pairs',r,s,len(pp),flush=True)
 save(OUT/'pairs.json',pairs)
 # 固定前10 strict审计，数值计算前确定。
 audits=[]
 for r in REGIONS:
  for s in SEM:
   pp=[p for p in pairs if p['region']==r and p['semantic_type']==s and p['tier']=='strict'][:10]
   for p in pp:
    audits.append({'pair':p,'A':records[p['a']],'B':records[p['b']],'factorized_A':mapped[p['a']],'factorized_B':mapped[p['b']]})
 save(OUT/'pair_audit_sample.json',audits)
 protocol={'training':False,'IRRA_forward_calls':0,'retrieval_calls':0,'rank_rule':'v5 float64 numerical rank: max(matrix.shape)*eps64*largest singular value','seed':42,'centroids':'all legal train region frames mean within value, equal-weight value centroids centered; no region input to SVD','region_localization':'upper/lower alpha mean across3 existing slot readers; full_body uniform explicit object span','type_refinement':'upper existing 4 groups; lower lexical trousers/jeans/shorts/leggings/tights/capri/skirt; full_body dress/gown/one_piece/full_body_clothing; original labels untouched','strict_controls':'color fixes nonnull type; type fixes nonnull color; length fixes nonnull color+type; minimize remaining factorized+7 other-slot Hamming; numeric row tie; no strict candidate -> relaxed; unordered dedup','full_body_old_pair_analysis':'reuse v5.2 lower strict full-body failed pairs; cross-region transitions separately marked; no reselection','old_sha256':hashes,'source_sha256':{n:digest(ROOT/n) for n in ('factorized.py','test_factorized.py')},'region_cache_sha256':digest(ROOT/'cache/region_z.fp32.npy'),'train_count':len(train),'validation_count':len(val)}
 save(OUT/'protocol.json',protocol)
 save(OUT/'selection_frozen.json',{n:digest(OUT/n) for n in ('mapped_labels.json','pairs.json','pair_audit_sample.json','coverage.json','ontology_audit.json','localization_audit.json','protocol.json')})
 print('selection frozen; no energy outcomes consulted',flush=True)
 return split,records,mapped,pairs,hashes


def evaluate(split,records,mapped,pairs,hashes):
 zr=np.load(ROOT/'cache/region_z.fp32.npy',mmap_mode='r');z=np.load(ZFILE,mmap_mode='r');corpus={};rng=np.random.default_rng(42)
 for s in SEM:
  groups=collections.defaultdict(list)
  for i in split['train_indices']:
   for ri,r in enumerate(REGIONS):
    value=mapped[i]['labels'][f'{r}:{s}']
    if value is not None:groups[value].append((i,ri))
  values=sorted(groups);c=[];counts=[]
  for v in values:
   rows=groups[v];c.append(np.mean(np.array([zr[i,ri] for i,ri in rows],dtype=np.float64),axis=0));counts.append(len(rows))
  assert len(c)>0
  corpus[s]={'values':values,'counts':counts,'centroids':np.array(c).tolist(),'geometry':basis(c,rng)}
 save(OUT/'corpus.json',corpus)
 Q={s:np.array(corpus[s]['geometry']['Q']) for s in SEM};R={s:np.array(corpus[s]['geometry']['Q_random']) for s in SEM}
 # 度量子空间几何：小子空间方向落入大空间的平均能量；不解释成纯度。
 overlap={}
 for s in SEM:
  for t in SEM:
   a=Q[s].T@Q[t];sv=np.linalg.svd(a,compute_uv=False)
   overlap[f'{s}->{t}']={'source_rank':Q[s].shape[1],'target_rank':Q[t].shape[1],'mean_source_projection_energy':float(np.sum(a*a)/Q[s].shape[1]) if Q[s].shape[1] else None,'principal_cosines':sv.tolist()}
 save(OUT/'overlap.json',overlap)
 classifications={}
 for r in REGIONS:
  classifications[r]={}
  for s in SEM:
   ids=[i for i in split['validation_indices'] if mapped[i]['labels'][f'{r}:{s}'] is not None];vals=corpus[s]['values'];cs=np.array(corpus[s]['centroids']);acc={};unseen=0
   if not ids:classifications[r][s]={'count':0};continue
   X=np.array(zr[ids,REGIONS.index(r)],dtype=np.float64);y=np.array([vals.index(mapped[i]['labels'][f'{r}:{s}']) if mapped[i]['labels'][f'{r}:{s}'] in vals else -1 for i in ids]);unseen=int((y<0).sum())
   for name,q in [('full',np.eye(512)),('semantic',Q[s]),('random',R[s])]:
    xp=X@q;cp=cs@q;pred=np.empty(len(X),int)
    for lo in range(0,len(X),512):pred[lo:lo+512]=np.argmin(np.sum((xp[lo:lo+512,None,:]-cp[None,:,:])**2,axis=-1),axis=1)
    acc[name]=float((pred==y).mean())
   classifications[r][s]={'count':len(ids),'unseen_value_count':unseen,'accuracy':acc}
 save(OUT/'classification.json',classifications)
 flatcorpus={x['slot']:x for x in load(V5/'outputs/full/corpus.json')};metrics={};results={};bridge={}
 for r in REGIONS:
  for s in SEM:
   target=f'{r}:{s}';ps=[p for p in pairs if p['target']==target];metrics[target]={}
   if not ps:continue
   ia=[p['a'] for p in ps];ib=[p['b'] for p in ps];ri=REGIONS.index(r);delta=np.array(zr[ib,ri],dtype=np.float64)-np.array(zr[ia,ri],dtype=np.float64);E,valid=energies(delta,Q);ER,_=energies(delta,R)
   for tier in ('strict','relaxed'):
    mask=valid&np.array([p['tier']==tier for p in ps]);metrics[target][tier]={'pair_count':int(sum(p['tier']==tier for p in ps)),'factorized':summarize(E,ER,mask,s)}
   EF=RF=EB=RB=None
   if r!='full_body':
    slot=f'{r}_clothing_'+{'color':'color','clothing_type':'type','clothing_length':'length'}[s];si=split['slots'].index(slot);df=np.array(z[ib,si],dtype=np.float64)-np.array(z[ia,si],dtype=np.float64)
    fq={t:flatcorpus[f'{r}_clothing_'+{'color':'color','clothing_type':'type','clothing_length':'length'}[t]]['geometry']['bases']['euclidean'] for t in SEM}
    fr={t:flatcorpus[f'{r}_clothing_'+{'color':'color','clothing_type':'type','clothing_length':'length'}[t]]['geometry']['bases']['random'] for t in SEM}
    EF,vf=energies(df,fq);RF,_=energies(df,fr);EB,vb=energies(df,Q);RB,_=energies(df,R)
    for tier in ('strict','relaxed'):
     m=np.array([p['tier']==tier for p in ps]);metrics[target][tier]['flat_same_pairs']=summarize(EF,RF,vf&m,s);metrics[target][tier]['shared_basis_old_slot_z']=summarize(EB,RB,vb&m,s)
   for k,p in enumerate(ps):
    results[p['pair_id']]={'valid':bool(valid[k]),'semantic':{t:float(E[t][k]) for t in SEM} if valid[k] else None,'random':{t:float(ER[t][k]) for t in SEM} if valid[k] else None,'delta_norm':float(np.linalg.norm(delta[k])),'flat':{t:float(EF[t][k]) for t in SEM} if EF is not None and np.isfinite(EF[s][k]) else None,'shared_basis_old_slot_z':{t:float(EB[t][k]) for t in SEM} if EB is not None and np.isfinite(EB[s][k]) else None}
   # 固定pair中的特定type transition观察，不另行挑pair。
   if s=='clothing_type':
    transitions=collections.defaultdict(list)
    for k,p in enumerate(ps):transitions['<->'.join(sorted((p['value_a'],p['value_b'])))].append(k)
    for tr,ix in transitions.items():
     for tier in ('strict','relaxed'):
      mask=np.array([k in ix and p['tier']==tier for k,p in enumerate(ps)])&valid
      bridge[f'{r}|{tr}|{tier}']=summarize(E,ER,mask,s)
 save(OUT/'metrics.json',metrics);save(OUT/'energy.json',results);save(OUT/'type_transitions.json',bridge)
 # v5.2 exact full-body nonhit cohort：原有pairs不修改，region变化独立标识。
 oldgroups=load(V52/'outputs/groups.json');oldE=load(V51/'outputs/energy.json');oldP={p['pair_id']:p for g in load(V51/'outputs/pairs.json') for p in g['pairs']};dress=[]
 for g in oldgroups:
  pid=g['pair_id']
  if g['target']!='lower_clothing_type' or not g['has_full_body'] or oldE[pid]['semantic_hit']:continue
  p=oldP[pid];end=[];reason=None
  for i in (p['a'],p['b']):
   candidates=[r for r in ('lower','full_body') if mapped[i]['labels'][f'{r}:clothing_type'] is not None]
   if len(candidates)!=1:reason='ambiguous_or_unresolved_endpoint_region';break
   rr=candidates[0]
   if rr=='full_body' and not mapped[i]['full_body_token_indices']:reason='full_body_localization_unresolved';break
   end.append(rr)
  item={'pair_id':pid,'a':p['a'],'b':p['b'],'row_a':p['row_a'],'row_b':p['row_b'],'old_group':g['group'],'old_type_energy':oldE[pid]['semantic']['lower_clothing_type'],'old_color_energy':oldE[pid]['semantic']['lower_clothing_color'],'endpoint_regions':end,'unresolved':reason}
  if not reason:
   d=np.asarray(zr[p['b'],REGIONS.index(end[1])],dtype=np.float64)-np.asarray(zr[p['a'],REGIONS.index(end[0])],dtype=np.float64);e,v=energies(d[None],Q);er,_=energies(d[None],R)
   color_a=mapped[p['a']]['labels'][end[0]+':color'];color_b=mapped[p['b']]['labels'][end[1]+':color']
   item.update({'cross_region':end[0]!=end[1],'color_a':color_a,'color_b':color_b,'color_control_preserved':color_a is not None and color_a==color_b,'semantic':{t:float(e[t][0]) for t in SEM} if v[0] else None,'random':{t:float(er[t][0]) for t in SEM} if v[0] else None})
   si=split['slots'].index('lower_clothing_type');dold=np.array(z[p['b'],si],dtype=np.float64)-np.array(z[p['a'],si],dtype=np.float64);be,bv=energies(dold[None],Q)
   item['shared_basis_old_slot_z']={t:float(be[t][0]) for t in SEM} if bv[0] else None
  dress.append(item)
 assert len(dress)==1392,('v52 failed full-body cohort regression',len(dress))
 save(OUT/'dress_failed_pairs.json',dress);dm={}
 for name,rows in [('all_usable',[x for x in dress if x.get('semantic')]),('color_control_preserved',[x for x in dress if x.get('semantic') and x['color_control_preserved']]),('same_region',[x for x in dress if x.get('semantic') and not x['cross_region']]),('cross_region',[x for x in dress if x.get('semantic') and x['cross_region']])]:
  if not rows:dm[name]={'count':0};continue
  ee={s:np.array([x['semantic'][s] for x in rows]) for s in SEM};rr={s:np.array([x['random'][s] for x in rows]) for s in SEM};dm[name]=summarize(ee,rr,np.ones(len(rows),bool),'clothing_type')
  dm[name]['old_type']=stats([x['old_type_energy'] for x in rows]);dm[name]['old_color']=stats([x['old_color_energy'] for x in rows]);dm[name]['old_margin']=stats([x['old_type_energy']-x['old_color_energy'] for x in rows])
  be={s:np.array([x['shared_basis_old_slot_z'][s] for x in rows]) for s in SEM};dm[name]['basis_only_type']=stats(be['clothing_type']);dm[name]['basis_only_color']=stats(be['color']);dm[name]['basis_only_type_gt_color']=float((be['clothing_type']>be['color']).mean())
 dm['unresolved']=dict(collections.Counter(x['unresolved'] for x in dress if x['unresolved']));dm['original_count']=len(dress);save(OUT/'dress_metrics.json',dm)
 frozen=load(OUT/'selection_frozen.json');assert all(digest(OUT/n)==h for n,h in frozen.items())
 assert all(digest(p)==h for p,h in hashes.items()),'original inputs changed'
 protocol=load(OUT/'protocol.json');assert all(digest(ROOT/n)==h for n,h in protocol['source_sha256'].items());assert digest(ROOT/'cache/region_z.fp32.npy')==protocol['region_cache_sha256']
 import subprocess
 tracked=subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True);staged=subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True);assert not tracked and not staged
 save(OUT/'source_audit.json',{'old_inputs_unchanged':True,'old_hash_count':len(hashes),'selection_unchanged':True,'sources_unchanged':True,'training':False,'IRRA_forward_calls':0,'feature_reextraction_calls':0,'retrieval_calls':0,'region_cache':'derived pooling only from existing H/Z','tracked_diff':tracked,'staged_diff':staged})
 print('evaluation complete',flush=True)
 return corpus,metrics,dm

if __name__=='__main__':
 t=time.time();args=prepare();evaluate(*args);save(OUT/'completed.json',{'elapsed_seconds':time.time()-t,'mode':'cached feasibility','training':False,'full_IRRA_extraction':False});print('done seconds',time.time()-t,flush=True)
