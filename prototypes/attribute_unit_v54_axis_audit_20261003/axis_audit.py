"""v5.4只审计phrase/length语义轴和cached centroid rank，不训练/IRRA forward。"""
import os,sys,re,json,hashlib,collections,time
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','8');os.environ.setdefault('OMP_NUM_THREADS','8');os.environ.setdefault('XDG_CACHE_HOME','/home/lzf/ldx/cache/xdg')
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent;OUT=ROOT/'outputs';V53=REPO/'prototypes/attribute_unit_v53_factorized_ontology_20261003'
sys.path.insert(0,str(V53));import factorized as old
REGIONS=['head','upper','lower','full_body']
# 固定nominal inventory：正式alias的lexical head，加显式实际出现的其它服装名；不把词形当collapsed canonical。
LEX={
 't_shirt':('upper',[r't[ -]?shirts?',r'tshirts?',r'tee(?:[ -]+shirts?)?']),
 'shirt':('upper',[r'shirts?',r'dress[ -]+shirts?']), 'blouse':('upper',[r'blouses?']), 'polo':('upper',[r'polo(?:[ -]+shirts?)?']), 'jersey':('upper',[r'jerseys?']),
 'jacket':('upper',[r'jackets?']), 'coat':('upper',[r'coats?']), 'blazer':('upper',[r'blazers?']), 'overcoat':('upper',[r'overcoats?']), 'parka':('upper',[r'parkas?']), 'windbreaker':('upper',[r'windbreakers?']),
 'hoodie':('upper',[r'hoodies?']), 'sweater':('upper',[r'sweaters?']), 'sweatshirt':('upper',[r'sweat[ -]?shirts?']), 'jumper':('upper',[r'jumpers?']), 'cardigan':('upper',[r'cardigans?']), 'pullover':('upper',[r'pullovers?']),
 'vest':('upper',[r'vests?']), 'tank_top':('upper',[r'tank[ -]+tops?']), 'top_unspecified':('upper',[r'(?:sleeveless[ -]+)?tops?',r'upper[ -]+(?:clothes|clothing)']), 'tunic':('upper',[r'tunics?']), 'waistcoat':('upper',[r'waistcoats?']), 'poncho':('upper',[r'ponchos?']),
 'pants_trousers':('lower',[r'pants',r'trousers?',r'dress[ -]+pants',r'dress[ -]+trousers?']), 'jeans':('lower',[r'jeans?']), 'shorts':('lower',[r'shorts',r'(?:jeans?|denim)[ -]+shorts']), 'skirt':('lower',[r'skirts?']),
 'slacks':('lower',[r'slacks']), 'leggings':('lower',[r'leggings']), 'tights':('lower',[r'tights']), 'capri':('lower',[r'capris?(?:[ -]+pants)?']), 'khakis':('lower',[r'khakis']), 'bottoms_unspecified':('lower',[r'bottoms',r'lower[ -]+(?:clothes|clothing)']),
 'dress':('full_body',[r'dresses',r'dress']), 'gown':('full_body',[r'(?:dressing[ -]+|night[ -]?)?gowns?']), 'jumpsuit':('full_body',[r'jump[ -]?suits?']), 'romper':('full_body',[r'rompers?']), 'overalls':('full_body',[r'overalls']),
 'robe':('unresolved_region',[r'(?:bath[ -]?)?robes?']), 'uniform':('unresolved_region',[r'uniforms?']), 'suit':('unresolved_region',[r'suits?']),
 'one_piece':('full_body',[r'one[ -]+piece']), 'dress_shoes':('not_clothing_type',[r'dress[ -]+shoes?'])}
PATTERNS=[(k,region,re.compile(r'(?<!\w)(?:'+ '|'.join(patterns)+r')(?!\w)',re.I)) for k,(region,patterns) in LEX.items()]
SLEEVE=re.compile(r'\b(?:short|long)(?:[ -]+[a-z]+){0,2}[ -]+sleeve[sd]?\b',re.I)
HAIR=re.compile(r'\b(?:short|long)(?:[ -]+[a-z]+){0,3}[ -]+hair(?:ed)?\b',re.I)
LANDMARK=re.compile(r'\b(?:knee|ankle|thigh|calf|floor|waist|hip|mid|full)[ -]+length\b',re.I)
BARRIER=re.compile(r'\b(?:hair|haircut|head|face|boobs|boots?|shoes?|socks?|bags?|backpacks?|purses?|ties?)\b',re.I)
OUTER={'jacket','coat','blazer','overcoat','parka','windbreaker','vest','cardigan','tunic','waistcoat'}
THRESHOLD={'train_captions':50,'train_pids':10,'validation_captions':10,'validation_pids':5}
load=old.load;save=old.save;digest=old.digest;checked=old.checked;key=old.key

def lexical(text):
 candidates=[]
 for val,region,pat in PATTERNS:
  for m in pat.finditer(text):candidates.append({'value':val,'region':region,'raw':m.group(),'start':m.start(),'end':m.end()})
 # 最长span先占据字符，防止t-shirt被计为shirt、sweatshirt被shirt、dress pants被dress。
 selected=[]
 for m in sorted(candidates,key=lambda x:(-(x['end']-x['start']),x['start'],x['value'])):
  if not any(m['start']<x['end'] and x['start']<m['end'] for x in selected):selected.append(m)
 result=[]
 for m in sorted(selected,key=lambda x:(x['start'],x['end'])):
  if m['value']=='one_piece':
   if re.match(r'\s+of\b',text[m['end']:],re.I):continue
   m['role']='construction'
  elif m['value']=='dress_shoes':m['role']='excluded_compound'
  elif m['value']=='top_unspecified' and re.search(r'\bon\s*$',text[max(0,m['start']-8):m['start']],re.I):m['role']='location_word'
  else:m['role']='head'
  before=text[max(0,m['start']-65):m['start']]
  actions=list(re.finditer(r'\b(?:wearing|wears|wear|dressed|carrying|carries|holding|holds)\b',before,re.I))
  last=actions[-1].group().lower() if actions else None
  m['scope_risk']=(last in ('carrying','carries','holding','holds')) or bool(re.search(r'\b(?:without|not|no)\b[^.!?;,]*$',before,re.I))
  result.append(m)
 for m in result:
  if m['value']=='jeans' and m['role']=='head':
   following=[g for g in result if g['role']=='head' and g['start']>=m['end'] and g['value']!='jeans']
   if following:
    g=following[0];gap=text[m['end']:g['start']]
    if (not gap.strip() or m['raw'].lower()=='jean' and len(gap.split())<=2) and not re.search(r'[.!?;,]|\b(?:and|with|over|under)\b',gap,re.I):m['role']='material_modifier'
 return result

def object_head(text,obj):
 if not checked(text,obj):return None,'bad_object_span'
 lo=max(0,obj['start']-20);hi=min(len(text),obj['end']+24);context=lexical(text[lo:hi]);cover=[g for g in context if g['start']<=obj['start']-lo and g['end']>=obj['end']-lo and g['role']=='head']
 ms=cover if len(cover)==1 else [m for m in lexical(obj['raw']) if m['role']=='head']
 if len(ms)!=1:return None,'unknown_or_multihead_object'
 m=ms[0]
 if m['value']=='jeans':
  tail=text[obj['end']:obj['end']+35];ng=[g for g in lexical(tail) if g['role']=='head']
  if ng and (not tail[:ng[0]['start']].strip() or m['raw'].lower()=='jean' and len(tail[:ng[0]['start']].split())<=2) and not re.search(r'[.!?;,]|\b(?:and|with|over|under)\b',tail[:ng[0]['start']],re.I):return m,'linked_modifier_not_head'
 if m['region']=='unresolved_region':return m,'unresolved_object_region'
 if m['value']=='top_unspecified' and re.search(r'\bon\s*$',text[max(0,obj['start']-8):obj['start']],re.I):return m,'spatial_top'
 return m,None

def classify_length(text,slot,canonical,m):
 obj=m.get('linked_object');record={'source_slot':slot,'canonical_value':canonical,'mention':m,'axis':'ambiguous','region':'unresolved_region','value':canonical,'garment':'unknown','binding':'unresolved','reason':None}
 if not checked(text,m):record['reason']='bad_evidence_span';return record
 if slot=='hair_length':
  raw=m['raw'].lower();v='bald' if re.search(r'\bbald\b',raw) else 'shoulder_length' if re.search(r'\bshoulder[ -]+length\b',raw) else 'medium' if re.search(r'\bmedium\b',raw) else 'short' if re.search(r'\bshort\b',raw) else 'long' if re.search(r'\blong\b',raw) else None
  if v is None:record['reason']='unknown_hair_value';return record
  record.update(axis='hair_state' if v=='bald' else 'hair_landmark' if v=='shoulder_length' else 'hair_length',region='head',value=v,garment='hair',binding='valid',reason='original_hair_provenance');return record
 h,reason=object_head(text,obj);snippet=text[m['start']:min(len(text),m['end']+45)]
 sleeve=next((x for x in SLEEVE.finditer(snippet) if x.start()==0),None);hair=next((x for x in HAIR.finditer(snippet) if x.start()==0),None)
 if h:record.update(region=h['region'],garment=h['value'])
 if hair:
  record.update(axis='hair_length',region='head',binding='conflict',reason='hair_evidence_bound_to_clothing');return record
 if sleeve:
  mid=sleeve.group().split()[1:-1]
  if any(x.lower() in ('with','and','in','over','under','wearing') for x in mid):sleeve=None
 if sleeve:
  record['axis']='sleeve_length'
  if h and h['region']=='lower':record['region']='unresolved_region'
 if reason:
  record['reason']=reason;return record
 # 弱/跨对象绑定不改变明确的sleeve语义，但不进入candidate axis corpus。
 if m['end']>obj['start']:record['reason']='postposed_or_overlapping_binding';return record
 gap=text[m['end']:obj['start']];gap_heads=[x for x in lexical(gap) if x['role']=='head'];gap_heads=[x for x in gap_heads if not(x['value']=='jeans' and h['value']=='shorts')]
 if len(gap.split())>8 or BARRIER.search(gap) or gap_heads or re.search(r'[.!?;]|\band\b|\b(?:wearing|wears|with)\b',gap,re.I):record['reason']='cross_object_or_weak_binding';return record
 if sleeve:
  if h['region']=='lower':record['reason']='sleeve_evidence_bound_to_lower_garment';record['binding']='conflict';record['region']='unresolved_region';return record
  record.update(binding='valid',reason='explicit_sleeve_noun');return record
 if h['value'] in ('dress','gown','skirt'):
  record.update(axis='hem_extent',binding='valid',reason='direct_short_long_hem_descriptor');return record
 if h['region']=='lower':
  record.update(axis='lower_garment_length',binding='valid',reason='direct_leg_extent_descriptor');return record
 if h['value'] in OUTER or re.search(r'\b(?:body|hem|length|waist|hip|cropp)\w*\b',gap,re.I):
  record.update(axis='garment_body_length',binding='valid',reason='direct_outer_or_explicit_body_extent_descriptor');return record
 record['reason']='unmarked_upper_length_axis';return record

def raw_landmarks(text,heads):
 result=[]
 for m in LANDMARK.finditer(text):
  nearby=[]
  for g in heads:
   if g['role']!='head' or g['region'] not in ('upper','lower','full_body'):continue
   lo=min(m.end(),g['end']);hi=max(m.start(),g['start']);gap=text[lo:hi] if hi>=lo else ''
   if len(gap.split())>6 or BARRIER.search(gap) or re.search(r'[.!?;,]|\band\b',gap,re.I):continue
   if any(x is not g and x['role']=='head' and min(m.start(),g['start'])<x['start']<max(m.start(),g['start']) for x in heads):continue
   nearby.append((abs(g['start']-m.start()),g))
  nearby.sort(key=lambda x:(x[0],x[1]['start']))
  record={'source_slot':'raw_landmark','canonical_value':None,'mention':{'raw':m.group(),'start':m.start(),'end':m.end()},'axis':'ambiguous','region':'unresolved_region','value':re.sub(r'[ -]+','_',m.group().lower()),'garment':'unknown','binding':'unresolved','reason':'no_unique_nearby_garment'}
  if nearby and (len(nearby)==1 or nearby[0][0]!=nearby[1][0]):
   g=nearby[0][1];value=record['value'];record.update(region=g['region'],garment=g['value'],linked_object=g)
   if g['scope_risk']:record['reason']='carried_or_negated_object'
   elif g['value'] in ('dress','gown','skirt') and value in ('knee_length','ankle_length','thigh_length','calf_length','floor_length'):
    record.update(axis='hem_landmark',binding='valid',reason='explicit_landmark_near_hem_garment')
   elif g['region']=='upper' and value in ('knee_length','thigh_length','waist_length','hip_length'):
    record.update(axis='body_landmark',binding='valid',reason='explicit_upper_body_landmark')
   elif g['region']=='lower' and value in ('knee_length','ankle_length','calf_length'):
    record.update(axis='leg_landmark',binding='valid',reason='explicit_leg_landmark')
   else:record['reason']='non_anatomical_or_uncertain_length_landmark'
  result.append(record)
 return result

def anchor(m,canonical):
 raw=m['raw'];x=re.search(r'\b(?:short|long)\b',raw,re.I);return m['start']+(x.start() if x else 0)

def counter_stats(rows):
 return {'mentions':len(rows),'distinct_captions':len({x['row_id'] for x in rows}),'distinct_pids':len({x['pid'] for x in rows})}

def between_rank(c):
 c=np.array(c,dtype=np.float64);center=c.mean(0);C=c-center;_,s,vh=np.linalg.svd(C,full_matrices=False);tol=max(C.shape)*np.finfo(np.float64).eps*(s[0] if len(s) else 0);rank=int((s>tol).sum())
 return {'rank':rank,'singular_values':s.tolist(),'tolerance':float(tol),'centroids':c.tolist(),'center':center.tolist(),'Q':vh[:rank].T.tolist()}


def run():
 start=time.time();protocol53=load(V53/'outputs/protocol.json');hashes=dict(protocol53['old_sha256'])
 for n in ['protocol.json','final_audit.json','coverage.json','localization_audit.json','selection_frozen.json','mapped_labels.json']:hashes[str(V53/'outputs'/n)]=digest(V53/'outputs'/n)
 hashes[str(V53/'cache/region_z.fp32.npy')]=protocol53['region_cache_sha256']
 for n,h in protocol53['source_sha256'].items():hashes[str(V53/n)]=h
 for path,h in hashes.items():assert digest(path)==h,('old source mismatch',path)
 split=load(old.V4/'outputs/split.json');lookup={x['row_id']:i for i,x in enumerate(split['records'])};train=set(split['train_indices']);val=set(split['validation_indices']);assert not train&val
 assert not set(split['train_pids'])&set(split['validation_pids']);records=[None]*len(lookup)
 with old.LABELS.open() as fh:
  for l in fh:
   a=json.loads(l);i=lookup.get(a['row_id'])
   if i is None:continue
   assert a['caption']==split['records'][i]['caption'];records[i]=a
 assert all(records);inventory=[];length=[];candidates=[];denom=collections.defaultdict(set)
 def add_candidate(i,region,axis,value,source,obj=None,qualified=True,reason=None):
  field=region+':'+axis;denom[field].add(i)
  candidates.append({'i':i,'row_id':records[i]['row_id'],'pid':split['records'][i]['pid'],'split':'train' if i in train else 'validation','region':region,'axis':axis,'value':value,'source':source,'object':obj,'qualified':qualified,'reason':reason})
 for i,a in enumerate(records):
  text=a['caption'];heads=lexical(text);part='train' if i in train else 'validation';pid=split['records'][i]['pid'];pr=a.get('provenance') or {}
  for g in heads:
   cm=[];linked=[]
   for slot,p in pr.items():
    for m in (p or {}).get('mentions',[]):
     if slot in ('upper_clothing_type','lower_clothing_type') and checked(text,m) and m['start']<g['end'] and g['start']<m['end']:cm.append(slot+'='+p['canonical'])
     ob=m.get('linked_object')
     if checked(text,ob) and ob['start']<g['end'] and g['start']<ob['end']:linked.append(slot+'='+p['canonical'])
   inventory.append(dict(g,row_id=a['row_id'],pid=pid,i=i,split=part,canonical_mapping=cm,linked_mapping=linked))
  # type候选优先原selected type provenance；fullbody未知词允许明确raw-only head，但保留来源。
  type_heads=[]
  for slot in ('upper_clothing_type','lower_clothing_type'):
   if a['attributes'][slot]=='null':continue
   for m in (pr.get(slot) or {}).get('mentions',[]):
    h,rr=object_head(text,m)
    if h and h['role']=='head':
     h=dict(h,start=m['start'],end=m['end'],raw=m['raw'])
     risk=next((g['scope_risk'] for g in heads if g['start']<=m['start']<g['end']),False)
     add_candidate(i,h['region'],'garment_type',h['value'],slot,m,qualified=rr is None and not risk and not h['value'].endswith('_unspecified'),reason=rr or ('carried_or_negated' if risk else 'generic_type' if h['value'].endswith('_unspecified') else None));type_heads.append((h['start'],h['end']))
    else:add_candidate(i,'unresolved_region','garment_type',None,slot,m,False,rr or 'unknown_head')
  for g in heads:
   if g['role']=='head' and g['region'] in ('upper','lower') and a['attributes'][g['region']+'_clothing_type']=='null':
    add_candidate(i,g['region'],'garment_type',g['value'],'raw_only_when_primary_type_missing',g,not g['scope_risk'] and not g['value'].endswith('_unspecified'),'scope_risk' if g['scope_risk'] else 'generic_type' if g['value'].endswith('_unspecified') else None)
   if g['region']=='full_body' and g['role']=='head' and not any(x<g['end'] and g['start']<y for x,y in type_heads):add_candidate(i,'full_body','garment_type',g['value'],'raw_only_full_body',g,not g['scope_risk'],'carried_or_negated' if g['scope_risk'] else None)
   if g['role']=='construction':add_candidate(i,'full_body','construction','one_piece','raw_construction',g,not g['scope_risk'],'scope_risk' if g['scope_risk'] else None)
  for slot in ('upper_clothing_length','lower_clothing_length','hair_length'):
   pp=pr.get(slot) or {}
   for m in pp.get('mentions',[]):
    x=classify_length(text,slot,pp['canonical'],m);x.update(row_id=a['row_id'],pid=pid,i=i,split=part,anchor=anchor(m,pp['canonical']));length.append(x)
    # 即使axis可判但binding冲突，也作为候选的unresolved记录，不能偷偷重新绑定到新reader。
    add_candidate(i,x['region'],x['axis'],x['value'],slot,m.get('linked_object'),x['binding']=='valid',x['reason'])
  for x in raw_landmarks(text,heads):
   x.update(row_id=a['row_id'],pid=pid,i=i,split=part,anchor=x['mention']['start']);length.append(x)
   add_candidate(i,x['region'],x['axis'],x['value'],'raw_landmark',x.get('linked_object'),x['binding']=='valid',x['reason'])
 # summary inventories：出现数/caption/PID/direct provenance 各有明确单位。
 table={}
 for value,(region,_) in LEX.items():
  table[value]={'region':region}
  for part in ('train','validation'):
   rows=[x for x in inventory if x['value']==value and x['split']==part];direct=[x for x in rows if x['canonical_mapping']]
   table[value][part]={**counter_stats(rows),'direct_provenance':counter_stats(direct),'raw_forms':dict(collections.Counter(x['raw'].lower() for x in rows)),'canonical_mapping':dict(collections.Counter(y for x in rows for y in set(x['canonical_mapping']))),'linked_mapping':dict(collections.Counter(y for x in rows for y in set(x['linked_mapping']))),'scope_risk_mentions':sum(x['scope_risk'] for x in rows),'roles':dict(collections.Counter(x['role'] for x in rows)),'head_only':counter_stats([x for x in rows if x['role']=='head'])}
 save(OUT/'garment_inventory.json',table);save(OUT/'garment_mentions.json',inventory)
 # 所有原抽取upper/lower length provenance逐条守恒；额外raw landmarks与hair单独标来源。
 original=[x for x in length if x['source_slot'] in ('upper_clothing_length','lower_clothing_length')];expected=sum(len((r.get('provenance',{}).get(s) or {}).get('mentions',[])) for r in records for s in ('upper_clothing_length','lower_clothing_length'))
 assert len(original)==expected
 summary={}
 for axis in sorted({x['axis'] for x in original}):
  summary[axis]={}
  for part in ('train','validation'):
   rows=[x for x in original if x['axis']==axis and x['split']==part]
   summary[axis][part]={**counter_stats(rows),'valid_binding':counter_stats([x for x in rows if x['binding']=='valid']),'values':dict(collections.Counter(x['canonical_value'] for x in rows)),'garments':dict(collections.Counter(x['garment'] for x in rows)),'regions':dict(collections.Counter(x['region'] for x in rows)),'reasons':dict(collections.Counter(x['reason'] for x in rows)),'source_slots':dict(collections.Counter(x['source_slot'] for x in rows))}
 save(OUT/'length_classification.json',{'original_clothing_mentions':expected,'by_axis':summary,'unit':'provenance mentions; distinct captions/PIDs separately; explicit axis with wrong binding retained but excluded from resolved corpus'})
 save(OUT/'length_mentions.json',length)
 conflict={}
 for value in ('short','long'):
  conflict[value]={}
  for part in ('train','validation'):
   rows=[x for x in length if x['canonical_value']==value and x['split']==part and x['source_slot']!='raw_landmark'];unique={}
   for x in rows:unique.setdefault((x['row_id'],x['anchor'],x['canonical_value']),[]).append(x)
   distributions=collections.Counter();dup=0
   for rr in unique.values():
    dup+=int(len(rr)>1);axes=set(x['axis'] for x in rr)
    if 'hair_length' in axes:axis='hair_length'
    elif 'sleeve_length' in axes:axis='sleeve_length'
    elif len(axes-{'ambiguous'})==1:axis=next(iter(axes-{'ambiguous'}))
    else:axis='ambiguous' if axes=={'ambiguous'} else 'conflicting_axes'
    distributions[axis]+=1
   conflict[value][part]={'provenance_mentions':len(rows),'unique_evidence_anchors':len(unique),'duplicated_anchor_count':dup,'unique_anchor_axis_counts':dict(distributions),'source_axis_mentions':dict(collections.Counter(x['source_slot']+'|'+x['axis'] for x in rows)),'hair_raw_value_counts':dict(collections.Counter(x['value'] for x in rows if x['source_slot']=='hair_length'))}
 save(OUT/'short_long_conflict.json',conflict)
 # 每caption/axis保留唯一qualified value；多garment/value冲突unresolved，不凭canonical挑值。
 fields=sorted({x['region']+':'+x['axis'] for x in candidates});frames=[];coverage={}
 for field in fields:
  xs=[x for x in candidates if x['region']+':'+x['axis']==field];by=collections.defaultdict(list)
  for x in xs:by[x['i']].append(x)
  counts={'train':collections.Counter(),'validation':collections.Counter()};why={'train':collections.Counter(),'validation':collections.Counter()}
  for i,rr in sorted(by.items(),key=lambda x:key(records[x[0]]['row_id'])):
   part='train' if i in train else 'validation';valid=[x for x in rr if x['qualified']];values=sorted(set(x['value'] for x in valid if x['value'] is not None));bad=[x for x in rr if not x['qualified']]
   # 同axis有未解决的冲突证据不视为完整label；逐caption审计不改源provenance。
   value=values[0] if len(values)==1 and not bad else None;reason='multiple_values' if len(values)>1 else 'unresolved_evidence' if bad else 'no_qualified_value' if not values else None
   region,axis=field.split(':');counts[part][value or 'unresolved']+=1
   if reason:why[part][reason]+=1
   frames.append({'i':i,'row_id':records[i]['row_id'],'pid':split['records'][i]['pid'],'split':part,'region':region,'axis':axis,'value':value,'reason':reason,'sources':rr})
  coverage[field]={'counts':counts,'unresolved_reasons':why}
 save(OUT/'candidate_frames.json',frames);save(OUT/'candidate_coverage.json',coverage)
 samples=[]
 for axis in sorted({x['axis'] for x in original}):
  for binding in ('valid','unresolved','conflict'):
   for x in sorted([x for x in original if x['axis']==axis and x['binding']==binding],key=lambda x:(key(x['row_id']),x['mention']['start']))[:5]:samples.append({'record':x,'caption':records[x['i']]['caption'],'original_attributes':records[x['i']]['attributes']})
 save(OUT/'fixed_audit_sample.json',samples)
 protocol={'training':False,'IRRA_forward_calls':0,'retrieval_calls':0,'formal_ontology_modified':False,'split':'fixed v4 PID 54246/13700','readiness':THRESHOLD,'readiness_scope':'engineering coverage screen only; >=2 qualified values; no significance/generalization claim; semantic overlap and absence of negatives can veto recommendation','length_rule':'explicit local sleeve/hair noun precedes nominal extent; binding reliability separate; bare shirt/sweater/top short/long ambiguous, outer explicit extent/body descriptor resolved; skirt/dress/gown hem; pants/etc leg; raw anatomical landmark separate from short/long extent','type_rule':'longest exact lexical alias/nominal phrase including compound whole-caption context for partial polo/shirt etc provenance spans; parent vs subtype/qualifier distinguished in proposal; Stage01 primary type provenance for upper/lower; if current primary type null, raw-only specific region head permitted with multi-value/null conflicts retained; raw-only explicit full_body when original alias absent; one-piece construction, shoulder-length hair landmark separated','frame_rule':'one region/axis/value per caption; any unresolved source or multiple values -> unresolved; no filtering by ranks','geometry':'train only value means in original512D existing cache, center equal value centroids, v5 numerical rank; no classifier/control/intervention','old_sha256':hashes,'source_sha256':{n:digest(ROOT/n) for n in ('axis_audit.py','test_axis_audit.py')}}
 save(OUT/'protocol.json',protocol);save(OUT/'classification_frozen.json',{n:digest(OUT/n) for n in ['garment_inventory.json','garment_mentions.json','length_classification.json','length_mentions.json','short_long_conflict.json','candidate_frames.json','candidate_coverage.json','fixed_audit_sample.json','protocol.json']})
 print('text/provenance classification frozen',expected,flush=True)
 geometry(split,records,frames,hashes)
 save(OUT/'completed.json',{'elapsed_seconds':time.time()-start,'cached_only':True,'training':False});print('done',time.time()-start,flush=True)


def geometry(split,records,frames,hashes):
 zr=np.load(V53/'cache/region_z.fp32.npy',mmap_mode='r');z=np.load(old.ZFILE,mmap_mode='r');H=np.load(old.HFILE,mmap_mode='r');m53=load(V53/'outputs/mapped_labels.json');extra={};alignment_fail={};source_counts=collections.Counter()
 sys.path.insert(0,str(old.CODE/'src'));sys.path.insert(0,'/home/lzf/Attack/IRRA-main/IRRA-main')
 from attributes.irra_token_alignment import align_irra_tokens,_span_token_indices
 from utils.simple_tokenizer import SimpleTokenizer
 tokenizer=SimpleTokenizer()
 def representation(i,region):
  if region=='head':source_counts['v4_hair_z']+=1;return np.array(z[i,split['slots'].index('hair_length')],dtype=np.float64)
  if region in ('upper','lower'):source_counts['v53_'+region+'_region_z']+=1;return np.array(zr[i,0 if region=='upper' else 1],dtype=np.float64)
  if region!='full_body':return None
  if m53[i]['full_body_token_indices']:source_counts['v53_full_body_region_z']+=1;return np.array(zr[i,2],dtype=np.float64)
  if i not in extra:
   text=records[i]['caption'];gs=[g for g in lexical(text) if g['region']=='full_body' and g['role']=='head' and not g['scope_risk']];a=align_irra_tokens(text,tokenizer);ix=set();reasons=[]
   for g in gs:
    t,rs=_span_token_indices(text,g,a['all_content_tokens']);ix.update(t);reasons.extend(rs)
   if a['status']!='valid':reasons.extend(a['reasons'])
   if not ix or max(ix,default=0)>split['records'][i]['content_count']:reasons.append('object_missing_or_truncated')
   if reasons:extra[i]=None;alignment_fail[records[i]['row_id']]=sorted(set(reasons))
   else:extra[i]=np.mean(H[i,sorted(ix)],axis=0,dtype=np.float64)
  if extra[i] is not None:source_counts['existing_H_full_body_object_pooling']+=1
  return extra[i]
 summary={};corpus={}
 for field in sorted({x['region']+':'+x['axis'] for x in frames}):
  rows=[x for x in frames if x['region']+':'+x['axis']==field];nonnull=[x for x in rows if x['value'] is not None];values=sorted({x['value'] for x in nonnull});info={};centroids=[];usedvalues=[];cachecov={}
  for value in values:
   d={};ready=True
   for part in ('train','validation'):
    rr=[x for x in nonnull if x['value']==value and x['split']==part];st=counter_stats(rr);d[part]=st
    ready&=st['distinct_captions']>=THRESHOLD[part+'_captions'] and st['distinct_pids']>=THRESHOLD[part+'_pids']
   d['coverage_ready']=bool(ready);info[value]=d
   tr=[x for x in nonnull if x['value']==value and x['split']=='train'];xs=[]
   for x in tr:
    v=representation(x['i'],x['region'])
    if v is not None:xs.append(v)
   cachecov[value]={'train_label_captions':len(tr),'cached_train_vectors':len(xs)}
   if xs:centroids.append(np.mean(xs,axis=0,dtype=np.float64));usedvalues.append(value)
  unresolved={}
  for part in ('train','validation'):
   rr=[x for x in rows if x['split']==part];unresolved[part]={'candidate_captions':len(rr),'unresolved_captions':sum(x['value'] is None for x in rr),'unresolved_ratio':sum(x['value'] is None for x in rr)/len(rr) if rr else None}
  readyvalues=[v for v,d in info.items() if d['coverage_ready']];geo=between_rank(centroids) if centroids else None
  corpus[field]={'values':usedvalues,'cache_coverage':cachecov,'geometry':geo}
  summary[field]={'observed_value_count':len(values),'values':info,'eligible_values':readyvalues,'coverage_ready_axis':len(readyvalues)>=2,'unresolved':unresolved,'cache_available_value_count':len(usedvalues),'effective_rank':geo['rank'] if geo else None,'singular_values':geo['singular_values'] if geo else None,'rank_caveat':'numerical between-centroid rank/data coverage only, no evidence of selectivity or disentanglement'}
 save(OUT/'candidate_axes.json',summary);save(OUT/'centroid_rank.json',corpus);save(OUT/'cache_pooling_audit.json',{'representation_sources':source_counts,'additional_fullbody_cached_pool_caption_count':sum(x is not None for x in extra.values()),'alignment_unresolved':alignment_fail,'IRRA_forward_calls':0,'new_reader_training':False,'rule':'reuse v53 region Z; head uses existing v4 hair Z; unsupported new fullbody noun uses same object-uniform rule on existing H, no new IRRA feature extraction'})
 frozen=load(OUT/'classification_frozen.json');assert all(digest(OUT/n)==h for n,h in frozen.items());assert all(digest(path)==h for path,h in hashes.items());protocol=load(OUT/'protocol.json');assert all(digest(ROOT/n)==h for n,h in protocol['source_sha256'].items())
 import subprocess
 assert not subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True);assert not subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True)
 save(OUT/'source_audit.json',{'old_input_hash_count':len(hashes),'old_inputs_unchanged':True,'classification_frozen_unchanged':True,'source_unchanged':True,'training':False,'IRRA_forward_calls':0,'feature_reextraction_calls':0,'retrieval_calls':0,'official_tracked_staged_clean':True,'all_original_clothing_length_mentions_accounted':True})
 print('rank audit complete',flush=True)

if __name__=='__main__':run()
