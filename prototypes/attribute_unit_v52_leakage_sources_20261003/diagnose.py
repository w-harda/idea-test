"""v5.2：文本/provenance先冻结分组，随后只读v5.1 strict energy；无模型调用。"""
import sys,json,re,hashlib,collections,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent;OUT=ROOT/'outputs'
V51=REPO/'prototypes/attribute_unit_v51_selectivity_20261003';OLD=V51/'outputs'
V4=REPO/'prototypes/attribute_unit_v4_learned_reader_20261003'
LABELS=Path('/home/lzf/ldx/outputs/idea-TBPS-test1/stage03/cuhk_all_scores.jsonl')
TARGETS=['lower_clothing_type','upper_clothing_type']
# 词表在读取energy前固定，只作审计分组，不修改ontology/标签/representation。
FULL=re.compile(r'\b(?:dress(?:es)?(?!\s+(?:shirts?|pants|trousers|shoes|boots|code|style|jackets?|coats?)\b)|gowns?|jumpsuits?|rompers?|onesies?|overalls?|coveralls?|bodysuits?|body\s+suits?|one[-\s]+piece|full[-\s]+body\s+(?:clothing|garment|outfit))\b',re.I)
LOWER={'trousers_shorts':r'\b(?:pants|trousers|jeans?|shorts|slacks|leggings|tights|capris?|joggers|chinos|sweatpants|trackpants|breeches)\b','skirt_dress':r'\bskirts?\b'}
UPPER={'t_shirt_shirt':r'\b(?:shirts?|t[-\s\x27]?shirts?|tees?|blouses?|polos?|jerseys?)\b','hoodie_sweater':r'\b(?:hoodies?|sweaters?|sweatshirts?|cardigans?|pullovers?|jumpers?|turtlenecks?)\b','jacket_coat':r'\b(?:jackets?|coats?|blazers?|parkas?|windbreakers?)\b','vest_sleeveless':r'\b(?:vests?|tanks?|singlets?|camisoles?)\b'}
COLORS=re.compile(r'\b(?:black|blue|brown|green|gr[ae]y|orange|pink|purple|red|white|yellow|tan|beige|khaki)\b',re.I)
DISTRACTOR=re.compile(r'\b(?:shoes?|boots?|sandals?|slippers?|bags?|backpacks?|purses?|hats?)\b',re.I)

def load(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rowkey(p):return tuple(int(x) for x in p.split(':')[1:])
def mentions(x,slot):return (x['provenance'].get(slot) or {}).get('mentions',[])
def span(x):return (x.get('start'),x.get('end'))
def valid_span(text,m):return isinstance(m.get('start'),int) and isinstance(m.get('end'),int) and 0<=m['start']<m['end']<=len(text) and text[m['start']:m['end']]==m.get('raw')
def lower_family(raw):
 if FULL.search(raw):return 'full_body'
 for val,rx in LOWER.items():
  if re.search(rx,raw,re.I):return val
 return None
def upper_family(raw):
 for val,rx in UPPER.items():
  if re.search(rx,raw,re.I):return val
 return None
def classify_caption(x,region):
 text=x['caption'];typ=region+'_clothing_type';color=region+'_clothing_color';other=('upper' if region=='lower' else 'lower')+'_clothing_color';tm=mentions(x,typ);cm=mentions(x,color);value=x['attributes'].get(typ);flags=set();full=[{'raw':m.group(),'start':m.start(),'end':m.end()} for m in FULL.finditer(text)]
 if value in (None,'null') or not tm:flags.add('null_or_missing_type_provenance')
 if x['attributes'].get(color) in (None,'null') or not cm:flags.add('null_or_missing_color_provenance')
 if not x.get('provenance',{}).get(typ) or (x['provenance'][typ].get('canonical')!=value):flags.add('type_provenance_canonical_mismatch')
 if cm and x['provenance'][color].get('canonical')!=x['attributes'][color]:flags.add('color_provenance_canonical_mismatch')
 expected=LOWER if region=='lower' else UPPER
 recognized=[]
 for t in tm:
  if not valid_span(text,t):flags.add('invalid_type_span');continue
  fam=lower_family(t['raw']) if region=='lower' else upper_family(t['raw']);recognized.append(fam)
  if fam is None:flags.add('weak_or_unrecognized_type_object')
  elif fam=='full_body':
   if value!='skirt_dress':flags.add('type_object_value_mismatch')
  elif fam!=value:flags.add('type_object_value_mismatch')
 # 同region不同canonical garment family是叠穿/对象多重风险；重复同一对象不因重复词被算多件。
 families={v for v,rx in expected.items() if re.search(rx,text,re.I)}
 if region=='lower' and full:families.add('full_body')
 if region=='upper':
  for gm in re.finditer(r'\btops?\b(?!\s+hat\b)',text,re.I):
   if not re.search(r'(?:on|at)\s+(?:the\s+)?$',text[:gm.start()],re.I):families.add('weak_generic_top')
 if len(families)>1:flags.add('multiple_same_region_garment_families')
 if full and any(re.search(rx,text,re.I) for rx in UPPER.values()):flags.add('full_body_plus_upper_garment_layering')
 type_spans={span(t) for t in tm};other_evidence={span(t) for t in mentions(x,other)}
 for c in cm:
  if not valid_span(text,c):flags.add('invalid_color_span');continue
  obj=c.get('linked_object')
  if not obj:flags.add('missing_color_linked_object');continue
  if not valid_span(text,obj):flags.add('invalid_linked_object_span');continue
  if span(obj) not in type_spans:flags.add('color_object_not_in_selected_type_provenance')
  fam=lower_family(obj['raw']) if region=='lower' else upper_family(obj['raw'])
  if fam is None:flags.add('color_linked_object_wrong_or_weak_region')
  elif fam not in (value,'full_body'):flags.add('color_linked_object_value_mismatch')
  if span(c) in other_evidence:flags.add('color_evidence_shared_between_regions')
  lo=min(c['start'],obj['start']);hi=max(c['end'],obj['end']);between=text[lo:hi]
  if DISTRACTOR.search(between):flags.add('color_binding_crosses_accessory_object')
  for rx in list(LOWER.values())+list(UPPER.values())+[FULL.pattern]:
   for gm in re.finditer(rx,between,re.I):
    gs,ge=lo+gm.start(),lo+gm.end()
    if not (obj['start']<=gs and ge<=obj['end']):flags.add('color_binding_crosses_other_garment')
  if re.search(r'[.;]',between):flags.add('color_binding_crosses_sentence_boundary')
  # 若颜色-对象间有多个规范色（grey/gray、tan/beige/khaki合并）则是多色binding风险。
  cols={ {'gray':'grey','tan':'brown','beige':'brown','khaki':'brown'}.get(v.lower(),v.lower()) for v in COLORS.findall(between)}
  if len(cols)>1:flags.add('multiple_colors_in_color_object_span')
  suffix=re.split(r'[.;]',text[c['end']:],maxsplit=1)[0]
  if region=='lower' and re.match(r'\s+(?:on\s+(?:the\s+)?top|on\s+(?:the\s+)?upper|at\s+(?:the\s+)?top)\b',suffix,re.I):flags.add('lower_color_explicit_upper_region_qualifier')
  if region=='upper' and re.match(r'\s+(?:from\s+(?:the\s+)?waist\s+down|on\s+(?:the\s+)?bottom)\b',suffix,re.I):flags.add('upper_color_explicit_lower_region_qualifier')
  if re.search(r'\b(?:multi[-\s]?colou?red|multicolou?r|two[-\s]+tone)\b',between,re.I):flags.add('multicolor_binding')
 clear=bool(tm) and all(f in expected for f in recognized) and len(recognized)==len(tm) and not full
 group='C_binding_risk' if flags or not (clear or full) else ('B_full_body' if full else 'A_clear_region')
 return {'row_id':x['row_id'],'region':region,'group':group,'flags':sorted(flags),'has_full_body':bool(full),'full_body_mentions':full,'recognized_type_families':recognized,'clear_region_objects':clear,'type':value,'color':x['attributes'].get(color),'type_provenance':x['provenance'].get(typ),'color_provenance':x['provenance'].get(color)}
def pair_group(a,b):
 if a['group']=='C_binding_risk' or b['group']=='C_binding_risk':return 'C_binding_risk'
 if a['has_full_body'] or b['has_full_body']:return 'B_full_body'
 return 'A_clear_region'
def prepare():
 assert not (OUT/'groups.json').exists(),'refuse overwrite frozen grouping'
 prot=load(OLD/'protocol.json');done=load(OLD/'completed.json');frozen=load(OLD/'pairing_frozen.json')
 paths=[OLD/n for n in ['protocol.json','completed.json','pairing_frozen.json','pairs.json','energy.json','metrics.json','audit.json']]+[V4/'outputs/split.json',LABELS,REPO/'prototypes/attribute_unit_v5_contextual_subspace_20261003/outputs/full/corpus.json']
 # 此处energy只hash，不读取任何selectivity数值。
 hashes={str(p):digest(p) for p in paths}
 assert hashes[str(OLD/'pairs.json')]==frozen['pairs_sha256'];assert hashes[str(LABELS)]==prot['baseline_sha256'][str(LABELS)]
 for name,h in done['source_sha256'].items():assert digest(V51/name)==h
 for p in [V4/'outputs/split.json',REPO/'prototypes/attribute_unit_v5_contextual_subspace_20261003/outputs/full/corpus.json']:assert hashes[str(p)]==prot['baseline_sha256'][str(p)]
 split=load(V4/'outputs/split.json');records=split['records'];val=set(split['validation_indices']);pairs=[p for g in load(OLD/'pairs.json') if g['target'] in TARGETS for p in g['pairs'] if p['tier']=='strict'];assert len(pairs)==8705+992
 needed={p[k] for p in pairs for k in ['row_a','row_b']};labels={}
 with LABELS.open() as f:
  for line in f:
   x=json.loads(line)
   if x['row_id'] in needed:labels[x['row_id']]={k:x[k] for k in ['row_id','caption','attributes','provenance']}
 assert set(labels)==needed
 cc={};groups=[]
 for p in pairs:
  region=p['target'].split('_')[0]
  for k,index in [('row_a','a'),('row_b','b')]:
   rid=p[k];i=p[index];assert i in val and records[i]['row_id']==rid and records[i]['caption']==labels[rid]['caption']
   expected={slot:split['values'][j][lab] for j,(slot,lab) in enumerate(zip(split['slots'],records[i]['labels']))};assert expected==labels[rid]['attributes']
   if (rid,region) not in cc:cc[(rid,region)]=classify_caption(labels[rid],region)
  a,b=cc[(p['row_a'],region)],cc[(p['row_b'],region)];groups.append({'pair_id':p['pair_id'],'target':p['target'],'group':pair_group(a,b),'has_full_body':a['has_full_body'] or b['has_full_body'],'flags':sorted(set(a['flags']+b['flags'])),'a':a,'b':b,'non_target_hamming':p['non_target_hamming'],'same_pid':p['same_pid']})
 save(OUT/'caption_records.json',labels);save(OUT/'groups.json',groups)
 protocol={'targets':TARGETS,'lower_strict':8705,'upper_strict':992,'pair_source':'unchanged v5.1 strict only','classification':'deterministic raw caption + original Stage01 provenance, no energy access','precedence':'C risk > B full-body > A clear region; dress/all full-body flag also independently retained','full_lexicon':FULL.pattern,'lower_lexicon':LOWER,'upper_lexicon':UPPER,'flags':'provenance/span/canonical mismatch; color-linked-object not in type provenance; unrecognized or weak region; multi same-region families; full-body+upper layering; exact evidence shared across regions; accessory/other-garment/sentence crossing; multi-colors in color-object span; explicit wrong-region qualifier','strict_risk_note':'C is conservative risk, not verified error; other null slots not automatically C because current strict type/color known','dress_shirt_pants_note':'dress modifier for shirt/pants/shoes etc excluded; dressed excluded; generic two-piece suit not automatically full-body','sample_rule':'after frozen grouping, first10 hits and first10 nonhits in lower strict numeric(row_a,row_b); upper first3/3 same rule; no energy magnitude ranking; main group statistics keep every pair','training':False,'IRRA_forward_calls':0,'feature_extraction_calls':0,'retrieval_calls':0,'old_sha256':hashes,'source_sha256':{p.name:digest(p) for p in sorted(ROOT.glob('*.py'))}}
 save(OUT/'protocol.json',protocol);save(OUT/'groups_frozen.json',{'groups_sha256':digest(OUT/'groups.json'),'caption_records_sha256':digest(OUT/'caption_records.json'),'protocol_sha256':digest(OUT/'protocol.json'),'energy_values_read':False})
 for target in TARGETS:
  gg=[g for g in groups if g['target']==target];print('GROUPS',target,dict(collections.Counter(g['group'] for g in gg)),'full_body',sum(g['has_full_body'] for g in gg),'flags',dict(collections.Counter(f for g in gg for f in g['flags'])),flush=True)
 print('GROUPING_FROZEN',OUT,flush=True)


def summarize(gs,energy,target):
 import numpy as np
 if not gs:return {'count':0}
 region=target.split('_')[0];color=region+'_clothing_color'
 def one(method):
  et=np.array([energy[g['pair_id']][method][target] for g in gs]);ec=np.array([energy[g['pair_id']][method][color] for g in gs]);margin=et-ec
  stats=lambda v:{'median':float(np.median(v)),'mean':float(v.mean()),'min':float(v.min()),'max':float(v.max())}
  return {'E_type':stats(et),'E_color':stats(ec),'margin':stats(margin),'hit_rate':float((margin>0).mean()),'all_original_sibling_hit_rate':float(np.mean([energy[g['pair_id']][method+'_hit'] for g in gs]))}
 same_rank=float(np.mean([energy[g['pair_id']]['semantic'][target]>energy[g['pair_id']]['random'][target] for g in gs]))
 reuse=collections.Counter(r['row_id'] for g in gs for r in [g['a'],g['b']]);h=[g for g in gs if g['non_target_hamming']==0]
 return {'count':len(gs),'semantic':one('semantic'),'random':one('random'),'type_gt_same_rank_random_rate':same_rank,'H0_count':len(h),'H0_semantic_hit_rate':None if not h else float(np.mean([energy[g['pair_id']]['semantic'][target]>energy[g['pair_id']]['semantic'][color] for g in h])),'same_pid_count':sum(g['same_pid'] for g in gs),'unique_endpoints':len(reuse),'maximum_endpoint_reuse':max(reuse.values())}
def verify_frozen():
 f=load(OUT/'groups_frozen.json');p=load(OUT/'protocol.json')
 for name,k in [('groups.json','groups_sha256'),('caption_records.json','caption_records_sha256'),('protocol.json','protocol_sha256')]:assert digest(OUT/name)==f[k]
 for name,h in p['source_sha256'].items():assert digest(ROOT/name)==h,'source changed after grouping freeze'
 for path,h in p['old_sha256'].items():assert digest(path)==h,'old artifact changed'
 return p
def measure():
 assert not (OUT/'metrics.json').exists(),'refuse overwrite group measurement';p=verify_frozen();groups=load(OUT/'groups.json');records=load(OUT/'caption_records.json');energy=load(OLD/'energy.json');metrics={};samples=[]
 for target in TARGETS:
  gs=[g for g in groups if g['target']==target];assert all(energy[g['pair_id']]['valid'] for g in gs)
  parts={'all':gs,**{group:[g for g in gs if g['group']==group] for group in ['A_clear_region','B_full_body','C_binding_risk']},'contains_full_body':[g for g in gs if g['has_full_body']],'no_full_body':[g for g in gs if not g['has_full_body']],'full_body_with_risk':[g for g in gs if g['has_full_body'] and g['group']=='C_binding_risk'],'no_full_body_with_risk':[g for g in gs if not g['has_full_body'] and g['group']=='C_binding_risk']}
  metrics[target]={k:summarize(x,energy,target) for k,x in parts.items()}
  n=10 if target.startswith('lower') else 3
  for hit in [True,False]:
   ordered=sorted([g for g in gs if energy[g['pair_id']]['semantic'][target]>energy[g['pair_id']]['semantic'][target.replace('_type','_color')]],key=lambda g:(rowkey(g['a']['row_id']),rowkey(g['b']['row_id']))) if hit else sorted([g for g in gs if not energy[g['pair_id']]['semantic'][target]>energy[g['pair_id']]['semantic'][target.replace('_type','_color')]],key=lambda g:(rowkey(g['a']['row_id']),rowkey(g['b']['row_id'])))
   for g in ordered[:n]:samples.append({**g,'sample_hit':hit,'energy':energy[g['pair_id']]})
  print('RESULT',target,json.dumps(metrics[target],ensure_ascii=False),flush=True)
 save(OUT/'metrics.json',metrics);save(OUT/'binding_audit_sample.json',samples)
 lines=['# Stage03 v5.2 固定row_id成功/失败审计','','分组在读取energy数值前冻结；下衣按numeric pair(row_a,row_b)分别取前10 hit/非hit，上衣简化为前3/3；按type>color定义hit，不按margin幅度挑选。主分组统计包含全部strict pairs；人工判断不改标签、样本或group。人工审计仅表示文本判读，不能单独证明representation leakage的因果来源。','']
 for g in samples:
  target=g['target'];region=target.split('_')[0];color=region+'_clothing_color';en=g['energy'];lines += [f"## {g['pair_id']} ({'成功' if g['sample_hit'] else '失败'})",'',f"group={g['group']}；has_full_body={g['has_full_body']}；flags={g['flags']}；Hamming={g['non_target_hamming']}",'']
  for k in ['a','b']:
   c=g[k];x=records[c['row_id']];lines += [f"### {k.upper()} {c['row_id']}",'',x['caption'],'',f"{target}={c['type']}；{color}={c['color']}",'','type provenance：','```json',json.dumps(c['type_provenance'],ensure_ascii=False,indent=2),'```','color provenance / linked_object：','```json',json.dumps(c['color_provenance'],ensure_ascii=False,indent=2),'```','']
  lines += [f"semantic type={en['semantic'][target]:.6f}；color={en['semantic'][color]:.6f}；M={en['semantic'][target]-en['semantic'][color]:.6f}",f"same-rank random type={en['random'][target]:.6f}；color={en['random'][color]:.6f}",'']
 (OUT/'binding_audit.md').write_text('\n'.join(lines),encoding='utf-8')
 verify_frozen();assert digest(OUT/'groups.json')==load(OUT/'groups_frozen.json')['groups_sha256']
 tracked=subprocess.check_output(['git','diff','--stat'],cwd=REPO,text=True);staged=subprocess.check_output(['git','diff','--cached','--stat'],cwd=REPO,text=True);assert tracked==staged==''
 save(OUT/'source_audit.json',{'old_artifacts_exact_unchanged':True,'old_artifact_count':len(p['old_sha256']),'groups_unchanged':True,'source_unchanged':True,'tracked_diff':tracked,'staged_diff':staged,'training':False,'IRRA_forward_calls':0,'new_feature_extraction_calls':0,'new_retrieval_calls':0,'new_projection_calls':0,'energy_source':'read unchanged v5.1 energy.json','sample_count':len(samples)})
 print('MEASURED',OUT,flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','measure'],required=True);args=p.parse_args()
 if args.phase=='prepare':prepare()
 else:measure()
