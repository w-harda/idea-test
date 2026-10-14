from __future__ import annotations
import sys,math,hashlib,json,copy
from pathlib import Path
from dataclasses import dataclass,field,asdict,replace
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent.parent
V54=REPO/'prototypes/attribute_unit_v54_axis_audit_20261003'
sys.path.insert(0,str(V54));import axis_audit as audit
AXES=('age','gender','hair_length','color','garment_type','sleeve_length','lower_garment_length','hem_extent','hat','glasses','backpack','bag')
DIRECT={'age':'person','gender':'person','hat':'head','glasses':'head','backpack':'carried','bag':'carried'}

def valid_span(text,span):
 if not isinstance(span,dict):return False
 a,b=span.get('start'),span.get('end')
 return type(a) is int and type(b) is int and 0<=a<b<=len(text) and isinstance(span.get('raw'),str) and text[a:b]==span['raw']

@dataclass(frozen=True)
class SemanticRepresentation:
 projected_context:tuple[float,...]
 value_centroid:tuple[float,...]
 basis_rank:int
 basis_id:str
 pooling_source:str
 has_between_value_contrast:bool
 # 不将rank0的常量value centroid伪称为非零semantic subspace。

@dataclass(frozen=True)
class AttributeInstance:
 region:str
 semantic_axis:str|None
 canonical_value:str|None
 raw_text:str|None
 char_start:int|None
 char_end:int|None
 linked_object:dict|None
 provenance:dict
 unresolved:bool
 row_id:str=''
 unresolved_reasons:tuple[str,...]=()
 semantic_representation:SemanticRepresentation|None=None
 representation_status:str='not_attached'
 def __post_init__(self):
  if self.semantic_axis not in AXES and not(self.semantic_axis is None and self.unresolved):raise ValueError('only fixed 12 axes; unmapped audit records must be unresolved')
  if not self.unresolved and (self.canonical_value is None or self.raw_text is None or type(self.char_start) is not int or type(self.char_end) is not int or self.char_start<0 or self.char_end<=self.char_start):raise ValueError('resolved instance requires real source span/value')
 @property
 def source_span(self):return {'raw':self.raw_text,'start':self.char_start,'end':self.char_end}
 @property
 def instance_id(self):
  key=(self.row_id,self.provenance.get('stage01_slot'),self.region,self.semantic_axis,self.canonical_value,self.char_start,self.char_end,self.linked_object)
  return hashlib.sha256(json.dumps(key,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
 def to_dict(self):return dict(asdict(self),instance_id=self.instance_id,source_span=self.source_span)


def convert_record(rec)->list[AttributeInstance]:
 text=rec['caption'];attrs=rec['attributes'];pr=rec.get('provenance') or {};out=[]
 # 此映射仅做已有span的binding核对；它的raw-only补充不进入本adapter。
 oldmap=audit.old.map_record(rec)
 for slot,canonical in attrs.items():
  if canonical in (None,'null'):continue
  pp=pr.get(slot) or {};mentions=pp.get('mentions') or [None]
  for mention in mentions:
   m=copy.deepcopy(mention) if isinstance(mention,dict) else {};axis=None;region='unresolved';value=canonical;obj=m.get('linked_object');why=[];detail={}
   good=valid_span(text,m)
   if not good:why.append('invalid_or_missing_source_span')
   if pp.get('canonical')!=canonical:why.append('provenance_canonical_mismatch')
   if slot in DIRECT:axis=slot;region=DIRECT[slot]
   elif slot in ('hair_length','upper_clothing_length','lower_clothing_length'):
    if good:
     x=audit.classify_length(text,slot,canonical,m);region=x['region'];value=x['value'];detail={'interpreted_axis':x['axis'],'garment':x['garment'],'binding':x['binding'],'reason':x['reason']}
     axis=x['axis'] if x['axis'] in AXES else None
     if axis is None:why.append('outside_fixed_attack_axes:'+x['axis'])
     if x['binding']!='valid':why.append('unreliable_binding:'+str(x['reason']))
     if slot=='hair_length' and value not in ('short','medium','long'):axis=None;why.append('outside_fixed_hair_length_values')
    else:axis='hair_length' if slot=='hair_length' else None
   elif slot in ('upper_clothing_type','lower_clothing_type'):
    axis='garment_type'
    if good:
     h,reason=audit.object_head(text,m)
     if h:
      region=h['region'];value=h['value'];detail={'lexical_value':value,'original_family':canonical,'role':h['role']}
      if h['role']!='head' or value.endswith('_unspecified'):why.append('not_specific_garment_head')
      # 原source是名词本身；linked_object保持原样，不自行补一个新的source span。
      local=[x for x in audit.lexical(text) if x['start']<=m['start']<x['end']]
      if any(x['scope_risk'] for x in local):why.append('carried_or_negated_garment')
     else:why.append('unknown_garment_head')
     if reason:why.append(reason)
   elif slot in ('upper_clothing_color','lower_clothing_color'):
    axis='color'
    if good:
     h,reason=audit.object_head(text,obj)
     if h:region=h['region'];detail={'garment':h['value'],'original_family':canonical}
     if reason:why.append(reason)
     events=[x for x in oldmap['sources'].values() for x in x]
     bound=any(x['source']==slot and x['semantic_type']=='color' and x['evidence']==m for x in events)
     if not bound:why.append('unreliable_color_binding')
   else:why.append('unsupported_stage01_slot')
   if obj is not None and not valid_span(text,obj):why.append('invalid_linked_object_span')
   if region in ('unresolved','unresolved_region','not_clothing_type'):why.append('unresolved_region')
   prov={'stage01_slot':slot,'stage01_canonical':canonical,'stage01_provenance':copy.deepcopy(pp),'interpretation':detail,'source_policy':'original provenance only; no string search/relocation/raw-only addition'}
   out.append(AttributeInstance(region,axis,value,m.get('raw'),m.get('start'),m.get('end'),copy.deepcopy(obj),prov,bool(why),rec['row_id'],tuple(sorted(set(why)))))
 # 只去重完全相同源slot/span/metadata，不合并同值不同位置。
 return list({a.instance_id:a for a in out}.values())


def text_candidates(instances,caption,require_representation=False):
 result=[]
 for a in instances:
  if a.unresolved or a.semantic_axis not in AXES or not valid_span(caption,a.source_span):continue
  if require_representation and a.semantic_representation is None:continue
  result.append(a)
 return result

def score(attribute_instance:AttributeInstance)->float:
 raise NotImplementedError('S(a_i) intentionally undefined; supply an external future scorer')


def dynamic_topk(scored:list[tuple[AttributeInstance,float]],beta=1.0):
 if isinstance(beta,bool) or not isinstance(beta,(int,float)) or not math.isfinite(beta) or not 0<beta<=1:raise ValueError('beta must satisfy 0 < beta <= 1')
 if any(a.unresolved or a.semantic_axis not in AXES for a,s in scored):raise ValueError('unresolved instances cannot enter Top-K')
 if len({a.instance_id for a,s in scored})!=len(scored):raise ValueError('duplicate instance ID')
 if any(isinstance(s,bool) or not isinstance(s,(int,float)) or not math.isfinite(s) or s<0 for a,s in scored):raise ValueError('prototype Top-K contract requires finite nonnegative supplied S; no signed-score policy defined')
 # 同一旧entropy/effective-count公式；opaque instance ID不含ontology解读。
 total=math.fsum(s for a,s in scored);weights=[s/total if total else 0. for a,s in scored]
 entropy=-math.fsum(w*math.log(w) for w in weights if w>0);neff=math.exp(entropy) if total else 0.;k=min(len(scored),math.ceil(beta*neff))
 order=sorted(range(len(scored)),key=lambda i:(-scored[i][1],i))[:k]
 return {'beta':beta,'weights':dict((a.instance_id,w) for (a,s),w in zip(scored,weights)),'entropy':entropy,'n_eff':neff,'k_star':k,'selected_instances':[scored[i][0] for i in order]}


def attack_payload(a,caption=None):
 if a.unresolved or a.semantic_representation is None:raise ValueError('attack payload needs resolved source and attached representation')
 if caption is not None and not valid_span(caption,a.source_span):raise ValueError('source span roundtrip failed')
 # 图像与文本拿同一e对象；不实现G_I/G_T，也不额外把完整caption编码喂给图像攻击。
 return {'image':{'e':a.semantic_representation},'text':{'source_span':a.source_span,'e':a.semantic_representation}}


def verify_hashes(expected):
 for p,h in expected.items():
  if audit.digest(Path(p))!=h:raise ValueError('frozen source/cache mismatch: '+str(p))


def build_bases():
 c54=audit.load(V54/'outputs/centroid_rank.json');c53=audit.load(audit.V53/'outputs/corpus.json');c5=audit.load(audit.old.V5/'outputs/full/corpus.json')
 grouped={a:{} for a in AXES};origins={a:[] for a in AXES}
 def add(axis,values,centroids,counts,origin):
  origins[axis].append(origin)
  for val,c,n in zip(values,centroids,counts):
   if n<=0:continue
   bucket=grouped[axis].setdefault(val,[]);bucket.append((np.asarray(c,dtype=np.float64),n))
 for x in c5:
  if x['slot'] in DIRECT:add(x['slot'],x['values'],x['centroids'],x['counts'][:len(x['values'])],'v5-full:'+x['slot'])
 add('color',c53['color']['values'],c53['color']['centroids'],c53['color']['counts'],'v53-shared:color')
 for field,x in c54.items():
  region,axis=field.split(':')
  if axis not in ('garment_type','sleeve_length','lower_garment_length','hem_extent','hair_length') or not x['geometry']:continue
  vals=x['values'];counts=[x['cache_coverage'][v]['cached_train_vectors'] for v in vals]
  add(axis,vals,x['geometry']['centroids'],counts,'v54:'+field)
 result={}
 for axis in AXES:
  vals=sorted(grouped[axis]);C=np.array([sum(c*n for c,n in grouped[axis][v])/sum(n for c,n in grouped[axis][v]) for v in vals]);g=audit.between_rank(C)
  bid=hashlib.sha256(C.astype('<f8').tobytes()+json.dumps(vals).encode()).hexdigest()
  result[axis]={'values':vals,'counts':{v:sum(n for c,n in grouped[axis][v]) for v in vals},'Q':g['Q'],'center':g['center'],'centroids':g['centroids'],'rank':g['rank'],'singular_values':g['singular_values'],'basis_id':bid,'source_centroid_groups':origins[axis]}
 return result


class CachedReader:
 def __init__(self,split,bases):
  self.split=split;self.bases=bases;self.Z=np.load(audit.old.ZFILE,mmap_mode='r');self.H=np.load(audit.old.HFILE,mmap_mode='r');self.RZ=np.load(audit.V53/'cache/region_z.fp32.npy',mmap_mode='r')
  if self.Z.shape!=(67946,13,512) or self.H.shape!=(67946,77,512) or self.RZ.shape!=(67946,3,512):raise ValueError('cache shape mismatch')
  sys.path.insert(0,str(audit.old.CODE/'src'));sys.path.insert(0,'/home/lzf/Attack/IRRA-main/IRRA-main')
  from attributes.irra_token_alignment import align_irra_tokens,_span_token_indices
  from utils.simple_tokenizer import SimpleTokenizer
  self.align=align_irra_tokens;self.span_tokens=_span_token_indices;self.tokenizer=SimpleTokenizer();self.alignments={}
 def attach(self,a,i,caption):
  if a.unresolved:return a
  b=self.bases[a.semantic_axis]
  if a.canonical_value not in b['values']:return replace(a,representation_status='unseen_training_value')
  al=self.alignments.get(i)
  if al is None:al=self.align(caption,self.tokenizer);self.alignments[i]=al
  spans=[a.source_span]+([a.linked_object] if a.linked_object else []);why=[]
  for span in spans:
   ix,rr=self.span_tokens(caption,span,al['all_content_tokens']);why.extend(rr)
   if not ix:why.append('missing_token_support')
  if al['status']!='valid':why.extend(al['reasons'])
  if why:return replace(a,representation_status='token_alignment_unresolved:'+','.join(sorted(set(why))))
  if a.semantic_axis in DIRECT or a.semantic_axis=='hair_length':
   slot=a.semantic_axis;z=np.array(self.Z[i,self.split['slots'].index(slot)],dtype=np.float64);src='frozen_v4_slot_Z:'+slot
  elif a.region in ('upper','lower'):
   z=np.array(self.RZ[i,0 if a.region=='upper' else 1],dtype=np.float64);src='frozen_v4_three_slot_region_mean:'+a.region
  elif a.region=='full_body':
   obj=a.source_span if a.semantic_axis=='garment_type' else a.linked_object
   if not valid_span(caption,obj):return replace(a,representation_status='full_body_missing_original_object_span')
   ix,rr=self.span_tokens(caption,obj,al['all_content_tokens'])
   if rr or not ix:return replace(a,representation_status='full_body_object_alignment_unresolved')
   z=np.mean(self.H[i,sorted(ix)],axis=0,dtype=np.float64);src='existing_block8_object_uniform:full_body'
  else:return replace(a,representation_status='unavailable_region_localization')
  Q=np.asarray(b['Q'],dtype=np.float64).reshape(512,b['rank']);center=np.asarray(b['center']);p=Q@(Q.T@(z-center));centroid=np.asarray(b['centroids'][b['values'].index(a.canonical_value)])
  if not np.isfinite(p).all() or not np.isfinite(centroid).all():raise ValueError('nonfinite representation')
  e=SemanticRepresentation(tuple(p),tuple(centroid),b['rank'],b['basis_id'],src,b['rank']>0)
  return replace(a,semantic_representation=e,representation_status='ready' if b['rank'] else 'ready_single_value_no_contrast')
