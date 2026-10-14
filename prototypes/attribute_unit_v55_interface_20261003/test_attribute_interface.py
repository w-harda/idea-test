import unittest,tempfile,json,math,sys,random
from pathlib import Path
from dataclasses import replace
from attribute_interface import *

def rec(text,slot,value,raw,start,obj=None):
 m={'raw':raw,'start':start,'end':start+len(raw)}
 if obj is not None:m['linked_object']=obj
 attrs={s:'null' for s in ('age','gender','hair_length','upper_clothing_length','upper_clothing_color','upper_clothing_type','lower_clothing_length','lower_clothing_color','lower_clothing_type','hat','glasses','backpack','bag')};attrs[slot]=value
 return {'row_id':'test:1:0','caption':text,'attributes':attrs,'provenance':{slot:{'canonical':value,'mentions':[m]}}}
def obj(text,raw):
 i=text.index(raw);return {'raw':raw,'start':i,'end':i+len(raw)}
def merge(*rs):
 out=rs[0]
 for r in rs[1:]:out['attributes'].update({s:v for s,v in r['attributes'].items() if v!='null'});out['provenance'].update(r['provenance'])
 return out

class InterfaceTests(unittest.TestCase):
 def test_axes_exact(self):self.assertEqual(len(AXES),12);self.assertNotIn('garment_body_length',AXES)
 def test_construct(self):
  a=convert_record(rec('woman','gender','female','woman',0))[0];self.assertFalse(a.unresolved);self.assertEqual(a.region,'person')
 def test_roundtrip(self):
  text='black shirt';a=convert_record(rec(text,'upper_clothing_color','black','black',0,obj(text,'shirt')))[0];self.assertEqual(text[a.char_start:a.char_end],a.raw_text);self.assertFalse(a.unresolved)
 def test_same_value_different_span(self):
  text='short hair and short sleeve shirt';r=merge(rec(text,'hair_length','short','short',0),rec(text,'upper_clothing_length','short','short',15,obj(text,'shirt')));aa=convert_record(r)
  self.assertEqual([a.semantic_axis for a in aa if not a.unresolved],['hair_length','sleeve_length']);self.assertNotEqual(aa[0].source_span,aa[1].source_span);self.assertNotEqual(aa[0].instance_id,aa[1].instance_id)
 def test_same_color_different_object(self):
  text='black shirt and black pants';r=merge(rec(text,'upper_clothing_color','black','black',0,obj(text,'shirt')),rec(text,'lower_clothing_color','black','black',16,obj(text,'pants')));aa=convert_record(r);self.assertEqual([a.region for a in aa],['upper','lower']);self.assertEqual(len(text_candidates(aa,text)),2)
 def test_no_search_relocation(self):
  r=rec('black shirt','upper_clothing_color','black','black',1,{'raw':'shirt','start':6,'end':11});a=convert_record(r)[0];self.assertTrue(a.unresolved);self.assertEqual(a.char_start,1);self.assertEqual(text_candidates([a],r['caption']),[])
 def test_negative_span(self):self.assertFalse(valid_span('short',{'raw':'short','start':-5,'end':5}))
 def test_missing_provenance(self):
  r=rec('woman','gender','female','woman',0);r['provenance']={};self.assertTrue(convert_record(r)[0].unresolved)
 def test_canonical_mismatch(self):
  r=rec('woman','gender','female','woman',0);r['provenance']['gender']['canonical']='male';self.assertTrue(convert_record(r)[0].unresolved)
 def test_medium_hair(self):
  a=convert_record(rec('medium hair','hair_length','long','medium',0))[0];self.assertEqual(a.canonical_value,'medium');self.assertEqual(a.provenance['stage01_canonical'],'long')
 def test_bald_not_new_axis(self):
  a=convert_record(rec('bald man','hair_length','bald','bald',0))[0];self.assertTrue(a.unresolved);self.assertIsNone(a.semantic_axis)
 def test_body_length_excluded(self):
  t='short red jacket';a=convert_record(rec(t,'upper_clothing_length','short','short',0,obj(t,'jacket')))[0];self.assertTrue(a.unresolved);self.assertIsNone(a.semantic_axis)
 def test_lower_leg(self):
  t='long black pants';a=convert_record(rec(t,'lower_clothing_length','long','long',0,obj(t,'pants')))[0];self.assertFalse(a.unresolved);self.assertEqual(a.semantic_axis,'lower_garment_length')
 def test_hem_fullbody(self):
  t='long black dress';a=convert_record(rec(t,'lower_clothing_length','long','long',0,obj(t,'dress')))[0];self.assertFalse(a.unresolved);self.assertEqual((a.region,a.semantic_axis),('full_body','hem_extent'))
 def test_dress_type_preserves_family(self):
  a=convert_record(rec('dress','lower_clothing_type','skirt_dress','dress',0))[0];self.assertEqual((a.region,a.canonical_value),('full_body','dress'));self.assertEqual(a.provenance['stage01_canonical'],'skirt_dress')
 def test_shared_hair_evidence_rejected(self):
  t='short dark hair blue shirt';r=merge(rec(t,'hair_length','short','short',0),rec(t,'upper_clothing_length','short','short',0,obj(t,'shirt')));aa=convert_record(r);self.assertEqual(len(text_candidates(aa,t)),1);self.assertTrue(aa[1].unresolved)
 def test_no_raw_only_new_attributes(self):
  r=rec('woman in dress','gender','female','woman',0);self.assertEqual(len(convert_record(r)),1)
 def test_representation_required(self):
  t='woman';a=convert_record(rec(t,'gender','female','woman',0))[0];self.assertEqual(text_candidates([a],t,True),[])
 def test_payload_same_e(self):
  a=convert_record(rec('woman','gender','female','woman',0))[0];e=SemanticRepresentation((0.,)*512,(1.,)*512,0,'basis','cache',False);a=replace(a,semantic_representation=e);p=attack_payload(a,'woman');self.assertIs(p['image']['e'],p['text']['e']);self.assertFalse(e.has_between_value_contrast)
 def test_payload_reject_changed_caption(self):
  a=convert_record(rec('woman','gender','female','woman',0))[0];e=SemanticRepresentation((0.,)*512,(1.,)*512,1,'basis','cache',True);a=replace(a,semantic_representation=e)
  with self.assertRaises(ValueError):attack_payload(a,'man')
 def test_score_undefined(self):
  with self.assertRaises(NotImplementedError):score(None)
 def test_legacy_topk_regression(self):
  sys.path.insert(0,str(REPO/'src'));from attributes.dynamic_topk import select_attributes,SLOT_ORDER
  slots=list(SLOT_ORDER);rng=random.Random(42)
  for n in (0,1,5,12):
   aa=[AttributeInstance('head','hat','yes','hat',i*4,i*4+3,None,{'stage01_slot':'hat'},False,row_id='test:1:0') for i in range(n)]
   for zero in (False,True):
    scores=[0 if zero else rng.randint(0,9) for i in range(n)]
    for beta in (.4,1.):
     old=select_attributes({s:'yes' for s in slots[:n]},dict(zip(slots[:n],scores)),beta);new=dynamic_topk(list(zip(aa,scores)),beta)
     self.assertAlmostEqual(old['entropy'],new['entropy']);self.assertEqual(old['k_star'],new['k_star']);self.assertEqual([slots[aa.index(a)] for a in new['selected_instances']],old['selected_slots'])
 def test_float_topk_no_axis_collapse(self):
  text='black shirt and black pants';aa=convert_record(merge(rec(text,'upper_clothing_color','black','black',0,obj(text,'shirt')),rec(text,'lower_clothing_color','black','black',16,obj(text,'pants'))));r=dynamic_topk([(aa[0],.3),(aa[1],.3)]);self.assertEqual(r['k_star'],2)
 def test_topk_reject_invalid(self):
  a=convert_record(rec('woman','gender','female','woman',0))[0]
  for s in (-1,float('nan'),True):
   with self.assertRaises(ValueError):dynamic_topk([(a,s)])
  with self.assertRaises(ValueError):dynamic_topk([(replace(a,unresolved=True),1)])
 def test_frozen_hash_reject_substitution(self):
  with tempfile.TemporaryDirectory(dir=ROOT) as d:
   p=Path(d)/'fixture';p.write_bytes(b'cache');h=audit.digest(p);verify_hashes({str(p):h});p.write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_hashes({str(p):h})
 def test_twelve_basis_shape_orthogonality(self):
  b=build_bases();self.assertEqual(tuple(b),AXES)
  for x in b.values():
   q=np.asarray(x['Q']).reshape(512,x['rank']);self.assertTrue(np.allclose(q.T@q,np.eye(x['rank']),atol=1e-10));self.assertEqual(len(x['center']),512)
  self.assertEqual(b['backpack']['rank'],0)
 def test_excluded_diagnostic_not_used_as_cache_index(self):
  import run_interface
  split={'records':[{'row_id':'cuhk:8:0'},{'row_id':'cuhk:9:0'}],'validation_indices':[1]}
  self.assertEqual(run_interface.smoke_indices(split,16,{'cuhk:0:0','cuhk:8:0'}),[1])
 def test_full_mode_guard(self):
  import run_interface,argparse
  with self.assertRaises(ValueError):run_interface.run(argparse.Namespace(mode='full',allow_full_validation=False,limit=16))

if __name__=='__main__':unittest.main()
