import unittest,copy
import numpy as np
import factorized as f

def rec(text, entries):
 a={'row_id':'cuhk:1:0','caption':text,'attributes':{s:'null' for s in f.CLOTH+f.OTHERS},'provenance':{}}
 for slot,val,ev,obj in entries:
  def span(word):
   i=text.index(word);return {'raw':word,'start':i,'end':i+len(word)}
  m=span(ev)
  if obj:m['linked_object']=span(obj)
  a['attributes'][slot]=val;a['provenance'][slot]={'canonical':val,'mentions':[m]}
 return a

class Tests(unittest.TestCase):
 def test_dress_compounds_and_gown(self):
  self.assertEqual([x['type'] for x in f.body_mentions('A dress shirt and dress pants under a blue gown.')],['gown'])
  self.assertEqual(f.body_mentions('one piece of clothing material'),[])
 def test_factorize_and_preserve(self):
  a=rec('She wears a black dress.', [('lower_clothing_type','skirt_dress','dress',None),('lower_clothing_color','black','black','dress')]);old=copy.deepcopy(a);m=f.map_record(a)
  self.assertEqual(a,old);self.assertEqual(m['labels']['full_body:clothing_type'],'dress');self.assertEqual(m['labels']['full_body:color'],'black');self.assertIsNone(m['labels']['lower:clothing_type'])
 def test_refine_original_merged_lower(self):
  for raw,want in [('jeans','jeans'),('shorts','shorts'),('pants','trousers')]:
   m=f.map_record(rec('He wears blue '+raw+'.',[('lower_clothing_type','trousers_shorts',raw,None)]));self.assertEqual(m['labels']['lower:clothing_type'],want)
 def test_raw_only_no_invented_color(self):
  m=f.map_record(rec('A woman wearing a red gown.',[]));self.assertEqual(m['labels']['full_body:clothing_type'],'gown');self.assertIsNone(m['labels']['full_body:color'])
 def test_full_body_panel(self):
  m=f.map_record(rec('The dress is black on top and colored from the waist down.',[('lower_clothing_type','skirt_dress','dress',None),('lower_clothing_color','black','black','dress')]));self.assertIsNone(m['labels']['full_body:color']);self.assertEqual(m['labels']['full_body:clothing_type'],'dress')
 def test_bad_binding_and_compound(self):
  a=rec('black boobs and a skirt',[('lower_clothing_color','black','black','skirt')]);self.assertIsNone(f.map_record(a)['labels']['lower:color'])
  a=rec('gray cutoff jean shorts',[('lower_clothing_color','grey','gray','shorts')]);self.assertEqual(f.map_record(a)['labels']['lower:color'],'grey')
  a=rec('short hair and a shirt',[('upper_clothing_length','short','short','shirt')]);self.assertIsNone(f.map_record(a)['labels']['upper:clothing_length'])
 def test_carried_body_label_rejected(self):
  m=f.map_record(rec('A woman carrying a dress.', [('lower_clothing_type','skirt_dress','dress',None)]));self.assertIsNone(m['labels']['full_body:clothing_type'])
 def test_conflict_is_unresolved(self):
  a=rec('a dress and a gown',[('lower_clothing_type','skirt_dress','dress',None)]);self.assertIsNone(f.map_record(a)['labels']['full_body:clothing_type'])
 def test_region_weight_equivalence(self):
  rng=np.random.default_rng(1);H=rng.normal(size=(20,512));A=rng.uniform(size=(3,20));A/=A.sum(1,keepdims=True);np.testing.assert_allclose((A@H).mean(0),A.mean(0)@H,atol=1e-14);self.assertAlmostEqual(float(A.mean(0).sum()),1.)
 def test_rank_span_random(self):
  C=np.zeros((3,512));C[0,0]=1;C[1,1]=1;C[2,0]=-1;b=f.basis(C);Q=np.array(b['Q']);R=np.array(b['Q_random']);self.assertEqual(b['rank'],2);self.assertEqual(R.shape,Q.shape);np.testing.assert_allclose(Q.T@Q,np.eye(2),atol=1e-14);np.testing.assert_allclose(R.T@R,np.eye(2),atol=1e-14);np.testing.assert_allclose((C-C.mean(0))@Q@Q.T,C-C.mean(0),atol=1e-14)
 def test_energy_non_probability(self):
  x=np.zeros((1,512));x[0,0]=2;Q=np.eye(512)[:,:1];e,v=f.energies(x,{'a':Q,'b':Q});self.assertEqual(float(e['a'][0]+e['b'][0]),2.);self.assertTrue(v[0])
 def test_pair_controls_and_numeric_tie(self):
  records=[];mapped=[]
  for row,col,typ in [('cuhk:9:0','black','shirt'),('cuhk:10:0','red','shirt'),('cuhk:2:0','red','shirt'),('cuhk:1:0','red','jacket')]:
   a=rec('shirt',[]);a['row_id']=row;records.append(a);m={'labels':{x:None for x in f.FIELDS},'full_body_token_indices':None};m['labels']['upper:color']=col;m['labels']['upper:clothing_type']=typ;mapped.append(m)
  p=f.pair_select(records,mapped,list(range(4)),'upper','color');x=[p for p in p if p['value_a']=='black' or p['value_b']=='black'];self.assertEqual(len(x),3);self.assertEqual(sum(p['tier']=='strict' for p in x),2);self.assertTrue(any(p['row_a']=='cuhk:2:0' and p['row_b']=='cuhk:9:0' for p in x))
 def test_symmetric_duplicate_fixed(self):
  rr=[];mm=[]
  for i,c in enumerate(('black','red')):
   a=rec('shirt',[]);a['row_id']=f'cuhk:{i}:0';rr.append(a);m={'labels':{x:None for x in f.FIELDS},'full_body_token_indices':None};m['labels']['upper:color']=c;m['labels']['upper:clothing_type']='shirt';mm.append(m)
  self.assertEqual(len(f.pair_select(rr,mm,[0,1],'upper','color')),1)

if __name__=='__main__':unittest.main()
