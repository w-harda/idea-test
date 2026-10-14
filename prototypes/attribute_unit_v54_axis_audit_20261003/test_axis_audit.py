import unittest,copy
import numpy as np
import axis_audit as a

def mention(text,word,obj=None):
 i=text.index(word);m={'raw':word,'start':i,'end':i+len(word)}
 if obj:
  j=text.index(obj);m['linked_object']={'raw':obj,'start':j,'end':j+len(obj)}
 return m
class TestAxisAudit(unittest.TestCase):
 def test_longest_heads(self):
  x=a.lexical('a t-shirt, sweatshirt, dress shirt, dress pants, dress shoes and jean shorts')
  self.assertEqual([m['value'] for m in x],['t_shirt','sweatshirt','shirt','pants_trousers','dress_shoes','shorts'])
 def test_jean_material_not_lower_garment(self):
  self.assertEqual(a.lexical('a blue jean jacket')[0]['role'],'material_modifier');self.assertEqual(a.lexical('jeans and a jacket')[0]['role'],'head')
 def test_compound_partial_provenance_same_head(self):
  t='a polo shirt';p=mention(t,'polo');q=mention(t,'shirt');self.assertEqual(a.object_head(t,p)[0]['value'],'polo');self.assertEqual(a.object_head(t,q)[0]['value'],'polo')
 def test_onepiece_is_construction(self):
  self.assertEqual([(m['value'],m['role']) for m in a.lexical('one-piece pink dress')],[('one_piece','construction'),('dress','head')]);self.assertEqual(a.lexical('one piece of material'),[])
 def test_sleeve_axis(self):
  t='a long sleeve white shirt';m=mention(t,'long','shirt');x=a.classify_length(t,'upper_clothing_length','long',m);self.assertEqual((x['axis'],x['binding']),('sleeve_length','valid'))
 def test_dress_sleeve_not_hem(self):
  t='a short sleeve knee length dress';x=a.classify_length(t,'lower_clothing_length','short',mention(t,'short','dress'));self.assertEqual((x['axis'],x['region'],x['binding']),('sleeve_length','full_body','valid'));rr=a.raw_landmarks(t,a.lexical(t));self.assertEqual((rr[0]['axis'],rr[0]['value']),('hem_landmark','knee_length'))
 def test_sleeve_lower_binding(self):
  t='a shirt with short sleeves and tan shorts';x=a.classify_length(t,'lower_clothing_length','short',mention(t,'short','shorts'));self.assertEqual(x['axis'],'sleeve_length');self.assertNotEqual(x['binding'],'valid');self.assertEqual(x['region'],'unresolved_region')
 def test_hair_misbound(self):
  t='short black hair and a shirt';x=a.classify_length(t,'upper_clothing_length','short',mention(t,'short','shirt'));self.assertEqual((x['axis'],x['binding']),('hair_length','conflict'))
 def test_body_descriptor_vs_sleeve(self):
  t='a short red jacket with long sleeves';x=a.classify_length(t,'upper_clothing_length','short',mention(t,'short','jacket'));self.assertEqual(x['axis'],'garment_body_length');self.assertEqual(x['binding'],'valid')
 def test_unmarked_shirt_ambiguous(self):
  t='a long gray shirt';x=a.classify_length(t,'upper_clothing_length','long',mention(t,'long','shirt'));self.assertEqual(x['axis'],'ambiguous');self.assertNotEqual(x['binding'],'valid')
 def test_leg_and_hem(self):
  for t,obj,want in [('long black pants','pants','lower_garment_length'),('short blue skirt','skirt','hem_extent'),('long red dress','dress','hem_extent')]:
   word=t.split()[0];x=a.classify_length(t,'lower_clothing_length',word,mention(t,word,obj));self.assertEqual(x['axis'],want);self.assertEqual(x['binding'],'valid')
 def test_hair_medium_not_long(self):
  t='medium hair';x=a.classify_length(t,'hair_length','long',mention(t,'medium'));self.assertEqual(x['value'],'medium');self.assertEqual(x['canonical_value'],'long')
 def test_shoulder_hair_landmark_not_extent(self):
  t='shoulder length hair';x=a.classify_length(t,'hair_length','long',mention(t,'shoulder length'));self.assertEqual(x['axis'],'hair_landmark')
 def test_spatial_top_not_garment(self):
  self.assertEqual(a.lexical('on top of a shirt')[0]['role'],'location_word')
 def test_bald_not_length(self):
  x=a.classify_length('bald man','hair_length','bald',mention('bald man','bald'));self.assertEqual(x['axis'],'hair_state')
 def test_landmark_not_hair_shoe(self):
  self.assertNotEqual(a.raw_landmarks('ankle length boots',a.lexical('ankle length boots'))[0]['binding'],'valid')
 def test_scope_wearing_reset(self):
  x=a.lexical('carrying a bag and wearing a gown');self.assertFalse(x[0]['scope_risk']);self.assertTrue(a.lexical('carrying a gown')[0]['scope_risk'])
 def test_rank_not_fixed(self):
  c=np.zeros((3,512));c[0,0]=1;c[1,1]=1;c[2,0]=-1;g=a.between_rank(c);self.assertEqual(g['rank'],2);Q=np.array(g['Q']);np.testing.assert_allclose(Q.T@Q,np.eye(2),atol=1e-14);self.assertEqual(a.between_rank(np.zeros((1,512)))['rank'],0)
 def test_original_span_not_mutated(self):
  t='short sleeves dress';m=mention(t,'short','dress');old=copy.deepcopy(m);a.classify_length(t,'lower_clothing_length','short',m);self.assertEqual(m,old)
if __name__=='__main__':unittest.main()
