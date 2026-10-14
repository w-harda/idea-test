"""分组只依赖原文/provenance；不含selectivity值。"""
import unittest
import diagnose as d

def example(text,garment,color='black',region='lower',canonical=None):
 start=text.index(garment);obj={'raw':garment,'start':start,'end':start+len(garment)};cs=text.index(color);c={'raw':color,'start':cs,'end':cs+len(color),'linked_object':dict(obj)}
 typ=region+'_clothing_type';col=region+'_clothing_color'
 if canonical is None:canonical=d.lower_family(garment) if region=='lower' else d.upper_family(garment)
 if canonical=='full_body':canonical='skirt_dress'
 return {'row_id':'cuhk:1:0','caption':text,'attributes':{typ:canonical,col:color},'provenance':{typ:{'canonical':canonical,'mentions':[obj]},col:{'canonical':color,'mentions':[c]}}}

class Rules(unittest.TestCase):
 def test_clear_lower_and_upper(self):
  self.assertEqual(d.classify_caption(example('black skirt with white shoes','skirt'),'lower')['group'],'A_clear_region')
  self.assertEqual(d.classify_caption(example('black shirt and blue jeans','shirt',region='upper'),'upper')['group'],'A_clear_region')
 def test_fullbody_and_false_dress_modifiers(self):
  self.assertEqual(d.classify_caption(example('black dress with white shoes','dress'),'lower')['group'],'B_full_body')
  self.assertFalse(d.FULL.search('dress pants and dress shirt, man dressed in a suit'))
  self.assertTrue(d.FULL.search('one-piece gown'))
 def test_layering_and_shared_color(self):
  x=example('black dress over black leggings','dress');self.assertIn('multiple_same_region_garment_families',d.classify_caption(x,'lower')['flags'])
  x=example('black dress','dress');x['provenance']['upper_clothing_color']=x['provenance']['lower_clothing_color'];self.assertIn('color_evidence_shared_between_regions',d.classify_caption(x,'lower')['flags'])
 def test_wrong_object_and_missing_link(self):
  x=example('black shirt with black skirt','skirt');x['provenance']['lower_clothing_color']['mentions'][0]['linked_object']={'raw':'shirt','start':6,'end':11};flags=d.classify_caption(x,'lower')['flags'];self.assertIn('color_object_not_in_selected_type_provenance',flags)
  del x['provenance']['lower_clothing_color']['mentions'][0]['linked_object'];self.assertIn('missing_color_linked_object',d.classify_caption(x,'lower')['flags'])
 def test_explicit_upper_color_qualifier(self):
  x=example('dress black on top and multi-colored from waist down','dress');self.assertIn('lower_color_explicit_upper_region_qualifier',d.classify_caption(x,'lower')['flags'])
 def test_invalid_span_and_canonical(self):
  x=example('black skirt','skirt');x['provenance']['lower_clothing_type']['mentions'][0]['start']=0;self.assertIn('invalid_type_span',d.classify_caption(x,'lower')['flags'])
  x=example('black skirt','skirt',canonical='trousers_shorts');self.assertIn('type_object_value_mismatch',d.classify_caption(x,'lower')['flags'])
 def test_pair_precedence(self):
  a={'group':'A_clear_region','has_full_body':False};b={'group':'B_full_body','has_full_body':True};c={'group':'C_binding_risk','has_full_body':True};self.assertEqual(d.pair_group(a,b),'B_full_body');self.assertEqual(d.pair_group(b,c),'C_binding_risk');self.assertEqual(d.pair_group(a,a),'A_clear_region')
 def test_color_binding_crosses_another_garment(self):
  x=example('black shirt and skirt','skirt');self.assertIn('color_binding_crosses_other_garment',d.classify_caption(x,'lower')['flags'])
 def test_generic_top_under_jacket_is_layering_risk(self):
  x=example('grey top and black jacket on top','jacket',region='upper');self.assertIn('multiple_same_region_garment_families',d.classify_caption(x,'upper')['flags'])
 def test_repetition_not_layering(self):
  x=example('black skirt. The skirt is black','skirt');self.assertEqual(d.classify_caption(x,'lower')['group'],'A_clear_region')

if __name__=='__main__':unittest.main()
