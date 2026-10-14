import unittest,json,math,copy,sys
from pathlib import Path
from selection import *
ROOT=Path(__file__).resolve().parent

def candidate(aid,score,start=0):
    return {'instance_id':aid,'region':'upper','semantic_axis':'color','canonical_value':'black','source_span':{'raw':'black','start':start,'end':start+5},'provenance':{'original':'Stage01','binding':{'span':[start,start+5]}},'semantic_representation':{'basis_rank':10,'projected_context':[1.,2.],'value_centroid':[3.,4.]},'unresolved':False,'S_raw':score}

class SelectionTests(unittest.TestCase):
    def test_gamma1_original_probabilities(self):
        xs=[candidate('a',2.),candidate('b',3.)];x=select('c',xs,1.)
        self.assertAlmostEqual(x['probabilities']['a'],.4,places=15);self.assertAlmostEqual(x['probabilities']['b'],.6,places=15)
        self.assertAlmostEqual(x['N_eff'],math.exp(-.4*math.log(.4)-.6*math.log(.6)),places=14)
    def test_power_formula_known_distribution(self):
        x=select('c',[candidate('a',2.),candidate('b',3.)],2.)
        self.assertAlmostEqual(x['probabilities']['a'],4/13,places=15);self.assertAlmostEqual(x['probabilities']['b'],9/13,places=15)
    def test_positive_sharpening_keeps_top1_and_ranking(self):
        xs=[candidate('a',.5),candidate('b',3.),candidate('c',1.)]
        for g in GAMMAS:self.assertEqual([a['instance_id'] for a in select('c',xs,g)['ranked_attributes']],['b','c','a'])
    def test_equal_score_ties_keep_input_order(self):
        xs=[candidate('b',1),candidate('a',1),candidate('c',1)]
        for g in GAMMAS:self.assertEqual([a['instance_id'] for a in select('c',xs,g)['ranked_attributes']],['b','a','c'])
    def test_uniform_scores_do_not_sharpen_and_cap_is_four(self):
        xs=[candidate(str(i),1.) for i in range(6)]
        for g in GAMMAS:
            x=select('c',xs,g);self.assertAlmostEqual(x['N_eff'],6);self.assertEqual(x['k_star'],4)
    def test_entropy_and_k_monotonic(self):
        xs=[candidate(str(i),s) for i,s in enumerate([9.,2.,1.,.5,.1])];results=[select('c',xs,g) for g in GAMMAS]
        for a,b in zip(results,results[1:]):
            self.assertLessEqual(b['N_eff'],a['N_eff']);self.assertLessEqual(b['k_star'],a['k_star']);self.assertEqual(b['selected_attributes'],a['selected_attributes'][:b['k_star']])
    def test_ceiling_instead_of_rounding_or_floor(self):
        x=select('c',[candidate('a',9.),candidate('b',1.)],1.);self.assertGreater(x['N_eff'],1);self.assertLess(x['N_eff'],1.5);self.assertEqual(x['k_star'],2)
    def test_empty_and_single_candidate(self):
        for g in GAMMAS:
            x=select('c',[],g);self.assertEqual((x['candidate_count'],x['entropy'],x['N_eff'],x['k_star']),(0,0,0,0));self.assertEqual(x['probabilities'],{})
            x=select('c',[candidate('a',1e-9)],g);self.assertEqual(x['probabilities'],{'a':1.});self.assertEqual(x['k_star'],1)
    def test_scale_invariance(self):
        xs=[candidate(str(i),s) for i,s in enumerate([3.,2.,.1])]
        for g in GAMMAS:
            a=select('c',xs,g);b=select('c',[dict(x,S_raw=x['S_raw']*1000) for x in xs],g)
            self.assertEqual(a['k_star'],b['k_star'])
            for k,v in a['probabilities'].items():self.assertAlmostEqual(v,b['probabilities'][k],places=14)
    def test_large_and_small_finite_scores_no_power_overflow(self):
        for xs in [[candidate('a',1e300),candidate('b',1e299)],[candidate('a',1e-300),candidate('b',1e-301)]]:
            x=select('c',xs,3.);self.assertAlmostEqual(x['probabilities']['a'],1000/1001,places=13);self.assertTrue(math.isfinite(x['entropy']))
    def test_zero_negative_nonfinite_or_bool_rejected_by_selector(self):
        for s in [0.,-1.,float('inf'),float('nan'),True]:
            with self.assertRaises(ValueError):select('c',[candidate('a',s)],1.)
    def test_only_fixed_gammas(self):
        for g in [.5,2.5,4.,True,float('nan')]:
            with self.assertRaises(ValueError):select('c',[candidate('a',1.)],g)
    def test_unresolved_and_duplicate_ids_rejected(self):
        a=candidate('a',1.)
        with self.assertRaises(ValueError):select('c',[dict(a,unresolved=True)],1.)
        with self.assertRaises(ValueError):select('c',[a,a],1.)
    def test_selected_original_metadata_and_score_unchanged(self):
        xs=[candidate('a',2),candidate('b',1,start=9)];before=copy.deepcopy(xs)
        for g in GAMMAS:
            out=select('c',xs,g);self.assertEqual(out['ranked_attributes'],xs);self.assertIs(out['selected_attributes'][0],xs[0])
        self.assertEqual(xs,before)
    def test_source_adapter_copies_instance_preserves_different_spans(self):
        aa=[]
        for i,s in enumerate([2.,1.]):
            inst=candidate(str(i),s,start=i*9);score=inst.pop('S_raw');aa.append({'status':'scored','instance':inst,'instance_id':str(i),'rank':10,'S_raw':score})
        record={'attributes':aa};before=copy.deepcopy(record);xs=candidates_from_record(record)
        self.assertEqual(len(xs),2);self.assertNotEqual(xs[0]['source_span'],xs[1]['source_span']);self.assertEqual(record,before)
        self.assertIsNot(xs[0]['provenance'],aa[0]['instance']['provenance'])
    def test_source_adapter_excludes_unsupported_unresolved_zero(self):
        inst=candidate('a',1);inst.pop('S_raw');rows=[{'status':'score_unsupported'},{'status':'unresolved_excluded'},{'status':'scored','instance':inst,'instance_id':'a','rank':10,'S_raw':0.}]
        self.assertEqual(candidates_from_record({'attributes':rows}),[])
    def test_payload_contract(self):
        x=select('c',[candidate('a',1.)],1.5);self.assertEqual(set(x),{'caption_id','candidate_count','gamma','probabilities','entropy','N_eff','k_star','ranked_attributes','selected_attributes'});json.dumps(x,allow_nan=False)
    def test_saved25_gamma1_regression_and_monotonicity(self):
        lock=json.loads((ROOT/'protocol.lock.json').read_text());source=json.loads((Path(lock['source_root'])/'results.json').read_text());self.assertEqual(len(source),25)
        for row in source:
            xs=candidates_from_record(row);results=[select(row['row_id'],xs,g) for g in GAMMAS];b=results[0];old=row['topk']['S_raw']
            self.assertEqual([a['instance_id'] for a in b['ranked_attributes']],old['order']);self.assertAlmostEqual(b['N_eff'],old['n_eff'],places=12);self.assertEqual(b['k_star'],min(old['k_star'],4))
            for a,c in zip(results,results[1:]):self.assertLessEqual(c['N_eff'],a['N_eff']+1e-12);self.assertLessEqual(c['k_star'],a['k_star'])
    def test_no_model_dependencies(self):
        forbidden=[m for m in sys.modules if m=='torch' or m.startswith(('torch.','numpy','attributes.','model.','learned_reader','importance'))];self.assertEqual(forbidden,[])

if __name__=='__main__':unittest.main()
