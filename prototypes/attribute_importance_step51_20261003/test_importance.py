import unittest,json,math,tempfile
from unittest.mock import patch
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from importance import *
from attribute_interface import verify_hashes
from run import parser,summarize_bias

def instance(axis='color',value='black',start=0,unresolved=False,rank=2):
    e=SemanticRepresentation((0.,)*512,(1.,)*512,rank,axis+'basis','test_frozen',rank>0)
    return AttributeInstance('upper',axis,value,'black',start,start+5,None,{'stage01_slot':axis},unresolved,'test:0',semantic_representation=e,representation_status='ready')

def bases(rank=2):
    return {'color':{'rank':rank,'basis_id':'colorbasis','Q':torch.eye(512,dtype=torch.float64)[:,:rank].tolist()}}

class ScoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);self.G=torch.randn(4,1,512);self.w=torch.ones(4)/2;self.Q=torch.linalg.qr(torch.randn(512,7,dtype=torch.float64)).Q
    def test_projection_norm_and_rank_normalization(self):
        x=projected_score(self.G,self.w,self.Q)
        self.assertAlmostEqual(x['S_raw'],float((self.Q.T@x['g'].double()).norm()),places=12)
        self.assertAlmostEqual(x['S_norm'],x['S_raw']/math.sqrt(7),places=12);self.assertGreaterEqual(x['S_raw'],0)
    def test_equal_budget_and_gradient_inner_product(self):
        x=projected_score(self.G,self.w,self.Q);delta=equal_budget_delta(self.w,x['direction'])
        self.assertAlmostEqual(float(delta.double().norm()),.01,places=8)
        predicted=float((self.G.double()*delta.double()).sum());self.assertAlmostEqual(predicted,.01*x['S_raw'],places=7)
    def test_linear_objective_first_order_exact(self):
        x=projected_score(self.G,self.w,self.Q);delta=equal_budget_delta(self.w,x['direction']);H=torch.zeros_like(self.G,dtype=torch.float64)
        F=lambda h:(h*self.G.double()).sum();self.assertAlmostEqual(float(F(H+delta)-F(H)),.01*x['S_raw'],places=7)
    def test_score_is_maximum_within_unit_subspace(self):
        x=projected_score(self.G,self.w,self.Q);gen=torch.Generator().manual_seed(1)
        for _ in range(10):
            u=torch.randn(7,dtype=torch.float64,generator=gen);u/=u.norm();d=(self.Q@u).float();delta=equal_budget_delta(self.w,d)
            self.assertLessEqual(float((self.G.double()*delta).sum()),.01*x['S_raw']+1e-7)
    def test_retrieval_backward_once_for_many_instances(self):
        class Probe:
            def encode(_,caption,layer):
                H=torch.zeros(4,1,512,requires_grad=True);return H.square().sum()+H.sum(),H,torch.zeros_like(H)
            def objective(_,feature):return feature,feature
        attrs=[instance(start=0),instance(start=6)]
        with patch('torch.autograd.grad',wraps=torch.autograd.grad) as grad:
            _,_,G,_,_,_=baseline_gradient(Probe(),'black black');rs=score_instances(attrs,'black black',G,{a.instance_id:self.w for a in attrs},bases())
            self.assertEqual(grad.call_count,1);self.assertEqual(len(rs),2)
    def test_id_mapping_same_value_different_span(self):
        a,b=instance(start=0),instance(start=6);self.assertNotEqual(a.instance_id,b.instance_id)
        rs=score_instances([b,a],'black black',self.G,{a.instance_id:self.w,b.instance_id:self.w},bases());self.assertEqual([x['instance_id'] for x in rs],[b.instance_id,a.instance_id])
        self.assertEqual(rs[0]['instance']['char_start'],6)
    def test_unresolved_excluded(self):
        a=instance(unresolved=True);rs=score_instances([a],'black',self.G,{},bases());self.assertEqual(rs[0]['status'],'unresolved_excluded');self.assertIsNone(rs[0]['S_raw'])
        self.assertEqual(topk_report([a],rs)['S_raw']['m'],0)
    def test_rank_zero_safe_exclusion_no_fallback(self):
        a=instance(rank=0);rs=score_instances([a],'black',self.G,{},bases(0));self.assertEqual(rs[0]['status'],'score_unsupported');self.assertIsNone(rs[0]['S_norm']);self.assertEqual(topk_report([a],rs)['S_norm']['k_star'],0)
    def test_representation_missing_excluded(self):
        a=replace(instance(),semantic_representation=None);rs=score_instances([a],'black',self.G,{},bases());self.assertEqual(rs[0]['status'],'representation_or_source_unavailable')
    def test_duplicate_id_rejected(self):
        a=instance()
        with self.assertRaises(ValueError):score_instances([a,a],'black',self.G,{},bases())
    def test_basis_identity_mismatch_rejected(self):
        a=instance()
        with self.assertRaises(ValueError):score_instances([a],'black',self.G,{a.instance_id:self.w},{'color':dict(bases()['color'],basis_id='bad')})
    def test_nonnegative_zero_score_no_unit_direction(self):
        x=projected_score(torch.zeros_like(self.G),self.w,self.Q);self.assertEqual(x['S_raw'],0);self.assertEqual(x['S_norm'],0);self.assertIsNone(equal_budget_delta(self.w,x['direction']))
    def test_invalid_inputs_rejected(self):
        for w,Q in [(self.w*2,self.Q),(self.w,self.Q*2),(self.w,torch.empty(512,0)),(self.w,self.Q.requires_grad_())]:
            with self.assertRaises(ValueError):projected_score(self.G,w,Q)
        self.Q=self.Q.detach()
        with self.assertRaises(ValueError):equal_budget_delta(self.w,torch.zeros(512))
    def test_dynamic_topk_adapter_and_rank_normalized_change(self):
        a,b=instance(start=0),instance(start=6);rows=[{'status':'scored','instance_id':a.instance_id,'S_raw':4.,'S_norm':1.},{'status':'scored','instance_id':b.instance_id,'S_raw':2.,'S_norm':2.}]
        out=topk_report([a,b],rows);self.assertEqual(out['S_raw']['order'],[a.instance_id,b.instance_id]);self.assertEqual(out['S_norm']['order'],[b.instance_id,a.instance_id]);self.assertEqual(out['S_raw']['k_star'],2)
    def test_invalid_span_excluded_not_relocated(self):
        a=instance(start=1);rs=score_instances([a],'black',self.G,{},bases());self.assertEqual(rs[0]['status'],'representation_or_source_unavailable')
    def test_rank_correlation_ties_and_undefined(self):
        self.assertAlmostEqual(rank_correlation([1,1,2],[3,3,4]),1);self.assertIsNone(rank_correlation([1,1],[2,3]));self.assertIsNone(rank_correlation([],[]))
    def test_fixed_selection_never_full(self):
        records=[{'row_id':'cuhk:10:1'},{'row_id':'cuhk:2:0'},{'row_id':'cuhk:2:1'}];split={'records':records,'validation_indices':[0,2,1]};old={'results':[{'row_id':'cuhk:99:1'}]}
        self.assertEqual([r['row_id'] for r in select_rows(split,old,'smoke')],['cuhk:99:1','cuhk:2:0','cuhk:2:1']);self.assertEqual(len(select_rows(split,old,'diagnostic')),4)
        with self.assertRaises(ValueError):select_rows(split,old,'full')
    def test_cli_default_smoke_and_full_rejected(self):
        self.assertEqual(parser().parse_args([]).mode,'smoke')
        with self.assertRaises(SystemExit):parser().parse_args(['--mode','full'])
    def test_diagnostic_requires_explicit_user_guard(self):
        from run import main
        with self.assertRaises(SystemExit):main(['--mode','diagnostic'])
    def test_same_rank_random_fixed_seed(self):
        bs={a:{'rank':0 if a=='backpack' else 2} for a in AXES};x=random_controls(bs);y=random_controls(bs)
        for a in AXES:
            self.assertTrue(torch.equal(x[a],y[a]));self.assertEqual(x[a].shape,(512,bs[a]['rank']))
            self.assertTrue(torch.allclose(x[a].T@x[a],torch.eye(bs[a]['rank'],dtype=torch.float64),atol=1e-12))
    def test_frozen_source_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as tmp:
            p=Path(tmp)/'test.txt';p.write_text('a');import hashlib;h=hashlib.sha256(b'a').hexdigest();verify_hashes({str(p):h});p.write_text('b')
            with self.assertRaises(ValueError):verify_hashes({str(p):h})
    def test_frozen_artifact_manifest_has_required_sources(self):
        lock=json.loads((ROOT/'provenance.lock.json').read_text())['frozen_sha256'];names=[Path(p).name for p in lock]
        for name in ('semantic_bases.json','block8.fp32.npy','reader_final.pt','split.json','irra_attribute_unit_prototype.py','irra_soft_rank_gradient.py','attribute_interface.py','clean-train-gallery.pt'):self.assertIn(name,names)
        self.assertEqual(lock[str(V55/'outputs/full/semantic_bases.json')],'94f114b9e330cb1f205c5ef4d1ca0aa9df9a6e7cd2e7906cf6ac69f1b68f32d4')

if __name__=='__main__':unittest.main()
