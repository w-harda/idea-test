"""v5.1 确定性匹配和cached投影诊断的针对性测试。"""
import unittest
import numpy as np
import selectivity as m

def split(rows):
    slots=['age','gender','hair_length','upper_clothing_length','upper_clothing_color','upper_clothing_type','lower_clothing_length','lower_clothing_color','lower_clothing_type','backpack','bag','glasses','hat']
    values=[['a','b','c','null'] for _ in slots]
    records=[]
    for row_id,updates in rows:
        y=[0]*13
        for k,v in updates.items():y[slots.index(k)]=v
        records.append({'row_id':row_id,'pid':1,'labels':y,'caption':row_id,'truncated':False})
    return {'slots':slots,'values':values,'records':records,'validation_indices':list(range(len(records)))}

def brute(s,target):
    slots=s['slots'];j=slots.index(target);siblings=[slots.index(t) for t in m.region(target) if t!=target];records=s['records'];eligible=[i for i in s['validation_indices'] if records[i]['labels'][j]!=3];pairs={}
    for a in eligible:
        ya=records[a]['labels'];c=[b for b in eligible if records[b]['labels'][j]!=ya[j]]
        strict=[b for b in c if all(ya[t]!=3 and records[b]['labels'][t]!=3 and ya[t]==records[b]['labels'][t] for t in siblings)]
        if not c:continue
        def h(b):return sum(ya[t]!=records[b]['labels'][t] for t in range(13) if t!=j)
        if strict:b=min(strict,key=lambda b:(h(b),m.key(records[b])));tier='strict'
        else:
            b=min(c,key=lambda b:(sum(ya[t]!=records[b]['labels'][t] for t in siblings),sum(ya[t]==3 or records[b]['labels'][t]==3 for t in siblings),h(b),m.key(records[b])));tier='relaxed'
        aa,bb=sorted((a,b),key=lambda i:m.key(records[i]));pairs[(aa,bb)]=tier
    return pairs

class PairTests(unittest.TestCase):
    def test_exact_brute_force_and_signature_compression(self):
        rng=np.random.default_rng(42);rows=[('cuhk:%d:0'%i,dict(zip(['age','gender','upper_clothing_color','upper_clothing_type','upper_clothing_length'],rng.integers(0,4,5).tolist()))) for i in range(40)]
        rows+= [('cuhk:100:0',rows[0][1]),('cuhk:101:0',rows[1][1])]
        s=split(rows)
        for target in m.TARGETS:
            g=m.select_pairs(s,target);self.assertEqual({(p['a'],p['b']):p['tier'] for p in g['pairs']},brute(s,target))
            for p in g['pairs']:self.assertNotEqual(p['value_a'],p['value_b']);self.assertNotIn('null',(p['value_a'],p['value_b']));self.assertTrue(p['same_pid'])
    def test_numeric_tie_and_dedup(self):
        s=split([('cuhk:1:0',{}),('cuhk:10:0',{'upper_clothing_color':1}),('cuhk:2:0',{'upper_clothing_color':1}),('cuhk:20:0',{'upper_clothing_color':3})])
        g=m.select_pairs(s,'upper_clothing_color');p=next(p for p in g['pairs'] if 'cuhk:1:0' in p['selected_by']);self.assertEqual(p['row_b'],'cuhk:2:0');self.assertEqual(len(g['pairs']),2);self.assertEqual(g['anchor_counts']['strict'],3)
    def test_unknown_is_relaxed(self):
        s=split([('cuhk:1:0',{'upper_clothing_type':3}),('cuhk:2:0',{'upper_clothing_type':3,'upper_clothing_color':1})])
        g=m.select_pairs(s,'upper_clothing_color');self.assertEqual(g['pairs'][0]['tier'],'relaxed');self.assertEqual(g['pairs'][0]['unknown_regional_controls'],['upper_clothing_type']);self.assertEqual(g['pairs'][0]['non_target_hamming'],0)
    def test_region_priority_over_other_attributes(self):
        s=split([('cuhk:1:0',{}),('cuhk:2:0',{'upper_clothing_color':1,'age':1,'gender':1,'hair_length':1}),('cuhk:3:0',{'upper_clothing_color':1,'upper_clothing_type':1})])
        g=m.select_pairs(s,'upper_clothing_color');p=next(p for p in g['pairs'] if 'cuhk:1:0' in p['selected_by']);self.assertEqual(p['row_b'],'cuhk:2:0');self.assertEqual(p['tier'],'strict')
    def test_no_candidate(self):
        g=m.select_pairs(split([('cuhk:1:0',{}),('cuhk:2:0',{})]),'upper_clothing_color');self.assertEqual(g['pairs'],[]);self.assertEqual(g['anchor_counts']['no_candidate'],2)

class EnergyTests(unittest.TestCase):
    def test_projection_equivalence_rotation_and_scaling(self):
        rng=np.random.default_rng(42);q=np.linalg.qr(rng.normal(size=(512,10)))[0];rot=np.linalg.qr(rng.normal(size=(10,10)))[0];x=rng.normal(size=(5,512));a,v,_=m.energy(x,{'q':q});b,_,_=m.energy(-3*x,{'q':q@rot});expected=np.linalg.norm(x@(q@q.T),axis=1)**2/np.linalg.norm(x,axis=1)**2;np.testing.assert_allclose(a['q'],expected,rtol=1e-13);np.testing.assert_allclose(a['q'],b['q'],rtol=1e-13)
    def test_overlap_is_not_probability_and_zero_is_undefined(self):
        q=np.eye(3)[:,:1];a,v,den=m.energy([[1,0,0],[0,0,0]],{'s':q,'t':q});self.assertEqual(a['s'][0]+a['t'][0],2);self.assertFalse(v[1]);self.assertTrue(np.isnan(a['s'][1]))
    def test_all_sibling_hit_and_random_comparison(self):
        e={'s':np.array([.5,.5]),'t':np.array([.2,.4]),'u':np.array([.6,.1])};r={'s':np.array([.01,.01]),'t':np.array([.02,.02]),'u':np.array([.01,.01])};d=m.metrics('s',['t','u'],e,r,np.array([True,True]));self.assertEqual(d['semantic']['hit_rate'],.5);self.assertEqual(d['semantic']['sibling_hit_rates']['t'],1);self.assertEqual(d['random']['hit_rate'],0);self.assertEqual(d['target_gt_same_rank_random_rate'],1);self.assertAlmostEqual(d['semantic']['margin']['median'],0)

if __name__=='__main__':unittest.main()
