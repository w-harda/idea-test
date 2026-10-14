import unittest
import torch
from subspace import *

class GeometryTests(unittest.TestCase):
    def test_equal_value_center_and_translation(self):
        C=torch.tensor([[1.,0.,1.],[3.,2.,1.],[5.,4.,1.]],dtype=torch.float64);a=geometry_from_centroids(C);b=geometry_from_centroids(C+torch.tensor([20.,10.,40.]))
        self.assertEqual(a['euclidean'].shape[1],1);torch.testing.assert_close(a['euclidean']@a['euclidean'].T,b['euclidean']@b['euclidean'].T)
        self.assertTrue(torch.equal(C.mean(0),torch.tensor([3.,2.,1.],dtype=torch.float64)))
    def test_log_map_angle_and_small_angle(self):
        mu=torch.tensor([1.,0.,0.]);x=torch.tensor([[1.,0.,0.],[0.,1.,0.],[1.,1e-9,0.]],dtype=torch.float64);v,theta=log_map(x,mu)
        torch.testing.assert_close(v@mu.double(),torch.zeros(3,dtype=torch.float64));self.assertAlmostEqual(float(v[1].norm()),torch.pi/2,places=12);self.assertTrue(torch.isfinite(v).all());self.assertLess(float(v[0].norm()),1e-12)
    def test_degenerate_and_antipodal(self):
        C=torch.ones(3,8,dtype=torch.float64);g=geometry_from_centroids(C);self.assertEqual(g['euclidean'].shape[1],0);self.assertEqual(g['tangent'].shape[1],0)
        with self.assertRaises(ValueError):log_map(torch.tensor([[-1.,0.]]),torch.tensor([1.,0.]))
        with self.assertRaises(ValueError):normalize(torch.zeros(3,8))
    def test_same_rank_random_reproducible(self):
        a=random_basis(32,4,torch.Generator().manual_seed(42));b=random_basis(32,4,torch.Generator().manual_seed(42));self.assertTrue(torch.equal(a,b));torch.testing.assert_close(a.T@a,torch.eye(4,dtype=torch.float64))
    def test_overlap_unequal_ranks(self):
        A=torch.eye(8,dtype=torch.float64)[:,:2];B=torch.eye(8,dtype=torch.float64)[:,:4];r=overlap(A,B);self.assertEqual(r['left_fraction'],1.);self.assertEqual(r['right_fraction'],.5);self.assertAlmostEqual(r['symmetric_overlap'],2/(8**.5))
    def test_full_vs_between_value_nearest(self):
        torch.manual_seed(42);C=torch.randn(4,32,dtype=torch.float64);Z=torch.randn(20,32,dtype=torch.float64);Q=geometry_from_centroids(C)['euclidean'];self.assertEqual(Q.shape[1],3);self.assertTrue(torch.equal(nearest(Z,C),nearest(Z,C,Q)))
    def test_projected_gradient_budget_first_order(self):
        torch.manual_seed(42);G=torch.randn(7,32,dtype=torch.float64);w=torch.randn(7,dtype=torch.float64);w/=w.norm();g=w@G;Q=random_basis(32,4,torch.Generator().manual_seed(42));p,S,d=projected_direction(g,Q);delta=.01*w[:,None]*d[None,:]
        self.assertAlmostEqual(float(delta.norm()),.01,places=12);self.assertAlmostEqual(float((G*delta).sum()),.01*S,places=12);self.assertLessEqual(S,float(g.norm())+1e-12)
        _,s,d0=projected_direction(g,torch.empty(32,0,dtype=torch.float64));self.assertEqual(s,0);self.assertFalse(d0.count_nonzero())
    def test_tangent_basis_orthogonal_to_reference(self):
        torch.manual_seed(42);a=geometry_from_centroids(torch.randn(5,32,dtype=torch.float64)+3);self.assertEqual(a['tangent'].shape[1],4);torch.testing.assert_close(a['reference']@a['tangent'],torch.zeros(4,dtype=torch.float64),atol=1e-12,rtol=0)
    def test_smoke_bounded_and_deterministic(self):
        split={'values':[['a','b','null']],'records':[{'labels':[i%3]} for i in range(100)],'train_indices':list(range(100))};a=smoke_indices(split,'train_indices');self.assertEqual(a,[0,1,3,4]);self.assertEqual(a,smoke_indices(split,'train_indices'))

class DegenerateRandomTests(unittest.TestCase):
    def test_single_value_and_identical_centroids_exact_zero_span(self):
        generator=torch.Generator().manual_seed(123)
        for _ in range(20):
            C=torch.randn(1,512,dtype=torch.float64,generator=generator)
            for rows in (C,C.expand(4,-1).clone()):
                geo=geometry_from_centroids(rows)
                self.assertEqual(geo['euclidean'].shape[1],0)
                self.assertEqual(geo['tangent'].shape[1],0)

if __name__=='__main__':unittest.main()
