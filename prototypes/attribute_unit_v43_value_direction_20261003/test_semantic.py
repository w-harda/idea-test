import unittest
import torch
from semantic import value_direction,retrieval_readout

class SemanticTests(unittest.TestCase):
    def test_common_weight_shift_invariance(self):
        torch.manual_seed(42);W=torch.randn(5,8,dtype=torch.float64);B=torch.linalg.qr(torch.randn(32,8,dtype=torch.float64)).Q
        _,q,d=value_direction(W,B,2);_,q2,d2=value_direction(W+torch.randn(8,dtype=torch.float64),B,2)
        torch.testing.assert_close(q,q2,rtol=1e-13,atol=1e-13);torch.testing.assert_close(d,d2,rtol=1e-13,atol=1e-13)
    def test_coordinate_rotation_invariance(self):
        torch.manual_seed(42);W=torch.randn(5,8,dtype=torch.float64);B=torch.linalg.qr(torch.randn(32,8,dtype=torch.float64)).Q;R=torch.linalg.qr(torch.randn(8,8,dtype=torch.float64)).Q
        _,_,d=value_direction(W,B,1);_,_,d2=value_direction(W@R,B@R,1);torch.testing.assert_close(d,d2,rtol=1e-13,atol=1e-13)
    def test_signed_budget_and_first_order(self):
        torch.manual_seed(42);B=torch.linalg.qr(torch.randn(32,8,dtype=torch.float64)).Q;W=torch.randn(5,8,dtype=torch.float64);_,_,d=value_direction(W,B,1)
        for sign_expected in (-1,1):
            raw=sign_expected*d;w=torch.randn(7,dtype=torch.float64);w/=w.norm();G=w[:,None]*raw[None,:]
            r,S,sign=retrieval_readout(raw,d,float((B.T@raw).norm()));self.assertEqual(sign,sign_expected)
            delta=.01*sign*w[:,None]*d[None,:];self.assertAlmostEqual(float(delta.norm()),.01,places=12);self.assertAlmostEqual(float((G*delta).sum()),.01*S,places=12)
        r,S,sign=retrieval_readout(torch.zeros_like(d),d,0.);self.assertEqual((r,S,sign),(0.,0.,0))
    def test_zero_centered_direction_is_rejected(self):
        with self.assertRaises(ValueError):value_direction(torch.ones(3,8),torch.eye(8),0)

if __name__=='__main__':unittest.main()
