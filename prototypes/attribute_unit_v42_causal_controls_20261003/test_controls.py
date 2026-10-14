import unittest
import torch
from controls import ControlProjection,projector_distance,orthogonal_basis

class ControlsTests(unittest.TestCase):
    def test_seed_and_shared_initialization(self):
        models=[]
        for mode in ('B','C','D'):
            torch.manual_seed(42);models.append(ControlProjection([3,4,5],mode,dim=32,rank=8))
        b,c,d=models
        self.assertFalse(torch.equal(b.basis()[0],b.basis()[1]))
        self.assertTrue(torch.equal(c.basis(),d.basis()))
        self.assertTrue(torch.equal(c.basis(),b.basis()[:1].expand_as(c.basis())))
        for k,v in b.state_dict().items():
            if k.startswith('classifiers.'):self.assertTrue(torch.equal(v,c.state_dict()[k]) and torch.equal(v,d.state_dict()[k]))
    def test_frozen_vs_learning(self):
        for mode in ('B','C','D'):
            torch.manual_seed(42);m=ControlProjection([3,4,5],mode,dim=32,rank=8);before=m.raw_basis.detach().clone();head=m.classifiers[0].weight.detach().clone()
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001);z=torch.randn(20,3,32);_,logits=m(z)
            loss=sum(torch.nn.functional.cross_entropy(x,torch.zeros(20,dtype=torch.long)) for x in logits);loss.backward();opt.step()
            self.assertFalse(torch.equal(head,m.classifiers[0].weight))
            if mode!='D':self.assertTrue(torch.equal(before,m.raw_basis));self.assertIsNone(m.raw_basis.grad)
            else:self.assertFalse(torch.equal(m.basis()[0],m.basis()[1]))
    def test_projector_rotation_invariance_and_budget(self):
        torch.manual_seed(42);B=orthogonal_basis(torch.randn(32,8)).double();R=orthogonal_basis(torch.randn(8,8)).double()
        distance=projector_distance((B@R)[None],B,['s'])['s'];self.assertLess(distance,2e-6)
        w=torch.randn(7,dtype=torch.float64);w=w/w.norm();G=torch.randn(7,32,dtype=torch.float64);g=B.T@(w@G);S=g.norm();d=B@(g/S);delta=.01*w[:,None]*d[None,:]
        self.assertAlmostEqual(float(delta.norm()),.01,places=8);self.assertAlmostEqual(float((G*delta).sum()),float(.01*S),places=8)

if __name__=='__main__':unittest.main()
