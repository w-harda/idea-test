import copy,json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
import common as c
from generator_model import ConditionalGenerator,compose,generate_once,training_loss,total_variation
from conditioning import aggregate_condition
torch.set_num_threads(2)
def conditions(n=2):
 return F.normalize(torch.randn(n,512),dim=1),F.normalize(torch.randn(n,512),dim=1)
def plan():
 out=[]
 for i,s in enumerate((2.,1.)):
  e=[0.]*512;e[i]=1.
  out.append(SimpleNamespace(instance_id=str(i),metadata={"S_raw":s,"semantic_representation":{"projected_context":e}}))
 return out
class GeneratorTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(42);self.g=ConditionalGenerator(width=4,blocks=1,condition_hidden=8);self.im=torch.rand(2,3,32,16);self.z,self.a=conditions()
 def test_delta_shape(self):self.assertEqual(self.g(self.im,self.z,self.a).shape,self.im.shape)
 def test_odd_dimensions(self):
  image=torch.rand(2,3,35,19);self.assertEqual(self.g(image,self.z,self.a).shape,image.shape)
 def test_requested_budget(self):self.assertLessEqual(float(self.g(self.im,self.z,self.a).abs().max()),8/255+1e-7)
 def test_clipped_effective_budget(self):
  x=torch.cat([torch.zeros(1,3,32,16),torch.ones(1,3,32,16)]);d=torch.full_like(x,8/255)
  y=compose(x,d);self.assertLessEqual(float((y-x).abs().max()),8/255+1e-7);self.assertTrue((y>=0).all() and (y<=1).all())
 def test_condition_shape(self):
  with self.assertRaises(ValueError):self.g(self.im,self.z[:1],self.a)
 def test_condition_requires_grad(self):
  with self.assertRaises(ValueError):self.g(self.im,self.z.requires_grad_(),self.a)
 def test_pixels_invalid(self):
  with self.assertRaises(ValueError):self.g(self.im-2,self.z,self.a)
 def test_nonfinite_condition(self):
  self.z[0,0]=float("nan")
  with self.assertRaises(ValueError):self.g(self.im,self.z,self.a)
 def test_budget_not_increased(self):
  with self.assertRaises(ValueError):ConditionalGenerator(epsilon=.1)
 def test_one_forward_no_backward(self):
  calls=[];h=self.g.register_forward_hook(lambda *a:calls.append(1))
  with patch("torch.autograd.grad",side_effect=AssertionError):
   r=generate_once(self.g,self.im,self.z,self.a)
  h.remove();self.assertEqual(len(calls),1);self.assertFalse(r["adversarial"].requires_grad)
 def test_repeat_exact(self):
  r=generate_once(self.g,self.im,self.z,self.a);q=generate_once(self.g,self.im,self.z,self.a)
  self.assertTrue(torch.equal(r["adversarial"],q["adversarial"]))
 def test_batch_independence(self):
  batch=generate_once(self.g,self.im,self.z,self.a)["delta"]
  single=generate_once(self.g,self.im[:1],self.z[:1],self.a[:1])["delta"]
  self.assertTrue(torch.allclose(batch[:1],single,atol=1e-7,rtol=1e-5))
 def test_caption_condition_used(self):
  x=self.g(self.im,self.z,self.a);y=self.g(self.im,self.z.flip(0),self.a);self.assertGreater(float((x-y).abs().max()),0)
 def test_attribute_condition_used(self):
  x=self.g(self.im,self.z,self.a);y=self.g(self.im,self.z,self.a.flip(0));self.assertGreater(float((x-y).abs().max()),0)
 def test_no_train_eval_buffer_change(self):
  state={k:v.clone() for k,v in self.g.state_dict().items()};generate_once(self.g,self.im,self.z,self.a)
  self.assertTrue(all(torch.equal(v,state[k]) for k,v in self.g.state_dict().items()))
class ConditionTests(unittest.TestCase):
 def setUp(self):
  self.p=plan();self.z=F.normalize(torch.ones(1,512),dim=1);self.v={"0":torch.eye(512)[2:3]*3,"1":torch.eye(512)[3:4]*6}
 def test_score_weights(self):
  x,w=aggregate_condition(self.z,self.p,self.v);self.assertAlmostEqual(w[0],2/3);self.assertAlmostEqual(w[1],1/3)
 def test_context_pool(self):
  x,w=aggregate_condition(self.z,self.p,self.v);self.assertAlmostEqual(float(x["c_attr"][0]),2/3,places=6);self.assertAlmostEqual(float(x["c_attr"][1]),1/3,places=6)
 def test_JVP_scale_retained(self):
  x,_=aggregate_condition(self.z,self.p,self.v);self.assertAlmostEqual(float(x["v_local"][2]),2.);self.assertAlmostEqual(float(x["v_local"][3]),2.)
 def test_metadata_unchanged(self):
  old=copy.deepcopy([a.metadata for a in self.p]);aggregate_condition(self.z,self.p,self.v);self.assertEqual(old,[a.metadata for a in self.p])
 def test_K0(self):
  x,w=aggregate_condition(self.z,[],{});self.assertEqual(w,[]);self.assertFalse(x["has_local"]);self.assertFalse(x["v_local"].any())
 def test_S_scale_invariant(self):
  x,_=aggregate_condition(self.z,self.p,self.v)
  for a in self.p:a.metadata["S_raw"]*=10
  y,_=aggregate_condition(self.z,self.p,self.v);self.assertTrue(torch.equal(x["c_attr"],y["c_attr"]))
 def test_score_invalid(self):
  self.p[0].metadata["S_raw"]=0
  with self.assertRaises(ValueError):aggregate_condition(self.z,self.p,self.v)
 def test_zero_representation_rejected(self):
  self.p[0].metadata["semantic_representation"]["projected_context"]=[0.]*512
  with self.assertRaises(ValueError):aggregate_condition(self.z,self.p,self.v)
 def test_conditions_detached(self):
  z=self.z.requires_grad_();x,_=aggregate_condition(z,self.p,self.v);self.assertFalse(any(v.requires_grad for v in x.values() if torch.is_tensor(v)))
class LossTests(unittest.TestCase):
 def test_global_minimizes_matching(self):
  z=torch.eye(512)[:1];d=torch.zeros(1,3,16,16)
  a,_=training_loss(z,z,torch.zeros_like(z),torch.tensor([False]),d)
  b,_=training_loss(-z,z,torch.zeros_like(z),torch.tensor([False]),d);self.assertLess(float(b),float(a))
 def test_local_minimizes_JVP_dot(self):
  z=torch.eye(512)[:1];v=torch.eye(512)[1:2];d=torch.zeros(1,3,16,16)
  a,_=training_loss(v,z,v,torch.tensor([True]),d,global_weight=0)
  b,_=training_loss(-v,z,v,torch.tensor([True]),d,global_weight=0);self.assertLess(float(b),float(a))
 def test_K0_local_zero(self):
  z=F.normalize(torch.ones(1,512),dim=1);_,terms=training_loss(z,z,torch.zeros_like(z),torch.tensor([False]),torch.zeros(1,3,16,16))
  self.assertEqual(float(terms["local"]),0.)
 def test_TV_constant_zero(self):self.assertEqual(float(total_variation(torch.ones(1,3,16,16))),0.)
 def test_TV_nonconstant_positive(self):self.assertGreater(float(total_variation(torch.randn(1,3,16,16))),0.)
 def test_regularization_units(self):
  z=torch.eye(512)[:1];_,terms=training_loss(z,z,z,torch.tensor([False]),torch.full((1,3,16,16),8/255))
  self.assertAlmostEqual(float(terms["perturbation"]),1.)
 def test_frozen_targets(self):
  z=torch.eye(512)[:1].requires_grad_()
  with self.assertRaises(ValueError):training_loss(z,z,z,torch.tensor([True]),torch.zeros(1,3,16,16))
 def test_gradient_through_frozen_surrogate_only_updates_G(self):
  from train import step
  torch.manual_seed(42);g=ConditionalGenerator(width=4,blocks=1,condition_hidden=8)
  class Toy(nn.Module):
   def __init__(self):super().__init__();self.linear=nn.Linear(3*32*16,512,bias=False);self.requires_grad_(False)
   def encode_images(self,x):return F.normalize(self.linear(x.flatten(1)),dim=1)
  victim=Toy();before=copy.deepcopy(g.state_dict());vbefore=copy.deepcopy(victim.state_dict())
  z,a=conditions();batch={"image":torch.rand(2,3,32,16),"z_t":z,"c_attr":a,"v_local":a,"has_local":torch.ones(2,dtype=torch.bool)}
  config={"training":{"global_weight":1.,"local_weight":1.,"perturbation_weight":.001,"tv_weight":.001}}
  opt=torch.optim.Adam(g.parameters(),lr=1e-3);stats=step(g,victim,batch,opt,config,diagnose=True)
  self.assertGreater(stats["G_gradient_norm"],0);self.assertGreater(stats["global_input_gradient_norm"],0);self.assertGreater(stats["local_input_gradient_norm"],0)
  self.assertTrue(any(not torch.equal(before[k],v) for k,v in g.state_dict().items()))
  self.assertTrue(all(torch.equal(vbefore[k],v) for k,v in victim.state_dict().items()));self.assertTrue(all(p.grad is None for p in victim.parameters()))
class GateTests(unittest.TestCase):
 def test_prepare_default_no_GPU(self):
  import prepare
  with patch.dict(sys.modules,{"integration":None}):prepare.main([])
 def test_training_requires_flag(self):
  import train
  with self.assertRaises(SystemExit):train.main([])
 def test_unseen_caption_no_hidden_backward(self):
  import generate
  with patch.object(c,"checked_cache",return_value={"train":[],"validation":[]}):
   with self.assertRaisesRegex(ValueError,"uncached"):
    generate.main(["--checkpoint","not_loaded","--cache","not_loaded","--row-id","unknown","--output",str(c.ROOT/"outputs/not_created.pt")])
 def test_write_boundary(self):
  with self.assertRaises(ValueError):c.safe("/home/lzf/TBPS/Datasets/x")
 def test_hash_corruption(self):
  (c.ROOT/"outputs").mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=c.ROOT/"outputs") as d:
   p=Path(d)/"x";p.write_text("x");h={str(p):c.sha(p)};p.write_text("y")
   with self.assertRaises(ValueError):c.verify(h)
if __name__=="__main__":unittest.main()
