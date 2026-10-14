"""Phase 2.1 CPU单元测试；无victim/GPU/训练循环/optimizer step。"""
import ast, copy, inspect, json, math, os, subprocess, sys, unittest
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from image_only_generator import ImageOnlyGenerator
from budget_projection import (DEFAULT_EPSILON, add_projected_delta, project_image,
                               projection_bounds, checked_epsilon)
from losses import (attribute_losses, joint_losses, retain_loss, retain_reference)
torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parent

def pixels(value=.5, shape=(1,3,8,8), dtype=torch.float32):
    return torch.full(shape,value,dtype=dtype)

def vector(index, scale=1., batch=1, dtype=torch.float32):
    x=torch.zeros(batch,512,dtype=dtype); x[:,index]=scale
    return x

class GeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); self.g=ImageOnlyGenerator(width=4,blocks=1)
    def test_forward_image_only_signature(self):
        self.assertEqual(list(inspect.signature(self.g.forward).parameters),["image"])
        with self.assertRaises(TypeError): self.g(pixels(),vector(0))
    def test_standard_shape(self):
        x=torch.rand(2,3,32,16); self.assertEqual(self.g(x).shape,x.shape)
    def test_official_resolution(self):
        x=torch.rand(1,3,384,128); self.assertEqual(self.g(x).shape,x.shape)
    def test_odd_resolution(self):
        x=torch.rand(2,3,35,19); self.assertEqual(self.g(x).shape,x.shape)
    def test_minimum_resolution(self): self.assertEqual(self.g(pixels()).shape,(1,3,8,8))
    def test_too_small_rejected(self):
        with self.assertRaises(ValueError): self.g(torch.rand(1,3,7,8))
    def test_single_round_budget(self):
        with torch.no_grad(): self.g.output.bias.fill_(100)
        delta=self.g(pixels()); self.assertLessEqual(float(delta.double().abs().max()),DEFAULT_EPSILON)
    def test_zero_budget(self):
        g=ImageOnlyGenerator(width=4,blocks=1,epsilon=0)
        self.assertFalse(g(pixels()).any())
    def test_invalid_budget(self):
        for eps in (-1,.1,float("nan"),float("inf"),True):
            with self.subTest(epsilon=eps), self.assertRaises(ValueError):
                ImageOnlyGenerator(epsilon=eps)
    def test_invalid_architecture(self):
        for kwargs in ({"width":0},{"blocks":0},{"width":True},{"blocks":1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): ImageOnlyGenerator(**kwargs)
    def test_invalid_pixels(self):
        for x in (pixels(-.1),pixels(1.1),pixels(float("nan")),torch.zeros(1,1,8,8),pixels().half()):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError): self.g(x)
    def test_repeat_exact(self):
        x=torch.rand(1,3,16,8); self.assertTrue(torch.equal(self.g(x),self.g(x)))
    def test_no_stateful_norm_or_text_branch(self):
        self.assertFalse(any(isinstance(m,(nn.Linear,nn.Embedding)) for m in self.g.modules()))
        self.assertFalse(list(self.g.buffers()))
    def test_generator_input_gradients(self):
        x=torch.rand(1,3,16,8,requires_grad=True); self.g(x).square().mean().backward()
        self.assertGreater(float(x.grad.norm()),0)
    def test_double_dtype(self):
        x=pixels(dtype=torch.float64); self.assertEqual(self.g.double()(x).dtype,torch.float64)

class ProjectionTests(unittest.TestCase):
    def test_cumulative_two_proposals(self):
        original=pixels(); cap=projection_bounds(original)[1]-original
        first=add_projected_delta(original,original,cap)
        second=add_projected_delta(original,first,cap)
        self.assertTrue(torch.equal(first,second))
        self.assertLessEqual(float((second.double()-original.double()).abs().max()),DEFAULT_EPSILON)
    def test_both_budget_signs_strict_no_tolerance(self):
        original=torch.linspace(0,1,3000).reshape(1,3,40,25)
        for proposed in (torch.full_like(original,-10),torch.full_like(original,10)):
            result=project_image(original,proposed)
            self.assertLessEqual(float((result.double()-original.double()).abs().max()),DEFAULT_EPSILON)
            self.assertGreaterEqual(float(result.min()),0); self.assertLessEqual(float(result.max()),1)
    def test_float64_budget_strict(self):
        original=torch.rand(1,3,16,8,dtype=torch.float64)
        for sign in (-1,1):
            result=project_image(original,original+sign)
            self.assertLessEqual(float((result-original).abs().max()),DEFAULT_EPSILON)
    def test_random_many_proposals(self):
        original=torch.rand(2,3,16,8); current=original.clone()
        # 只重复纯投影算子，不调用G/loss/optimizer，不实现K轮训练调度。
        for _ in range(20):
            delta=(torch.rand_like(original)-.5)*.06
            current=add_projected_delta(original,current,delta)
            self.assertLessEqual(float((current.double()-original.double()).abs().max()),DEFAULT_EPSILON)
    def test_pixels_zero_one(self):
        original=torch.cat([pixels(0),pixels(1)])
        result=project_image(original,original+torch.cat([-torch.ones_like(pixels()),torch.ones_like(pixels())]))
        self.assertTrue(torch.equal(result,original))
    def test_zero_epsilon_exact(self):
        original=pixels(); result=project_image(original,original+.1,epsilon=0)
        self.assertTrue(torch.equal(result,original))
    def test_interior_gradient_is_one(self):
        proposed=pixels().requires_grad_(); project_image(pixels(),proposed).sum().backward()
        self.assertTrue(torch.equal(proposed.grad,torch.ones_like(proposed)))
    def test_budget_saturation_gradient_is_zero(self):
        for sign in (-1,1):
            proposed=pixels(.5+sign*.1).requires_grad_()
            project_image(pixels(),proposed).sum().backward()
            self.assertFalse(proposed.grad.any())
    def test_pixel_saturation_gradient_is_zero(self):
        for anchor,proposal in ((0.,-.01),(1.,1.01)):
            proposed=pixels(proposal).requires_grad_()
            project_image(pixels(anchor),proposed).sum().backward()
            self.assertFalse(proposed.grad.any())
    def test_boundary_autograd_convention(self):
        lower,upper=projection_bounds(pixels())
        for endpoint in (lower,upper):
            proposed=endpoint.clone().requires_grad_()
            project_image(pixels(),proposed).sum().backward()
            self.assertTrue(torch.equal(proposed.grad,torch.ones_like(proposed)))
    def test_mixed_saturation_routes_only_interior(self):
        original=pixels(shape=(1,3,1,3))
        proposed=torch.tensor([.4,.5,.6]).reshape(1,1,1,3).expand_as(original).clone().requires_grad_()
        project_image(original,proposed).sum().backward()
        self.assertTrue(torch.equal(proposed.grad,torch.tensor([0.,1.,0.]).reshape(1,1,1,3).expand_as(original)))
    def test_original_reference_stop_gradient(self):
        original=pixels().requires_grad_(); proposed=pixels().requires_grad_()
        project_image(original,proposed).sum().backward()
        self.assertIsNone(original.grad); self.assertIsNotNone(proposed.grad)
    def test_delta_and_current_receive_interior_gradients(self):
        original=pixels(); current=pixels().requires_grad_(); delta=torch.zeros_like(original,requires_grad=True)
        add_projected_delta(original,current,delta).sum().backward()
        self.assertTrue(torch.equal(delta.grad,torch.ones_like(delta)))
        self.assertTrue(torch.equal(current.grad,torch.ones_like(current)))
    def test_no_implicit_detach_current(self):
        original=pixels(); current=(original+.001).requires_grad_()
        add_projected_delta(original,current,torch.zeros_like(current)).sum().backward()
        self.assertIsNotNone(current.grad)
    def test_interior_gradcheck(self):
        original=pixels(dtype=torch.float64); proposed=(original+.001).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(lambda p:project_image(original,p),(proposed,),fast_mode=True))
    def test_invalid_current_budget_rejected(self):
        with self.assertRaises(ValueError): add_projected_delta(pixels(),pixels(.6),torch.zeros_like(pixels()))
    def test_invalid_single_delta_rejected(self):
        with self.assertRaises(ValueError): add_projected_delta(pixels(),pixels(),pixels(.1))
    def test_proposal_metadata_and_nonfinite_rejected(self):
        for p in (pixels(dtype=torch.float64),pixels(float("inf")),torch.zeros(1,3,8,7)):
            with self.subTest(shape=p.shape), self.assertRaises(ValueError): project_image(pixels(),p)
    def test_epsilon_validation(self):
        with self.assertRaises(ValueError): checked_epsilon(DEFAULT_EPSILON+1e-8)

class LossTests(unittest.TestCase):
    def setUp(self):
        self.f=vector(0)+vector(1,2)+vector(2,3)
        self.z=vector(0); self.v=vector(1,3); self.previous=vector(2)
    def test_first_round_exact_numbers(self):
        r=attribute_losses(self.f,self.z,self.v)
        self.assertEqual({k:float(v) for k,v in r.items()},{"global":1.,"local":6.,"retain":0.,"total":7.})
    def test_recovery_penalized(self):
        r=attribute_losses(self.f,self.z,self.v,previous_jvp=self.previous,reference_score=torch.tensor([2.]))
        self.assertEqual(float(r["retain"]),1.); self.assertEqual(float(r["total"]),8.)
    def test_improved_previous_attribute_not_penalized(self):
        self.assertEqual(float(retain_loss(self.f,self.previous,torch.tensor([4.]))),0.)
    def test_reference_equality_relu_zero_grad(self):
        f=self.f.clone().requires_grad_()
        retain_loss(f,self.previous,torch.tensor([3.])).backward()
        self.assertFalse(f.grad.any())
    def test_reference_stop_gradient(self):
        f=self.f.clone().requires_grad_(); reference=torch.tensor([2.],requires_grad=True)
        retain_loss(f,self.previous,reference).backward()
        self.assertIsNone(reference.grad); self.assertTrue(torch.equal(f.grad,self.previous))
    def test_previous_feature_stop_gradient(self):
        previous_features=(self.f-.1).requires_grad_()
        reference=retain_reference(previous_features,self.previous)
        self.assertFalse(reference.requires_grad)
        f=self.f.clone().requires_grad_(); retain_loss(f,self.previous,reference).backward()
        self.assertIsNone(previous_features.grad)
    def test_first_round_zero_gradient(self):
        f=self.f.clone().requires_grad_(); retain_loss(f).backward(); self.assertFalse(f.grad.any())
    def test_only_immediate_previous_direction(self):
        f=self.f.clone().requires_grad_(); retain_loss(f,self.previous,torch.tensor([2.])).backward()
        self.assertTrue(torch.equal(f.grad,vector(2)))
    def test_retain_batch_mean(self):
        f=self.f.expand(2,-1).clone().requires_grad_()
        r=retain_loss(f,self.previous.expand(2,-1),torch.tensor([2.,4.])); r.backward()
        self.assertEqual(float(r),.5); self.assertEqual(float(f.grad[0,2]),.5); self.assertFalse(f.grad[1].any())
    def test_previous_reference_pair_required(self):
        for kwargs in ({"previous_jvp":self.previous},{"reference_score":torch.tensor([2.])}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): retain_loss(self.f,**kwargs)
    def test_reference_shape_rejected(self):
        with self.assertRaises(ValueError): retain_loss(self.f,self.previous,torch.tensor([[2.]]))
    def test_reference_nan_rejected(self):
        with self.assertRaises(ValueError): retain_loss(self.f,self.previous,torch.tensor([float("nan")]))
    def test_supervision_grad_rejected(self):
        for name in ("z_t","current_jvp"):
            z=self.z.clone(); v=self.v.clone()
            if name=="z_t": z.requires_grad_()
            else: v.requires_grad_()
            with self.subTest(name=name), self.assertRaises(ValueError): attribute_losses(self.f,z,v)
    def test_previous_JVP_grad_rejected(self):
        with self.assertRaises(ValueError): retain_loss(self.f,self.previous.requires_grad_(),torch.tensor([2.]))
    def test_global_local_minimize_positive_dot(self):
        plus=attribute_losses(self.f,self.z,self.v)["total"]
        minus=attribute_losses(-self.f,self.z,self.v)["total"]
        self.assertLess(float(minus),float(plus))
    def test_attribute_batch_reduction(self):
        r=attribute_losses(torch.cat([self.f,-self.f]),self.z.expand(2,-1),self.v.expand(2,-1))
        self.assertEqual(float(r["total"]),0.)
    def test_JVP_scale_not_normalized(self):
        a=attribute_losses(self.f,self.z,self.v)["local"]
        b=attribute_losses(self.f,self.z,2*self.v)["local"]
        self.assertEqual(float(b),2*float(a))
    def test_joint_exact_numbers(self):
        r=joint_losses(self.f,self.z,self.v,torch.tensor([True]))
        self.assertEqual(float(r["total"]),7.); self.assertEqual(set(r),{"global","local","total"})
    def test_joint_K0_zero_and_global_gradient(self):
        f=self.f.clone().requires_grad_()
        r=joint_losses(f,self.z,torch.zeros_like(self.v),torch.tensor([False])); r["total"].backward()
        self.assertEqual(float(r["local"]),0.); self.assertTrue(torch.equal(f.grad,self.z))
    def test_joint_mixed_mask_mean_all_samples(self):
        f=self.f.expand(2,-1); v=torch.cat([self.v,torch.zeros_like(self.v)])
        r=joint_losses(f,self.z.expand(2,-1),v,torch.tensor([True,False]))
        self.assertEqual(float(r["local"]),3.); self.assertEqual(float(r["total"]),4.)
    def test_joint_uses_raw_S_weighted_cached_target(self):
        jvps=torch.stack([vector(1,2)[0],vector(1,4)[0]])
        v=(torch.tensor([.6,.4])[:,None]*jvps).sum(0,keepdim=True)
        r=joint_losses(self.f,self.z,v,torch.tensor([True]))
        self.assertAlmostEqual(float(r["local"]),5.6,places=5)
    def test_joint_invalid_K0_nonzero_rejected(self):
        with self.assertRaises(ValueError): joint_losses(self.f,self.z,self.v,torch.tensor([False]))
    def test_joint_mask_dtype_shape_rejected(self):
        for mask in (torch.tensor([1]),torch.tensor(True)):
            with self.subTest(mask=mask), self.assertRaises(ValueError): joint_losses(self.f,self.z,self.v,mask)
    def test_invalid_feature_shape_nonfinite_and_dtype(self):
        for f in (torch.ones(1,511),torch.full((1,512),float("nan")),torch.ones(1,512,dtype=torch.int64)):
            with self.subTest(shape=f.shape), self.assertRaises(ValueError): attribute_losses(f,self.z,self.v)
    def test_global_gradient_is_z_plus_local(self):
        f=self.f.clone().requires_grad_(); attribute_losses(f,self.z,self.v)["total"].backward()
        self.assertTrue(torch.equal(f.grad,self.z+self.v))

class FrozenToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(3*8*8,512,bias=False)
        self.requires_grad_(False); self.eval()
    def forward(self, image):
        return F.normalize(self.projection(F.adaptive_avg_pool2d(image,(8,8)).flatten(1)),dim=1)

class GradientIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.g=ImageOnlyGenerator(width=4,blocks=1)
        self.encoder=FrozenToyEncoder()
        self.original=torch.rand(2,3,16,8)*.6+.2
        self.z=F.normalize(torch.randn(2,512),dim=1)
        self.local=torch.randn(2,512)*.01
    def assert_gradient(self, joint):
        state=copy.deepcopy(self.encoder.state_dict()); gstate=copy.deepcopy(self.g.state_dict())
        d=self.g(self.original); d.retain_grad()
        adv=add_projected_delta(self.original,self.original,d)
        f=self.encoder(adv)
        if joint:
            r=joint_losses(f,self.z,self.local,torch.tensor([True,True]))
        else:
            prev=torch.randn(2,512)*.01
            reference=retain_reference(self.encoder(self.original),prev)-.01
            r=attribute_losses(f,self.z,self.local,previous_jvp=prev,reference_score=reference)
        r["total"].backward()
        grads=[p.grad for p in self.g.parameters() if p.grad is not None]
        self.assertTrue(grads); self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(float(sum(g.square().sum() for g in grads).sqrt()),0)
        self.assertGreater(float(d.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in self.encoder.parameters()))
        self.assertTrue(all(torch.equal(state[k],v) for k,v in self.encoder.state_dict().items()))
        self.assertTrue(all(torch.equal(gstate[k],v) for k,v in self.g.state_dict().items()))
        self.assertLessEqual(float((adv.double()-self.original.double()).abs().max()),DEFAULT_EPSILON)
    def test_attribute_backprop_through_frozen_encoder_to_G(self): self.assert_gradient(False)
    def test_joint_backprop_through_frozen_encoder_to_G(self): self.assert_gradient(True)
    def test_saturated_projection_blocks_G_gradient(self):
        delta=self.g(self.original); delta.retain_grad()
        current=projection_bounds(self.original)[1]
        proposed=current+delta.abs()+.01
        f=self.encoder(project_image(self.original,proposed))
        joint_losses(f,self.z,self.local,torch.tensor([True,True]))["total"].backward()
        self.assertFalse(delta.grad.any())
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in self.g.parameters()))
    def test_actual_phase1_cache_cpu_joint_interface(self):
        source=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_v2_20261014/outputs/smoke")
        if not (source/"manifest.json").exists(): self.skipTest("optional original server smoke assets unavailable")
        import cache_io as io
        entries=io.load(source/"manifest.json")["entries"]
        self.assertEqual(len(entries),5)
        for e in entries:
            with self.subTest(row_id=e["row_id"]):
                v,_,_=io.read_entry(e)
                feature=v["z_t"].unsqueeze(0).clone().requires_grad_()
                r=joint_losses(feature,v["z_t"][None],v["v_local"][None],torch.tensor([v["has_local"]]))
                r["total"].backward()
                self.assertTrue(torch.allclose(feature.grad,v["z_t"][None]+v["v_local"][None]))
    def test_new_module_imports_are_independent(self):
        allowed={"torch","math","numbers","budget_projection"}
        for name in ("image_only_generator.py","budget_projection.py","losses.py"):
            imports=set()
            for node in ast.walk(ast.parse((ROOT/name).read_text())):
                if isinstance(node,ast.Import): imports.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node,ast.ImportFrom): imports.add(node.module.split(".")[0])
            self.assertLessEqual(imports,allowed)
    def test_import_and_backward_with_frontend_modules_forbidden(self):
        script="""import sys
from unittest.mock import patch
with patch.dict(sys.modules, {n:None for n in ("integration","conditioning","common","runtime","deps")}):
 import torch
 from image_only_generator import ImageOnlyGenerator
 from budget_projection import add_projected_delta
 from losses import joint_losses
 g=ImageOnlyGenerator(width=2,blocks=1)
 x=torch.full((1,3,8,8),.5)
 y=add_projected_delta(x,x,g(x))
 feature=y.flatten(1).repeat(1,3)[:,:512]
 z=torch.ones_like(feature).detach()
 joint_losses(feature,z,torch.zeros_like(z),torch.tensor([False]))["total"].backward()
 assert any(p.grad is not None for p in g.parameters())
 assert not torch.cuda.is_initialized()
"""
        result=subprocess.run([sys.executable,"-c",script],cwd=ROOT,
            env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
    def test_no_training_loop_file(self): self.assertFalse((ROOT/"train.py").exists())

if __name__=="__main__": unittest.main(verbosity=2)
