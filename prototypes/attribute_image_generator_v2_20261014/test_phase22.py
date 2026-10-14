"""Phase 2.2仅CPU合成样本，真实optimizer更新与单样本机制验证。"""
import ast, copy, gc, math, os, subprocess, sys, unittest, weakref
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from condition_schema import build
from image_only_generator import ImageOnlyGenerator
from budget_projection import DEFAULT_EPSILON, epsilon_tensor, projection_bounds
import serial_training as st
from serial_training import LossWeights, train_single_sample
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parent

def condition(k):
    gen=torch.Generator().manual_seed(100+k)
    z=F.normalize(torch.randn(512,generator=gen),dim=0)
    vectors=torch.randn(k,512,generator=gen)*.01
    scores=[float(k-i) for i in range(k)]
    weights=[s/sum(scores) for s in scores]
    return build({"row_id":"cuhk:10:0","image_rel_path":"synthetic/image.png","caption_index":0,
        "split":"train","role":"train","pid":1,"caption":"Synthetic caption."},
        [f"attr{i}" for i in range(k)],scores,weights,vectors,z)

class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(3*8*8,512,bias=False)
        self.register_buffer("sentinel",torch.tensor([1.]))
    def forward(self,image):
        return F.normalize(self.projection(F.adaptive_avg_pool2d(image,(8,8)).flatten(1)),dim=1)

class CountingSGD(torch.optim.SGD):
    def __init__(self,params,events=None,**kwargs):
        super().__init__(params,**kwargs); self.steps=0; self.zero_calls=0
        self.before=[]; self.after=[]; self.events=events
    def parameter_snapshot(self):
        return [p.detach().clone() for group in self.param_groups for p in group["params"]]
    def zero_grad(self,*args,**kwargs):
        self.zero_calls+=1
        if self.events is not None: self.events.append("zero")
        return super().zero_grad(*args,**kwargs)
    def step(self,*args,**kwargs):
        self.before.append(self.parameter_snapshot())
        result=super().step(*args,**kwargs)
        self.steps+=1; self.after.append(self.parameter_snapshot())
        if self.events is not None: self.events.append("step")
        return result

def setup(k=3,events=None):
    torch.manual_seed(42)
    g=ImageOnlyGenerator(width=2,blocks=1,epsilon=.004)
    encoder=ToyEncoder()
    optimizer=CountingSGD(g.parameters(),lr=.01,events=events)
    image=torch.rand(1,3,16,8)*.6+.2
    return g,encoder,optimizer,image,condition(k)

def same_state(a,b): return all(torch.equal(x,y) for x,y in zip(a,b))

class SerialBehaviorTests(unittest.TestCase):
    def test_K0_1_3_7_exact_steps_and_forwards(self):
        for k in (0,1,3,7):
            with self.subTest(K=k):
                g,e,opt,image,c=setup(k); forwards=[]; enc=[]
                h=g.register_forward_hook(lambda *args:forwards.append(1))
                eh=e.register_forward_hook(lambda *args:enc.append(1))
                r=train_single_sample(g,e,opt,image,c)
                h.remove(); eh.remove()
                self.assertEqual((opt.steps,opt.zero_calls,len(forwards),len(enc)),(k+1,)*4)
                self.assertEqual((r["optimizer_steps"],r["G_forwards"],r["encoder_forwards"]),(k+1,)*3)
                self.assertEqual([x["phase"] for x in r["records"]],["attribute"]*k+["joint"])
    def test_all_K_0_through_7_supported_without_cap(self):
        for k in range(8):
            with self.subTest(K=k):
                g,e,opt,image,c=setup(k)
                r=train_single_sample(g,e,opt,image,c)
                self.assertEqual(len(r["records"]),k+1)
                self.assertEqual([x["instance_id"] for x in r["records"][:-1]],c["selected_instance_ids"])
    def test_each_G_input_is_previous_detached_adversarial(self):
        g,e,opt,image,c=setup(7); inputs=[]; observed=[]
        h=g.register_forward_pre_hook(lambda module,args:inputs.append(
            (args[0].detach().clone(),args[0].requires_grad,args[0].grad_fn)))
        train_single_sample(g,e,opt,image,c,observer=observed.append); h.remove()
        self.assertTrue(torch.equal(inputs[0][0],image))
        for i in range(1,7): self.assertTrue(torch.equal(inputs[i][0],observed[i-1]["adversarial"]))
        self.assertTrue(all(not requires and fn is None for _,requires,fn in inputs))
    def test_next_forward_uses_immediately_updated_parameters(self):
        g,e,opt,image,c=setup(7); states=[]
        h=g.register_forward_pre_hook(lambda module,args:states.append([p.detach().clone() for p in g.parameters()]))
        train_single_sample(g,e,opt,image,c); h.remove()
        for i in range(1,8): self.assertTrue(same_state(states[i],opt.after[i-1]))
        self.assertTrue(all(not same_state(a,b) for a,b in zip(opt.before,opt.after)))
    def test_zero_forward_encode_backward_step_order(self):
        events=[]; g,e,opt,image,c=setup(3,events)
        h=g.register_forward_pre_hook(lambda *args:events.append("forward"))
        eh=e.register_forward_pre_hook(lambda *args:events.append("encode"))
        gh=next(g.parameters()).register_hook(lambda grad:events.append("backward"))
        train_single_sample(g,e,opt,image,c)
        h.remove(); eh.remove(); gh.remove()
        self.assertEqual(events,["zero","forward","encode","backward","step"]*4)
    def test_joint_restarts_original_with_updated_G(self):
        g,e,opt,image,c=setup(3); observed=[]; inputs=[]
        h=g.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
        r=train_single_sample(g,e,opt,image,c,observer=observed.append); h.remove()
        self.assertFalse(torch.equal(observed[2]["adversarial"],image))
        self.assertTrue(torch.equal(inputs[-1],image))
        self.assertTrue(torch.equal(observed[-1]["input_image"],image))
        self.assertEqual(r["records"][-1]["phase"],"joint")
        self.assertIsNone(r["records"][-1]["previous_instance_id"])
    def test_K0_only_global_joint(self):
        g,e,opt,image,c=setup(0)
        r=train_single_sample(g,e,opt,image,c)
        self.assertTrue(torch.equal(r["serial_adversarial"],image))
        rec=r["records"][0]
        self.assertEqual(rec["phase"],"joint"); self.assertEqual(rec["local"],0.)
        self.assertEqual(rec["retain"],0.); self.assertAlmostEqual(rec["total"],rec["global"])
    def test_Retain_previous_JVP_and_previous_image_score(self):
        g,e,opt,image,c=setup(7); observed=[]
        original=st.attribute_losses
        with patch("serial_training.attribute_losses",wraps=original) as spy:
            r=train_single_sample(g,e,opt,image,c,observer=observed.append)
        self.assertEqual(spy.call_count,7)
        for i,call in enumerate(spy.call_args_list):
            self.assertTrue(torch.equal(call.args[2],c["attribute_jvps"][i:i+1]))
            if i==0:
                self.assertIsNone(call.kwargs["previous_jvp"])
                self.assertIsNone(call.kwargs["reference_score"])
                self.assertEqual(r["records"][0]["retain"],0.)
            else:
                previous=c["attribute_jvps"][i-1:i]
                ref=(observed[i-1]["feature"]*previous).sum(1)
                self.assertTrue(torch.equal(call.kwargs["previous_jvp"],previous))
                self.assertTrue(torch.equal(call.kwargs["reference_score"],ref))
                self.assertFalse(ref.requires_grad); self.assertIsNone(call.kwargs["reference_score"].grad_fn)
                expected=torch.relu((observed[i]["feature"]*previous).sum(1)-ref).mean()
                self.assertAlmostEqual(r["records"][i]["retain"],float(expected),places=8)
    def test_no_old_forward_graphs_survive_into_next_round(self):
        g,e,opt,image,c=setup(7); refs=[]; alive_counts=[]
        def before(module,args):
            gc.collect()
            alive_counts.append(sum(ref() is not None for ref in refs))
        h=g.register_forward_pre_hook(before)
        gh=g.register_forward_hook(lambda module,args,out:refs.append(weakref.ref(out)))
        eh=e.register_forward_hook(lambda module,args,out:refs.append(weakref.ref(out)))
        train_single_sample(g,e,opt,image,c)
        h.remove(); gh.remove(); eh.remove(); gc.collect()
        self.assertEqual(alive_counts,[0]*8)
        self.assertTrue(all(ref() is None for ref in refs))
    def test_observer_and_return_values_have_no_graph(self):
        g,e,opt,image,c=setup(3); observed=[]
        r=train_single_sample(g,e,opt,image,c,observer=observed.append)
        for event in observed:
            for value in event.values():
                if torch.is_tensor(value):
                    self.assertFalse(value.requires_grad); self.assertIsNone(value.grad_fn)
        for key in ("serial_adversarial","joint_adversarial"):
            self.assertFalse(r[key].requires_grad); self.assertIsNone(r[key].grad_fn)
    def test_encoder_frozen_and_G_changes(self):
        g,e,opt,image,c=setup(3); old=copy.deepcopy(e.state_dict()); gb=copy.deepcopy(g.state_dict())
        train_single_sample(g,e,opt,image,c)
        self.assertFalse(e.training); self.assertTrue(all(not p.requires_grad and p.grad is None for p in e.parameters()))
        self.assertTrue(all(torch.equal(old[k],v) for k,v in e.state_dict().items()))
        self.assertTrue(any(not torch.equal(gb[k],v) for k,v in g.state_dict().items()))
    def test_original_input_and_condition_not_mutated_or_differentiated(self):
        g,e,opt,image,c=setup(3); image.requires_grad_(); before=image.detach().clone(); cb=copy.deepcopy(c)
        train_single_sample(g,e,opt,image,c)
        self.assertIsNone(image.grad); self.assertTrue(torch.equal(image.detach(),before))
        for k,v in c.items():
            if torch.is_tensor(v): self.assertTrue(torch.equal(v,cb[k]))
            else: self.assertEqual(v,cb[k])
    def test_logging_finite_and_budget_all_rounds(self):
        g,e,opt,image,c=setup(7); observed=[]
        r=train_single_sample(g,e,opt,image,c,observer=observed.append)
        for rec in r["records"]:
            for key in ("global","local","retain","total","G_gradient_l2","cumulative_linf","budget_saturation_ratio"):
                self.assertTrue(math.isfinite(rec[key]))
            self.assertLessEqual(rec["cumulative_linf"],DEFAULT_EPSILON)
            for key in ("budget_saturation_ratio","pixel_saturation_ratio","strict_projection_clipping_ratio"):
                self.assertTrue(0<=rec[key]<=1)
        self.assertTrue(all(torch.isfinite(x["adversarial"]).all() for x in observed))
    def test_joint_directly_uses_cached_v_local(self):
        g,e,opt,image,c=setup(3)
        original=st.joint_losses
        with patch("serial_training.joint_losses",wraps=original) as spy:
            train_single_sample(g,e,opt,image,c)
        self.assertEqual(spy.call_count,1)
        self.assertTrue(torch.equal(spy.call_args.args[1],c["z_t"][None]))
        self.assertTrue(torch.equal(spy.call_args.args[2],c["v_local"][None]))
    def test_Adam_runs_without_graph_reuse(self):
        g,e,_,image,c=setup(7); opt=torch.optim.Adam(g.parameters(),lr=.001)
        r=train_single_sample(g,e,opt,image,c)
        self.assertEqual(r["optimizer_steps"],8)
        self.assertTrue(all(int(state["step"])==8 for state in opt.state.values()))

class PositiveGenerator(nn.Module):
    def __init__(self):
        super().__init__(); self.bias=nn.Parameter(torch.tensor(1.))
    def forward(self,image):
        return (epsilon_tensor(image)*torch.tanh(self.bias)).expand_as(image)

class MeanEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,image):
        x=image.mean(dim=(1,2,3))[:,None]*self.scale
        return x.expand(-1,512)

class BudgetAndWeightsTests(unittest.TestCase):
    def test_original_anchor_budget_not_current_anchor_and_saturated_grad(self):
        g=PositiveGenerator(); e=MeanEncoder(); opt=CountingSGD(g.parameters(),lr=.001)
        image=torch.full((1,3,8,8),.5); c=condition(7); observed=[]
        r=train_single_sample(g,e,opt,image,c,observer=observed.append)
        upper=projection_bounds(image)[1]
        self.assertTrue(torch.equal(observed[1]["adversarial"],upper))
        self.assertFalse(torch.equal(observed[-1]["adversarial"],upper))
        for rec in r["records"]:
            self.assertLessEqual(rec["cumulative_linf"],DEFAULT_EPSILON)
        for rec in r["records"][1:7]:
            self.assertEqual(rec["budget_saturation_ratio"],1.)
            self.assertEqual(rec["strict_projection_clipping_ratio"],1.)
            self.assertEqual(rec["G_gradient_l2"],0.)
        self.assertGreater(r["records"][-1]["G_gradient_l2"],0)
    def test_configurable_weights_applied_to_each_loss(self):
        g,e,opt,image,c=setup(3)
        weights=LossWeights(.2,3.,4.)
        r=train_single_sample(g,e,opt,image,c,weights=weights)
        for rec in r["records"]:
            expected=.2*rec["global"]+3*rec["local"]+4*rec["retain"]
            self.assertAlmostEqual(rec["total"],expected,places=6)
    def test_zero_weights_still_exact_step_count(self):
        g,e,opt,image,c=setup(3); before=opt.parameter_snapshot()
        r=train_single_sample(g,e,opt,image,c,weights=LossWeights(0,0,0))
        self.assertEqual(opt.steps,4); self.assertTrue(same_state(before,opt.parameter_snapshot()))
        self.assertTrue(all(x["total"]==0 and x["G_gradient_l2"]==0 for x in r["records"]))
    def test_default_weights_are_one(self):
        self.assertEqual(LossWeights(),LossWeights(1,1,1))
    def test_invalid_weight_values(self):
        for value in (-1.,float("nan"),float("inf"),True,"1"):
            for name in ("lambda_global","lambda_local","lambda_retain"):
                with self.subTest(value=value,name=name),self.assertRaises(ValueError):
                    LossWeights(**{name:value})
    def test_float64_serial_loop(self):
        g,e,opt,image,c=setup(3); g.double(); e.double(); image=image.double()
        # optimizer参数引用仍是同一组Parameter。
        r=train_single_sample(g,e,opt,image,c)
        self.assertEqual(r["joint_adversarial"].dtype,torch.float64)
        self.assertTrue(all(x["cumulative_linf"]<=DEFAULT_EPSILON for x in r["records"]))

class RejectionTests(unittest.TestCase):
    def test_invalid_condition_no_optimizer_step(self):
        g,e,opt,image,c=setup(3); c["S_raw"]=c["S_raw"].flip(0)
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_training_target_requires_grad_rejected(self):
        g,e,opt,image,c=setup(3); c["attribute_jvps"].requires_grad_()
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
    def test_multiple_samples_rejected(self):
        g,e,opt,image,c=setup()
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image.expand(2,-1,-1,-1),c)
        self.assertEqual(opt.steps,0)
    def test_encoder_parameters_in_optimizer_rejected(self):
        g,e,_,image,c=setup(); opt=CountingSGD([*g.parameters(),*e.parameters()],lr=.01)
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_missing_G_parameter_rejected(self):
        g,e,_,image,c=setup(); opt=CountingSGD([next(g.parameters())],lr=.01)
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
    def test_nonfinite_input_rejected(self):
        g,e,opt,image,c=setup(); image[0,0,0,0]=float("nan")
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_no_grad_context_rejected(self):
        g,e,opt,image,c=setup()
        with torch.no_grad(),self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_detached_encoder_output_rejected(self):
        g,e,opt,image,c=setup()
        original=e.forward
        e.forward=lambda x:original(x).detach()
        with self.assertRaisesRegex(ValueError,"gradient path"): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_nonfinite_encoder_output_rejected_before_step(self):
        g,e,opt,image,c=setup(); e.forward=lambda x:torch.full((1,512),float("nan"))
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_nonfinite_gradients_rejected_before_step(self):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx,x): return x.clone()
            @staticmethod
            def backward(ctx,grad): return torch.full_like(grad,float("nan"))
        g,e,opt,image,c=setup(); original=e.forward
        e.forward=lambda x:BadGradient.apply(original(x))
        with self.assertRaisesRegex(ValueError,"gradients"): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)
    def test_nonfinite_optimizer_update_is_detected(self):
        g,e,opt,image,c=setup()
        original=opt.step
        def bad_step():
            original()
            with torch.no_grad(): next(g.parameters()).fill_(float("nan"))
        opt.step=bad_step
        with self.assertRaisesRegex(ValueError,"optimizer produced"): train_single_sample(g,e,opt,image,c)
    def test_non_LossWeights_config_rejected(self):
        g,e,opt,image,c=setup()
        with self.assertRaises(ValueError): train_single_sample(g,e,opt,image,c,weights={"lambda_global":1})
    def test_mutating_encoder_buffer_detected(self):
        g,e,opt,image,c=setup(); original=e.forward
        def bad(x):
            e.sentinel.add_(1); return original(x)
        e.forward=bad
        with self.assertRaisesRegex(ValueError,"buffers"): train_single_sample(g,e,opt,image,c)
        self.assertEqual(opt.steps,0)

class IndependenceTests(unittest.TestCase):
    def test_import_allowlist(self):
        allowed={"dataclasses","math","numbers","torch","condition_schema","budget_projection","losses"}
        imports=set()
        for n in ast.walk(ast.parse((ROOT/"serial_training.py").read_text())):
            if isinstance(n,ast.Import): imports.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n,ast.ImportFrom): imports.add(n.module.split(".")[0])
        self.assertLessEqual(imports,allowed)
    def test_training_without_frontend_imports_or_CUDA(self):
        script="""import sys
from unittest.mock import patch
with patch.dict(sys.modules,{n:None for n in ("integration","common","conditioning","runtime","deps","selection_audit","attribute_tta")}):
 from test_phase22 import setup
 from serial_training import train_single_sample
 import torch
 g,e,opt,image,c=setup(3)
 r=train_single_sample(g,e,opt,image,c)
 assert opt.steps==4
 assert not torch.cuda.is_initialized()
"""
        p=subprocess.run([sys.executable,"-c",script],cwd=ROOT,env={**os.environ,
             "CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=="__main__": unittest.main(verbosity=2)
