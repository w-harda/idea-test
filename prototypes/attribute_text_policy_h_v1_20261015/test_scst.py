"""合成训练图文特征的奖励、stop-gradient和H更新验证。"""
import math, unittest
import torch
from test_policy import example, model
from rewards import TrainGallery, RewardEvaluator
from scst import scst_loss, synthetic_scst_step

def features():
    clean=torch.zeros(3,512);clean[0,0]=1;clean[1,1]=1;clean[2,2]=1
    adv=clean.clone();adv[0,:3]=torch.tensor([.8,.6,0.])
    return clean,adv

def context(**kw):
    clean,adv=features()
    values={"clean":clean,"adversarial":adv,
            "positive_mask":torch.tensor([True,False,False]),"paired_index":0,
            "query_split":"train","gallery_splits":("train",)*3,
            "query_role":"train","gallery_roles":("train",)*3}
    return TrainGallery(**{**values,**kw})

def encoder(texts):
    # 显式合成：从字符串决定单位特征，不加载任何真实模型。
    result=torch.zeros(len(texts),512)
    for i,t in enumerate(texts):
        if t=="coat":result[i,0]=1
        else:result[i,1]=1
    return result

class RewardTests(unittest.TestCase):
    def test_exact_sim_margin_and_synergy_values(self):
        r=RewardEvaluator(context(),encoder).evaluate_many("coat",("cоat",))[0]
        self.assertAlmostEqual(r.R_sim,.2,places=6)
        self.assertAlmostEqual(r.R_rank,1.2,places=6)
        self.assertAlmostEqual(r.clean_sim_drop,1.,places=6)
        self.assertAlmostEqual(r.synergy_increment,-.8,places=6)
        self.assertAlmostEqual(r.total,1.4,places=6)
    def test_unchanged_caption_zero_reward(self):
        r=RewardEvaluator(context(),encoder).evaluate_many("coat",("coat",))[0]
        self.assertEqual(r.total,0.)
        self.assertEqual(r.R_sim,0.)
        self.assertEqual(r.R_rank,0.)
    def test_reward_weights_configurable(self):
        r=RewardEvaluator(context(),encoder,lambda_sim=2,lambda_rank=3).evaluate_many("coat",("changed",))[0]
        self.assertAlmostEqual(r.total,4.,places=6)
    def test_invalid_reward_weights(self):
        for a,b in ((-1,1),(0,0),(float("nan"),1)):
            with self.assertRaises(ValueError):RewardEvaluator(context(),encoder,a,b)
    def test_test_or_validation_feedback_rejected(self):
        for split in ("test","val","validation"):
            with self.subTest(split=split),self.assertRaises(ValueError):context(query_split=split)
            with self.subTest(split=split),self.assertRaises(ValueError):context(gallery_splits=("train",split,"train"))
    def test_internal_validation_role_rejected_even_when_official_train(self):
        with self.assertRaises(ValueError):context(query_role="validation")
        with self.assertRaises(ValueError):context(gallery_roles=("train","validation","train"))
    def test_missing_positive_or_negative_rejected(self):
        for mask in (torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool)):
            with self.assertRaises(ValueError):context(positive_mask=mask)
    def test_wrong_paired_image_rejected(self):
        for index in (1,-1,True,99):
            with self.assertRaises(ValueError):context(paired_index=index)
    def test_unfrozen_gallery_rejected(self):
        clean,adv=features();clean.requires_grad_()
        with self.assertRaises(ValueError):context(clean=clean)
    def test_mismatched_or_nonfinite_gallery_rejected(self):
        clean,adv=features()
        with self.assertRaises(ValueError):context(adversarial=adv[:2])
        clean[0,0]=float("nan")
        with self.assertRaises(ValueError):context(clean=clean)
    def test_reward_callback_no_grad_and_text_only(self):
        calls=[]
        def callback(texts):
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(all(isinstance(t,str) for t in texts))
            calls.append(texts);return encoder(texts)
        RewardEvaluator(context(),callback).evaluate_many("coat",("changed","coat"))
        self.assertEqual(calls,[("coat","changed","coat")])
    def test_multi_positive_uses_best_positive_margin(self):
        ctx=context(positive_mask=torch.tensor([True,True,False]))
        r=RewardEvaluator(ctx,encoder).evaluate_many("coat",("changed",))[0]
        self.assertAlmostEqual(r.R_rank,-.2,places=6)
    def test_callback_invalid_features_rejected(self):
        with self.assertRaises(ValueError):
            RewardEvaluator(context(),lambda texts:torch.ones(len(texts),512)).evaluate_many("coat",("changed",))

class SCSTTests(unittest.TestCase):
    def test_positive_advantage_sign(self):
        lp=torch.tensor(-2.,requires_grad=True);loss,a=scst_loss(lp,3.,1.)
        self.assertEqual(float(loss),4.);loss.backward()
        self.assertEqual(float(lp.grad),-2.)
    def test_negative_advantage_sign(self):
        lp=torch.tensor(-2.,requires_grad=True);loss,a=scst_loss(lp,1.,3.)
        loss.backward();self.assertEqual(float(lp.grad),2.)
    def test_reward_and_baseline_stop_gradient(self):
        lp=torch.tensor(-2.,requires_grad=True)
        sr=torch.tensor(3.,requires_grad=True);gr=torch.tensor(1.,requires_grad=True)
        loss,a=scst_loss(lp,sr,gr);loss.backward()
        self.assertIsNone(sr.grad);self.assertIsNone(gr.grad)
        self.assertFalse(a.requires_grad)
    def test_equal_rewards_zero_gradient(self):
        lp=torch.tensor(-2.,requires_grad=True);loss,_=scst_loss(lp,1.,1.);loss.backward()
        self.assertEqual(float(lp.grad),0.)
    def test_nonfinite_rewards_rejected(self):
        with self.assertRaises(ValueError):scst_loss(torch.tensor(-1.),float("nan"),0.)
    def test_synthetic_optimizer_updates_H_with_finite_gradients(self):
        torch.manual_seed(7);p=model();opt=torch.optim.Adam(p.parameters(),lr=.001)
        ev=RewardEvaluator(context(),encoder)
        before={n:x.detach().clone() for n,x in p.named_parameters()}
        records=[]
        # 少量合成step寻找非零advantage；不使用真实Caption/图像。
        for seed in range(12):
            records.append(synthetic_scst_step(p,example(),ev,opt,torch.Generator().manual_seed(seed)))
            if any(not torch.equal(before[n],x) for n,x in p.named_parameters()):break
        self.assertTrue(any(not torch.equal(before[n],x) for n,x in p.named_parameters()))
        self.assertTrue(any(r["gradient_norm"]>0 for r in records))
        self.assertTrue(all(math.isfinite(r["loss"]) for r in records))
    def test_optimizer_cannot_contain_external_model(self):
        p=model();external=torch.nn.Parameter(torch.ones(1))
        opt=torch.optim.Adam([*p.parameters(),external])
        with self.assertRaises(ValueError):
            synthetic_scst_step(p,example(),RewardEvaluator(context(),encoder),opt)
    def test_gallery_features_unchanged_after_H_step(self):
        p=model();ctx=context();before=(ctx.clean.clone(),ctx.adversarial.clone())
        synthetic_scst_step(p,example(),RewardEvaluator(ctx,encoder),
                            torch.optim.Adam(p.parameters()),torch.Generator().manual_seed(1))
        self.assertTrue(torch.equal(ctx.clean,before[0]))
        self.assertTrue(torch.equal(ctx.adversarial,before[1]))
    def test_K0_scst_step_safe_identity(self):
        from contracts import TextExample
        p=model();out=synthetic_scst_step(p,TextExample("coat",()),RewardEvaluator(context(),encoder),
                    torch.optim.Adam(p.parameters()),torch.Generator().manual_seed(2))
        self.assertEqual(out["sampled"]["adversarial"],"coat")
        self.assertEqual(out["advantage"],0.)
        self.assertEqual(out["gradient_norm"],0.)
    def test_policy_output_independent_of_training_pair_context(self):
        p=model();x=example()
        before=p.attack(x).record()
        ctx=context(paired_index=1,positive_mask=torch.tensor([False,True,False]))
        RewardEvaluator(ctx,encoder)  # 完全不同配对只能改变奖励，不能成为策略输入。
        self.assertEqual(before,p.attack(x).record())

if __name__=="__main__":unittest.main()
