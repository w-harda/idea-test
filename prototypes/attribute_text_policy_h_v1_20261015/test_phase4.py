"""Phase4 CPU：SCST真实接口核心、事务Checkpoint与精确恢复，不运行GPU。"""
from dataclasses import replace
from pathlib import Path
import json
import random
import tempfile
import subprocess
import sys
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from test_phase2 import fixture,GMock
from test_phase3 import EncoderFixture
from tokenizer_probe import official_probe
from text_frontend import CaptionFrontend
from real_feedback import InferenceRuntime
from phase3_inputs import import_gv2
from phase4_inputs import query_order,validate_config
from phase4_training import train_step,optimizer_for,check_optimizer,action_stats
from phase4_real_reward import FreshReward
from phase4_checkpoint import (capture_rng,restore_rng,same,begin_step,publish,
                               latest,restore,verify_reload)
from feedback_backend import state_sha
from train_phase4 import policy_factory
from phase2_sources import ROOT,digest,sha
import train_phase4

class SCSTFixture:
    def __init__(self,total=8):
        self.samples,_,_,self.catalog=fixture()
        self.gallery=self.samples[:3];self.queries=self.samples[:3]
        self.probe=official_probe();self.frontend=CaptionFrontend()
        self.config=json.loads((ROOT/"config_phase4.json").read_text())
        self.config["architecture"]={"width":16,"layers":1,"heads":4,"max_bytes":1024}
        self.config["total_steps"]=total
        self.protocol={"fixture":"CPU only, no real training","config":self.config,
                       "query_order":query_order(self.queries,total)}
        self.policy=policy_factory(self.probe,self.config)
        self.opt=optimizer_for(self.policy,self.config)
        self.sampler=torch.Generator().manual_seed(42)
        inference,_=import_gv2()
        self.runtime=InferenceRuntime(GMock(),EncoderFixture(),inference.generate_once,self.probe,"cpu",
                            {"G":0,"clean":0,"adv":0,"text":3*total},"cpu_test_fixture")
        rng=torch.Generator().manual_seed(9)
        self.features={k:torch.nn.functional.normalize(torch.randn(3,512,generator=rng),dim=1)
                       for k in ("clean","adversarial")}
        self.lookup={s.row_id:s for s in self.queries}
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
    def step(self,step):
        q=self.lookup[self.protocol["query_order"][step-1]]
        evaluator=lambda k:FreshReward(self.catalog,q,self.gallery,self.features,self.runtime,self.frontend,k)
        return train_step(self.policy,self.opt,self.frontend(q.caption),self.sampler,evaluator,step,q.row_id)
    def close(self):self.runtime.close()

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.f=SCSTFixture()
        self.addCleanup(self.f.close)
    def test_real_interface_scalar_backward(self):
        records=[self.f.step(i)[0] for i in range(1,5)]
        self.assertTrue(any(r["gradient_norm"]>0 for r in records))
        self.assertTrue(any(r["parameter_delta_l2"]>0 for r in records))
        self.assertEqual(self.f.runtime.samples,{"G":0,"clean":0,"adv":0,"text":12})
        self.assertEqual(self.f.runtime.calls["text"],4)
        self.f.runtime.finish()
    def test_reward_formula_and_synergy_preserved(self):
        record,_=self.f.step(1)
        for mode in ("sampled","greedy"):
            r=record[mode+"_reward"]
            self.assertEqual(r["total"],r["R_sim"]+r["R_rank"])
            self.assertEqual(r["synergy_increment"],r["adv_sim_drop"]-r["clean_sim_drop"])
        self.assertEqual(record["advantage"],float(torch.tensor(
            record["sampled_reward"]["total"]-record["greedy_reward"]["total"],dtype=torch.float32)))
    def test_no_frozen_model_gradients(self):
        self.f.step(1)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in self.f.runtime.models for p in m.parameters()))
    def test_only_H_optimizer(self):
        wrong=torch.optim.Adam(list(self.f.policy.parameters())+list(self.f.runtime.generator.parameters()))
        with self.assertRaises(ValueError):check_optimizer(self.f.policy,wrong)
    def test_duplicate_optimizer_parameter_rejected(self):
        opt=self.f.opt;opt.param_groups[0]["params"].append(opt.param_groups[0]["params"][0])
        with self.assertRaises(ValueError):check_optimizer(self.f.policy,opt)
    def test_inference_H_context_rejected(self):
        with torch.inference_mode(),self.assertRaises(ValueError):self.f.step(1)
    def test_current_H_state_bound_to_log(self):
        before=state_sha(self.f.policy);r,_=self.f.step(1)
        self.assertEqual(before,r["H_state_sha256_before"])
        self.assertEqual(state_sha(self.f.policy),r["H_state_sha256_after"])
    def test_unicode_budget_and_trace(self):
        r,_=self.f.step(1)
        self.assertLessEqual(len(r["sampled"]["edits"]),r["K"])
        self.assertLessEqual(len(r["greedy"]["edits"]),r["K"])
        self.assertIn("STOP_ratio",r["sampled_action_stats"])
    def test_fresh_text_each_step(self):
        self.f.step(1);before=self.f.runtime.calls["text"]
        self.f.step(2)
        self.assertEqual(self.f.runtime.calls["text"],before+1)
    def test_validation_feedback_rejected(self):
        q=self.f.samples[3]
        with self.assertRaises(ValueError):
            FreshReward(self.f.catalog,q,self.f.gallery,self.f.features,self.f.runtime,self.f.frontend,1)
    def test_text_features_detached_and_finite(self):
        r,x=self.f.step(1)
        self.assertEqual(x.shape,(3,512));self.assertFalse(x.requires_grad)
        self.assertTrue(torch.isfinite(x).all());self.assertEqual(r["IRRA_text_samples"],3)
    def test_zero_advantage_does_not_force_edits(self):
        from rewards import Reward
        class Eval:
            runtime=self.f.runtime
            features=torch.nn.functional.normalize(torch.ones(3,512),dim=1)
            def evaluate_many(self,o,v):return (Reward(0,0,0,0,0,0,0),)*2
            def diagnostics(self):return {}
        q=self.f.queries[0]
        r,_=train_step(self.f.policy,self.f.opt,self.f.frontend(q.caption),self.f.sampler,
                       lambda k:Eval(),1,q.row_id)
        self.assertEqual(r["gradient_norm"],0.)
        self.assertEqual(r["SCST_loss"],0.)
        self.assertEqual(r["optimizer_steps_this_step"],1)
    def test_nonfinite_gradient_rejected(self):
        handle=next(self.f.policy.parameters()).register_hook(lambda x:x*float("nan"))
        self.addCleanup(handle.remove)
        with self.assertRaises(ValueError):self.f.step(1)

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.f=SCSTFixture()
        self.addCleanup(self.f.close)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.parent=None
    def publish0(self):
        work=begin_step(self.root,0,self.f.protocol,None)
        self.parent=publish(self.root,work,0,self.f.policy,self.f.opt,self.f.sampler,
                            self.f.protocol,{"step":0},None)
    def publish1(self):
        self.publish0()
        work=begin_step(self.root,1,self.f.protocol,self.parent)
        r,x=self.f.step(1)
        self.parent=publish(self.root,work,1,self.f.policy,self.f.opt,self.f.sampler,self.f.protocol,r,self.parent,x)
    def test_roundtrip_H_Adam_RNG(self):
        self.publish1()
        h,opt,g,e=verify_reload(self.root,self.f.policy,self.f.opt,self.f.sampler,self.f.protocol,
                    lambda:policy_factory(self.f.probe,self.f.config),lambda h:optimizer_for(h,self.f.config))
        self.assertTrue(e["H_exact"] and e["Adam_exact"] and e["RNG_exact"])
        self.assertEqual(e["extra_optimizer_steps"],0)
    def test_corrupt_checkpoint_rejected(self):
        self.publish1();p,_=latest(self.root,self.f.protocol)
        (p/"checkpoint.pt").write_bytes(b"broken")
        with self.assertRaises(ValueError):latest(self.root,self.f.protocol)
    def test_log_SHA_rejected(self):
        self.publish1();p,_=latest(self.root,self.f.protocol)
        (p/"sample.json").write_text("{}")
        with self.assertRaises(ValueError):latest(self.root,self.f.protocol)
    def test_feature_SHA_rejected(self):
        self.publish1();p,_=latest(self.root,self.f.protocol)
        (p/"text_features.pt").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):latest(self.root,self.f.protocol)
    def test_config_mismatch_rejected(self):
        self.publish1();changed=json.loads(json.dumps(self.f.protocol));changed["config"]["optimizer"]["lr"]=2e-4
        with self.assertRaises(ValueError):latest(self.root,changed)
    def test_source_mismatch_rejected(self):
        self.publish1();changed={**self.f.protocol,"source_sha":"different"}
        with self.assertRaises(ValueError):latest(self.root,changed)
    def test_pending_refuses_automatic_retry(self):
        self.publish0();begin_step(self.root,1,self.f.protocol,self.parent)
        with self.assertRaises(ValueError):latest(self.root,self.f.protocol)
        with self.assertRaises(FileExistsError):begin_step(self.root,1,self.f.protocol,self.parent)
    def test_process_exit_after_intent_refuses_retry(self):
        self.publish0()
        original=(self.root/"checkpoints/step_0000/checkpoint.pt").read_bytes()
        code="from phase4_checkpoint import begin_step; import os; begin_step("+repr(str(self.root))+",1,"+repr(self.f.protocol)+","+repr(self.parent)+"); os._exit(9)"
        completed=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=dict(os.environ),capture_output=True)
        self.assertEqual(completed.returncode,9,completed.stderr.decode())
        self.assertEqual(original,(self.root/"checkpoints/step_0000/checkpoint.pt").read_bytes())
        with self.assertRaises(ValueError):latest(self.root,self.f.protocol)
    def test_process_exit_after_publication_restorable(self):
        self.publish0()
        code="""import os
from test_phase4 import SCSTFixture
from phase4_checkpoint import begin_step,publish,restore
f=SCSTFixture()
root=ROOT_VALUE
restore(__import__('pathlib').Path(root)/'checkpoints/step_0000',f.policy,f.opt,f.sampler,f.protocol)
work=begin_step(root,1,f.protocol,PARENT_VALUE)
log,features=f.step(1)
publish(root,work,1,f.policy,f.opt,f.sampler,f.protocol,log,PARENT_VALUE,features)
os._exit(7)
""".replace("ROOT_VALUE",repr(str(self.root))).replace("PARENT_VALUE",repr(self.parent))
        completed=subprocess.run([sys.executable,"-c",code],cwd=ROOT,env=dict(os.environ),capture_output=True)
        self.assertEqual(completed.returncode,7,completed.stderr.decode())
        path,md=latest(self.root,self.f.protocol)
        value=restore(path,self.f.policy,self.f.opt,self.f.sampler,self.f.protocol)
        self.assertEqual(value["next_step"],2)
        self.assertTrue(self.f.opt.state)
    def test_atomic_failure_preserves_prior_checkpoint(self):
        self.publish0();work=begin_step(self.root,1,self.f.protocol,self.parent)
        r,x=self.f.step(1)
        with patch("phase4_checkpoint.os.rename",side_effect=OSError("simulated publication failure")):
            with self.assertRaises(OSError):
                publish(self.root,work,1,self.f.policy,self.f.opt,self.f.sampler,self.f.protocol,r,self.parent,x)
        self.assertTrue((self.root/"checkpoints/step_0000/checkpoint.pt").exists())
        self.assertFalse((self.root/"checkpoints/step_0001").exists())
        with self.assertRaises(ValueError):latest(self.root,self.f.protocol)
    def test_immutable_step_not_overwritten(self):
        self.publish0();work=begin_step(self.root,0,self.f.protocol,None)
        with self.assertRaises(FileExistsError):
            publish(self.root,work,0,self.f.policy,self.f.opt,self.f.sampler,self.f.protocol,{"step":0},None)
    def test_corrupt_highest_no_silent_fallback(self):
        self.publish1();p,_=latest(self.root,self.f.protocol);(p/"checkpoint.json").write_text("{}")
        with self.assertRaises((ValueError,KeyError)):latest(self.root,self.f.protocol)
    def test_step_budget(self):
        with self.assertRaises(ValueError):begin_step(self.root,9,self.f.protocol,None)
    def test_rng_all_CPU_sources(self):
        state=capture_rng(self.f.sampler)
        expected=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=self.f.sampler))
        restore_rng(state,self.f.sampler)
        actual=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=self.f.sampler))
        self.assertTrue(same(expected,actual))

class RootedAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.logs={}
        for i in range(2):
            p=self.root/"checkpoints"/f"step_{i:04d}"/"sample.json"
            p.parent.mkdir(parents=True);p.write_text(json.dumps({"step":i}))
            self.logs[str(p)]=sha(p)
    def test_rooted_logs_exact(self):
        from phase4_audit import verify_rooted_logs
        verify_rooted_logs(self.root,{"step_log_sha256":self.logs})
    def test_log_tampering_even_with_changed_sidecar_rejected(self):
        from phase4_audit import verify_rooted_logs
        p=Path(next(iter(self.logs)));p.write_text("changed")
        (p.parent/"checkpoint.json").write_text(json.dumps({"sample_log_sha256":sha(p)}))
        with self.assertRaises(ValueError):verify_rooted_logs(self.root,{"step_log_sha256":self.logs})
    def test_missing_log_coverage_rejected(self):
        from phase4_audit import verify_rooted_logs
        with self.assertRaises(ValueError):verify_rooted_logs(self.root,{"step_log_sha256":{}})

class ResumeAndEntryTests(unittest.TestCase):
    def test_uninterrupted_vs_restored_exact(self):
        continuous=SCSTFixture(total=6)
        try:
            logs_a=[continuous.step(i)[0] for i in range(1,7)]
            h_a={k:x.clone() for k,x in continuous.policy.state_dict().items()}
            adam_a=continuous.opt.state_dict();rng_a=capture_rng(continuous.sampler)
        finally:continuous.close()
        split=SCSTFixture(total=6)
        try:
            with tempfile.TemporaryDirectory() as d:
                parent=None
                work=begin_step(d,0,split.protocol,None)
                parent=publish(d,work,0,split.policy,split.opt,split.sampler,split.protocol,{"step":0},None)
                logs_b=[]
                for i in range(1,4):
                    work=begin_step(d,i,split.protocol,parent);r,x=split.step(i);logs_b.append(r)
                    parent=publish(d,work,i,split.policy,split.opt,split.sampler,split.protocol,r,parent,x)
                split.policy,split.opt,split.sampler,e=verify_reload(d,split.policy,split.opt,split.sampler,
                    split.protocol,lambda:policy_factory(split.probe,split.config),
                    lambda h:optimizer_for(h,split.config))
                logs_b.extend(split.step(i)[0] for i in range(4,7))
                self.assertTrue(same(h_a,split.policy.state_dict()))
                self.assertTrue(same(adam_a,split.opt.state_dict()))
                self.assertTrue(same(rng_a,capture_rng(split.sampler)))
                self.assertEqual(logs_a,logs_b)
        finally:split.close()
    def test_query_schedule_deterministic_and_bounded(self):
        samples,_,_,_=fixture()
        a=query_order(samples[:3]);b=query_order(samples[:3])
        self.assertEqual(a,b);self.assertEqual(len(a),32)
        self.assertEqual(set(a),{s.row_id for s in samples[:3]})
        with self.assertRaises(ValueError):query_order(samples[:3],33)
    def test_real_gallery_preflight_no_model_or_directory_write(self):
        from phase4_inputs import preflight
        with patch.object(Path,"mkdir",side_effect=AssertionError("read-only source")), \
             patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("no model forward")):
            catalog,gallery,queries,features,probe,protocol=preflight()
        self.assertEqual(len(gallery),10);self.assertEqual(len(queries),10)
        self.assertEqual(features["clean"].shape,(10,512))
        self.assertEqual(protocol["new_G_forwards"],0)
        self.assertEqual(protocol["new_IRRA_image_forwards"],0)
        self.assertFalse(torch.cuda.is_initialized())
    def test_default_no_GPU(self):
        with patch.object(train_phase4,"preflight",return_value=(None,None,None,None,None,{})), \
             patch.object(train_phase4,"run",side_effect=AssertionError("GPU")):
            train_phase4.main([])
    def test_explicit_training_authorization(self):
        with patch.object(train_phase4,"preflight",side_effect=AssertionError("no preflight")):
            with self.assertRaises(ValueError):train_phase4.main(["--action","run"])
    def test_frozen_configuration(self):
        config=json.loads((ROOT/"config_phase4.json").read_text());validate_config(config)
        config["reward_weights"]["lambda_sim"]=2.
        with self.assertRaises(ValueError):validate_config(config)
    def test_no_updates_after_completion(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);(output/"protocol.json").write_text(json.dumps({"config":{}}));(output/"completed.json").write_text("{}")
            with patch.object(train_phase4,"safe_output",return_value=output), \
                 patch.object(train_phase4,"readonly_verify",return_value=None) as verify, \
                 patch.object(train_phase4,"load_runtime",side_effect=AssertionError("GPU")):
                train_phase4.run(None,None,None,None,None,{"config":{}},output,resume=True)
                verify.assert_called_once()
    def test_fresh_output_will_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d,patch.object(train_phase4,"safe_output",return_value=Path(d)):
            with self.assertRaises(FileExistsError):
                train_phase4.run(None,None,None,None,None,{"config":{}},d)

if __name__=="__main__":unittest.main()
