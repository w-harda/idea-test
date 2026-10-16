"""Stage 3B一次性入口、计数及完整动作比较的CPU护栏。"""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from tbps import smoke_attacks as s
from tbps.artifacts import ArtifactError

class Echo(nn.Module):
    def forward(self,x):return x+1

class HMock(nn.Module):
    def forward(self,example):return torch.ones(1)

class SmokeAttackTests(unittest.TestCase):
    def test_action_tuple_list_normalization_exact(self):
        a={"H_forward":1,"budget_k":2,"predicted_importance":(.1,.2),"edits":({"position":1},)}
        b={"H_forward":1,"budget_k":2,"predicted_importance":[.1,.2],"edits":[{"position":1}]}
        self.assertTrue(s.exact_action(a,b)["exact"])
    def test_same_string_different_full_action_refused(self):
        a={"H_forward":1,"adversarial":"red","budget_k":2}
        for field,value in (("budget_k",1),("H_forward",0),("decisions",[])):
            b=dict(a);b[field]=value
            with self.subTest(field=field),self.assertRaises(ArtifactError):s.exact_action(a,b)
    def test_importance_last_bit_difference_refused(self):
        a={"H_forward":1,"predicted_importance":[.1]}
        with self.assertRaises(ArtifactError):
            s.exact_action(a,{"H_forward":1,"predicted_importance":[.10000000000000002]})
    def test_image_budget_four_and_captures_cpu_compact(self):
        ledger=s.ForwardLedger();model=Echo().eval()
        with ExitStack() as stack,torch.inference_mode():
            ledger.attach(model,"G",stack)
            for phase in ("native","plugin"):
                ledger.phase=phase
                for row in s.ROW_IDS:
                    ledger.sample=row;model(torch.zeros(1,3,8,8))
            with self.assertRaisesRegex(ArtifactError,"budget"):model(torch.zeros(1,3,8,8))
        self.assertEqual(ledger.completed["G"],4)
        self.assertEqual(len(ledger.g_captures),4)
        for record in ledger.g_captures.values():
            self.assertEqual(record["delta"].device.type,"cpu")
            self.assertEqual(record["delta"].untyped_storage().nbytes(),record["delta"].numel()*4)
    def test_H_budget_and_label_fields(self):
        ledger=s.ForwardLedger();model=HMock().eval()
        with ExitStack() as stack:
            ledger.attach(model,"H",stack)
            for _ in range(4):model(SimpleNamespace(caption="red",attributes=()))
            with self.assertRaises(ArtifactError):model(SimpleNamespace(caption="red",attributes=()))
        ledger=s.ForwardLedger()
        with ExitStack() as stack:
            ledger.attach(model,"H",stack)
            with self.assertRaises(ArtifactError):
                model(SimpleNamespace(caption="red",attributes=(),pid=1))
        self.assertEqual(ledger.started["H"],0)
    def test_failed_forward_not_marked_completed(self):
        class Fails(nn.Module):
            def forward(self,x):raise RuntimeError("fixture")
        model=Fails();ledger=s.ForwardLedger()
        with ExitStack() as stack:
            ledger.attach(model,"G",stack)
            with self.assertRaises(RuntimeError):model(torch.zeros(1,3,8,8))
        self.assertEqual(ledger.started["G"],1);self.assertEqual(ledger.completed["G"],0)
    def test_preflight_default_never_runs_gpu(self):
        with patch.object(s,"preflight",return_value=({},None,(),(),(),{},{"model_loaded":False})), \
             patch.object(s,"run_once",side_effect=AssertionError("GPU forbidden")):
            self.assertEqual(s.main(["--config","fixture"]),0)
    def test_existing_attempt_refuses_reentry_before_GPU(self):
        with tempfile.TemporaryDirectory(prefix="tbps-framework-smoke-") as base:
            out=Path(base)/"once";out.mkdir();(out/"attempt").mkdir()
            with patch.object(s,"BASE",base),patch.object(s,"gpu_available",side_effect=AssertionError("GPU forbidden")):
                with self.assertRaises(ArtifactError):s.run_once({},None,(),(),(),{}, {},out)
    def test_historical_outputs_rejected(self):
        with self.assertRaises(ArtifactError):
            s.run_once({},None,(),(),(),{},{},"/home/lzf/ldx/projects/idea-TBPS-test1/outputs")
    def test_CUDA_environment_explicit(self):
        with tempfile.TemporaryDirectory(prefix="tbps-framework-smoke-") as base:
            with patch.object(s,"BASE",base),patch.dict(s.os.environ,{"CUDA_VISIBLE_DEVICES":""}):
                with self.assertRaises(ArtifactError):s.run_once({},None,(),(),(),{}, {},Path(base)/"once")
    def test_fixed_limits_threshold_and_samples(self):
        self.assertEqual(s.LIMITS,{"G":4,"H":4,"IRRA":0})
        self.assertEqual(len(s.ROW_IDS),2)
        for kwargs in ({"row_ids":["other","sample"]},{"limits":{"G":5,"H":4,"IRRA":0}},
                       {"max_abs_error":.01},{"seed":43}):
            c={"schema":"tbps-gh-equivalence-smoke-1","dataset_sources":{},"attack_sources":{},
               "row_ids":list(s.ROW_IDS),"expected_eligible":[True,False],"limits":dict(s.LIMITS),
               "max_abs_error":1e-6,"seed":42,"precision":"fp32/math/no_tf32",
               "physical_gpu":0,"minimum_free_mib":8192,"maximum_utilization_percent":10}
            c.update(kwargs)
            with tempfile.TemporaryDirectory() as base:
                import json
                p=Path(base)/"config.json";p.write_text(json.dumps(c))
                with self.subTest(change=kwargs),self.assertRaises(ArtifactError):s.preflight(p)

if __name__=="__main__":unittest.main()
