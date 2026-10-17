"""新IRRA-only入口CPU门禁/资源/预算测试；真实启动流程沿用Stage4C2R回归。"""
from pathlib import Path
import ast
import json
import unittest
from unittest.mock import patch
from tbps.artifacts import ArtifactError
from tbps.engines.replay_control import ResourceLimit
from tbps.materialization import irra_only as runner

class IRRAOnlyTests(unittest.TestCase):
    def test_default_entry_only_CPU_no_output_no_GPU(self):
        prepared=({},None,None,{"fixture":True},())
        with patch.object(runner,"preflight",return_value=prepared), \
             patch.object(runner,"ArtifactStore",side_effect=AssertionError("output")), \
             patch("tbps.smoke_irra.gpu_available",side_effect=AssertionError("GPU")):
            self.assertEqual(runner.run("fixture"),0)
    def test_preflight_failure_never_loads_or_claims(self):
        with patch.object(runner,"preflight",side_effect=ArtifactError("source failure")), \
             patch.object(runner,"ArtifactStore") as store:
            with self.assertRaises(ArtifactError):runner.run("fixture",allow_gpu=True)
            store.assert_not_called()
    def test_single_existing_attempt_refused_before_preflight(self):
        with patch.object(runner.Path,"exists",return_value=True),patch.object(runner,"preflight") as pre:
            with self.assertRaises(ArtifactError):runner.run("fixture",allow_gpu=True)
            pre.assert_not_called()
    def test_no_2GiB_RSS_limit(self):
        guard=runner.SystemGuard(memory=lambda:{"MemAvailable":12*1024**3},sample=lambda:4*1024**3)
        guard.check()
        self.assertEqual(guard.snapshot()["peak_rss_bytes"],4*1024**3)
        self.assertIsNone(guard.snapshot()["rss_hard_limit_bytes"])
    def test_low_available_RAM_stops(self):
        guard=runner.SystemGuard(memory=lambda:{"MemAvailable":1024**3},sample=lambda:512*1024**2)
        with self.assertRaises(ResourceLimit):guard.check()
        self.assertIn("RAM",guard.reason)
    def test_total_time_limit_still_enforced(self):
        now=[0.]
        guard=runner.SystemGuard(memory=lambda:{"MemAvailable":12*1024**3},sample=lambda:512*1024**2,clock=lambda:now[0])
        now[0]=301
        with self.assertRaises(ResourceLimit):guard.check()
    def test_fixed_counts_and_threshold(self):
        self.assertEqual(runner.LIMITS,{"G":0,"H":0,"IRRA_image":4,"IRRA_text":4})
        self.assertEqual(runner.THRESHOLD,1e-6)
    def test_no_G_H_generation_or_loader_or_ranking_calls(self):
        tree=ast.parse(Path(runner.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.id if isinstance(node.func,ast.Name) else getattr(node.func,"attr","")
                self.assertNotIn(name,("load_generator","load_legacy_plugins","generate_once","generate_image_once",
                    "generate_text_once","seal_collection","rank","backward","step"))
    def config(self):
        path=Path(__file__).resolve().parents[2]/"configs/framework/stage4c3_irra_only.server.json"
        return json.loads(path.read_text())
    def test_changed_numerical_contract_refused_before_IO(self):
        for k,v in (("threshold",1e-5),("limits",{**runner.LIMITS,"G":2}),("seed",43),("query",["wrong"])):
            with self.subTest(key=k),patch.object(runner,"read_json",return_value={**self.config(),k:v}), \
                 patch.object(runner,"checked") as checked,self.assertRaises(ArtifactError):
                runner.read_control("fixture")
            checked.assert_not_called()
    def test_old_RSS_limit_cannot_silently_reappear(self):
        with patch.object(runner,"read_json",return_value={**self.config(),"rss_hard_limit_bytes":2*1024**3}):
            with self.assertRaises(ArtifactError):runner.read_control("fixture")
    def test_missing_repair_receipt_rejected(self):
        with patch.object(runner,"read_json",return_value=self.config()), \
             patch.object(runner,"checked",side_effect=ArtifactError("missing SHA")):
            with self.assertRaises(ArtifactError):runner.read_control("fixture")
    def test_cli_defaults_CPU(self):
        with patch.object(runner,"run",return_value=0) as run:
            runner.main(["--config","fixture"]);run.assert_called_once_with("fixture",allow_gpu=False)

    def test_historical_tokenizer_before_after_schema(self):
        from types import SimpleNamespace as NS
        action={"original":"red shirt","tokenizer_audit":{"before":{"model_tokens":[1,2]},"after":{"model_tokens":[1,3]}}}
        inputs=NS(source=None,texts=[{"action":action,"original_caption":"red shirt","adversarial_caption":"rеd shirt"}])
        tokenizer=NS(fingerprint="fixture",inspect=lambda text:NS(ids=(1,2) if text=="red shirt" else (1,3),
                     untruncated_length=2,truncated=False))
        with patch("tbps.models.irra_io.official_tokenizer",return_value=tokenizer):
            report=runner.tokenizer_check(inputs)
        self.assertTrue(report["historical_tokens_exact"]);self.assertEqual(len(report["records"]),2)
    def test_historical_tokenizer_id_difference_refused(self):
        from types import SimpleNamespace as NS
        inputs=NS(source=None,texts=[{"action":{"original":"red","tokenizer_audit":{"before":{"model_tokens":[1,2]}}},
                                    "original_caption":"red","adversarial_caption":"rеd"}])
        tokenizer=NS(fingerprint="fixture",inspect=lambda text:NS(ids=(1,9),untruncated_length=2,truncated=False))
        with patch("tbps.models.irra_io.official_tokenizer",return_value=tokenizer),self.assertRaises(ArtifactError):
            runner.tokenizer_check(inputs)
