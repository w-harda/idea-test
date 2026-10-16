"""Stage 2.5 CPU护栏；不加载真实模型，不初始化CUDA。"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

from tbps import smoke_irra as s
from tbps.artifacts import ArtifactError, ArtifactStore, read_json
from test_stage2_models import adapter


class SmokeBudgetTests(unittest.TestCase):
    def test_exact_five_image_four_text_samples(self):
        c = s.ForwardBudget()
        for kind, n in (("image", 2), ("text", 2), ("image", 2), ("text", 2), ("image", 1)):
            c.wrap(kind, lambda x: x.clone())(torch.zeros(n, 1))
        self.assertEqual(c.started, s.LIMITS)
        self.assertEqual(c.completed, s.LIMITS)
        self.assertEqual(len(c.calls), 5)

    def test_excess_rejected_before_native_call(self):
        c = s.ForwardBudget()
        called = []
        f = c.wrap("image", lambda x: called.append(len(x)))
        f(torch.zeros(2, 1)); f(torch.zeros(2, 1)); f(torch.zeros(1, 1))
        with self.assertRaisesRegex(ArtifactError, "budget"):
            f(torch.zeros(1, 1))
        self.assertEqual(called, [2, 2, 1])

    def test_text_excess_independently_rejected(self):
        c = s.ForwardBudget()
        f = c.wrap("text", lambda x: x)
        f(torch.zeros(2, 1)); f(torch.zeros(2, 1))
        with self.assertRaises(ArtifactError):
            f(torch.zeros(1, 1))
        self.assertEqual(c.started["image"], 0)

    def test_large_batch_rejected(self):
        with self.assertRaises(ArtifactError):
            s.ForwardBudget().wrap("image", lambda x: x)(torch.zeros(3, 1))

    def test_empty_batch_rejected(self):
        with self.assertRaises(ArtifactError):
            s.ForwardBudget().wrap("image", lambda x: x)(torch.zeros(0, 1))

    def test_exception_counts_attempt_without_retry(self):
        c = s.ForwardBudget()
        def failed(x):
            raise RuntimeError("fixture failed")
        with self.assertRaises(RuntimeError):
            c.wrap("image", failed)(torch.zeros(1, 1))
        self.assertEqual(c.started["image"], 1)
        self.assertEqual(c.completed["image"], 0)
        self.assertFalse(c.calls[0]["returned"])

    def test_counter_retains_no_tensor_or_graph(self):
        c = s.ForwardBudget()
        x = torch.ones(1, 1, requires_grad=True)
        c.phase = "surrogate"
        c.wrap("image", lambda x: x * 2)(x)
        json.dumps(c.report(), allow_nan=False)
        self.assertEqual(c.calls[0]["phase"], "surrogate")
        self.assertTrue(c.calls[0]["grad_enabled"])


class SmokeNumericsTests(unittest.TestCase):
    def test_identical_tokens_exact(self):
        self.assertTrue(s.compare(torch.ones(2, 77, dtype=torch.long),
                                  torch.ones(2, 77, dtype=torch.long), exact=True)["exact"])

    def test_token_mismatch_rejected(self):
        with self.assertRaises(ArtifactError):
            s.compare(torch.zeros(2, 77, dtype=torch.long), torch.ones(2, 77, dtype=torch.long), exact=True)

    def test_subthreshold_accepted_and_reported(self):
        report = s.compare(torch.zeros(2, 2), torch.full((2, 2), 5e-7))
        self.assertLess(report["max_abs_error"], 1e-6)
        self.assertFalse(report["exact"])

    def test_threshold_not_relaxed(self):
        with self.assertRaisesRegex(ArtifactError, "1e-6"):
            s.compare(torch.zeros(2, 2), torch.full((2, 2), 1.1e-6))

    def test_exact_preprocess_refuses_small_difference(self):
        with self.assertRaisesRegex(ArtifactError, "exact"):
            s.compare(torch.zeros(1), torch.tensor([1e-8]), exact=True)

    def test_nonfinite_refused(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ArtifactError):
                s.compare(torch.tensor([value]), torch.tensor([value]))

    def test_shape_and_dtype_refused(self):
        for b in (torch.ones(2), torch.ones(1).double()):
            with self.subTest(shape=b.shape), self.assertRaises(ArtifactError):
                s.compare(torch.ones(1), b)

    def test_features_normalized_and_native_dim(self):
        a = torch.nn.functional.normalize(torch.ones(2, 512), dim=-1)
        report = s.feature_comparison(a, a.clone())
        self.assertEqual(report["shape"], [2, 512])
        self.assertEqual(len(report["official_l2_norms"]), 2)
        with self.assertRaises(ArtifactError):
            s.feature_comparison(torch.ones(2, 512), torch.ones(2, 512))
        with self.assertRaises(ArtifactError):
            s.feature_comparison(torch.ones(2, 37), torch.ones(2, 37))

    def test_gradient_finite_nonzero_and_model_unchanged(self):
        a = adapter()
        before = s.model_snapshot(a._backend)
        x = torch.rand(1, 3, 384, 128, requires_grad=True)
        a.surrogate().encode_images(x).payload[0, 0].backward()
        self.assertTrue(s.gradient_report(x)["nonzero"])
        self.assertEqual(before, s.model_snapshot(a._backend))

    def test_missing_zero_nonfinite_gradient_rejected(self):
        for grad in (None, torch.zeros(1), torch.tensor([float("nan")])):
            x = torch.ones(1, requires_grad=True)
            x.grad = grad
            with self.subTest(grad=grad), self.assertRaises(ArtifactError):
                s.gradient_report(x)

    def test_state_hash_detects_parameter_and_buffer_change(self):
        for kind in ("parameter", "buffer"):
            a = adapter()
            before = s.model_snapshot(a._backend)
            with torch.no_grad():
                tensor = next(a._backend.parameters()) if kind == "parameter" else next(a._backend.buffers())
                tensor.add_(1)
            self.assertNotEqual(before, s.model_snapshot(a._backend))

    def test_model_must_remain_frozen_and_eval(self):
        a = adapter()
        a._backend.train()
        with self.assertRaises(ArtifactError):
            s.model_snapshot(a._backend)
        a._backend.eval()
        next(a._backend.parameters()).requires_grad_(True)
        with self.assertRaises(ArtifactError):
            s.model_snapshot(a._backend)


class SmokePermissionTests(unittest.TestCase):
    def gpu_mock(self, free=24000, util=0, process=""):
        return [subprocess.CompletedProcess([], 0, f"NVIDIA GeForce RTX 4090, {free}, 40, {util}\n", ""),
                subprocess.CompletedProcess([], 0, process, "")]

    def test_free_gpu_allowed_without_torch_cuda(self):
        with patch.object(s.subprocess, "run", side_effect=self.gpu_mock()):
            self.assertEqual(s.gpu_available({"minimum_free_mib":8192,"maximum_utilization_percent":10})["free_mib"], 24000)
        self.assertFalse(torch.cuda.is_initialized())

    def test_low_memory_safe_stop(self):
        with patch.object(s.subprocess, "run", side_effect=self.gpu_mock(free=4000)):
            with self.assertRaises(ArtifactError):
                s.gpu_available({"minimum_free_mib":8192,"maximum_utilization_percent":10})

    def test_busy_gpu_safe_stop(self):
        with patch.object(s.subprocess, "run", side_effect=self.gpu_mock(util=60)):
            with self.assertRaises(ArtifactError):
                s.gpu_available({"minimum_free_mib":8192,"maximum_utilization_percent":10})

    def test_existing_process_safe_stop(self):
        with patch.object(s.subprocess, "run", side_effect=self.gpu_mock(process="123, python, 2000")):
            with self.assertRaises(ArtifactError):
                s.gpu_available({"minimum_free_mib":8192,"maximum_utilization_percent":10})

    def test_default_cli_is_cpu_only(self):
        resolved = ({}, None, (), {}, {"model_loaded":False})
        with patch.object(s, "resolve_config", return_value=resolved), patch.object(s, "run_once", side_effect=AssertionError("GPU forbidden")):
            self.assertEqual(s.main(["--config","fixture"]), 0)

    def test_used_output_refuses_gpu_reentry(self):
        with tempfile.TemporaryDirectory(prefix="tbps-framework-smoke-") as base:
            out = Path(base) / "once";out.mkdir();(out/"failure.json").write_text("{}")
            with patch.object(s, "BASE", base), patch.object(s, "gpu_available", side_effect=AssertionError("GPU forbidden")):
                with self.assertRaisesRegex(ArtifactError, "already used"):
                    s.run_once({}, None, (), {}, {}, out)

    def test_historical_output_rejected(self):
        with self.assertRaises(ArtifactError):
            s.run_once({}, None, (), {}, {}, "/home/lzf/ldx/projects/idea-TBPS-test1/outputs")

    def test_explicit_device_environment_required(self):
        with tempfile.TemporaryDirectory(prefix="tbps-framework-smoke-") as base:
            with patch.object(s, "BASE", base), patch.dict(s.os.environ, {"CUDA_VISIBLE_DEVICES":""}):
                with self.assertRaisesRegex(ArtifactError, "CUDA_VISIBLE_DEVICES"):
                    s.run_once({}, None, (), {}, {}, Path(base)/"once")

    def test_attempt_record_is_immutable(self):
        with tempfile.TemporaryDirectory(prefix="tbps-framework-smoke-") as base:
            store = ArtifactStore(Path(base)/"once", new_base=base)
            store.publish("attempt", {"preflight.json": b"{}"}, {"purpose":"one"})
            with self.assertRaises(ArtifactError):
                store.publish("attempt", {"preflight.json": b"{}"}, {"purpose":"two"})
            store.verify("attempt")

    def test_import_is_lazy_no_models_or_torch(self):
        script = """import builtins,sys
old=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0] in {'torch','torchvision','model','prototypes'}:raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=guard
import tbps.smoke_irra
assert 'torch' not in sys.modules
print('LAZY_IMPORT_PASS')
"""
        p = subprocess.run([sys.executable,"-B","-c",script],capture_output=True,text=True,check=True)
        self.assertIn("LAZY_IMPORT_PASS",p.stdout)

    def test_invalid_configuration_refused_before_source_io(self):
        good = {"schema":"tbps-irra-smoke-1","sources":{},"row_ids":list(s.ROW_IDS),
                "limits":dict(s.LIMITS),"max_abs_error":1e-6,"minimum_free_mib":8192,
                "maximum_utilization_percent":10,"precision":"fp32/math/no_tf32",
                "physical_gpu":0,"membership_sha256":"1"*64}
        for key, value in (("limits",{"image":6,"text":4}),("row_ids",["test:0","test:1"]),
                           ("max_abs_error",1e-3),("minimum_free_mib",4096),
                           ("precision","fp16"),("physical_gpu",1),("maximum_utilization_percent",50)):
            c = dict(good);c[key] = value
            with tempfile.TemporaryDirectory() as root:
                p = Path(root)/"config.json";p.write_text(json.dumps(c))
                with self.subTest(key=key), self.assertRaises(ArtifactError):
                    s.resolve_config(p)


if __name__ == "__main__":
    unittest.main()
