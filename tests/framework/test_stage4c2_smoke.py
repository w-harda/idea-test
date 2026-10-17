"""CPU合成验证门禁、计数、不可放宽阈值和独立会话，不加载真实模型。"""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import torch
from tbps.artifacts import ArtifactError
from tbps.materialization import smoke,smoke_support as support

class SmokeTests(unittest.TestCase):
    def test_limits_are_fixed(self):
        self.assertEqual(support.LIMITS,{"G":2,"H":0,"IRRA_image":4,"IRRA_text":4})
        self.assertEqual(support.RSS,2*1024**3);self.assertEqual(support.SECONDS,300)
        self.assertEqual(support.THRESHOLD,1e-6)
    def test_fixed_gallery_and_query_are_independent(self):
        self.assertEqual(support.GALLERY,("cuhk:37132:0","cuhk:37147:0"))
        self.assertEqual(support.QUERY,("cuhk:37132:0","cuhk:37147:1"))
    def test_ledger_exact_and_inference(self):
        ledger=support.Ledger();x=torch.zeros(1,3)
        with torch.inference_mode():
            f=ledger.wrap("G",lambda x:x+1);f(x);f(x)
            with self.assertRaises(ArtifactError):f(x)
        self.assertEqual(ledger.completed["G"],2)
    def test_ledger_forbids_H(self):
        with torch.inference_mode(),self.assertRaises(ArtifactError):
            support.Ledger().wrap("H",lambda x:x)(torch.zeros(1,4))
    def test_ledger_forbids_gradient_execution(self):
        with self.assertRaises(ArtifactError):
            support.Ledger().wrap("G",lambda x:x)(torch.zeros(1,4))
    def test_failed_forward_stays_counted(self):
        ledger=support.Ledger()
        def fail(x):raise RuntimeError("fixture fail")
        with torch.inference_mode(),self.assertRaises(RuntimeError):
            ledger.wrap("G",fail)(torch.zeros(1,4))
        self.assertEqual(ledger.started["G"],1);self.assertEqual(ledger.completed["G"],0)
    def test_large_batch_rejected_before_forward(self):
        f=Mock()
        with torch.inference_mode(),self.assertRaises(ArtifactError):
            support.Ledger().wrap("G",f)(torch.zeros(3,4))
        f.assert_not_called()
    def test_comparison_reports_bitwise_exact(self):
        x=torch.randn(2,512);r=support.compare(x,x.clone(),exact=True)
        self.assertTrue(r["bitwise_exact"]);self.assertEqual(r["max_abs_error"],0)
    def test_storage_clone_does_not_change_comparison(self):
        base=torch.zeros(1000,512);view=base[:2]
        clone=view.detach().clone()
        self.assertEqual(clone.untyped_storage().nbytes(),2*512*4)
        self.assertTrue(support.compare(view,clone,exact=True)["bitwise_exact"])
    def test_threshold_not_relaxed(self):
        x=torch.zeros(2,512);y=x.clone();y[0,0]=2e-6
        with self.assertRaises(ArtifactError):support.compare(x,y)
    def test_exact_readback_rejects_subthreshold_change(self):
        x=torch.zeros(2,512);y=x.clone();y[0,0]=1e-7
        self.assertLess(support.compare(x,y)["max_abs_error"],1e-6)
        with self.assertRaises(ArtifactError):support.compare(x,y,exact=True)
    def test_bad_shape_dtype_nan_rejected(self):
        x=torch.zeros(2,512)
        for bad in (torch.zeros(1,512),x.double(),torch.full_like(x,float("nan")),torch.full_like(x,float("inf"))):
            with self.subTest(shape=bad.shape),self.assertRaises(ArtifactError):support.compare(x,bad)
    def test_default_launch_does_not_touch_GPU_or_outputs(self):
        with patch.object(smoke,"preflight",return_value={"fixture":True}) as pre, \
             patch.object(smoke,"record_store",side_effect=AssertionError("write")), \
             patch("tbps.smoke_irra.gpu_available",side_effect=AssertionError("GPU")):
            self.assertEqual(smoke.launch("fixture"),0);pre.assert_called_once()
    def test_CPU_rejection_prevents_gpu_attempt(self):
        with patch.object(smoke,"preflight",side_effect=ArtifactError("closure failed")), \
             patch("tbps.smoke_irra.gpu_available") as gpu,patch.object(smoke,"record_store") as store:
            with self.assertRaises(ArtifactError):smoke.launch("fixture",allow_gpu=True)
            gpu.assert_not_called();store.assert_not_called()
    def test_wrong_output_refused_before_preflight(self):
        with patch.object(smoke,"preflight") as pre,self.assertRaises(ArtifactError):
            smoke.launch("fixture",output="/tmp/unauthorized")
        pre.assert_not_called()
    def test_existing_attempt_refused_before_preflight(self):
        with patch.object(smoke.Path,"exists",return_value=True),patch.object(smoke,"preflight") as pre:
            with self.assertRaises(ArtifactError):smoke.launch("fixture",allow_gpu=True)
            pre.assert_not_called()
    def test_cli_defaults_cpu_and_child_requires_token(self):
        with patch.object(smoke,"launch",return_value=0) as run:
            smoke.main(["--config","fixture"]);run.assert_called_once_with("fixture",allow_gpu=False)
        with self.assertRaises(ArtifactError):smoke.main(["--config","fixture","--private-stage","g"])
    def test_child_without_real_attempt_cannot_start_model(self):
        with patch.object(smoke,"record_store") as store,patch.object(smoke,"stage_g") as g:
            store.return_value.verify.side_effect=ArtifactError("no attempt")
            with self.assertRaises(ArtifactError):smoke.child("g","fixture",support.OUTPUT,"fake")
            g.assert_not_called()
    def test_no_full_seal_or_ranking_or_H_factory(self):
        tree=ast.parse(Path(smoke.__file__).read_text())
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                name=n.func.id if isinstance(n.func,ast.Name) else getattr(n.func,"attr","")
                self.assertNotIn(name,("seal_collection","seal_generated_collection","rank","backward","step"))
        self.assertIn('components=("gv2",)',Path(smoke.__file__).read_text())
    def test_independent_scores_match_fixture(self):
        g=torch.nn.functional.normalize(torch.randn(2,37),dim=-1)
        q=torch.nn.functional.normalize(torch.randn(2,37),dim=-1)
        self.assertTrue(support.compare(q@g.T,q.clone()@g.clone().T)["bitwise_exact"])
    def test_config_rejects_changed_budget_before_source_IO(self):
        fixed={"schema":"tbps-materialized-smoke-1","gallery":list(support.GALLERY),"query":list(support.QUERY),
            "gallery_indices":[0,15],"limits":dict(support.LIMITS),"threshold":1e-6,"rss_bytes":support.RSS,"seconds":300,
            "seed":42,"precision":"fp32/math/no_tf32","physical_gpu":0,"minimum_free_mib":8192,"maximum_utilization_percent":10}
        for k in ("replay_config","dataset_sources","gh_config","derivation","c1_result","b2_result"):
            fixed[k]={"path":"/fixture/"+k,"sha256":"a"*64}
        for key,value in (("threshold",1e-5),("rss_bytes",4*1024**3),("seconds",600),("seed",41),
                          ("limits",{**support.LIMITS,"G":3}),("gallery",list(reversed(support.GALLERY)))):
            with self.subTest(key=key),patch.object(support,"read_json",return_value={**fixed,key:value}), \
                 patch.object(support,"checked") as read,self.assertRaises(ArtifactError):
                support.config("fixture")
            read.assert_not_called()
    def test_config_accepts_fixed_fields_with_pinned_refs(self):
        import json
        path=Path(__file__).resolve().parents[2]/"configs/framework/stage4c2_smoke.server.json"
        with patch.object(support,"checked",return_value={}) as check:
            c=support.config(path);self.assertEqual(c["limits"],support.LIMITS);self.assertEqual(check.call_count,6)

    def test_observed_failure_early_cuda_init_is_rejected_before_IRRA_load(self):
        # 复现本次FAIL的启动顺序，证明CPU门禁没有被绕过；不是GPU修复/重试。
        from tbps.runtime_closure import cpu_only
        order=[];initialized=[False];model_load=Mock()
        def reset():
            order.append("CUDA initialization");initialized[0]=True
        def cpu_sources_then_load():
            order.append("CPU sources")
            with cpu_only():pass
            model_load()
        with patch("torch.cuda.reset_peak_memory_stats",side_effect=reset), \
             patch("torch.cuda.is_initialized",side_effect=lambda:initialized[0]):
            torch.cuda.reset_peak_memory_stats()
            with self.assertRaisesRegex(ArtifactError,"isolated CPU process required"):
                cpu_sources_then_load()
        self.assertEqual(order,["CUDA initialization","CPU sources"])
        model_load.assert_not_called()
