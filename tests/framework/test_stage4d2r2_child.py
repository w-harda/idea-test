"""实际child + 原StartupGate + 真实预处理/首图恢复组合；仅CUDA/权重为Mock。"""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import time
import unittest
import torch
from tbps.artifacts import ArtifactError,ArtifactStore,read_json,json_bytes,digest
from tbps.materialization.gpu_safety import GObservation
from tbps.materialization.claim_recovery import verify_recovery_completion
from tbps.materialization.full_gpu import control
import test_stage4d2r_startup as startup_fixtures
from test_stage4d2r2_recovery import World,ref
from dataclasses import asdict

class V3ChildTests(unittest.TestCase):
    def test_actual_child_recovers_first_claim_and_runs_original_preprocessing(self):
        flow=startup_fixtures.ChildFlowTests();flow.setUp()
        tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-v3-child-world-")
        w=World(Path(tmp.name))
        try:
            flow.store=w.store
            # 新World的Plan不同，发布独立Fixture launch；原Fixture回执不覆盖。
            flow.output=flow.root/"attempt-v3"
            flow.records=ArtifactStore(flow.output,new_base=flow.root)
            flow.records.publish("attempt",{
                "preflight.json":json_bytes({"status":"G_FULL_CPU_PREFLIGHT_PASS","plan_id":w.store.plan.id,
                    "CPU_contexts_restored":True,"GH_status":"PASS","old_pixel_reference":{}}),
                "control.json":json_bytes({"token_sha256":digest(flow.token),
                    "config":asdict(ref(flow.config)),"started_unix":time.time()}),
                "old_files.json":b"{}"},{"fixture":True})
            flow.c["recovery"]=asdict(ref(w.path))
            with w.plugin(guarded=False) as (_,observer):
                flow.c["preprocess_sources"]=observer.preprocess_sources
            original_prepare=flow.prepare
            def prepare(path):
                value=original_prepare(path);legacy=value[2]
                source=legacy.root/"prototypes/attribute_image_generator_v2_20261014/condition_schema.py"
                legacy.config["runtime_sources"][str(source)]=ref(source).sha256
                return value
            flow.prepare=prepare
            def observed(*args,**kwargs):
                value=GObservation(*args,**kwargs);flow.observers.append(value);return value
            flow.observer=observed
            @contextmanager
            def plugins(legacy,**kwargs):
                flow.events.append("G_load_entry")
                flow.assertIn("load_guard_ready",flow.events)
                flow.assertIn("no_opt_ready",flow.events)
                torch.load(w.store.plan.g.checkpoint.path,weights_only=True)
                with w.plugin(guarded=False) as (plugin,_):
                    yield {"gv2":plugin}
            flow.plugins=plugins
            code,report=flow.run_child()
            # Fixture两图不放宽正式3074总数门禁。
            self.assertEqual(code,1)
            self.assertEqual(report["error"],"exact 3074 generation / one restoration")
            self.assertEqual(report["actual_G_forwards"]["started"],2)
            self.assertEqual(report["actual_G_forwards"]["completed"],2)
            self.assertEqual(report["actual_G_forwards"]["verified_preprocess_Resize_calls"],4)
            self.assertEqual(report["counts"]["G_load_attempts"],1)
            self.assertEqual(report["counts"]["G_loads"],1)
            self.assertEqual(report["model_before"],report["model_after"])
            self.assertEqual(report["stream"]["generated"],2)
            self.assertEqual(report["stream"]["recovered_first"],1)
            proof=verify_recovery_completion(w.store,w.recovery,flow.records)
            self.assertTrue(proof["old_claim_preserved"])
            self.assertLess(flow.events.index("CUDA_mock"),flow.events.index("no_opt_ready"))
            self.assertLess(flow.events.index("load_guard_ready"),flow.events.index("G_load_entry"))
        finally:
            flow.tearDown();tmp.cleanup()

    def test_v3_configuration_is_strict_and_old_v2_is_rejected(self):
        root=Path(__file__).resolve().parents[2]
        cfg=root/"configs/framework/stage4d2r2_generation.server.json"
        value=control(cfg)
        self.assertTrue(value["output"].endswith("/generation_v3"))
        self.assertEqual(value["gallery"],3074)
        self.assertEqual(value["query"],6156)
        self.assertEqual(value["max_seconds"],7200)
        self.assertEqual(len(value["entry_sources"]),4)
        with self.assertRaises(ArtifactError):
            control(root/"configs/framework/stage4d2r_generation.server.json")
