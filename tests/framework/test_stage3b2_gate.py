"""只用Mock验证启动顺序和不可绕过CPU门禁；不加载真实模型。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tbps.artifacts import ArtifactError
from tbps import smoke_attacks_gated as gate

class GateTests(unittest.TestCase):
    def test_default_runs_cpu_only(self):
        with patch.object(gate,"prepare",return_value=({},None,(),(),(),{},{})) as cpu, \
             patch.object(gate.smoke_attacks,"run_once") as gpu:
            self.assertEqual(gate.launch("config","derivation"),0)
            cpu.assert_called_once_with("config","derivation");gpu.assert_not_called()

    @patch("tbps.smoke_attacks_gated.Path.exists",return_value=False)
    def test_cpu_failure_prevents_gpu(self,_exists):
        with patch.object(gate,"prepare",side_effect=ArtifactError("source mismatch")), \
             patch.object(gate.smoke_attacks,"run_once") as gpu:
            with self.assertRaisesRegex(ArtifactError,"source mismatch"):
                gate.launch("config","derivation",allow_gpu_smoke=True)
            gpu.assert_not_called()

    @patch("tbps.smoke_attacks_gated.Path.exists",return_value=False)
    def test_gpu_called_once_after_strict_cpu_success(self,_exists):
        order=[]
        def cpu(*args):order.append("CPU");return ({},None,(),(),(),{},{"strict":"PASS"})
        def gpu(*args,**kwargs):order.append("GPU");return 0
        with patch.object(gate,"prepare",side_effect=cpu), \
             patch.object(gate.smoke_attacks,"run_once",side_effect=gpu) as run:
            self.assertEqual(gate.launch("config","derivation",allow_gpu_smoke=True),0)
            run.assert_called_once();self.assertEqual(order,["CPU","GPU"])
            self.assertEqual(run.call_args.kwargs["output"],gate.OUTPUT)

    @patch("tbps.smoke_attacks_gated.Path.exists",return_value=False)
    def test_failed_gpu_never_retried(self,_exists):
        with patch.object(gate,"prepare",return_value=({},None,(),(),(),{},{})), \
             patch.object(gate.smoke_attacks,"run_once",return_value=1) as run:
            self.assertEqual(gate.launch("config","derivation",allow_gpu_smoke=True),1)
            run.assert_called_once()

    def test_old_fail_or_other_output_refused_before_cpu(self):
        with patch.object(gate,"prepare") as cpu:
            for output in (str(Path(gate.OUTPUT).with_name("equivalence_v1")),"/tmp/new_attempt"):
                with self.subTest(output=output),self.assertRaises(ArtifactError):
                    gate.launch("c","d",allow_gpu_smoke=True,output=output)
            cpu.assert_not_called()

    def test_existing_attempt_refused_without_cpu_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(gate,"OUTPUT",str(Path(tmp).resolve())), \
                 patch.object(gate,"prepare") as cpu,patch.object(gate.smoke_attacks,"run_once") as gpu:
                with self.assertRaisesRegex(ArtifactError,"already used"):
                    gate.launch("c","d",allow_gpu_smoke=True,output=tmp)
                cpu.assert_not_called();gpu.assert_not_called()

    def test_derivation_required_by_cli(self):
        with self.assertRaises(SystemExit) as exc:
            gate.main(["--config","fixture"])
        self.assertEqual(exc.exception.code,2)

    def test_prepare_must_capture_actual_preflight(self):
        with patch.object(gate,"check_derived",return_value={"status":"PASS","runtime_source_count":75}), \
             patch.object(gate.smoke_attacks,"run_once") as gpu:
            with self.assertRaisesRegex(ArtifactError,"execute once"):gate.prepare("c","d")
            gpu.assert_not_called()

    def test_wrong_source_count_or_failed_status_refused(self):
        for report in ({"status":"FAIL","runtime_source_count":75},{"status":"PASS","runtime_source_count":74}):
            with self.subTest(report=report),patch.object(gate,"check_derived",return_value=report):
                with self.assertRaisesRegex(ArtifactError,"75 fixed"):gate.prepare("c","d")

if __name__=="__main__":unittest.main()
