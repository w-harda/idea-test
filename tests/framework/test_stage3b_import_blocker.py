"""FAIL现场CPU回归：像素读取器的传递依赖未锁定时必须拒绝；不修闸门或重试GPU。"""
from contextlib import ExitStack
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactError
from tbps.attacks.legacy_runtime import legacy_imports
from stage3a_fixtures import support

class RuntimeClosureBlockerTests(unittest.TestCase):
    def test_pixel_reader_current_sources_refuse_unpinned_condition_schema(self):
        # 记录当前明确阻断项；后续批准修复时须替换为完整闭包预检通过测试。
        with patch.object(torch,"load",side_effect=AssertionError("weight restore forbidden")), \
             patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("neural forward forbidden")):
            with legacy_imports(support(),allow_import=True) as old:
                with self.assertRaisesRegex(ArtifactError,"unpinned legacy import: .*condition_schema.py"):
                    old("irra_image_adapter")
        self.assertFalse(torch.cuda.is_initialized())

    def test_strict_preflight_rejects_current_closure_without_model_restore(self):
        from tbps.check_attack_runtime import check
        with patch("tbps.check_attack_runtime.preflight",return_value=({},support(),(),(),(),{},{})):
            with self.assertRaisesRegex(ArtifactError,"condition_schema.py"):check("fixture")

if __name__=="__main__":unittest.main()
