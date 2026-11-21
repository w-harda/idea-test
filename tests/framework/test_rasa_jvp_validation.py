"""弱信号舍入回归、真实偏差拒绝与有限差分回执审计。"""
import unittest

import torch

from tbps.artifacts import ArtifactError
from tbps.native_aug.jvp_validation import audit_fd_receipt, verify_directional_derivative


class DirectionalDerivativeTests(unittest.TestCase):
    def test_normal_step_unchanged(self):
        truth = torch.tensor([0.5, -0.3], dtype=torch.float64)
        receipt = verify_directional_derivative(lambda step: truth * step, truth)
        self.assertEqual(receipt['FD_method'], 'centered')
        audit_fd_receipt({'zero_exact': True, **receipt})

    def test_fp32_cancellation_recovers_without_relaxing_threshold(self):
        truth = torch.tensor([0.004, -0.0015], dtype=torch.float64)

        def quantized(step):
            return torch.tensor([1.0, 1.0]) + (truth * step).round(decimals=5).float()

        receipt = verify_directional_derivative(quantized, truth)
        self.assertEqual(receipt['FD_method'], 'richardson')
        self.assertGreater(receipt['FD_initial_relative_error'], 0.02)
        self.assertLess(receipt['FD_relative_error'], 0.02)
        audit_fd_receipt({'zero_exact': True, **receipt})

    def test_stable_reference_rejects_wrong_gradients(self):
        truth = torch.tensor([0.5, -0.3], dtype=torch.float64)
        with self.assertRaisesRegex(ArtifactError, 'directional derivative mismatch'):
            verify_directional_derivative(lambda step: truth * step, truth * 0.8)

    def test_nonconverged_estimates_rejected(self):
        truth = torch.tensor([0.5, -0.3], dtype=torch.float64)

        def inconsistent(step):
            multipliers = {0.01: 2, 0.04: 2, 0.08: 1, 0.16: 3, 0.32: 1, 0.64: 5}
            return truth * step * multipliers[abs(step)]

        with self.assertRaisesRegex(ArtifactError, 'did not converge'):
            verify_directional_derivative(inconsistent, truth)

    def test_audit_rejects_modified_fallback_drift(self):
        proof = {'zero_exact': True, 'FD_method': 'richardson', 'FD_step': 0.08,
                 'FD_coarse_step': 0.16, 'FD_initial_relative_error': 0.03,
                 'FD_relative_error': 0.005, 'FD_scale_drift': 0.02}
        with self.assertRaises(ArtifactError):
            audit_fd_receipt(proof)


if __name__ == '__main__':
    unittest.main()
