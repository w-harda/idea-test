"""IRRA transfer adapter contract tests."""
import pytest
import torch

from scripts.run_irra_transfer_poc import _mapped_victim_image


def test_image_transfer_preserves_zero_delta_and_budget():
    clean = torch.full((1, 3, 384, 128), 0.5)
    source = torch.full((1, 3, 224, 224), 0.5)
    unchanged, zero_linf = _mapped_victim_image(None, clean, source, source)
    assert torch.equal(unchanged, clean)
    assert zero_linf == 0.0

    adv, linf = _mapped_victim_image(None, clean, source, source + 8 / 255)
    assert adv.shape == clean.shape
    assert 0 < linf <= 8 / 255 + 1e-6
    assert torch.allclose(adv - clean, torch.full_like(clean, 8 / 255), atol=1e-6)


def test_image_transfer_rejects_source_budget_violation():
    clean = torch.full((1, 3, 384, 128), 0.5)
    source = torch.full((1, 3, 224, 224), 0.5)
    with pytest.raises(AssertionError, match="source TTA image exceeds"):
        _mapped_victim_image(None, clean, source, source + 9 / 255)
