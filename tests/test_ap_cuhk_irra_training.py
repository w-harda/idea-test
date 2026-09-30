"""Baseline 3 的 train-only、可微映射和来源合约。"""
import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from attributes.ap_cuhk_irra_training import ap_irra_losses, transport_for_training
from attributes.ap_gallery_baseline import sha256
from scripts import run_ap_cuhk_irra_surrogate_baseline as evaluation
from scripts import train_ap_cuhk_irra_baseline as training


def test_config_fixes_baseline2_generator_recipe(tmp_path):
    config, base = training.read_config(training.DEFAULT_CONFIG)
    g = config["generator"]
    assert (g["epochs"], g["batch_size"], g["instances_per_identity"]) == (60, 48, 4)
    assert g["irra_weight"] == base["generator"]["reid_weight"] == 10
    assert g["semantic_weight"] == base["generator"]["semantic_weight"] == 10
    assert g["architecture"] == base["generator"]["architecture"]
    altered = json.loads(json.dumps(config))
    altered["generator"]["margin"] = .4
    import yaml
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(altered))
    with pytest.raises(ValueError, match="仅替换 IDE"):
        training.read_config(path)


def test_transport_is_differentiable_and_bounded():
    clean = torch.zeros(2, 3, 256, 128)
    adv = torch.full_like(clean, .04, requires_grad=True)
    victim = torch.full((2, 3, 384, 128), .5)
    native_clean, native_adv, victim_adv = transport_for_training(clean, adv, victim)
    assert torch.allclose(native_clean, torch.full_like(native_clean, .5))
    assert torch.allclose(native_adv, torch.full_like(native_adv, .52))
    assert torch.allclose(victim_adv, torch.full_like(victim_adv, .52))
    victim_adv.sum().backward()
    assert adv.grad is not None and torch.isfinite(adv.grad).all()
    assert adv.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="超过 8/255"):
        transport_for_training(clean, torch.full_like(clean, .2), victim)


class TinyIRRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(3), requires_grad=False)

    def encode_images(self, pixels):
        return F.normalize(pixels.mean(dim=(2, 3)) * self.scale, dim=-1)


class TinySemantic(nn.Module):
    def forward(self, pixels, ids):
        x = pixels.mean(dim=(2, 3))
        return None, torch.cat([x + i * .1 for i in range(5)], dim=1)


def test_two_losses_reach_adversarial_pixels_without_irra_weight_grads():
    clean = torch.zeros(2, 3, 256, 128)
    adv = torch.full_like(clean, .02, requires_grad=True)
    victim = torch.tensor([.35, .55, .75]).view(1, 3, 1, 1).expand(2, 3, 384, 128)
    ids = torch.tensor([0, 1])
    irra = TinyIRRA()
    original = {"adv_TripletLoss": lambda margin: (
        lambda clean, adv, ids: (adv - clean).square().mean()
    )}
    total, image_loss, semantic_loss, info = ap_irra_losses(
        clean, adv, victim, ids, irra, TinySemantic(), original)
    assert torch.isfinite(torch.stack((total, image_loss, semantic_loss))).all()
    assert torch.allclose(total, image_loss + semantic_loss)
    assert info["adv_irra_feature"].shape == (2, 3)
    assert torch.allclose(info["adv_irra_feature"].norm(dim=-1), torch.ones(2))
    total.backward()
    assert adv.grad is not None and adv.grad.abs().sum() > 0
    assert irra.scale.grad is None


def test_evaluation_accepts_only_formal_epoch60_matching_source(tmp_path):
    cp = tmp_path / "G.pth.tar"
    cp.write_bytes(b"model state")
    meta = {
        "mode": "formal", "training_dataset": "CUHK-PEDES/train",
        "surrogate": "official_IRRA_image_encoder",
        "irra_checkpoint_sha256": evaluation.OFFICIAL_CUHK_SHA256,
        "training_uses_captions": False, "training_uses_irra_text": False,
        "epoch": 60, "checkpoint_sha256": sha256(cp),
    }
    sidecar = cp.with_suffix(cp.suffix + ".json")
    sidecar.write_text(json.dumps(meta))
    assert evaluation.irra_surrogate_provenance(cp) == meta
    for key, bad in (("mode", "smoke"), ("training_uses_captions", True),
                     ("epoch", 59), ("checkpoint_sha256", "bad")):
        changed = {**meta, key: bad}
        sidecar.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="来源不匹配"):
            evaluation.irra_surrogate_provenance(cp)
