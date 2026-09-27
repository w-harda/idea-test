"""IRRA-only source and TTA bridge contracts without loading a checkpoint."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from attributes.irra_surrogate import IRRASurrogate
from attributes.irra_victim import MEAN, STD
from attributes.tta_adapter import _tokenize
from attributes.tta_irra import _FullIRRABridge, IRRATTAImageCallback
from scripts.run_irra_surrogate_poc import _output_guard, DEFAULT_OUTPUT


class FakeIRRAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.3, 0.5, -0.2],
            ]))

    def encode_image(self, normalized):
        return self.projection(normalized.mean(dim=(2, 3)))


class FakeVictim(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeIRRAModel()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def encode_images(self, pixels):
        return F.normalize(
            self.model.encode_image((pixels - self.mean) / self.std), dim=-1)

    def encode_texts(self, texts):
        return F.normalize(torch.tensor([
            [1.0, 0.4 + len(text) * 0.001, -0.2, 0.1]
            for text in texts
        ], device=self.mean.device), dim=-1)


def test_irra_source_image_gradient_and_retrieval_scores():
    source = IRRASurrogate(FakeVictim(), require_cuda=False)
    pixels = torch.full((1, 3, 384, 128), 0.6, requires_grad=True)
    normalized = (pixels - source.victim.mean) / source.victim.std
    raw = source.inference_image(normalized)["image_feat"]
    assert torch.allclose(raw, source.encode_images(pixels))
    assert torch.allclose(
        source.inference_text(_tokenize(["red shirt"]))["text_feat"],
        source.encode_texts(["red shirt"]))
    loss = source.attack_loss(pixels, ["red shirt"])
    gradient = torch.autograd.grad(loss, pixels)[0]
    assert gradient.shape == pixels.shape
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0
    assert not any(parameter.requires_grad for parameter in source.parameters())

    gallery = torch.cat([raw.detach(), -raw.detach()], dim=0)
    scores = source.retrieval_scores(["red shirt"], gallery)
    assert scores.shape == (1, 2)
    rank = source.soft_rank(["red shirt"], gallery, [1], [1, 2], 0.05)
    assert rank.shape == (1,) and torch.isfinite(rank).all()


def test_tta_image_callback_uses_irra_gradient_and_budget(monkeypatch):
    import attributes.tta_irra as adapter

    source = IRRASurrogate(FakeVictim(), require_cuda=False)
    original = torch.full((1, 3, 384, 128), 0.6)

    class FakeOfficialAttack:
        def __init__(self, _ref, tokenizer, *, imgs_eps, step_size):
            self.tokenizer = tokenizer
            assert imgs_eps == pytest.approx(8 / 255)
            assert step_size == pytest.approx(2 / 255)

        def img_attack(self, model, texts, image, clean, ids, steps,
                       momentum, device, *, scales):
            assert model is source
            assert texts == ["red shirt", "red"]
            assert ids == [0, 0] and steps == 10
            assert scales == (0.5, 0.75, 1.25, 1.5)
            image = image.detach().requires_grad_(True)
            normalized = (image - source.victim.mean) / source.victim.std
            img = model.inference_image(normalized)["image_feat"]
            txt = model.inference_text(self.tokenizer(texts))["text_feat"]
            gradient = torch.autograd.grad(-(img @ txt.T).sum(), image)[0]
            assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
            return image.detach() + gradient.sign() * 0.1, momentum

    monkeypatch.setattr(adapter, "_check_source", lambda *_: None)
    monkeypatch.setattr(adapter, "load_official_attack",
                        lambda *_: FakeOfficialAttack)
    callback = IRRATTAImageCallback(source, "unused", original)
    request = SimpleNamespace(
        state=SimpleNamespace(
            image=original, text="red shirt",
            targets={"upper_clothing_color": {"mentions": [{"raw": "red"}]}}),
        attribute=SimpleNamespace(slot="upper_clothing_color"))
    result = callback(request)
    assert result.guidance["attribute_word"] == "red"
    assert (result.image - original).abs().max() <= 8 / 255 + 1e-6


def test_full_tta_text_bridge_and_output_isolation():
    source = IRRASurrogate(FakeVictim(), require_cuda=False)

    class FakeBertTokenizer:
        def decode(self, ids):
            assert ids == [1, 2]
            return "[CLS] red shirt [SEP] [PAD]"

    bridge = _FullIRRABridge(source, FakeBertTokenizer())
    ids = SimpleNamespace(input_ids=torch.tensor([[1, 2]]))
    assert torch.allclose(
        bridge.inference_text(ids)["text_feat"],
        source.encode_texts(["red shirt"]))
    with pytest.raises(ValueError, match="cannot overwrite"):
        _output_guard(DEFAULT_OUTPUT.parent / "irra-victim-poc")
    _output_guard(DEFAULT_OUTPUT)
