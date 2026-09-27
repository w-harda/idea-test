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

def test_irra_500_output_guard_preserves_old_results():
    from scripts.run_irra_surrogate_poc import OLD20_OUTPUT

    with pytest.raises(ValueError, match="cannot overwrite"):
        _output_guard(OLD20_OUTPUT)
    with pytest.raises(ValueError, match="cannot overwrite"):
        _output_guard(OLD20_OUTPUT / "arms" / "clean")
    _output_guard(DEFAULT_OUTPUT)


def test_irra_manifest_resume_by_row_id(tmp_path):
    import json
    from scripts.run_irra_surrogate_poc import _prior_rows, SOURCE_LABEL

    path = tmp_path / "results.jsonl"
    rows = [
        {"row_id": "q:12", "arm": "clean", "attack_source": SOURCE_LABEL,
         "checkpoint_sha256": "checkpoint", "config_sha256": "config",
         "query_manifest_sha256": "manifest"},
        {"row_id": "q:3", "arm": "clean", "attack_source": SOURCE_LABEL,
         "checkpoint_sha256": "checkpoint", "config_sha256": "config",
         "query_manifest_sha256": "manifest"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")
    result = _prior_rows(path, "clean", {"q:3", "q:12", "q:99"},
                         "checkpoint", "config", "manifest")
    assert set(result) == {"q:12", "q:3"}
    with pytest.raises(ValueError, match="metadata differs"):
        _prior_rows(path, "clean", {"q:3", "q:12"},
                    "checkpoint", "config", "changed")
    with pytest.raises(ValueError, match="outside manifest"):
        _prior_rows(path, "clean", {"q:3"},
                    "checkpoint", "config", "manifest")


def test_irra_summary_requires_exact_manifest_rows(tmp_path):
    import json
    from scripts.run_cuhk_test_full import ARMS, Query
    from scripts.run_irra_surrogate_poc import SOURCE_LABEL, summarize

    queries = (
        Query("q:0", "a.jpg", 1, "first"),
        Query("q:1", "b.jpg", 2, "second"),
        Query("q:2", "c.jpg", 3, "third"),
        Query("q:3", "d.jpg", 4, "fourth"),
    )
    selected = (2, 0, 3)
    (tmp_path / "official_clean_validation.json").write_text(
        json.dumps({"checkpoint_sha256": "checkpoint",
                    "config_sha256": "config"}), encoding="utf-8")
    for arm in ARMS:
        directory = tmp_path / "arms" / arm
        directory.mkdir(parents=True)
        rows = [
            {"row_id": "q:0", "arm": arm, "attack_source": SOURCE_LABEL,
             "checkpoint_sha256": "checkpoint", "config_sha256": "config",
             "query_manifest_sha256": "manifest", "clean_rank": 1,
             "rank": 1 if arm == "clean" else 2,
             "text_changed": False, "linf": 0.0},
            {"row_id": "q:2", "arm": arm, "attack_source": SOURCE_LABEL,
             "checkpoint_sha256": "checkpoint", "config_sha256": "config",
             "query_manifest_sha256": "manifest", "clean_rank": 3,
             "rank": 3 if arm == "clean" else 4,
             "text_changed": False, "linf": 0.0},
            {"row_id": "q:3", "arm": arm, "attack_source": SOURCE_LABEL,
             "checkpoint_sha256": "checkpoint", "config_sha256": "config",
             "query_manifest_sha256": "manifest", "clean_rank": 4,
             "rank": 4 if arm == "clean" else 5,
             "text_changed": False, "linf": 0.0},
        ]
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summarize(tmp_path, queries, selected, "manifest")
    report = json.loads(
        (tmp_path / "summary_500_irra_white_box_8arms.json").read_text(
            encoding="utf-8"))
    assert report["query_row_ids"] == ["q:2", "q:0", "q:3"]
    assert all(item["query_count"] == 3 for item in report["arms"].values())
    extra = tmp_path / "arms" / "clean" / "results.jsonl"
    with extra.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "row_id": "q:1", "arm": "clean", "attack_source": SOURCE_LABEL,
            "checkpoint_sha256": "checkpoint", "config_sha256": "config",
            "query_manifest_sha256": "manifest", "clean_rank": 2,
            "rank": 2, "text_changed": False, "linf": 0.0}) + "\n")
    with pytest.raises(ValueError, match="exactly the manifest"):
        summarize(tmp_path, queries, selected, "manifest")
