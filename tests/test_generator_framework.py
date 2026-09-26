from copy import deepcopy
import json

import pytest
torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from attributes.cuhk_training import load_cuhk_training_index
from attributes.generator_framework import (
    CharacterEdit, GeneratorTrainer, TextDistribution, TrainingBatch,
    apply_text_edits, prepare_query, project_image, validate_gallery_subset,
)
from attributes.rank_objective import (
    calibrate_tau, exact_first_hit_rank, query_utility, rank_loss,
    soft_first_hit_rank,
)


def record(selected=True):
    slots = ["upper_clothing_color"] if selected else []
    return {
        "row_id": "cuhk:0:0", "caption": "red shirt", "id": 999, "image": "secret.png",
        "attributes": {"upper_clothing_color": "red"},
        "shared_attributes": {"upper_clothing_color": "red"},
        "scores": {"upper_clothing_color": 2},
        "k_star": len(slots), "selected_slots": slots,
        "selected_attributes": {slot: "red" for slot in slots},
        "provenance": {"upper_clothing_color": {
            "canonical": "red", "mentions": [{"raw": "red", "start": 0, "end": 3}],
        }},
    }


def allowed(query, slot, offset):
    assert not hasattr(query, "id") and not hasattr(query, "image")
    return ("X",)


class ToySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.calls = 0

    def encode_images(self, images):
        self.calls += 1
        value = images.mean(dim=(1, 2, 3)) * self.weight
        return F.normalize(torch.stack((value, 1 - value), dim=1), dim=1)

    def encode_texts(self, texts):
        self.calls += 1
        values = [[0.0, 1.0] if "X" in text else [1.0, 0.0] for text in texts]
        return torch.tensor(values, dtype=torch.float32)


class ToyImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(0.005))

    def forward(self, images):
        return images + self.shift


class ToyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.preference = nn.Parameter(torch.tensor(0.1))
        self.select_calls = 0

    def distribution(self, query):
        assert query.plan[0].slot == "upper_clothing_color"
        return TextDistribution(((), (CharacterEdit("upper_clothing_color", 0, "X"),)),
                                torch.stack((self.preference * 0, self.preference)))

    def select(self, query):
        self.select_calls += 1
        return (CharacterEdit("upper_clothing_color", 0, "X"),)


def make_trainer():
    source, gi, gt = ToySource(), ToyImage(), ToyText()
    trainer = GeneratorTrainer(source, gi, gt, allowed,
                               torch.optim.SGD(gi.parameters(), lr=0.1),
                               torch.optim.SGD(gt.parameters(), lr=0.1), tau=0.1)
    return trainer, source, gi, gt


def test_soft_rank_counts_wrong_images_not_wrong_ids_and_has_gradient():
    scores = torch.tensor([[0.8, 0.7, 0.9, 0.85]], requires_grad=True)
    rank = soft_first_hit_rank(scores, [1], [1, 1, 2, 2], 0.01)
    assert 2.9 < rank.item() < 3.1
    assert exact_first_hit_rank(scores, [1], [1, 1, 2, 2]).item() == 3
    rank.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 2] > 0
    assert query_utility(torch.tensor([-2.0, 0.0, 3.0])).tolist() == pytest.approx(
        [-2.0, 0.0, torch.log(torch.tensor(4.0)).item()]
    )
    assert rank_loss(scores.detach(), scores, [1], [1, 1, 2, 2], 0.1).item() == pytest.approx(0)
    assert calibrate_tau(scores.detach(), [1], [1, 1, 2, 2]) > 0


def test_query_budget_and_stage05_mapping():
    query = prepare_query(record())
    assert [item.slot for item in query.plan] == ["upper_clothing_color"]
    assert apply_text_edits(query, (CharacterEdit("upper_clothing_color", 0, "X"),), allowed) == "Xed shirt"
    for edits in ((CharacterEdit("upper_clothing_color", 4, "X"),),
                  (CharacterEdit("upper_clothing_color", 0, "X"),
                   CharacterEdit("upper_clothing_color", 1, "X")),
                  (CharacterEdit("upper_clothing_color", 0, "ZZ"),)):
        with pytest.raises(ValueError):
            apply_text_edits(query, edits, allowed)
    broken = deepcopy(record())
    broken["provenance"]["upper_clothing_color"]["mentions"][0]["start"] = 1
    with pytest.raises(ValueError):
        prepare_query(broken)


def test_image_projection_is_relative_to_original():
    original = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 3, 1, 1)
    projected = project_image(original, original + 1)
    assert projected.flatten().tolist() == pytest.approx([8 / 255, 0.5 + 8 / 255, 1.0])
    assert (projected - original).abs().max() <= 8 / 255 + 1e-7
    with pytest.raises(ValueError):
        project_image(original, torch.full_like(original, float("nan")))


def test_subset_requires_all_same_id_images():
    validate_gallery_subset([1], ["a", "b", "c"], [1, 1, 2],
                            ["a", "b", "c", "d"], [1, 1, 2, 3])
    with pytest.raises(ValueError):
        validate_gallery_subset([1], ["a", "c"], [1, 2],
                                ["a", "b", "c"], [1, 1, 2])


def test_alternating_train_and_frozen_test_use_distinct_information():
    trainer, source, gi, gt = make_trainer()
    images = torch.stack([torch.full((3, 2, 2), value) for value in (0.2, 0.3, 0.8)])
    batch = TrainingBatch(images, ("a", "b", "c"), (1, 1, 2),
                          (record(),), (1,), ("a", "b", "c", "d"), (1, 1, 2, 3))
    before_gi, before_gt = gi.shift.item(), gt.preference.item()
    result = trainer.train_step(batch)
    assert set(result) == {"image_loss", "text_loss"}
    assert gi.shift.item() != before_gi
    assert gt.preference.item() < before_gt
    assert source.weight.grad is None
    calls = source.calls
    adv = trainer.infer_gallery(images, ("a", "b", "c"))
    texts = trainer.infer_queries((record(), record(False)))
    assert source.calls == calls
    assert texts == ("Xed shirt", "red shirt")
    assert gt.select_calls == 2  # training G_I and test; zero-k skips
    assert (adv - images).abs().max() <= 8 / 255 + 1e-7
    with pytest.raises(ValueError):
        trainer.infer_gallery(images, ("a", "a", "c"))


def test_cuhk_supervision_joins_training_rows_only(tmp_path):
    annotation = [
        {"split": "train", "file_path": "a.png", "id": 7, "captions": ["red shirt"]},
        {"split": "test", "file_path": "b.png", "id": 8, "captions": ["blue shirt"]},
    ]
    annotation_path = tmp_path / "reid_raw.json"
    stage04_path = tmp_path / "stage04.jsonl"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    row = record()
    row.update({"split": "train", "image": "a.png"})
    stage04_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    index = load_cuhk_training_index(annotation_path, stage04_path)
    assert index.gallery_paths == ("a.png",)
    assert index.gallery_ids == (7,)
    assert index.query_ids == (7,)
    assert len(index.records) == 1
    row["caption"] = "wrong caption"
    stage04_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pairing mismatch"):
        load_cuhk_training_index(annotation_path, stage04_path)


def test_test_gallery_invokes_image_generator_independently():
    trainer, _, _, _ = make_trainer()

    class BatchAwareImage(nn.Module):
        def forward(self, images):
            return images + images.mean() * 0.01

    trainer.image_generator = BatchAwareImage()
    images = torch.stack([torch.full((3, 2, 2), value) for value in (0.2, 0.8)])
    together = trainer.infer_gallery(images, ("a", "b"))
    separate = torch.cat((trainer.infer_gallery(images[:1], ("a",)),
                          trainer.infer_gallery(images[1:], ("b",))), dim=0)
    assert torch.allclose(together, separate)
