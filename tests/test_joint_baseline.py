import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from attributes.attack_scheduler import ImageAttackResult, TextAttackResult, run_attack
from attributes.confusable_attack import CONFUSABLES, ConfusableTextCallback, character_candidates
from attributes.tta_adapter import TTAImageCallback


def record():
    text = "black pants"
    return {
        "row_id": "cuhk:0:0", "caption": text,
        "attributes": {"upper_clothing_color": "black", "lower_clothing_type": "trousers_shorts"},
        "shared_attributes": {"upper_clothing_color": "black",
                              "lower_clothing_type": "trousers_shorts"},
        "scores": {"upper_clothing_color": 2, "lower_clothing_type": 1},
        "k_star": 2,
        "selected_slots": ["upper_clothing_color", "lower_clothing_type"],
        "selected_attributes": {"upper_clothing_color": "black",
                                "lower_clothing_type": "trousers_shorts"},
        "provenance": {
            "upper_clothing_color": {"canonical": "black", "mentions": [
                {"start": 0, "end": 5, "raw": "black",
                 "linked_object": {"start": 6, "end": 11, "raw": "pants"}}]},
            "lower_clothing_type": {"canonical": "trousers_shorts", "mentions": [
                {"start": 6, "end": 11, "raw": "pants"}]},
        },
    }


def test_candidates_edit_only_one_character_inside_mention():
    row = record()
    candidates = list(character_candidates(row["caption"],
                                            row["provenance"]["upper_clothing_color"]))
    assert candidates
    for offset, replacement, text in candidates:
        assert 0 <= offset < 5
        assert replacement in CONFUSABLES[row["caption"][offset]]
        assert len(text) == len(row["caption"])
        assert [i for i, (a, b) in enumerate(zip(row["caption"], text)) if a != b] == [offset]
    assert list(character_candidates(row["caption"],
                                     row["provenance"]["upper_clothing_color"],
                                     set(range(5)))) == []


def test_text_callback_respects_stage05_order_and_preserves_future_spans():
    row = record()
    events = []

    def scorer(_image, texts):
        return torch.arange(len(texts), dtype=torch.float32)

    callback = ConfusableTextCallback(scorer)

    def image(request):
        events.append(("image", request.attribute.slot, request.state.text))
        return ImageAttackResult(request.state.image)

    def text(request):
        events.append(("text", request.attribute.slot, request.state.text))
        return callback(request)

    result = run_attack(row, image="pixels", image_attack=image, text_attack=text)
    assert [event[:2] for event in events] == [
        ("image", "upper_clothing_color"), ("text", "upper_clothing_color"),
        ("image", "lower_clothing_type"), ("text", "lower_clothing_type"),
    ]
    assert len(result.state.text) == len(row["caption"])
    assert len([i for i, (a, b) in enumerate(zip(row["caption"], result.state.text))
                if a != b]) == 2
    assert len(callback.used_offsets) == 2
    assert all(event["selected"] for event in callback.events)


def test_skip_is_allowed_and_does_not_change_text():
    row = record()
    callback = ConfusableTextCallback(
        lambda _image, texts: torch.tensor([1.0] + [0.0] * (len(texts) - 1)))
    result = run_attack(
        row, image="pixels",
        image_attack=lambda request: ImageAttackResult(request.state.image),
        text_attack=callback,
    )
    assert result.state.text == row["caption"]
    assert not any(event["selected"] for event in callback.events)


def test_tta_callback_projects_every_round_against_original(monkeypatch):
    seen_texts = []

    class FakeAttack:
        N_trans = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def img_attack(self, _model, texts, imgs, _origin, ids, _steps,
                       momentum, _device, scales=None):
            seen_texts.append((texts, ids, scales))
            return imgs + 0.1, momentum + 1

    monkeypatch.setattr("attributes.tta_adapter.load_official_attack",
                        lambda _root: FakeAttack)
    class Source:
        resolution = 2
    source = Source()
    original = torch.full((1, 3, 2, 2), 0.5)
    callback = TTAImageCallback(source, "/unused", original,
                                steps=1, transforms_per_scale=1)
    row = record()
    result = run_attack(
        row, image=original, image_attack=callback,
        text_attack=lambda request: TextAttackResult(request.state.text),
    )
    assert (result.state.image - original).abs().max().item() <= 8 / 255 + 1e-6
    assert torch.all(callback.momentum == 2)
    assert seen_texts == [(["black pants", "black"], [0, 0], (0.5, 0.75, 1.25, 1.5)),
                          (["black pants", "pants"], [0, 0], (0.5, 0.75, 1.25, 1.5))]


def test_vanilla_tta_uses_only_original_caption_and_two_image_passes(monkeypatch):
    from attributes.tta_adapter import vanilla_tta_image_attack

    calls = []

    class FakeAttack:
        N_trans = 0

        def __init__(self, *_args, **kwargs):
            assert kwargs["imgs_eps"] == pytest.approx(8 / 255)
            assert kwargs["step_size"] == pytest.approx(2 / 255)

        def img_attack(self, _model, texts, imgs, origin, ids, steps,
                       momentum, _device, scales=None):
            calls.append((texts, ids, steps, scales, momentum.clone(),
                          (imgs - origin).abs().max().item(), self.N_trans))
            return imgs + 0.1, momentum + 1

    monkeypatch.setattr("attributes.tta_adapter.load_official_attack",
                        lambda _root: FakeAttack)

    class Source:
        resolution = 2

    original = torch.full((1, 3, 2, 2), 0.5)
    attacked = vanilla_tta_image_attack(Source(), "/unused", original,
                                        "black pants")
    assert len(calls) == 2
    assert all(call[0] == ["black pants"] and call[1] == [0]
               and call[2] == 10 and call[3] == (0.5, 0.75, 1.25, 1.5)
               and call[6] == 6 for call in calls)
    assert torch.all(calls[0][4] == 0) and torch.all(calls[1][4] == 1)
    assert calls[1][5] <= 8 / 255 + 1e-6
    assert (attacked - original).abs().max().item() <= 8 / 255 + 1e-6
    with pytest.raises(ValueError):
        vanilla_tta_image_attack(Source(), "/unused", original, "")
