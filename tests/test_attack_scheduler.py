from copy import deepcopy

import pytest

from attributes.attack_scheduler import (
    ImageAttackResult,
    TextAttackResult,
    plan_attack,
    run_attack,
)
from attributes.text_extractor import extract_with_provenance


CAPTION = "A young woman wears a red shirt and black pants."


def make_record(slots=("lower_clothing_color", "upper_clothing_color")):
    attributes = extract_with_provenance(CAPTION)["attributes"]
    selected = {slot: attributes[slot] for slot in slots}
    assert all(value != "null" for value in selected.values())
    return {
        "row_id": "cuhk:0:0",
        "caption": CAPTION,
        "attributes": attributes,
        "shared_attributes": selected,
        "scores": {"lower_clothing_color": 1, "upper_clothing_color": 10},
        "k_star": len(slots),
        "selected_slots": list(slots),
        "selected_attributes": selected,
    }


def shift_targets(targets, distance):
    updated = deepcopy(targets)
    for target in updated.values():
        for mention in target["mentions"]:
            mention["start"] += distance
            mention["end"] += distance
            if "linked_object" in mention:
                mention["linked_object"]["start"] += distance
                mention["linked_object"]["end"] += distance
    return updated


def test_complete_rounds_preserve_stage04_order_and_transfer_current_state():
    row = make_record()
    events = []

    def image_attack(request):
        events.append(("image", request.round_index, request.attribute.slot))
        if request.round_index == 2:
            assert request.state.text.startswith("Very ")
            assert request.state.image == "image|lower_clothing_color"
        return ImageAttackResult(
            f"{request.state.image}|{request.attribute.slot}",
            guidance=f"guide-{request.round_index}",
        )

    def text_attack(request):
        events.append(("text", request.round_index, request.attribute.slot))
        assert request.state.image.endswith(request.attribute.slot)
        assert request.image_result.guidance == f"guide-{request.round_index}"
        mention = request.target["mentions"][0]
        assert request.state.text[mention["start"]:mention["end"]] == mention["raw"]
        if request.round_index == 1:
            return TextAttackResult(
                "Very " + request.state.text,
                shift_targets(request.state.targets, 5),
            )
        return TextAttackResult(request.state.text)

    result = run_attack(
        row, image="image", image_attack=image_attack, text_attack=text_attack,
    )
    assert events == [
        ("image", 1, "lower_clothing_color"),
        ("text", 1, "lower_clothing_color"),
        ("image", 2, "upper_clothing_color"),
        ("text", 2, "upper_clothing_color"),
    ]
    assert result.state.image == "image|lower_clothing_color|upper_clothing_color"
    assert result.state.text == "Very " + CAPTION
    assert result.state.targets == {}
    assert [item.slot for item in result.completed] == row["selected_slots"]
    assert [item.score for item in result.completed] == [1, 10]


def test_zero_k_does_not_invoke_attack_callbacks():
    row = make_record(())
    called = []

    def attack(_request):
        called.append(True)
        raise AssertionError("no attack call expected")

    result = run_attack(row, image="original", image_attack=attack, text_attack=attack)
    assert result.state.image == "original"
    assert result.state.text == CAPTION
    assert result.completed == ()
    assert called == []


@pytest.mark.parametrize("change", [
    {"k_star": 1},
    {"selected_slots": ["upper_clothing_color", "upper_clothing_color"]},
    {"selected_attributes": {"upper_clothing_color": "blue"}},
])
def test_inconsistent_stage04_selection_is_rejected(change):
    row = make_record()
    row.update(change)
    with pytest.raises(ValueError):
        plan_attack(row)


def test_stage01_attribute_mismatch_is_rejected_before_callbacks():
    row = make_record()
    row["attributes"] = {**row["attributes"], "age": "adult"}
    with pytest.raises(ValueError, match="Stage 01"):
        run_attack(
            row, image="image",
            image_attack=lambda _: pytest.fail("unexpected image call"),
            text_attack=lambda _: pytest.fail("unexpected text call"),
        )


def test_text_edit_must_keep_remaining_target_spans_valid():
    row = make_record()

    with pytest.raises(ValueError, match="span"):
        run_attack(
            row,
            image="image",
            image_attack=lambda request: ImageAttackResult(request.state.image),
            text_attack=lambda request: TextAttackResult("Very " + request.state.text),
        )


def test_wrong_callback_result_is_rejected():
    row = make_record(("upper_clothing_color",))
    with pytest.raises(TypeError, match="ImageAttackResult"):
        run_attack(
            row, image="image",
            image_attack=lambda _: "image",
            text_attack=lambda _: pytest.fail("unexpected text call"),
        )
