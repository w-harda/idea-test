"""Random-K 调度的确定性、候选池与 Stage 05 顺序测试。"""
from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import pytest

from attributes.attack_scheduler import ImageAttackResult, TextAttackResult, run_attack
from attributes.random_k import apply_random_k, random_k_entry
from scripts.run_cuhk_test_full import (
    GUIDED_ARMS, IMAGE_ARMS, Query, load_random_k_manifest, pairwise_rank_gap,
)


CAPTION = "A man in a red shirt and blue pants."


def record(row_id="cuhk:1:0"):
    values = {"gender": "male", "upper_clothing_color": "red",
              "lower_clothing_color": "blue"}
    words = {"gender": "man", "upper_clothing_color": "red",
             "lower_clothing_color": "blue"}
    return {
        "row_id": row_id, "caption": CAPTION,
        "attributes": values.copy(), "shared_attributes": values.copy(),
        "scores": {"gender": 100, "upper_clothing_color": 10,
                   "lower_clothing_color": 1},
        "provenance": {
            slot: {"canonical": value, "mentions": [{
                "raw": words[slot], "start": CAPTION.index(words[slot]),
                "end": CAPTION.index(words[slot]) + len(words[slot]),
            }]}
            for slot, value in values.items()
        },
        "k_star": 2,
        "selected_slots": ["gender", "upper_clothing_color"],
        "selected_attributes": {
            slot: values[slot] for slot in ("gender", "upper_clothing_color")
        },
    }


def test_per_query_randomness_ignores_global_rng_scores_and_execution_order():
    first = record()
    second = record("cuhk:2:0")
    original_rng = random.getstate()
    try:
        random.seed(1234)
        before = random.getstate()
        entry = random_k_entry(first, seed=42)
        assert random.getstate() == before
        random_k_entry(second, seed=42)
        assert random_k_entry(first, seed=42) == entry
        random.seed(999)
        assert random_k_entry(first, seed=42) == entry
    finally:
        random.setstate(original_rng)
    altered = deepcopy(first)
    altered["scores"] = {"gender": 1, "upper_clothing_color": 500,
                         "lower_clothing_color": 900}
    assert random_k_entry(altered, seed=42)["random_attack_order"] == (
        entry["random_attack_order"])
    assert len(entry["random_attack_order"]) == len(set(entry["random_attack_order"])) == 2
    assert set(entry["random_selected_slots"]) == set(entry["random_attack_order"])
    assert entry["candidate_slots"] == [
        "gender", "upper_clothing_color", "lower_clothing_color"]


def test_random_schedule_preserves_image_then_text_and_original_record():
    original = record()
    original_copy = deepcopy(original)
    entry = next(
        random_k_entry(original, seed=seed)
        for seed in range(100)
        if random_k_entry(original, seed=seed)["random_attack_order"]
        != original["selected_slots"]
    )
    scheduled = apply_random_k(original, entry, seed=entry["seed"])
    assert original == original_copy
    assert scheduled["selected_slots"] == entry["random_attack_order"]
    assert scheduled["k_star"] == original["k_star"]
    events = []
    def image(request):
        events.append(("image", request.attribute.slot))
        return ImageAttackResult(request.state.image)
    def text(request):
        events.append(("text", request.attribute.slot))
        return TextAttackResult(request.state.text)
    result = run_attack(scheduled, image="image", image_attack=image, text_attack=text)
    assert [item.slot for item in result.completed] == entry["random_attack_order"]
    assert events == [
        (kind, slot) for slot in entry["random_attack_order"]
        for kind in ("image", "text")
    ]
    assert {"random_k_tta_only", "random_k_tta_text"} <= GUIDED_ARMS & IMAGE_ARMS


def test_zero_k_skips_callbacks_and_manifest_order_is_checked(tmp_path: Path):
    zero = record()
    zero["k_star"] = 0
    zero["selected_slots"] = []
    zero["selected_attributes"] = {}
    entry = random_k_entry(zero)
    assert entry["random_selected_slots"] == entry["random_attack_order"] == []
    scheduled = apply_random_k(zero, entry)
    def fail(_request):
        pytest.fail("k_star=0 must not call attack callbacks")
    assert run_attack(scheduled, image="image", image_attack=fail,
                      text_attack=fail).completed == ()

    one = record()
    two = record("cuhk:2:0")
    queries = (Query(one["row_id"], "a.jpg", 1, one["caption"]),
               Query(two["row_id"], "b.jpg", 2, two["caption"]))
    records = {one["row_id"]: one, two["row_id"]: two}
    path = tmp_path / "random_k_seed42.jsonl"
    rows = [random_k_entry(one), random_k_entry(two)]
    path.write_text("".join(json.dumps(row) + chr(10) for row in rows),
                    encoding="utf-8")
    assert list(load_random_k_manifest(path, queries, (0, 1), records)) == [
        one["row_id"], two["row_id"]]
    path.write_text("".join(json.dumps(row) + chr(10) for row in reversed(rows)),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="schedule differs"):
        load_random_k_manifest(path, queries, (0, 1), records)


def test_pairwise_rank_gap_uses_matched_row_ids():
    topk = [{"row_id": "a", "rank": 6}, {"row_id": "b", "rank": 4},
            {"row_id": "c", "rank": 1}]
    random_rows = [{"row_id": "a", "rank": 3}, {"row_id": "b", "rank": 4},
                   {"row_id": "c", "rank": 4}]
    assert pairwise_rank_gap(topk, random_rows) == {
        "topk_stronger_count": 1, "same_count": 1,
        "random_k_stronger_count": 1,
        "mean_rank_gap_topk_minus_random": 0.0,
        "median_rank_gap_topk_minus_random": 0.0,
    }
    with pytest.raises(ValueError, match="query IDs"):
        pairwise_rank_gap(topk, list(reversed(random_rows)))
