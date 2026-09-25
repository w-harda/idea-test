import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from attributes.dynamic_topk import select_attributes


def test_concentrated_scores_select_one():
    result = select_attributes(
        {"age": "young", "gender": "female", "hat": "yes"},
        {"age": 20, "gender": 0, "hat": 0},
    )
    assert result["weights"] == {"age": 1.0, "gender": 0.0, "hat": 0.0}
    assert result["entropy"] == 0
    assert result["n_eff"] == 1
    assert result["k_star"] == 1
    assert result["selected_slots"] == ["age"]
    assert result["selected_attributes"] == {"age": "young"}


def test_uniform_scores_select_all_and_ties_use_canonical_order():
    result = select_attributes(
        {"hat": "yes", "age": "adult", "gender": "female"},
        {"hat": 5, "age": 5, "gender": 5},
    )
    assert result["entropy"] == pytest.approx(math.log(3))
    assert result["n_eff"] == pytest.approx(3)
    assert result["k_star"] == 3
    assert result["selected_slots"] == ["age", "gender", "hat"]


def test_beta_controls_budget_and_scores_determine_rank():
    result = select_attributes(
        {"hat": "yes", "age": "adult", "gender": "female", "bag": "yes"},
        {"hat": 10, "age": 10, "gender": 10, "bag": 10},
        beta=0.5,
    )
    assert result["k_star"] == 2
    assert result["selected_slots"] == ["age", "gender"]
    assert result["selected_attributes"] == {"age": "adult", "gender": "female"}


@pytest.mark.parametrize("shared,scores", [
    ({}, {}),
    ({"age": "young", "hat": "yes"}, {"age": 0, "hat": 0}),
])
def test_zero_signal_selects_nothing(shared, scores):
    result = select_attributes(shared, scores)
    assert result["k_star"] == 0
    assert result["selected_attributes"] == {}
    assert result["selected_slots"] == []
    assert result["n_eff"] == 0
    assert sum(result["weights"].values()) == 0


@pytest.mark.parametrize("beta", [0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        select_attributes({"age": "young"}, {"age": 1}, beta)


@pytest.mark.parametrize("shared,scores", [
    ({"age": "young"}, {"age": -1}),
    ({"age": "young"}, {"age": True}),
    ({"age": "young"}, {}),
    ({"upper_clothing_type": "jacket_coat"}, {"upper_clothing_type": 1}),
    ({"age": "null"}, {"age": 1}),
])
def test_invalid_stage03_scoring_rejected(shared, scores):
    with pytest.raises(ValueError):
        select_attributes(shared, scores)


def test_cli_preserves_stage03_records_and_all_splits(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "stage03.jsonl"
    output = tmp_path / "stage04.jsonl"
    rows = [
        {
            "dataset": "CUHK-PEDES", "split": split, "row_id": f"cuhk:{index}:0",
            "caption": f"caption {index}", "attributes": {"age": "young"},
            "shared_attributes": {"age": "young", "hat": "yes"},
            "scores": {"age": 3, "hat": 1}, "candidate_count": 7,
        }
        for index, split in enumerate(("train", "val", "test"))
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    subprocess.run([
        sys.executable, str(root / "scripts" / "select_dynamic_topk.py"),
        "--dataset", "cuhk", "--input", str(source), "--output", str(output),
    ], check=True)
    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 3
    for before, after in zip(rows, results, strict=True):
        assert all(after[key] == value for key, value in before.items())
        assert after["selected_slots"] == ["age", "hat"]
        assert after["selected_attributes"] == {"age": "young", "hat": "yes"}


def test_cli_rejects_duplicate_id_without_replacing_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "stage03.jsonl"
    output = tmp_path / "stage04.jsonl"
    row = {
        "dataset": "CUHK-PEDES", "row_id": "cuhk:0:0",
        "shared_attributes": {"age": "young"}, "scores": {"age": 1},
    }
    source.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    output.write_text("existing", encoding="utf-8")
    process = subprocess.run([
        sys.executable, str(root / "scripts" / "select_dynamic_topk.py"),
        "--dataset", "cuhk", "--input", str(source), "--output", str(output),
    ], capture_output=True, text=True)
    assert process.returncode != 0
    assert "duplicate row_id" in process.stderr
    assert output.read_text(encoding="utf-8") == "existing"
