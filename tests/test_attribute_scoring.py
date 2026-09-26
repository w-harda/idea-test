import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from attributes.attribute_scoring import AttributeGallery, SUPPORTED_SLOTS
from attributes.text_extractor import extract_with_provenance
from attributes.visual_canonicalizer import SLOTS


def attrs(**values):
    return {slot: values.get(slot, "null") for slot in SLOTS}


def gallery(*items):
    return AttributeGallery([
        {"image": f"image-{index}", "attributes": item}
        for index, item in enumerate(items)
    ])


def reference(query, images):
    shared = {slot: query[slot] for slot in SUPPORTED_SLOTS if query[slot] != "null"}
    slots = tuple(shared)
    valid = [
        index for index, image in enumerate(images)
        if sum(image[slot] == "null" for slot in slots) <= len(slots) / 2
    ]

    def candidates(selected):
        return {
            index for index in valid
            if all(images[index][slot] in ("null", shared[slot]) for slot in selected)
            and any(images[index][slot] == shared[slot] for slot in selected)
        }

    baseline = candidates(slots)
    return len(valid), len(baseline), {
        slot: len(candidates(tuple(other for other in slots if other != slot)) - baseline)
        for slot in slots
    }


def test_conflict_removal_counts_new_entrants_only():
    query = attrs(gender="female", hat="yes")
    images = [
        attrs(gender="female", hat="yes"),  # baseline
        attrs(gender="female", hat="no"),   # enters after deleting hat
        attrs(gender="male", hat="yes"),    # enters after deleting gender
        attrs(gender="female", hat="null"), # baseline, then may leave
        attrs(gender="null", hat="null"),   # excluded, fixed for both deletions
        attrs(gender="male", hat="no"),     # two conflicts
    ]
    result = gallery(*images).score(query)
    assert result["valid_gallery_count"] == 5
    assert result["candidate_count"] == 2
    assert result["scores"] == {"gender": 1, "hat": 1}


def test_fixed_valid_gallery_and_match_requirement():
    query = attrs(age="adult", gender="female", hat="yes")
    images = [
        attrs(age="young", gender="female", hat="null"),
        attrs(age="adult", gender="null", hat="null"),
        attrs(age="null", gender="null", hat="yes"),
        attrs(age="adult", gender="female", hat="yes"),
    ]
    result = gallery(*images).score(query)
    valid, candidates, scores = reference(query, images)
    assert (result["valid_gallery_count"], result["candidate_count"], result["scores"]) == (
        valid, candidates, scores
    )
    assert valid == 2  # two nulls remain excluded after deletion


def test_unsupported_upper_type_and_empty_query():
    indexed = gallery(attrs(upper_clothing_type="null", gender="female"))
    result = indexed.score(attrs(upper_clothing_type="jacket_coat"))
    assert result["shared_attributes"] == {}
    assert result["valid_gallery_count"] == 1
    assert result["candidate_count"] == 0
    assert result["scores"] == {}


def test_random_cases_match_direct_set_definition():
    rng = random.Random(28)
    slots = ("age", "gender", "hat", "bag")
    values = {
        "age": ("adult", "young", "null"),
        "gender": ("female", "male", "null"),
        "hat": ("yes", "no", "null"),
        "bag": ("yes", "no", "null"),
    }
    images = [
        attrs(**{slot: rng.choice(values[slot]) for slot in slots})
        for _ in range(100)
    ]
    indexed = gallery(*images)
    for _ in range(30):
        query = attrs(**{slot: rng.choice(values[slot]) for slot in slots})
        result = indexed.score(query)
        valid, candidates, scores = reference(query, images)
        assert (result["valid_gallery_count"], result["candidate_count"], result["scores"]) == (
            valid, candidates, scores
        )


def test_gallery_rejects_duplicate_paths_and_bad_attributes():
    record = {"image": "same.jpg", "attributes": attrs(gender="female")}
    with pytest.raises(ValueError, match="duplicate"):
        AttributeGallery([record, record])
    with pytest.raises(ValueError, match="13 string slots"):
        gallery({"gender": "female"})


def test_cli_scores_all_splits_and_preserves_duplicate_legacy_id(tmp_path):
    root = Path(__file__).resolve().parents[1]
    annotation = tmp_path / "annotation.json"
    gallery_path = tmp_path / "gallery.jsonl"
    output = tmp_path / "scores.jsonl"
    rows = [
        {"file_path": "same.jpg", "split": "train", "captions": ["A woman with a hat."]},
        {"file_path": "same.jpg", "split": "train", "captions": ["A man without a hat."]},
        {"file_path": "other.jpg", "split": "test", "captions": ["A woman in red."]},
    ]
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    gallery_path.write_text("\n".join(json.dumps(record) for record in (
        {"image": "same.jpg", "attributes": attrs(gender="female", hat="yes")},
        {"image": "other.jpg", "attributes": attrs(gender="female", hat="no")},
    )) + "\n", encoding="utf-8")
    subprocess.run([
        sys.executable, str(root / "scripts" / "score_attributes.py"),
        "--dataset", "icfg", "--annotation", str(annotation),
        "--gallery", str(gallery_path), "--output", str(output),
    ], check=True)
    results = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(results) == 3
    assert [row["split"] for row in results] == ["train", "train", "test"]
    assert [row["row_id"] for row in results] == ["icfg:0:0", "icfg:1:0", "icfg:2:0"]
    assert results[0]["id"] == results[1]["id"] == "same.jpg#0"
    assert [row["caption"] for row in results] == [r["captions"][0] for r in rows]
    assert all(row["dataset"] == "ICFG-PEDES" for row in results)
    assert all(row["gallery_count"] == 2 for row in results)
    assert all(tuple(row["attributes"]) == SLOTS for row in results)
    assert all(
        row["provenance"] == extract_with_provenance(row["caption"])["provenance"]
        for row in results
    )
    assert all(row["scores"].keys() == row["shared_attributes"].keys() for row in results)


def test_cli_rejects_cross_dataset_gallery_before_writing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    annotation = tmp_path / "annotation.json"
    gallery_path = tmp_path / "wrong_gallery.jsonl"
    output = tmp_path / "scores.jsonl"
    annotation.write_text(json.dumps([
        {"file_path": "cuhk.jpg", "split": "train", "captions": ["A woman."]}
    ]), encoding="utf-8")
    gallery_path.write_text(json.dumps({
        "image": "icfg.jpg", "attributes": attrs(gender="female")
    }) + "\n", encoding="utf-8")
    process = subprocess.run([
        sys.executable, str(root / "scripts" / "score_attributes.py"),
        "--dataset", "cuhk", "--annotation", str(annotation),
        "--gallery", str(gallery_path), "--output", str(output),
    ], capture_output=True, text=True)
    assert process.returncode != 0
    assert "different image sets" in process.stderr
    assert not output.exists()
