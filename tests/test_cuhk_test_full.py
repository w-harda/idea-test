"""全量 CUHK 运行入口的续跑完整性测试。"""
import json
from pathlib import Path

import pytest

from scripts.run_cuhk_test_full import (
    Query, _read_completed, _sha256, available_manifest_rows,
    build_sample500_manifest, load_query_manifest,
)


QUERIES = (
    Query("cuhk:1:0", "a.jpg", 1, "first"),
    Query("cuhk:2:0", "b.jpg", 2, "second"),
)


def _row(row_id: str, arm: str = "text_only") -> bytes:
    return (json.dumps({"row_id": row_id, "arm": arm}) + "\n").encode()


def test_completed_prefix_and_truncated_tail(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(_row("cuhk:1:0") + b'{"row_id":"cuhk:2:0"')
    assert _read_completed(path, QUERIES, "text_only") == 1
    assert path.read_bytes() == _row("cuhk:1:0")
    path.write_bytes(path.read_bytes() + _row("cuhk:2:0"))
    assert _read_completed(path, QUERIES, "text_only") == 2


def test_wrong_prefix_rejected(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(_row("cuhk:2:0"))
    with pytest.raises(ValueError, match="prefix"):
        _read_completed(path, QUERIES, "text_only")


def test_invalid_middle_row_rejected_without_mutation(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    original = _row("cuhk:1:0") + b"invalid\n" + _row("cuhk:2:0")
    path.write_bytes(original)
    with pytest.raises(ValueError, match="before final"):
        _read_completed(path, QUERIES, "text_only")
    assert path.read_bytes() == original


def test_sample500_reproducible_and_diverse(tmp_path: Path):
    annotation = tmp_path / "annotation.json"
    annotation.write_text("fixture", encoding="utf-8")
    queries = tuple(
        Query(f"q:{index}", f"image:{index // 2}.jpg", index // 6, "caption")
        for index in range(6156)
    )
    one = build_sample500_manifest(annotation, queries, seed=42)
    assert one == build_sample500_manifest(annotation, queries, seed=42)
    assert one != build_sample500_manifest(annotation, queries, seed=43)
    assert [row["row_id"] for row in one["queries"][:20]] == [
        query.row_id for query in queries[:20]]
    assert one["unique_person_ids"] == 484
    assert one["unique_images"] == 490
    manifest = tmp_path / "sample500_seed42.json"
    manifest.write_text(json.dumps(one), encoding="utf-8")
    selected = load_query_manifest(manifest, annotation, queries)
    assert selected[:20] == tuple(range(20))
    assert len(selected) == len(set(selected)) == 500
    assert all(index >= 20 for index in selected[20:])
    one["queries"][21]["person_id"] = -1
    manifest.write_text(json.dumps(one), encoding="utf-8")
    with pytest.raises(ValueError, match="pairing"):
        load_query_manifest(manifest, annotation, queries)


def test_manifest_resume_reuses_full_rows_and_skips_extra(tmp_path: Path):
    queries = tuple(
        Query(f"q:{index}", f"image:{index}.jpg", index, "caption")
        for index in range(5)
    )
    arm = "vanilla_image"
    arm_dir = tmp_path / "arms" / arm
    arm_dir.mkdir(parents=True)
    full = arm_dir / "results.jsonl"
    def row(index):
        return {"row_id": f"q:{index}", "arm": arm, "person_id": index,
                "image": f"image:{index}.jpg", "rank": index + 1}
    full.write_text("".join(json.dumps(row(i)) + chr(10) for i in range(3)),
                    encoding="utf-8")
    manifest = tmp_path / "sample500_seed42.json"
    manifest.write_text("{}", encoding="utf-8")
    selected = (0, 1, 4)
    assert set(available_manifest_rows(tmp_path, arm, queries, manifest, selected)) == {
        "q:0", "q:1"}
    extra = arm_dir / "sample500_seed42.jsonl"
    (arm_dir / "sample500_seed42.sha256").write_text(
        _sha256(manifest) + chr(10), encoding="ascii")
    extra.write_text(json.dumps(row(4)) + chr(10), encoding="utf-8")
    assert set(available_manifest_rows(tmp_path, arm, queries, manifest, selected)) == {
        "q:0", "q:1", "q:4"}
    assert full.read_text(encoding="utf-8").count(chr(10)) == 3
    manifest.write_text("{ }", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        available_manifest_rows(tmp_path, arm, queries, manifest, selected)
