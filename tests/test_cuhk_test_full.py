"""全量 CUHK 运行入口的续跑完整性测试。"""
import json
from pathlib import Path

import pytest

from scripts.run_cuhk_test_full import Query, _read_completed


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
