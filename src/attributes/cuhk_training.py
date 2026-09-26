"""CUHK-PEDES 训练配对和 ID：仅供损失与图库子集校验。"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CuhkTrainingIndex:
    gallery_paths: tuple[str, ...]
    gallery_ids: tuple[int | str, ...]
    records: tuple[Mapping[str, Any], ...]
    query_ids: tuple[int | str, ...]


def load_cuhk_training_index(annotation_path: str | Path,
                             stage04_path: str | Path) -> CuhkTrainingIndex:
    """只选 train，并验证 Stage 04 row_id、文本、图片与原始标注一致。"""
    with Path(annotation_path).open(encoding="utf-8") as handle:
        annotation = json.load(handle)
    if not isinstance(annotation, list):
        raise ValueError("CUHK annotation must be a JSON array")
    galleries: dict[str, int | str] = {}
    expected: dict[str, tuple[str, str, int | str]] = {}
    for row_index, row in enumerate(annotation):
        if not isinstance(row, dict) or row.get("split") != "train":
            continue
        image, person_id, captions = row.get("file_path"), row.get("id"), row.get("captions")
        if (not isinstance(image, str) or not image
                or not isinstance(person_id, (str, int)) or isinstance(person_id, bool)
                or not isinstance(captions, list)
                or any(not isinstance(caption, str) for caption in captions)):
            raise ValueError(f"invalid train annotation row {row_index}")
        if image in galleries and galleries[image] != person_id:
            raise ValueError(f"conflicting person ID for {image}")
        galleries[image] = person_id
        for caption_index, caption in enumerate(captions):
            expected[f"cuhk:{row_index}:{caption_index}"] = (image, caption, person_id)
    records: list[Mapping[str, Any]] = []
    ids: list[int | str] = []
    seen: set[str] = set()
    with Path(stage04_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Stage 04 line {line_number} is not an object")
            if record.get("split") != "train":
                continue
            row_id = record.get("row_id")
            if row_id in seen or row_id not in expected:
                raise ValueError(f"duplicate or unknown train row_id: {row_id}")
            image, caption, person_id = expected[row_id]
            if record.get("image") != image or record.get("caption") != caption:
                raise ValueError(f"Stage 04 pairing mismatch: {row_id}")
            seen.add(row_id)
            records.append(record)
            ids.append(person_id)
    if seen != set(expected):
        raise ValueError(f"Stage 04 missing {len(set(expected) - seen)} train captions")
    return CuhkTrainingIndex(tuple(galleries), tuple(galleries.values()),
                             tuple(records), tuple(ids))
