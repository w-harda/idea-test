import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from attributes.dataset_adapter import (
    CaptionRecord, CuhkPedesAdapter, IcfgPedesAdapter, JsonCaptionAdapter, RstpReidAdapter,
)


def test_generic_jsonl_adapter_keeps_data_loading_separate() -> None:
    path = Path(__file__).parent / "fixtures" / "captions.jsonl"
    assert list(JsonCaptionAdapter(path).iter_records()) == [
        CaptionRecord("p1", "a man wearing a white shirt"),
        CaptionRecord(1, "without a hat"),
    ]


def test_cuhk_adapter_flattens_captions_and_filters_split() -> None:
    path = Path(__file__).parent / "fixtures" / "cuhk_reid_raw.json"
    assert list(CuhkPedesAdapter(path, "train").iter_records()) == [
        CaptionRecord("train_query/p1.jpg#0", "a man wearing a white shirt and black pants"),
        CaptionRecord("train_query/p1.jpg#1", "without a hat"),
    ]
    assert len(list(CuhkPedesAdapter(path).iter_records())) == 3


def _read_annotations(adapter_type: type, data: object, split: str | None = None) -> list[CaptionRecord]:
    with patch.object(Path, "open", return_value=StringIO(json.dumps(data))):
        return list(adapter_type("annotation.json", split).iter_records())


def test_icfg_adapter_filters_split_and_expands_every_caption() -> None:
    data = [
        {"split": "train", "file_path": "train/a.jpg", "id": 1,
         "processed_tokens": [["ignored"]], "captions": ["A woman wearing a red shirt."]},
        {"split": "test", "file_path": "test/b.jpg", "id": 2,
         "captions": ["caption one", "caption two"]},
    ]
    assert _read_annotations(IcfgPedesAdapter, data, "train") == [
        CaptionRecord("train/a.jpg#0", "A woman wearing a red shirt."),
    ]
    assert _read_annotations(IcfgPedesAdapter, data) == [
        CaptionRecord("train/a.jpg#0", "A woman wearing a red shirt."),
        CaptionRecord("test/b.jpg#0", "caption one"),
        CaptionRecord("test/b.jpg#1", "caption two"),
    ]


def test_rstp_adapter_expands_captions_and_uses_record_split() -> None:
    data = [
        {"id": 0, "img_path": "0000.jpg", "captions": ["caption one", "caption two"], "split": "train"},
        {"id": 1, "img_path": "0001.jpg", "captions": ["test caption"], "split": "custom-test"},
    ]
    assert _read_annotations(RstpReidAdapter, data, "train") == [
        CaptionRecord("0000.jpg#0", "caption one"),
        CaptionRecord("0000.jpg#1", "caption two"),
    ]
    assert _read_annotations(RstpReidAdapter, data, "custom-test") == [
        CaptionRecord("0001.jpg#0", "test caption"),
    ]
    assert len(_read_annotations(RstpReidAdapter, data)) == 3


@pytest.mark.parametrize(
    ("adapter_type", "image_key"),
    [(IcfgPedesAdapter, "file_path"), (RstpReidAdapter, "img_path")],
)
def test_image_caption_adapters_reject_invalid_annotations(adapter_type: type, image_key: str) -> None:
    cases = [
        ({}, "顶层必须是数组"),
        (["not a record"], "记录必须是对象"),
        ([{"captions": []}], image_key),
        ([{image_key: "image.jpg", "captions": "not a list"}], "captions 必须是列表"),
        ([{image_key: "image.jpg", "captions": [1]}], "captions\\[0\\] 必须是字符串"),
    ]
    for data, message in cases:
        with pytest.raises(ValueError, match=message):
            _read_annotations(adapter_type, data)
