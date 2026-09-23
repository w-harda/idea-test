from pathlib import Path

from attributes.dataset_adapter import CaptionRecord, CuhkPedesAdapter, JsonCaptionAdapter


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
