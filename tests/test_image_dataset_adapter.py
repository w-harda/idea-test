import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from attributes.image_dataset_adapter import ImageRecord, TbpsImageAdapter


@pytest.mark.parametrize("dataset,key", [
    ("cuhk", "file_path"), ("icfg", "file_path"), ("rstp", "img_path")])
def test_image_only_deduplicates_and_filters_split(dataset, key) -> None:
    data = [
        {"split": "train", key: "a/p1.jpg", "captions": "malformed but unread"},
        {"split": "train", key: "a/p1.jpg", "attributes": {"gender": "male"}},
        {"split": "train", key: "a\\p2.jpg", "provenance": ["ignored"]},
        {"split": "test", key: "a/p3.jpg"},
    ]
    with patch.object(Path, "open", return_value=StringIO(json.dumps(data))):
        records = list(TbpsImageAdapter("annotations.json", "images", dataset, "train").iter_records())
    assert records == [
        ImageRecord("a/p1.jpg", Path("images") / "a" / "p1.jpg"),
        ImageRecord("a/p2.jpg", Path("images") / "a" / "p2.jpg"),
    ]


@pytest.mark.parametrize("image", ["../escape.jpg", "/abs/p.jpg", "C:/abs/p.jpg", "", "."])
def test_rejects_unsafe_image_paths(image) -> None:
    with patch.object(Path, "open", return_value=StringIO(json.dumps([{"file_path": image}]))):
        with pytest.raises(ValueError):
            list(TbpsImageAdapter("annotations.json", "images", "cuhk").iter_records())
