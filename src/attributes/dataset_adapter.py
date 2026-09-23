"""caption 数据读取边界；具体数据集格式可在此接口下单独接入。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol


@dataclass(frozen=True)
class CaptionRecord:
    id: str | int
    caption: str


class CaptionAdapter(Protocol):
    def iter_records(self) -> Iterator[CaptionRecord]: ...


class JsonCaptionAdapter:
    """通用 JSON 数组或 JSONL：每项为 caption 字符串或带 id/caption 的对象。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def iter_records(self) -> Iterator[CaptionRecord]:
        if self.path.suffix.lower() == ".jsonl":
            with self.path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if line.strip():
                        yield self._record(json.loads(line), index)
        elif self.path.suffix.lower() == ".json":
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("JSON annotation 顶层必须是数组")
            for index, item in enumerate(data):
                yield self._record(item, index)
        else:
            raise ValueError("输入必须是 .json 或 .jsonl 文件")

    @staticmethod
    def _record(item: object, index: int) -> CaptionRecord:
        if isinstance(item, str):
            return CaptionRecord(index, item)
        if isinstance(item, dict) and isinstance(item.get("caption"), str):
            record_id = item.get("id", index)
            if not isinstance(record_id, (str, int)) or isinstance(record_id, bool):
                raise ValueError(f"第 {index} 条记录的 id 必须是字符串或整数")
            return CaptionRecord(record_id, item["caption"])
        raise ValueError(f"第 {index} 条记录必须是 caption 字符串或包含 caption 的对象")


class CuhkPedesAdapter:
    """读取 CUHK-PEDES 原始 reid_raw.json，每条 caption 单独输出。"""

    def __init__(self, path: str | Path, split: str | None = None):
        self.path = Path(path)
        self.split = split

    def iter_records(self) -> Iterator[CaptionRecord]:
        with self.path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("CUHK-PEDES annotation 顶层必须是数组")
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"第 {index} 条 CUHK 记录必须是对象")
            if self.split is not None and item.get("split") != self.split:
                continue
            captions = item.get("captions")
            image_path = item.get("file_path")
            if not isinstance(image_path, str) or not isinstance(captions, list):
                raise ValueError(f"第 {index} 条 CUHK 记录缺少 file_path 或 captions")
            for caption_index, caption in enumerate(captions):
                if not isinstance(caption, str):
                    raise ValueError(f"第 {index} 条 CUHK 记录的 captions[{caption_index}] 不是字符串")
                yield CaptionRecord(f"{image_path}#{caption_index}", caption)


def _iter_image_captions(
    path: Path, split: str | None, image_key: str, dataset_name: str
) -> Iterator[CaptionRecord]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{dataset_name} annotation 顶层必须是数组")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 条 {dataset_name} 记录必须是对象")
        if split is not None and item.get("split") != split:
            continue
        image_path = item.get(image_key)
        captions = item.get("captions")
        if not isinstance(image_path, str):
            raise ValueError(f"第 {index} 条 {dataset_name} 记录的 {image_key} 必须是字符串")
        if not isinstance(captions, list):
            raise ValueError(f"第 {index} 条 {dataset_name} 记录的 captions 必须是列表")
        for caption_index, caption in enumerate(captions):
            if not isinstance(caption, str):
                raise ValueError(
                    f"第 {index} 条 {dataset_name} 记录的 captions[{caption_index}] 必须是字符串"
                )
            yield CaptionRecord(f"{image_path}#{caption_index}", caption)


class IcfgPedesAdapter:
    """读取 ICFG-PEDES.json 中每张图像的全部 captions。"""

    def __init__(self, path: str | Path, split: str | None = None):
        self.path = Path(path)
        self.split = split

    def iter_records(self) -> Iterator[CaptionRecord]:
        yield from _iter_image_captions(self.path, self.split, "file_path", "ICFG-PEDES")


class RstpReidAdapter:
    """读取 data_captions.json 中每张图像的全部 captions。"""

    def __init__(self, path: str | Path, split: str | None = None):
        self.path = Path(path)
        self.split = split

    def iter_records(self) -> Iterator[CaptionRecord]:
        yield from _iter_image_captions(self.path, self.split, "img_path", "RSTPReid")
