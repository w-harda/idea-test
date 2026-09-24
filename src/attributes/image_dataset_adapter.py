"""只读取图像路径与 split；绝不读取 captions 或文本属性。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator


IMAGE_KEYS = {"cuhk": "file_path", "icfg": "file_path", "rstp": "img_path"}


@dataclass(frozen=True)
class ImageRecord:
    image: str
    path: Path


class TbpsImageAdapter:
    def __init__(self, annotation: str | Path, image_root: str | Path,
                 dataset: str, split: str | None = None):
        if dataset not in IMAGE_KEYS:
            raise ValueError(f"未知数据集: {dataset}")
        self.annotation = Path(annotation)
        self.image_root = Path(image_root)
        self.dataset = dataset
        self.split = split

    def iter_records(self) -> Iterator[ImageRecord]:
        with self.annotation.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("图像 annotation 顶层必须是数组")
        seen: set[str] = set()
        key = IMAGE_KEYS[self.dataset]
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"第 {index} 条图像记录必须是对象")
            if self.split is not None and item.get("split") != self.split:
                continue
            image = item.get(key)
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"第 {index} 条图像记录缺少有效 {key}")
            image = image.replace("\\", "/")
            relative = PurePosixPath(image)
            if (relative.is_absolute() or not relative.parts or
                    any(part == ".." for part in relative.parts) or ":" in image):
                raise ValueError(f"非法图像相对路径: {image}")
            normalized = relative.as_posix()
            if normalized in seen:
                continue
            seen.add(normalized)
            path = self.image_root.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(self.image_root.resolve()):
                raise ValueError(f"图像路径超出 image_root: {image}")
            yield ImageRecord(normalized, path)
