"""对通用 JSON/JSONL caption annotation 批量提取文本属性。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attributes.dataset_adapter import CuhkPedesAdapter, JsonCaptionAdapter
from attributes.text_extractor import TextAttributeExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="批量提取 TBPS caption 属性")
    parser.add_argument("--input", required=True, type=Path, help="输入 .json 或 .jsonl")
    parser.add_argument("--output", required=True, type=Path, help="输出 .json 或 .jsonl")
    parser.add_argument("--adapter", choices=("flat", "cuhk"), default="flat", help="annotation 格式")
    parser.add_argument("--split", help="CUHK split 名称，例如 train、val、test；默认全部")
    parser.add_argument("--limit", type=int, help="仅处理前 N 条 caption，便于抽查")
    args = parser.parse_args()
    if args.output.suffix.lower() not in (".json", ".jsonl"):
        parser.error("输出文件必须使用 .json 或 .jsonl 后缀")
    if args.input.resolve() == args.output.resolve():
        parser.error("输入和输出不能是同一文件")
    if args.split and args.adapter != "cuhk":
        parser.error("--split 仅适用于 --adapter cuhk")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")

    extractor = TextAttributeExtractor()
    adapter = CuhkPedesAdapter(args.input, args.split) if args.adapter == "cuhk" else JsonCaptionAdapter(args.input)
    captions = adapter.iter_records()
    if args.limit is not None:
        captions = itertools.islice(captions, args.limit)
    records = (
        {"id": record.id, "caption": record.caption, "attributes": extractor.extract(record.caption)}
        for record in captions
    )
    with args.output.open("w", encoding="utf-8") as handle:
        if args.output.suffix.lower() == ".jsonl":
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(list(records), handle, ensure_ascii=False, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
