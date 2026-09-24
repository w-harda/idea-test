"""单图或 TBPS 数据集的 GPU batch 图像属性提取。"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attributes.image_dataset_adapter import ImageRecord, TbpsImageAdapter
from attributes.upar_inference import Upar40Predictor
from attributes.visual_canonicalizer import UPAR40_NAMES, VisualCanonicalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预训练 UPAR40 图像属性 GPU 推理")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="单张图像路径")
    inputs.add_argument("--annotation", type=Path, help="TBPS 数据集 JSON annotation")
    parser.add_argument("--dataset", choices=("cuhk", "icfg", "rstp"))
    parser.add_argument("--image-root", type=Path, help="annotation 中图像相对路径的根目录")
    parser.add_argument("--split", help="只处理指定 split")
    parser.add_argument("--upar-source", required=True, type=Path,
                        help="仓库外的官方 upar_challenge 源码目录")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="输出 JSONL；包含原始 UPAR40 概率")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="最多处理 N 张不同图像")
    parser.add_argument("--resume", action="store_true", help="跳过输出文件中已完成的图像并追加结果")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="每完成 N 张新图像报告一次进度")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.annotation and (args.dataset is None or args.image_root is None):
        parser.error("--annotation 需要同时提供 --dataset 和 --image-root")
    if args.image and (args.dataset or args.image_root or args.split):
        parser.error("--image 不使用 --dataset、--image-root 或 --split")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size 和 --limit 必须大于 0")
    if args.progress_every < 1:
        parser.error("--progress-every 必须大于 0")
    if args.output.suffix.lower() != ".jsonl":
        parser.error("--output 必须是 .jsonl")
    if args.annotation and args.annotation.resolve() == args.output.resolve():
        parser.error("输入 annotation 和输出不能是同一文件")
    return args


def completed_images(output: Path) -> set[str]:
    """读取已写入的图像路径；遇到损坏的结果时停止，避免静默跳过。"""
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{output}:{line_number} 不是完整 JSON，不能续跑") from exc
            image = row.get("image") if isinstance(row, dict) else None
            if not isinstance(image, str) or not image or image in completed:
                raise ValueError(f"{output}:{line_number} 的图像路径无效或重复")
            completed.add(image)
    return completed


def main() -> None:
    args = parse_args()
    completed = completed_images(args.output) if args.resume else set()
    if args.image:
        records = iter((ImageRecord(args.image.name, args.image),))
    else:
        records = TbpsImageAdapter(
            args.annotation, args.image_root, args.dataset, args.split).iter_records()
    if args.limit is not None:
        records = itertools.islice(records, args.limit)
    records = (record for record in records if record.image not in completed)
    batch = list(itertools.islice(records, args.batch_size))
    if not batch:
        print(f"没有待推理图像；已有 {len(completed)} 张，结果: {args.output}", flush=True)
        return
    print(f"开始加载 UPAR 模型；已有 {len(completed)} 张可续跑图像", flush=True)
    predictor = Upar40Predictor(args.upar_source, args.checkpoint, args.device)
    canonicalizer = VisualCanonicalizer()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    started = time.monotonic()
    next_report = args.progress_every
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        while batch:
            probabilities = predictor.predict_paths([record.path for record in batch])
            for record, scores in zip(batch, probabilities, strict=True):
                result = {
                    "image": record.image,
                    "attributes": canonicalizer.canonicalize(scores),
                    "upar40": dict(zip(UPAR40_NAMES, scores, strict=True)),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            if count >= next_report:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(f"进度: 本次 {count} 张，累计 {len(completed) + count} 张，"
                      f"平均 {count / elapsed:.2f} 张/秒", flush=True)
                next_report = (count // args.progress_every + 1) * args.progress_every
            batch = list(itertools.islice(records, args.batch_size))
    print(f"已推理 {count} 张新图像，累计 {len(completed) + count} 张，结果: {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
