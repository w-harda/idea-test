"""Stream Stage 03 JSONL into complete Dynamic Top-K results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attributes.dynamic_topk import select_attributes


DATASET_NAMES = {"cuhk": "CUHK-PEDES", "icfg": "ICFG-PEDES", "rstp": "RSTPReid"}


def select_dataset(dataset: str, source: Path, output: Path, beta: float) -> int:
    count = 0
    seen_row_ids: set[str] = set()
    temporary: Path | None = None
    try:
        with source.open("r", encoding="utf-8") as reader, tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as writer:
            temporary = Path(writer.name)
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    raise ValueError(f"{source}:{line_number}: empty line")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{source}:{line_number}: expected an object")
                if row.get("dataset") != DATASET_NAMES[dataset]:
                    raise ValueError(f"{source}:{line_number}: dataset mismatch")
                row_id = row.get("row_id")
                if not isinstance(row_id, str) or not row_id.startswith(f"{dataset}:"):
                    raise ValueError(f"{source}:{line_number}: invalid row_id")
                if row_id in seen_row_ids:
                    raise ValueError(f"{source}:{line_number}: duplicate row_id: {row_id}")
                seen_row_ids.add(row_id)
                if "selected_attributes" in row:
                    raise ValueError(f"{source}:{line_number}: already contains Stage 04 data")
                selected = select_attributes(
                    row.get("shared_attributes"), row.get("scores"), beta
                )
                writer.write(json.dumps({**row, **selected}, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError(f"{source}: no Stage 03 records")
        os.replace(temporary, output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Select Stage 04 Dynamic Top-K")
    parser.add_argument("--dataset", required=True, choices=tuple(DATASET_NAMES))
    parser.add_argument("--input", required=True, type=Path, help="Stage 03 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Stage 04 JSONL")
    parser.add_argument("--beta", default=1.0, type=float)
    args = parser.parse_args()
    if args.input.suffix.lower() != ".jsonl" or args.output.suffix.lower() != ".jsonl":
        parser.error("input and output must end in .jsonl")
    if args.input.resolve() == args.output.resolve():
        parser.error("output must differ from input")
    if not 0 < args.beta <= 1:
        parser.error("--beta must satisfy 0 < beta <= 1")
    if not args.output.parent.is_dir():
        parser.error("output parent directory must exist")
    count = select_dataset(args.dataset, args.input, args.output, args.beta)
    print(f"{DATASET_NAMES[args.dataset]}: selected {count} captions (beta={args.beta})")


if __name__ == "__main__":
    main()
