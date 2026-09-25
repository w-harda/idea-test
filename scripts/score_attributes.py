"""Score precomputed text attributes against a precomputed visual gallery."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from attributes.attribute_scoring import AttributeGallery


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                yield value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Stage 03 attribute scores")
    parser.add_argument("--queries", required=True, type=Path, help="Stage 01 JSONL")
    parser.add_argument("--gallery", required=True, type=Path, help="Stage 02 JSONL")
    parser.add_argument("--output", required=True, type=Path, help="score JSONL")
    parser.add_argument("--limit", type=int, help="first N queries for inspection")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.output.suffix.lower() != ".jsonl":
        parser.error("--output must end in .jsonl")
    if args.output.resolve() in (args.queries.resolve(), args.gallery.resolve()):
        parser.error("output must differ from inputs")

    gallery = AttributeGallery(read_jsonl(args.gallery))
    queries = read_jsonl(args.queries)
    if args.limit is not None:
        queries = itertools.islice(queries, args.limit)
    with args.output.open("w", encoding="utf-8") as handle:
        for query in queries:
            if "id" not in query:
                raise ValueError("query record needs id")
            result = gallery.score(query.get("attributes"))
            handle.write(json.dumps({"id": query["id"], **result}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
