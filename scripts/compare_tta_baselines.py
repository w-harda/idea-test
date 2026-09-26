#!/usr/bin/env python3
"""合并相同三条 query 的 Clean、Vanilla 与属性引导三臂结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_GALLERY_SHA256 = "e1505afef38cb62aa5e22dacf9d60a4e62871243763669e4a50e0a492fd88788"


def compare(attribute_report: dict, vanilla_report: dict) -> dict:
    if (attribute_report["queries"] != 3
            or attribute_report["gallery_images"] != 29
            or vanilla_report["gallery_images"] != 29
            or vanilla_report["gallery_sha256"] != EXPECTED_GALLERY_SHA256
            or attribute_report["seed"] != vanilla_report["seed"]
            or abs(attribute_report["tau"] - vanilla_report["tau"]) > 1e-6):
        raise ValueError("reports do not use the same three-query gallery setup")
    left, right = attribute_report["results"], vanilla_report["results"]
    if len(left) != 3 or len(right) != 3:
        raise ValueError("expected three query results in both reports")
    merged = []
    labels = ("clean", "vanilla_tta", "attribute_guided_tta_only",
              "text_only", "attribute_guided_tta_text")
    for attribute, vanilla in zip(left, right, strict=True):
        for field in ("row_id", "image", "caption", "person_id", "clean_rank"):
            if attribute[field] != vanilla[field]:
                raise ValueError(f"query mismatch for {field}")
        if abs(attribute["clean_soft_rank"] - vanilla["clean_soft_rank"]) > 1e-5:
            raise ValueError("clean source scores differ")
        arms = attribute["arms"]
        ranks = {
            "clean": attribute["clean_rank"],
            "vanilla_tta": vanilla["vanilla_rank"],
            "attribute_guided_tta_only": arms["tta_only"]["rank"],
            "text_only": arms["text_only"]["rank"],
            "attribute_guided_tta_text": arms["tta_text"]["rank"],
        }
        soft_ranks = {
            "clean": attribute["clean_soft_rank"],
            "vanilla_tta": vanilla["vanilla_soft_rank"],
            "attribute_guided_tta_only": arms["tta_only"]["soft_rank"],
            "text_only": arms["text_only"]["soft_rank"],
            "attribute_guided_tta_text": arms["tta_text"]["soft_rank"],
        }
        merged.append({"row_id": attribute["row_id"], "ranks": ranks,
                       "soft_ranks": soft_ranks})
    mean_ranks = {name: sum(item["ranks"][name] for item in merged) / 3
                  for name in labels}
    mean_deltas = {name: mean_ranks[name] - mean_ranks["clean"]
                   for name in labels}
    return {
        "dataset": "CUHK-PEDES train", "queries": 3, "gallery_images": 29,
        "gallery_sha256": EXPECTED_GALLERY_SHA256,
        "vanilla_definition": "two official img_attack passes, original caption only, no text attack",
        "mean_ranks": mean_ranks, "mean_rank_deltas": mean_deltas,
        "results": merged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribute-report", required=True)
    parser.add_argument("--vanilla-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.attribute_report).open(encoding="utf-8") as handle:
        attribute_report = json.load(handle)
    with Path(args.vanilla_report).open(encoding="utf-8") as handle:
        vanilla_report = json.load(handle)
    comparison = compare(attribute_report, vanilla_report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"mean_ranks": comparison["mean_ranks"],
                      "mean_rank_deltas": comparison["mean_rank_deltas"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
