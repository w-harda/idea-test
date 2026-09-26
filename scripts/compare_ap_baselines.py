#!/usr/bin/env python3
"""固定三条 query / 29 图库的九项 TTA 与 AP-Attack 对照。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_GALLERY_SHA256 = "e1505afef38cb62aa5e22dacf9d60a4e62871243763669e4a50e0a492fd88788"
LABELS = (
    "clean", "vanilla_tta_image_only", "vanilla_tta_full", "ap_attack_full",
    "text_only", "attribute_guided_tta_only", "attribute_guided_tta_text",
    "attribute_guided_ap_only", "attribute_guided_ap_text",
)


def compare(six: dict, ap_full: dict, ap_guided: dict) -> dict:
    for name, report in (("full", ap_full), ("guided", ap_guided)):
        if (report["mode"] != "development_paired_image_ap_attack_" + name
                or report["queries"] != 3 or report["gallery_images"] != 29
                or report["gallery_sha256"] != EXPECTED_GALLERY_SHA256
                or report["attribute_conditioned_inference"] is not False
                or len(report["results"]) != 3):
            raise ValueError(f"invalid AP-Attack {name} report")
    if (six["queries"] != 3 or six["gallery_images"] != 29
            or six["gallery_sha256"] != EXPECTED_GALLERY_SHA256
            or len(six["results"]) != 3
            or ap_full["seed"] != ap_guided["seed"]
            or ap_full["ap_checkpoint_sha256"] != ap_guided["ap_checkpoint_sha256"]
            or abs(ap_full["tau"] - ap_guided["tau"]) > 1e-6):
        raise ValueError("reports do not share the same three-query setup")

    merged = []
    for base, full, guided in zip(
            six["results"], ap_full["results"], ap_guided["results"], strict=True):
        for other in (full, guided):
            if (other["row_id"] != base["row_id"]
                    or other["clean_rank"] != base["ranks"]["clean"]
                    or abs(other["clean_soft_rank"] - base["soft_ranks"]["clean"]) > 1e-5):
                raise ValueError("query or clean score mismatch")
        for field in ("row_id", "image", "caption", "person_id", "clean_rank"):
            if full[field] != guided[field]:
                raise ValueError(f"AP-Attack full/guided mismatch for {field}")
        if (full["text"] != full["caption"] or full["generator_calls"] != 1
                or guided["arms"]["ap_only"]["text"] != guided["caption"]):
            raise ValueError("AP-Attack image-only arm changed text")
        for arm in guided["arms"].values():
            if (arm["rounds"] != guided["selected_slots"]
                    or arm["generator_calls"] != len(guided["selected_slots"])):
                raise ValueError("guided AP-Attack order/call count mismatch")
        for item in (full, *guided["arms"].values()):
            if item["linf_native"] > 8 / 255 + 1e-6 or item["linf_clip"] > 8 / 255 + 1e-6:
                raise ValueError("AP-Attack image exceeds 8/255")
        ranks = {
            "clean": base["ranks"]["clean"],
            "vanilla_tta_image_only": base["ranks"]["vanilla_tta_image_only"],
            "vanilla_tta_full": base["ranks"]["vanilla_tta_full"],
            "ap_attack_full": full["full_rank"],
            "text_only": base["ranks"]["text_only"],
            "attribute_guided_tta_only": base["ranks"]["attribute_guided_tta_only"],
            "attribute_guided_tta_text": base["ranks"]["attribute_guided_tta_text"],
            "attribute_guided_ap_only": guided["arms"]["ap_only"]["rank"],
            "attribute_guided_ap_text": guided["arms"]["ap_text"]["rank"],
        }
        soft = {
            "clean": base["soft_ranks"]["clean"],
            "vanilla_tta_image_only": base["soft_ranks"]["vanilla_tta_image_only"],
            "vanilla_tta_full": base["soft_ranks"]["vanilla_tta_full"],
            "ap_attack_full": full["full_soft_rank"],
            "text_only": base["soft_ranks"]["text_only"],
            "attribute_guided_tta_only": base["soft_ranks"]["attribute_guided_tta_only"],
            "attribute_guided_tta_text": base["soft_ranks"]["attribute_guided_tta_text"],
            "attribute_guided_ap_only": guided["arms"]["ap_only"]["soft_rank"],
            "attribute_guided_ap_text": guided["arms"]["ap_text"]["soft_rank"],
        }
        merged.append({"row_id": base["row_id"], "ranks": ranks, "soft_ranks": soft})
    means = {name: sum(row["ranks"][name] for row in merged) / 3 for name in LABELS}
    return {
        "dataset": "CUHK-PEDES train", "queries": 3, "gallery_images": 29,
        "gallery_sha256": EXPECTED_GALLERY_SHA256,
        "ap_checkpoint_sha256": ap_full["ap_checkpoint_sha256"],
        "ap_full_definition": "one frozen image-only generator pass; semantic guidance was used during Duke stage2 training",
        "ap_guided_definition": "Stage 04 order gates repeated image-only generator calls; generator has no attribute input",
        "mean_ranks": means,
        "mean_rank_deltas": {name: means[name] - means["clean"] for name in LABELS},
        "results": merged,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--six-report", required=True)
    parser.add_argument("--ap-full-report", required=True)
    parser.add_argument("--ap-guided-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = []
    for path in (args.six_report, args.ap_full_report, args.ap_guided_report):
        with Path(path).open(encoding="utf-8") as handle:
            reports.append(json.load(handle))
    result = compare(*reports)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps({"mean_ranks": result["mean_ranks"],
                      "mean_rank_deltas": result["mean_rank_deltas"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
