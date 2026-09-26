#!/usr/bin/env python3
"""CUHK-PEDES 开发样本的 Stage 04→05→TTA/字符替换三臂 PoC。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attributes.attack_scheduler import ImageAttackResult, TextAttackResult, run_attack
from attributes.confusable_attack import CONFUSABLES, ConfusableTextCallback, character_candidates
from attributes.cuhk_training import load_cuhk_training_index
from attributes.frozen_clip import FrozenCLIP
from attributes.rank_objective import calibrate_tau, exact_first_hit_rank, soft_first_hit_rank
from attributes.tta_adapter import TTAImageCallback


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--stage04", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--negative-images", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--transforms-per-scale", type=int, default=6)
    parser.add_argument("--scales", default="0.5,0.75,1.25,1.5")
    parser.add_argument("--seed", type=int, default=20260926)
    return parser.parse_args()


def pick_queries(index, count):
    selected = []
    seen_ids = set()
    for record, person_id in zip(index.records, index.query_ids, strict=True):
        if person_id in seen_ids or not 1 <= record["k_star"] <= 2:
            continue
        if not any(list(character_candidates(record["caption"], record["provenance"][slot]))
                   for slot in record["selected_slots"]):
            continue
        selected.append((record, person_id))
        seen_ids.add(person_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("insufficient real train queries with 1–2 attack attributes")
    return selected


def make_gallery(index, selected, negative_count, seed):
    positives = {person_id for _, person_id in selected}
    positive_paths = [path for path, person_id in
                      zip(index.gallery_paths, index.gallery_ids, strict=True)
                      if person_id in positives]
    negatives = [(path, person_id) for path, person_id in
                 zip(index.gallery_paths, index.gallery_ids, strict=True)
                 if person_id not in positives]
    random.Random(seed).shuffle(negatives)
    chosen = positive_paths + [path for path, _ in negatives[:negative_count]]
    id_by_path = dict(zip(index.gallery_paths, index.gallery_ids, strict=True))
    return chosen, [id_by_path[path] for path in chosen]


def main():
    args = arguments()
    if args.queries < 1 or args.negative_images < 1:
        raise ValueError("queries and negative-images must be positive")
    scales = None if args.scales == "none" else tuple(float(x) for x in args.scales.split(","))
    index = load_cuhk_training_index(args.annotation, args.stage04)
    selected = pick_queries(index, args.queries)
    paths, gallery_ids = make_gallery(index, selected, args.negative_images, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = FrozenCLIP(args.checkpoint, device=device).to(device)
    prep = transforms.Compose([
        transforms.Resize(source.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(source.resolution),
        transforms.ToTensor(),
    ])
    root = Path(args.image_root).resolve()
    images = []
    for path in paths:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("gallery path escapes image root")
        with Image.open(resolved) as handle:
            images.append(prep(handle.convert("RGB")))
    gallery_images = torch.stack(images).to(device)
    with torch.no_grad():
        gallery_features = source.encode_images(gallery_images)
        query_features = source.encode_texts([record["caption"] for record, _ in selected])
        clean_scores = query_features @ gallery_features.T
    query_ids = [person_id for _, person_id in selected]
    tau = calibrate_tau(clean_scores, query_ids, gallery_ids)
    results = []
    for query_index, (record, person_id) in enumerate(selected):
        paired = paths.index(record["image"])
        original = gallery_images[paired:paired + 1]
        base_rank = int(exact_first_hit_rank(clean_scores[query_index:query_index + 1],
                                             [person_id], gallery_ids)[0])
        arms = {}
        for arm in ("tta_only", "text_only", "tta_text"):
            torch.manual_seed(args.seed + query_index)
            np.random.seed(args.seed + query_index)
            if arm != "text_only":
                image_callback = TTAImageCallback(
                    source, args.tta_root, original, steps=args.steps,
                    transforms_per_scale=args.transforms_per_scale, scales=scales,
                )
            else:
                image_callback = lambda request: ImageAttackResult(request.state.image)

            def candidate_scores(current_image, texts):
                with torch.no_grad():
                    features = gallery_features.clone()
                    features[paired:paired + 1] = source.encode_images(current_image)
                    scores = source.encode_texts(texts) @ features.T
                    return soft_first_hit_rank(scores, [person_id] * len(texts),
                                               gallery_ids, tau)

            if arm != "tta_only":
                text_callback = ConfusableTextCallback(candidate_scores)
            else:
                text_callback = lambda request: TextAttackResult(request.state.text)
            result = run_attack(record, image=original, image_attack=image_callback,
                                text_attack=text_callback)
            events = getattr(text_callback, "events", [])
            edits = {event["offset"]: event for event in events if event["selected"]}
            differences = {offset for offset, (before, after) in enumerate(
                zip(record["caption"], result.state.text)) if before != after}
            if (len(result.state.text) != len(record["caption"])
                    or differences != set(edits)
                    or (result.state.image - original).abs().max() > 8 / 255 + 1e-6):
                raise AssertionError("attack exceeded the image/text hard budget")
            for offset, event in edits.items():
                mentions = record["provenance"][event["slot"]]["mentions"]
                if (not any(mention["start"] <= offset < mention["end"]
                            for mention in mentions)
                        or result.state.text[offset] not in CONFUSABLES.get(record["caption"][offset], ())):
                    raise AssertionError("text edit is outside selected attribute or not confusable")
            with torch.no_grad():
                modified_features = gallery_features.clone()
                modified_features[paired:paired + 1] = source.encode_images(result.state.image)
                final_score = source.encode_texts([result.state.text]) @ modified_features.T
            rank = int(exact_first_hit_rank(final_score, [person_id], gallery_ids)[0])
            arms[arm] = {
                "rank": rank, "rank_delta": rank - base_rank,
                "soft_rank": float(soft_first_hit_rank(final_score, [person_id],
                                                       gallery_ids, tau)[0]),
                "linf": float((result.state.image - original).abs().max()),
                "text": result.state.text,
                "rounds": [item.slot for item in result.completed],
                "text_events": events,
            }
        results.append({
            "row_id": record["row_id"], "image": record["image"],
            "caption": record["caption"], "person_id": person_id,
            "selected_slots": record["selected_slots"], "clean_rank": base_rank,
            "clean_soft_rank": float(soft_first_hit_rank(
                clean_scores[query_index:query_index + 1], [person_id], gallery_ids, tau)[0]),
            "arms": arms,
        })
        print(json.dumps({"row_id": record["row_id"], "clean_rank": base_rank,
                          "ranks": {key: value["rank"] for key, value in arms.items()}},
                         ensure_ascii=False), flush=True)
    summary = {arm: {
        "mean_rank": sum(item["arms"][arm]["rank"] for item in results) / len(results),
        "mean_rank_delta": sum(item["arms"][arm]["rank_delta"]
                               for item in results) / len(results),
        "mean_soft_rank": sum(item["arms"][arm]["soft_rank"]
                              for item in results) / len(results),
    } for arm in ("tta_only", "text_only", "tta_text")}
    report = {
        "mode": "development_paired_image_poc",
        "dataset": "CUHK-PEDES train", "source": "Frozen CLIP ViT-B/16",
        "seed": args.seed, "tta_steps": args.steps,
        "tta_transforms_per_scale": args.transforms_per_scale,
        "tta_scales": scales,
        "gallery_images": len(paths), "queries": len(results), "tau": tau,
        "summary": summary, "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")


if __name__ == "__main__":
    main()
