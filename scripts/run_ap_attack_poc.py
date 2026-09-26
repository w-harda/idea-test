#!/usr/bin/env python3
"""固定三条 CUHK query 的 AP-Attack full / Stage 05 属性调度 PoC。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attributes.ap_attack_adapter import (
    AP_SEMANTIC_CHECKPOINT_SHA256, APAttackGenerator, APImageCallback,
    map_native_delta_to_clip,
)
from attributes.frozen_clip import FrozenCLIP
from attributes.rank_objective import calibrate_tau, exact_first_hit_rank, soft_first_hit_rank
from run_vanilla_tta_poc import EXPECTED_GALLERY_SHA256, SEED, raw_sample


def load_selected_records(stage04_path: str | Path, row_ids: set[str]) -> dict:
    selected = {}
    with Path(stage04_path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["row_id"] in row_ids:
                if record["row_id"] in selected:
                    raise ValueError("duplicate Stage 04 query")
                selected[record["row_id"]] = record
    if set(selected) != row_ids:
        raise ValueError("Stage 04 records missing for fixed query set")
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "guided"), required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--stage04")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ap-root", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "guided" and not args.stage04:
        parser.error("--stage04 is required for guided mode")
    if args.mode == "full" and args.stage04:
        parser.error("--stage04 is forbidden for full mode")

    queries, paths, gallery_ids = raw_sample(args.annotation)
    gallery_sha = hashlib.sha256(json.dumps(
        list(zip(paths, gallery_ids, strict=True)), ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    if gallery_sha != EXPECTED_GALLERY_SHA256:
        raise ValueError("gallery differs from the existing 29-image PoC")
    records = None
    if args.mode == "guided":
        records = load_selected_records(
            args.stage04, {row_id for row_id, *_ in queries})
        for row_id, caption, image, person_id in queries:
            record = records[row_id]
            if record["caption"] != caption or record["image"] != image:
                raise ValueError("Stage 04 query differs from raw annotation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = FrozenCLIP(args.checkpoint, device=device).to(device)
    clip_prep = transforms.Compose([
        transforms.Resize(source.resolution,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(source.resolution),
        transforms.ToTensor(),
    ])
    native_prep = transforms.Compose([
        transforms.Resize((256, 128),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    root = Path(args.image_root).resolve()
    gallery_clip, gallery_native, raw_sizes = [], [], []
    for path in paths:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("gallery path escapes image root")
        with Image.open(resolved) as handle:
            rgb = handle.convert("RGB")
            raw_sizes.append((rgb.height, rgb.width))
            gallery_clip.append(clip_prep(rgb))
            gallery_native.append(native_prep(rgb))
    gallery_images = torch.stack(gallery_clip).to(device)
    with torch.no_grad():
        gallery_features = source.encode_images(gallery_images)
        query_features = source.encode_texts([caption for _, caption, _, _ in queries])
        clean_scores = query_features @ gallery_features.T
    query_ids = [person_id for _, _, _, person_id in queries]
    tau = calibrate_tau(clean_scores, query_ids, gallery_ids)
    attack = APAttackGenerator(args.ap_root, args.generator, device)
    results = []

    for index, (row_id, caption, image, person_id) in enumerate(queries):
        torch.manual_seed(SEED + index)
        np.random.seed(SEED + index)
        paired = paths.index(image)
        native_original = gallery_native[paired].unsqueeze(0).to(device)
        clean_clip = gallery_images[paired:paired + 1]
        raw_size = raw_sizes[paired]
        clean_rank = int(exact_first_hit_rank(
            clean_scores[index:index + 1], [person_id], gallery_ids)[0])
        clean_soft = float(soft_first_hit_rank(
            clean_scores[index:index + 1], [person_id], gallery_ids, tau)[0])

        def evaluate(attacked_image, text):
            if (attacked_image - clean_clip).abs().max() > 8 / 255 + 1e-6:
                raise AssertionError("AP-Attack exceeded CLIP image budget")
            with torch.no_grad():
                modified = gallery_features.clone()
                modified[paired:paired + 1] = source.encode_images(attacked_image)
                scores = source.encode_texts([text]) @ modified.T
            return (
                int(exact_first_hit_rank(scores, [person_id], gallery_ids)[0]),
                float(soft_first_hit_rank(scores, [person_id], gallery_ids, tau)[0]),
            )

        result = {
            "row_id": row_id, "image": image, "caption": caption,
            "person_id": person_id, "clean_rank": clean_rank,
            "clean_soft_rank": clean_soft,
        }
        if args.mode == "full":
            native_attacked = attack.attack_native(native_original, native_original)
            attacked = map_native_delta_to_clip(
                native_original, native_attacked, clean_clip,
                raw_size, source.resolution)
            rank, soft_rank = evaluate(attacked, caption)
            result.update({
                "full_rank": rank, "full_soft_rank": soft_rank,
                "full_rank_delta": rank - clean_rank,
                "linf_native": float((native_attacked - native_original).abs().max()),
                "linf_clip": float((attacked - clean_clip).abs().max()),
                "generator_calls": 1,
                "text": caption,
            })
            summary = {"row_id": row_id, "clean_rank": clean_rank, "full_rank": rank}
        else:
            from attributes.attack_scheduler import TextAttackResult, run_attack
            from attributes.confusable_attack import CONFUSABLES, ConfusableTextCallback

            record = records[row_id]
            arms = {}
            for arm in ("ap_only", "ap_text"):
                callback = APImageCallback(
                    attack, native_original, clean_clip, raw_size, source.resolution)

                def candidate_scores(current_image, texts):
                    with torch.no_grad():
                        modified = gallery_features.clone()
                        modified[paired:paired + 1] = source.encode_images(current_image)
                        scores = source.encode_texts(texts) @ modified.T
                        return soft_first_hit_rank(
                            scores, [person_id] * len(texts), gallery_ids, tau)

                if arm == "ap_text":
                    text_callback = ConfusableTextCallback(candidate_scores)
                else:
                    text_callback = lambda request: TextAttackResult(request.state.text)
                state = run_attack(
                    record, image=clean_clip, image_attack=callback,
                    text_attack=text_callback)
                events = getattr(text_callback, "events", [])
                edits = {event["offset"]: event for event in events if event["selected"]}
                differences = {offset for offset, (before, after) in enumerate(
                    zip(caption, state.state.text)) if before != after}
                if (len(state.state.text) != len(caption)
                        or differences != set(edits)
                        or callback.calls != record["k_star"]
                        or (state.state.image - clean_clip).abs().max() > 8 / 255 + 1e-6):
                    raise AssertionError("guided AP-Attack violated image/text budget")
                for offset, event in edits.items():
                    mentions = record["provenance"][event["slot"]]["mentions"]
                    if (not any(mention["start"] <= offset < mention["end"]
                                for mention in mentions)
                            or state.state.text[offset] not in
                            CONFUSABLES.get(caption[offset], ())):
                        raise AssertionError("AP text edit outside selected attribute")
                rank, soft_rank = evaluate(state.state.image, state.state.text)
                arms[arm] = {
                    "rank": rank, "soft_rank": soft_rank,
                    "rank_delta": rank - clean_rank,
                    "linf_native": float(
                        (callback.native_current - native_original).abs().max()),
                    "linf_clip": float((state.state.image - clean_clip).abs().max()),
                    "generator_calls": callback.calls,
                    "rounds": [item.slot for item in state.completed],
                    "text": state.state.text, "text_events": events,
                }
            result.update({
                "selected_slots": record["selected_slots"],
                "arms": arms,
            })
            summary = {"row_id": row_id, "clean_rank": clean_rank,
                       "ranks": {name: item["rank"] for name, item in arms.items()}}
        results.append(result)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    report = {
        "mode": "development_paired_image_ap_attack_" + args.mode,
        "dataset": "CUHK-PEDES train", "source": "Frozen CLIP ViT-B/16",
        "ap_training": "DukeMTMC-reID stage2 ReID+semantic 10/10 reproduction",
        "ap_checkpoint_sha256": AP_SEMANTIC_CHECKPOINT_SHA256,
        "ap_native_resolution": [256, 128],
        "attribute_conditioned_inference": False,
        "seed": SEED, "queries": 3, "gallery_images": len(paths),
        "gallery_sha256": gallery_sha, "tau": tau, "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
