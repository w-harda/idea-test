#!/usr/bin/env python3
"""固定三条 CUHK query 的 Vanilla TTA 官方完整图文攻击 PoC。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attributes.frozen_clip import FrozenCLIP
from attributes.rank_objective import calibrate_tau, exact_first_hit_rank, soft_first_hit_rank
from attributes.tta_full import FullVanillaTTA
from run_vanilla_tta_poc import (
    EXPECTED_GALLERY_SHA256, SEED, raw_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta-root", required=True)
    parser.add_argument("--bert", required=True)
    parser.add_argument("--glove", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries, paths, gallery_ids = raw_sample(args.annotation)
    gallery_sha = hashlib.sha256(json.dumps(
        list(zip(paths, gallery_ids, strict=True)), ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    if gallery_sha != EXPECTED_GALLERY_SHA256:
        raise ValueError("gallery differs from the existing 29-image PoC")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = FrozenCLIP(args.checkpoint, device=device).to(device)
    preprocess = transforms.Compose([
        transforms.Resize(source.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(source.resolution),
        transforms.ToTensor(),
    ])
    root = Path(args.image_root).resolve()
    pixels = []
    for path in paths:
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("gallery path escapes image root")
        with Image.open(resolved) as image:
            pixels.append(preprocess(image.convert("RGB")))
    gallery_images = torch.stack(pixels).to(device)
    with torch.no_grad():
        gallery_features = source.encode_images(gallery_images)
        query_features = source.encode_texts([caption for _, caption, _, _ in queries])
        clean_scores = query_features @ gallery_features.T
    query_ids = [person_id for _, _, _, person_id in queries]
    tau = calibrate_tau(clean_scores, query_ids, gallery_ids)
    attack = FullVanillaTTA(source, args.tta_root, args.bert, args.glove)
    results = []
    for index, (row_id, caption, image, person_id) in enumerate(queries):
        torch.manual_seed(SEED + index)
        np.random.seed(SEED + index)
        paired = paths.index(image)
        original = gallery_images[paired:paired + 1]
        attacked, attacked_text = attack(original, caption)
        linf = float((attacked - original).abs().max())
        if linf > 8 / 255 + 1e-6:
            raise AssertionError("full vanilla TTA exceeded cumulative 8/255")
        with torch.no_grad():
            attacked_features = gallery_features.clone()
            attacked_features[paired:paired + 1] = source.encode_images(attacked)
            adv_scores = source.encode_texts([attacked_text]) @ attacked_features.T
        clean_rank = int(exact_first_hit_rank(
            clean_scores[index:index + 1], [person_id], gallery_ids)[0])
        rank = int(exact_first_hit_rank(adv_scores, [person_id], gallery_ids)[0])
        results.append({
            "row_id": row_id, "image": image, "caption": caption,
            "person_id": person_id, "clean_rank": clean_rank,
            "clean_soft_rank": float(soft_first_hit_rank(
                clean_scores[index:index + 1], [person_id], gallery_ids, tau)[0]),
            "full_rank": rank, "full_rank_delta": rank - clean_rank,
            "full_soft_rank": float(soft_first_hit_rank(
                adv_scores, [person_id], gallery_ids, tau)[0]),
            "attacked_text": attacked_text, "text_changed": attacked_text != caption,
            "linf": linf,
        })
        print(json.dumps({"row_id": row_id, "clean_rank": clean_rank,
                          "full_rank": rank, "text_changed": attacked_text != caption},
                         ensure_ascii=False), flush=True)
    report = {
        "mode": "development_paired_image_vanilla_tta_full",
        "dataset": "CUHK-PEDES train", "source": "Frozen CLIP ViT-B/16",
        "seed": SEED, "gallery_images": len(paths),
        "gallery_sha256": gallery_sha, "tau": tau,
        "tta_flow": "official Image_1 -> Text_1 -> Image_2",
        "tta_steps_per_image_pass": 10, "tta_text_steps": 1,
        "tta_transforms_per_scale": 6,
        "tta_scales": [0.5, 0.75, 1.25, 1.5],
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
