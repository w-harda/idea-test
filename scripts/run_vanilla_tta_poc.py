#!/usr/bin/env python3
"""固定三条 CUHK query 的图像侧 Vanilla TTA；不读取 Stage 04/05/M。"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attributes.frozen_clip import FrozenCLIP
from attributes.rank_objective import calibrate_tau, exact_first_hit_rank, soft_first_hit_rank
from attributes.tta_adapter import vanilla_tta_image_attack

ROW_IDS = ("cuhk:5:1", "cuhk:11:0", "cuhk:20:0")
SEED = 20260926
NEGATIVE_IMAGES = 20
EXPECTED_GALLERY_SHA256 = "e1505afef38cb62aa5e22dacf9d60a4e62871243763669e4a50e0a492fd88788"


def raw_sample(annotation_path: str | Path):
    """仅从 CUHK 原始标注重建同一批 query 与图库顺序。"""
    with Path(annotation_path).open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("CUHK annotation must be a JSON array")
    gallery: dict[str, int | str] = {}
    queries = {}
    for row_index, row in enumerate(rows):
        if row["split"] != "train":
            continue
        image, person_id = row["file_path"], row["id"]
        if image in gallery and gallery[image] != person_id:
            raise ValueError("conflicting ID for gallery image")
        gallery[image] = person_id
        for caption_index, caption in enumerate(row["captions"]):
            row_id = f"cuhk:{row_index}:{caption_index}"
            if row_id in ROW_IDS:
                queries[row_id] = (caption, image, person_id)
    if set(queries) != set(ROW_IDS):
        raise ValueError("fixed three query IDs missing from raw train annotation")
    ordered = [(row_id, *queries[row_id]) for row_id in ROW_IDS]
    positive_ids = {person_id for _, _, _, person_id in ordered}
    positive_paths = [path for path, person_id in gallery.items()
                      if person_id in positive_ids]
    negatives = [(path, person_id) for path, person_id in gallery.items()
                 if person_id not in positive_ids]
    random.Random(SEED).shuffle(negatives)
    paths = positive_paths + [path for path, _ in negatives[:NEGATIVE_IMAGES]]
    if len(paths) != 29:
        raise ValueError("fixed PoC gallery must contain exactly 29 images")
    return ordered, paths, [gallery[path] for path in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    queries, paths, gallery_ids = raw_sample(args.annotation)
    gallery_fingerprint = hashlib.sha256(json.dumps(
        list(zip(paths, gallery_ids, strict=True)), ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()
    if gallery_fingerprint != EXPECTED_GALLERY_SHA256:
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
    results = []
    for index, (row_id, caption, image, person_id) in enumerate(queries):
        torch.manual_seed(SEED + index)
        np.random.seed(SEED + index)
        paired = paths.index(image)
        original = gallery_images[paired:paired + 1]
        attacked = vanilla_tta_image_attack(source, args.tta_root, original, caption)
        linf = float((attacked - original).abs().max())
        if linf > 8 / 255 + 1e-6:
            raise AssertionError("vanilla TTA exceeded cumulative 8/255")
        with torch.no_grad():
            attacked_features = gallery_features.clone()
            attacked_features[paired:paired + 1] = source.encode_images(attacked)
            adv_scores = source.encode_texts([caption]) @ attacked_features.T
        clean_rank = int(exact_first_hit_rank(
            clean_scores[index:index + 1], [person_id], gallery_ids)[0])
        rank = int(exact_first_hit_rank(adv_scores, [person_id], gallery_ids)[0])
        results.append({
            "row_id": row_id, "image": image, "caption": caption,
            "person_id": person_id, "clean_rank": clean_rank,
            "clean_soft_rank": float(soft_first_hit_rank(
                clean_scores[index:index + 1], [person_id], gallery_ids, tau)[0]),
            "vanilla_rank": rank, "vanilla_rank_delta": rank - clean_rank,
            "vanilla_soft_rank": float(soft_first_hit_rank(
                adv_scores, [person_id], gallery_ids, tau)[0]),
            "linf": linf,
        })
        print(json.dumps({"row_id": row_id, "clean_rank": clean_rank,
                          "vanilla_rank": rank}, ensure_ascii=False), flush=True)
    report = {
        "mode": "development_paired_image_vanilla_tta",
        "source": "Frozen CLIP ViT-B/16", "dataset": "CUHK-PEDES train",
        "seed": SEED, "gallery_images": len(paths), "gallery_sha256": gallery_fingerprint,
        "gallery_paths": paths, "tau": tau,
        "tta_image_passes": 2, "tta_steps_per_pass": 10,
        "tta_transforms_per_scale": 6, "tta_scales": [0.5, 0.75, 1.25, 1.5],
        "text_attack": "disabled", "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
