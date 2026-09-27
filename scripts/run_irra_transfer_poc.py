#!/usr/bin/env python3
"""Twenty-query CLIP-to-IRRA transfer diagnostic on the official CUHK test gallery."""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from attributes.attack_scheduler import ImageAttackResult, TextAttackResult, run_attack
from attributes.confusable_attack import ConfusableTextCallback
from attributes.frozen_clip import FrozenCLIP
from attributes.generator_framework import IMAGE_EPSILON
from attributes.irra_victim import FrozenIRRA, IMAGE_SIZE, file_sha256
from attributes.random_k import apply_random_k, random_k_entry
from attributes.rank_objective import soft_first_hit_rank
from attributes.tta_adapter import TTAImageCallback, vanilla_tta_image_attack
from attributes.tta_full import FullVanillaTTA
from scripts.run_cuhk_test_full import (
    ARMS, GUIDED_ARMS, IMAGE_ARMS, RANDOM_ARMS, SEED, TAU,
    _rank, _text_events_are_valid, cache_metadata, load_or_build_features,
    load_stage04_test, load_test_protocol, preprocess,
)

ROOT = Path("/home/lzf/ldx")
DEFAULT_OUTPUT = ROOT / "outputs/idea-TBPS-test1/irra-victim-poc"
DEFAULT_STAGE1 = ROOT / "outputs/idea-TBPS-test1/cuhk-test-stage1"
DEFAULT_REPO = Path("/home/lzf/Attack/IRRA-main/IRRA-main")
DEFAULT_CHECKPOINT = Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth")
DEFAULT_CONFIG = ROOT / "projects/idea-TBPS-test1/configs/irra_cuhk_official.yaml"
DEFAULT_ANNOTATION = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
DEFAULT_IMAGE_ROOT = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")
DEFAULT_STAGE04 = ROOT / "outputs/idea-TBPS-test1/stage04/cuhk_all_topk.jsonl"
DEFAULT_SOURCE = ROOT / "cache/clip/ViT-B-16.pt"
DEFAULT_TTA = ROOT / "external/Transform_to_Transfer_Attack"
DEFAULT_BERT = ROOT / "cache/tta/bert-base-uncased"
DEFAULT_GLOVE = ROOT / "cache/tta/glove-wiki-gigaword-300/glove-wiki-gigaword-300.model"
FIRST20 = 20


def checked_paths(args):
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("IRRA experiment output must stay under /home/lzf/ldx")
    queries, paths, ids = load_test_protocol(args.annotation.resolve())
    manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
    expected = [query.row_id for query in queries[:FIRST20]]
    actual = [row["row_id"] for row in manifest["queries"][:FIRST20]]
    if expected != actual or any(not row["from_first20"] for row in manifest["queries"][:FIRST20]):
        raise ValueError("first 20 query IDs differ from the existing common manifest")
    return output, queries, paths, ids


def _save_array(path: Path, value: np.ndarray):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    temporary.replace(path)


def victim_clean(args, output, queries, paths, ids):
    if not torch.cuda.is_available():
        raise RuntimeError("IRRA victim requires CUDA")
    device = torch.device("cuda")
    victim = FrozenIRRA(args.irra_repo, args.irra_checkpoint, args.irra_config, device="cuda")
    cache = output / "cache-official"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_sha256": file_sha256(args.irra_checkpoint),
        "config_sha256": file_sha256(args.irra_config),
        "annotation_sha256": file_sha256(args.annotation),
        "query_manifest_sha256": file_sha256(args.query_manifest),
        "gallery_count": len(paths), "query_count": len(queries),
        "image_size": list(IMAGE_SIZE), "text_length": 77,
        "repo": str(args.irra_repo.resolve()),
    }
    meta_path = cache / "metadata.json"
    g_path = cache / "gallery.npy"
    q_path = cache / "queries.npy"
    if meta_path.exists() and g_path.exists() and q_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("IRRA cache metadata differs")
        gallery = np.load(g_path, allow_pickle=False)
        query = np.load(q_path, allow_pickle=False)
        if gallery.shape != (3074, 512) or query.shape != (6156, 512):
            raise ValueError("IRRA feature cache shape differs")
        g = torch.from_numpy(gallery.copy()).to(device)
        q = torch.from_numpy(query.copy()).to(device)
    else:
        images = []
        for start in range(0, len(paths), 64):
            pixels = []
            for relative in paths[start:start + 64]:
                path = (args.image_root / relative).resolve()
                if not path.is_relative_to(args.image_root.resolve()):
                    raise ValueError("gallery image escapes root")
                with Image.open(path) as handle:
                    pixels.append(victim.pixels(handle))
            with torch.no_grad():
                images.append(victim.encode_images(torch.stack(pixels).to(device)).cpu())
            if start % 512 == 0:
                print(json.dumps({"irra_gallery_done": min(start + 64, len(paths))}), flush=True)
        gallery = torch.cat(images).numpy()
        texts = []
        for start in range(0, len(queries), 128):
            with torch.no_grad():
                texts.append(victim.encode_texts([
                    query.caption for query in queries[start:start + 128]
                ]).cpu())
        query = torch.cat(texts).numpy()
        _save_array(g_path, gallery)
        _save_array(q_path, query)
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        g = torch.from_numpy(gallery.copy()).to(device)
        q = torch.from_numpy(query.copy()).to(device)

    scores = q @ g.T
    order = scores.argsort(dim=1, descending=True)
    gallery_ids = torch.tensor(ids, device=device)
    query_ids = torch.tensor([item.person_id for item in queries], device=device)
    hits = gallery_ids[order].eq(query_ids[:, None])
    if not bool(hits.any(dim=1).all()):
        raise AssertionError("some test IDs are missing from gallery")
    ranks = hits.to(torch.int32).argmax(dim=1) + 1
    report = {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "config_sha256": metadata["config_sha256"],
        "test_queries": len(queries), "gallery_images": len(paths),
        "r1": float((ranks <= 1).float().mean()),
        "r5": float((ranks <= 5).float().mean()),
        "r10": float((ranks <= 10).float().mean()),
        "mean_first_hit_rank": float(ranks.float().mean()),
    }
    if not 0.70 <= report["r1"] <= 0.77:
        raise RuntimeError(f"IRRA clean validation far from reported 73.38%: {report}")
    report_path = output / "official_clean_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return victim, g, q, report


def first20_random_entries(args, queries, records):
    entries = {row["row_id"]: row for row in map(
        json.loads, args.random_manifest.open(encoding="utf-8"))}
    for query in queries[:FIRST20]:
        if entries.get(query.row_id) != random_k_entry(records[query.row_id], 42):
            raise ValueError(f"Random-K manifest differs: {query.row_id}")
    return entries


def _mapped_victim_image(victim, clean_pixels, source_clean, source_adv):
    delta = (source_adv - source_clean).detach()
    if float(delta.abs().max()) > 8 / 255 + 1e-6:
        raise AssertionError("source TTA image exceeds 8/255")
    mapped = F.interpolate(delta, size=IMAGE_SIZE, mode="bilinear",
                           align_corners=False).clamp(-IMAGE_EPSILON, IMAGE_EPSILON)
    adv = (clean_pixels + mapped).clamp(0, 1)
    linf = float((adv - clean_pixels).abs().max())
    if linf > 8 / 255 + 1e-6:
        raise AssertionError("IRRA-view image exceeds 8/255")
    return adv, linf


def run_one_arm(args, output, queries, paths, ids):
    if args.arm not in ARMS:
        raise ValueError("unknown arm")
    victim, victim_gallery, victim_queries, _ = victim_clean(args, output, queries, paths, ids)
    gallery_indices = {path: index for index, path in enumerate(paths)}
    records = load_stage04_test(args.stage04, queries) if args.arm in GUIDED_ARMS else None
    random_entries = (first20_random_entries(args, queries, records)
                      if args.arm in RANDOM_ARMS else None)
    source = None
    source_gallery = None
    full_attack = None
    prep = None
    if args.arm != "clean":
        source = FrozenCLIP(args.source_checkpoint, device="cuda").to("cuda")
        if not next(source.model.parameters()).is_cuda:
            raise RuntimeError("CLIP source is not on CUDA")
        meta = cache_metadata(args.annotation, args.source_checkpoint,
                              queries, paths, ids)
        source_gallery, _ = load_or_build_features(
            args.source_cache, meta, source, queries, paths, args.image_root,
            torch.device("cuda"))
        prep = preprocess(source.resolution)
        if args.arm == "vanilla_full":
            full_attack = FullVanillaTTA(source, args.tta_root, args.bert, args.glove)

    arm_dir = output / "arms" / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    result_path = arm_dir / "results.jsonl"
    previous = {}
    if result_path.exists():
        for line in result_path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["arm"] != args.arm or row["row_id"] in previous:
                raise ValueError("invalid prior IRRA arm row")
            previous[row["row_id"]] = row
    expected = {q.row_id for q in queries[:FIRST20]}
    if set(previous) - expected:
        raise ValueError("IRRA result contains row outside first 20")
    completed = len(previous)
    started = time.monotonic()
    with result_path.open("a", encoding="utf-8") as handle:
        for index, query in enumerate(queries[:FIRST20]):
            if query.row_id in previous:
                continue
            try:
                torch.manual_seed(SEED + index)
                np.random.seed(SEED + index)
                random.seed(SEED + index)
                paired = gallery_indices[query.image]
                clean_scores = victim_queries[index] @ victim_gallery.T
                clean_rank = _rank(clean_scores, query.person_id, ids)
                attacked_text = query.caption
                events, rounds = [], []
                source_clean = None
                source_adv = None
                if args.arm != "clean":
                    image_path = (args.image_root / query.image).resolve()
                    if not image_path.is_relative_to(args.image_root.resolve()):
                        raise ValueError("query image escapes root")
                    with Image.open(image_path) as image:
                        source_clean = prep(image.convert("RGB")).unsqueeze(0).to("cuda")
                        victim_clean_pixels = victim.pixels(image).unsqueeze(0).to("cuda")
                    source_adv = source_clean
                    if args.arm == "vanilla_image":
                        source_adv = vanilla_tta_image_attack(
                            source, args.tta_root, source_clean, query.caption)
                    elif args.arm == "vanilla_full":
                        source_adv, attacked_text = full_attack(source_clean, query.caption)
                    elif args.arm in GUIDED_ARMS:
                        record = records[query.row_id]
                        if args.arm in RANDOM_ARMS:
                            record = apply_random_k(
                                record, random_entries[query.row_id], 42)
                        if args.arm == "text_only" or record["k_star"] == 0:
                            image_callback = lambda request: ImageAttackResult(request.state.image)
                        else:
                            image_callback = TTAImageCallback(
                                source, args.tta_root, source_clean)
                        def candidate_scores(current_image, texts):
                            with torch.no_grad():
                                if args.arm == "text_only":
                                    gallery = source_gallery
                                else:
                                    gallery = source_gallery.clone()
                                    gallery[paired:paired + 1] = source.encode_images(current_image)
                                values = source.encode_texts(texts) @ gallery.T
                                return soft_first_hit_rank(
                                    values, [query.person_id] * len(texts), ids, TAU)
                        if args.arm in {"attribute_tta_only", "random_k_tta_only"}:
                            text_callback = lambda request: TextAttackResult(request.state.text)
                        else:
                            text_callback = ConfusableTextCallback(candidate_scores)
                        result = run_attack(record, image=source_clean,
                                            image_attack=image_callback,
                                            text_attack=text_callback)
                        source_adv = result.state.image
                        attacked_text = result.state.text
                        events = getattr(text_callback, "events", [])
                        rounds = [item.slot for item in result.completed]
                        if rounds != record["selected_slots"]:
                            raise AssertionError("Stage 05 attribute order changed")
                        _text_events_are_valid(record, attacked_text, events)

                linf = 0.0
                source_linf = 0.0
                if args.arm == "clean":
                    rank = clean_rank
                else:
                    source_linf = float((source_adv - source_clean).abs().max())
                    with torch.no_grad():
                        gallery = victim_gallery
                        if args.arm in IMAGE_ARMS and source_linf > 0:
                            adv_pixels, linf = _mapped_victim_image(
                                victim, victim_clean_pixels, source_clean, source_adv)
                            gallery = victim_gallery.clone()
                            gallery[paired:paired + 1] = victim.encode_images(adv_pixels)
                        query_feature = victim.encode_texts([attacked_text])[0]
                        rank = _rank(query_feature @ gallery.T, query.person_id, ids)
                row = {
                    "row_id": query.row_id, "arm": args.arm,
                    "person_id": query.person_id, "image": query.image,
                    "clean_rank": clean_rank, "rank": rank,
                    "text": attacked_text, "text_changed": attacked_text != query.caption,
                    "linf": linf, "source_linf": source_linf,
                    "rounds": rounds, "text_events": events,
                    "victim": "official_irra_cuhk",
                }
                if args.arm in RANDOM_ARMS:
                    row["random_attack_order"] = rounds
                    row["random_k_manifest_sha256"] = file_sha256(args.random_manifest)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                print(json.dumps({"arm": args.arm, "done": completed, "rank": rank,
                                  "clean_rank": clean_rank,
                                  "elapsed_sec": round(time.monotonic() - started, 1)}),
                      flush=True)
            except Exception as exc:
                error_path = arm_dir / "errors.jsonl"
                with error_path.open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({
                        "row_id": query.row_id, "arm": args.arm,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                    }) + "\n")
                raise


def summarize(output, queries):
    by_arm = {}
    ids = [query.row_id for query in queries[:FIRST20]]
    for arm in ARMS:
        path = output / "arms" / arm / "results.jsonl"
        rows = {row["row_id"]: row for row in map(
            json.loads, path.open(encoding="utf-8"))}
        if set(rows) != set(ids):
            raise ValueError(f"{arm} does not cover exactly the first 20")
        by_arm[arm] = [rows[row_id] for row_id in ids]
    clean = np.array([r["rank"] for r in by_arm["clean"]])
    metrics = {}
    for arm, rows in by_arm.items():
        ranks = np.array([r["rank"] for r in rows])
        refs = np.array([r["clean_rank"] for r in rows])
        if not np.array_equal(refs, clean):
            raise ValueError(f"{arm} clean ranks differ")
        delta = ranks - clean
        metrics[arm] = {
            "query_count": FIRST20,
            "rank1": float((ranks <= 1).mean()),
            "rank5": float((ranks <= 5).mean()),
            "rank10": float((ranks <= 10).mean()),
            "mean_first_hit_rank": float(ranks.mean()),
            "median_first_hit_rank": float(np.median(ranks)),
            "mean_rank_delta": float(delta.mean()),
            "median_rank_delta": float(np.median(delta)),
            "rank_delta_positive_count": int((delta > 0).sum()),
            "rank_delta_zero_count": int((delta == 0).sum()),
            "rank_delta_negative_count": int((delta < 0).sum()),
            "rank_delta_positive": float((delta > 0).mean()),
            "rank_delta_zero": float((delta == 0).mean()),
            "rank_delta_negative": float((delta < 0).mean()),
            "text_changed": sum(row["text_changed"] for row in rows),
            "mean_linf": float(np.mean([row["linf"] for row in rows])),
            "max_linf": max(row["linf"] for row in rows),
        }
    result = {
        "victim": "official_irra_cuhk",
        "attack_source": "frozen_clip",
        "protocol": "reid_raw.json test, 6156 queries, 3074 gallery; first 20 only",
        "query_row_ids": ids,
        "arms": metrics,
        "per_query": [
            {"row_id": row_id, **{arm: by_arm[arm][i]["rank"] for arm in ARMS}}
            for i, row_id in enumerate(ids)
        ],
    }
    path = output / "summary_20_irra_8arms.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-clean", "run", "summarize"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--irra-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--irra-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--irra-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--query-manifest", type=Path,
                        default=DEFAULT_STAGE1 / "sample500_seed42.json")
    parser.add_argument("--random-manifest", type=Path,
                        default=DEFAULT_STAGE1 / "random_k_seed42.jsonl")
    parser.add_argument("--stage04", type=Path, default=DEFAULT_STAGE04)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_STAGE1 / "cache")
    parser.add_argument("--tta-root", type=Path, default=DEFAULT_TTA)
    parser.add_argument("--bert", type=Path, default=DEFAULT_BERT)
    parser.add_argument("--glove", type=Path, default=DEFAULT_GLOVE)
    args = parser.parse_args()
    output, queries, paths, ids = checked_paths(args)
    if args.command == "summarize":
        summarize(output, queries)
    elif args.command == "validate-clean":
        _, _, _, report = victim_clean(args, output, queries, paths, ids)
        print(json.dumps(report), flush=True)
    else:
        if args.arm is None:
            parser.error("run requires --arm")
        run_one_arm(args, output, queries, paths, ids)


if __name__ == "__main__":
    main()
