#!/usr/bin/env python3
"""Manifest-scoped official IRRA white-box attack entry; attacks run only on request."""
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

from attributes.attack_scheduler import ImageAttackResult, TextAttackResult, run_attack
from attributes.confusable_attack import ConfusableTextCallback
from attributes.generator_framework import IMAGE_EPSILON
from attributes.irra_surrogate import IRRASurrogate
from attributes.irra_victim import FrozenIRRA, IMAGE_SIZE, file_sha256
from attributes.random_k import apply_random_k
from attributes.rank_objective import calibrate_tau
from attributes.tta_irra import (
    IRRAFullVanillaTTA, IRRATTAImageCallback, vanilla_tta_irra_image_attack,
)
from scripts.run_cuhk_test_full import (
    ARMS, GUIDED_ARMS, IMAGE_ARMS, RANDOM_ARMS, SEED,
    _rank, _text_events_are_valid, load_stage04_test, load_test_protocol,
    load_query_manifest, load_random_k_manifest,
)
from scripts.run_irra_transfer_poc import (
    DEFAULT_ANNOTATION, DEFAULT_BERT, DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG, DEFAULT_GLOVE, DEFAULT_IMAGE_ROOT, DEFAULT_REPO,
    DEFAULT_STAGE04, DEFAULT_STAGE1, DEFAULT_TTA, victim_clean,
)

ROOT = Path("/home/lzf/ldx")
DEFAULT_OUTPUT = ROOT / "outputs/idea-TBPS-test1/irra-surrogate-500"
TRANSFER_OUTPUT = ROOT / "outputs/idea-TBPS-test1/irra-victim-poc"
OLD20_OUTPUT = ROOT / "outputs/idea-TBPS-test1/irra-surrogate-poc"
SOURCE_LABEL = "official_irra_cuhk_white_box"
TAU_CALIBRATION_COUNT = 20  # Preserve the existing IRRA source objective.


def _output_guard(output: Path) -> None:
    if not output.is_relative_to(ROOT):
        raise ValueError("IRRA experiment output must stay under /home/lzf/ldx")
    if output.is_relative_to(TRANSFER_OUTPUT) or output.is_relative_to(OLD20_OUTPUT):
        raise ValueError("IRRA surrogate output cannot overwrite previous results")


def protocol_paths(args):
    output = args.output_dir.resolve()
    _output_guard(output)
    annotation = args.annotation.resolve()
    queries, paths, ids = load_test_protocol(annotation)
    selected = load_query_manifest(args.query_manifest.resolve(), annotation, queries)
    return output, queries, paths, ids, selected


def _source_tau(query_features: torch.Tensor, gallery_features: torch.Tensor,
                queries, gallery_ids) -> float:
    """Calibrate the existing soft-rank rule using IRRA clean scores only."""
    with torch.no_grad():
        scores = query_features[:TAU_CALIBRATION_COUNT] @ gallery_features.T
        return calibrate_tau(
            scores, [item.person_id for item in queries[:TAU_CALIBRATION_COUNT]],
            gallery_ids)


def _prior_rows(path: Path, arm: str, expected_ids: set[str],
                checkpoint_sha256: str, config_sha256: str,
                manifest_sha256: str):
    prior = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if (row.get("arm") != arm or row.get("row_id") in prior
                    or row.get("attack_source") != SOURCE_LABEL
                    or row.get("checkpoint_sha256") != checkpoint_sha256
                    or row.get("config_sha256") != config_sha256
                    or row.get("query_manifest_sha256") != manifest_sha256):
                raise ValueError("IRRA surrogate result metadata differs")
            prior[row["row_id"]] = row
    if set(prior) - expected_ids:
        raise ValueError("IRRA surrogate result contains query outside manifest")
    return prior


def run_one_arm(args, output, queries, paths, ids, selected, manifest_sha256):
    if args.arm not in ARMS:
        raise ValueError("unknown arm")
    victim, gallery_features, query_features, validation = victim_clean(
        args, output, queries, paths, ids)
    source = IRRASurrogate(victim)
    if not next(source.victim.model.parameters()).is_cuda:
        raise RuntimeError("IRRA attack source is not on CUDA")
    tau = _source_tau(query_features, gallery_features, queries, ids)
    gallery_indices = {path: index for index, path in enumerate(paths)}
    records = load_stage04_test(args.stage04, queries) if args.arm in GUIDED_ARMS else None
    random_entries = (
        load_random_k_manifest(
            args.random_manifest, queries, selected, records, 42)
        if args.arm in RANDOM_ARMS else None
    )
    random_sha = file_sha256(args.random_manifest) if args.arm in RANDOM_ARMS else None
    full_attack = (
        IRRAFullVanillaTTA(source, args.tta_root, args.bert, args.glove)
        if args.arm == "vanilla_full" else None
    )

    arm_dir = output / "arms" / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    result_path = arm_dir / "results.jsonl"
    expected_ids = {queries[index].row_id for index in selected}
    prior = _prior_rows(
        result_path, args.arm, expected_ids,
        validation["checkpoint_sha256"], validation["config_sha256"],
        manifest_sha256)
    if args.arm in RANDOM_ARMS and any(
            row.get("random_k_manifest_sha256") != random_sha
            for row in prior.values()):
        raise ValueError("Random-K manifest changed during resume")
    completed = len(prior)
    started = time.monotonic()
    with result_path.open("a", encoding="utf-8") as handle:
        for index in selected:
            query = queries[index]
            if query.row_id in prior:
                continue
            try:
                torch.manual_seed(SEED + index)
                np.random.seed(SEED + index)
                random.seed(SEED + index)
                paired = gallery_indices[query.image]
                clean_scores = query_features[index] @ gallery_features.T
                clean_rank = _rank(clean_scores, query.person_id, ids)
                attacked_text = query.caption
                events, rounds = [], []
                clean_pixels = None
                adv_pixels = None
                if args.arm != "clean":
                    image_path = (args.image_root / query.image).resolve()
                    if not image_path.is_relative_to(args.image_root.resolve()):
                        raise ValueError("query image escapes root")
                    with Image.open(image_path) as image:
                        clean_pixels = source.pixels(image).unsqueeze(0).to("cuda")
                    adv_pixels = clean_pixels
                    if args.arm == "vanilla_image":
                        adv_pixels = vanilla_tta_irra_image_attack(
                            source, args.tta_root, clean_pixels, query.caption)
                    elif args.arm == "vanilla_full":
                        adv_pixels, attacked_text = full_attack(
                            clean_pixels, query.caption)
                    elif args.arm in GUIDED_ARMS:
                        record = records[query.row_id]
                        if args.arm in RANDOM_ARMS:
                            record = apply_random_k(
                                record, random_entries[query.row_id], 42)
                        if args.arm == "text_only" or record["k_star"] == 0:
                            image_callback = lambda request: ImageAttackResult(
                                request.state.image)
                        else:
                            image_callback = IRRATTAImageCallback(
                                source, args.tta_root, clean_pixels)

                        def candidate_scores(current_image, texts):
                            with torch.no_grad():
                                gallery = gallery_features
                                if args.arm != "text_only":
                                    gallery = gallery_features.clone()
                                    gallery[paired:paired + 1] = source.encode_images(
                                        current_image)
                                return source.soft_rank(
                                    texts, gallery,
                                    [query.person_id] * len(texts), ids, tau)

                        if args.arm in {"attribute_tta_only", "random_k_tta_only"}:
                            text_callback = lambda request: TextAttackResult(
                                request.state.text)
                        else:
                            text_callback = ConfusableTextCallback(candidate_scores)
                        result = run_attack(
                            record, image=clean_pixels, image_attack=image_callback,
                            text_attack=text_callback)
                        adv_pixels = result.state.image
                        attacked_text = result.state.text
                        events = getattr(text_callback, "events", [])
                        rounds = [item.slot for item in result.completed]
                        if rounds != record["selected_slots"]:
                            raise AssertionError("Stage 05 attribute order changed")
                        _text_events_are_valid(record, attacked_text, events)

                linf = 0.0
                if args.arm == "clean":
                    rank = clean_rank
                else:
                    linf = float((adv_pixels - clean_pixels).abs().max())
                    if linf > IMAGE_EPSILON + 1e-6:
                        raise AssertionError("IRRA attack image exceeds 8/255")
                    with torch.no_grad():
                        gallery = gallery_features
                        if args.arm in IMAGE_ARMS and linf > 0:
                            gallery = gallery_features.clone()
                            gallery[paired:paired + 1] = source.encode_images(
                                adv_pixels)
                        scores = source.retrieval_scores([attacked_text], gallery)[0]
                        rank = _rank(scores, query.person_id, ids)
                row = {
                    "row_id": query.row_id, "arm": args.arm,
                    "person_id": query.person_id, "image": query.image,
                    "clean_rank": clean_rank, "rank": rank,
                    "text": attacked_text, "text_changed": attacked_text != query.caption,
                    "linf": linf, "rounds": rounds, "text_events": events,
                    "attack_source": SOURCE_LABEL,
                    "checkpoint_sha256": validation["checkpoint_sha256"],
                    "config_sha256": validation["config_sha256"],
                    "query_manifest_sha256": manifest_sha256,
                    "irra_soft_rank_tau": tau,
                    "image_size": list(IMAGE_SIZE),
                }
                if args.arm in RANDOM_ARMS:
                    row["random_attack_order"] = rounds
                    row["random_k_manifest_sha256"] = random_sha
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                print(json.dumps({
                    "arm": args.arm, "done": completed, "rank": rank,
                    "clean_rank": clean_rank,
                    "elapsed_sec": round(time.monotonic() - started, 1),
                }), flush=True)
            except Exception as exc:
                with (arm_dir / "errors.jsonl").open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({
                        "row_id": query.row_id, "arm": args.arm,
                        "error": repr(exc), "traceback": traceback.format_exc(),
                    }) + "\n")
                raise


def summarize(output, queries, selected, manifest_sha256):
    validation = json.loads(
        (output / "official_clean_validation.json").read_text(encoding="utf-8"))
    ids = [queries[index].row_id for index in selected]
    by_arm = {}
    for arm in ARMS:
        path = output / "arms" / arm / "results.jsonl"
        rows = {}
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if (row.get("row_id") in rows or row.get("arm") != arm
                    or row.get("attack_source") != SOURCE_LABEL
                    or row.get("checkpoint_sha256") != validation["checkpoint_sha256"]
                    or row.get("config_sha256") != validation["config_sha256"]
                    or row.get("query_manifest_sha256") != manifest_sha256):
                raise ValueError(f"{arm} result metadata differs")
            rows[row["row_id"]] = row
        if set(rows) != set(ids):
            raise ValueError(f"{arm} does not cover exactly the manifest")
        by_arm[arm] = [rows[row_id] for row_id in ids]

    clean = np.array([row["rank"] for row in by_arm["clean"]])
    metrics = {}
    for arm, rows in by_arm.items():
        ranks = np.array([row["rank"] for row in rows])
        refs = np.array([row["clean_rank"] for row in rows])
        if not np.array_equal(refs, clean):
            raise ValueError(f"{arm} IRRA clean ranks differ")
        delta = ranks - clean
        metrics[arm] = {
            "query_count": len(selected),
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
        "attack_source": SOURCE_LABEL,
        "protocol": "reid_raw.json test, 6156 queries, 3074 gallery; manifest only",
        "query_manifest_sha256": manifest_sha256,
        "query_row_ids": ids,
        "arms": metrics,
        "per_query": [
            {"row_id": row_id, **{arm: by_arm[arm][i]["rank"] for arm in ARMS}}
            for i, row_id in enumerate(ids)
        ],
    }
    path = output / "summary_500_irra_white_box_8arms.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def gradient_check(args):
    if not torch.cuda.is_available():
        raise RuntimeError("IRRA gradient check requires CUDA")
    victim = FrozenIRRA(
        args.irra_repo, args.irra_checkpoint, args.irra_config, device="cuda")
    source = IRRASurrogate(victim)
    pixels = torch.full(
        (1, 3, *IMAGE_SIZE), 0.5, device="cuda", requires_grad=True)
    loss = source.attack_loss(
        pixels, ["A person wearing a red shirt and black pants."])
    gradient = torch.autograd.grad(loss, pixels)[0]
    if not torch.isfinite(gradient).all() or not bool((gradient.abs() > 0).any()):
        raise RuntimeError("IRRA image gradient is zero or nonfinite")
    print(json.dumps({
        "source": SOURCE_LABEL, "device": str(pixels.device),
        "loss": float(loss.detach()),
        "gradient_abs_max": float(gradient.abs().max()),
        "checkpoint_sha256": file_sha256(args.irra_checkpoint),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("gradient-check", "validate-clean", "run", "summarize"))
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
    parser.add_argument("--tta-root", type=Path, default=DEFAULT_TTA)
    parser.add_argument("--bert", type=Path, default=DEFAULT_BERT)
    parser.add_argument("--glove", type=Path, default=DEFAULT_GLOVE)
    args = parser.parse_args()
    if args.command == "gradient-check":
        gradient_check(args)
        return
    output, queries, paths, ids, selected = protocol_paths(args)
    manifest_sha256 = file_sha256(args.query_manifest)
    if args.command == "summarize":
        summarize(output, queries, selected, manifest_sha256)
    elif args.command == "validate-clean":
        _, _, _, report = victim_clean(args, output, queries, paths, ids)
        print(json.dumps(report), flush=True)
    else:
        if args.arm is None:
            parser.error("run requires --arm")
        run_one_arm(args, output, queries, paths, ids, selected, manifest_sha256)


if __name__ == "__main__":
    main()
