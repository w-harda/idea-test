#!/usr/bin/env python3
"""CUHK-PEDES test 6156-query / 3074-gallery 六组可续跑评测。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from attributes.attack_scheduler import ImageAttackResult, TextAttackResult, plan_attack, run_attack
from attributes.confusable_attack import CONFUSABLES, ConfusableTextCallback
from attributes.frozen_clip import FrozenCLIP
from attributes.rank_objective import exact_first_hit_rank, soft_first_hit_rank
from attributes.tta_adapter import TTAImageCallback, vanilla_tta_image_attack
from attributes.tta_full import FullVanillaTTA

ARMS = ("clean", "text_only", "vanilla_image", "vanilla_full",
        "attribute_tta_only", "attribute_tta_text")
GUIDED_ARMS = {"text_only", "attribute_tta_only", "attribute_tta_text"}
IMAGE_ARMS = {"vanilla_image", "vanilla_full", "attribute_tta_only", "attribute_tta_text"}
SEED = 20260926
TAU = 0.07774211466312408  # Earlier train PoC calibration; fixed before test.
EXPECTED_QUERIES = 6156
EXPECTED_GALLERY = 3074
EXPECTED_IDENTITIES = 1000


@dataclass(frozen=True)
class Query:
    row_id: str
    image: str
    person_id: int | str
    caption: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_protocol(annotation: Path):
    """只读取原作者 reid_raw.json 的 test split，保留原始顺序。"""
    with annotation.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("CUHK annotation must be a JSON array")
    gallery = {}
    queries = []
    for row_index, row in enumerate(rows):
        if row["split"] != "test":
            continue
        path, person_id = row["file_path"], row["id"]
        if path in gallery:
            raise ValueError(f"duplicate test gallery image: {path}")
        gallery[path] = person_id
        for caption_index, caption in enumerate(row["captions"]):
            queries.append(Query(
                f"cuhk:{row_index}:{caption_index}", path, person_id, caption))
    if (len(queries) != EXPECTED_QUERIES or len(gallery) != EXPECTED_GALLERY
            or len(set(gallery.values())) != EXPECTED_IDENTITIES
            or any(not query.caption for query in queries)):
        raise ValueError("CUHK test split does not match 6156/3074/1000 protocol")
    return tuple(queries), tuple(gallery), tuple(gallery.values())


def load_stage04_test(path: Path, queries: tuple[Query, ...]) -> dict[str, dict]:
    expected = {query.row_id: query for query in queries}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            if record["split"] != "test":
                continue
            row_id = record["row_id"]
            if row_id not in expected or row_id in records:
                raise ValueError(f"duplicate/unknown Stage 04 test row at {line_number}")
            query = expected[row_id]
            if record["image"] != query.image or record["caption"] != query.caption:
                raise ValueError(f"Stage 04 test pairing differs: {row_id}")
            plan_attack(record)
            if record["k_star"] and not isinstance(record.get("provenance"), dict):
                raise ValueError(f"Stage 04 test provenance missing: {row_id}")
            records[row_id] = record
    if set(records) != set(expected):
        raise ValueError(f"Stage 04 missing {len(set(expected) - set(records))} test rows")
    return records


def _fingerprint(pairs) -> str:
    return hashlib.sha256(json.dumps(
        list(pairs), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def cache_metadata(annotation: Path, checkpoint: Path, queries, paths, ids):
    return {
        "protocol": "CUHK-PEDES reid_raw.json test only",
        "annotation_sha256": _sha256(annotation),
        "clip_sha256": _sha256(checkpoint),
        "gallery_sha256": _fingerprint(zip(paths, ids, strict=True)),
        "query_sha256": _fingerprint(
            (query.row_id, query.image, query.person_id, query.caption)
            for query in queries),
        "queries": len(queries), "gallery_images": len(paths),
        "identities": len(set(ids)),
    }


def preprocess(resolution: int):
    return transforms.Compose([
        transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def load_or_build_features(
    cache_dir: Path, metadata: dict, source: FrozenCLIP, queries,
    paths, image_root: Path, device: torch.device,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    gallery_path = cache_dir / "gallery_features.npy"
    query_path = cache_dir / "query_features.npy"
    if meta_path.is_file() and gallery_path.is_file() and query_path.is_file():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing == metadata:
            gallery = np.load(gallery_path, allow_pickle=False)
            query = np.load(query_path, allow_pickle=False)
            if (gallery.shape == (EXPECTED_GALLERY, 512)
                    and query.shape == (EXPECTED_QUERIES, 512)
                    and np.isfinite(gallery).all() and np.isfinite(query).all()):
                return (
                    torch.from_numpy(gallery.copy()).to(device),
                    torch.from_numpy(query.copy()).to(device),
                )
        raise ValueError("feature cache exists but metadata or tensors differ")

    prep = preprocess(source.resolution)
    gallery_chunks = []
    for start in range(0, len(paths), 64):
        pixels = []
        for relative in paths[start:start + 64]:
            path = (image_root / relative).resolve()
            if not path.is_relative_to(image_root):
                raise ValueError("test gallery path escapes image root")
            with Image.open(path) as handle:
                pixels.append(prep(handle.convert("RGB")))
        with torch.no_grad():
            gallery_chunks.append(
                source.encode_images(torch.stack(pixels).to(device)).cpu())
        if start % 512 == 0:
            print(json.dumps({"cache_gallery_done": min(start + 64, len(paths))}),
                  flush=True)
    gallery = torch.cat(gallery_chunks).numpy()
    query_chunks = []
    for start in range(0, len(queries), 128):
        with torch.no_grad():
            query_chunks.append(source.encode_texts([
                item.caption for item in queries[start:start + 128]]).cpu())
    query = torch.cat(query_chunks).numpy()
    _save_npy_atomic(gallery_path, gallery)
    _save_npy_atomic(query_path, query)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    return torch.from_numpy(gallery.copy()).to(device), torch.from_numpy(query.copy()).to(device)


def _read_completed(path: Path, queries: tuple[Query, ...], arm: str) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete JSONL tail")
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if handle.read(1):
                    raise ValueError("invalid JSONL row before final tail")
                handle.truncate(offset)
                break
            if (count >= len(queries) or row.get("row_id") != queries[count].row_id
                    or row.get("arm") != arm):
                raise ValueError("JSONL result prefix differs from test protocol")
            count += 1
    return count


def _rank(scores: torch.Tensor, person_id, gallery_ids) -> int:
    return int(exact_first_hit_rank(scores[None], [person_id], gallery_ids)[0])


def _soft(scores: torch.Tensor, person_id, gallery_ids) -> float:
    return float(soft_first_hit_rank(
        scores[None], [person_id], gallery_ids, TAU)[0])


def _text_events_are_valid(record: dict, final_text: str, events: list[dict]) -> None:
    original = record["caption"]
    edits = {event["offset"]: event for event in events if event["selected"]}
    differences = {offset for offset, (before, after) in enumerate(
        zip(original, final_text, strict=True)) if before != after}
    if len(original) != len(final_text) or differences != set(edits):
        raise AssertionError("text edit count or position differs from callback events")
    for offset, event in edits.items():
        if (event["round"] < 1 or event["round"] > record["k_star"]
                or not any(
                    mention["start"] <= offset < mention["end"]
                    for mention in record["provenance"][event["slot"]]["mentions"])
                or final_text[offset] not in CONFUSABLES.get(original[offset], ())):
            raise AssertionError("text edit outside current selected attribute")


def run_arm(args):
    annotation = Path(args.annotation).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    image_root = Path(args.image_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    queries, paths, gallery_ids = load_test_protocol(annotation)
    records = None
    if args.arm in GUIDED_ARMS:
        if not args.stage04:
            raise ValueError("guided arm requires --stage04")
        records = load_stage04_test(Path(args.stage04), queries)
    elif args.stage04:
        raise ValueError("vanilla and clean arms must not receive --stage04")
    if not torch.cuda.is_available():
        raise RuntimeError("formal attack requires server GPU")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    source = FrozenCLIP(checkpoint, device=device).to(device)
    if not source.mean.is_cuda or not next(source.model.parameters()).is_cuda:
        raise RuntimeError("Frozen CLIP did not load on CUDA")
    metadata = cache_metadata(annotation, checkpoint, queries, paths, gallery_ids)
    gallery_features, query_features = load_or_build_features(
        output_dir / "cache", metadata, source, queries, paths, image_root, device)
    gallery_indices = {path: index for index, path in enumerate(paths)}
    arm_dir = output_dir / "arms" / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    output = arm_dir / "results.jsonl"
    log_path = arm_dir / "run.log"
    error_path = arm_dir / "errors.jsonl"
    def emit(payload):
        message = json.dumps(payload, ensure_ascii=False)
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(message + "\n")
    start = _read_completed(output, queries, args.arm)
    if start == len(queries):
        emit({"arm": args.arm, "completed": start, "status": "already_complete"})
        return
    stop = min(start + args.batch_size, len(queries)) if args.batch_size else len(queries)
    emit({"arm": args.arm, "start": start, "stop": stop,
          "total": len(queries), "device": str(device)})
    prep = preprocess(source.resolution)
    full_attack = None
    if args.arm == "vanilla_full":
        full_attack = FullVanillaTTA(
            source, args.tta_root, args.bert, args.glove)
        if next(full_attack.ref_net.parameters()).device.type != "cuda":
            raise RuntimeError("TTA text model did not load on CUDA")
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle:
        for index in range(start, stop):
            try:
                query = queries[index]
                torch.manual_seed(SEED + index)
                np.random.seed(SEED + index)
                random.seed(SEED + index)
                paired = gallery_indices[query.image]
                clean_scores = query_features[index] @ gallery_features.T
                clean_rank = _rank(clean_scores, query.person_id, gallery_ids)
                clean_soft = _soft(clean_scores, query.person_id, gallery_ids)
                image = None
                if args.arm != "clean":
                    path = (image_root / query.image).resolve()
                    if not path.is_relative_to(image_root):
                        raise ValueError("query image path escapes image root")
                    with Image.open(path) as raw:
                        image = prep(raw.convert("RGB")).unsqueeze(0).to(device)
                    if not image.is_cuda:
                        raise RuntimeError("attack input did not load on CUDA")
                attacked = image
                attacked_text = query.caption
                events = []
                rounds = []
                if args.arm == "vanilla_image":
                    attacked = vanilla_tta_image_attack(
                        source, args.tta_root, image, query.caption)
                elif args.arm == "vanilla_full":
                    attacked, attacked_text = full_attack(image, query.caption)
                elif args.arm in GUIDED_ARMS:
                    record = records[query.row_id]
                    if args.arm == "text_only" or record["k_star"] == 0:
                        image_callback = lambda request: ImageAttackResult(request.state.image)
                    else:
                        image_callback = TTAImageCallback(
                            source, args.tta_root, image)
                    def candidate_scores(current_image, texts):
                        with torch.no_grad():
                            if args.arm == "text_only":
                                gallery = gallery_features
                            else:
                                gallery = gallery_features.clone()
                                gallery[paired:paired + 1] = source.encode_images(current_image)
                            scores = source.encode_texts(texts) @ gallery.T
                            return soft_first_hit_rank(
                                scores, [query.person_id] * len(texts), gallery_ids, TAU)
                    if args.arm == "attribute_tta_only":
                        text_callback = lambda request: TextAttackResult(request.state.text)
                    else:
                        text_callback = ConfusableTextCallback(candidate_scores)
                    result = run_attack(
                        record, image=image, image_attack=image_callback,
                        text_attack=text_callback)
                    attacked = result.state.image
                    attacked_text = result.state.text
                    events = getattr(text_callback, "events", [])
                    rounds = [item.slot for item in result.completed]
                    if rounds != record["selected_slots"]:
                        raise AssertionError("Stage 05 attribute order changed")
                    _text_events_are_valid(record, attacked_text, events)
                if args.arm == "clean":
                    rank, soft = clean_rank, clean_soft
                    linf = 0.0
                else:
                    if not isinstance(attacked, torch.Tensor) or not attacked.is_cuda:
                        raise RuntimeError("attack output did not remain on CUDA")
                    linf = float((attacked - image).abs().max())
                    if linf > 8 / 255 + 1e-6:
                        raise AssertionError("TTA cumulative image perturbation exceeded 8/255")
                    with torch.no_grad():
                        if args.arm == "text_only":
                            gallery = gallery_features
                        else:
                            gallery = gallery_features.clone()
                            gallery[paired:paired + 1] = source.encode_images(attacked)
                        final_scores = source.encode_texts([attacked_text])[0] @ gallery.T
                    rank = _rank(final_scores, query.person_id, gallery_ids)
                    soft = _soft(final_scores, query.person_id, gallery_ids)
                row = {
                    "row_id": query.row_id, "arm": args.arm, "person_id": query.person_id,
                    "image": query.image, "clean_rank": clean_rank, "rank": rank,
                    "clean_soft_rank": clean_soft, "soft_rank": soft,
                    "linf": linf, "rounds": rounds, "text": attacked_text,
                    "text_changed": attacked_text != query.caption,
                    "text_events": events,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if (index + 1) % 25 == 0 or index + 1 == stop:
                    elapsed = time.perf_counter() - started
                    emit({"arm": args.arm, "done": index + 1, "total": len(queries),
                          "elapsed_this_run_sec": round(elapsed, 1),
                          "last_rank": rank, "last_clean_rank": clean_rank})

            except Exception as exc:
                error = {"arm": args.arm, "index": index,
                         "row_id": queries[index].row_id,
                         "error": repr(exc), "traceback": traceback.format_exc()}
                with error_path.open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps(error, ensure_ascii=False) + "\n")
                emit({"arm": args.arm, "failed_index": index,
                      "row_id": queries[index].row_id, "error": repr(exc)})
                raise

def summarize(args):
    annotation = Path(args.annotation).resolve()
    queries, paths, gallery_ids = load_test_protocol(annotation)
    output_dir = Path(args.output_dir).resolve()
    rows_by_arm = {}
    selected = (args.arm,) if args.arm else ARMS
    to_load = tuple(dict.fromkeys(("clean",) + selected))
    for arm in to_load:
        path = output_dir / "arms" / arm / "results.jsonl"
        if _read_completed(path, queries, arm) != len(queries):
            raise ValueError(f"arm incomplete: {arm}")
        with path.open(encoding="utf-8") as handle:
            rows_by_arm[arm] = [json.loads(line) for line in handle]
    for index, query in enumerate(queries):
        reference = rows_by_arm["clean"][index]["rank"]
        for arm in selected:
            row = rows_by_arm[arm][index]
            if row["person_id"] != query.person_id or row["clean_rank"] != reference:
                raise ValueError(f"clean reference mismatch: {arm} {query.row_id}")
    summary = {}
    for arm in selected:
        rows = rows_by_arm[arm]
        ranks = np.array([row["rank"] for row in rows], dtype=np.int32)
        clean = np.array([row["clean_rank"] for row in rows], dtype=np.int32)
        summary[arm] = {
            "query_count": len(rows),
            "mean_first_hit_rank": float(ranks.mean()),
            "median_first_hit_rank": float(np.median(ranks)),
            "mean_rank_delta": float((ranks - clean).mean()),
            "median_rank_delta": float(np.median(ranks - clean)),
            "rank_delta_positive": float(((ranks - clean) > 0).mean()),
            "rank_delta_zero": float(((ranks - clean) == 0).mean()),
            "rank_delta_negative": float(((ranks - clean) < 0).mean()),
            "rank1": float((ranks <= 1).mean()),
            "rank5": float((ranks <= 5).mean()),
            "rank10": float((ranks <= 10).mean()),
            "text_changed": sum(row["text_changed"] for row in rows),
            "mean_linf": float(np.mean([row["linf"] for row in rows])),
            "max_linf": max(row["linf"] for row in rows),
        }
    cache = json.loads((output_dir / "cache" / "metadata.json").read_text())
    report = {
        "dataset": "CUHK-PEDES", "split": "test",
        "protocol": "original reid_raw.json test: 6156 captions, 3074 images, 1000 IDs",
        "gallery_sha256": cache["gallery_sha256"],
        "query_sha256": cache["query_sha256"],
        "clip_sha256": cache["clip_sha256"],
        "tau": TAU,
        "attack_scope": "per-query paired image only; other test gallery images clean",
        "text_candidate_selection": "known test ID source soft-rank oracle",
        "arms": summary,
    }
    path = output_dir / (f"summary_{args.arm}.json" if args.arm else "summary.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--stage04")
    parser.add_argument("--image-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tta-root")
    parser.add_argument("--bert")
    parser.add_argument("--glove")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="本次最多处理多少条剩余 query；0 表示跑完")
    args = parser.parse_args()
    if args.batch_size < 0:
        parser.error("--batch-size must be nonnegative")
    if args.command == "run":
        if not args.arm or not args.image_root or not args.checkpoint:
            parser.error("run requires --arm, --image-root and --checkpoint")
        if args.arm in IMAGE_ARMS and not args.tta_root:
            parser.error("TTA image arm requires --tta-root")
        if args.arm == "vanilla_full" and (not args.bert or not args.glove):
            parser.error("vanilla_full requires --bert and --glove")
        run_arm(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
