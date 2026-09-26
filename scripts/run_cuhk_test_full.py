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
from collections import defaultdict
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


def build_sample500_manifest(annotation: Path, queries: tuple[Query, ...],
                             seed: int = 42) -> dict:
    """固定前 20 条，再从其他 ID 各取一条，避免标注顺序聚集。"""
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError("sample500 requires the complete CUHK test split")
    first = queries[:20]
    first_ids = {query.person_id for query in first}
    by_id = defaultdict(list)
    for query in queries[20:]:
        if query.person_id not in first_ids:
            by_id[query.person_id].append(query)
    if len(by_id) < 480:
        raise ValueError("not enough new person IDs for sample500")
    rng = random.Random(seed)
    identities = sorted(by_id, key=str)
    rng.shuffle(identities)
    added = [rng.choice(by_id[person_id]) for person_id in identities[:480]]
    rng.shuffle(added)
    chosen = (*first, *added)
    return {
        "protocol": "CUHK-PEDES reid_raw.json test",
        "annotation_sha256": _sha256(annotation),
        "seed": seed,
        "sampling": "first 20 annotation queries; 480 distinct new IDs sampled with Python random.Random",
        "first20_count": 20,
        "added_count": 480,
        "unique_person_ids": len({query.person_id for query in chosen}),
        "unique_images": len({query.image for query in chosen}),
        "queries": [
            {"row_id": query.row_id, "person_id": query.person_id,
             "image": query.image, "from_first20": index < 20}
            for index, query in enumerate(chosen)
        ],
    }


def create_sample500_manifest(args) -> None:
    annotation = Path(args.annotation).resolve()
    queries, _, _ = load_test_protocol(annotation)
    if not args.query_manifest:
        raise ValueError("create-manifest requires --query-manifest")
    manifest = build_sample500_manifest(annotation, queries, args.seed)
    path = Path(args.query_manifest).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("existing manifest differs from deterministic sample")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    print(json.dumps({"manifest": str(path), "query_count": 500,
                      "first20": 20, "added": 480,
                      "unique_person_ids": manifest["unique_person_ids"],
                      "unique_images": manifest["unique_images"]}), flush=True)


def load_query_manifest(path: Path, annotation: Path,
                        queries: tuple[Query, ...]) -> tuple[int, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("queries")
    if (data.get("annotation_sha256") != _sha256(annotation)
            or data.get("first20_count") != 20
            or data.get("added_count") != 480
            or not isinstance(rows, list) or len(rows) != 500):
        raise ValueError("query manifest protocol or annotation differs")
    indices = {query.row_id: index for index, query in enumerate(queries)}
    selected = []
    seen = set()
    for position, row in enumerate(rows):
        row_id = row.get("row_id")
        if row_id not in indices or row_id in seen:
            raise ValueError(f"unknown or duplicate manifest row: {row_id}")
        index = indices[row_id]
        query = queries[index]
        if (row.get("person_id") != query.person_id or row.get("image") != query.image
                or row.get("from_first20") is not (position < 20)
                or (position < 20 and index != position)
                or (position >= 20 and index < 20)):
            raise ValueError(f"manifest query pairing differs: {row_id}")
        seen.add(row_id)
        selected.append(index)
    if (len({queries[index].person_id for index in selected})
            != data.get("unique_person_ids")
            or len({queries[index].image for index in selected})
            != data.get("unique_images")):
        raise ValueError("manifest diversity counts differ")
    return tuple(selected)


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


def _manifest_result_path(output_dir: Path, arm: str, manifest: Path) -> Path:
    return output_dir / "arms" / arm / (manifest.stem + ".jsonl")


def _manifest_binding_path(output_dir: Path, arm: str, manifest: Path) -> Path:
    return output_dir / "arms" / arm / (manifest.stem + ".sha256")


def _check_manifest_binding(output_dir: Path, arm: str, manifest: Path,
                            *, create: bool = False) -> None:
    results = _manifest_result_path(output_dir, arm, manifest)
    binding = _manifest_binding_path(output_dir, arm, manifest)
    digest = _sha256(manifest)
    if binding.exists():
        if binding.read_text(encoding="ascii").strip() != digest:
            raise ValueError("query manifest changed after results were written")
    elif results.exists():
        raise ValueError("manifest results exist without a SHA-256 binding")
    elif create:
        binding.write_text(digest + chr(10), encoding="ascii")


def _read_result_map(path: Path, queries: tuple[Query, ...], arm: str,
                     allowed_ids: set[str]) -> dict[str, dict]:
    """按 row_id 读取增量文件；残缺最终行可安全恢复。"""
    if not path.exists():
        return {}
    expected = {query.row_id: query for query in queries}
    rows = {}
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                if not line.endswith(bytes((10,))):
                    raise ValueError("incomplete JSONL tail")
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if handle.read(1):
                    raise ValueError("invalid manifest JSONL row before final tail")
                handle.truncate(offset)
                break
            row_id = row.get("row_id")
            if (row_id not in allowed_ids or row_id in rows
                    or row.get("arm") != arm
                    or row.get("person_id") != expected[row_id].person_id
                    or row.get("image") != expected[row_id].image):
                raise ValueError(f"invalid or duplicate result row: {row_id}")
            rows[row_id] = row
    return rows


def available_manifest_rows(output_dir: Path, arm: str,
                            queries: tuple[Query, ...],
                            manifest: Path, selected: tuple[int, ...]) -> dict[str, dict]:
    """合并旧全量前缀和本 manifest 的增量文件，不改旧结果。"""
    full = output_dir / "arms" / arm / "results.jsonl"
    completed = _read_completed(full, queries, arm)
    allowed = {queries[index].row_id for index in selected}
    rows = {}
    if completed:
        with full.open(encoding="utf-8") as handle:
            for index in range(completed):
                row = json.loads(handle.readline())
                if row["row_id"] in allowed:
                    query = queries[index]
                    if row["person_id"] != query.person_id or row["image"] != query.image:
                        raise ValueError(f"existing result pairing differs: {query.row_id}")
                    rows[row["row_id"]] = row
    _check_manifest_binding(output_dir, arm, manifest)
    extra = _read_result_map(
        _manifest_result_path(output_dir, arm, manifest), queries, arm, allowed)
    for row_id, row in extra.items():
        if row_id in rows and row != rows[row_id]:
            raise ValueError(f"full and manifest results disagree: {row_id}")
        rows[row_id] = row
    return rows


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
    manifest = Path(args.query_manifest).resolve() if args.query_manifest else None
    selected = load_query_manifest(manifest, annotation, queries) if manifest else None
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
    output = (_manifest_result_path(output_dir, args.arm, manifest) if manifest
              else arm_dir / "results.jsonl")
    log_path = arm_dir / ((manifest.stem + ".log") if manifest else "run.log")
    error_path = arm_dir / ((manifest.stem + ".errors.jsonl") if manifest
                            else "errors.jsonl")
    def emit(payload):
        message = json.dumps(payload, ensure_ascii=False)
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(message + "\n")
    if manifest:
        _check_manifest_binding(output_dir, args.arm, manifest, create=True)
        existing = available_manifest_rows(
            output_dir, args.arm, queries, manifest, selected)
        pending = [index for index in selected
                   if queries[index].row_id not in existing]
        completed = len(existing)
        total = len(selected)
    else:
        completed = _read_completed(output, queries, args.arm)
        pending = list(range(completed, len(queries)))
        total = len(queries)
    work = pending[:args.batch_size] if args.batch_size else pending
    if not work:
        emit({"arm": args.arm, "completed": completed, "total": total,
              "status": "already_complete",
              "manifest": str(manifest) if manifest else None})
        return
    emit({"arm": args.arm, "completed_before": completed,
          "this_run": len(work), "total": total,
          "manifest": str(manifest) if manifest else None,
          "device": str(device)})
    prep = preprocess(source.resolution)
    full_attack = None
    if args.arm == "vanilla_full":
        full_attack = FullVanillaTTA(
            source, args.tta_root, args.bert, args.glove)
        if next(full_attack.ref_net.parameters()).device.type != "cuda":
            raise RuntimeError("TTA text model did not load on CUDA")
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle:
        for processed, index in enumerate(work, 1):
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
                if processed % 25 == 0 or processed == len(work):
                    elapsed = time.perf_counter() - started
                    emit({"arm": args.arm, "done": completed + processed,
                          "total": total,
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
    selected_arms = (args.arm,) if args.arm else ARMS
    to_load = tuple(dict.fromkeys(("clean",) + selected_arms))
    manifest = Path(args.query_manifest).resolve() if args.query_manifest else None
    if manifest:
        selected_indices = load_query_manifest(manifest, annotation, queries)
        selected_queries = [queries[index] for index in selected_indices]
        for arm in to_load:
            available = available_manifest_rows(
                output_dir, arm, queries, manifest, selected_indices)
            missing = [query.row_id for query in selected_queries
                       if query.row_id not in available]
            if missing:
                raise ValueError(f"arm incomplete: {arm}; missing {len(missing)} manifest rows")
            rows_by_arm[arm] = [available[query.row_id] for query in selected_queries]
    else:
        selected_queries = queries
        for arm in to_load:
            path = output_dir / "arms" / arm / "results.jsonl"
            if _read_completed(path, queries, arm) != len(queries):
                raise ValueError(f"arm incomplete: {arm}")
            with path.open(encoding="utf-8") as handle:
                rows_by_arm[arm] = [json.loads(line) for line in handle]
    for index, query in enumerate(selected_queries):
        reference = rows_by_arm["clean"][index]["rank"]
        for arm in selected_arms:
            row = rows_by_arm[arm][index]
            if (row["row_id"] != query.row_id or row["person_id"] != query.person_id
                    or row["image"] != query.image or row["clean_rank"] != reference):
                raise ValueError(f"query or clean reference mismatch: {arm} {query.row_id}")
    summary = {}
    for arm in selected_arms:
        rows = rows_by_arm[arm]
        ranks = np.array([row["rank"] for row in rows], dtype=np.int32)
        clean = np.array([row["clean_rank"] for row in rows], dtype=np.int32)
        summary[arm] = {
            "query_count": len(rows),
            "mean_first_hit_rank": float(ranks.mean()),
            "median_first_hit_rank": float(np.median(ranks)),
            "mean_rank_delta": float((ranks - clean).mean()),
            "median_rank_delta": float(np.median(ranks - clean)),
            "rank_delta_positive_count": int(((ranks - clean) > 0).sum()),
            "rank_delta_zero_count": int(((ranks - clean) == 0).sum()),
            "rank_delta_negative_count": int(((ranks - clean) < 0).sum()),
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
    if manifest:
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        report["query_manifest"] = str(manifest)
        report["query_manifest_sha256"] = _sha256(manifest)
        report["unique_person_ids"] = manifest_data["unique_person_ids"]
        report["unique_images"] = manifest_data["unique_images"]
    suffix = ("_" + args.arm) if args.arm else ""
    suffix += ("_" + manifest.stem) if manifest else ""
    path = output_dir / ("summary" + suffix + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def status(args):
    if not args.query_manifest:
        raise ValueError("status requires --query-manifest")
    annotation = Path(args.annotation).resolve()
    queries, _, _ = load_test_protocol(annotation)
    manifest = Path(args.query_manifest).resolve()
    selected = load_query_manifest(manifest, annotation, queries)
    output_dir = Path(args.output_dir).resolve()
    arms = (args.arm,) if args.arm else ARMS
    counts = {}
    for arm in arms:
        existing = available_manifest_rows(output_dir, arm, queries, manifest, selected)
        counts[arm] = {"done": len(existing), "remaining": len(selected) - len(existing)}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"query_manifest": str(manifest), "query_count": len(selected),
                      "first20": data["first20_count"], "added": data["added_count"],
                      "unique_person_ids": data["unique_person_ids"],
                      "unique_images": data["unique_images"],
                      "arms": counts}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "summarize", "status", "create-manifest"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--stage04")
    parser.add_argument("--image-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tta-root")
    parser.add_argument("--bert")
    parser.add_argument("--glove")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-manifest")
    parser.add_argument("--seed", type=int, default=42)
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
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "status":
        status(args)
    else:
        create_sample500_manifest(args)


if __name__ == "__main__":
    main()
