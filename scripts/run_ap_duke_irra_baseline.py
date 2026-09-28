#!/usr/bin/env python3
"""AP-Attack (G_Duke -> CUHK-PEDES -> IRRA)；生成与 TBPS 评测分离。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# 第三方源码位于只读区时也不产生 __pycache__。
sys.dont_write_bytecode = True
os.environ.setdefault("TORCH_HOME", "/home/lzf/ldx/cache/torch")
os.environ.setdefault("HF_HOME", "/home/lzf/ldx/cache/huggingface")

import torch
from PIL import Image

from attributes.ap_gallery_baseline import (
    DEFAULT_G_SHA, ROOT, FrozenAPGenerator, atomic_json, atomic_torch,
    build_cache, cache_spec, cached_batches, fingerprint, gallery_paths,
    image_path, native_pixels, read_progress, sha256, transport_delta, write_path,
)

DEFAULT_OUTPUT = ROOT / "outputs/idea-TBPS-test1/baseline-ap-duke-irra"
DEFAULT_AP = ROOT / "projects/AP-Attack"
DEFAULT_G = ROOT / "outputs/AP-Attack/stage2_reid_semantic_10_10/best_G_V.pth.tar"
ANNOTATION = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGE_ROOT = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")
IRRA_REPO = Path("/home/lzf/Attack/IRRA-main/IRRA-main")
IRRA_CHECKPOINT = Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth")
IRRA_CONFIG = ROOT / "projects/idea-TBPS-test1/configs/irra_cuhk_official.yaml"
MANIFEST = ROOT / "outputs/idea-TBPS-test1/cuhk-test-stage1/sample500_seed42.json"
MANIFEST_SHA = "696cca268c042073b4e021415acac48aaa1b2402cbed97342056c041290d2306"
BASELINE = "AP-Attack (G_Duke → CUHK-PEDES → IRRA)"


def generator_spec(args):
    paths = gallery_paths(args.annotation)
    return paths, cache_spec(
        args.annotation, args.image_root, paths, args.ap_source,
        args.g_checkpoint, args.g_sha256, args.g_origin, args.generator_batch_size)


def generate(args, output, target=None):
    paths, spec = generator_spec(args)
    cache = output / "adversarial-gallery"
    progress = read_progress(cache, spec)
    target = len(paths) if target is None else min(target, len(paths))
    if target <= progress["done"]:
        return paths, spec
    if not torch.cuda.is_available():
        raise RuntimeError("AP 生成需要 CUDA")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    factory = lambda: FrozenAPGenerator(
        args.ap_source, args.g_checkpoint, args.g_sha256, device="cuda")
    build_cache(cache, spec, paths, args.image_root, factory,
                max_images=target - progress["done"])
    return paths, spec


def selected_protocol(args, scope, gallery):
    # 评测阶段复用冻结的官方 test 和 500-query manifest 读取接口。
    from scripts.run_cuhk_test_full import load_test_protocol, load_query_manifest

    queries, paths, ids = load_test_protocol(args.annotation)
    if tuple(gallery) != paths:
        raise ValueError("生成图库与官方 TBPS gallery 顺序不同")
    if scope == "500":
        if sha256(args.query_manifest) != MANIFEST_SHA:
            raise ValueError("固定 sample500_seed42.json SHA-256 不匹配")
        selected = load_query_manifest(args.query_manifest, args.annotation, queries)
        return [queries[i] for i in selected], paths, ids
    if scope == "full":
        return list(queries), paths, ids
    paths, ids = paths[:args.smoke_gallery], ids[:args.smoke_gallery]
    selected = [q for q in queries if q.image in paths][:args.smoke_queries]
    if not selected:
        raise ValueError("Smoke 图库没有可评测文本")
    return selected, paths, ids


def victim_metadata(args, spec, progress, count):
    from attributes.irra_victim import OFFICIAL_CUHK_SHA256
    from attributes.tbps_retrieval_metrics import IRRA_METRICS_SHA

    if sha256(args.irra_checkpoint) != OFFICIAL_CUHK_SHA256:
        raise ValueError("IRRA 不是固定官方 checkpoint")
    if sha256(args.irra_repo / "utils/metrics.py") != IRRA_METRICS_SHA:
        raise ValueError("IRRA 官方指标源码不匹配")
    return {
        "gallery_spec_sha256": fingerprint(spec),
        "chunks": [c for c in progress["chunks"] if c["start"] < count],
        "gallery_count": count, "irra_checkpoint_sha256": OFFICIAL_CUHK_SHA256,
        "irra_config_sha256": sha256(args.irra_config),
        "irra_source_hashes": {str(p): sha256(args.irra_repo / p) for p in
                              ("model/build.py", "model/clip_model.py",
                               "utils/simple_tokenizer.py", "utils/metrics.py")},
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "pixel_mapping": "official clean384 + bilinear(adv256-clean256), clamp[0,1]",
        "metric_protocol": "IRRA text-to-image; all same person-ID positives; no camera exclusion",
        "torch_version": str(torch.__version__),
    }


def gallery_features(args, output, spec, count, metadata, get_victim):
    feature_path = output / "irra-gallery.pt"
    if feature_path.exists():
        data = torch.load(feature_path, map_location="cpu", weights_only=True)
        if data["metadata"] != metadata:
            raise ValueError("IRRA gallery feature cache 配置不同")
        for key in ("clean", "adversarial"):
            if data[key].shape != (count, 512) or not torch.isfinite(data[key]).all():
                raise ValueError("IRRA gallery feature cache 形状或数值错误")
        return data
    victim = get_victim()
    clean_features, adv_features, all_linf = [], [], []
    for paths, raw_hashes, native_adv in cached_batches(
            output / "adversarial-gallery", spec, count):
        clean_native, clean_victim = [], []
        for relative, raw_sha in zip(paths, raw_hashes):
            path = image_path(args.image_root, relative)
            if sha256(path) != raw_sha:
                raise ValueError("原图内容与 AP 生成时不同")
            with Image.open(path) as image:
                clean_native.append(native_pixels(image))
                clean_victim.append(victim.pixels(image))
        clean_pixels = torch.stack(clean_victim).to("cuda")
        adv_pixels, linf = transport_delta(
            torch.stack(clean_native).to("cuda"), native_adv.to("cuda"), clean_pixels)
        with torch.no_grad():
            clean_features.append(victim.encode_images(clean_pixels).cpu())
            adv_features.append(victim.encode_images(adv_pixels).cpu())
        all_linf.extend(linf)
        print(json.dumps({"irra_gallery_encoded": len(all_linf), "target": count}),
              flush=True)
    data = {
        "metadata": metadata, "clean": torch.cat(clean_features),
        "adversarial": torch.cat(adv_features), "linf_victim": all_linf,
    }
    atomic_torch(feature_path, data)
    return data


def query_features(output, scope, queries, metadata, batch_size, get_victim):
    path = output / f"irra-queries-{scope}.pt"
    query_metadata = {
        "victim": metadata,
        "queries": [{"row_id": q.row_id, "person_id": q.person_id,
                     "caption": q.caption} for q in queries],
    }
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data["metadata"] != query_metadata:
            raise ValueError("IRRA text feature cache 配置不同")
        features = data["features"]
        if (features.shape != (len(queries), 512) or not torch.isfinite(features).all()):
            raise ValueError("IRRA text feature cache 形状或数值错误")
        return features
    victim = get_victim()
    chunks = []
    for start in range(0, len(queries), batch_size):
        with torch.no_grad():
            chunks.append(victim.encode_texts(
                [q.caption for q in queries[start:start + batch_size]]).cpu())
    features = torch.cat(chunks)
    atomic_torch(path, {"metadata": query_metadata, "features": features})
    return features


def write_query_rows(path, rows):
    path = write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = write_path(path.with_suffix(".jsonl.tmp"))
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def make_summary(output, scope):
    from attributes.tbps_retrieval_metrics import paired_summary

    directory = output / "evaluations" / scope
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (directory / "per-query.jsonl").read_text(encoding="utf-8").splitlines()]
    if ([r["row_id"] for r in rows] != meta["query_row_ids"]
            or len(rows) != meta["query_count"]):
        raise ValueError("评测结果尚未完整覆盖对应 query scope")
    expected_count = {"500": 500, "full": 6156}
    if scope in expected_count and (
            len(rows) != expected_count[scope] or meta["gallery_count"] != 3074
            or not meta["gallery_complete"]):
        raise ValueError("正式 TBPS 协议不完整")
    clean = [{**r["clean"], "row_id": r["row_id"], "person_id": r["person_id"]}
             for r in rows]
    adv = [{**r["adversarial"], "row_id": r["row_id"], "person_id": r["person_id"]}
           for r in rows]
    summary = {
        "baseline": BASELINE, "scope": scope,
        "smoke_only": scope == "smoke", "metadata": meta,
        **paired_summary(clean, adv),
    }
    atomic_json(directory / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "metadata"},
                     ensure_ascii=False), flush=True)
    return summary


def evaluate(args, output, scope):
    from attributes.irra_victim import FrozenIRRA
    from attributes.tbps_retrieval_metrics import text_to_image_metrics

    gallery, spec = generator_spec(args)
    queries, paths, ids = selected_protocol(args, scope, gallery)
    progress = read_progress(output / "adversarial-gallery", spec)
    if progress["done"] < len(paths):
        raise ValueError("先生成整个评测图库，不能混用 clean/adv gallery")
    if scope != "smoke" and progress["done"] != 3074:
        raise ValueError("正式 baseline 必须攻击全部 3074 张图库图像")
    if not torch.cuda.is_available():
        raise RuntimeError("IRRA 评测需要 CUDA")
    metadata = victim_metadata(args, spec, progress, len(paths))
    holder = []
    def get_victim():
        if not holder:
            holder.append(FrozenIRRA(args.irra_repo, args.irra_checkpoint,
                                     args.irra_config, device="cuda"))
        return holder[0]
    g = gallery_features(args, output, spec, len(paths), metadata, get_victim)
    q = query_features(output, scope, queries, metadata,
                       args.text_batch_size, get_victim)
    per_arm = {}
    for name in ("clean", "adversarial"):
        per_arm[name] = []
        for start in range(0, len(queries), args.text_batch_size):
            with torch.no_grad():
                scores = q[start:start + args.text_batch_size].to("cuda") @ g[name].to("cuda").T
                _, rows = text_to_image_metrics(
                    scores, [x.person_id for x in
                             queries[start:start + args.text_batch_size]], ids)
            per_arm[name].extend(rows)
    rows = [{
        "row_id": query.row_id, "person_id": query.person_id,
        "text_unchanged": True,
        "clean": per_arm["clean"][i], "adversarial": per_arm["adversarial"][i],
        "delta_rank": per_arm["adversarial"][i]["first_hit_rank"] -
                      per_arm["clean"][i]["first_hit_rank"],
    } for i, query in enumerate(queries)]
    directory = output / "evaluations" / scope
    report_metadata = {
        "victim": metadata, "generator": spec,
        "gallery_count": len(paths), "gallery_complete": progress["done"] == 3074,
        "query_count": len(queries), "query_row_ids": [q.row_id for q in queries],
        "query_manifest_sha256": sha256(args.query_manifest) if scope == "500" else None,
        "attack_modality": "image/gallery", "retrieval_task": "Text-to-Image Person Search",
        "victim_information_used_during_generation": "none",
        "mean_linf_native": sum(progress["linf"][:len(paths)]) / len(paths),
        "max_linf_native": max(progress["linf"][:len(paths)]),
        "mean_linf_victim": sum(g["linf_victim"]) / len(paths),
        "max_linf_victim": max(g["linf_victim"]),
        "AP_INP_per_query_unit": "fraction",
    }
    write_query_rows(directory / "per-query.jsonl", rows)
    atomic_json(directory / "metadata.json", report_metadata)
    return make_summary(output, scope)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "evaluate", "run",
                                            "smoke", "summary", "status"))
    parser.add_argument("--scope", choices=("500", "full"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--ap-source", type=Path, default=DEFAULT_AP)
    parser.add_argument("--g-checkpoint", type=Path, default=DEFAULT_G)
    parser.add_argument("--g-sha256", default=DEFAULT_G_SHA)
    parser.add_argument("--g-origin", choices=("local_reproduction",
                                               "author_pretrained_verified"),
                        default="local_reproduction")
    parser.add_argument("--generator-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--query-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--irra-repo", type=Path, default=IRRA_REPO)
    parser.add_argument("--irra-checkpoint", type=Path, default=IRRA_CHECKPOINT)
    parser.add_argument("--irra-config", type=Path, default=IRRA_CONFIG)
    parser.add_argument("--max-images", type=int,
                        help="生成图库前 N 张；N 须为 generator batch size 的整数倍")
    parser.add_argument("--smoke-gallery", type=int, default=4)
    parser.add_argument("--smoke-queries", type=int, default=4)
    args = parser.parse_args()
    if min(args.generator_batch_size, args.text_batch_size) < 1:
        parser.error("batch size 必须为正数")
    if args.command in {"run", "evaluate", "summary"} and args.scope is None:
        parser.error("正式评测/汇总必须显式选择 --scope 500 或 full")
    if args.command != "generate" and args.max_images is not None:
        parser.error("--max-images 只用于 generate")
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images 必须为正数")
    output = write_path(args.output_dir)
    if args.command == "smoke":
        if not (1 <= args.smoke_gallery <= 16 and 1 <= args.smoke_queries <= 32):
            parser.error("smoke 最多 16 图 / 32 query")
        output = write_path(output / "smoke")
        args.generator_batch_size = args.smoke_gallery
        generate(args, output, args.smoke_gallery)
        evaluate(args, output, "smoke")
    elif args.command == "generate":
        generate(args, output, args.max_images)
    elif args.command == "run":
        generate(args, output)
        evaluate(args, output, args.scope)
    elif args.command == "evaluate":
        evaluate(args, output, args.scope)
    elif args.command == "summary":
        make_summary(output, args.scope)
    else:
        _, spec = generator_spec(args)
        progress = read_progress(output / "adversarial-gallery", spec)
        print(json.dumps({"gallery_cached": progress["done"], "gallery_target": 3074,
                          "evaluations": [p.parent.name for p in
                           (output / "evaluations").glob("*/summary.json")]},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
