"""Smoke 的分进程 GPU 生产及独立 CPU 审计。"""
from dataclasses import asdict
from functools import partial
from pathlib import Path
import json
import os
import sys
import time

from tbps.artifacts import digest, json_bytes, sha_file
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.runtime import ref, tensor_bytes
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan, prepare_caption
from tbps.baselines.cpgc_sep.sep_source import classifier
from tbps.baselines.cpgc_sep.victims import VictimSource
from tbps.engines.metrics import ranking
from tbps.models.irra_io import official_tokenizer

from . import attack, isolation, native, parity
from .common import (TARGETS, artifact_json, eval_manifest,
                     progress, publish, read, seed, settings, source_store,
                     store_at, tensor_sha, train_manifest)


def input_bundle(out):
    store = store_at(out, "preflight")
    return artifact_json(store, "source-input", "bundle.json"), store.root / "source-input"


def eval_sets(out):
    return artifact_json(store_at(out, "preflight"), "evaluation-input", "manifests.json")


def cpu():
    import torch
    require(not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "independent CPU subprocess must hide CUDA")


def token_receipt(tokenizer, texts):
    tokens, reports = tokenizer.batch(texts, truncate=True)
    require(list(tokens.shape) == [len(texts), 77], "native IRRA token shape")
    return {"token_sha256": tensor_sha(tokens), "reports": json.loads(json_bytes([asdict(r) for r in reports]))}


def parity_texts(manifest):
    clean = manifest.queries[0].caption
    other = manifest.queries[1].caption
    words = clean.split()
    return [clean, " ".join(words[1:]), " ".join(words[:-1]), other]


def gpu_parity(c, base):
    import torch
    out, ctl = c["output"], c["control"]
    manifest = eval_manifest(eval_sets(out)["cuhk"])
    texts = parity_texts(manifest)
    require(len(set(texts)) == len(texts), "distinct parity candidates")
    matrices, evidence = {}, {}
    report = partial(progress, ctl, purpose="parity")
    for name in ("irra", "rasa"):
        start = time.monotonic()
        with native.native(base, name, "cuhk") as (adapter, model, src):
            before = native.state_sha(model)
            bank = native.Gallery(adapter, name, base["datasets"]["cuhk"]["root"], manifest.gallery, report)
            scores, pairs = native.score(adapter, name, bank, texts)
            require(name != "rasa" or pairs == len(texts)*128, "RaSa Top128 coverage")
            matrices[name] = scores
            require(before == native.state_sha(model), "parity model changed")
            evidence[name] = {"checkpoint": asdict(src.checkpoint), "model_state_sha256": before,
                              "text_forwards": len(texts), "image_forwards": len(bank.ids),
                              "fusion_pairs": pairs, "seconds": time.monotonic()-start}
        del adapter, model, bank
        torch.cuda.empty_cache()
    store = store_at(out, "parity")
    store.publish("native-scores", {"scores.pt": tensor_bytes(matrices), "evidence.json": json_bytes(evidence),
                                    "texts.json": json_bytes(texts)}, {"kind": "immutable native FP32 scores"},
                  sources=(ref(Path(out)/"preflight/evaluation-input/manifest.json"),))


def cpu_parity(c, base):
    import torch
    cpu()
    out = c["output"]
    store = store_at(out, "parity")
    store.verify("native-scores")
    root = store.root / "native-scores"
    manifest = eval_manifest(eval_sets(out)["cuhk"])
    texts = read(root/"texts.json")
    require(texts == parity_texts(manifest), "parity candidate stream changed")
    matrices = torch.load(root/"scores.pt", map_location="cpu", weights_only=True)
    require(set(matrices) == {"irra", "rasa"}, "exact two-model score capture")
    report = parity.compare(base, manifest, texts, matrices)
    store.publish("exact", {"result.json": json_bytes(report)}, {"kind": "CPU bitwise parity"},
                  sources=(ref(root/"manifest.json"),))


def guard(bundle, path, out, control):
    manifest = train_manifest(bundle["train_gallery"])
    work = Path(out)/"source"
    work.mkdir(exist_ok=True)
    scratch = work/"scratch"
    scratch.mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(scratch), CUDA_CACHE_PATH=str(scratch), XDG_CACHE_HOME=str(scratch))
    # Only this exact progress file is writable outside the new Source output.
    log = Path(control)/"progress.jsonl"
    log.touch(exist_ok=True)
    abi = isolation.restrict_source(isolation.source_paths(bundle, manifest, path), [work, log])
    isolation.must_deny([Path(out)/"preflight/evaluation-input/manifests.json"])
    return manifest, abi


def generate(c, out, ctl):
    import nltk
    import torch
    bundle, path = input_bundle(out)
    manifest, abi = guard(bundle, path, out, ctl)
    config = bundle["config"]
    for asset, expected in bundle["nltk_files"].items():
        require(sha_file(asset) == expected, "frozen SEP POS asset SHA")
    require(set(config["victims"]) == {"irra"} and set(config["victims"]["irra"]["datasets"]) == {"cuhk"},
            "generation must have CUHK Source model capability only")
    require(all(set(row) == {"key", "caption"} for row in bundle["captions"]),
            "Caption-only generation input")
    nltk.data.path[:] = [config["nltk_root"]]
    source = source_store(out)
    binding = ref(path/"manifest.json")
    seed()
    torch.cuda.manual_seed_all(0)
    vocabulary = attack.resources(config)
    with native.native(config, "irra", "cuhk") as (adapter, model, src):
        before = native.state_sha(model)
        bank_start = time.monotonic()
        report = partial(progress, ctl, purpose="source")
        bank = native.Gallery(adapter, "irra", bundle["train_root"], manifest.gallery, report)
        gallery_evidence = {"images": len(bank.ids), "image_forwards": len(bank.ids),
                            "features_sha256": tensor_sha(bank.globals),
                            "pid_count": len({g.pid for g in manifest.gallery}),
                            "train_manifest_sha256": digest(bundle["train_gallery"]),
                            "seconds": time.monotonic()-bank_start,
                            "checkpoint": asdict(src.checkpoint), "state_sha256": before, "landlock_abi": abi}
        source.publish("gallery", {"evidence.json": json_bytes(gallery_evidence),
                                   "features.pt": tensor_bytes(bank.globals)}, {"kind": "Source Train bank"},
                       sources=(binding,))
        predictor = classifier(adapter, model, "irra", bank, manifest)
        completed = []
        for item in bundle["captions"]:
            key = item["key"]
            item_report = partial(report, query=key)
            item_report("QUERY_BEGIN")
            publish(source, f"claim-{key}", claim={"key": key, "input_sha256": digest(item)})
            require(not predictor.cache, "cache reset at Caption boundary")
            with attack.deadline(c["source_query_timeout_seconds"]):
                row, logits = attack.attack_caption(config, item["caption"], predictor, "cuda:0",
                                                   vocabulary, item_report)
            row.update(key=key, cache_keys=list(predictor.cache),
                       tokenizer=token_receipt(adapter._tokenizer, [row["matched_clean"], row["selected"]]))
            predictor.cache.clear()
            source.publish(key, {"text.json": json_bytes(row), "clean_pid_logits.pt": tensor_bytes(logits)},
                           {"kind": "frozen source-only adversarial text", "key": key},
                           sources=(binding, ref(source.root/"gallery/manifest.json")))
            completed.append({"key": key, "text_sha256": digest(row["adversarial"]),
                              "manifest_sha256": sha_file(source.root/key/"manifest.json")})
            item_report("QUERY_FROZEN", seconds=row["elapsed_seconds"], noop=row["noop"],
                        text_forwards=row["text_forwards"], candidates=len(row["candidates"]))
        require(before == native.state_sha(model), "Source IRRA weights or buffers changed")
    source.publish("completed", {"index.json": json_bytes(completed)}, {"kind": "frozen all six source texts"},
                   sources=(binding, *(ref(source.root/x["key"]/"manifest.json") for x in completed)))


def frozen_texts(out, manifest):
    store = source_store(out)
    records = artifact_json(store, "completed", "index.json")
    index = {x["key"]: x for x in records}
    adv = []
    for q in manifest.queries:
        row = artifact_json(store, q.query_id, "text.json")
        require(row["key"] == q.query_id and row["original"] == q.caption
                and digest(row["adversarial"]) == index[q.query_id]["text_sha256"]
                and sha_file(store.root/q.query_id/"manifest.json") == index[q.query_id]["manifest_sha256"],
                "frozen text index/official raw Caption mismatch")
        adv.append(row["adversarial"])
    return adv


def target(c, base, aptm, key):
    import torch
    out, ctl = c["output"], c["control"]
    # Freeze verification must occur before any Target model is loaded.
    ds, name = TARGETS[key]
    manifest = eval_manifest(eval_sets(out)[ds])
    adv = frozen_texts(out, manifest)
    raw = [q.caption for q in manifest.queries]
    report = partial(progress, ctl, purpose="target", target=key)
    started = time.monotonic()
    with native.native(base, name, ds, aptm) as (adapter, model, src):
        before = native.state_sha(model)
        bank = native.Gallery(adapter, name, base["datasets"][ds]["root"], manifest.gallery, report)
        clean_scores, clean_pairs = native.score(adapter, name, bank, raw)
        adv_scores, adv_pairs = native.score(adapter, name, bank, adv)
        pairs = clean_pairs + adv_pairs
        expected_pairs = 0 if name == "irra" else len(raw+adv)*128
        require(pairs == expected_pairs and before == native.state_sha(model), "target fusion/frozen invariant")
        scores = {"Clean": clean_scores, "Text-only": adv_scores}
        require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in scores.values()),
                "finite native score")
    evidence = {"model": name, "dataset": ds, "checkpoint": asdict(src.checkpoint), "state_sha256": before,
                "image_forwards": len(bank.ids), "text_forwards": len(raw+adv), "fusion_pairs": pairs,
                "elapsed_seconds": time.monotonic()-started, "adv_text_sha256": [digest(x) for x in adv],
                "scores_sha256": {k: tensor_sha(v) for k,v in scores.items()}}
    store_at(out, "target").publish(key, {"scores.pt": tensor_bytes(scores), "evidence.json": json_bytes(evidence)},
                                    {"kind": "independent Target smoke scores"},
                                    sources=(ref(Path(out)/"source/completed/manifest.json"),
                                             ref(Path(out)/"preflight/evaluation-input/manifest.json")))


def audit(c, base, aptm):
    import nltk
    import torch
    cpu()
    out = c["output"]
    bundle, path = input_bundle(out)
    config = bundle["config"]
    source = source_store(out)
    gallery = artifact_json(source, "gallery", "evidence.json")
    require(gallery["landlock_abi"] >= 3 and gallery["train_manifest_sha256"] == digest(bundle["train_gallery"])
            and gallery["checkpoint"] == asdict(VictimSource(config, "irra", "cuhk").checkpoint),
            "Source capability / Train manifest binding")
    require(gallery["features_sha256"] == tensor_sha(torch.load(source.root/"gallery/features.pt",
                                                            map_location="cpu", weights_only=True)),
            "Source feature digest")
    plan = PIDPlan.build(tuple(x.pid for x in train_manifest(bundle["train_gallery"]).gallery))
    vocab = attack.resources(config)
    nltk.data.path[:] = [config["nltk_root"]]
    pos = attack.sep_functions(config)["attack"].__globals__["criteria"].get_pos
    tokenizer = official_tokenizer(VictimSource(config, "irra", "cuhk"))
    rows = {}
    completed = artifact_json(source, "completed", "index.json")
    require([x["key"] for x in completed] == [x["key"] for x in bundle["captions"]], "six frozen texts")
    for item in bundle["captions"]:
        key = item["key"]
        row = artifact_json(source, key, "text.json")
        logits = torch.load(source.root/key/"clean_pid_logits.pt", map_location="cpu", weights_only=True)
        require(logits.dtype == torch.float32 and logits.shape == (1, 11003) and torch.isfinite(logits).all(),
                "complete Source clean PID logits")
        top = int(logits[0].argmax())
        require(row["pseudo_pid_index"] == top and row["pseudo_pid"] == plan.order[top].value
                and row["original"] == item["caption"], "frozen Source-only pseudo PID")
        clean = prepare_caption(config, item["caption"])
        choice = row["candidates"][0]["text"] if row["candidates"] else clean
        require(row["matched_clean"] == clean and row["selected"] == choice
                and row["adversarial"] == (item["caption"] if row["noop"] else choice)
                and row["noop"] == (choice == clean)
                and row["source_oracle"][0] == {"texts": [clean], "source_top1_changed": [False]},
                "selection / Source oracle / raw-preserving NOOP")
        tau = [x["tau"] for x in row["candidates"]]
        require(tau == sorted(tau), "official tau order")
        require(row["word_edits"] == attack.verify_edits(clean, choice, vocab, pos(clean.split()))
                and row["tokenizer"] == token_receipt(tokenizer, [clean, choice]), "budget, vocab and tokenizer")
        ordered = [x for request in row["requests"] for x in request["texts"]]
        require(list(dict.fromkeys(ordered)) == row["cache_keys"] and len(ordered) == row["calls"]
                and row["calls"] == row["cache_hits"] + row["text_forwards"]
                and row["text_forwards"] == len(row["cache_keys"]), "candidate stream and cache evidence")
        rows[key] = {k:row[k] for k in ("elapsed_seconds", "text_forwards", "cache_hits", "cache_hit_rate", "noop")}
        rows[key].update(candidates=len(tau), edits=row["word_edits"]["word_change_count"])
    results = {}
    targets = store_at(out, "target")
    manifests = eval_sets(out)
    for key, (ds, name) in TARGETS.items():
        manifest = eval_manifest(manifests[ds])
        adv = frozen_texts(out, manifest)
        ev = artifact_json(targets, key, "evidence.json")
        if name == "aptm":
            from tbps.models.aptm import APTMSource
            checkpoint = APTMSource(aptm).checkpoint
        else:
            checkpoint = VictimSource(base, name, ds).checkpoint
        require(ev["checkpoint"] == asdict(checkpoint) and ev["adv_text_sha256"] == [digest(x) for x in adv],
                "Target identity and shared frozen texts")
        scores = torch.load(targets.root/key/"scores.pt", map_location="cpu", weights_only=True)
        require(set(scores) == {"Clean", "Text-only"}
                and {k:tensor_sha(v) for k,v in scores.items()} == ev["scores_sha256"], "Target score closure")
        results[key] = {"metrics": {k:ranking(v, manifest.evaluation_identity())[0] for k,v in scores.items()},
                        "evidence": ev}
    require(results["cuhk-rasa"]["evidence"]["adv_text_sha256"] ==
            results["cuhk-aptm"]["evidence"]["adv_text_sha256"], "one CUHK frozen text artifact")
    parity_store = store_at(out, "parity")
    exact = artifact_json(parity_store, "exact", "result.json")
    require(exact["status"] == "EXACT", "real model parity")
    publish(store_at(out, "audit"), "completed", report={"status":"PASS", "scope":"smoke", "parity":exact,
            "gallery":gallery, "queries":rows, "targets":results, "CUDA_initialized":torch.cuda.is_initialized()})


def main():
    import torch
    torch.set_num_threads(2)
    phase = sys.argv[1]
    if phase == "source":
        # Source sees only the three CLI scalars and a capability bundle. Config
        # and Target manifests are consumed only by the separate prep/eval processes.
        out, ctl, timeout = sys.argv[2:]
        generate({"source_query_timeout_seconds": int(timeout)}, out, ctl)
        return
    c, base, aptm = settings()
    if phase == "parity-gpu":
        gpu_parity(c, base)
    elif phase == "parity-cpu":
        cpu_parity(c, base)
    elif phase == "target":
        target(c, base, aptm, sys.argv[2])
    elif phase == "audit":
        audit(c, base, aptm)
    else:
        raise ValueError("Smoke phases only")


if __name__ == "__main__":
    main()
