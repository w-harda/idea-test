"""只读验证既有100句、clean cache、annotation及冻结SHA。"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import torch
from pid_metrics import MODES

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
SOURCE = REPO / "prototypes/attribute_tta_selection_value_20261005/outputs/diagnostic"
GALLERY = Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt")
ANNOTATION = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
METRIC_SOURCE = REPO / "src/attributes/tbps_retrieval_metrics.py"

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

def verify(hashes):
    for name, expected in hashes.items():
        if digest(name) != expected:
            raise ValueError("SHA changed: " + name)

def own_hashes():
    return {str(p): digest(p) for p in sorted(ROOT.glob("*.py"))} | {str(METRIC_SOURCE): digest(METRIC_SOURCE)}

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def checked_inputs():
    protocol = load(SOURCE / "protocol.json")
    completed = load(SOURCE / "completed.json")
    if not completed["method_audit_passed"] or completed["case_count"] != 100 or not completed["diagnostic_run"]:
        raise ValueError("completed100 diagnostic required")
    if protocol["formal_shared_adversarial_gallery"] or protocol["gallery_count"] != 34054:
        raise ValueError("unexpected source protocol")
    if protocol["modes"] != list(MODES) or protocol["steps_per_round"] != 10 or protocol["tau"] != 0.02:
        raise ValueError("frozen modes/steps/tau mismatch")
    locked = dict(protocol["frozen_sha256"])
    for name in ("protocol.json", "completed.json", "audit.json", "selection_manifest.json", "per_caption_results.json"):
        p = SOURCE / name
        locked[str(p)] = digest(p)
    locked.update(own_hashes())
    print("VERIFY_FROZEN_SHA", len(locked), flush=True)
    verify(locked)
    cases = load(SOURCE / "per_caption_results.json")
    manifest = load(SOURCE / "selection_manifest.json")
    if len(cases) != 100 or len({c["caption_id"] for c in cases}) != 100 or [c["caption_id"] for c in cases] != [r["caption_id"] for r in manifest["rows"]]:
        raise ValueError("fixed100 manifest order changed")
    raw = load(ANNOTATION)
    gallery = torch.load(GALLERY, map_location="cpu", weights_only=False)
    spec, features = gallery["spec"], gallery["features"]
    if spec["split"] != "train" or features.shape != (34054, 512) or features.dtype != torch.float32:
        raise ValueError("invalid frozen train feature cache")
    if str(torch.__version__) != spec["torch_version"]:
        raise ValueError("torch version differs from original diagnostic")
    for key, path in [("annotation_sha256", ANNOTATION),
                      ("checkpoint_sha256", Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth"))]:
        if spec[key] != locked[str(path)]:
            raise ValueError("cache provenance mismatch: " + key)
    official = Path("/home/lzf/Attack/IRRA-main/IRRA-main")
    for rel, expected in spec["source_sha256"].items():
        if digest(official / rel) != expected:
            raise ValueError("official source mismatch: " + rel)
    audit_spec = dict(spec)
    paths = audit_spec.pop("gallery_paths")
    audit_spec["gallery_paths_sha256"] = hashlib.sha256(json.dumps(paths).encode()).hexdigest()
    if audit_spec != protocol["gallery"]:
        raise ValueError("cache metadata differs from source diagnostic")
    if len(paths) != len(set(paths)) or not torch.isfinite(features).all() or not torch.allclose(features.norm(dim=1), torch.ones(len(paths)), atol=2e-4, rtol=0):
        raise ValueError("gallery paths/features invalid")
    train = {r["file_path"]: r["id"] for r in raw if r["split"] == "train"}
    if len(train) != len(paths) or set(train) != set(paths):
        raise ValueError("gallery does not cover exactly the train image paths")
    pids = torch.tensor([train[p] for p in paths], dtype=torch.long)
    index = {p: i for i, p in enumerate(paths)}
    grouped = defaultdict(list)
    image_hashes = defaultdict(lambda: {m: {} for m in MODES})
    compact = []
    for c, row in zip(cases, manifest["rows"]):
        _, image_id, caption_id = c["caption_id"].split(":")
        record = raw[int(image_id)]
        if record["split"] != "train" or record["captions"][int(caption_id)] != c["caption"] or record["id"] != c["pid"]:
            raise ValueError("caption/PID/raw annotation mismatch: " + c["caption_id"])
        paired_path = record["file_path"]
        paired = index[paired_path]
        if c["selection_ids"] != row["selection_ids"] or c["natural_k"] != row["natural_k"]:
            raise ValueError("selection changed")
        for mode in MODES:
            result = c["modes"][mode]
            if result["fixed_sequence"] != c["selection_ids"][mode] or len(result["states"]) != c["natural_k"] + 1:
                raise ValueError("sequence/state count mismatch")
            if len(result["rounds"]) != c["natural_k"] or any(len(r["steps"]) != 10 for r in result["rounds"]):
                raise ValueError("not completed10-step rounds")
        clean = c["modes"][MODES[0]]["states"][0]
        if any(c["modes"][m]["states"][0] != clean for m in MODES):
            raise ValueError("different clean state across modes")
        grouped[paired_path].append(c["caption_id"])
        for mode in MODES:
            image_hashes[paired_path][mode][c["caption_id"]] = c["modes"][mode]["rounds"][-1]["post_projection_image_sha256"]
        compact.append({"caption_id": c["caption_id"], "caption": c["caption"], "pid": c["pid"],
            "natural_k": c["natural_k"], "paired_path": paired_path, "paired_index": paired,
            "clean": clean, "final": {m: c["modes"][m]["states"][-1] for m in MODES},
            "selection_ids": c["selection_ids"]})
    query_identity = [[c["caption_id"], c["caption"], c["pid"]] for c in compact]
    contract = {"task": "TBPS text-to-image", "dataset": "CUHK-PEDES",
        "query_scope": "fixed100 held-out reader-validation captions drawn from original train",
        "gallery_scope": "34054 clean train images; query-specific one-image replacement",
        "protocol_kind": "paired_replacement_PID_diagnostic",
        "relevance": "all gallery images with same PID; no camera exclusion",
        "query_count": len(compact), "query_set_sha256": canonical_sha(query_identity),
        "gallery_layout_sha256": canonical_sha(list(zip(paths, pids.tolist()))),
        "attack_set_id": locked[str(SOURCE / "per_caption_results.json")],
        "modes": list(MODES), "shared_adversarial_gallery": False}
    audit = {"source_method_audit_passed": True, "query_count": len(compact),
        "gallery_count": len(paths), "distinct_query_PIDs": len({c["pid"] for c in compact}),
        "distinct_attacked_source_images": len(grouped),
        "source_images_with_multiple_queries": {p: ids for p, ids in grouped.items() if len(ids) > 1},
        "multiple_query_source_image_final_SHA": {p: h for p, h in image_hashes.items() if len(grouped[p]) > 1},
        "different_final_image_for_same_source_by_mode": {m: sum(len(set(h[m].values())) > 1 for h in image_hashes.values()) for m in MODES},
        "attacked_source_image_coverage": len(grouped) / len(paths),
        "query_conditioned_attack": True, "shared_adversarial_gallery": False,
        "adversarial_pixels_saved_in_source_output": False,
        "adversarial_image_features_saved_in_source_output": False,
        "reconstruction": "recompute clean text-vs-gallery scores, replace paired score with saved final cosine",
        "not_formal_test_evaluation": True,
        "K_distribution": dict(Counter(c["natural_k"] for c in compact)),
        "all_same_PID_images_are_positive": True, "source_sha_verified": True}
    return compact, features, pids, spec, locked, contract, audit
