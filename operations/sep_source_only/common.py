"""固定 Smoke 契约、输入投影和独立产物。"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import subprocess
import time

from tbps.artifacts import ArtifactStore, checked, digest, json_bytes, sha_file
from tbps.baselines.cpgc_sep.dataset import SPECS, require
from tbps.baselines.cpgc_sep.runtime import ref
from tbps.contracts import DatasetManifest, FileRef, GalleryItem, PersonID, QueryItem

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/framework/sep_source_only_smoke.server.json"
PROTOCOL = "SEP-TBPS-SRefPseudoPID-S1"
TARGETS = {"cuhk-rasa": ("cuhk", "rasa"), "cuhk-aptm": ("cuhk", "aptm"),
           "rstp-irra": ("rstp", "irra"), "icfg-irra": ("icfg", "irra")}
BASE = "769648d0b8330e7a9bf93a096abac1397fb13940"
IRRA_SHA = "9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"


def read(path):
    return json.loads(Path(path).read_text())


def pinned(reference):
    return checked(FileRef(**reference))


def settings():
    c = read(CONFIG)
    base = ROOT / c["baseline_config"]["path"]
    aptm = ROOT / c["aptm_config"]["path"]
    checked(FileRef(str(base), c["baseline_config"]["sha256"]))
    checked(FileRef(str(aptm), c["aptm_config"]["sha256"]))
    return c, read(base), aptm


def source_config(base):
    model = base["victims"]["irra"]
    source = {**model, "datasets": {"cuhk": model["datasets"]["cuhk"]}}
    require(source["datasets"]["cuhk"]["checkpoint"]["sha256"] == IRRA_SHA,
            "locked CUHK Source IRRA")
    return {"victims": {"irra": source}, "runtime_versions": {"victim": base["runtime_versions"]["victim"]},
            "official": {"sep": base["official"]["sep"]}, "nltk_root": base["nltk_root"],
            "resources": {k: base["resources"][k] for k in ("sep_vectors", "sep_neighbors")}}


def source_store(out):
    return ArtifactStore(Path(out) / "source", new_base=out)


def store_at(out, name):
    return ArtifactStore(Path(out) / name, new_base=out)


def artifact_json(store, key, filename):
    store.verify(key)
    return read(store.root / key / filename)


def publish(store, key, **fields):
    return store.publish(key, {f"{name}.json": json_bytes(value) for name, value in fields.items()},
                         {"protocol": PROTOCOL, "scope": "smoke"})


def progress(control, event, **fields):
    path = Path(control) / "progress.jsonl"
    with path.open("ab") as stream:
        stream.write(json_bytes({"at": time.time(), "event": event, **fields}) + b"\n")


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("xb") as f:
        f.write(json_bytes(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def image_path(root, rel):
    base = Path(root).resolve() / "imgs"
    path = (base / rel).resolve()
    require(path.is_relative_to(base) and path.is_file() and path.stat().st_size > 0,
            "invalid official image path")
    return path


def gallery_item(ds, root, path_key, i, row, namespace, split):
    rel = row[path_key]
    return GalleryItem(f"{ds}:{i}:0", rel, sha_file(image_path(root, rel)),
                       PersonID(namespace, str(row["id"])), split)


def preregister(c, base):
    """此准备进程持有原标注，生成进程只接收文本与 Train-only 投影。"""
    source_rows, eval_sets, captions = [], {}, []
    for ds, entries in c["samples"].items():
        cfg = base["datasets"][ds]
        pinned(cfg["annotation"])
        _, _, path_key, ng, nq, npid = SPECS[ds]
        rows = read(cfg["annotation"]["path"])
        require(all(r["split"] in {"train", "val", "test"} for r in rows), "unexpected split")
        test = [(i, r) for i, r in enumerate(rows) if r["split"] == "test"]
        require((len(test), sum(len(r["captions"]) for _, r in test),
                 len({r["id"] for _, r in test})) == (ng, nq, npid), "official Test population")
        require(len({r[path_key] for _, r in test}) == ng, "duplicate Test paths")
        ids = {split: {r["id"] for r in rows if r["split"] == split} for split in ("train", "val", "test")}
        require(not (ids["train"] & ids["test"] or ids["val"] & ids["test"] or ids["train"] & ids["val"]),
                "official split PID overlap")
        if ds == "cuhk":
            train = [(i, r) for i, r in enumerate(rows) if r["split"] == "train"]
            train_paths = {r[path_key] for _, r in train}
            require((len(train), len(train_paths), len(ids["train"])) == (34054, 34054, 11003)
                    and not train_paths.intersection(r[path_key] for r in rows if r["split"] != "train"),
                    "complete isolated CUHK official Train")
            source_rows = [gallery_item(ds, cfg["root"], path_key, i, r, "cuhk_train", "train")
                           for i, r in train]
        # Preserve annotation order. The fixed early gallery is extended by preregistered
        # image IDs by displacing its last entries; relevance never enters generation.
        by_index = dict(test)
        selected = {entry["row"] for entry in entries}
        require(selected <= by_index.keys(), "non-Test sample")
        require(len({(e["row"], e["caption"]) for e in entries}) == len(entries) == 2,
                "two preregistered captions per dataset")
        if ds == "cuhk":
            earlier = {(i, j) for i, r in test for j in range(len(r["captions"]))}
            earlier = set(sorted(earlier)[:45])
            require(not earlier.intersection((e["row"], e["caption"]) for e in entries),
                    "old stopped Query scope is frozen")
        chosen = list(selected) + [i for i, _ in test if i not in selected][:128 - len(selected)]
        namespace = "rstp_reid" if ds == "rstp" else ds + "_pedes"
        gallery = tuple(gallery_item(ds, cfg["root"], path_key, i, by_index[i], namespace, "test")
                        for i in sorted(chosen))
        queries = []
        for n, entry in enumerate(entries):
            i, j = entry["row"], entry["caption"]
            row = by_index[i]
            raw = row["captions"][j]
            key = f"{ds}-{n}"
            require(type(raw) is str and bool(raw.strip()), "raw caption")
            queries.append(QueryItem(key, raw, j, f"{ds}:{i}:0", PersonID(namespace, str(row["id"])), "test"))
            # Opaque ID, raw text, and no other annotation fields.
            captions.append({"key": key, "caption": raw})
        eval_sets[ds] = asdict(DatasetManifest(namespace, "test", FileRef(**cfg["annotation"]), gallery, tuple(queries)))
    source_hashes = {x.image_sha256 for x in source_rows}
    require(not any(x["image_sha256"] in source_hashes for m in eval_sets.values() for x in m["gallery"]),
            "Source Train / Smoke gallery exact image bytes overlap")
    return [asdict(x) for x in source_rows], captions, eval_sets


def train_manifest(rows):
    gallery = tuple(GalleryItem(**{**r, "pid": PersonID(**r["pid"])}) for r in rows)
    require(len(gallery) == 34054 and len({g.pid for g in gallery}) == 11003
            and all(g.split == "train" and g.pid.dataset == "cuhk_train" for g in gallery),
            "complete Source Train manifest")
    return SimpleNamespace(gallery=gallery)


def eval_manifest(data):
    def rows(kind, constructor):
        return tuple(constructor(**{**r, "pid": PersonID(**r["pid"])}) for r in data[kind])
    return DatasetManifest(data["dataset_id"], "test", FileRef(**data["annotation"]),
                           rows("gallery", GalleryItem), rows("queries", QueryItem))


def checked_runtime(config):
    from importlib.metadata import version
    expected = config["runtime_versions"]["victim"]
    require(all(version(k) == v for k, v in expected.items()), "native runtime versions")


def code_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def seed():
    import random
    import numpy as np
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def bits(x):
    import torch
    return x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_sha(x):
    return hashlib.sha256(bits(x)).hexdigest()
