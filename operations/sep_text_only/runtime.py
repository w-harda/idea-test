"""SEP Text-only 的最小数据投影和执行契约，原数值模块只读复用。"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import sys
import time

from tbps.artifacts import checked, digest, json_bytes, sha_file
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.runtime import ref
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from operations.sep_source_only.common import ROOT, atomic, eval_manifest, read, train_manifest

CONFIG = ROOT / "configs/framework/sep_text_only_full.server.json"
DATASETS = ("cuhk", "rstp", "icfg")
VICTIMS = ("irra", "rasa")


def config():
    c = read(CONFIG)
    from tbps.contracts import FileRef
    checked(FileRef(str(ROOT/c["base_config"]), c["base_config_sha256"]))
    return c, read(ROOT/c["base_config"])


def store(c, path):
    return VerifiedArtifactStore(Path(c["output"])/path, new_base=c["output"])


def metadata(c, key, filename):
    s = store(c, "inputs")
    s.verify(key)
    return read(s.root/key/filename)


def event(c, kind, **fields):
    record = {"time":time.time(), "event":kind, **fields}
    with (Path(c["control"])/"progress.jsonl").open("ab") as stream:
        stream.write(json_bytes(record)+b"\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def query_key(dataset, query_id):
    return hashlib.sha256((dataset+"\0"+query_id).encode()).hexdigest()


def source_inputs(c):
    return metadata(c, "source", "source.json")


def full_manifest(c, ds):
    return eval_manifest(metadata(c, "evaluation-"+ds, "manifest-data.json"))


def selected(c, ds, scope):
    if scope == "full":
        return tuple(full_manifest(c, ds).queries)
    require(scope == "smoke", "scope")
    manifest = full_manifest(c, ds)
    keys = set(metadata(c, "selection", "queries.json")[ds])
    return tuple(q for q in manifest.queries if query_key(ds,q.query_id) in keys)


def target_manifest(c, ds, scope):
    from dataclasses import replace
    manifest = full_manifest(c, ds)
    if scope == "full":
        return manifest
    queries = selected(c, ds, scope)
    # Evaluation identity is local to this process and never sent to generation.
    required = {q.paired_image_id for q in queries}
    ids = set(g.image_id for g in manifest.gallery[:128]) | required
    gallery = tuple(g for g in manifest.gallery if g.image_id in ids)
    return replace(manifest, gallery=gallery, queries=queries)


def selected_source(c, ds, scope):
    bundle = metadata(c, "captions-"+ds, "captions.json")
    require(all(set(x) == {"key","caption"} and len(x["key"]) == 64 for x in bundle), "Caption-only inputs")
    if scope == "full":
        return bundle
    keys = set(metadata(c, "selection", "queries.json")[ds])
    return [x for x in bundle if x["key"] in keys]


def source_file(c, ds, key):
    return store(c, "source/"+ds), key


def frozen_rows(c, ds, queries):
    rows = []
    for q in queries:
        key = query_key(ds, q.query_id)
        s, key = source_file(c, ds, key)
        s.verify(key)
        row = read(s.root/key/"text.json")
        require(row["key"] == key and row["original"] == q.caption, "frozen original/query binding")
        rows.append(row)
    return rows


def source_guard(c, bundle):
    """生成子进程只允许 Train 图片和 Source checkpoint。拒绝 Target 标注/权重/身份产物。"""
    train_paths = {str((Path(bundle["train_root"])/"imgs"/x["image_rel_path"]).resolve())
                   for x in bundle["train_gallery"]}
    allowed_checkpoint = Path(bundle["config"]["victims"]["irra"]["datasets"]["cuhk"]["checkpoint"]["path"]).resolve()
    allowed_bootstrap = Path(bundle["config"]["victims"]["irra"]["clip_bootstrap"]["path"]).resolve()
    dataset_root = Path("/home/lzf/TBPS/Datasets").resolve()
    inputs = Path(c["output"])/"inputs"
    protected = (Path("/home/lzf/Attack/IRRA/logs"), Path("/home/lzf/ldx/checkpoints/RaSa"),
                 Path("/home/lzf/ldx/checkpoints/APTM"))
    output = Path(c["output"]).resolve()
    control = Path(c["control"]).resolve()

    def audit(name, args):
        if name != "open" or not isinstance(args[0], (str,bytes,os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1], args[2]
        write = bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)) if isinstance(flags,int) else False
        write = write or (isinstance(mode,str) and any(x in mode for x in "wax+"))
        if write:
            if not (p.is_relative_to(output/"source") or p.is_relative_to(control) or str(p)=="/dev/null"):
                raise PermissionError("Source may only write its new output and progress")
            return
        if p.is_relative_to(dataset_root) and str(p) not in train_paths:
            raise PermissionError("Target annotation/Test image is inaccessible to Source")
        if p.is_relative_to(inputs) and any(x.startswith("evaluation-") for x in p.parts):
            raise PermissionError("evaluation identity is inaccessible to Source")
        if p.suffix in (".pt",".pth") and p not in (allowed_checkpoint,allowed_bootstrap) and not p.is_relative_to(output/"source"):
            raise PermissionError("Only frozen CUHK Source tensors may be opened in generation")
        if p != allowed_checkpoint and any(p.is_relative_to(root) for root in protected):
            raise PermissionError("Target checkpoint is inaccessible to Source")

    # PyTorch/torchvision imports create local Python helpers. Keep these writes
    # inside the new Source output before installing the file-access hook.
    import tempfile
    scratch=output/"source"/"scratch"
    scratch.mkdir(parents=True,exist_ok=True)
    os.environ["TMPDIR"]=str(scratch)
    os.environ["CUDA_CACHE_PATH"]=str(scratch)
    tempfile.tempdir=str(scratch)
    sys.addaudithook(audit)
    return audit


def prepare(c, base):
    from tbps.baselines.cpgc_sep.dataset import read_test
    from tbps.baselines.cpgc_sep.victims import VictimSource
    require(not Path(c["output"]).exists() and not Path(c["control"]).exists(), "fresh Attempt required")
    Path(c["output"]).mkdir()
    Path(c["control"]).mkdir()
    s = store(c, "inputs")
    original = Path(c["source_reference"])
    require(sha_file(original/"bundle.json") == c["source_reference_bundle_sha256"], "fixed Source reference SHA")
    inherited = read(original/"bundle.json")
    require(digest(inherited["train_gallery"]) == c["source_train_manifest_sha256"], "Source Train identity")
    bundle = {k:inherited[k] for k in ("config","train_root","train_gallery","nltk_files")}
    train_manifest(bundle["train_gallery"])
    require(set(bundle["config"]["victims"]) == {"irra"} and
            set(bundle["config"]["victims"]["irra"]["datasets"]) == {"cuhk"}, "single frozen Source capability")
    VictimSource(bundle["config"], "irra", "cuhk").verify()
    selection = {}
    source_hashes = {g["image_sha256"] for g in bundle["train_gallery"]}
    s.publish("source", {"source.json":json_bytes(bundle)}, {"protocol":c["protocol"]},
              sources=(ref(CONFIG),ref(original/"manifest.json"),ref(original/"bundle.json")))
    for ds in DATASETS:
        event(c,"PREPARE_DATASET",dataset=ds)
        manifest = read_test(base["datasets"][ds])
        require(not source_hashes.intersection(g.image_sha256 for g in manifest.gallery), "Source Train/Test image overlap")
        captions = [{"key":query_key(ds,q.query_id),"caption":q.caption} for q in manifest.queries]
        eligible = [(i,x) for i,x in enumerate(captions) if len(x["caption"].split())>=4]
        chosen = sorted(eligible,key=lambda pair:(len(pair[1]["caption"].split()),pair[0]))[:c["smoke_queries"]]
        require(len(chosen)==8,"eight preregistered Smoke queries")
        selection[ds] = [x["key"] for _,x in chosen]
        s.publish("evaluation-"+ds,{"manifest-data.json":json_bytes(asdict(manifest))},
                  {"kind":"evaluation-only identity","dataset":ds},sources=(ref(CONFIG),))
        s.publish("captions-"+ds,{"captions.json":json_bytes(captions)},
                  {"kind":"Caption-only generation input","dataset":ds},sources=(ref(CONFIG),))
        for v in VICTIMS:
            VictimSource(base,v,ds).verify()
    s.publish("selection",{"queries.json":json_bytes(selection)},{"kind":"fixed model-independent Smoke"},
              sources=(ref(CONFIG),))
    event(c,"PREFLIGHT_PASS",source_manifest_sha=c["source_train_manifest_sha256"],
          inherited_cpu_tests=176,new_tests="targeted only",target_models=6,smoke_queries_per_dataset=8)
