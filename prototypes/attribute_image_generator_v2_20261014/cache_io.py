"""仅写G-v2独立outputs；逐样本目录原子发布，完成缓存不覆盖。"""
import hashlib, json, os, shutil, uuid, fcntl
from contextlib import contextmanager
from pathlib import Path
import torch
from condition_schema import validate, IDENTITY
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""): h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
        separators=(",",":"), allow_nan=False).encode()).hexdigest()

def safe(path):
    p = Path(path).resolve()
    if not p.is_relative_to(OUTPUTS.resolve()): raise ValueError("write must remain inside G-v2 outputs")
    if not OUTPUTS.resolve().is_relative_to(ROOT.resolve()): raise ValueError("outputs symlink escapes G-v2")
    return p

def verify(locks):
    for name, expected in locks.items():
        if sha(name) != expected: raise ValueError("SHA mismatch: " + name)

def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def save_json(path, value):
    p = safe(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = safe(p.parent / (".pending-" + uuid.uuid4().hex))
    try:
        with tmp.open("x", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p); fsync_dir(p.parent)
    finally:
        if tmp.exists(): tmp.unlink()

@contextmanager
def output_lock(path):
    p = safe(path); p.mkdir(parents=True, exist_ok=True)
    with (p / ".prepare.lock").open("a") as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError("another G-v2 preparation is active")
        try: yield p
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def read_entry(entry, protocol_hash=None):
    verify({entry["tensor_path"]:entry["tensor_sha256"], entry["metadata_path"]:entry["metadata_sha256"]})
    meta = load(entry["metadata_path"])
    value = torch.load(entry["tensor_path"], map_location="cpu", weights_only=True)
    report = validate(value, entry)
    if any(meta.get(k) != value[k] for k in IDENTITY): raise ValueError("metadata identity mismatch")
    if (meta.get("K") != value["K"] or entry.get("K") != value["K"]
        or meta.get("selected_instance_ids") != value["selected_instance_ids"]
        or meta.get("tensor_sha256") != entry["tensor_sha256"]): raise ValueError("metadata/K/ID mismatch")
    if protocol_hash is not None and meta.get("protocol_hash") != protocol_hash:
        raise ValueError("sample protocol mismatch")
    if meta.get("aggregate_reconstruction") != report: raise ValueError("aggregate audit report mismatch")
    selected = meta["source_selection"]["selected_attributes"]
    if ([a["instance_id"] for a in selected] != value["selected_instance_ids"]
        or [a["S_raw"] for a in selected] != value["S_raw"].tolist()
        or meta["source_selection"]["natural_k"] != value["K"]): raise ValueError("frozen selection changed")
    return value, meta, report

def entry_at(directory):
    d = Path(directory); meta = load(d / "metadata.json")
    return {**{k:meta[k] for k in IDENTITY}, "K":meta["K"],
            "tensor_path":str(d/"condition.pt"), "tensor_sha256":meta["tensor_sha256"],
            "metadata_path":str(d/"metadata.json"), "metadata_sha256":sha(d/"metadata.json")}

def write_entry(output, value, metadata):
    report = validate(value)
    out = safe(output); target = safe(out/"samples"/value["row_id"].replace(":","_"))
    if target.exists(): raise ValueError("published sample exists; use checked resume")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = safe(target.parent/(".pending-"+uuid.uuid4().hex)); stage.mkdir()
    try:
        p = stage/"condition.pt"
        with p.open("xb") as f:
            torch.save(value, f); f.flush(); os.fsync(f.fileno())
        meta = {**metadata, **{k:value[k] for k in IDENTITY}, "K":value["K"],
                "selected_instance_ids":value["selected_instance_ids"],
                "tensor_sha256":sha(p), "aggregate_reconstruction":report}
        save_json(stage/"metadata.json", meta); fsync_dir(stage)
        os.rename(stage, target); fsync_dir(target.parent)
    finally:
        if stage.exists(): shutil.rmtree(safe(stage))
    entry = entry_at(target); read_entry(entry, metadata["protocol_hash"])
    return entry

def checked_cache(path):
    out = Path(path); done = load(out/"completed.json")
    verify({str(out/"manifest.json"):done["manifest_sha256"],
            str(out/"summary.json"):done["summary_sha256"]})
    m = load(out/"manifest.json")
    verify({m["protocol_path"]:m["protocol_sha256"]})
    protocol = load(m["protocol_path"]); verify(protocol["source_sha256"])
    if not done.get("method_audit_passed") or m["schema"] != "gv2-condition-manifest-1":
        raise ValueError("incomplete G-v2 cache")
    entries = m["entries"]
    if len({e["row_id"] for e in entries}) != len(entries) or len(entries) != m["count"]:
        raise ValueError("duplicate/missing sample")
    if [{k:e[k] for k in (*IDENTITY,"K")} for e in entries] != protocol["samples"]:
        raise ValueError("manifest/protocol coverage mismatch")
    pids = {role:{e["pid"] for e in entries if e["role"] == role} for role in ("train","validation")}
    if pids["train"] & pids["validation"]: raise ValueError("PID leakage")
    for e in entries: read_entry(e, digest(protocol))
    return m
