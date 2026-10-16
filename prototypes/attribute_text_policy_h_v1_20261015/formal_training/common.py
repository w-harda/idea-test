"""稳定来源集合、不可变文件及运行锁；不枚举整个源码目录。"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import signal
import uuid
import torch
from phase2_sources import ROOT, G2, digest, sha, checked, read_json
from real_feature_cache import fsync_dir

BASE = ROOT/"outputs/formal_v1"
MODULES = {
    "data": ("common.py", "data.py"),
    "gallery": ("common.py", "data.py", "storage.py", "runtime.py"),
    "training": ("common.py", "data.py", "storage.py", "runtime.py", "checkpoint.py", "training.py"),
    "validation": ("common.py", "data.py", "storage.py", "runtime.py", "checkpoint.py", "validation.py"),
}
# 原有Phase3显式依赖清单继续严格核验；新增文件不改变历史协议。
def sources(kind):
    result = dict(read_json(ROOT/"config_phase3.json")["source_sha256"])
    from phase2_sources import REPO
    from phase3_inputs import OFFICIAL
    from tokenizer_probe import TOKENIZER_SHA, BPE_SHA
    result.update({str(REPO/p):v for p,v in read_json(ROOT/"config_phase2.json")["frontend_sources"].items()})
    result[str(OFFICIAL/"utils/simple_tokenizer.py")]=TOKENIZER_SHA
    result[str(OFFICIAL/"data/bpe_simple_vocab_16e6.txt.gz")]=BPE_SHA
    for path, expected in result.items():
        checked(path, expected)
    for name in (*MODULES[kind], "cli.py") if kind != "data" else MODULES[kind]:
        path = Path(__file__).parent/name
        result[str(path)] = sha(path)
    for name in ("phase4_training.py", "phase4_checkpoint.py", "phase4_real_reward.py",
                 "real_feedback.py", "real_feature_cache.py", "phase3_inputs.py",
                 "config_phase3.json", "config_phase4.json"):
        path = ROOT/name
        result[str(path)] = sha(path)
    return result

def verify_sources(protocol):
    for path, expected in protocol["source_sha256"].items():
        checked(path, expected)

def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+"\n").encode()

def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != json.loads(json_bytes(value)):
            raise ValueError("immutable JSON differs: "+str(path))
        return
    work = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with work.open("xb") as f:
        f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
    try:
        os.link(work, path)  # 原子、不覆盖；读者永远不会看见半个JSON。
    finally:
        work.unlink()
    fsync_dir(path.parent)

def safe_root(path):
    path = Path(path).resolve()
    if path != BASE.resolve() and BASE.resolve() not in path.parents:
        raise ValueError("formal output must stay under outputs/formal_v1")
    return path

@contextmanager
def run_lock(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root/".lock").open("a") as f:
        try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise ValueError("another writer owns this formal output") from e
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

class Stop:
    """只在样本/缓存单元边界停止，不在forward中抛异常。"""
    def __init__(self):
        self.requested = False
        self.previous = {}
    def __enter__(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: setattr(self, "requested", True))
        return self
    def __exit__(self, *args):
        for sig, handler in self.previous.items(): signal.signal(sig, handler)

def cpu_only():
    if torch.cuda.is_initialized():
        raise ValueError("CPU operation unexpectedly initialized CUDA")
