"""Linux Landlock 文件访问门禁。只给生成进程 Source 资产的读取权限。"""
import ctypes as c
import os
import platform
from pathlib import Path
import sys

from tbps.baselines.cpgc_sep.dataset import require


def restrict_source(read_paths, write_paths=()):
    require(platform.machine() == "x86_64" and sys.platform == "linux", "Landlock platform")
    libc = c.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    require(abi >= 3, "Landlock ABI 3+ required for Source isolation")
    read = (1 << 0) | (1 << 2) | (1 << 3)  # EXECUTE, READ_FILE, READ_DIR
    writable = ((1 << 15) - 1) & ~read

    class Rule(c.Structure):
        _pack_ = 1
        _fields_ = [("access", c.c_uint64), ("parent", c.c_int32)]

    def success(result):
        if result < 0:
            err = c.get_errno()
            raise OSError(err, os.strerror(err))
        return result

    handled = c.c_uint64(read | writable)
    fd = success(libc.syscall(444, c.byref(handled), c.sizeof(handled), 0))
    try:
        system = [p for p in ("/usr", "/lib", "/lib64", "/etc", "/sys", "/proc") if Path(p).exists()]
        for paths, mask in (([*read_paths, *system, sys.prefix], read),
                            ([*write_paths, "/dev"], read | writable)):
            for path in dict.fromkeys(str(Path(p).resolve(strict=True)) for p in paths):
                with_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    file_mask = mask if Path(path).is_dir() else mask & ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 14))
                    rule = Rule(file_mask, with_fd)
                    success(libc.syscall(445, fd, 1, c.byref(rule), 0))
                finally:
                    os.close(with_fd)
        success(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        success(libc.syscall(446, fd, 0))
    finally:
        os.close(fd)
    return abi


def source_paths(bundle, manifest, bundle_path):
    from .common import ROOT, image_path
    from tbps.baselines.cpgc_sep.victims import VictimSource
    c = bundle["config"]
    model = VictimSource(c, "irra", "cuhk")
    assets = [model.root, model.checkpoint.path, model.clip_bootstrap.path, model.bpe.path,
              model.data["base_config"]["path"], *model.data["files"],
              c["official"]["sep"]["root"], c["nltk_root"],
              *(x["path"] for x in c["resources"].values())]
    return [ROOT, bundle_path, *assets,
            *(image_path(bundle["train_root"], item.image_rel_path) for item in manifest.gallery)]


def must_deny(paths):
    for path in paths:
        try:
            with Path(path).open("rb"):
                raise AssertionError(f"Source could open restricted asset {path}")
        except PermissionError:
            continue
