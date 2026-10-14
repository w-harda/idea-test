"""不可变epoch目录+原子current指针；保留原checkpoint payload与last.pt入口。"""
import contextlib
import json
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path
import torch
import common as c

def _slot(path):
    # 最后一级可能为last.pt别名，不能resolve后覆盖不可变版本。
    path=Path(path)
    return c.safe(path.parent)/path.name

def _store(path):
    return c.safe(path.parent/("."+path.name+".checkpoints"))

def _pointer(path):
    return path.parent/("."+path.name+".current")

def _sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def _sync_file(path):
    with Path(path).open("rb") as f:os.fsync(f.fileno())

def _fault(stage):
    """测试注入点，正式执行不做任何操作。"""

def _read_pair(file,device):
    file=c.safe(file)
    side=c.load(str(file)+".json")
    if c.sha(file)!=side["sha256"]:raise ValueError("checkpoint SHA mismatch: "+str(file))
    payload=torch.load(file,map_location=device,weights_only=True)
    if payload["schema"]!="conditional-G-image-1":raise ValueError("wrong G checkpoint")
    for key in ("epoch","scope","cache_manifest_sha256"):
        if payload[key]!=side[key]:raise ValueError("checkpoint/sidecar metadata mismatch: "+key)
    return payload,side

def resolve_checkpoint(path,device="cpu"):
    """只读选择最近完整有效版本；临时目录永不进入恢复候选。"""
    path=_slot(path);store=_store(path);candidates=[]
    if store.exists():
        for directory in store.iterdir():
            if directory.is_dir() and (directory.name.startswith("epoch_") or directory.name=="legacy"):
                try:
                    meta=c.load(directory/"checkpoint.pt.json")
                    candidates.append((int(meta["epoch"]),directory.name,directory/"checkpoint.pt"))
                except (OSError,ValueError,KeyError,TypeError):
                    warnings.warn("ignoring incomplete checkpoint directory: "+str(directory))
    failures=[]
    for _,_,file in sorted(candidates,reverse=True):
        try:
            payload,side=_read_pair(file,device)
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError) as exc:
            failures.append(str(file)+": "+str(exc));continue
        if failures:warnings.warn("checkpoint rollback after invalid newer version: "+"; ".join(failures))
        return payload,side,file
    if candidates or failures:
        raise ValueError("no valid committed checkpoint: "+"; ".join(failures))
    # 兼容修复前普通.pt+.json；未知SHA/格式错误不能跳过。
    payload,side=_read_pair(path,device)
    return payload,side,path

def _symlink_replace(link,target):
    tmp=link.parent/("."+link.name+".link-"+uuid.uuid4().hex)
    try:
        os.symlink(os.path.relpath(target,link.parent),tmp)
        os.replace(tmp,link)
        _sync_dir(link.parent)
    finally:
        if tmp.is_symlink():tmp.unlink()

def _aliases(path):
    pointer=_pointer(path)
    for link,target in ((path,pointer/"checkpoint.pt"),(Path(str(path)+".json"),pointer/"checkpoint.pt.json")):
        if link.is_symlink() and os.readlink(link)==os.path.relpath(target,link.parent):continue
        _symlink_replace(link,target)
        _fault("after_alias:"+link.name)

def _publish(path,file):
    # 切换current之前两个永久别名均安装好；已有current时它们始终指向同一旧版本。
    _aliases(path)
    _fault("before_pointer")
    _symlink_replace(_pointer(path),file.parent)
    _fault("after_pointer")

def _import_legacy(path):
    store=_store(path);legacy=store/"legacy"
    if _pointer(path).exists() or legacy.exists():return
    if not path.exists():return
    payload,side=_read_pair(path,"cpu")
    pending=store/(".pending-legacy-"+uuid.uuid4().hex);pending.mkdir()
    shutil.copyfile(path,pending/"checkpoint.pt")
    shutil.copyfile(str(path)+".json",pending/"checkpoint.pt.json")
    _sync_file(pending/"checkpoint.pt");_sync_file(pending/"checkpoint.pt.json");_sync_dir(pending)
    os.rename(pending,legacy);_sync_dir(store)
    # 先指向同内容的legacy，再逐个安装别名，中断时普通文件和别名仍内容一致。
    _symlink_replace(_pointer(path),legacy)
    _aliases(path)

def save_checkpoint(path,payload):
    path=_slot(path);path.parent.mkdir(parents=True,exist_ok=True)
    store=_store(path);store.mkdir(exist_ok=True);_sync_dir(path.parent)
    _import_legacy(path)
    final=store/("epoch_"+format(payload["epoch"],"06d")+"_"+str(time.time_ns())+"_"+uuid.uuid4().hex)
    pending=store/(".pending-"+uuid.uuid4().hex);pending.mkdir()
    file=pending/"checkpoint.pt"
    with file.open("wb") as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    _fault("after_tensor")
    side={"sha256":c.sha(file),"epoch":payload["epoch"],"scope":payload["scope"],"cache_manifest_sha256":payload["cache_manifest_sha256"]}
    with Path(str(file)+".json").open("w",encoding="utf-8") as f:
        json.dump(side,f,ensure_ascii=False,allow_nan=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
    _fault("after_sidecar")
    _read_pair(file,"cpu");_sync_dir(pending)
    _fault("before_commit")
    os.rename(pending,final);_sync_dir(store)
    _fault("after_commit")
    _publish(path,final/"checkpoint.pt")
    return final/"checkpoint.pt"

def restore_checkpoint(path,device):
    path=_slot(path)
    payload,side,file=resolve_checkpoint(path,device)
    c.verify(payload["inference_source_sha256"])
    if file==path:
        # 验证过的旧格式可继续读取；下一次保存时再无损导入。
        return payload
    _publish(path,file)
    return payload

def checkpoint_exists(path):
    path=_slot(path)
    return path.exists() or _pointer(path).is_symlink() or _store(path).exists()

@contextlib.contextmanager
def training_lock(output):
    """目录和锁文件是取得锁所必需的唯一先行创建；协议/产物写入均在锁内。"""
    import fcntl
    output=c.safe(output);output.mkdir(parents=True,exist_ok=True)
    with c.safe(output/".run.lock").open("a") as guard:
        fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield output
        finally:fcntl.flock(guard.fileno(),fcntl.LOCK_UN)

def verify_training_smoke(smoke):
    """保留原GPU smoke；IO修复只由锁定的新CPU回归收据衔接，不改历史记录。"""
    if not smoke["method_audit_passed"]:raise ValueError("successful smoke required")
    hashes=smoke["source_sha256"]
    changed={p:(h,c.sha(p)) for p,h in hashes.items() if c.sha(p)!=h}
    if not changed:
        c.verify(hashes);return
    receipt=c.load(c.ROOT/"outputs/checkpoint_repair/completed.json")
    if not receipt["method_audit_passed"] or not receipt["attack_training_core_AST_unchanged"]:
        raise ValueError("checkpoint-only repair validation required")
    if receipt["prior_smoke_sha256"]!=c.sha(c.ROOT/"outputs/smoke/completed.json"):
        raise ValueError("historical GPU smoke receipt changed")
    allowed=receipt["source_transitions"]
    if set(changed)!=set(allowed):raise ValueError("unexpected source changes after GPU smoke")
    for p,(old,new) in changed.items():
        if allowed[p]!={"before":old,"after":new}:raise ValueError("unvalidated source transition: "+p)
    c.verify(receipt["source_sha256"])
