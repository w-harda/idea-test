"""周期事务Checkpoint；提交段不可变，未提交段只能显式保留并回滚。"""
from pathlib import Path
import json
import os
import uuid
import torch
from phase2_sources import digest, sha, read_json
from phase4_checkpoint import capture_rng, restore_rng, same
from real_feature_cache import fsync_dir
from .common import immutable_json, json_bytes

SCHEMA = "hv1-formal-periodic-checkpoint-1"
def begin(root, protocol, parent):
    root=Path(root)
    work=root/"pending"
    work.mkdir(exist_ok=False)
    immutable_json(work/"intent.json",{"protocol_sha256":digest(protocol),"parent_checkpoint_sha256":parent})
    (work/"steps.jsonl").touch(exist_ok=False)
    fsync_dir(work);fsync_dir(root)
    return work

def append(work, log):
    # JSON序列化显式拒绝Tensor/NaN，不持有旧计算图。
    with (Path(work)/"steps.jsonl").open("ab") as f:
        f.write(json_bytes(log));f.flush();os.fsync(f.fileno())

def logs(path):
    content=Path(path).read_bytes()
    if content and not content.endswith(b"\n"): raise ValueError("partial/uncommitted log line")
    return [json.loads(line) for line in content.splitlines()]

def check_logs(records, start, end, updates_before, updates_after, protocol):
    if len(records)!=end-start: raise ValueError("log visit range mismatch")
    count=updates_before
    for visit,log in enumerate(records,start+1):
        if log["visit"]!=visit or log["row_id"]!=protocol["query_order"][visit-1]:
            raise ValueError("associated sample order mismatch")
        if log["optimizer_steps_this_step"] not in (0,1): raise ValueError("invalid update count")
        if log["optimizer_steps_this_step"]!=int(protocol["eligible_by_row"][log["row_id"]]):
            raise ValueError("qualified/ineligible update policy mismatch")
        count+=log["optimizer_steps_this_step"]
        if log["committed_update_candidate"]!=count: raise ValueError("step accounting mismatch")
    if count!=updates_after: raise ValueError("checkpoint Adam/log count mismatch")

def publish(root,work,policy,optimizer,sampler,protocol,cursor,updates,parent,start_cursor,start_updates,reason):
    root,work=Path(root),Path(work)
    intent={"protocol_sha256":digest(protocol),"parent_checkpoint_sha256":parent}
    if read_json(work/"intent.json")!=intent: raise ValueError("checkpoint intent mismatch")
    records=logs(work/"steps.jsonl")
    check_logs(records,start_cursor,cursor,start_updates,updates,protocol)
    logsha=sha(work/"steps.jsonl")
    payload={"schema":SCHEMA,"protocol":protocol,"protocol_sha256":digest(protocol),
        "cursor":cursor,"updates":updates,"start_cursor":start_cursor,"start_updates":start_updates,
        "query_order":protocol["query_order"],"policy":policy.state_dict(),
        "optimizer":optimizer.state_dict(),"rng":capture_rng(sampler),
        "sample_log_sha256":logsha,"parent_checkpoint_sha256":parent,"reason":reason}
    torch.save(payload,work/"checkpoint.pt")
    with (work/"checkpoint.pt").open("rb") as f:os.fsync(f.fileno())
    md={"schema":SCHEMA,"protocol_sha256":digest(protocol),"cursor":cursor,"updates":updates,
        "start_cursor":start_cursor,"start_updates":start_updates,"parent_checkpoint_sha256":parent,
        "sample_log_sha256":logsha,"checkpoint_sha256":sha(work/"checkpoint.pt"),
        "intent_sha256":sha(work/"intent.json"),"reason":reason}
    immutable_json(work/"checkpoint.json",md)
    fsync_dir(work)
    directory=root/"checkpoints";directory.mkdir(exist_ok=True)
    target=directory/f"visit_{cursor:08d}"
    if target.exists(): raise FileExistsError("immutable checkpoint already exists")
    os.rename(work,target);fsync_dir(directory);fsync_dir(root)
    return target,md["checkpoint_sha256"]

def verify(path,protocol,parent,start_cursor,start_updates):
    path=Path(path)
    md=read_json(path/"checkpoint.json")
    if (md["schema"]!=SCHEMA or md["protocol_sha256"]!=digest(protocol)
            or md["checkpoint_sha256"]!=sha(path/"checkpoint.pt")
            or md["sample_log_sha256"]!=sha(path/"steps.jsonl")
            or md["intent_sha256"]!=sha(path/"intent.json")
            or md["parent_checkpoint_sha256"]!=parent
            or md["start_cursor"]!=start_cursor or md["start_updates"]!=start_updates
            or path.name!=f'visit_{md["cursor"]:08d}'
            or read_json(path/"intent.json")!={"protocol_sha256":digest(protocol),"parent_checkpoint_sha256":parent}):
        raise ValueError("checkpoint/source/associated log SHA mismatch")
    payload=torch.load(path/"checkpoint.pt",map_location="cpu",weights_only=False)
    if payload["schema"]!=SCHEMA or payload["protocol"]!=protocol or payload["query_order"]!=protocol["query_order"]:
        raise ValueError("checkpoint training config/order mismatch")
    for key in ("protocol_sha256","cursor","updates","start_cursor","start_updates",
                "sample_log_sha256","parent_checkpoint_sha256","reason"):
        if payload[key]!=md[key]: raise ValueError("checkpoint metadata/payload mismatch")
    if not 0<=md["cursor"]<=len(protocol["query_order"]) or md["updates"]<start_updates:
        raise ValueError("checkpoint progress invalid")
    records=logs(path/"steps.jsonl")
    check_logs(records,start_cursor,md["cursor"],start_updates,md["updates"],protocol)
    for state in payload["optimizer"]["state"].values():
        if not 0<state["step"].item()<=md["updates"]: raise ValueError("Adam parameter step outside committed update range")
        for value in state.values():
            if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("nonfinite Adam state")
    if any(not torch.isfinite(x).all() for x in payload["policy"].values()):
        raise ValueError("nonfinite H checkpoint")
    return payload,md

def audit(root,protocol):
    """只读校验全部发布版本；不恢复CUDA或进行模型前向。"""
    root=Path(root)
    versions=sorted((root/"checkpoints").glob("visit_*"))
    if not versions: raise ValueError("no full published checkpoint")
    parent=None;cursor=updates=0;items=[]
    for i,path in enumerate(versions):
        payload,md=verify(path,protocol,parent,cursor,updates)
        if (i==0 and md["cursor"]!=0) or (i and md["cursor"]<=cursor):
            raise ValueError("checkpoint history missing initial/monotonic progress")
        parent=md["checkpoint_sha256"];cursor,updates=md["cursor"],md["updates"]
        items.append({"path":str(path),"sha256":parent,"cursor":cursor,"updates":updates,"reason":md["reason"]})
    return payload,{"published":items,"committed_visits":cursor,"committed_updates":updates,
                    "uncommitted_tail_exists":(root/"pending").exists()}

def rollback(root,protocol,acknowledge=False):
    root=Path(root)
    payload,result=audit(root,protocol)
    work=root/"pending"
    if work.exists():
        if not acknowledge:
            raise ValueError("uncommitted tail retained; resume requires explicit --rollback-uncommitted")
        intent=read_json(work/"intent.json")
        if intent!={"protocol_sha256":digest(protocol),
                    "parent_checkpoint_sha256":result["published"][-1]["sha256"]}:
            raise ValueError("unknown pending segment; cannot roll back automatically")
        archive=root/"abandoned"/uuid.uuid4().hex
        archive.parent.mkdir(exist_ok=True)
        os.rename(work,archive);fsync_dir(root);fsync_dir(archive.parent)
        immutable_json(archive/"rollback.json",{
            "acknowledged":True,"restore_cursor":payload["cursor"],"restore_updates":payload["updates"],
            "preserved_files_sha256":{p.name:sha(p) for p in archive.iterdir() if p.is_file()},
            "policy":"tail never committed; may contain executed/unlogged updates; replay from full checkpoint; do not count twice"})
    return payload,result

def restore(payload,policy,optimizer,sampler,allow_cuda=True):
    policy.load_state_dict(payload["policy"],strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"],sampler,allow_cuda=allow_cuda)
    if not same(policy.state_dict(),payload["policy"]) or not same(optimizer.state_dict(),payload["optimizer"]):
        raise ValueError("H/Adam restore mismatch")
    rng=capture_rng(sampler)
    if not allow_cuda: rng["torch_cuda"]=payload["rng"]["torch_cuda"]
    if not same(rng,payload["rng"]): raise ValueError("RNG restore mismatch")
