"""独立不可变Epoch checkpoint；SHA/fsync/原子发布，拒绝损坏，不支持step恢复。"""
import os,random,uuid,sys
from pathlib import Path
import numpy as np
import torch
import cache_io as io

def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value

def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}

def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_initialized(): raise ValueError("CUDA RNG recovery requires initialized same CUDA environment")
        torch.cuda.set_rng_state_all(state["cuda"])

def save_epoch(output,generator,optimizer,epoch,contract,progress):
    if type(epoch) is not int or epoch<1: raise ValueError("completed Epoch required")
    out=io.safe(output); versions=out/"checkpoints";versions.mkdir(parents=True,exist_ok=True)
    final=io.safe(versions/f"epoch_{epoch:04d}")
    if final.exists(): raise ValueError("immutable Epoch checkpoint already exists")
    stage=io.safe(versions/(".pending-"+uuid.uuid4().hex));stage.mkdir()
    payload={"schema":"gv2-epoch-checkpoint-1","completed_epoch":epoch,"next_epoch":epoch+1,
        "contract":contract,"contract_sha256":io.digest(contract),"generator":cpu_tree(generator.state_dict()),
        "optimizer":cpu_tree(optimizer.state_dict()),"rng":rng_state(),"progress":progress}
    with (stage/"checkpoint.pt").open("xb") as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    sha=io.sha(stage/"checkpoint.pt")
    io.save_json(stage/"checkpoint.json",{"sha256":sha,"epoch":epoch,"contract_sha256":io.digest(contract)})
    # 在发布前验证真实payload和sidecar。
    verify_version(stage,contract)
    io.fsync_dir(stage);os.rename(stage,final);io.fsync_dir(versions)
    io.save_json(out/"current.json",{"version":str(final),"epoch":epoch,"sha256":sha})
    return final

def verify_version(version,contract):
    p=Path(version);side=io.load(p/"checkpoint.json")
    io.verify({str(p/"checkpoint.pt"):side["sha256"]})
    if side["contract_sha256"]!=io.digest(contract): raise ValueError("checkpoint source/config/cache contract mismatch")
    value=torch.load(p/"checkpoint.pt",map_location="cpu",weights_only=False)
    if (value["schema"]!="gv2-epoch-checkpoint-1" or io.digest(value["contract"])!=io.digest(contract)
        or value["contract_sha256"]!=io.digest(contract)
        or value["completed_epoch"]!=side["epoch"] or value["next_epoch"]!=side["epoch"]+1):
        raise ValueError("checkpoint payload/sidecar mismatch")
    if p.name.startswith("epoch_") and p.name!=f"epoch_{value['completed_epoch']:04d}":
        raise ValueError("checkpoint directory/epoch mismatch")
    return value

def load_latest(output,contract,generator,optimizer):
    out=io.safe(output)
    versions=sorted((out/"checkpoints").glob("epoch_[0-9][0-9][0-9][0-9]"))
    if not versions: raise ValueError("no published Epoch checkpoint")
    # pending忽略；最高已发布版本损坏直接拒绝，不静默回退/择优。
    version=versions[-1];value=verify_version(version,contract)
    generator.load_state_dict(value["generator"],strict=True)
    optimizer.load_state_dict(value["optimizer"])
    restore_rng(value["rng"])
    io.save_json(out/"current.json",{"version":str(version),"epoch":value["completed_epoch"],
                                    "sha256":io.sha(version/"checkpoint.pt")})
    return value
