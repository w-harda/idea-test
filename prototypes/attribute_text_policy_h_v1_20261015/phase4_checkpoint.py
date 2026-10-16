"""H-v1独立step边界Checkpoint：日志SHA绑定、不可变版本、原子发布及RNG恢复。"""
from pathlib import Path
import os
import random
import numpy as np
import torch
from phase2_sources import digest,sha,read_json,new_json
from real_feature_cache import fsync_dir

def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def capture_rng(sampler):
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state().clone(),
            "sample_generator":sampler.get_state().clone(),
            "torch_cuda":tuple(x.cpu().clone() for x in torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else ()}

def restore_rng(state,sampler,allow_cuda=True):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    sampler.set_state(state["sample_generator"])
    if state["torch_cuda"] and allow_cuda:
        if not torch.cuda.is_initialized():
            raise ValueError("CUDA RNG restore requires already-loaded frozen runtime")
        torch.cuda.set_rng_state_all(list(state["torch_cuda"]))

def begin_step(root,step,protocol,parent_sha):
    if type(step) is not int or not 0<=step<=protocol["config"]["total_steps"]:
        raise ValueError("checkpoint step outside controlled budget")
    root=Path(root)
    (root/"checkpoints").mkdir(parents=True,exist_ok=True)
    work=root/"pending"/f"step_{step:04d}"
    work.mkdir(parents=True,exist_ok=False)
    new_json(work/"intent.json",{"step":step,"protocol_sha256":digest(protocol),"parent_checkpoint_sha256":parent_sha})
    fsync_dir(work);fsync_dir(work.parent);fsync_dir(root)
    return work

def publish(root,work,step,policy,optimizer,sampler,protocol,log,parent_sha,text_features=None):
    root,work=Path(root),Path(work)
    intent=read_json(work/"intent.json")
    expected={"step":step,"protocol_sha256":digest(protocol),"parent_checkpoint_sha256":parent_sha}
    if intent!=expected or log["step"]!=step:
        raise ValueError("step publication identity mismatch")
    payload={"schema":"hv1-scst-step-checkpoint-1","step":step,"next_step":step+1,
        "protocol":protocol,"protocol_sha256":digest(protocol),"config":protocol["config"],
        "policy":policy.state_dict(),"optimizer":optimizer.state_dict(),"rng":capture_rng(sampler),
        "query_order":protocol["query_order"],"sample_cursor":step}
    checkpoint=work/"checkpoint.pt"
    torch.save(payload,checkpoint)
    with checkpoint.open("rb") as f:os.fsync(f.fileno())
    new_json(work/"sample.json",log)
    text_sha=None
    if text_features is not None:
        from rewards import frozen_features
        frozen_features(text_features,torch.Size((3,512)))
        torch.save(text_features.detach().cpu().clone(),work/"text_features.pt")
        with (work/"text_features.pt").open("rb") as stream:os.fsync(stream.fileno())
        text_sha=sha(work/"text_features.pt")
    md={**expected,"checkpoint_sha256":sha(checkpoint),"sample_log_sha256":sha(work/"sample.json"),
        "intent_sha256":sha(work/"intent.json"),"text_features_sha256":text_sha}
    new_json(work/"checkpoint.json",md)
    fsync_dir(work)
    target=root/"checkpoints"/f"step_{step:04d}"
    if target.exists():
        raise FileExistsError("immutable step checkpoint already published")
    os.rename(work,target)
    fsync_dir(target.parent);fsync_dir(work.parent)
    return md["checkpoint_sha256"]

def check_version(path,protocol,parent_sha):
    path=Path(path)
    md=read_json(path/"checkpoint.json")
    step=int(path.name.removeprefix("step_"))
    if (md["protocol_sha256"]!=digest(protocol) or md["step"]!=step
            or md["parent_checkpoint_sha256"]!=parent_sha
            or md["checkpoint_sha256"]!=sha(path/"checkpoint.pt")
            or md["sample_log_sha256"]!=sha(path/"sample.json")
            or md["intent_sha256"]!=sha(path/"intent.json")
            or read_json(path/"intent.json")!={"step":step,"protocol_sha256":digest(protocol),
                                              "parent_checkpoint_sha256":parent_sha}):
        raise ValueError("checkpoint/source/linked-log SHA mismatch")
    if md.get("text_features_sha256") is not None and md["text_features_sha256"]!=sha(path/"text_features.pt"):
        raise ValueError("associated real text feature SHA mismatch")
    log=read_json(path/"sample.json")
    if log["step"]!=step:
        raise ValueError("checkpoint log step mismatch")
    if step and log["row_id"]!=protocol["query_order"][step-1]:
        raise ValueError("checkpoint associated sample order mismatch")
    return md

def latest(root,protocol):
    root=Path(root)
    pending=root/"pending"
    if pending.exists() and any(p.is_dir() for p in pending.iterdir()):
        raise ValueError("unpublished step intent exists; preserve evidence; automatic retry forbidden")
    versions=sorted((root/"checkpoints").glob("step_*"))
    if not versions:
        raise ValueError("no published H checkpoint")
    parent=None
    for i,path in enumerate(versions):
        if path.name!=f"step_{i:04d}" or i>protocol["config"]["total_steps"]:
            raise ValueError("noncontiguous/out-of-budget checkpoint history")
        md=check_version(path,protocol,parent)
        parent=md["checkpoint_sha256"]
    return versions[-1],md

def restore(path,policy,optimizer,sampler,protocol,restore_random=True,allow_cuda=True):
    path=Path(path)
    # latest()核对完整历史；本函数仍独立校验当前文件与日志，不信任torch.load前未核验的数据。
    md=read_json(path/"checkpoint.json")
    check_version(path,protocol,md["parent_checkpoint_sha256"])
    value=torch.load(path/"checkpoint.pt",map_location="cpu",weights_only=False)
    if (value["schema"]!="hv1-scst-step-checkpoint-1" or value["protocol_sha256"]!=digest(protocol)
            or digest(value["protocol"])!=digest(protocol) or value["config"]!=protocol["config"]
            or value["query_order"]!=protocol["query_order"] or value["sample_cursor"]!=value["step"]
            or value["step"]!=md["step"] or value["next_step"]!=value["step"]+1):
        raise ValueError("checkpoint payload/config/progress mismatch")
    policy.load_state_dict(value["policy"],strict=True)
    optimizer.load_state_dict(value["optimizer"])
    if restore_random:
        restore_rng(value["rng"],sampler,allow_cuda)
    return value

def verify_reload(root,policy,optimizer,sampler,protocol,make_policy,make_optimizer):
    """无额外forward/backward/step；独立新对象验证，然后用恢复对象继续。"""
    before_rng=capture_rng(sampler)
    old_h={k:v.detach().clone() for k,v in policy.state_dict().items()}
    old_adam=optimizer.state_dict()
    path,md=latest(root,protocol)
    new_h=make_policy()
    new_opt=make_optimizer(new_h)
    new_sampler=torch.Generator(device="cpu")
    value=restore(path,new_h,new_opt,new_sampler,protocol)
    if (not same(old_h,new_h.state_dict()) or not same(old_adam,new_opt.state_dict())
            or not same(before_rng,capture_rng(new_sampler))):
        raise ValueError("independent H/Adam/RNG restore mismatch")
    evidence={"step":value["step"],"checkpoint_sha256":md["checkpoint_sha256"],
              "H_exact":True,"Adam_exact":True,"RNG_exact":True,"query_order_exact":True,
              "extra_optimizer_steps":0,"extra_model_forwards":0}
    return new_h,new_opt,new_sampler,evidence
