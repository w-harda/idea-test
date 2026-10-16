"""独立Validation四组官方检索指标；全部Query保留，仅用于模型选择。"""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import os
import torch
from phase2_sources import digest, sha, read_json
from phase3_inputs import OFFICIAL
from rewards import frozen_features
from constraints import replay
from feedback_backend import state_sha
from .common import immutable_json, sources, verify_sources
from .data import example_from
from .storage import fsync_dir

def metrics(query,gallery,qids,gids):
    frozen_features(query);frozen_features(gallery)
    if len(gallery)<10 or len(query)!=len(qids) or len(gallery)!=len(gids):
        raise ValueError("fixed validation dimensions/ranks mismatch")
    if not set(qids.tolist()).issubset(set(gids.tolist())):
        raise ValueError("validation query without positive")
    path=OFFICIAL/"utils/metrics.py"
    spec=importlib.util.spec_from_file_location("hv1_formal_official_metrics",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with torch.no_grad():
        cmc,map_,minp,_=module.rank(query@gallery.T,qids,gids,max_rank=10,get_mAP=True)
    return {"mAP":float(map_),"R@1":float(cmc[0]),"R@5":float(cmc[4]),"R@10":float(cmc[9]),"mINP":float(minp)}

def four_groups(original,edited,features,pids):
    return {name:metrics(text,images,pids,pids) for name,text,images in (
        ("Clean",original,features["clean"]),("Image-only",original,features["adversarial"]),
        ("Text-only",edited,features["clean"]),("Joint",edited,features["adversarial"]))}

def query_feature(root,protocol,sample,entry,policy,original,runtime,probe):
    unit=root/"queries"/digest(asdict(sample))
    if unit.exists():
        md,receipt=read_json(unit/"metadata.json"),read_json(unit/"receipt.json")
        if (md["protocol_sha256"]!=digest(protocol) or md["identity"]!=asdict(sample)
                or receipt!={"metadata_sha256":sha(unit/"metadata.json"),"feature_sha256":sha(unit/"text.pt")}
                or md["feature_sha256"]!=receipt["feature_sha256"]):
            raise ValueError("validation query feature/identity/SHA mismatch")
        x=torch.load(unit/"text.pt",map_location="cpu",weights_only=True).clone()
        frozen_features(x,torch.Size((1,512)))
        return x
    work=unit.with_name(unit.name+".pending")
    work.mkdir(parents=True,exist_ok=False)
    immutable_json(work/"intent.json",{"protocol_sha256":digest(protocol),"identity":asdict(sample)})
    if entry["eligible"]:
        example=example_from(entry)
        output=policy.attack(example)
        if replay(example,output.edits,output.budget_k)!=output.adversarial:
            raise ValueError("H validation trace mismatch")
        text=output.adversarial;action=output.record()
    else:
        text=sample.caption
        action={"NOOP":True,"reasons":entry["reasons"],"edits":[],"budget_k":0}
    if text==sample.caption:
        x=original.clone();text_samples=0
    else:
        x=runtime.encode_text((text,));text_samples=1
    frozen_features(x,torch.Size((1,512)))
    torch.save(x.detach().cpu().clone(),work/"text.pt")
    with (work/"text.pt").open("rb") as f:os.fsync(f.fileno())
    immutable_json(work/"metadata.json",{"protocol_sha256":digest(protocol),"identity":asdict(sample),
            "adversarial":text,"action":action,"feature_sha256":sha(work/"text.pt"),
            "token_sha256":digest(probe.inspect(text).model_tokens),
            "IRRA_text_samples_this_query":text_samples,"H_caption_only":True})
    immutable_json(work/"receipt.json",{"metadata_sha256":sha(work/"metadata.json"),"feature_sha256":sha(work/"text.pt")})
    fsync_dir(work);os.rename(work,unit);fsync_dir(unit.parent)
    return x

def evaluate(root,train_protocol,payload,checkpoint_sha,store,qualification,policy,runtime,probe,stop):
    if store.role!="validation": raise ValueError("independent validation gallery required")
    if train_protocol.get("validation_feedback") is not False:
        raise ValueError("invalid training isolation contract")
    evidence=store.audit();samples=store.samples
    if [e["identity"] for e in qualification["entries"]] != [asdict(s) for s in samples]:
        raise ValueError("fixed validation identities changed")
    protocol={"schema":"hv1-formal-validation-1","checkpoint_sha256":checkpoint_sha,
        "training_protocol_sha256":digest(train_protocol),"cursor":payload["cursor"],
        "validation_gallery_protocol_sha256":store.hash,"validation_content_sha256":evidence["content_sha256"],
        "qualification_sha256":digest(qualification),"identities":[asdict(s) for s in samples],
        "source_sha256":sources("validation"),"selection":"minimum Joint mAP; tie earliest committed visit"}
    protocol["source_sha256"][str(OFFICIAL/"utils/metrics.py")]=sha(OFFICIAL/"utils/metrics.py")
    root=Path(root)/checkpoint_sha
    immutable_json(root/"protocol.json",protocol)
    if (root/"completed.json").exists():
        return audit_result(root,store,qualification,train_protocol)
    policy.eval().requires_grad_(False)
    if state_sha(policy)!=state_sha_from(payload["policy"]): raise ValueError("wrong validation H")
    before=state_sha(policy)
    features,original=store.gather(),store.originals()
    edited=[]
    for i,(sample,entry) in enumerate(zip(samples,qualification["entries"])):
        if stop.requested:
            print("VALIDATION_SAFE_STOP",i,len(samples),flush=True)
            return None
        edited.append(query_feature(root,protocol,sample,entry,policy,original[i:i+1],runtime,probe))
        if (i+1)%32==0:print("VALIDATION_PROGRESS",i+1,len(samples),flush=True)
    if state_sha(policy)!=before or any(p.grad is not None for p in policy.parameters()):
        raise ValueError("validation modified H")
    pids=torch.tensor([s.pid for s in samples])
    groups=four_groups(original,torch.cat(edited),features,pids)
    files={str(p.relative_to(root)):sha(p) for p in (root/"queries").rglob("*") if p.is_file()}
    summary={"protocol_sha256":digest(protocol),"checkpoint_sha256":checkpoint_sha,"cursor":payload["cursor"],
        "groups":groups,"Joint_minus_ImageOnly_mAP_pp":groups["Joint"]["mAP"]-groups["Image-only"]["mAP"],
        "gallery":len(samples),"queries":len(samples),"all_queries_retained":True,
        "query_files_sha256":files,"H_unchanged":True,"H_state_sha256":before,
        "limitation":"128 Gallery/Query small internal validation; not official Test; no training reward"}
    immutable_json(root/"summary.json",summary)
    immutable_json(root/"completed.json",{"summary_sha256":sha(root/"summary.json"),"protocol_sha256":digest(protocol),
                                          "checkpoint_sha256":checkpoint_sha,"queries":len(samples)})
    print("VALIDATION_COMPLETE",checkpoint_sha,groups["Joint"]["mAP"],flush=True)
    return summary

def state_sha_from(state):
    import hashlib
    h=hashlib.sha256()
    for name,x in sorted(state.items()):
        h.update(name.encode());h.update(str((tuple(x.shape),x.dtype)).encode())
        h.update(x.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def audit_result(root,store,qualification,train_protocol):
    root=Path(root)
    p,s,d=read_json(root/"protocol.json"),read_json(root/"summary.json"),read_json(root/"completed.json")
    verify_sources(p)
    if (d!={"summary_sha256":sha(root/"summary.json"),"protocol_sha256":digest(p),
            "checkpoint_sha256":p["checkpoint_sha256"],"queries":len(store.samples)}
            or s["protocol_sha256"]!=digest(p) or s["checkpoint_sha256"]!=p["checkpoint_sha256"]
            or p["validation_gallery_protocol_sha256"]!=store.hash
            or p["validation_content_sha256"]!=store.audit()["content_sha256"]
            or p["qualification_sha256"]!=digest(qualification)
            or p["training_protocol_sha256"]!=digest(train_protocol)
            or p["identities"]!=[asdict(x) for x in store.samples]
            or s["queries"]!=len(store.samples) or not s["all_queries_retained"]):
        raise ValueError("validation completion/protocol mismatch")
    if list(root.rglob("*.pending")): raise ValueError("uncommitted validation query")
    actual={str(x.relative_to(root)):sha(x) for x in (root/"queries").rglob("*") if x.is_file()}
    if actual!=s["query_files_sha256"] or len(list((root/"queries").iterdir()))!=len(store.samples):
        raise ValueError("validation query evidence mismatch")
    if s["Joint_minus_ImageOnly_mAP_pp"]!=s["groups"]["Joint"]["mAP"]-s["groups"]["Image-only"]["mAP"]:
        raise ValueError("validation metric summary mismatch")
    return s

def select_best(results):
    if not results: raise ValueError("no audited validation checkpoints")
    best=min(results,key=lambda r:(r["groups"]["Joint"]["mAP"],r["cursor"]))
    return {"criterion":"minimum independent Validation Joint mAP, earliest visit tie",
            "selected_checkpoint_sha256":best["checkpoint_sha256"],"cursor":best["cursor"],
            "Joint_mAP":best["groups"]["Joint"]["mAP"],
            "Joint_minus_ImageOnly_mAP_pp":best["Joint_minus_ImageOnly_mAP_pp"],
            "candidates":[{"checkpoint_sha256":r["checkpoint_sha256"],"cursor":r["cursor"],
                           "groups":r["groups"],"Joint_minus_ImageOnly_mAP_pp":r["Joint_minus_ImageOnly_mAP_pp"]}
                          for r in results],"uses_official_Test":False}
