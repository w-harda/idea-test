"""确定性Epoch调度；H从seed42新建，GPU仅提供冻结文本奖励。"""
from dataclasses import asdict
from pathlib import Path
import random
import time
import numpy as np
import torch
from policy import CharacterPolicy
from phase4_training import train_step, optimizer_for
from feedback_backend import state_sha
from phase2_sources import digest, read_json, sha, ROOT
from .common import immutable_json, sources, verify_sources
from .data import members, example_from, INELIGIBLE
from . import checkpoint as ck

def config(epochs=10, interval=256):
    if type(epochs) is not int or not 1<=epochs<=100 or type(interval) is not int or interval<1:
        raise ValueError("positive explicit epoch/checkpoint interval required")
    original=read_json(ROOT/"config_phase4.json")
    return {"schema":"hv1-formal-training-config-1","epochs":epochs,"checkpoint_interval_updates":interval,
            "seed":42,"architecture":original["architecture"],"optimizer":original["optimizer"],
            "batch":1,"reward_weights":{"lambda_sim":1.,"lambda_rank":1.},
            "H_initialization":"fresh seed42; never load scst32 weights","ineligible_handling":INELIGIBLE}

def order(samples,epochs,seed=42):
    result=[]
    for epoch in range(epochs):
        indices=torch.randperm(len(samples),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        result.extend(samples[i].row_id for i in indices)
    return result

def protocol_for(catalog,store,qualification,cfg):
    if store.role!="train": raise ValueError("training requires Train-only cache")
    samples=members(catalog,"train")
    evidence=store.audit()
    return {"schema":"hv1-formal-SCST-training-1","execution_backend":store.protocol["execution_backend"],
            "config":cfg,"catalog_sha256":catalog.fingerprint,"gallery_protocol_sha256":store.hash,
            "gallery_content_sha256":evidence["content_sha256"],
            "qualification_sha256":digest(qualification),
            "train_identities":[asdict(s) for s in samples],"source_sha256":sources("training"),
            "query_order":order(samples,cfg["epochs"],cfg["seed"]),
            "epoch_size":len(samples),"validation_feedback":False,
            "eligible_by_row": {e["identity"]["row_id"]:e["eligible"] for e in qualification["entries"]}}

def fresh_policy(probe,cfg):
    random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
    return CharacterPolicy(probe,**cfg["architecture"])

def execute(root,protocol,catalog,qualification,store,probe,bank,runtime,stop,
            resume=False,rollback_uncommitted=False,max_visits=None):
    """max_visits只用于CPU fixture；真实入口没有此实验扩展参数。"""
    root=Path(root);cfg=protocol["config"]
    verify_sources(protocol)
    if protocol["execution_backend"]=="official_irra_real" and max_visits is not None:
        raise ValueError("real partial experiment is controlled only by user stop/epochs")
    samples=members(catalog,"train")
    if protocol["query_order"]!=order(samples,cfg["epochs"],cfg["seed"]):
        raise ValueError("query order changed")
    qualification_entries={e["identity"]["row_id"]:e for e in qualification["entries"]}
    if set(qualification_entries)!={s.row_id for s in samples}:
        raise ValueError("training qualification identity mismatch")
    lookup={s.row_id:s for s in samples}
    policy=fresh_policy(probe,cfg);optimizer=optimizer_for(policy,cfg)
    sampler=torch.Generator().manual_seed(cfg["seed"])
    immutable_json(root/"protocol.json",protocol)
    cursor=updates=0;parent=None
    if resume:
        payload,result=ck.rollback(root,protocol,rollback_uncommitted)
        ck.restore(payload,policy,optimizer,sampler)
        cursor,updates=payload["cursor"],payload["updates"]
        parent=result["published"][-1]["sha256"]
    else:
        if list((root/"checkpoints").glob("*")) or (root/"pending").exists():
            raise ValueError("existing formal run requires --resume")
        work=ck.begin(root,protocol,None)
        _,parent=ck.publish(root,work,policy,optimizer,sampler,protocol,0,0,None,0,0,"initial_seed42")
    initial_cursor=cursor
    work=None;start_cursor=cursor;start_updates=updates
    while cursor<len(protocol["query_order"]):
        if stop.requested: break
        if work is None:
            work=ck.begin(root,protocol,parent);start_cursor=cursor;start_updates=updates
        visit=cursor+1;q=lookup[protocol["query_order"][cursor]]
        catalog.require_train(q)
        entry=qualification_entries[q.row_id]
        if entry["identity"]!=asdict(q) or entry["caption"]!=q.caption:
            raise ValueError("qualification Caption/PID/identity mismatch")
        start=time.monotonic()
        if entry["eligible"]:
            ex=example_from(entry)
            record,_=train_step(policy,optimizer,ex,sampler,
                lambda k:bank.bind(q,runtime,ex,k),updates+1,q.row_id)
            updates+=1
        else:
            record={"row_id":q.row_id,"optimizer_steps_this_step":0,"skip_reasons":entry["reasons"],
                    "H_state_sha256_after":state_sha(policy),"NOOP":True,"IRRA_text_samples":0}
        cursor=visit
        record.update(visit=visit,epoch=(visit-1)//len(samples)+1,
                      epoch_position=(visit-1)%len(samples)+1,committed_update_candidate=updates,
                      seconds=time.monotonic()-start)
        record["GPU_peak_allocated_MiB"]=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_initialized() else 0.
        record["GPU_peak_reserved_MiB"]=torch.cuda.max_memory_reserved()/2**20 if torch.cuda.is_initialized() else 0.
        ck.append(work,record)
        runtime.assert_frozen()
        boundary=cursor%len(samples)==0
        partial=max_visits is not None and cursor-initial_cursor>=max_visits
        if boundary or (updates-start_updates>=cfg["checkpoint_interval_updates"]) or stop.requested or partial:
            reason="epoch_complete" if boundary else "safe_stop" if stop.requested or partial else "periodic"
            path,parent=ck.publish(root,work,policy,optimizer,sampler,protocol,cursor,updates,parent,
                                   start_cursor,start_updates,reason)
            print("H_CHECKPOINT_COMMITTED",cursor,updates,path,flush=True)
            work=None
        if cursor%32==0: print("SCST_PROGRESS",cursor,len(protocol["query_order"]),updates,flush=True)
        if partial: break
    if work is not None:
        _,parent=ck.publish(root,work,policy,optimizer,sampler,protocol,cursor,updates,parent,
                            start_cursor,start_updates,"safe_stop")
    audit_payload,evidence=ck.audit(root,protocol)
    if not ck.same(policy.state_dict(),audit_payload["policy"]) or not ck.same(optimizer.state_dict(),audit_payload["optimizer"]):
        raise ValueError("published state does not match running H/Adam")
    # 独立对象恢复验证；最后恢复原RNG，避免验证改变后续轨迹。
    rng=ck.capture_rng(sampler)
    other=fresh_policy(probe,cfg);other_opt=optimizer_for(other,cfg);other_sampler=torch.Generator()
    ck.restore(audit_payload,other,other_opt,other_sampler)
    ck.restore_rng(rng,sampler)
    result={"committed_visits":cursor,"committed_updates":updates,
            "complete":cursor==len(protocol["query_order"]),"checkpoint_sha256":parent,
            "restored_H_Adam_RNG_exact":True,"runtime":runtime.finish(),"audit":evidence}
    immutable_json(root/"attempts"/(str(cursor)+"_"+str(time.time_ns())+".json"),result)
    print("FORMAL_TRAIN_COMPLETE" if result["complete"] else "FORMAL_TRAIN_SAFE_STOP",cursor,updates,flush=True)
    return result

def log_statistics(root,protocol):
    """只从已提交日志累计标量；没有历史Tensor/图保留。"""
    from collections import Counter
    epochs={}
    for path in sorted((Path(root)/"checkpoints").glob("visit_*")):
        for r in ck.logs(path/"steps.jsonl"):
            e=epochs.setdefault(str(r["epoch"]),{"visits":0,"updates":0,"skipped":0,"scalars":{},
                                                  "K_distribution":Counter(),"sampled_NOOP":0,"greedy_NOOP":0})
            e["visits"]+=1;e["updates"]+=r["optimizer_steps_this_step"]
            if not r["optimizer_steps_this_step"]:
                e["skipped"]+=1;continue
            e["K_distribution"][str(r["K"])]+=1
            for mode in ("sampled","greedy"):
                e[mode+"_NOOP"]+=r[mode+"_action_stats"]["NOOP_output"]
            values={k:r[k] for k in ("gradient_norm","SCST_loss","advantage","parameter_delta_l2","seconds")}
            for mode in ("sampled","greedy"):
                values.update({mode+"_"+k:r[mode+"_reward"][k] for k in ("R_sim","R_rank","total","synergy_increment")})
            for name,v in values.items():
                a=e["scalars"].setdefault(name,{"sum":0.,"min":v,"max":v,"count":0})
                a["sum"]+=v;a["min"]=min(a["min"],v);a["max"]=max(a["max"],v);a["count"]+=1
    for e in epochs.values():
        e["K_distribution"]=dict(e["K_distribution"])
        for v in e["scalars"].values():v["mean"]=v["sum"]/v["count"]
    return epochs
