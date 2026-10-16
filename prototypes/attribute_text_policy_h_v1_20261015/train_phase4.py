"""H-v1真实32-step SCST；默认CPU preflight，GPU更新需显式授权。"""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
from pathlib import Path
import random
import time
import traceback
from unittest.mock import patch
import numpy as np
import torch
from phase2_sources import ROOT,digest,sha,read_json,new_json
from phase4_inputs import preflight,safe_output,verify_protocol
from phase4_checkpoint import (begin_step,publish,latest,restore,verify_reload,
                              capture_rng,same,fsync_dir)
from phase4_real_reward import load_runtime,FreshReward
from phase4_training import optimizer_for,train_step
from policy import CharacterPolicy
from feedback_backend import state_sha
from text_frontend import CaptionFrontend
from smoke_phase3 import StopBoundary

def policy_factory(probe,config):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        h=CharacterPolicy(probe,**config["architecture"])
    return h.cpu().train().requires_grad_(True)

def aggregate(logs):
    result={}
    for mode in ("sampled","greedy"):
        result[mode]={}
        for key in ("R_sim","R_rank","total","synergy_increment"):
            values=[r[mode+"_reward"][key] for r in logs]
            result[mode][key]={"mean":sum(values)/len(values),"min":min(values),"max":max(values)}
        actions=[r[mode+"_action_stats"] for r in logs]
        result[mode]["edits_total"]=sum(a["edits"] for a in actions)
        result[mode]["NOOP_output_ratio"]=sum(a["NOOP_output"] for a in actions)/len(actions)
        total=sum(a["attribute_decisions"] for a in actions)
        result[mode]["STOP_ratio"]=sum(a["STOP_count"] for a in actions)/total if total else 0.
        positions=sum(a["position_decisions"] for a in actions)
        result[mode]["position_NOOP_ratio"]=sum(a["position_NOOP_count"] for a in actions)/positions if positions else 0.
    result["K_distribution"]=dict(Counter(r["K"] for r in logs))
    for key in ("gradient_norm","SCST_loss","advantage","parameter_delta_l2"):
        values=[r[key] for r in logs]
        result[key]={"mean":sum(values)/len(values),"min":min(values),"max":max(values)}
    result["zero_gradient_steps"]=[r["step"] for r in logs if r["gradient_norm"]==0]
    result["unchanged_parameter_steps"]=[r["step"] for r in logs if r["parameter_delta_l2"]==0]
    return result

def readonly_verify(output,protocol=None):
    output=safe_output(output)
    saved=read_json(output/"protocol.json")
    if protocol is not None and digest(saved)!=digest(protocol):
        raise ValueError("resume/verify protocol changed")
    verify_protocol(saved)
    path,md=latest(output,saved)
    logs=[read_json(p/"sample.json") for p in sorted((output/"checkpoints").glob("step_*"))[1:]]
    completed=output/"completed.json"
    if completed.exists():
        done=read_json(completed)
        if (done["protocol_sha256"]!=digest(saved) or done["checkpoint_sha256"]!=md["checkpoint_sha256"]
                or sha(output/"summary.json")!=done["summary_sha256"] or len(logs)!=32 or md["step"]!=32):
            raise ValueError("controlled training completion mismatch")
        summary=read_json(output/"summary.json")
        if summary["optimizer_steps"]!=32 or summary["new_G_forwards"]!=0 or summary["new_IRRA_image_forwards"]!=0:
            raise ValueError("completed training budget mismatch")
    print("HV1_PHASE4_CPU_CHECKPOINT_VERIFIED",md["step"],md["checkpoint_sha256"],flush=True)
    return path,md,logs

def run(catalog,gallery,queries,features,probe,protocol,output,resume=False,stop_after_updates=32):
    config=protocol["config"]
    if type(stop_after_updates) is not int or not 1<=stop_after_updates<=32:
        raise ValueError("bounded updates must be in [1,32]")
    output=safe_output(output)
    if not resume:output.mkdir(parents=True,exist_ok=False)
    elif not output.is_dir():raise ValueError("resume requires an existing independent Phase4 run")
    with (output/".training.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/"protocol.json").exists():
            if digest(read_json(output/"protocol.json"))!=digest(protocol):
                raise ValueError("frozen training protocol changed")
        else:
            if resume:raise ValueError("resume protocol missing")
            new_json(output/"protocol.json",protocol);fsync_dir(output)
        if (output/"completed.json").exists():
            readonly_verify(output,protocol)
            return
        make_h=lambda:policy_factory(probe,config)
        make_opt=lambda h:optimizer_for(h,config)
        policy=make_h();optimizer=make_opt(policy)
        sampler=torch.Generator(device="cpu").manual_seed(config["sample_generator_seed"])
        prior=0;parent=None
        if resume:
            path,md=latest(output,protocol)
            value=restore(path,policy,optimizer,sampler,protocol,restore_random=False)
            prior=value["step"];parent=md["checkpoint_sha256"]
            if prior>=32:
                raise ValueError("all updates already published; inspect evidence, do not repeat")
        elif state_sha(policy)!=protocol["initial_H_state_sha256"]:
            raise ValueError("initial H differs from reviewed seed42 policy")
        attempts=output/"attempts";attempts.mkdir(exist_ok=True)
        index=1
        while (attempts/f"attempt_{index:04d}").exists():index+=1
        attempt=attempts/f"attempt_{index:04d}";attempt.mkdir()
        runtime=None
        started=time.perf_counter()
        context={"step":prior,"stage":"loading"}
        with StopBoundary() as stop,ExitStack() as guards:
            guards.enter_context(patch.object(torch.autograd,"grad",side_effect=AssertionError("no scoring/JVP")))
            try:
                verify_protocol(protocol)
                torch.cuda.reset_peak_memory_stats()
                runtime=load_runtime(protocol,probe,32-prior)
                random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
                if resume:
                    value=restore(path,policy,optimizer,sampler,protocol)
                else:
                    work=begin_step(output,0,protocol,None)
                    parent=publish(output,work,0,policy,optimizer,sampler,protocol,
                                   {"step":0,"initial_H_state_sha256":state_sha(policy)},None)
                frontend=CaptionFrontend()
                lookup={q.row_id:q for q in queries}
                restored=[]
                limit=min(32,prior+stop_after_updates)
                for step in range(prior+1,limit+1):
                    stop.check()
                    query=lookup[protocol["query_order"][step-1]]
                    catalog.require_train(query)
                    context={"step":step,"row_id":query.row_id,"stage":"SCST_update"}
                    work=begin_step(output,step,protocol,parent)
                    t=time.perf_counter()
                    make_reward=lambda k:FreshReward(catalog,query,gallery,features,runtime,frontend,k)
                    log,text_features=train_step(policy,optimizer,frontend(query.caption),sampler,
                                                 make_reward,step,query.row_id)
                    torch.cuda.synchronize()
                    log.update({"seconds_update":time.perf_counter()-t,
                        "GPU_peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
                        "GPU_peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20})
                    parent=publish(output,work,step,policy,optimizer,sampler,protocol,log,parent,text_features)
                    del text_features
                    print("HV1_PHASE4_STEP",step,"/32",query.row_id,"K",log["K"],
                        "reward",log["sampled_reward"]["total"],log["greedy_reward"]["total"],
                        "advantage",log["advantage"],"grad",log["gradient_norm"],
                        "delta",log["parameter_delta_l2"],flush=True)
                    if step in (16,32):
                        policy,optimizer,sampler,evidence=verify_reload(output,policy,optimizer,sampler,
                                                                      protocol,make_h,make_opt)
                        restored.append(evidence)
                        new_json(attempt/f"restore_step_{step:04d}.json",evidence)
                        print("HV1_PHASE4_RESTORE_EXACT",step,flush=True)
                    stop.check()
                final=runtime.finish()
                if final["samples"]!={"G":0,"clean":0,"adv":0,"text":3*(limit-prior)}:
                    raise ValueError("actual text-only training forward totals mismatch")
                execution={"prior_step":prior,"last_step":limit,"optimizer_steps_this_run":limit-prior,
                    "runtime":final,"restore_checks":restored,"seconds":time.perf_counter()-started,
                    "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
                    "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20}
                new_json(attempt/"execution.json",execution)
                verify_protocol(protocol)
                if limit==32:
                    path,md,logs=readonly_verify(output,protocol)
                    if (len(logs)!=32 or sum(x["optimizer_steps_this_step"] for x in logs)!=32
                            or sum(x["IRRA_text_calls"] for x in logs)!=32
                            or sum(x["IRRA_text_samples"] for x in logs)!=96):
                        raise ValueError("published step/text log coverage mismatch")
                    summary={"method_audit_passed":True,"protocol_sha256":digest(protocol),
                        "optimizer_steps":32,"new_G_forwards":0,"new_IRRA_image_forwards":0,
                        "IRRA_text_calls":32,"IRRA_text_samples":96,"query_count":10,"gallery_count":10,
                        "statistics":aggregate(logs),"first8":aggregate(logs[:8]),"last8":aggregate(logs[-8:]),
                        "initial_H_state_sha256":protocol["initial_H_state_sha256"],
                        "final_H_state_sha256":state_sha(policy),"execution":execution,
                        "final_checkpoint":str(path/"checkpoint.pt"),"checkpoint_sha256":md["checkpoint_sha256"],
                        "step_log_sha256":{str(p/"sample.json"):sha(p/"sample.json")
                            for p in (output/"checkpoints").glob("step_*")},
                        "Adam_parameter_step_distribution":dict(Counter(int(v["step"]) for v in optimizer.state.values())),
                        "not_model_selection":True,"NaN_or_anomalies":[],
                        "evaluation_goal":"Joint Full Test mAP below Image-only; not judged by this run"}
                    new_json(output/"summary.json",summary)
                    new_json(output/"completed.json",{"method_audit_passed":True,"protocol_sha256":digest(protocol),
                        "checkpoint_sha256":md["checkpoint_sha256"],"summary_sha256":sha(output/"summary.json")})
                    print("HV1_PHASE4_SCST32_COMPLETE",summary["statistics"],execution,flush=True)
                return
            except BaseException as exc:
                new_json(attempt/("interruption.json" if isinstance(exc,InterruptedError) else "failure.json"),
                    {"error":repr(exc),"traceback":traceback.format_exc(),"context":context,
                     "runtime_samples":dict(runtime.samples) if runtime else None})
                raise
            finally:
                if runtime is not None:runtime.close()

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action",choices=("preflight","run","verify"),default="preflight")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/phase4/scst32")
    p.add_argument("--allow-scst32",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--stop-after-updates",type=int,default=32)
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    if a.action=="verify":
        readonly_verify(a.output)
        return
    if a.action=="run" and not a.allow_scst32:
        raise ValueError("explicit --allow-scst32 required")
    values=preflight()
    print("HV1_PHASE4_CPU_PREFLIGHT",digest(values[-1]),"Gallery reused 10; train queries 10; no image forward",flush=True)
    if a.action=="run":run(*values,a.output,resume=a.resume,stop_after_updates=a.stop_after_updates)
if __name__=="__main__":main()
