"""G-v2 Epoch11–60续训验证入口；默认CPU，本阶段只运行一次独立Smoke。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
from unittest.mock import patch
import argparse,json,os,signal,subprocess,time,traceback
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest,sha_file
from .gv2_epoch_state import (POINTS,need,ref,config,legacy,base_state,validate_contract,restore,adam_steps,
    tree_hash,validate_progress,pack,save_checkpoint,load_checkpoint,restore_boundary,session_lock,
    snapshot,check_snapshot,summarize,tensor_bytes)

@contextmanager
def no_cuda():
    def blocked(*a,**kw):raise ValueError("CPU preflight cannot initialize CUDA")
    need(not torch.cuda.is_initialized(),"fresh CPU process required")
    with patch.object(torch.cuda,"_lazy_init",blocked):yield
    need(not torch.cuda.is_initialized(),"CPU validation initialized CUDA")

def preflight(c):
    torch.set_num_threads(2)
    with no_cuda():
        contract,parent,m=base_state(c)
        from tbps.models.irra_source import IRRASource
        source=IRRASource.from_dict(read_json(c["irra_source_ref"]["path"])["irra"]);source.verify()
        previous=read_json(c["history_summary"])
        need(previous["method_audit_passed"] and previous["final_Adam_step"]==80680,"completed original ten Epochs")
        et,ck,_=legacy(c)
        train=[e for e in m["entries"] if e["role"]=="train"]
        ordered=et.epoch_order(train,42,11)
        need(8068==sum(e["K"]+1 for e in train),"original K+1 count")
        first=ordered[0]
        need(c["smoke_member"]=={"row_id":first["row_id"],"K":first["K"]},"GPU smoke fixed before execution")
        report={"status":"GV2_EPOCH60_CPU_PREFLIGHT_PASS","parent_checkpoint":c["parent_checkpoint"],
            "Adam_step":80680,"absolute_start_epoch":11,"original_epochs":10,"extension_target_epochs":60,
            "first_epoch11_sample":c["smoke_member"],"Epoch11_order_sha256":digest([e["row_id"] for e in ordered]),
            "scheduler":None,"lr":2e-4,"betas":[.5,.999],"contract_sha256":digest(contract),
            "train":2048,"validation":128,"new_JVP":0,"new_scoring":0,"CUDA":False,
            "history":[{"epoch":e["epoch"],"train":e["statistics"]["all"]["total"]["mean"],"validation":e["validation"]}
                for e in previous["epochs"]]}
    print("GV2_EPOCH60_CPU_PREFLIGHT_PASS",json.dumps(report,ensure_ascii=False),flush=True)
    return contract,parent,m,report

def resources(c):
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True,timeout=10).splitlines()[0]
    name,free,total,util=[x.strip() for x in gpu.split(",")]
    processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits"],text=True,timeout=10)
    mem={line.split(":")[0]:int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line}
    disk=os.statvfs("/home/lzf/ldx/outputs")
    need(int(free)>=c["minimum_free_mib"],"GPU memory insufficient; no retry")
    need(mem["MemAvailable"]>=c["minimum_ram_bytes"] and disk.f_bavail*disk.f_frsize>=c["minimum_disk_bytes"],"RAM/disk insufficient")
    return {"GPU":name,"free_mib":int(free),"total_mib":int(total),"utilization":int(util),"compute_processes":processes,
        "available_RAM_bytes":mem["MemAvailable"],"disk_bytes":disk.f_bavail*disk.f_frsize,"free_inodes":disk.f_favail,
        "policy":"shared_observe_only; no waiting, no intervention"}

def run_epoch(et,ck,g,encoder,opt,entries,val_entries,loader,epoch,log_path,weights,*,limit=None,observer=None):
    """只观测原train_epoch；单样本Smoke选取完整绝对Epoch顺序的首项。"""
    original_order=et.epoch_order(entries,42,epoch)
    need(limit in (None,1),"only a single-sample probe or complete Epoch")
    selected=original_order[:1] if limit==1 else entries
    sample_fn=et.train_single_sample;position=0
    with Path(log_path).open("x",encoding="utf-8") as handle:
        def observed(g,enc,opt,image,condition,**kwargs):
            nonlocal position
            result=sample_fn(g,enc,opt,image,condition,observer=observer,**kwargs);position+=1
            row={"epoch":epoch,"sample_position":position,"row_id":condition["row_id"],"role":"train","K":condition["K"],
                "optimizer_steps":result["optimizer_steps"],"rounds":result["records"]}
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
            if position%32==0 or limit:
                print("GV2_TRAIN_PROGRESS",epoch,position,"/",len(selected),"Adam",adam_steps(opt),flush=True)
            return result
        with patch.object(et,"train_single_sample",observed):
            progress=et.train_epoch(g,encoder,opt,selected,loader,42,epoch,weights,log_path=None)
    expected=[e["row_id"] for e in original_order[:1] if limit] if limit else [e["row_id"] for e in original_order]
    need(progress["sample_order"]==expected,"absolute Epoch sample ordering")
    before=(tree_hash(g.state_dict()),tree_hash(opt.state_dict()),tree_hash(ck.rng_state()))
    progress["validation"]=et.validate_epoch(g,encoder,val_entries,loader)
    need(before==(tree_hash(g.state_dict()),tree_hash(opt.state_dict()),tree_hash(ck.rng_state())),"validation changed G/Adam/RNG")
    progress.update(sample_log_path=str(Path(log_path).resolve()),sample_log_sha256=sha_file(log_path))
    validate_progress(progress,formal=limit is None)
    return progress

class Observation:
    """只记录计数/标量/摘要；不改G、encoder、Adam或训练Tensor。"""
    def __init__(self,g,encoder,opt):
        self.g,self.encoder,self.opt=g,encoder,opt;self.counts={"G":0,"IRRA":0,"backward":0,"step":0}
        self.input_gradient=[];self.inputs=[];self.parameter_versions=[];self.handles=[]
    def __enter__(self):
        def before_g(m,args):
            self.counts["G"]+=1
            need(args[0].grad_fn is None and not args[0].requires_grad,"G must consume detached pixels")
            self.inputs.append(tree_hash(args[0]));self.parameter_versions.append(tuple(p._version for p in m.parameters()))
        def before_encoder(m,args):
            self.counts["IRRA"]+=1
            if torch.is_grad_enabled():
                need(args[0].requires_grad,"IRRA must preserve image gradient")
                def hook(v):
                    need(bool(torch.isfinite(v).all()),"IRRA input gradient nonfinite")
                    self.input_gradient.append(float(v.detach().double().norm()))
                args[0].register_hook(hook)
        def backward(v):
            need(bool(torch.isfinite(v).all()),"G gradient nonfinite");self.counts["backward"]+=1
        def step(*a):self.counts["step"]+=1
        self.handles=[self.g.register_forward_pre_hook(before_g),self.encoder.register_forward_pre_hook(before_encoder),
            next(self.g.parameters()).register_hook(backward),self.opt.register_step_post_hook(step)]
        return self
    def __exit__(self,*exc):
        for h in self.handles:h.remove()

def failure(store,c,exc,counts):
    if not (store.root/"failure").exists():
        store.publish("failure",{"report.json":json_bytes({"status":"FAIL","error":repr(exc),
            "traceback":traceback.format_exc(),"counts":counts})},{"kind":"extension_failure"},sources=(ref(c["path"]),))

def execute(c,*,smoke=False,resume=False):
    contract,parent,m,pre=preflight(c);validate_contract(contract)
    et,ck,oldio=legacy(c)
    root=Path(c["smoke_output"] if smoke else c["formal_output"])
    need(not smoke or not resume,"Smoke never resumes/retries")
    with session_lock(c["output_base"]):
        need(not smoke or not root.exists(),"existing Smoke Attempt; preserve, no retry")
        boundary=None if smoke else restore_boundary(root,contract,resume=resume)
        state=parent if boundary is None else boundary
        start_epoch=11 if boundary is None else boundary["next_epoch"]
        need(11<=start_epoch<=60,"authorized absolute Epoch range")
        r=resources(c);protection=snapshot(c)
        store=ArtifactStore(root,new_base=c["output_base"])
        if not resume:
            store.publish("protocol",{"protocol.json":json_bytes(contract),"preflight.json":json_bytes(pre),
                "resources.json":json_bytes(r),"protection.json":json_bytes(protection)},
                {"kind":"gv2-extension-protocol","scope":"smoke" if smoke else "formal"},
                sources=(ref(c["path"]),FileRef(**c["parent_checkpoint"])))
        else:
            store.verify("protocol");need(read_json(root/"protocol/protocol.json")==contract,"immutable resume protocol")
        need(not store.pending(),"pending session; stop")
        logs=root/"logs";logs.mkdir(exist_ok=True)
        started=time.monotonic();counts={};summary=None;stop={"requested":False}
        def request_stop(signum,frame):stop["requested"]=True
        old_handlers={s:signal.signal(s,request_stop) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            torch.set_num_threads(2)
            g=et.ImageOnlyGenerator(**contract["original_training_contract"]["config"]["architecture"]).to("cuda")
            import irra_image_adapter
            encoder=irra_image_adapter.load_official("cuda")
            opt=torch.optim.Adam(g.parameters(),**contract["original_training_contract"]["config"]["optimizer"])
            initial=restore(state,g,opt,ck);initial.update(absolute_next_epoch=start_epoch,source_checkpoint=c["parent_checkpoint"],
                source_state_epoch=state["completed_epoch"],GPU=True,IRRA_loads=1,G_initializations=1)
            need(adam_steps(opt)==[(start_epoch-1)*8068],"pre-update Adam count")
            first_epoch_source=tree_hash(state["generator"]);del parent,boundary,state
            key="initial_restore" if not resume else f"restore_before_{start_epoch:04d}"
            store.publish(key,{"report.json":json_bytes(initial)},{"kind":"G-Adam-RNG-exact-restore"},
                sources=(ref(c["path"]),FileRef(**c["parent_checkpoint"])))
            frozen=tree_hash(encoder.state_dict())
            loader=et.make_loader(m,"cuda")
            entries=[e for e in m["entries"] if e["role"]=="train"]
            vals=[e for e in m["entries"] if e["role"]=="validation"]
            weights=et.LossWeights(**contract["original_training_contract"]["config"]["weights"])
            epochs=(11,) if smoke else range(start_epoch,61)
            cs=None if smoke else ArtifactStore(root/"checkpoints",new_base=c["output_base"])
            for epoch in epochs:
                claim="probe_claim" if smoke else f"claim_{epoch:04d}"
                need(not (root/claim).exists(),"existing epoch Claim; no replay")
                order=et.epoch_order(entries,42,epoch)
                store.publish(claim,{"claim.json":json_bytes({"scope":"smoke" if smoke else "formal",
                    "absolute_epoch":epoch,"complete_epoch_will_be_published":not smoke,
                    "parent":c["parent_checkpoint"],"order_sha256":digest([e["row_id"] for e in order]),
                    "smoke_member":c["smoke_member"] if smoke else None})},{"kind":"pre-update-claim"},sources=(ref(c["path"]),))
                begin=time.monotonic();before_g=tree_hash(g.state_dict());events=[]
                with Observation(g,encoder,opt) as observation:
                    progress=run_epoch(et,ck,g,encoder,opt,entries,vals[:1] if smoke else vals,loader,epoch,
                        logs/f"epoch_{epoch:04d}_samples.jsonl",weights,limit=1 if smoke else None,
                        observer=(lambda e:events.append({k:tree_hash(e[k]) if torch.is_tensor(e[k]) else e[k]
                            for k in ("record","input_image","adversarial","previous_jvp","previous_reference")})) if smoke else None)
                counts=observation.counts
                expected=progress["optimizer_steps"];val_count=len(progress["validation"])
                need(counts=={"G":expected+val_count,"IRRA":expected+val_count,"backward":expected,"step":expected},"actual forward/backward/step counts")
                expected_adam=80680+expected if smoke else epoch*8068
                need(adam_steps(opt)==[expected_adam],"complete Adam cumulative steps")
                need(tree_hash(g.state_dict())!=before_g and tree_hash(encoder.state_dict())==frozen
                    and not encoder.training and all(not p.requires_grad and p.grad is None for p in encoder.parameters()),"G changed; IRRA frozen")
                checked(FileRef(**c["parent_checkpoint"]));oldio.verify(contract["original_training_contract"]["training_source_sha256"])
                verified_cache=et.checked_cache(c["cache"],require_full=True)
                need(et.contract_for(verified_cache,c["cache"],contract["original_training_contract"]["config"])==
                    contract["original_training_contract"],"cache/source changed during epoch")
                del verified_cache
                summary=summarize(progress,time.monotonic()-begin,opt.param_groups[0]["lr"])
                summary.update(Adam_step=expected_adam,counts=counts,IRRA_parameters_buffers_exact=True,
                    IRRA_input_gradient_norms=observation.input_gradient if smoke else {
                        "min":min(observation.input_gradient),"max":max(observation.input_gradient)},
                    maximum_linf=max(t["cumulative_linf"] for s in progress["sample_logs"] for t in s["rounds"]),
                    zero_target_feedback=True)
                if smoke:
                    need(counts["step"]==c["smoke_member"]["K"]+1 and
                        any(t["G_gradient_l2"]>0 for t in progress["sample_logs"][0]["rounds"]),"single real sample gradient")
                    n=expected
                    need(observation.inputs[0]==observation.inputs[n-1],"last Joint restarts original input")
                    for i in range(n-1):
                        need(observation.parameter_versions[i]!=observation.parameter_versions[i+1],"G not immediately updated")
                    for i in range(1,n-1):need(observation.inputs[i]==events[i-1]["adversarial"],"serial input not previous detached image")
                    probe={"schema":"gv2-epoch11-smoke-probe-1","completed_epoch":None,"absolute_training_epoch":11,
                        "samples_updated":1,"resume_as_formal_allowed":False,"contract":contract,
                        "generator":ck.cpu_tree(g.state_dict()),"optimizer":ck.cpu_tree(opt.state_dict()),
                        "rng":ck.rng_state(),"progress":progress}
                    store.publish("probe_state",{"state.pt":tensor_bytes(probe)},{"kind":"partial-smoke-state-not-an-epoch"},
                        sources=(ref(progress["sample_log_path"]),ref(c["path"]),FileRef(**c["parent_checkpoint"])))
                    store.verify("probe_state")
                    loaded=torch.load(root/"probe_state/state.pt",map_location="cpu",weights_only=False)
                else:
                    value=pack(g,opt,epoch,contract,progress,ck)
                    checkpoint=save_checkpoint(cs,value);loaded=load_checkpoint(cs,epoch,contract)
                    summary["checkpoint"]=asdict(checkpoint);del value
                copy_g=et.ImageOnlyGenerator(**contract["original_training_contract"]["config"]["architecture"]).to("cuda")
                copy_opt=torch.optim.Adam(copy_g.parameters(),**contract["original_training_contract"]["config"]["optimizer"])
                restored=restore(loaded,copy_g,copy_opt,ck)
                need(tree_hash(copy_g.state_dict())==tree_hash(g.state_dict()) and tree_hash(copy_opt.state_dict())==tree_hash(opt.state_dict()),"live/saved exact state")
                del loaded,copy_g,copy_opt
                summary.update(restore_check=restored,seconds=time.monotonic()-begin,
                    GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                if smoke:summary.update(scope="smoke",completed_epoch=None,formal_epoch11_published=False,
                    first_sample_is_absolute_epoch11_prefix=True,events=events,G_forward_input_sha256=observation.inputs)
                store.publish("probe_result" if smoke else f"epoch_{epoch:04d}",{"summary.json":json_bytes(summary)},
                    {"kind":"gv2-smoke-diagnostics" if smoke else "gv2-epoch-diagnostics"},sources=(ref(progress["sample_log_path"]),))
                print("GV2_SMOKE_UPDATE_COMPLETE" if smoke else "GV2_EPOCH_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
                if stop["requested"]:
                    if not smoke:
                        store.publish(f"boundary_stop_{epoch:04d}",{"report.json":json_bytes({"completed_epoch":epoch,"next_epoch":epoch+1,
                            "safe_boundary":True})},{"kind":"requested-safe-stop"},sources=(checkpoint,))
                        print("GV2_SAFE_EPOCH_BOUNDARY_STOP",epoch,flush=True);return
                    need(False,"stop requested during Smoke")
            protection_result=check_snapshot(protection)
            from tbps.engines.replay_control import rss_bytes
            report={"status":"PASS","scope":"smoke" if smoke else "formal",
                "parent_checkpoint":c["parent_checkpoint"],"initial_restore":initial,"final_restore":summary["restore_check"],
                "completed_epoch":None if smoke else 60,"absolute_training_epoch":11 if smoke else 60,
                "formal_epoch11_published":not smoke,"counts":counts,"Adam_step":expected_adam,
                "source_G_initial_sha256":first_epoch_source,"IRRA_frozen":True,
                "new_JVP":0,"new_scoring":0,"seconds":time.monotonic()-started,
                "peak_RSS_bytes":rss_bytes(),"GPU_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "GPU_peak_reserved_bytes":torch.cuda.max_memory_reserved(),"protection":protection_result,
                "CUDA_bitwise_historical_trajectory_claim":False}
            store.publish("completed",{"report.json":json_bytes(report)},{"kind":"gv2-resume-smoke" if smoke else "gv2-training-completed"},
                sources=(ref(c["path"]),FileRef(**c["parent_checkpoint"])))
            print("GV2_EPOCH60_GPU_SMOKE_PASS" if smoke else "GV2_EPOCH60_TRAINING_PASS",flush=True)
        except BaseException as exc:
            failure(store,c,exc,counts);raise
        finally:
            for s,h in old_handlers.items():signal.signal(s,h)

def audit_smoke(c):
    contract,_,m,_=preflight(c);root=Path(c["smoke_output"]);store=ArtifactStore(root,new_base=c["output_base"])
    need(not store.pending() and not (root/"failure").exists(),"Smoke failed/pending")
    for key in ("protocol","initial_restore","probe_claim","probe_state","probe_result","completed"):store.verify(key)
    p=torch.load(root/"probe_state/state.pt",map_location="cpu",weights_only=False)
    need(p["schema"]=="gv2-epoch11-smoke-probe-1" and p["completed_epoch"] is None
        and p["resume_as_formal_allowed"] is False and p["absolute_training_epoch"]==11 and p["contract"]==contract,"probe not a formal epoch")
    validate_progress(p["progress"],formal=False)
    need(p["progress"]["samples"]==1 and p["progress"]["sample_order"]==[c["smoke_member"]["row_id"]],"one fixed training member")
    need(adam_steps(p["optimizer"])==[80680+c["smoke_member"]["K"]+1],"probe Adam state")
    initial=read_json(root/"initial_restore/report.json");result=read_json(root/"completed/report.json")
    need(all(initial[k] for k in ("G_exact","Adam_exact","RNG_exact")) and initial["Adam_steps"]==[80680]
        and result["status"]=="PASS" and result["completed_epoch"] is None,"actual exact restore")
    need(not Path(c["formal_output"]).exists(),"formal training was started")
    protected=check_snapshot(read_json(root/"protocol/protection.json"))
    store.publish("cpu_audit",{"report.json":json_bytes({"status":"PASS","CUDA":False,
        "formal_epoch11_published":False,"probe_steps":c["smoke_member"]["K"]+1,"protection":protected})},
        {"kind":"independent-smoke-CPU-audit"},sources=(ref(root/"completed/manifest.json"),ref(root/"probe_state/manifest.json")))
    print("GV2_EPOCH60_SMOKE_CPU_AUDIT_PASS",flush=True)

def audit_train(c):
    contract,_,_,_=preflight(c);root=Path(c["formal_output"]);store=ArtifactStore(root,new_base=c["output_base"])
    store.verify("completed");need(read_json(root/"completed/report.json")["completed_epoch"]==60,"training incomplete")
    cs=ArtifactStore(root/"checkpoints",new_base=c["output_base"])
    for epoch in range(11,61):
        store.verify(f"claim_{epoch:04d}");load_checkpoint(cs,epoch,contract)
    protected=check_snapshot(read_json(root/"protocol/protection.json"))
    store.publish("cpu_audit",{"report.json":json_bytes({"status":"PASS","epochs":50,"Adam_step":484080,
        "CUDA":False,"protection":protected})},{"kind":"independent-training-CPU-audit"},
        sources=(ref(root/"completed/manifest.json"),))
    print("GV2_EPOCH60_TRAIN_CPU_AUDIT_PASS",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--action",choices=("preflight","smoke","smoke-audit","train","train-audit"),default="preflight")
    p.add_argument("--allow-gpu-smoke",action="store_true");p.add_argument("--allow-training",action="store_true")
    p.add_argument("--resume",action="store_true");a=p.parse_args();c=config(a.config)
    if a.action=="preflight":preflight(c)
    elif a.action=="smoke":need(a.allow_gpu_smoke,"explicit GPU Smoke flag");execute(c,smoke=True)
    elif a.action=="smoke-audit":audit_smoke(c)
    elif a.action=="train":need(a.allow_training,"formal training requires separate authorization and flag");execute(c,resume=a.resume)
    else:audit_train(c)
if __name__=="__main__":main()
