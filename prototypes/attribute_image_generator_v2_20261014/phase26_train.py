"""仅授权Epoch 1的正式入口；持久化总目标10，禁止自动续跑。"""
import argparse,json,os,subprocess,threading,time,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import cache_io as io
import epoch_training as et
import epoch_checkpoint as ck
import irra_image_adapter as adapter
from image_only_generator import ImageOnlyGenerator
from budget_projection import DEFAULT_EPSILON
from training_data import checked_cache,make_loader,check_pair,RAW,SPLIT
from losses import joint_losses
from inference_once import generate_once
from phase26_support import state_hash,summarize,validation_stats

CONFIG=io.ROOT/"config_phase26.json"
CACHE=io.OUTPUTS/"conditions_full"
OUTPUT=io.OUTPUTS/"train_v2"
EXPECTED_MANIFEST="506caf30b8f82b8a29fc35e06df7bb04a5b2bcddc58a1f159b524bb107e7511e"
EXPECTED_PROTOCOL="9ad4903a6de6dc221df1a15af48d9e63e6332d7fcd3a9a62ecd3f11a2ed91c9d"

def formal_configuration():
    config=et.configuration("formal",10,42,"cuda")
    config["architecture"]["epsilon"]=DEFAULT_EPSILON
    config["data"]={"train":2048,"validation":128,"manifest_sha256":EXPECTED_MANIFEST,"protocol_sha256":EXPECTED_PROTOCOL}
    config["surrogate"]={"name":"official IRRA","checkpoint_sha256":adapter.CHECKPOINT_SHA,"precision":"fp32","frozen":True}
    config["algorithm"]="A";config["budget_epsilon"]=DEFAULT_EPSILON
    config["entry_source_sha256"]=et.file_contract(("phase26_train.py","phase26_support.py"))
    return config

def preflight():
    config=io.load(CONFIG)
    if io.digest(config)!=io.digest(formal_configuration()):raise ValueError("frozen formal config/source changed")
    m=checked_cache(CACHE,require_full=True)
    if io.sha(CACHE/"manifest.json")!=EXPECTED_MANIFEST or m["protocol_sha256"]!=EXPECTED_PROTOCOL:
        raise ValueError("Phase 2.5 cache SHA changed")
    raw=io.load(RAW);split=io.load(SPLIT)
    for e in m["entries"]:
        c,meta,_=io.read_entry(e,io.digest(m["protocol"]));check_pair(e,c,meta,raw,split)
    io.verify(io.load(io.OUTPUTS/"phase26/protected_sources.json"))
    return config,m

def launch_once(runner,cache,output,config,**kwargs):
    return runner(cache,output,config,stop_after_epochs=1,**kwargs)

def finite_optimizer(opt):
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():raise ValueError("nonfinite Adam state")

def run():
    config,m=preflight()
    if OUTPUT.exists():raise ValueError("formal output exists; this entry never automatically resumes/starts Epoch 2")
    OUTPUT.mkdir()
    io.save_json(OUTPUT/"frozen_config.json",config)
    io.save_json(OUTPUT/"execution_limit.json",{"epochs_this_invocation":1,"total_target":10,"Epoch2_authorized":False})
    loader=make_loader(m,"cuda")
    validation=[e for e in m["entries"] if e["role"]=="validation"]
    handles=[];box={};records=[];baseline=[];phase="initial_validation"
    counts={"train_G":0,"train_IRRA":0,"backward":0,"step":0,"zero_grad":0,
            "initial_G":0,"initial_IRRA":0,"validation_G":0,"validation_IRRA":0,"samples":0}
    context={};versions=None;stop=threading.Event();started=time.perf_counter()
    round_file=(OUTPUT/"round_audit.jsonl").open("x",encoding="utf-8")
    def monitor():
        with (OUTPUT/"resource_monitor.jsonl").open("x",encoding="utf-8") as f:
            while True:
                row={"seconds":time.perf_counter()-started,"samples":counts["samples"],"steps":counts["step"]}
                if torch.cuda.is_initialized():
                    row.update(allocated_MiB=torch.cuda.memory_allocated()/2**20,reserved_MiB=torch.cuda.memory_reserved()/2**20)
                try:
                    for name,query in (("GPU","--query-gpu=index,memory.used,memory.total,utilization.gpu"),
                                       ("processes","--query-compute-apps=pid,used_memory")):
                        p=subprocess.run(["nvidia-smi",query,"--format=csv,noheader"],capture_output=True,text=True,timeout=5)
                        row[name]=p.stdout.strip()
                except Exception as exc:row["monitor_error"]=repr(exc)
                f.write(json.dumps(row)+"\n");f.flush()
                if stop.wait(10):break
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    original_g=et.ImageOnlyGenerator;original_encoder=adapter.load_official;original_sample=et.train_single_sample
    original_validation=et.validate_epoch;original_save=et.save_epoch;Adam=torch.optim.Adam
    def before_g(module,args):
        if phase=="train":
            counts["train_G"]+=1
            if args[0].requires_grad or args[0].grad_fn is not None or not torch.equal(args[0],context["expected_image"]):
                raise ValueError("wrong serial detached input/original joint restart")
            if versions is not None and versions!=tuple(p._version for p in module.parameters()):
                raise ValueError("next G forward not using immediately updated parameters")
        else:counts["initial_G" if phase=="initial_validation" else "validation_G"]+=1
    def encoder_before(module,args):
        if phase=="train":
            counts["train_IRRA"]+=1
            if not args[0].requires_grad:raise ValueError("IRRA input gradient detached")
            def image_grad(grad):
                if not torch.isfinite(grad).all():raise ValueError("nonfinite IRRA input gradient")
                context["image_grad_norm"]=float(grad.double().norm())
            args[0].register_hook(image_grad)
        else:
            counts["initial_IRRA" if phase=="initial_validation" else "validation_IRRA"]+=1
            if torch.is_grad_enabled() or args[0].requires_grad:raise ValueError("validation gradient enabled")
    def g_factory(*args,**kwargs):
        g=original_g(*args,**kwargs);box["g"]=g
        handles.append(g.register_forward_pre_hook(before_g));return g
    def load_encoder(*args,**kwargs):
        encoder=original_encoder(*args,**kwargs);box["encoder"]=encoder
        box["frozen_encoder"]=state_hash(encoder.state_dict())
        handles.append(encoder.register_forward_pre_hook(encoder_before))
        g=box["g"];rng_before=state_hash(ck.rng_state())
        with torch.no_grad():
            for e in validation:
                image,c=loader(e);z=c["z_t"][None].cuda();v=c["v_local"][None].cuda()
                mask=torch.tensor([c["has_local"]],device="cuda")
                clean=joint_losses(encoder(image),z,v,mask)
                adv=generate_once(g,image)["adversarial"];initial=joint_losses(encoder(adv),z,v,mask)
                baseline.append({"row_id":e["row_id"],"clean_global":float(clean["global"]),"clean_local":float(clean["local"]),
                    "initial_global":float(initial["global"]),"initial_local":float(initial["local"])})
        if rng_before!=state_hash(ck.rng_state()):raise ValueError("validation changed RNG")
        io.save_json(OUTPUT/"validation_baseline.json",baseline)
        box["G_initial_sha256"]=state_hash(g.state_dict());torch.cuda.reset_peak_memory_stats()
        return encoder
    class AuditAdam(Adam):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);box["opt"]=self
            def backward(grad):
                if not torch.isfinite(grad).all():raise ValueError("nonfinite G gradient")
                counts["backward"]+=1
            handles.append(next(box["g"].parameters()).register_hook(backward))
        def zero_grad(self,*args,**kwargs):
            counts["zero_grad"]+=1;return super().zero_grad(*args,**kwargs)
        def step(self,*args,**kwargs):
            nonlocal versions
            with torch.no_grad():
                before=torch.cat([p.detach().reshape(-1) for p in box["g"].parameters()])
                result=super().step(*args,**kwargs)
                after=torch.cat([p.detach().reshape(-1) for p in box["g"].parameters()])
                if not torch.isfinite(after).all():raise ValueError("nonfinite updated G")
                context["changed"]=int(torch.count_nonzero(before!=after))
            counts["step"]+=1;versions=tuple(p._version for p in box["g"].parameters())
            return result
    def sample(g,encoder,opt,image,c,**kwargs):
        nonlocal phase
        phase="train";position=counts["samples"]+1;k=c["K"]
        context.update(row_id=c["row_id"],K=k,expected_image=image.detach(),previous_feature=None)
        before=counts["step"]
        def observe(event):
            r=event["record"];idx=r["update_index"]-1
            if r["phase"]=="attribute":
                if idx:
                    v=c["attribute_jvps"][idx-1:idx].cuda();ref=(context["previous_feature"]*v).sum(1)
                    if not torch.equal(event["previous_jvp"],v) or not torch.equal(event["previous_reference"],ref):
                        raise ValueError("Retain previous attribute/reference mismatch")
                context["expected_image"]=image.detach() if idx==k-1 else event["adversarial"]
            elif not torch.equal(event["input_image"],image):raise ValueError("joint restart mismatch")
            context["previous_feature"]=event["feature"]
            record={**r,"row_id":c["row_id"],"K":k,"sample_position":position,
                "changed_parameter_count":context["changed"],"IRRA_input_gradient_l2":context["image_grad_norm"]}
            if record["cumulative_linf"]>DEFAULT_EPSILON:raise ValueError("budget violation")
            records.append(record);round_file.write(json.dumps(record,allow_nan=False)+"\n")
        result=original_sample(g,encoder,opt,image,c,observer=observe,**kwargs)
        if counts["step"]-before!=k+1:raise ValueError("K+1 update count")
        counts["samples"]+=1;round_file.flush();os.fsync(round_file.fileno())
        if counts["samples"]%32==0:print("TRAIN_PROGRESS",counts["samples"],"steps",counts["step"],"seconds",time.perf_counter()-started,flush=True)
        return result
    def validate(*args,**kwargs):
        nonlocal phase
        phase="validation";before_g=state_hash(box["g"].state_dict());before_o=state_hash(box["opt"].state_dict())
        result=original_validation(*args,**kwargs)
        if state_hash(box["g"].state_dict())!=before_g or state_hash(box["opt"].state_dict())!=before_o:
            raise ValueError("validation changed G/Adam")
        io.save_json(OUTPUT/"validation_epoch1.json",result);return result
    def save(out,g,opt,epoch,contract,progress):
        if epoch!=1 or counts["step"]!=8068 or counts["samples"]!=2048:raise ValueError("formal Epoch 1 counts")
        finite_optimizer(opt)
        if state_hash(box["encoder"].state_dict())!=box["frozen_encoder"]:raise ValueError("IRRA changed")
        round_file.flush();os.fsync(round_file.fileno())
        stop.set();thread.join(timeout=12)
        if thread.is_alive():raise ValueError("resource log writer did not stop")
        progress["supplementary_log_sha256"]={str(OUTPUT/name):io.sha(OUTPUT/name) for name in
            ("round_audit.jsonl","validation_baseline.json","validation_epoch1.json","resource_monitor.jsonl")}
        return original_save(out,g,opt,epoch,contract,progress)
    try:
        with patch.object(et,"ImageOnlyGenerator",new=g_factory),patch.object(adapter,"load_official",new=load_encoder), \
             patch.object(et.torch.optim,"Adam",new=AuditAdam),patch.object(et,"train_single_sample",new=sample), \
             patch.object(et,"validate_epoch",new=validate),patch.object(et,"save_epoch",new=save), \
             patch.object(torch.autograd,"grad",side_effect=AssertionError("JVP/scoring grad forbidden")):
            g,opt,progress=launch_once(et.run_epochs,CACHE,OUTPUT,config,loader=loader)
        torch.cuda.synchronize()
        expected={"train_G":8068,"train_IRRA":8068,"backward":8068,"step":8068,"zero_grad":8068,
                  "initial_G":128,"initial_IRRA":256,"validation_G":128,"validation_IRRA":128,"samples":2048}
        if counts!=expected or progress["completed_epoch"]!=1:raise ValueError("full count mismatch")
        current_g=state_hash(g.state_dict());current_o=state_hash(opt.state_dict());current_rng=state_hash(ck.rng_state())
        restored_g=ImageOnlyGenerator(**config["architecture"]).cuda();restored_o=Adam(restored_g.parameters(),**config["optimizer"])
        restored=ck.load_latest(OUTPUT,progress["contract"],restored_g,restored_o)
        restored_exact=(current_g==state_hash(restored_g.state_dict()) and current_o==state_hash(restored_o.state_dict())
                        and current_rng==state_hash(ck.rng_state()))
        if not restored_exact or restored["next_epoch"]!=2 or restored["contract"]["config"]["epochs"]!=10:
            raise ValueError("formal checkpoint boundary restore mismatch")
        finite_optimizer(restored_o)
        steps=sorted({int(s["step"]) for s in restored_o.state.values()})
        if steps!=[8068]:raise ValueError("Adam state step count")
        groups=summarize(records)
        summary={"method_audit_passed":True,"config":config,"counts":counts,"completed_epoch":1,"next_epoch":2,"target_epochs":10,
            "statistics":groups,"validation_clean":{"global_mean":sum(r["clean_global"] for r in baseline)/128,
                "local_mean":sum(r["clean_local"] for r in baseline)/128},
            "validation_initial":{"global_mean":sum(r["initial_global"] for r in baseline)/128,
                "local_mean":sum(r["initial_local"] for r in baseline)/128},
            "validation_epoch1":validation_stats(progress["epochs_this_run"][0]["validation"]),
            "checkpoint_restore_G_Adam_RNG_exact":True,"Adam_steps":steps,"IRRA_parameters_buffers_exact":True,
            "G_initial_sha256":box["G_initial_sha256"],"G_final_sha256":current_g,"G_changed":current_g!=box["G_initial_sha256"],
            "seconds":time.perf_counter()-started,"peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20,
            "checkpoint_path":str(OUTPUT/"checkpoints/epoch_0001/checkpoint.pt"),
            "checkpoint_sha256":io.sha(OUTPUT/"checkpoints/epoch_0001/checkpoint.pt"),
            "sample_log_sha256":restored["progress"]["sample_log_sha256"],
            "config_sha256":io.sha(OUTPUT/"frozen_config.json"),"new_JVPs":0,"new_scoring_backwards":0,
            "Epoch2_started":False,"gallery_evaluation":False,"GPU_trajectory_bitwise_claim":False}
        io.verify(io.load(io.OUTPUTS/"phase26/protected_sources.json"));checked_cache(CACHE,require_full=True)
        io.save_json(OUTPUT/"phase26_summary.json",summary)
        io.save_json(OUTPUT/"phase26_completed.json",{"method_audit_passed":True,"summary_sha256":io.sha(OUTPUT/"phase26_summary.json")})
        print("FORMAL_EPOCH1_COMPLETE",json.dumps({k:v for k,v in summary.items() if k not in ("statistics","config")}),flush=True)
    except BaseException as exc:
        io.save_json(OUTPUT/"phase26_failure.json",{"error":repr(exc),"traceback":traceback.format_exc(),"counts":counts,
            "current":{k:v for k,v in context.items() if k in ("row_id","K")},"Epoch2_started":False})
        raise
    finally:
        stop.set();thread.join(timeout=12);round_file.close()
        for h in handles:h.remove()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--allow-epoch1",action="store_true")
    a=p.parse_args(argv)
    if a.allow_epoch1:run()
    else:
        _,m=preflight();print("CPU_PREFLIGHT",m["counts"],"total_target10; execution_limit1; no model/GPU",flush=True)
if __name__=="__main__":main()
