"""仅固定5条真实缓存的1个受控Epoch smoke；默认只做CPU缓存核验。"""
import argparse,hashlib,json,time,sys,traceback
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
import cache_io as io
import epoch_training as et
import epoch_checkpoint as ck
from training_data import checked_cache,make_loader
SMALL_IDS=("cuhk:26530:0","cuhk:26747:0","cuhk:9955:0","cuhk:26196:0","cuhk:17663:0")
from image_only_generator import ImageOnlyGenerator
from irra_image_adapter import load_official
from inference_once import generate_once,load_generator

def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def run(cache,output):
    m=checked_cache(cache)
    if (set(e["row_id"] for e in m["entries"])!=set(SMALL_IDS)
        or sorted((e["role"],e["K"]) for e in m["entries"])!=[("train",0),("train",1),("train",3),("train",7),("validation",4)]):
        raise ValueError("smoke requires exact five identities and roles/K")
    protected=io.load(io.OUTPUTS/"phase24/protected_sources.json");io.verify(protected)
    protocol={"schema":"gv2-phase24-smoke-1","cache_sha256":io.sha(Path(cache)/"manifest.json"),
        "sources":{**et.file_contract(et.TRAIN_FILES),**et.file_contract(("smoke_phase24.py",))},
        "epochs":1,"formal_training":False,"validation_optimizer_steps":0}
    with io.output_lock(output) as out:
        if (out/"completed.json").exists():
            done=io.load(out/"completed.json")
            io.verify({str(out/"summary.json"):done["summary_sha256"]})
            if io.digest(io.load(out/"protocol.json"))!=io.digest(protocol):raise ValueError("completed smoke protocol changed")
            print("ALREADY_COMPLETE_CPU_ONLY");return
        if (out/"protocol.json").exists():raise ValueError("preserve incomplete smoke; use a new output")
        io.save_json(out/"protocol.json",protocol)
        try:
            encoder=load_official()
            frozen=state_hash(encoder.state_dict())
            torch.manual_seed(42);g=ImageOnlyGenerator().cuda()
            g_initial=state_hash(g.state_dict())
            cfg=et.configuration("smoke",1,42,"cuda")
            loader=make_loader(m,"cuda")
            counts={"G_train":0,"IRRA_train":0,"backward":0,"optimizer_steps":0,"G_validation":0,"IRRA_validation":0}
            expected=None;versions=None;phase="train";sample_reports=[];gradient_norms=[];opt_box=[]
            def before_g(module,args):
                nonlocal versions
                counts["G_"+phase]+=1
                if phase=="train":
                    if args[0].requires_grad or args[0].grad_fn is not None or not torch.equal(args[0],expected):
                        raise ValueError("serial input is not previous detached image / original joint restart")
                    if versions is not None and versions!=tuple(p._version for p in g.parameters()):
                        raise ValueError("next forward did not use updated G")
            def before_encoder(module,args):
                counts["IRRA_"+phase]+=1
                if phase=="train":
                    if not args[0].requires_grad:raise ValueError("IRRA image input gradient detached")
                    args[0].register_hook(lambda grad:gradient_norms.append(float(grad.double().norm())))
                elif args[0].requires_grad:raise ValueError("validation unexpectedly has gradient")
            def backward(grad):
                if not torch.isfinite(grad).all():raise ValueError("nonfinite gradient")
                counts["backward"]+=1
            Adam=torch.optim.Adam
            class AuditAdam(Adam):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs);self.changes=[];opt_box.append(self)
                def step(self,*args,**kwargs):
                    nonlocal versions
                    before=state_hash(g.state_dict());r=super().step(*args,**kwargs)
                    self.changes.append(before!=state_hash(g.state_dict()))
                    counts["optimizer_steps"]+=1;versions=tuple(p._version for p in g.parameters())
                    return r
            original_sample=et.train_single_sample
            def sample(g_,enc,opt,image,c,**kwargs):
                nonlocal expected,versions
                expected=image.detach().clone();previous=None;k=c["K"]
                before_counts=counts.copy();start=time.perf_counter()
                def observe(event):
                    nonlocal expected,previous
                    r=event["record"];idx=r["update_index"]-1
                    if r["phase"]=="attribute":
                        if idx:
                            v=c["attribute_jvps"][idx-1:idx].cuda();ref=(previous*v).sum(1)
                            if (not torch.equal(event["previous_jvp"],v) or not torch.equal(event["previous_reference"],ref)
                                or event["previous_reference"].requires_grad):raise ValueError("Retain reference mismatch")
                        expected=image.detach().clone() if idx==k-1 else event["adversarial"].clone()
                    elif not torch.equal(event["input_image"],image):raise ValueError("joint original restart failed")
                    previous=event["feature"].clone()
                result=original_sample(g_,enc,opt,image,c,observer=observe,**kwargs)
                torch.cuda.synchronize()
                if any(counts[name]-before_counts[name]!=k+1 for name in ("G_train","IRRA_train","backward","optimizer_steps")):
                    raise ValueError("per sample K+1 count mismatch")
                sample_reports.append({"row_id":c["row_id"],"K":k,"optimizer_steps":k+1,
                    "seconds":time.perf_counter()-start,"records":result["records"]})
                return result
            original_validation=et.validate_epoch
            def validation(*args,**kwargs):
                nonlocal phase
                phase="validation";before_g=state_hash(g.state_dict());before_opt=state_hash(opt_box[0].state_dict())
                result=original_validation(*args,**kwargs)
                if before_g!=state_hash(g.state_dict()) or before_opt!=state_hash(opt_box[0].state_dict()):
                    raise ValueError("validation changed generator/optimizer")
                return result
            handles=[g.register_forward_pre_hook(before_g),encoder.register_forward_pre_hook(before_encoder),
                     next(g.parameters()).register_hook(backward)]
            torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            try:
                with patch.object(et,"train_single_sample",side_effect=sample),patch.object(et,"validate_epoch",side_effect=validation), \
                     patch.object(et.torch.optim,"Adam",AuditAdam), \
                     patch.object(encoder.model,"encode_text",side_effect=AssertionError("text supervision forbidden")):
                    g,opt,progress=et.run_epochs(cache,out/"epoch_run",cfg,generator=g,encoder=encoder,loader=loader)
                torch.cuda.synchronize()
                peak={"allocated_MiB":torch.cuda.max_memory_allocated()/2**20,"reserved_MiB":torch.cuda.max_memory_reserved()/2**20}
            finally:
                for h in handles:h.remove()
            expected_counts={"G_train":15,"IRRA_train":15,"backward":15,"optimizer_steps":15,"G_validation":1,"IRRA_validation":1}
            if counts!=expected_counts:raise ValueError("Epoch call counts mismatch")
            if state_hash(encoder.state_dict())!=frozen or any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
                raise ValueError("IRRA parameters/buffers changed")
            before_g=state_hash(g.state_dict());before_opt=state_hash(opt.state_dict());before_rng=state_hash(ck.rng_state())
            restored_g=ImageOnlyGenerator().cuda();restored_opt=Adam(restored_g.parameters(),**cfg["optimizer"])
            restored=ck.load_latest(out/"epoch_run",progress["contract"],restored_g,restored_opt)
            restored_exact=(before_g==state_hash(restored_g.state_dict()) and before_opt==state_hash(restored_opt.state_dict())
                            and before_rng==state_hash(ck.rng_state()))
            if not restored_exact or restored["next_epoch"]!=2:raise ValueError("GPU Epoch boundary restore not exact")
            # 推理模块只加载G；额外1次前向独立计数，不调用IRRA/缓存监督。
            cp=out/"epoch_run/checkpoints/epoch_0001/checkpoint.pt"
            inference_g=load_generator(cp,"cuda");calls=[];h=inference_g.register_forward_hook(lambda *a:calls.append(1))
            image,_=loader(m["entries"][0]);generated=generate_once(inference_g,image);h.remove()
            inference_linf=float((generated["adversarial"].double()-image.double()).abs().max())
            if len(calls)!=1 or inference_linf>8/255:raise ValueError("single forward inference budget")
            records=[r for s in sample_reports for r in s["records"]]
            if any(r["cumulative_linf"]>8/255 for r in records):raise ValueError("budget exceeded")
            if any(not np.isfinite(v) for v in gradient_norms) or not all(opt.changes):raise ValueError("gradient/parameter update failure")
            io.verify(protected);io.verify(protocol["sources"])
            summary={"method_audit_passed":True,"counts":counts,"samples":sample_reports,
                "epoch_order":progress["epochs_this_run"][0]["sample_order"],"updates_by_K":progress["epochs_this_run"][0]["updates_by_K"],
                "validation":progress["epochs_this_run"][0]["validation"],"validation_generator_optimizer_unchanged":True,
                "IRRA_parameters_buffers_exact":True,"G_changed":before_g!=g_initial,"G_parameter_changed_per_step":opt.changes,
                "IRRA_input_gradient_l2":gradient_norms,"zero_G_gradient_rounds":[(s["row_id"],r["update_index"]) for s in sample_reports for r in s["records"] if r["G_gradient_l2"]==0],
                "maximum_linf":max(r["cumulative_linf"] for r in records),
                "maximum_budget_saturation_ratio":max(r["budget_saturation_ratio"] for r in records),
                "maximum_strict_projection_clipping_ratio":max(r["strict_projection_clipping_ratio"] for r in records),
                "GPU_boundary_restore_G_Adam_RNG_exact":restored_exact,"next_epoch":restored["next_epoch"],
                "recovery_contract":"Epoch boundary only; no step-level or CUDA trajectory bitwise guarantee",
                "inference_G_forwards":len(calls),"inference_linf":inference_linf,
                "new_scoring_backwards_this_run":0,"new_JVPs_this_run":0,"text_forwards_this_run":0,
                "GPU_peak":peak,"seconds":time.perf_counter()-start,"protected_source_count":len(protected),
                "formal_training":False,"full_2176_preparation":False,"gallery_evaluation":False}
            io.save_json(out/"summary.json",summary)
            io.save_json(out/"completed.json",{"method_audit_passed":True,"summary_sha256":io.sha(out/"summary.json")})
            print("PHASE24_SMOKE_COMPLETE "+json.dumps({k:v for k,v in summary.items() if k not in ("samples","IRRA_input_gradient_l2")}),flush=True)
        except Exception as error:
            io.save_json(out/"failure.json",{"error":repr(error),"traceback":traceback.format_exc()});raise

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=io.OUTPUTS/"phase24/conditions_small")
    p.add_argument("--output",type=Path,default=io.OUTPUTS/"phase24/smoke")
    p.add_argument("--allow-smoke",action="store_true")
    a=p.parse_args(argv)
    if a.allow_smoke:run(a.cache,a.output)
    else:print("CPU_CACHE_PREFLIGHT",checked_cache(a.cache)["counts"])
if __name__=="__main__":main()
