"""默认CPU来源审计；显式--allow-smoke仅执行固定5条GPU诊断，不保存权重。"""
import argparse,hashlib,json,time,sys,os,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import cache_io as io
from irra_image_adapter import load_official,pixels_from_pil,MEAN,STD,SIZE
from phase23_inputs import audit_inputs
from image_only_generator import ImageOnlyGenerator
from serial_training import train_single_sample

def fingerprint(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class AuditAdam(torch.optim.Adam):
    def __init__(self,g):
        super().__init__(g.parameters(),lr=2e-4,betas=(.5,.999))
        self.g=g; self.steps=0; self.zeros=0; self.changes=[]; self.versions=None
    def zero_grad(self,*args,**kwargs):
        self.zeros+=1; return super().zero_grad(*args,**kwargs)
    def step(self,*args,**kwargs):
        before=fingerprint(self.g)
        result=super().step(*args,**kwargs)
        after=fingerprint(self.g)
        self.changes.append(before!=after); self.steps+=1
        self.versions=tuple(p._version for p in self.g.parameters())
        return result

def execute_case(encoder,row):
    entry=row["entry"]; c=row["condition"]; k=c["K"]
    with Image.open(row["image_path"]) as im:
        pixels=pixels_from_pil(im).unsqueeze(0).cuda()
        official_pixels=transforms.Compose([transforms.Resize(SIZE),transforms.ToTensor()])(im.convert("RGB")).unsqueeze(0)
    if not torch.equal(pixels.cpu(),official_pixels): raise ValueError("PIL preprocessing mismatch")
    # 训练以外5+5次图像forward单独计数；原图库混合精度仅报告差异。
    with torch.no_grad():
        clean=encoder(pixels)
        normalized=transforms.Normalize(MEAN,STD)(pixels[0]).unsqueeze(0)
        direct=F.normalize(encoder.model.encode_image(normalized).float(),dim=-1)
    if not torch.equal(clean,direct): raise ValueError("adapter vs official fp32 feature mismatch; stop")
    historical=row["clean_gallery_feature"].to(clean.device)
    historical_error=float((clean-historical).abs().max())
    historical_cosine=float(F.cosine_similarity(clean,historical))
    del direct,normalized,clean,historical
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    g=ImageOnlyGenerator().cuda(); opt=AuditAdam(g)
    initial_sha=fingerprint(g)
    if opt.state: raise ValueError("smoke optimizer state must be fresh")
    counts={"G":0,"IRRA":0,"backward":0}; image_gradients=[]; expected=pixels.detach().clone()
    input_checks=[]; reference_checks=[]; previous_feature=None
    def before_G(module,args):
        if args[0].requires_grad or args[0].grad_fn is not None: raise ValueError("non-detached round input")
        if not torch.equal(args[0],expected): raise ValueError("wrong previous/original image input")
        if counts["G"] and tuple(p._version for p in g.parameters())!=opt.versions:
            raise ValueError("next round does not use step-updated G")
        input_checks.append(True); counts["G"]+=1
    def before_encoder(module,args):
        counts["IRRA"]+=1
        if not args[0].requires_grad: raise ValueError("image gradient path detached")
        def gradient(grad):
            if not torch.isfinite(grad).all(): raise ValueError("nonfinite IRRA input gradient")
            image_gradients.append(float(grad.double().norm()))
        args[0].register_hook(gradient)
    def backward(grad):
        if not torch.isfinite(grad).all(): raise ValueError("nonfinite G gradient")
        counts["backward"]+=1
    def observe(event):
        nonlocal expected,previous_feature
        rec=event["record"]; idx=rec["update_index"]-1
        if rec["phase"]=="attribute":
            if idx:
                correct=c["attribute_jvps"][idx-1:idx].to(pixels.device)
                ref=(previous_feature*correct).sum(1)
                if (not torch.equal(event["previous_jvp"],correct)
                    or not torch.equal(event["previous_reference"],ref)
                    or event["previous_reference"].requires_grad):
                    raise ValueError("wrong Retain previous JVP/reference")
            reference_checks.append(True)
            expected=pixels.detach().clone() if idx==k-1 else event["adversarial"].clone()
        else:
            if not torch.equal(event["input_image"],pixels): raise ValueError("joint did not restart original")
        previous_feature=event["feature"].clone()
    handles=[g.register_forward_pre_hook(before_G),encoder.register_forward_pre_hook(before_encoder),
             next(g.parameters()).register_hook(backward)]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    try:
        result=train_single_sample(g,encoder,opt,pixels,c,observer=observe)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        peak_alloc=torch.cuda.max_memory_allocated()/2**20; peak_reserved=torch.cuda.max_memory_reserved()/2**20
    finally:
        for h in handles: h.remove()
    if (counts!={"G":k+1,"IRRA":k+1,"backward":k+1} or opt.steps!=k+1 or opt.zeros!=k+1
        or len(image_gradients)!=k+1): raise ValueError("K+1 count failure")
    if any(not torch.isfinite(torch.tensor(v)) for v in image_gradients) or not any(v>0 for v in image_gradients):
        raise ValueError("missing real IRRA input gradient")
    if any(r["cumulative_linf"]>8/255 for r in result["records"]): raise ValueError("original budget exceeded")
    report={"row_id":entry["row_id"],"role":entry["role"],"K":k,"diagnostic_only":True,
        "validation_not_formal_training":entry["role"]=="validation","counts":counts,
        "optimizer_steps":opt.steps,"zero_grad_calls":opt.zeros,"G_initial_sha256":initial_sha,
        "G_parameter_changed_per_step":opt.changes,"IRRA_input_gradient_l2":image_gradients,
        "records":result["records"],"zero_G_gradient_updates":[r["update_index"] for r in result["records"] if r["G_gradient_l2"]==0],
        "official_fp32_feature_max_abs_error":0.,"PIL_pixels_exact":True,
        "historical_mixed_precision_gallery_max_abs_error":historical_error,
        "historical_mixed_precision_gallery_cosine":historical_cosine,
        "input_parameter_reference_checks_passed":True,"joint_original_exact":True,
        "seconds":seconds,"peak_allocated_MiB":peak_alloc,"peak_reserved_MiB":peak_reserved,
        "checkpoint_saved":False}
    del result,g,opt,pixels,expected,previous_feature
    torch.cuda.empty_cache()
    return report

def same_protocol(saved,current):
    """JSON规范化比较，同时保持所有来源SHA严格。"""
    return io.digest(saved)==io.digest(current)

def run(output):
    rows,locks,audit=audit_inputs()
    locks.update({str(p):io.sha(p) for p in [*io.ROOT.glob("*.py"),io.ROOT/"config.json"]})
    protocol={"schema":"gv2-phase23-smoke-1","audit":audit,"source_sha256":locks,
        "G":{"width":32,"blocks":4,"seed_per_case":42},"Adam":{"lr":2e-4,"betas":[.5,.999]},
        "weights":{"lambda_global":1,"lambda_local":1,"lambda_retain":1},
        "test_used":False,"validation":"independent disposable diagnostic state; never saved"}
    with io.output_lock(output) as out:
        if (out/"completed.json").exists():
            done=io.load(out/"completed.json")
            io.verify({str(out/"summary.json"):done["summary_sha256"],str(out/"protocol.json"):done["protocol_sha256"]})
            if not same_protocol(io.load(out/"protocol.json"),protocol) or not done["method_audit_passed"]: raise ValueError("completed smoke changed")
            print("ALREADY_COMPLETE_CPU_ONLY",flush=True); return
        if (out/"protocol.json").exists(): raise ValueError("incomplete smoke; preserve evidence and use new output")
        io.save_json(out/"protocol.json",protocol)
        cases=[]
        try:
            encoder=load_official()
            frozen_before=fingerprint(encoder)
            with patch.object(encoder.model,"encode_text",side_effect=AssertionError("text forward forbidden")), \
                 patch.object(encoder.model.base_model,"encode_text",side_effect=AssertionError("text forward forbidden")):
                for row in rows:
                    report=execute_case(encoder,row)
                    if fingerprint(encoder)!=frozen_before: raise ValueError("IRRA parameter/buffer state changed")
                    io.save_json(out/(row["entry"]["row_id"].replace(":","_")+".json"),report)
                    cases.append(report)
                    print("GPU_CASE_COMPLETE "+json.dumps(report),flush=True)
            if any(p.requires_grad or p.grad is not None for p in encoder.parameters()) or encoder.training:
                raise ValueError("IRRA not frozen/eval")
            io.verify(locks)
            summary={"method_audit_passed":True,"count":5,"cases":cases,
                "total_training_G_forwards":sum(c["counts"]["G"] for c in cases),
                "total_training_IRRA_forwards":sum(c["counts"]["IRRA"] for c in cases),
                "total_training_backwards":sum(c["counts"]["backward"] for c in cases),
                "total_optimizer_steps":sum(c["optimizer_steps"] for c in cases),
                "extra_clean_adapter_forwards":5,"extra_direct_official_forwards":5,
                "maximum_linf":max(r["cumulative_linf"] for c in cases for r in c["records"]),
                "source_sha_count":len(locks),"source_SHA_unchanged":True,
                "IRRA_parameters_buffers_exact":True,"new_scoring_backwards":0,"new_TopK":0,"new_JVPs":0,
                "text_forwards":0,"formal_training":False,"checkpoints_saved":0,"Phase24_started":False,
                "all_independent_G_initial_SHA_equal":len({c["G_initial_sha256"] for c in cases})==1}
            if not summary["all_independent_G_initial_SHA_equal"]: raise ValueError("independent state init mismatch")
            io.save_json(out/"summary.json",summary)
            io.save_json(out/"completed.json",{"method_audit_passed":True,"summary_sha256":io.sha(out/"summary.json"),
                "protocol_sha256":io.sha(out/"protocol.json")})
            del encoder
            torch.cuda.empty_cache()
            print("PHASE23_COMPLETE "+json.dumps({k:v for k,v in summary.items() if k!="cases"}),flush=True)
        except Exception as error:
            io.save_json(out/"failure.json",{"error":repr(error),"traceback":traceback.format_exc(),
                "completed_cases":[c["row_id"] for c in cases]})
            raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-smoke",action="store_true")
    parser.add_argument("--output",type=Path,default=io.OUTPUTS/"phase23/smoke")
    args=parser.parse_args(argv)
    if args.allow_smoke: run(args.output)
    else:
        _,_,audit=audit_inputs()
        print("CPU_PREFLIGHT "+json.dumps(audit),flush=True)
if __name__=="__main__": main()
