"""只读CPU复核NOOP四步产物；不实例化模型、不执行Optimizer或神经前向。"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

FROZEN=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023")
DIAG=Path("/home/lzf/ldx/outputs/tbps-framework-gh-noop-determinism/noop_determinism_diag_v1")
CONTROL=Path("/home/lzf/ldx/tmp/gh_noop_determinism_diag_v1/execution_20261023_v1")
OUT=Path("/home/lzf/ldx/outputs/tbps-framework-gh-noop-determinism-review/readonly_v1")
LABELS=("A1_repeat_1","A1_repeat_2","A1_cross","B_cross")
SOURCES="06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b"

def require(ok,message):
    if not ok:raise ValueError(message)

def flatten(value,prefix=""):
    import torch
    if isinstance(value,torch.Tensor):return {prefix:value}
    result={}
    if isinstance(value,dict):
        for key,v in value.items():result.update(flatten(v,prefix+"/"+str(key)))
    elif isinstance(value,(list,tuple)):
        for key,v in enumerate(value):result.update(flatten(v,prefix+"/"+str(key)))
    return result

def differences(a,b):
    import torch
    aa,bb=flatten(a),flatten(b)
    require(set(aa)==set(bb),"different Tensor members")
    details=[];ss=0.;rr=0.;qq=0.;mx=0.;changed=0;total=0
    for key in sorted(aa):
        x,y=aa[key],bb[key]
        require(x.device.type==y.device.type=="cpu","CPU-only tensors")
        require(x.shape==y.shape and x.dtype==y.dtype,"shape/dtype conflict")
        dx=x.double()-y.double()
        d=float(torch.linalg.vector_norm(dx.flatten()))
        r=float(torch.linalg.vector_norm(x.double().flatten()))
        q=float(torch.linalg.vector_norm(y.double().flatten()))
        m=float(dx.abs().max()) if dx.numel() else 0.
        c=int(torch.count_nonzero(dx))
        details.append({"tensor":key,"shape":list(x.shape),"dtype":str(x.dtype),"max_abs":m,
            "L2":d,"relative_L2":d/max(r,1e-300),"reference_L2":r,"candidate_L2":q,
            "changed_elements":c,"elements":x.numel(),"exact":bool(torch.equal(x,y))})
        ss+=d*d;rr+=r*r;qq+=q*q;mx=max(mx,m);changed+=c;total+=x.numel()
    return {"exact":all(x["exact"] for x in details),"max_abs":mx,"L2":math.sqrt(ss),
        "relative_L2":math.sqrt(ss)/max(math.sqrt(rr),1e-300),"reference_L2":math.sqrt(rr),
        "candidate_L2":math.sqrt(qq),"changed_elements":changed,"elements":total,"per_tensor":details}

def moment_state(optimizer,kind):
    return {str(key):v[kind] for key,v in optimizer["state"].items()}

def amplification(a,b):
    """只读已保存Adam矩，代数解释最大参数差；绝不调用Optimizer.step。"""
    import torch
    metric=differences(a["G_after"],b["G_after"])
    row=max(metric["per_tensor"],key=lambda r:r["max_abs"])
    name=row["tensor"][1:]
    x,y=a["G_after"][name],b["G_after"][name]
    index=int((x.double()-y.double()).abs().flatten().argmax())
    names=list(a["parameter_gradients"])
    ids=[i for g in a["Adam_after"]["param_groups"] for i in g["params"]]
    require(len(ids)==len(names),"Adam parameter mapping")
    pid=ids[names.index(name)]
    group=next(g for g in a["Adam_after"]["param_groups"] if pid in g["params"])
    lr,beta1,beta2,eps=group["lr"],*group["betas"],group["eps"]
    def value(t):return float(t.flatten()[index])
    result={"parameter":name,"flat_index":index,"group":{"lr":lr,"betas":[beta1,beta2],"eps":eps},
        "before_parameter":value(a["G_before"][name]),"after_A":value(x),"after_B":value(y),
        "actual_parameter_difference":abs(value(x)-value(y))}
    prior=a["Adam_before"]["state"][pid]
    result["prior_exp_avg"]=value(prior["exp_avg"]);result["prior_exp_avg_sq"]=value(prior["exp_avg_sq"])
    predicted=[]
    for side,saved in (("A",a),("B",b)):
        state=saved["Adam_after"]["state"][pid]
        step=float(state["step"]);m=value(state["exp_avg"]);v=value(state["exp_avg_sq"])
        denom=math.sqrt(v)/math.sqrt(1-beta2**step)+eps
        update=lr/(1-beta1**step)*m/denom
        result[side]={"gradient":value(saved["parameter_gradients"][name]),"step":step,"exp_avg":m,
            "exp_avg_sq":v,"sqrt_second_moment":math.sqrt(v),"denominator_with_eps":denom,
            "predicted_subtracted_update_from_saved_moments":update,
            "actual_subtracted_update":result["before_parameter"]-value(saved["G_after"][name])}
        predicted.append(update)
    result["predicted_parameter_difference_from_saved_moments"]=abs(predicted[0]-predicted[1])
    result["no_optimizer_executed"]=True
    return result

def comparison(a,b):
    import torch
    result={}
    for stage,key in (("G_BEFORE","G_before"),("ADAM_BEFORE","Adam_before"),("RNG_BEFORE","rng_before"),
        ("INPUT","image"),("CONDITION","condition"),("G_FORWARD","G_forward"),
        ("PROJECTED_RGB","projected_RGB"),("IRRA_FEATURE","IRRA_feature")):
        result[stage]=differences(a[key],b[key])
    result["LOSS"]=differences({k:torch.tensor(a["loss_record"][k],dtype=torch.float64) for k in ("global","local","retain","total")},
                              {k:torch.tensor(b["loss_record"][k],dtype=torch.float64) for k in ("global","local","retain","total")})
    for stage,key in (("LOSS_TO_FEATURE_GRAD","IRRA_feature_gradient"),("IRRA_INPUT_GRAD","projected_RGB_gradient"),
        ("G_OUTPUT_GRAD","G_forward_gradient"),("G_PARAMETER_GRAD","parameter_gradients"),
        ("G_AFTER","G_after"),("ADAM_AFTER","Adam_after")):
        result[stage]=differences(a[key],b[key])
    for kind in ("exp_avg","exp_avg_sq","step"):
        result["Adam_"+kind]=differences(moment_state(a["Adam_after"],kind),moment_state(b["Adam_after"],kind))
    pre=a["G_layer_output_gradients"]["up1.2"];pre2=b["G_layer_output_gradients"]["up1.2"]
    post=a["G_layer_input_gradients"]["up2.0/0"];post2=b["G_layer_input_gradients"]["up2.0/0"]
    boundary={"post_interpolation_gradient":differences(post,post2),
        "pre_interpolation_gradient":differences(pre,pre2),
        "input_activation":differences(a["G_layers"]["up1.2"]["output"],b["G_layers"]["up1.2"]["output"]),
        "output_activation":differences(a["G_layers"]["up2.0"]["input"],b["G_layers"]["up2.0"]["input"]),
        "source":"image_only_generator.py:46 F.interpolate(..., mode=bilinear, align_corners=False)",
        "attribution":"second bilinear backward boundary; no CUDA kernel profiler/isolation; original Smoke kernel NOT_VERIFIED"}
    return {"stages":result,"coarse_first_divergence":next(k for k,v in result.items() if not v["exact"]),
        "second_bilinear_boundary":boundary,
        "layer_forward":differences(a["G_layers"],b["G_layers"]),
        "layer_input_gradients":differences(a["G_layer_input_gradients"],b["G_layer_input_gradients"]),
        "layer_output_gradients":differences(a["G_layer_output_gradients"],b["G_layer_output_gradients"]),
        "effective_G_modes_exact":a["effective_G_modes"]==b["effective_G_modes"],
        "IRRA_modes_exact":a["IRRA_modes"]==b["IRRA_modes"],
        "full_RNG_before_exact":a["rng_before_sha256"]==b["rng_before_sha256"],
        "full_RNG_after_exact":__import__("tbps.alternating.state",fromlist=["tree_sha"]).tree_sha(a["rng_after"])==__import__("tbps.alternating.state",fromlist=["tree_sha"]).tree_sha(b["rng_after"]),
        "Adam_amplification":amplification(a,b)}

def review():
    import torch
    from tbps.artifacts import ArtifactStore,json_bytes,FileRef,sha_file,digest,read_json
    from tbps.alternating.state import tree_sha
    require(not torch.cuda.is_initialized(),"CPU review must not initialize CUDA")
    require(not OUT.exists(),"read-only review identity consumed")
    require((CONTROL/"GPU.exit").read_text().strip()=="0" and not (DIAG/"FAIL").exists(),"native diagnostic failed")
    units=[]
    for p in sorted(DIAG.rglob("manifest.json")):
        ArtifactStore(p.parent.parent,new_base=DIAG.parent).verify(p.parent.name)
        units.append({"path":str(p),"sha256":sha_file(p)})
    require(not list(DIAG.rglob("*.pending*")),"pending")
    native=read_json(DIAG/"completed/result.json");plan=read_json(DIAG/"claim/plan.json")
    require(plan["git_commit"]=="8419b8c28a39f2ab148e2059ab6bd8587239c2c6" and plan["source_key"]==SOURCES,"diagnostic sources")
    require(native["status"]=="NOOP_DIAGNOSTIC_COMPLETE" and native["counts"]=={"G":4,"encoder":4,"Adam":4},"actual budgets")
    require(native["native_counts"]=={"G":4,"IRRA_image":4,"IRRA_text":0,"reader":0,"H":0,"JVP":0},"neural budgets")
    require(native["frozen_sources"] and native["original_smoke_control_metadata_unchanged"],"native freeze/protection")
    def forbidden(*a,**k):raise ValueError("CPU review forbids CUDA/neural/derivative/Optimizer")
    with ExitStack() as stack:
        for owner,name in ((torch.cuda,"init"),(torch.cuda,"_lazy_init"),(torch.nn.Module,"_call_impl"),
            (torch.Tensor,"backward"),(torch.autograd,"grad"),(torch.optim.Adam,"step")):
            stack.enter_context(patch.object(owner,name,forbidden))
        runs={label:torch.load(DIAG/label/"tensors.pt",map_location="cpu",weights_only=False) for label in LABELS}
        pre=read_json(DIAG/"claim/preflight.json")["parent_restore"]
        checks={}
        for label,x in runs.items():
            require(all(t.device.type=="cpu" and torch.isfinite(t).all() for t in flatten(x).values()),"finite CPU tensors")
            require(tree_sha(x["G_before"])==pre["G_state"] and tree_sha(x["Adam_before"])==pre["G_Adam"],"full parent state")
            require(x["rng_before_sha256"]==tree_sha(x["rng_before"])==pre["G_RNG"],"full restored RNG")
            require(not any(x["G_pending_gradients_before"].values()),"residual gradients")
            require(x["native_log"]["K"]==0 and x["native_log"]["optimizer_steps"]==1,"K0 single update")
            require(not any(x["IRRA_modes"]["training"].values()) and not any(x["IRRA_modes"]["requires_grad"].values()),"IRRA modes")
            require(all(float(v["step"])==484081 for v in x["Adam_after"]["state"].values()),"Adam absolute step")
            require(x["condition"]["K"]==0 and x["image"].shape==(1,3,384,128),"NOOP input")
            require(x["image"].dtype==x["projected_RGB"].dtype==torch.float32,"FP32")
            linf=float((x["projected_RGB"].double()-x["image"].double()).abs().max())
            require(linf<=8/255,"pixel budget")
            checks[label]={"parent_G":pre["G_state"],"parent_Adam":pre["G_Adam"],
                "parent_RNG":pre["G_RNG"],"after_G":tree_sha(x["G_after"]),"after_Adam":tree_sha(x["Adam_after"]),
                "after_rng":tree_sha(x["rng_after"]),"Adam_steps":[484081],"linf":linf,
                "loss_record":x["loss_record"],"runtime":x["runtime"],
                "before_G_training_flags":x["G_modes_before_train"]["training"],
                "effective_G_training_flags":x["effective_G_modes"]["training"]}
        comp={"same_arm":comparison(runs["A1_repeat_1"],runs["A1_repeat_2"]),
              "cross_arm":comparison(runs["A1_cross"],runs["B_cross"])}
        for pair,c in comp.items():
            for stage,stat in native["comparisons"][pair]["stages"].items():
                ours=c["stages"][stage]
                require(ours["exact"]==stat["exact"] and ours["max_abs"]==stat["max_abs"]
                    and ours["changed_elements"]==stat["changed_elements"],"native/independent exact stat mismatch")
                require(math.isclose(ours["L2"],stat["L2"],rel_tol=1e-12,abs_tol=1e-18),"CPU L2 recomputation")
        require(not torch.cuda.is_initialized(),"review initialized CUDA")
    files=[{"path":str(p),"sha256":sha_file(p),"bytes":p.stat().st_size,
        "mtime_ns":p.stat().st_mtime_ns,"inode":p.stat().st_ino} for p in sorted(DIAG.rglob("*")) if p.is_file()]
    output={"status":"NOOP_INDEPENDENT_CPU_REVIEW_PASS","comparison_source":"saved full tensors; independent CPU reduction",
        "checks":checks,"comparisons":comp,"native_report":native,"plan":plan,"artifact_units":len(units),
        "artifact_manifests":units,"all_artifact_files":files,
        "SAME_ARM_REPEAT_EXACT":all(v["exact"] for k,v in comp["same_arm"]["stages"].items()),
        "SAME_NOOP_GRADIENT_EXACT":comp["cross_arm"]["stages"]["G_PARAMETER_GRAD"]["exact"],
        "SAME_NOOP_GRADIENT_NUMERIC":{k:v for k,v in comp["cross_arm"]["stages"]["G_PARAMETER_GRAD"].items() if k!="per_tensor"},
        "SAME_NOOP_UPDATE_EXACT":comp["cross_arm"]["stages"]["G_AFTER"]["exact"],
        "SAME_NOOP_UPDATE_NUMERIC":{k:v for k,v in comp["cross_arm"]["stages"]["G_AFTER"].items() if k!="per_tensor"},
        "FIRST_DIVERGENCE_STAGE":"G backward: second bilinear interpolation boundary, up2.0 input grad exact / up1.2 output grad differs",
        "NONDETERMINISTIC_KERNEL_CONFIRMED":False,"STATE_LEAK_CONFIRMED":False,
        "current_original_environment_nonbitwise_repeatability_observed":True,
        "original_smoke_specific_kernel_causality":"NOT_VERIFIED","no_more_GPU_execution":True}
    store=ArtifactStore(OUT,new_base=OUT.parent)
    store.publish("completed",{"results.json":json_bytes(output)},{"review_script":sha_file(__file__),"original_source":SOURCES},
        sources=(FileRef(str(Path(__file__).resolve()),sha_file(__file__)),
                 FileRef(str(DIAG/"completed/manifest.json"),sha_file(DIAG/"completed/manifest.json"))))
    print("NOOP_INDEPENDENT_CPU_REVIEW_PASS",flush=True)
    print(json.dumps({"units":len(units),"files":len(files),"first_divergence":output["FIRST_DIVERGENCE_STAGE"],
        "same_arm":comp["same_arm"]["stages"]["G_PARAMETER_GRAD"],
        "cross_arm":comp["cross_arm"]["stages"]["G_PARAMETER_GRAD"]},ensure_ascii=False)[:1000],flush=True)

if __name__=="__main__":
    sys.path.insert(0,str(FROZEN/"src"));review()
