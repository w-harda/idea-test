"""仅已核验35:1、历史一致round1后的实测数值回归；不设置全局宽容差。"""
from pathlib import Path
import json,math,hashlib
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent

def historical_signature(prior):
    value={k:prior[k] for k in ("fixed_sequence","noise_sha256","random_trace","rng_end_sha256","states")}
    value["round_images"]=[e["post_projection_image_sha256"] for e in prior["rounds"]]
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def load_reference():
    manifest=json.loads((ROOT/"repeatability_reference_35.json").read_text())
    tensors=torch.load(ROOT/manifest["tensor_file"],map_location="cpu",weights_only=True)
    return {"manifest":manifest,"tensors":tensors}

def compare_numerical_repeatability(result,prior,current_images,execution_contract,reference):
    m=reference["manifest"];t=reference["tensors"]
    failures=[]
    def require(ok,message):
        if not ok:failures.append(message)
    require(m["kind"]=="observed_conditional_repeatability","unverified calibration")
    require(historical_signature(prior)==m["historical_signature"],"historical reference changed")
    require(execution_contract==m["execution_contract"],"steps/budget/transform settings changed")
    for key in ("fixed_sequence","noise_sha256","random_trace","rng_end_sha256"):
        require(result[key]==prior[key],key+" changed")
    require(len(result["rounds"])==2 and len(result["states"])==3,"state topology changed")
    if failures:return {"passed":False,"failures":failures}
    require(result["rounds"][0]["post_projection_image_sha256"]==prior["rounds"][0]["post_projection_image_sha256"],
            "round1 outside historical exact condition")
    require(result["states"][:2]==prior["states"][:2],"clean/round1 metrics changed")
    for e,old in zip(result["rounds"],prior["rounds"]):
        require(len(e["steps"])==m["execution_contract"]["steps"],"optimization steps changed")
        for key in ("instance_id","region","semantic_axis","canonical_value","S_raw","source_span","labels_passed","caption_loss_terms"):
            require(e[key]==old[key],"attribute contract "+key+" changed")
        require(e["cumulative_perturbation_linf"]<=m["execution_contract"]["epsilon"]+1e-6,"budget violated")
    for key in ("state_index","positive_image_rank","positive_image_rank_tie_count"):
        require(result["states"][2][key]==prior["states"][2][key],"discrete metric "+key+" changed")
    metric_report={}
    for key,b in m["metric_bounds"].items():
        x=result["states"][2][key];saved=prior["states"][2][key]
        require(math.isfinite(x) and b["min"]<=x<=b["max"],key+" outside observed range")
        metric_report[key]={"current":x,"saved":saved,"absolute_error":abs(x-saved),
                           "relative_error":abs(x-saved)/abs(saved) if saved else None,
                           "observed_min":b["min"],"observed_max":b["max"],"observed_max_absolute_error":b["max_abs"]}
    require(current_images is not None and len(current_images)==3,"actual image tensors required")
    if failures:return {"passed":False,"failures":failures,"metric_comparison":metric_report}
    images=[x.detach().cpu() for x in current_images]
    for i,x in enumerate(images):
        require(x.shape==t["reference"][i].shape and x.dtype==t["reference"][i].dtype,"image shape/dtype changed")
        require(bool(torch.isfinite(x).all()),"nonfinite image")
    if failures:return {"passed":False,"failures":failures,"metric_comparison":metric_report}
    require(torch.equal(images[0],t["reference"][0]),"clean image changed")
    require(torch.equal(images[1],t["reference"][1]),"round1 actual image changed")
    # 非变动坐标的min=max，必须逐值相同；不能用L∞放行整幅图上的任意变化。
    x=images[2];ref=t["reference"][2]
    require(bool(((x>=t["lower"]) & (x<=t["upper"])).all()),"pixel outside observed coordinate envelope")
    delta=x.double()-ref.double()
    absolute=float(delta.abs().max());l2=float(delta.norm());count=int((x!=ref).sum())
    direction=float(F.cosine_similarity((x-images[0]).double().flatten(),(ref-images[0]).double().flatten(),dim=0))
    require(absolute<=m["image_limits"]["max_abs"],"image max_abs outside envelope")
    require(l2<=m["image_limits"]["max_l2"],"image L2 outside envelope")
    require(count<=m["image_limits"]["max_changed_elements"],"too many changed pixels")
    require(direction>=math.nextafter(m["image_limits"]["min_perturbation_cosine"],-math.inf),
            "attack direction outside observed envelope")
    return {"passed":not failures,"failures":failures,"metric_comparison":metric_report,
            "image_comparison":{"max_abs_diff":absolute,"l2_diff":l2,"changed_elements":count,
                                "perturbation_cosine":direction,"limits":m["image_limits"],
                                "coordinate_envelope_checked":True},
            "condition":"historical exact round1; calibration run3/4/5; larger round1 branches remain rejected"}
