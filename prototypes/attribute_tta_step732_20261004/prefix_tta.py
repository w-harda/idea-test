"""Step7.3.2：仅将current-only NEW改为冻结JVP的prefix算术平均。"""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import torch
REPO = Path("/home/lzf/ldx/projects/idea-TBPS-test1")
P72 = REPO/"prototypes/attribute_tta_step72_20261003"
P73 = REPO/"prototypes/attribute_tta_step73_20261004"
for path in (REPO/"src", P72, P73):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from controlled_tta import ControlledTTA, CategoricalTape, metric_delta, norms, rng_fingerprint
MODES = ("SEQUENTIAL_NEW", "PREFIX_NEW")

def round_members(mode, plan, index):
    if mode not in MODES or not 1 <= index <= len(plan):
        raise ValueError("invalid scheduler/round")
    if [a.selection_rank for a in plan] != list(range(1, len(plan)+1)):
        raise ValueError("frozen selection order invalid")
    return tuple(plan[:index]) if mode == "PREFIX_NEW" else (plan[index-1],)

def cached_round_guidance(mode, plan, vectors):
    """一次JVP/实例；一次aggregate/round；不归一化、不使用S或caption。"""
    out = []
    for i in range(1, len(plan)+1):
        members = round_members(mode, plan, i)
        vs = [vectors[a.instance_id] for a in members]
        if any(v.ndim != 2 or v.shape[0] != 1 or not torch.isfinite(v).all() for v in vs):
            raise ValueError("invalid frozen JVP")
        # 单项直接返回原tensor，维持k=1以及round1逐值一致。
        aggregate = vs[0] if len(vs) == 1 else torch.stack(vs, dim=0).mean(dim=0).detach()
        out.append((members, aggregate))
    return tuple(out)


def _rng_snapshot():
    import random, numpy as np
    return {"python":random.getstate(),"numpy":np.random.get_state(),
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}

def _restore_rng(snapshot):
    import random, numpy as np
    random.setstate(snapshot["python"]);np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot["cuda"]:torch.cuda.set_rng_state_all(snapshot["cuda"])

def execute(source, original, plan, cached, evaluator, *, steps, transforms, scales, tape,
            first_round_capture=None, first_round_checkpoint=None, final_image_capture=None):
    """原ControlledTTA更新不变；可从同一次公共round1的完整状态分支。"""
    engine = ControlledTTA(source, original, "NEW", steps=steps, transforms=transforms, scales=scales, tape=tape)
    current = original.clone()
    if first_round_checkpoint is None:
        first = evaluator(current)
        states, previous, inputs = [dict(state_index=0, **first)], first, []
    else:
        cp=first_round_checkpoint
        if (not plan or cp["instance_id"]!=plan[0].instance_id or
            cp["settings"]!=(steps,transforms,scales) or
            not torch.equal(original,cp["original"]) or
            not torch.equal(cached[0][1],cp["guidance"]) or
            [a.instance_id for a in cached[0][0]]!=[cp["instance_id"]]):
            raise ValueError("common round1 checkpoint inputs changed")
        if not tape.replay or tape.position!=0 or tape.records[:cp["tape_position"]]!=cp["tape_records"]:
            raise ValueError("common round1 replay prefix changed")
        # 使用实际共同更新后的image/momentum/observer和RNG；不重跑或替换其结果。
        current=cp["image"]
        engine.momentum=cp["momentum"]
        engine.previous_actual_change=cp["previous_actual_change"]
        engine.previous_raw_gradient=cp["previous_raw_gradient"]
        engine.events=[copy.deepcopy(cp["event"])]
        tape.position=cp["tape_position"]
        _restore_rng(cp["rng"])
        first=cp["first"];previous=cp["metric"]
        states=copy.deepcopy(cp["states"]);inputs=[original]
    for index,(a, (members, aggregate)) in enumerate(zip(plan, cached)):
        if index==0 and first_round_checkpoint is not None:
            continue
        inputs.append(current)
        current = engine.step(current, a, aggregate, ["attribute_unit:"+a.instance_id])
        metric = evaluator(current)
        event = engine.events[-1]
        event.update({
            "prefix_length": len(members),
            "selected_prefix_length": a.selection_rank,
            "prefix_attribute_instance_ids": [x.instance_id for x in members],
            "attribute_loss_terms": len(members),
            "prefix_arithmetic": "unweighted mean of unchanged NEW JVP loss; no aggregate normalization",
            "guidance_aggregate_norm": norms(aggregate)["l2"],
            "metrics_before": previous, "metrics_after": metric,
            "metric_delta": metric_delta(previous, metric),
        })
        states.append(dict(state_index=a.selection_rank, **metric))
        previous = metric
        if index==0 and first_round_capture is not None:
            if first_round_capture:raise ValueError("common round1 capture must be empty")
            first_round_capture.append({
                "instance_id":a.instance_id,"settings":(steps,transforms,scales),
                "original":original,"guidance":aggregate,"image":current,"momentum":engine.momentum,
                "previous_actual_change":engine.previous_actual_change,
                "previous_raw_gradient":engine.previous_raw_gradient,
                "event":copy.deepcopy(event),"first":first,"metric":metric,"states":copy.deepcopy(states),
                "tape_position":tape.position,"tape_records":copy.deepcopy(tape.records[:tape.position]),
                "rng":_rng_snapshot()})
    trace = tape.finish()
    result = {
        "states": states, "rounds": engine.events,
        "clean_to_final": metric_delta(first, previous),
        "random_trace": trace,
        "noise_sha256": [e["initialization"]["noise_sha256"] for e in engine.events],
        "rng_end_sha256": rng_fingerprint(),
        "fixed_sequence": [e["instance_id"] for e in engine.events],
        "final_linf": norms(current-original)["linf"],
    }
    if final_image_capture is not None:final_image_capture.append(current)
    return result, inputs, engine.momentum

def gradient_decomposition(source, image, members, vectors):
    """同一未变换round输入点的梯度分解；不是实际IG各节点的归因。无RNG调用。"""
    x = image.detach().clone().requires_grad_(True)
    feature = source.encode_images(x)
    terms = [-(feature * vectors[a.instance_id].detach()).sum() for a in members]
    total = torch.stack(terms).sum()
    mean = total / len(terms)
    aggregate, = torch.autograd.grad(mean, x, retain_graph=True)
    gradients = [torch.autograd.grad(term, x, retain_graph=i<len(terms)-1)[0].detach()
                 for i, term in enumerate(terms)]
    reconstructed = torch.stack(gradients).mean(0)
    difference = norms(aggregate-reconstructed)["l2"]
    denominator = norms(aggregate)["l2"]
    return {
        "diagnostic_point": "round input before initialization; untransformed image; measured after all attacks",
        "attribute_terms": [
            {"instance_id": a.instance_id, "semantic_axis": a.metadata["semantic_axis"],
             "region": a.metadata["region"], "canonical_value": a.metadata["canonical_value"],
             "loss": float(term.detach()),
             "unweighted_image_gradient_norm": norms(grad)["l2"],
             "mean_weighted_image_gradient_contribution_norm": norms(grad/len(terms))["l2"]}
            for a, term, grad in zip(members, terms, gradients)],
        "prefix_total_loss": float(total.detach()),
        "prefix_mean_loss": float(mean.detach()),
        "prefix_aggregate_image_gradient_norm": denominator,
        "mean_gradient_reconstruction_absolute_error": difference,
        "mean_gradient_reconstruction_relative_error": difference/denominator if denominator else None,
        "caption_loss_terms": 0,
    }

def attach_decompositions(source, result, inputs, cached, vectors):
    before = rng_fingerprint()
    for event, image, (members, _) in zip(result["rounds"], inputs, cached):
        event["loss_gradient_decomposition"] = gradient_decomposition(source, image, members, vectors)
    if rng_fingerprint() != before:
        raise ValueError("diagnostic consumed RNG")

class RegressionMismatch(ValueError):
    def __init__(self, report):
        self.report = report
        first = report["failures"][0]
        super().__init__("SEQUENTIAL_NEW regression failed: "+str(first))

def saved_regression(result, prior, reference_variants=(), *, numerical_reference=None, current_images=None, execution_contract=None):
    """结构/图像/RNG/整数严格相等；浮点metrics采用项目既有fp32容差。"""
    import math
    atol, rtol = 2e-6, 1e-6  # Step7.2/7.3既有fp32回归容差；不按攻击效果调节。
    failures, differences = [], []
    numerical_check=None
    if numerical_reference is not None:
        from repeatability_gate import compare_numerical_repeatability
        numerical_check=compare_numerical_repeatability(result,prior,current_images,execution_contract,numerical_reference)
        if not numerical_check["passed"]:
            failures.append({"field":"conditional_numerical_contract","comparison":"measured and exact controls",
                             "reasons":numerical_check["failures"]})
    def strict(path, a, b):
        if a != b:
            failures.append({"field":path,"current":a,"saved":b,"comparison":"exact"})
    strict("fixed_sequence", result["fixed_sequence"], prior["fixed_sequence"])
    for key in ("random_trace", "noise_sha256", "rng_end_sha256"):
        strict(key,result[key],prior[key])
    strict("round_count",len(result["rounds"]),len(prior["rounds"]))
    for i,(a,b) in enumerate(zip(result["rounds"],prior["rounds"]),1):
        strict(f"rounds[{i}].post_projection_image_sha256",
               a["post_projection_image_sha256"],b["post_projection_image_sha256"])
    strict("state_count",len(result["states"]),len(prior["states"]))
    for i,(a,b) in enumerate(zip(result["states"],prior["states"])):
        strict(f"states[{i}].keys",sorted(a),sorted(b))
        for key in a.keys() & b.keys():
            # 下方按固定时间/字段顺序重排report；不会给整数rank/tie容差。
            av,bv=a[key],b[key]
            if key in ("paired_cosine","paired_soft_rank_F"):
                absolute=abs(av-bv)
                relative=absolute/abs(bv) if bv else None
                allowed=atol+rtol*abs(bv)
                passed=math.isfinite(av) and math.isfinite(bv) and absolute<=allowed
                if av!=bv or not passed:
                    item={"field":f"states[{i}].{key}","state":i,"current":av,"saved":bv,
                          "absolute_error":absolute,"relative_error":relative,"allowed_absolute_error":allowed,
                          "within_existing_fp32_tolerance":passed}
                    differences.append(item)
                    if not passed:failures.append(item)
            else:
                strict(f"states[{i}].{key}",av,bv)
    # 稳定排序，保证跨进程hash seed不改变第一次state mismatch的记录顺序。
    order={"state_index":0,"paired_cosine":1,"paired_soft_rank_F":2,
           "positive_image_rank":3,"positive_image_rank_tie_count":4}
    differences.sort(key=lambda x:(x["state"],order.get(x["field"].split(".")[-1],99)))
    report={"saved_round_images_bitwise_exact":not any("post_projection_image_sha256" in x["field"] for x in failures),
            "saved_metrics_exact":result["states"]==prior["states"],
            "saved_metrics_allclose_existing_fp32_tolerance":not any(x["field"].startswith("states[") for x in failures),
            "saved_noise_trace_rng_exact":all(result[k]==prior[k] for k in ("random_trace","noise_sha256","rng_end_sha256")),
            "metric_tolerance":{"atol":atol,"rtol":rtol,"source":"Step7.2/7.3既有fp32回归容差；仅cosine/F"},
            "metric_differences":differences,"failures":failures}
    if failures:
        for variant in reference_variants:
            if variant["kind"] != "verified_unmodified_reference_repeat":
                raise ValueError("unaudited reference variant forbidden")
            try:
                accepted=saved_regression(result,variant["reference"])
            except RegressionMismatch:
                continue
            return {"accepted_verified_reference_variant":True,
                    "reference_id":variant["reference_id"],
                    "historical_reference_comparison":report,
                    "accepted_reference_comparison":accepted,
                    "provenance":variant["provenance"]}
        if numerical_reference is not None:
            from repeatability_gate import compare_numerical_repeatability
            check=numerical_check
            if check["passed"]:
                return {"accepted_empirical_repeatability":True,
                        "historical_reference_comparison":report,"repeatability_comparison":check,
                        "scope":"35:1 only; historical exact round1; measured run3/4/5 envelope"}
            report["numerical_reference_rejected"]=check
        raise RegressionMismatch(report)
    if numerical_check is not None:report["repeatability_comparison"]=numerical_check
    return report

def budget_and_prefix_checks(mode, result, plan, epsilon):
    if result["fixed_sequence"] != [a.instance_id for a in plan] or len(result["states"]) != len(plan)+1:
        raise ValueError("fixed plan violated")
    for i, event in enumerate(result["rounds"], 1):
        members = round_members(mode, plan, i)
        if event["prefix_attribute_instance_ids"] != [a.instance_id for a in members] or event["caption_loss_terms"]:
            raise ValueError("prefix/attribute-only contract violated")
        if any(s["cumulative_perturbation_linf"] > epsilon+1e-6 for s in event["steps"]):
            raise ValueError("global step budget violated")
        if event["cumulative_perturbation_linf"] > epsilon+1e-6:
            raise ValueError("global round budget violated")
