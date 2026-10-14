"""运行后方法/预算/冻结审计；不是slow equivalence gate。"""
import json
from pathlib import Path
import deps
def audit(cases):
    from priority import MODES
    failures=[];known=[];actual=0;timing={}
    for c in cases:
        if not all(c["checks"].get(k) for k in ["round1_image_momentum_metrics_noise_transform_RNG_exact","IRRA_frozen","AttributeInstance_S_TopK_JVP_unchanged","zero_exact"]):
            failures.append({"caption_id":c["caption_id"],"field":"frozen/shared controls"})
        for mode in MODES:
            for e in c["modes"][mode]["rounds"]:
                if e["configured_steps_per_round"]!=10:failures.append({"caption_id":c["caption_id"],"field":"configured steps"})
                if any(s["cumulative_perturbation_linf"]>8/255+1e-6 for s in e["steps"]):failures.append({"caption_id":c["caption_id"],"field":"epsilon"})
                rows=e["priority_steps"]
                if not all(s["actual_momentum_input_verified"] and s["history_components_unchanged"] and s["only_new_attribute_modified"] for s in rows):
                    failures.append({"caption_id":c["caption_id"],"field":"real gradient path"})
                if mode==MODES[2] and e["round_index"]>1:
                    for s in rows:
                        if s["conflict"]:
                            if s["cosine_after"] is not None and abs(s["cosine_after"])>64*2**-23:
                                failures.append({"caption_id":c["caption_id"],"field":"post-conflict cosine","value":s["cosine_after"]})
                        elif not s["projected_new_unchanged_exact"] or s["projection_component_max_abs"]!=0.:
                            failures.append({"caption_id":c["caption_id"],"field":"no-conflict new gradient changed"})
                        known.append(dict(caption_id=c["caption_id"],round=e["round_index"],**s))
                if mode==MODES[0] or e["round_index"]>1:actual+=len(e["steps"])
            timing.setdefault(str(c["natural_k"]),{})[mode]=[
              {"round":e["round_index"],"seconds_per_step":e["attack_seconds_per_observed_step"],
               "backward_calls":e["priority_steps"][0]["image_gradient_backward_calls"]}
              for e in c["modes"][mode]["rounds"]]
    return {"passed":not failures,"failures":failures,"case_count":len(cases),"actual_optimization_steps":actual,
      "projection_measured_steps":len(known),"conflict_steps":sum(s["conflict"] for s in known),
      "no_conflict_steps":sum(not s["conflict"] for s in known),
      "maximum_conflict_cosine_after":max([abs(s["cosine_after"]) for s in known if s["conflict"] and s["cosine_after"] is not None],default=0.),
      "maximum_linf":max(c["modes"][m]["final_linf"] for c in cases for m in MODES),
      "timing_by_k":timing,"slow_equivalence_gate":False}
def require_smoke():
    from selection_audit import load,verify
    out=deps.ROOT/"outputs/smoke"
    if not (out/"completed.json").exists():raise ValueError("Step753 method smoke must pass before user diagnostic")
    completed=load(out/"completed.json");protocol=load(out/"protocol.json")
    result=audit(load(out/"per_caption_results.json"))
    if not completed.get("method_audit_passed") or not result["passed"] or result["case_count"]<3:
        raise ValueError("Step753 method smoke not passed")
    verify(protocol["frozen_sha256"])
    return result
if __name__=="__main__":
    from selection_audit import load,save
    out=deps.ROOT/"outputs/smoke";result=audit(load(out/"per_caption_results.json"))
    save(out/"method_audit.json",result);print(json.dumps(result,ensure_ascii=False))
