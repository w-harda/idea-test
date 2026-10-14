"""三层消融汇总；不按结果修改样本/公式。"""
from collections import Counter
from priority import MODES
from selection_audit import stats,best_prefix,METRICS,bin_for
def compare(cs,left,right):
    out={}
    for key in METRICS:
        ds=[c["modes"][right]["states"][-1][key]-c["modes"][left]["states"][-1][key] for c in cs]
        signs=[-v if key=="paired_cosine" else v for v in ds]
        out[key]={"right_wins":sum(v>0 for v in signs),"left_wins":sum(v<0 for v in signs),
                  "ties":sum(v==0 for v in signs),"right_minus_left":stats(ds)}
    return out
def retention(data):
    bs=[best_prefix(d) for d in data]
    return {"case_count":len(bs),"final_equals_best":sum(b["final_equals_best"] for b in bs),
            "final_weaker_than_best":sum(not b["final_equals_best"] for b in bs),
            "F_best_minus_final":stats(b["F_best_minus_final"] for b in bs),
            "rank_best_minus_final":stats(b["rank_best_minus_final"] for b in bs),
            "best_round_distribution":dict(Counter(str(b["best_round"]) for b in bs))}
def round_stats(events):
    ds=[e["metric_delta"]["paired_soft_rank_F"] for e in events]
    return {"round_count":len(events),"delta_F":stats(ds),
            "delta_rank":stats(e["metric_delta"]["positive_image_rank"] for e in events),
            "delta_cosine":stats(e["metric_delta"]["paired_cosine"] for e in events),
            "enhancement_count":sum(v>0 for v in ds),"recovery_count":sum(v<0 for v in ds),
            "tie_count":sum(v==0 for v in ds),"recovery_ratio":sum(v<0 for v in ds)/len(ds) if ds else None,
            "recovery_magnitude":stats(-v for v in ds if v<0),
            "post_projection_update_l2":stats(e["post_projection_actual_change"]["l2"] for e in events),
            "clipping_ratio":stats(e["projection_clipping_ratio_mean_over_steps"] for e in events)}
def conflict_stats(rows):
    return {"step_count":len(rows),"conflict_count":sum(x["conflict"] for x in rows),
            "no_conflict_count":sum(not x["conflict"] for x in rows),
            "conflict_rate":sum(x["conflict"] for x in rows)/len(rows) if rows else None,
            "projection_intervention_count":sum(x["projection_intervention"] for x in rows),
            "modification_ratio":stats(x["projection_modification_ratio"] for x in rows),
            "conflict_cosine":stats(x["cosine_before"] for x in rows if x["conflict"] and x["cosine_before"] is not None),
            "all_cosine":stats(x["cosine_before"] for x in rows if x["cosine_before"] is not None),
            "dot_after":stats(x["dot_after"] for x in rows),
            "removed_component_norm":stats(x["removed_conflict_component_norm"] for x in rows),
            "no_conflict_projection_modified_count":sum(not x["conflict"] and x["projection_intervention"] for x in rows),
            "no_conflict_exact_failures":sum(not x["conflict"] and x.get("no_conflict_projection_exact") is False for x in rows),
            "real_momentum_path_verified_count":sum(x["actual_momentum_input_verified"] for x in rows)}
def summarize(cases):
    summary={"case_count":len(cases),"modes":{},"comparisons":{},"by_natural_k":{}}
    extended=[c for c in cases if c["natural_k"]>4]
    for left,right in [(MODES[0],MODES[1]),(MODES[1],MODES[2])]:
        summary["comparisons"][right+"_vs_"+left]={"all":compare(cases,left,right),"k_gt4":compare(extended,left,right)}
    for k in sorted(set(c["natural_k"] for c in cases)):
        cs=[c for c in cases if c["natural_k"]==k]
        summary["by_natural_k"][str(k)]={"count":len(cs),"comparisons":{
            right+"_vs_"+left:compare(cs,left,right) for left,right in [(MODES[0],MODES[1]),(MODES[1],MODES[2])]}}
    maxk=max(c["natural_k"] for c in cases)
    rounds={};conflicts={};projection={}
    for mode in MODES:
        ds=[c["modes"][mode] for c in cases]
        summary["modes"][mode]={"clean_to_final":{key:stats(d["clean_to_final"][key] for d in ds) for key in METRICS},
                              "retention_all":retention(ds),"retention_k_gt4":retention([c["modes"][mode] for c in extended])}
        rounds[mode]={str(i):round_stats([d["rounds"][i-1] for d in ds if len(d["rounds"])>=i]) for i in range(1,maxk+1)}
        rounds[mode]["7+"]=round_stats([e for d in ds for e in d["rounds"][6:]])
        tagged=[(c,e,s) for c in cases for e in c["modes"][mode]["rounds"] for s in e["priority_steps"]]
        conflicts[mode]={"all":conflict_stats([s for c,e,s in tagged]),
                         "round2_plus":conflict_stats([s for c,e,s in tagged if e["round_index"]>1]),
                         "by_round":{str(i):conflict_stats([s for c,e,s in tagged if e["round_index"]==i]) for i in range(1,maxk+1)},
                         "by_natural_k":{str(k):conflict_stats([s for c,e,s in tagged if c["natural_k"]==k]) for k in sorted(set(c["natural_k"] for c in cases))},
                         "by_new_axis":{axis:conflict_stats([s for c,e,s in tagged if e["semantic_axis"]==axis]) for axis in sorted(set(e["semantic_axis"] for c,e,s in tagged))}}
        projection[mode]={"all":conflicts[mode]["all"],
                          "history_unchanged":all(s["history_components_unchanged"] for c,e,s in tagged),
                          "only_new_modified":all(s["only_new_attribute_modified"] for c,e,s in tagged),
                          "same_state_no_conflict_counterfactual":"combine() returns exactly the unprojected weighted tensor; not a claim that independent diverged trajectories have identical later gradients",
                          "IG_reconstruction_relative_error":stats(s["IG_reconstruction_relative_error"] for c,e,s in tagged if s.get("IG_reconstruction_relative_error") is not None)}
    return summary,rounds,conflicts,projection
