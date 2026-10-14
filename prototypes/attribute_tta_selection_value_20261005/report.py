"""选择价值诊断：保存最终和逐轮配对比较，不从短smoke选择方法。"""
from collections import Counter
from selection import MODES
from selection_audit import stats,METRICS,best_prefix,marginal
PAIRS=((MODES[0],MODES[1]),(MODES[0],MODES[2]),(MODES[1],MODES[2]))
def comparison(cases,left,right,round_index=-1):
    out={}
    for metric in METRICS:
        ds=[c["modes"][left]["states"][round_index][metric]-c["modes"][right]["states"][round_index][metric] for c in cases]
        signs=[-d if metric=="paired_cosine" else d for d in ds]
        out[metric]={"left_wins":sum(d>0 for d in signs),"right_wins":sum(d<0 for d in signs),
                     "ties":sum(d==0 for d in signs),"left_minus_right":stats(ds)}
    return out
def comparisons(cs,index=-1):
    return {l+"_vs_"+r:comparison(cs,l,r,index) for l,r in PAIRS}
def summarize(cases):
    summary={"case_count":len(cases),"comparisons":comparisons(cases),"by_K":{},"modes":{},
             "overlap":{key:{"intersection_count":stats(c["set_overlap"][key]["intersection_count"] for c in cases),
                             "jaccard":stats(c["set_overlap"][key]["jaccard"] for c in cases),
                             "identical_sets":sum(c["set_overlap"][key]["identical_set"] for c in cases)}
                        for key in cases[0]["set_overlap"]}}
    for k in sorted({c["natural_k"] for c in cases}):
        group=[c for c in cases if c["natural_k"]==k]
        summary["by_K"][str(k)]={"count":len(group),"comparisons":comparisons(group),
            "modes":{m:{metric:stats(c["modes"][m]["clean_to_final"][metric] for c in group) for metric in METRICS} for m in MODES}}
    rounds={}
    for m in MODES:
        best=[best_prefix(c["modes"][m]) for c in cases]
        summary["modes"][m]={"clean_to_final":{metric:stats(c["modes"][m]["clean_to_final"][metric] for c in cases) for metric in METRICS},
            "final_equals_best":sum(b["final_equals_best"] for b in best),
            "F_best_minus_final":stats(b["F_best_minus_final"] for b in best)}
    for r in range(1,max(c["natural_k"] for c in cases)+1):
        group=[c for c in cases if c["natural_k"]>=r]
        rounds[str(r)]={"case_count":len(group),"prefix_comparisons":comparisons(group,r),
            "modes":{m:{"marginal":marginal([c["modes"][m]["rounds"][r-1] for c in group]),
                         "clean_to_prefix":{metric:stats(c["modes"][m]["states"][r][metric]-c["modes"][m]["states"][0][metric] for c in group) for metric in METRICS}}
                     for m in MODES},
            "by_K":{str(k):comparisons([c for c in group if c["natural_k"]==k],r) for k in sorted({c["natural_k"] for c in group})}}
    return summary,rounds
