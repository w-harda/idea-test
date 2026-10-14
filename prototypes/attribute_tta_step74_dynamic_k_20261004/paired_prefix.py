"""共用实际前缀，capped为真实round4 checkpoint；不改PREFIX/TTA数学。"""
import copy,json,hashlib
from dataclasses import replace
import torch
import deps
from selection_audit import MODES
import prefix_tta as old
from attribute_tta import prepared_plan, seed_all, IMAGE_EPSILON
from controlled_tta import CategoricalTape,rng_fingerprint,metric_delta

def prepare_uncapped(payload,record,bases,device):
    chosen=payload["selected_attributes"];ranked=payload["ranked_attributes"]
    if chosen!=ranked[:payload["natural_k"]] or payload["k_star"]!=payload["natural_k"]:
        raise ValueError("natural fixed raw-ranked prefix changed")
    if len(chosen)!=payload["natural_k"] or payload["capped_k"]!=min(len(chosen),4):
        raise ValueError("K metadata mismatch")
    # 原Step72门禁一次最多4项：按连续chunk复用，恢复原全局selection_rank。
    out=[]
    for offset in range(0,len(chosen),4):
        chunk=chosen[offset:offset+4]
        partial={**payload,"ranked_attributes":chunk,"selected_attributes":chunk,"k_star":len(chunk)}
        checked=prepared_plan(partial,record,bases,device)
        out.extend(replace(a,selection_rank=offset+i) for i,a in enumerate(checked,1))
    if [a.instance_id for a in out]!=[a["instance_id"] for a in chosen]:
        raise ValueError("uncapped IDs changed")
    return tuple(out)

def execute_pair(source,image,plan,vectors,evaluator,*,steps,transforms,scales):
    cached=old.cached_round_guidance("PREFIX_NEW",plan,vectors)
    tape=CategoricalTape();snapshots=[]
    def observed_metric(current):
        metric=evaluator(current)
        # evaluator无RNG，记录每轮结束的完整随机条件，供capped endpoint核验。
        snapshots.append({"draw_count":tape.position,"rng":rng_fingerprint()})
        return metric
    seed_all(42)
    natural,inputs,momentum=old.execute(source,image,plan,cached,observed_metric,
             steps=steps,transforms=transforms,scales=scales,tape=tape)
    old.budget_and_prefix_checks("PREFIX_NEW",natural,plan,IMAGE_EPSILON)
    cap=min(len(plan),4);snap=snapshots[cap]
    prefix_records=tape.records[:snap["draw_count"]]
    encoded=json.dumps(prefix_records,separators=(",",":")).encode()
    capped=copy.deepcopy(natural)
    capped["states"]=capped["states"][:cap+1];capped["rounds"]=capped["rounds"][:cap]
    capped["fixed_sequence"]=capped["fixed_sequence"][:cap];capped["noise_sha256"]=capped["noise_sha256"][:cap]
    capped["clean_to_final"]=metric_delta(capped["states"][0],capped["states"][-1])
    capped["final_linf"]=capped["rounds"][-1]["cumulative_perturbation_linf"] if cap else 0.0
    capped["random_trace"]={"categorical_draw_count":snap["draw_count"],
        "categorical_trace_sha256":hashlib.sha256(encoded).hexdigest()}
    capped["rng_end_sha256"]=snap["rng"]
    old.budget_and_prefix_checks("PREFIX_NEW",capped,plan[:cap],IMAGE_EPSILON)
    if capped["states"]!=natural["states"][:cap+1] or capped["rounds"]!=natural["rounds"][:cap]:
        raise ValueError("common prefix not exact")
    if len(plan)<=4 and capped!=natural:raise ValueError("uncapped <=4 final mismatch")
    if len(snapshots)!=len(plan)+1:raise ValueError("round checkpoint missing")
    checks={"common_prefix_rounds":cap,"common_prefix_states_images_exact":True,
        "common_prefix_noise_transform_rng_exact":True,"natural_le4_final_exact":len(plan)<=4,
        "execution":"one actual original PREFIX trajectory; capped endpoint retained at min(k,4); only natural suffix continues",
        "caption_loss_terms":0,"score_backwards_during_attack":0}
    return {MODES[0]:capped,MODES[1]:natural},checks
