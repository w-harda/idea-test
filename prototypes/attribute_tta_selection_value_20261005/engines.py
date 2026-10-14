"""三组不同集合使用同一原始等权PREFIX；只耦合随机条件，不共享不同属性的round1。"""
import copy
from dataclasses import replace
import deps
import torch
from selection import MODES,mode_payload
from paired_prefix import prepare_uncapped
from controlled_tta import CategoricalTape
from attribute_tta import seed_all,IMAGE_EPSILON
import prefix_tta as original
baseline=deps.module("selection_value_frozen_baseline",deps.P753/"modes.py")
def plans_for(row,record,bases,device):
    return {m:prepare_uncapped(mode_payload(row,m),record,bases,device) for m in MODES}
def execute_three(source,image,plans,vectors,evaluator,*,steps=10,transforms=6,scales=(.5,.75,1.25,1.5),observed_limit=None):
    if observed_limit not in (None,1,2):raise ValueError("short smoke limit")
    if len({len(p) for p in plans.values()})!=1 or not plans[MODES[0]]:raise ValueError("same nonzero K required")
    results={};by_sequence={};tape=CategoricalTape();previous_limit=baseline._LIMIT
    baseline._LIMIT=observed_limit
    try:
        for mode in MODES:
            plan=plans[mode];seq=tuple(a.instance_id for a in plan)
            if seq in by_sequence:
                result=copy.deepcopy(results[by_sequence[seq]])
                result["execution_reuse_of"]=by_sequence[seq]
            else:
                replay=tape if mode==MODES[0] else CategoricalTape(tape.records)
                seed_all(42)
                with baseline.engine_for(baseline.MODES[0],plan,vectors):
                    result,_,_=original.execute(source,image,plan,
                        original.cached_round_guidance("PREFIX_NEW",plan,vectors),evaluator,
                        steps=steps,transforms=transforms,scales=scales,tape=replay)
                result["execution_reuse_of"]=None;by_sequence[seq]=mode
            original.budget_and_prefix_checks("PREFIX_NEW",result,plan,IMAGE_EPSILON)
            for e in result["rounds"]:
                e.pop("priority_steps",None)
                e.update(scheduler="PREFIX_BASELINE",score_weighting=False,priority_projection=False)
            if mode!=MODES[0]:
                for key in ("noise_sha256","random_trace","rng_end_sha256"):
                    if result[key]!=results[MODES[0]][key]:raise ValueError("controlled RNG differs: "+key)
            results[mode]=result
    finally:baseline._LIMIT=previous_limit
    checks={"same_K":True,"selected_sequence_exact":True,"same_clean_image":True,
            "noise_transform_RNG_exact":True,"unweighted_PREFIX_unchanged":True,
            "score_weighting":False,"priority_projection":False,"caption_loss_terms":0,
            "score_backwards_during_attack":0,
            "identical_sequence_reuse":"identical full sequence reuses the same actual trajectory; different selections always run their own image/momentum",
            "round1_shared_across_different_attributes":False}
    return results,checks
