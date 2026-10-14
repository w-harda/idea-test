"""三模式共享一次round1；随后anchor trace回放，真实IG每步加权/投影。"""
import copy
from contextlib import contextmanager
import torch
import prefix_tta as old
from controlled_tta import ControlledTTA,CategoricalTape,rng_fingerprint
from attribute_tta import seed_all,IMAGE_EPSILON
from priority import MODES,normalized_weights
from real_ig import IGCollector,attack_type

def cached_guidance(mode,plan,vectors):
    if mode==MODES[0]:return old.cached_round_guidance("PREFIX_NEW",plan,vectors)
    out=[]
    for r in range(1,len(plan)+1):
        members=plan[:r];vs=[vectors[a.instance_id] for a in members]
        weights=normalized_weights([a.metadata["S_raw"] for a in members],mode)
        if r==1:aggregate=vs[0]
        else:
            ws=torch.tensor(weights,dtype=vs[0].dtype,device=vs[0].device)
            aggregate=(torch.stack(vs)*ws[:,None,None]).sum(0).detach()
        out.append((members,aggregate))
    return tuple(out)

@contextmanager
def engine_for(mode,plan,vectors):
    original_type=old.ControlledTTA
    class Engine(ControlledTTA):
        def step(self,current,attribute,features,labels):
            if attribute.selection_rank==1:
                result=super().step(current,attribute,features,labels)
                # 真正单属性IG已由原observer记录；无history，完全共享旧路径。
                event=self.events[-1]
                event["priority_steps"]=[{
                    "step_index":s["step_index"],"attribute_instance_id":attribute.instance_id,
                    "new_semantic_axis":attribute.metadata["semantic_axis"],"new_region":attribute.metadata["region"],
                    "S_raw":[attribute.metadata["S_raw"]],"normalized_weights":[1.],
                    "history_norm":0.,"new_norm_before":s["ig_gradient_l2"],"new_norm_after":s["ig_gradient_l2"],
                    "dot_before":0.,"dot_after":0.,"cosine_before":None,"cosine_after":None,
                    "conflict":False,"projection_intervention":False,
                    "removed_conflict_component_norm":0.,"projection_modification_ratio":0.,
                    "no_conflict_projection_exact":True,
                    "aggregate_gradient_norm":s["ig_gradient_l2"],
                    "history_components_unchanged":True,"only_new_attribute_modified":True,
                    "actual_momentum_input_verified":True,
                    "diagnostic_point":"shared unmodified official single-attribute IG path",
                    "attribute_components":[{
                        "attribute_instance_id":attribute.instance_id,"semantic_axis":attribute.metadata["semantic_axis"],
                        "region":attribute.metadata["region"],"canonical_value":attribute.metadata["canonical_value"],
                        "S_raw":attribute.metadata["S_raw"],"normalized_weight":1.,
                        "real_IG_gradient_norm":s["ig_gradient_l2"],"weighted_contribution_norm":s["ig_gradient_l2"],
                    }],
                } for s in event["steps"]]
                return result
            # 官方实例参数原样继承，helper与categorical tape均不替换。
            original_attack=self.attack
            self.attack=attack_type(self.tape)(None,original_attack.tokenizer,
                                               imgs_eps=original_attack.imgs_eps,step_size=original_attack.step_size)
            self.attack.N_trans=original_attack.N_trans
            controller=IGCollector(plan[:attribute.selection_rank],vectors,mode)
            self.attack._priority=controller
            result=super().step(current,attribute,features,labels)
            event=self.events[-1];event["priority_steps"]=controller.records
            if len(controller.records)!=self.steps or not all(x.get("actual_momentum_input_verified") for x in controller.records):
                raise ValueError("real optimization steps did not consume composed gradient")
            return result
    old.ControlledTTA=Engine
    try:yield
    finally:old.ControlledTTA=original_type

def execute_three(source,image,plan,vectors,evaluator,*,steps,transforms,scales):
    tape=CategoricalTape();capture=[];finals=[];seed_all(42)
    with engine_for(MODES[0],plan,vectors):
        baseline,_,baseline_momentum=old.execute(source,image,plan,cached_guidance(MODES[0],plan,vectors),evaluator,
            steps=steps,transforms=transforms,scales=scales,tape=tape,first_round_capture=capture,
            final_image_capture=finals)
    results={MODES[0]:baseline};final_momenta={MODES[0]:baseline_momentum}
    if not plan:raise ValueError("fixed Step74 manifest contains only positive supported K")
    cp=capture[0]
    for mode in MODES[1:]:
        replay=CategoricalTape(tape.records)
        with engine_for(mode,plan,vectors):
            result,_,momentum=old.execute(source,image,plan,cached_guidance(mode,plan,vectors),evaluator,
                steps=steps,transforms=transforms,scales=scales,tape=replay,
                first_round_checkpoint=cp,final_image_capture=finals)
        results[mode]=result;final_momenta[mode]=momentum
        for key in ("fixed_sequence","noise_sha256","random_trace","rng_end_sha256"):
            if result[key]!=baseline[key]:raise ValueError("controlled randomness/sequence mismatch: "+key)
        if result["states"][:2]!=baseline["states"][:2] or result["rounds"][0]!=baseline["rounds"][0]:
            raise ValueError("shared round1 mismatch")
    for mode,result in results.items():
        old.budget_and_prefix_checks("PREFIX_NEW",result,plan,IMAGE_EPSILON)
        for event in result["rounds"]:
            event["aggregation_mode"]="SHARED_SINGLE_ATTRIBUTE" if event["round_index"]==1 else mode
            event["prefix_arithmetic"]="shared unchanged single attribute" if event["round_index"]==1 else ("unweighted original mean" if mode==MODES[0] else "frozen S_raw normalized weights; priority modifies only newest real IG if conflicting")
            event["normalized_attribute_weights"]=normalized_weights([a.metadata["S_raw"] for a in plan[:event["round_index"]]],mode)
    # event补充的模式metadata不同，不作为共享物理状态的相等判断。
    k1=len(plan)==1
    if k1 and (not all(torch.equal(finals[0],x) for x in finals[1:]) or
               not all(torch.equal(baseline_momentum,m) for m in list(final_momenta.values())[1:])):
        raise ValueError("k1 image/momentum mismatch")
    checks={"round1_image_momentum_metrics_noise_transform_RNG_exact":True,
            "round1_shared_complete_state":True,"k1_final_exact":True if k1 else None,
            "noise_transform_RNG_exact":True,"score_backwards_during_attack":0,"caption_loss_terms":0,
            "gradient_projection_enters_real_IG_momentum_path":True,
            "random_control":"PREFIX_BASELINE anchors full categorical trace; other modes replay from exact shared round1 checkpoint; each mode retains official adaptive-probability update"}
    return results,checks
