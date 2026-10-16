"""仅文本随机消融；不接收PID、图像、检索分数或奖励。"""
from collections import Counter
from dataclasses import asdict
import hashlib
import random
import torch
from policy import CharacterPolicy
from contracts import from_text_spans,word_at
from constraints import candidates,make_edit,replay
from phase5_joint.backend import TextAttacker,validate_action
from feedback_backend import state_sha

LEARNED_FIELDS={"original","adversarial","budget_k","predicted_importance","attributes",
 "selected_attribute_ids","edits","decisions","tokenizer_audit","eligible","reasons","H_forward"}
def validate_reference(caption,learned,probe,frontend):
    if type(caption)is not str or type(learned)is not dict or set(learned)!=LEARNED_FIELDS:
        raise ValueError("only verified caption-only H action fields accepted")
    validate_action(caption,learned,probe,frontend)
    return from_text_spans(caption,learned["attributes"])
def rng_for(caption,seed,kind):
    if type(seed)is not int or type(caption)is not str:raise TypeError("seed and Caption required")
    key=f"{seed}|{kind}|{caption}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest(),"big"))
def inference_settings():
    # 与Phase4 Validation和Phase5相同，早于任何H forward。
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
def build_untrained(architecture,probe):
    inference_settings()
    # 只替换并恢复CPU RNG，不重置已加载IRRA的CUDA随机状态。
    saved=torch.get_rng_state().clone()
    try:
        with torch.inference_mode(False):
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            policy=CharacterPolicy(probe,**architecture).cpu()
    finally:torch.set_rng_state(saved)
    return policy.eval().requires_grad_(False)

class RandomStrategy:
    def __init__(self,kind,seed,probe,frontend,max_nodes=256):
        if kind not in ("random_attribute","h_selected") or type(max_nodes)is not int or max_nodes<1:
            raise ValueError("explicit random kind and positive search budget")
        self.kind,self.seed,self.probe,self.frontend,self.limit=kind,seed,probe,frontend,max_nodes
    def attack(self,caption,learned):
        ex=validate_reference(caption,learned,self.probe,self.frontend)
        target=len(learned["edits"]);budget=learned["budget_k"]
        support=[e["attribute_id"] for e in learned["edits"]]
        mapping={a.attribute_id:i for i,a in enumerate(ex.attributes)}
        rng=rng_for(caption,self.seed,self.kind)
        best=[];nodes=0;limited=False;found=False
        def search(current,edits,used_ids,used_words):
            nonlocal best,nodes,limited,found
            if len(edits)>len(best):best=list(edits)
            if len(edits)==target:
                if not edits or self.probe.legal_change(caption,current):
                    best=list(edits);found=True;return True
                return False
            if nodes>=self.limit:limited=True;return False
            if self.kind=="h_selected":
                indices=[mapping[support[len(edits)]]]
            else:
                indices=[i for i,a in enumerate(ex.attributes) if a.attribute_id not in used_ids]
                rng.shuffle(indices)
            for i in indices:
                attr=ex.attributes[i]
                legal=candidates(ex,current,i,used_words,self.probe)
                positions=list(legal);rng.shuffle(positions)
                for pos in positions:
                    replacements=list(legal[pos]);rng.shuffle(replacements)
                    for replacement in replacements:
                        if nodes>=self.limit:limited=True;return False
                        nodes+=1
                        edit=make_edit(ex,i,pos,replacement)
                        next_text=current[:pos]+replacement+current[pos+1:]
                        if search(next_text,edits+[edit],used_ids|{attr.attribute_id},
                                  used_words|{word_at(caption,pos)}):return True
            return False
        if target:search(caption,[],set(),set())
        else:found=True
        if target and not found and not limited:
            # 原H编辑本身是合法见证；穷尽仍失败意味着实现不一致，不伪造不可行。
            raise ValueError("known feasible H witness but exhaustive randomized search failed")
        # 部分解也必须相对原文Token改变；不合法则明确回退空编辑并记账。
        while best and not self.probe.legal_change(caption,replay(ex,best,budget)):best.pop()
        adv=replay(ex,best,budget)
        selected=(list(learned["selected_attribute_ids"]) if self.kind=="h_selected"
                  else [e.attribute_id for e in best])
        result={"original":caption,"adversarial":adv,"budget_k":budget,
          "attributes":[asdict(a) for a in ex.attributes],
          "predicted_importance":list(learned["predicted_importance"]) if self.kind=="h_selected" else [],
          "selected_attribute_ids":selected,"edits":[asdict(e) for e in best],"decisions":[],
          "tokenizer_audit":self.probe.pair_report(caption,adv),"eligible":learned["eligible"],
          "reasons":list(learned["reasons"]),"H_forward":0}
        result["ablation"]={"kind":self.kind,"seed":self.seed,"target_edits":target,"actual_edits":len(best),
          "matched":len(best)==target,"source_H_budget_k":budget,
          "source_H_selected_ids":list(learned["selected_attribute_ids"]),"source_H_edited_ids":support,
          "failure_reason":None if len(best)==target else "search_node_limit_no_complete_match",
          "search_nodes":nodes,"search_limit":self.limit,"search_exhausted":limited,
          "support_definition":"H actual edited attributes in original order" if self.kind=="h_selected" else "all legal Caption attributes",
          "no_feedback":True}
        validate_action(caption,result,self.probe,self.frontend)
        if self.kind=="h_selected" and [e["attribute_id"] for e in result["edits"]]!=support[:len(best)]:
            raise ValueError("H actual attribute support/order changed")
        return result
    def finish(self):return {"kind":self.kind,"seed":self.seed,"H_forward_samples":0,"no_feedback":True}

class UntrainedStrategy:
    def __init__(self,architecture,probe,frontend):
        self.probe,self.frontend=probe,frontend
        self.attacker=TextAttacker(build_untrained(architecture,probe),frontend,probe)
        self.initial_sha=state_sha(self.attacker.policy)
    def attack(self,caption,learned):
        validate_reference(caption,learned,self.probe,self.frontend)
        result=self.attacker.attack(caption)
        result["ablation"]={"kind":"untrained","seed":42,"target_edits":len(learned["edits"]),
            "actual_edits":len(result["edits"]),"matched":len(result["edits"])==len(learned["edits"]),
            "failure_reason":None,"count_matching":"not forced; natural initial H greedy policy",
            "source_H_budget_k":learned["budget_k"],"no_feedback":True}
        return result
    def finish(self):
        return {**self.attacker.finish(),"initial_H_state_sha256":self.initial_sha,
                "seed":42,"SCST_training_steps":0}

def validate_trial_action(caption,action,learned,trial,probe,frontend,expected_limit=256):
    validate_reference(caption,learned,probe,frontend)
    validate_action(caption,action,probe,frontend)
    a=action["ablation"];target=len(learned["edits"]);actual=len(action["edits"])
    if (a["kind"]!=trial["kind"] or a["seed"]!=trial["seed"] or a["target_edits"]!=target
        or a["actual_edits"]!=actual or a["matched"]!=(actual==target) or a["no_feedback"] is not True):
        raise ValueError("ablation matching/seed accounting mismatch")
    if a["source_H_budget_k"]!=learned["budget_k"]:
        raise ValueError("reference H budget trace mismatch")
    if trial["kind"]!="untrained":
        if (a["source_H_selected_ids"]!=learned["selected_attribute_ids"]
                or a["source_H_edited_ids"]!=[e["attribute_id"] for e in learned["edits"]]
                or a["search_limit"]!=expected_limit or not 0<=a["search_nodes"]<=expected_limit):
            raise ValueError("random control reference/search provenance mismatch")
        if a["matched"] and a["failure_reason"] is not None:
            raise ValueError("matched query has false failure")
        if not a["matched"] and (not a["search_exhausted"] or a["search_nodes"]!=expected_limit):
            raise ValueError("matching deficit without exhausted configured search")
        if action["budget_k"]!=learned["budget_k"] or actual>target or action["H_forward"]!=0:
            raise ValueError("random control changed budget/count or invoked H")
        if not a["matched"] and a["failure_reason"]!="search_node_limit_no_complete_match":
            raise ValueError("unreported matching deficit")
        if trial["kind"]=="h_selected":
            support=[e["attribute_id"] for e in learned["edits"]]
            if (action["selected_attribute_ids"]!=learned["selected_attribute_ids"]
                or [e["attribute_id"] for e in action["edits"]]!=support[:actual]):
                raise ValueError("H-selected support changed")

def edit_statistics(actions):
    n=len(actions);a=[x["ablation"] for x in actions]
    matched=sum(x["matched"] for x in a)
    return {"queries":n,"target_edit_count_distribution":dict(Counter(str(x["target_edits"]) for x in a)),
        "actual_edit_count_distribution":dict(Counter(str(len(x["edits"])) for x in actions)),
        "NOOP_queries":sum(not x["edits"] for x in actions),
        "NOOP_ratio":sum(not x["edits"] for x in actions)/n if n else 0.,
        "exact_count_match_queries":matched,"exact_count_match_ratio":matched/n if n else 0.,
        "under_target_queries":sum(x["actual_edits"]<x["target_edits"] for x in a),
        "over_target_queries":sum(x["actual_edits"]>x["target_edits"] for x in a),
        "matching_failure_reasons":dict(Counter(x["failure_reason"] for x in a if x["failure_reason"])),
        "count_matching_scope":"forced random controls; descriptive only for untrained"}

def reference_statistics(learned):
    n=len(learned)
    return {"queries":n,"actual_edit_count_distribution":dict(Counter(str(len(a["edits"])) for a in learned)),
        "NOOP_queries":sum(not a["edits"] for a in learned),
        "NOOP_ratio":sum(not a["edits"] for a in learned)/n if n else 0.,
        "edits_total":sum(len(a["edits"]) for a in learned),
        "K_distribution":dict(Counter(str(a["budget_k"]) for a in learned))}
