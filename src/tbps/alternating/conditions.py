"""条件重建门禁和冻结算法调用接线；真实神经后端本轮未执行。"""
from dataclasses import dataclass, asdict
from .contracts import ConditionIdentity, require, text_sha
from tbps.artifacts import digest

@dataclass
class PreparedCondition:
    identity: ConditionIdentity
    payload: dict
    evidence: dict

def prepare_condition(arm, trace, original, g_anchors, *, legal_pairs, kernel):
    """kernel是已核验的原IRRA/定位/TextTransport组合，不能静默回退旧向量。"""
    require(arm in ("A1", "B"), "arm")
    trace.validate(legal_pairs)
    require(text_sha(trace.original) == original.identity.original_caption_sha256
            == original.identity.context_caption_sha256, "original condition identity")
    require(len(g_anchors) == len(original.identity.selected_g_ids), "original G anchor coverage")
    mappings = tuple(trace.mapping(a) for a in g_anchors)
    # A1忽略H监督反馈；NOOP仅在完整原空间/依赖已核验时允许显式复用。
    if arm == "A1" or trace.original == trace.adversarial:
        return PreparedCondition(original.identity, original.payload, {
            "mode": "A1_original" if arm == "A1" else "NOOP_identity_reuse",
            "parent_condition_key": original.identity.key, "action_sha256": trace.action_sha256,
            "new_JVPs": 0, "new_text_encodings": 0})
    identity = ConditionIdentity(
        text_sha(trace.original), text_sha(trace.adversarial),
        original.identity.selected_g_ids, original.identity.scores,
        original.identity.source_space_sha256, digest(mappings))
    value, evidence = kernel.recompute(trace.adversarial, mappings, original)
    require(evidence["context_caption_sha256"] == identity.context_caption_sha256
            and evidence["source_space_sha256"] == identity.source_space_sha256,
            "fresh context/space mismatch")
    require(evidence["new_JVPs"] == len(identity.selected_g_ids)
            and evidence["new_text_encodings"] >= 1 and evidence["fresh_localization"],
            "new context requires real encoding/localization/JVP; no relabel")
    require(evidence["zero_exact"] and all(
        x["zero_exact"] and x["central_fd_relative_error"] <= .02 for x in evidence["JVP_checks"]),
        "original TextTransport zero/FD gate")
    require(len(evidence["JVP_checks"]) == len(identity.selected_g_ids), "JVP receipt coverage")
    require(value["selected_instance_ids"] == list(identity.selected_g_ids)
            and value["K"] == len(identity.selected_g_ids), "G selection changed")
    return PreparedCondition(identity, value, evidence)

class LegacyContextKernel:
    """惰性依赖注入：复用FrozenLocalization.attach/TextTransport.read/condition_schema.build。
    locate/plan只做原算法I/O转换，真实来源与token对齐必须在GPU阶段另行验收。
    本类没有Loss、属性重评分、优化循环或自定义JVP。
    """
    def __init__(self, *, transport_factory, locate_and_plan, build, source_space_sha256, text_counter):
        self.transport_factory = transport_factory
        self.locate_and_plan = locate_and_plan
        self.build = build
        self.space = source_space_sha256
        self.text_counter = text_counter

    def recompute(self, caption, mappings, original):
        before = self.text_counter()
        transport = self.transport_factory(caption)
        # locate_and_plan必须用当前transport.H、原Canonical/basis、当前Span原生定位，
        # 返回原Attribute方向对象；不得返回旧cached.w/e或重新选K。
        plan, token_audit = self.locate_and_plan(caption, transport.H, mappings, original)
        ids = list(original.identity.selected_g_ids)
        require([a.instance_id for a in plan] == ids, "fresh plan order")
        require(token_audit["context_caption_sha256"] == text_sha(caption)
                and token_audit["fresh_localization"]
                and not token_audit["truncated"] and not token_audit["unresolved"],
                "new BPE/localization mapping unavailable")
        vectors, checks = [], []
        for a in plan:
            v, check = transport.read(a)
            require(check["instance_id"] == a.instance_id, "JVP ID")
            vectors.append(v); checks.append(check)
        import torch
        jvps = torch.cat(vectors).detach().cpu().clone() if vectors else torch.empty(0, 512)
        identity = {k: original.payload[k] for k in (
            "row_id", "image_rel_path", "caption_index", "split", "role", "pid", "caption")}
        identity["caption"] = caption
        scores = original.identity.scores
        weights = original.payload["score_weights"].tolist()
        result = self.build(identity, ids, scores, weights, jvps, transport.clean)
        return result, {
            "context_caption_sha256": text_sha(caption), "source_space_sha256": self.space,
            "new_JVPs": len(plan), "new_text_encodings": self.text_counter() - before,
            "count_semantics": "measured neural text sample counter",
            "fresh_localization": True, "zero_exact": True,
            "JVP_checks": checks, "tokenizer_audit": token_audit,
            "original_condition_key": original.identity.key,
            "old_JVP_relabelled": False}

class LegacySteps:
    """训练数学完全委托冻结原函数；不实现新Loss或Reward。"""
    def __init__(self, g_step, h_step):
        self.g_step, self.h_step = g_step, h_step

    def update_g(self, generator, optimizer, encoder, image, condition, **kwargs):
        result = self.g_step(generator, encoder, optimizer, image, condition.payload, **kwargs)
        # 原训练返回的两张临时图不进入逐步JSON；最终图库由后续generate_once生成。
        log = {k:v for k,v in result.items() if k not in ("serial_adversarial", "joint_adversarial")}
        from tbps.artifacts import json_bytes
        json_bytes(log)
        return log

    def update_h(self, policy, optimizer, example, sampler, bind_reward, step, row_id):
        log, _ = self.h_step(policy, optimizer, example, sampler, bind_reward, step, row_id)
        from tbps.artifacts import json_bytes
        json_bytes(log)
        return log


class CanonicalLocator:
    """仅Span/元数据接线；调用原FrozenLocalization，不重新提取Unicode攻击文本属性。"""
    def __init__(self, original_caption, original_instances, original_plan,
                 localizer, selected_type):
        self.original_caption = original_caption
        self.instances, self.plan = tuple(original_instances), tuple(original_plan)
        self.localizer, self.selected_type = localizer, selected_type

    def __call__(self, caption, hidden, mappings, original):
        from dataclasses import replace
        import copy
        require(len(caption)==len(self.original_caption), "only one-codepoint replacements")
        require(len(mappings)==len(self.instances)==len(self.plan), "anchor coverage")
        changed = []
        for instance, plan, mapping in zip(self.instances, self.plan, mappings):
            require(instance.instance_id==plan.instance_id, "original G instance/plan")
            a = mapping["original"]
            require(all(getattr(instance,k)==a[k] for k in a), "original Canonical identity")
            obj=copy.deepcopy(instance.linked_object)
            if obj:
                require(self.original_caption[obj["start"]:obj["end"]]==obj["raw"], "original linked-object Span")
                obj["raw"]=caption[obj["start"]:obj["end"]]
            changed.append(replace(instance,raw_text=mapping["edited_raw_text"],
                linked_object=obj,semantic_representation=None,representation_status="not_attached"))
        fresh, weights, alignment, _ = self.localizer.attach(changed,caption,hidden,cache_index=None)
        require(alignment["status"]=="valid" and len(fresh)==len(changed), "new token alignment")
        output=[]
        for new, old in zip(fresh,self.plan):
            require(new.representation_status=="ready" and new.semantic_representation is not None
                    and new.instance_id in weights, "new contextual localization unavailable")
            md=new.to_dict()
            md.update(instance_id=old.instance_id, context_instance_id=new.instance_id,
                      original_attribute_metadata_sha256=digest(old.metadata),
                      S_raw=old.metadata["S_raw"])
            # ID保持原Canonical锚点；新语境ID另记，不伪称新S_raw或旧prepared_plan。
            output.append(self.selected_type(md,old.selection_rank,weights[new.instance_id],old.Q))
        return tuple(output), dict(context_caption_sha256=text_sha(caption),
            fresh_localization=True,truncated=False,unresolved=False,
            alignment=alignment, original_condition_key=original.identity.key)


def trace_from_legacy(example, adversarial, edits, budget_k, native_action, *, replay):
    """先调用原constraints.replay核验完整EditRecord（码位/字节偏移/Span），再作字段转换。"""
    from dataclasses import asdict
    from .contracts import Anchor, Edit, TextTrace
    require(replay(example, edits, budget_k)==adversarial, "native Unicode replay mismatch")
    require(isinstance(native_action,dict) and native_action, "full native action required")
    evidence={"native_action":native_action,"native_edits":[asdict(e) for e in edits],
              "original_caption":example.caption,"adversarial_caption":adversarial,"budget_k":budget_k}
    return TextTrace(example.caption,adversarial,
        tuple(Anchor(**asdict(a)) for a in example.attributes),
        tuple(Edit(e.attribute_id,e.position,e.original_char,e.replacement_char) for e in edits),
        budget_k,digest(evidence))
