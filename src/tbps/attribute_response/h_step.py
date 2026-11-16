"""保留三头SCST，增加属性监督和完整合法位置/字符条件优势。"""
from dataclasses import asdict
import math
import torch
from torch.nn import functional as F
from unittest.mock import patch
from tbps.augmented_attraction.h_reward import BoundAlignmentReward
from .memory import semantic_key
from tbps.alternating.state import tree_sha
from .objectives import conditional_advantage, action_features
from .settings import require

def legal_alternatives(policy, example, sampled, edit_index, representations, constraints, probe):
    edit = sampled.edits[edit_index]
    attrs = {a.attribute_id: (i, a) for i, a in enumerate(example.attributes)}
    ai, attr = attrs[edit.attribute_id]
    other = [e for i, e in enumerate(sampled.edits) if i != edit_index]
    context = constraints.replay(example, other, sampled.budget_k)
    contracts = __import__("contracts")
    used = {contracts.word_at(example.caption, e.position) for e in other}
    support = constraints.candidates(example, context, ai, used, probe)
    pooled, chars, attributes = representations
    positions = list(range(attr.char_start, attr.char_end))
    with torch.no_grad():
        pf = torch.stack([torch.cat((chars[p], attributes[ai])) for p in positions]
                         + [torch.cat((pooled, attributes[ai]))])
        logits = policy.position_head(pf).flatten()
        mask = torch.tensor([p in support for p in positions] + [True])
        ppos = logits.masked_fill(~mask, -torch.inf).softmax(0)
        rows = [{"text": context, "probability": float(ppos[-1]), "position": None,
                 "replacement": None, "action": [0.] * 6, "position_logit": float(logits[-1]), "replacement_logit": None}]
        for j, pos in enumerate(positions):
            if pos not in support:
                continue
            rlogits = policy.replacement_head(torch.cat((chars[pos], attributes[ai])))
            rmask = torch.tensor([r in support[pos] for r in constraints.REPLACEMENTS])
            prep = rlogits.masked_fill(~rmask, -torch.inf).softmax(0)
            for ri, char in enumerate(constraints.REPLACEMENTS):
                if not rmask[ri]:
                    continue
                replacement = constraints.make_edit(example, ai, pos, char)
                text = constraints.replay(example, other + [replacement], sampled.budget_k)
                rows.append({"text": text, "probability": float(ppos[j] * prep[ri]),
                             "position": pos, "replacement": char, "position_logit": float(logits[j]), "replacement_logit": float(rlogits[ri]),
                             "action": action_features(replacement, example.caption, len(sampled.edits))})
    selected = [i for i, x in enumerate(rows)
                if x["position"] == edit.position and x["replacement"] == edit.replacement_char]
    require(len(selected) == 1 and rows[selected[0]]["text"] == sampled.adversarial,
            "sampled edit belongs to complete conditional action support")
    require(math.isclose(sum(x["probability"] for x in rows), 1., abs_tol=2e-6),
            "complete normalized conditional position/replacement/NOOP")
    return rows, selected[0]

def encode_bounded(runtime, texts, limit):
    unique = list(dict.fromkeys(texts)); values = {}
    for start in range(0, len(unique), limit):
        batch = unique[start:start + limit]
        feature = runtime.encode_text(batch)
        for text, row in zip(batch, feature):
            values[text] = row.detach().clone()
    return values

def train_step(phase, policy_module, constraints, policy, optimizer, example, sampler,
               make_reward, step, row_id, profile, partners, teacher, states,
               *, version, response_probe=None, feature_sink=None, max_counterfactuals=2048):
    phase.check_optimizer(policy, optimizer)
    require(torch.is_grad_enabled() and not torch.is_inference_mode_enabled(), "ordinary H autograd")
    policy.train(); optimizer.zero_grad(set_to_none=True)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    from feedback_backend import state_sha, validate_variant
    h_before = state_sha(policy)
    with torch.no_grad():
        greedy = policy.rollout(example, "greedy")
    choices = []; representation = []
    original_choose = policy_module.choose
    def choose(logits, mask, mode, generator=None):
        index, lp = original_choose(logits, mask, mode, generator)
        choices.append({"logits": logits, "mask": mask, "index": index, "log_prob": lp})
        return index, lp
    hook = policy.register_forward_hook(lambda m, args, output: representation.append(output))
    try:
        with patch.object(policy_module, "choose", choose):
            sampled = policy.rollout(example, "sample", generator=sampler)
    finally:
        hook.remove()
    require(len(representation) == 1 and len(choices) == len(sampled.decisions), "one H encoding/action trace")
    require(sampled.budget_k == greedy.budget_k, "same original Natural-K")
    for out in (sampled, greedy):
        require(constraints.replay(example, out.edits, out.budget_k) == out.adversarial, "Unicode replay")
    base = make_reward(sampled.budget_k)
    bound = BoundAlignmentReward(base, profile, partners)
    sr, gr = bound.evaluate_many(example.caption, (sampled.adversarial, greedy.adversarial))
    original_features = bound.features.detach().clone()
    scst, advantage = phase.scst_loss(sampled.log_prob, sr.total, gr.total)
    groups = []; pending = []
    for decision, choice in zip(sampled.decisions, choices):
        if decision["head"] == "attribute":
            pending = []
        elif decision["head"] == "position":
            pending = [choice["log_prob"]]
        elif decision["head"] == "replacement":
            groups.append(sum(pending + [choice["log_prob"]]))
    require(len(groups) == len(sampled.edits), "position/replacement credit per actual edit")
    supports = [legal_alternatives(policy, example, sampled, i, representation[0], constraints,
                                  base.runtime.probe) for i in range(len(sampled.edits))]
    require(sum(len(x[0]) for x in supports) <= max_counterfactuals, "bounded full conditional support; no truncation")
    texts = [x["text"] for rows, _ in supports for x in rows]
    for text in set(texts):
        validate_variant(example, text, sampled.budget_k, base.runtime.probe)
    encoded = encode_bounded(base.runtime, texts, teacher.settings.max_text_batch)
    feature_texts = list(encoded)
    feature_index = {t: i for i, t in enumerate(feature_texts)}
    credit_terms = []; diagnostics = []; updates = []
    features_before = teacher.state_dict()
    for i, ((rows, selected), lp) in enumerate(zip(supports, groups)):
        edit = sampled.edits[i]
        attr_index = next(j for j, a in enumerate(example.attributes) if a.attribute_id == edit.attribute_id)
        key = semantic_key(example.attributes[attr_index]); state = states[attr_index]
        batch = teacher.features([key] * len(rows), [state] * len(rows),
                                 [x["action"] for x in rows], version=version)
        prediction = teacher.predict(batch)[:, 1]
        exact = []
        for row in rows:
            current = encoded[row["text"]][None].expand(3, -1)
            exact.append(float(profile.terms(current, original_features[:1].expand(3, -1), partners)["reward"].mean()))
        values = [r + teacher.settings.response_weight * float(p) for r, p in zip(exact, prediction)]
        probabilities = [x["probability"] for x in rows]
        adv, baseline = conditional_advantage(values, probabilities, selected)
        credit_terms.append(-teacher.settings.credit_weight * adv * lp)
        diagnostic = {"attribute": key, "attribute_index": attr_index, "states": state, "selected": selected, "support": rows,
                      "feature_indices": [feature_index[x["text"]] for x in rows],
                      "exact_rewards": exact, "predicted_next_G_gains": prediction.tolist(),
                      "values": values, "baseline": float(baseline), "advantage": float(adv),
                      "position_replacement_log_prob": float(lp.detach()),
                      "teacher_context": {"key": key, "state": state, "history": batch["histories"][0], "version": version}}
        if response_probe is not None:
            response = response_probe(encoded[rows[selected]["text"]], edit, i)
            if response is not None:
                target, proof = response
            else:
                target, proof = None, None
            selected_batch = {k: [v[selected]] for k, v in batch.items()}
            if target is not None:
                updates.append((selected_batch, [target], 1))
                diagnostic["next_G_probe"] = proof
        diagnostics.append(diagnostic)
    credit = sum(credit_terms, sampled.log_prob * 0) / max(1, len(credit_terms))
    keys = [semantic_key(a) for a in example.attributes]
    batch = teacher.features(keys, states, version=version)
    pred = teacher.predict(batch)[:, 1]
    for d in diagnostics:
        pred[d["attribute_index"]] = max(d["values"])-d["baseline"]
    attr_choices = [x for d, x in zip(sampled.decisions, choices) if d["head"] == "attribute"]
    distill = sampled.log_prob * 0
    distill_log = {"optimized": False}
    if attr_choices and keys:
        logits = attr_choices[0]["logits"][:-1]; mask = attr_choices[0]["mask"][:-1]
        if mask.any():
            target = (logits.detach()[mask] + pred[mask]).softmax(0).detach()
            distill = teacher.settings.attribute_distill_weight * F.kl_div(
                F.log_softmax(logits[mask], 0), target, reduction="sum")
            distill_log = {"optimized": True, "target": target.tolist(),
                           "log_probs": F.log_softmax(logits[mask].detach(), 0).tolist(),
                           "mask": mask.tolist()}
    loss = scst + credit + distill
    require(torch.isfinite(loss), "finite guided H loss")
    loss.backward()
    grads = {n: p.grad for n, p in policy.named_parameters() if p.grad is not None}
    require(grads and all(torch.isfinite(x).all() for x in grads.values()), "finite three-head gradients")
    gradient = math.sqrt(sum(float(x.double().square().sum()) for x in grads.values()))
    group_grads = {prefix: math.sqrt(sum(float(x.double().square().sum()) for n, x in grads.items()
                          if n.startswith(prefix + "."))) for prefix in
                   ("attribute_head", "position_head", "replacement_head", "byte_embedding", "encoder")}
    optimizer.step()
    require(all(torch.isfinite(p).all() for p in policy.parameters()), "finite H update")
    changes = [p.detach().double() - before[n].double() for n, p in policy.named_parameters()]
    teacher_updates = [teacher.observe(*u) for u in updates]
    for d in diagnostics:
        gain = d.get("next_G_probe", {}).get("gain", 0.)
        if "next_G_probe" in d:
            teacher.remember(d["attribute"], d["states"], [0., gain], version=version)
    result = {"step": step, "row_id": row_id, "H_state_sha256_before": h_before,
              "H_state_sha256_after": state_sha(policy), "sampled_reward": asdict(sr), "greedy_reward": asdict(gr),
              "advantage": float(advantage), "SCST_loss": float(scst.detach()),
              "sequence_log_prob": float(sampled.log_prob.detach()), "gradient_norm": gradient,
              "group_gradient_norm": group_grads, "parameter_delta_l2": math.sqrt(sum(float(x.square().sum()) for x in changes)),
              "parameter_delta_max_abs": max(float(x.abs().max()) for x in changes), "K": sampled.budget_k,
              "sampled": sampled.record(), "greedy": greedy.record(),
              "sampled_action_stats": phase.action_stats(sampled), "greedy_action_stats": phase.action_stats(greedy),
              "optimizer_steps_this_step": 1, **bound.diagnostics()}
    result["IRRA_text_samples"] = 3 + len(set(texts))
    result["attribute_response"] = {"credits": diagnostics, "credit_loss": float(credit.detach()),
         "attribute_distill_loss": float(distill.detach()), "attribute_distill": distill_log,
         "optimized_loss": float(loss.detach()), "teacher_before": tree_sha(features_before),
         "teacher_after": tree_sha(teacher.state_dict()), "teacher_updates": teacher_updates,
         "response_weight": teacher.settings.response_weight, "credit_weight": teacher.settings.credit_weight,
         "distill_weight": teacher.settings.attribute_distill_weight,
         "original_features": original_features.tolist(), "counterfactual_unique_texts": len(set(texts)),
         "teacher_inference_only": True, "four_mode_reward_optimized": False}
    features = torch.stack([encoded[t] for t in feature_texts]) if feature_texts else torch.empty((0, original_features.shape[1]))
    result["attribute_response"]["counterfactual_features"] = (feature_sink(feature_texts, features) if feature_sink else
        {"texts": feature_texts, "features": features.tolist()})
    bound.runtime.assert_frozen()
    return result, original_features
