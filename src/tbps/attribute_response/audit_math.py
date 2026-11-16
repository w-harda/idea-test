"""不加载G/H/Victim的独立FP64算术审计；复核每个合法条件候选和虚拟更新隔离。"""
from pathlib import Path
import math
import re
import copy
import torch
from torch.nn import functional as F
from tbps.artifacts import read_json, sha_file
from tbps.alternating.state import tree_sha
from tbps.augmented_attraction.audit_math import audit_attraction as audit_base, audit_view_bank
from .settings import Settings, require
from .memory import key_bytes

def close(a, b, label, atol=5e-5):
    require(math.isfinite(float(a)) and math.isfinite(float(b))
            and math.isclose(float(a), float(b), abs_tol=atol, rel_tol=2e-4), label)

class Projection:
    def __init__(self, rows, project):
        self.rows, self.project = rows, project
    def __len__(self):
        return len(self.rows)
    def __iter__(self):
        for row in self.rows:
            yield self.project(row)

def base_g(g):
    g = dict(g)
    g["records"] = [{k: v for k, v in r.items() if k != "attribute_response"} for r in g["records"]]
    return g

def base_h(h):
    h = dict(h)
    if not h.get("fixed_NOOP"):
        # 基础审计只覆盖原始3文本SCST；额外候选及其计数由下面逐项覆盖。
        h["IRRA_text_samples"] = 3
    return h

def profile_reward(current, clean, partner, anchors, cfg):
    def logp(x):
        return F.log_softmax(F.normalize(x.double(), dim=-1) @ F.normalize(anchors.double(), dim=-1).T
                             / cfg["temperature"], dim=-1)
    pa, pc, pp = logp(current), logp(clean), logp(partner)
    pull = (pp.exp() * (pp-pa)).sum(-1)
    baseline = (pp.exp() * (pp-pc)).sum(-1)
    push = (pc.exp() * (pc-pa)).sum(-1)
    return (cfg["pull_weight"]*(baseline-pull)+cfg["push_weight"]*push).mean()

def audit_regression(row):
    if not row.get("updated"):
        return
    require(len(row["predictions"]) == len(row["targets"]) == len(row["directions"]),
            "complete teacher regression batch")
    dif = torch.tensor(row["predictions"], dtype=torch.float64)-torch.tensor(row["targets"], dtype=torch.float64)
    loss = torch.where(dif.abs() < 1, .5*dif.square(), dif.abs()-.5).mean()
    close(row["loss"], loss, "independent response regression")
    require(row["update"] > 0 and len(row["teacher_before"]) == len(row["teacher_after"]) == 64,
            "teacher update recorded")

def audit_isolation(proof):
    require(proof["isolated"] and proof["formal_before"] == proof["formal_after"]
            and proof["rng_before"] == proof["rng_after"], "virtual model/Adam/gradient/mode/RNG isolation")

def audit_utility(event, base_bank, anchors, kl_cfg, response):
    require(event["role"] == "train", "probe feedback is Train only")
    require(event["base_bank_sha256"] == tree_sha(base_bank), "exact frozen virtual base bank")
    bank = base_bank.clone()
    positions = set()
    for patch in event["patches"]:
        j = patch["index"]
        require(type(j) is int and 0 <= j < len(bank) and j not in positions, "unique in-range virtual bank patch")
        positions.add(j); bank[j] = torch.tensor(patch["features"], dtype=bank.dtype)
    require(tree_sha(bank) == event["bank_feature_sha256"], "virtual bank patch content")
    text = torch.tensor(event["text_features"], dtype=torch.float64)
    require(text.shape == (1+len(event["actions"]), anchors.shape[1]), "clean and all attacked probe text features")
    # 配对行由调用方写入的Train身份查回，不能用模型得分挑选。
    return bank, text

def audit_probe_utility(event, base_bank, anchors, kl_cfg, response, ids, positive_map=None):
    bank, text = audit_utility(event, base_bank, anchors, kl_cfg, response)
    require(event["row"] in ids, "probe member")
    index = ids.index(event["row"])
    partner = bank[index].double()
    require(event["probe_policy_mode"] == "conditional_fixed_legal_trajectory_support", "explicit finite probe policy")
    positive = torch.zeros(len(ids), dtype=torch.bool)
    require(event["positive_indices"] and len(set(event["positive_indices"])) == len(event["positive_indices"]),
            "nonempty unique positive probe indices")
    positive[event["positive_indices"]] = True
    require(positive.any() and (~positive).any(), "probe positive/negative coverage")
    if positive_map is not None:
        expected = [j for j, row in enumerate(ids) if positive_map[row] == positive_map[event["row"]]]
        require(event["positive_indices"] == expected, "Train PID used only in reward binding")
    probabilities = torch.tensor(event["log_probs"], dtype=torch.float64).softmax(0)
    rewards = []; margins = []
    for j, feature in enumerate(text[1:]):
        reward = profile_reward(feature[None].expand(3, -1), text[:1].expand(3, -1), partner, anchors, kl_cfg)
        scores = bank[:, 0].double() @ feature
        margin = scores[~positive].max()-scores[positive].max()
        close(event["per_action_KL"][j], reward, "per-action probe KL")
        close(event["per_action_margin"][j], margin, "per-action probe margin")
        close(event["probabilities"][j], probabilities[j], "conditional probe policy")
        rewards.append(reward); margins.append(margin)
    reward = probabilities @ torch.stack(rewards)
    margin = probabilities @ torch.stack(margins)
    close(event["KL_reward"], reward, "probe KL")
    close(event["negative_minus_positive_margin"], margin, "probe retrieval margin")
    close(event["utility"], reward+response.rank_weight*margin, "probe complete utility")


def audit_probe(record, base_bank, anchors, kl_cfg, response, ids, positive_map=None):
    require(record["support"] in ids and record["probe"] in ids
            and record["support"] != record["probe"] and record["role"] == "train",
            "distinct Train support/probe")
    audit_isolation(record["isolation"])
    for name in ("before", "after"):
        require(record[name]["row"] == record["probe"], "same probe before and after update")
        audit_probe_utility(record[name], base_bank, anchors, kl_cfg, response, ids, positive_map)
    require(record["before"]["actions"] == record["after"]["actions"], "same probe trajectories before/after update")
    close(record["gain"], record["after"]["utility"]-record["before"]["utility"], "next-update gain")

def candidate_features(md, ledger_root):
    ref = md["counterfactual_features"]
    if "features" in ref:
        return ref["texts"], torch.tensor(ref["features"], dtype=torch.float64)
    require(re.fullmatch(r"H_features_[0-9]{4}\.pt", ref["feature_file"])
            and re.fullmatch(r"H_texts_[0-9]{4}\.json", ref["text_file"]), "confined sealed H features")
    p = Path(ledger_root)
    require(sha_file(p/ref["feature_file"]) == ref["feature_sha256"]
            and sha_file(p/ref["text_file"]) == ref["texts_sha256"], "sealed candidate feature SHA")
    features = torch.load(p/ref["feature_file"], map_location="cpu", weights_only=True)
    texts = read_json(p/ref["text_file"])["texts"]
    require(features.ndim == 2 and len(texts) == len(features) and len(texts) == len(set(texts)),
            "complete unique candidate feature coverage")
    return texts, features.double()

def audit_conditional(credit, sampled, edit_index, texts, features, clean_text,
                      partner, anchors, kl_cfg, response, contracts, constraints, probe):
    require(len(credit["support"]) == len(credit["values"]) == len(credit["feature_indices"])
            == len(credit["exact_rewards"]) == len(credit["predicted_next_G_gains"]), "all conditional values")
    rows = credit["support"]; selected = credit["selected"]
    require(0 <= selected < len(rows), "selected conditional action")
    key_bytes(credit["attribute"])
    if contracts is not None:
        example = contracts.from_text_spans(sampled["original"], sampled["attributes"])
        edits = [constraints.EditRecord(**{**x, "source_span": tuple(x["source_span"])}) for x in sampled["edits"]]
        edit = edits[edit_index]
        other = [e for j, e in enumerate(edits) if j != edit_index]
        context = constraints.replay(example, other, sampled["budget_k"])
        ai = next(j for j, a in enumerate(example.attributes) if a.attribute_id == edit.attribute_id)
        used = {contracts.word_at(example.caption, e.position) for e in other}
        legal = constraints.candidates(example, context, ai, used, probe)
        expected = {(None, None)} | {(pos, char) for pos, chars in legal.items() for char in chars}
        require({(x["position"], x["replacement"]) for x in rows} == expected
                and len(rows) == len(expected), "complete legal positions/characters plus NOOP")
        for x in rows:
            action = other if x["position"] is None else other + [
                constraints.make_edit(example, ai, x["position"], x["replacement"])]
            require(constraints.replay(example, action, sampled["budget_k"]) == x["text"],
                    "counterfactual original Unicode replay")
    require(rows[selected]["text"] == sampled["adversarial"], "conditional selected text exact")
    positions = list(dict.fromkeys(x["position"] for x in rows))
    logits = torch.tensor([next(x["position_logit"] for x in rows if x["position"] == p)
                           for p in positions], dtype=torch.float64)
    pp = logits.softmax(0)
    for j, position in enumerate(positions):
        group = [i for i, x in enumerate(rows) if x["position"] == position]
        for i in group:
            close(rows[i]["position_logit"], logits[j], "shared position logit")
        rp = (torch.ones(1, dtype=torch.float64) if position is None else
              torch.tensor([rows[i]["replacement_logit"] for i in group], dtype=torch.float64).softmax(0))
        for i, p in zip(group, rp):
            close(rows[i]["probability"], pp[j]*p, "conditional policy probability")
    for i, row in enumerate(rows):
        j = credit["feature_indices"][i]
        require(0 <= j < len(features) and texts[j] == row["text"], "candidate feature text identity")
        reward = profile_reward(features[j:j+1].expand(3, -1), clean_text.expand(3, -1),
                                partner, anchors, kl_cfg)
        close(credit["exact_rewards"][i], reward, "exact counterfactual KL reward")
        close(credit["values"][i], reward+response.response_weight*credit["predicted_next_G_gains"][i],
              "exact reward plus learned next-update value")
    probabilities = torch.tensor([x["probability"] for x in rows], dtype=torch.float64)
    close(probabilities.sum(), 1., "full conditional probability mass")
    value = torch.tensor(credit["values"], dtype=torch.float64)
    baseline = (value*probabilities).sum()
    close(credit["baseline"], baseline, "conditional expectation baseline")
    close(credit["advantage"], value[selected]-baseline, "conditional edit advantage")

def audit_attraction(log, cfg, contracts, constraints, anchors, bank_features, view_bank,
                     augmentation, *, response_cfg, ledger_root=None, native_config=None,
                     expected_teacher_before=None):
    response = Settings(**response_cfg)
    projected = dict(log)
    projected["G"] = Projection(log["G"], base_g)
    projected["H"] = Projection(log["H"], base_h)
    report = audit_base(projected, cfg, contracts, constraints, anchors, bank_features, view_bank, augmentation)
    ids = log["bank"]["ordered_train_ids"]; previous = expected_teacher_before
    probe = None; positive_map = None
    if native_config is not None:
        probe_module = __import__("tokenizer_probe"); tokenizer_module = __import__("utils.simple_tokenizer", fromlist=["SimpleTokenizer"])
        probe = probe_module.TokenizerProbe(tokenizer_module.SimpleTokenizer(bpe_path=native_config["sources"]["bpe"]["path"]))
        entries = read_json(Path(native_config["prototype"]["history_root"])/"qualification/train.json")["entries"]
        positive_map = {e["identity"]["row_id"]: e["identity"]["pid"] for e in entries}
    if log["round"] == 1:
        previous_bank = anchors[:, None].expand(-1, 3, -1).clone()
    else:
        previous_bank = torch.load(Path(ledger_root).parent/("round_%04d" % (log["round"]-1))/"bank_view_features.pt",
                                   map_location="cpu", weights_only=True)
    virtual_g = virtual_h = probe_images = probe_texts = candidates_count = g_probes = h_probes = 0
    for g in log["G"]:
        md = g["attribute_response"]
        require(md["four_mode_reward_optimized"] is False and md["original_TopK_unchanged"], "new internal guidance declaration")
        if previous is not None:
            require(md["teacher_before"] == previous, "teacher continuous G state")
        weights = md["weights"]
        if weights["weights"]:
            logits = torch.tensor(weights["prior"], dtype=torch.float64).log() + torch.tensor(
                weights["prediction"], dtype=torch.float64)/response.weight_temperature
            expected = 1-response.weight_mix+response.weight_mix*len(logits)*logits.softmax(0)
            for a, b in zip(weights["weights"], expected):
                close(a, b, "meta-guided mean-one attribute weights")
        for index, rec in enumerate(g["records"]):
            d = rec["attribute_response"]
            w = 1. if rec["phase"] == "joint" else weights["weights"][index]
            close(d["local_weight"], w, "attribute update weight")
            close(rec["local"], d["unweighted_local"]*w, "weighted original attribute Local")
        look = md["lookahead"]
        if look["performed"]:
            g_probes += 1
            require(look["support"] != look["probe"], "distinct support/probe")
            baseline = look["candidates"][0]
            for item in look["candidates"]:
                merged = dict(item, support=look["support"], probe=look["probe"], role="train")
                audit_probe(merged, previous_bank, anchors, cfg, response, ids, positive_map)
                require(item["isolation"]["formal_before"] == baseline["isolation"]["formal_before"]
                        and item["isolation"]["rng_before"] == baseline["isolation"]["rng_before"],
                        "all lookahead candidates start from same model/Adam/RNG")
                require(item["virtual_G_steps"] == g["K"]+1 and item["virtual_H_steps"] == 1,
                        "full original virtual K+1 and single H step")
                require(item["counts"]["G"] == 3*(g["K"]+1)+6, "virtual G updates and support/probe generation count")
                require(item["counts"]["H"] == 6+2*len(item["before"]["actions"]), "all virtual H probe/train forwards")
                virtual_g += item["counts"]["G"]; virtual_h += item["counts"]["H"]
                probe_images += item["IRRA_images"]; probe_texts += item["IRRA_texts"]
            require(len(look["targets"]) == len(look["candidates"])-1, "all tested attribute targets")
            for target, item in zip(look["targets"], look["candidates"][1:]):
                close(target, item["gain"]-baseline["gain"], "relative next-H update target")
            audit_regression(look["teacher_update"])
        previous = md["teacher_after"]
    for h in log["H"]:
        if h.get("fixed_NOOP"):
            continue
        md = h["attribute_response"]
        if previous is not None:
            require(md["teacher_before"] == previous, "teacher continuous H state")
        require(md["four_mode_reward_optimized"] is False, "no old four-mode objective")
        texts, features = candidate_features(md, ledger_root)
        require(h["IRRA_text_samples"] == 3+len(texts), "all actual H counterfactual encodings")
        require(len(md["credits"]) == len(h["sampled"]["edits"]), "one conditional credit per actual edit")
        partner = torch.tensor(h["attraction"]["partner_view_features"], dtype=torch.float64)
        clean_text = torch.tensor(md["original_features"][:1], dtype=torch.float64)
        terms = []
        for j, credit in enumerate(md["credits"]):
            audit_conditional(credit, h["sampled"], j, texts, features, clean_text, partner,
                              anchors, cfg, response, contracts if probe else None, constraints, probe)
            candidates_count += len(credit["support"])
            terms.append(-response.credit_weight*credit["advantage"]*credit["position_replacement_log_prob"])
            if "next_G_probe" in credit:
                h_probes += 1
                p = credit["next_G_probe"]
                audit_probe(p, view_bank, anchors, cfg, response, ids, positive_map)
                require(p["Global_only"] and p["new_JVPs"] == 0 and not p["historical_JVP_relabelled"],
                        "explicit H->G global-only lookahead")
                require(p["counts"]["G"] == 15 and p["counts"]["H"] == 0, "one global virtual G step and two banks")
                virtual_g += p["counts"]["G"]; virtual_h += p["counts"]["H"]
                probe_images += p["IRRA_images"]; probe_texts += p["IRRA_texts"]
        close(md["credit_loss"], sum(terms)/max(1, len(terms)), "conditional credit loss")
        d = md["attribute_distill"]
        distill = 0.
        if d["optimized"]:
            target = torch.tensor(d["target"], dtype=torch.float64)
            lp = torch.tensor(d["log_probs"], dtype=torch.float64)
            distill = response.attribute_distill_weight*(target*(target.log()-lp)).sum()
        close(md["attribute_distill_loss"], distill, "attribute-head distillation")
        close(md["optimized_loss"], h["SCST_loss"]+md["credit_loss"]+md["attribute_distill_loss"],
              "full optimized H objective")
        for u in md["teacher_updates"]:
            audit_regression(u)
        previous = md["teacher_after"]
    require(previous == log["attribute_teacher_after"], "sealed end-of-round teacher identity")
    report.update(attribute_teacher_after=previous, conditional_candidates=candidates_count, G_to_H_probes=g_probes, H_to_G_probes=h_probes,
                  virtual_G_samples=virtual_g, virtual_H_forwards=virtual_h,
                  probe_IRRA_images=probe_images, probe_IRRA_texts=probe_texts,
                  no_G_H_victim_forward_in_audit=True)
    return report
