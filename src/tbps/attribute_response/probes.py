"""少量隔离一步前瞻：G->H使用K+1虚拟G步，H->G使用单个全局G步。"""
from dataclasses import asdict
import torch
from tbps.alternating.state import tree_sha
from .isolation import sandbox, clone_trainable
from .settings import require

class ResponseProber:
    def __init__(self, backend):
        self.b = backend

    def other(self, row):
        ids = self.b.p["samples"]
        start = ids.index(row)
        for offset in range(1, len(ids)):
            candidate = ids[(start + offset) % len(ids)]
            if self.b.entries[candidate]["eligible"]:
                return candidate
        raise ValueError("lookahead needs a distinct eligible Train probe row")

    def model_pair(self, side):
        b = self.b
        model, optimizer = (b.g, b.ga) if side == "G" else (b.h, b.ha)
        clone, opt = clone_trainable(model, optimizer)
        def count(module, args):
            b.counts[side] += len(args[0]) if side == "G" else 1
            b.virtual_counts[side] += len(args[0]) if side == "G" else 1
        clone.register_forward_pre_hook(count)
        return clone, opt

    def image_features(self, g, row):
        from budget_projection import add_projected_delta
        views = self.b.make_views(row)
        mode = g.training; g.eval()
        try:
            with torch.no_grad():
                adv = add_projected_delta(views, views, g(views))
                feature = self.b.encoder(adv).detach().cpu().clone()
        finally:
            g.train(mode)
        return feature

    def bank(self, g, support, probe):
        b = self.b
        # 前瞻近似：冻结全Train Bank，仅替换本次support/probe对应行；正式Bank不修改。
        views = b.current_view_features.detach().clone()
        for row in (support, probe):
            views[b.p["train_ids"].index(row)] = self.image_features(g, row)
        return views

    def reward(self, row, bank, example):
        b = self.b; runtime = b.native.get("formal_training.runtime")
        query = b.catalog.get_train(row)
        index = b.p["train_ids"].index(row)
        positive = torch.tensor([s.pid == query.pid for s in b.catalog.train])
        gallery = runtime.TrainGallery(b.clean, bank[:, 0].clone(), positive, index,
                        "train", ("train",) * len(bank), "train", ("train",) * len(bank))
        return lambda k: runtime.FreshReward(gallery, query, example, k, b.runtime)

    def utility(self, h, row, bank, actions=None):
        from .probe_policy import probe_actions, trajectory_log_prob
        b = self.b; example = b.example(row)
        if actions is None:
            actions = probe_actions(h, example, version=b.round_index)
        constraints = b.native.get("constraints")
        for out in actions:
            require(constraints.replay(example,
                [constraints.EditRecord(**{**e, "source_span": tuple(e["source_span"])}) for e in out["edits"]],
                out["budget_k"]) == out["adversarial"], "probe H native Unicode replay")
        with torch.no_grad():
            log_probs = torch.stack([trajectory_log_prob(h, example, out) for out in actions])
            probabilities = log_probs.softmax(0).detach().double()
        f = b.runtime.encode_text((example.caption, *[out["adversarial"] for out in actions]))
        index = b.p["train_ids"].index(row); partner = bank[index]
        query = b.catalog.get_train(row)
        positive = torch.tensor([s.pid == query.pid for s in b.catalog.train])
        require(positive.any() and (~positive).any(), "Train probe positive/negative candidates")
        rewards = []; margins = []
        for current in f[1:]:
            rewards.append(float(b.profile_cpu.terms(current[None].expand(3, -1),
                f[:1].expand(3, -1), partner)["reward"].mean()))
            scores = bank[:, 0] @ current
            margins.append(float(scores[~positive].max() - scores[positive].max()))
        kl = float(probabilities @ torch.tensor(rewards, dtype=torch.float64))
        margin = float(probabilities @ torch.tensor(margins, dtype=torch.float64))
        value = kl + b.response_settings.rank_weight * margin
        return value, {"row": row, "role": "train", "actions": actions,
            "probe_policy_mode": "conditional_fixed_legal_trajectory_support",
            "log_probs": log_probs.tolist(), "probabilities": probabilities.tolist(),
            "per_action_KL": rewards, "per_action_margin": margins, "KL_reward": kl,
            "negative_minus_positive_margin": margin,
            "positive_indices": positive.nonzero().flatten().tolist(), "utility": value,
            "bank_feature_sha256": tree_sha(bank), "text_features": f.tolist(),
            "base_bank_sha256": tree_sha(b.current_view_features),
            "patches": [{"index": j, "features": bank[j].tolist()} for j in range(len(bank))
                        if not torch.equal(bank[j], b.current_view_features[j])]}

    def g_to_h(self, row, condition, keys, states, clean, views):
        b = self.b; settings = b.response_settings; k = len(keys)
        if not k or not b.entries[row]["eligible"]:
            return {"performed": False, "reason": "no selected attribute or eligible H support"}
        probe = self.other(row)
        selected = [(b.probe_cursor + i) % k for i in range(min(settings.probe_attributes, k))]
        candidates = [torch.ones(k)]
        for index in selected:
            value = torch.ones(k); value[index] = 2.; value *= k / value.sum()
            candidates.append(value)
        records = []
        from .g_step import train_single_sample
        from tbps.augmented_attraction.h_reward import train_step as virtual_h_step
        for weights in candidates:
            count_before = dict(b.virtual_counts); image_before = b.counts["IRRA_image"]
            text_before = b.counts["IRRA_text"]
            with sandbox((b.g, b.h), (b.ga, b.ha), b.sampler) as proof:
                g, ga = self.model_pair("G"); h, ha = self.model_pair("H")
                result = train_single_sample(g, b.encoder, ga, views, condition.payload,
                       profile=b.profile_g, clean_features=clean, attribute_weights=weights,
                       coordinate_gradients=False)
                bank = self.bank(g, row, probe)
                before, before_evidence = self.utility(h, probe, bank)
                example = b.example(row)
                phase = b.native.get("phase4_training")
                log, _ = virtual_h_step(phase, h, ha, example, b.sampler,
                    self.reward(row, bank, example), 1, row, b.profile_cpu,
                    bank[b.p["train_ids"].index(row)])
                after, after_evidence = self.utility(h, probe, bank, before_evidence["actions"])
                del g, ga, h, ha
            records.append({"weights": weights.tolist(), "before": before_evidence,
                 "after": after_evidence, "gain": after-before, "isolation": proof,
                 "virtual_G_steps": result["optimizer_steps"], "virtual_H_steps": 1,
                 "virtual_H_loss": log["SCST_loss"],
                 "counts": {s: b.virtual_counts[s]-count_before[s] for s in count_before},
                 "IRRA_images": b.counts["IRRA_image"]-image_before,
                 "IRRA_texts": b.counts["IRRA_text"]-text_before})
        targets = [r["gain"]-records[0]["gain"] for r in records[1:]]
        batch = b.teacher.features([keys[i] for i in selected], [states[i] for i in selected],
                                   version=b.round_index)
        update = b.teacher.observe(batch, targets, 0)
        for i, target in zip(selected, targets):
            b.teacher.remember(keys[i], states[i], [target, 0.], version=b.round_index)
        b.probe_cursor += len(selected)
        return {"performed": True, "support": row, "probe": probe, "role": "train",
                "selected_attribute_indices": selected, "candidates": records,
                "targets": targets, "teacher_features": batch, "teacher_update": update,
                "bank_mode": "frozen_full_Train_with_support_probe_row_replacements",
                "not_fresh_full_virtual_bank": True}

    def h_to_g(self, row, text_feature, edit_index):
        b = self.b
        if edit_index >= b.response_settings.probe_attributes:
            return None
        probe = self.other(row)
        image_before = b.counts["IRRA_image"]; text_before = b.counts["IRRA_text"]
        count_before = dict(b.virtual_counts)
        from budget_projection import add_projected_delta
        with sandbox((b.g, b.h), (b.ga, b.ha), b.sampler) as proof:
            g, ga = self.model_pair("G")
            g.train()
            before_bank = self.bank(g, row, probe)
            before, before_evidence = self.utility(b.h, probe, before_bank)
            views = b.make_views(row)
            with torch.no_grad():
                clean = b.encoder(views).detach().clone()
            clean[0] = b.clean[b.p["train_ids"].index(row)].to(clean)
            ga.zero_grad(set_to_none=True)
            feature = b.encoder(add_projected_delta(views, views, g(views)))
            partner = text_feature.to(feature)[None].expand(3, -1).detach()
            loss = b.profile_g.terms(feature, clean, partner)["loss"].mean()
            require(torch.isfinite(loss), "finite virtual G global loss")
            loss.backward(); ga.step()
            after_bank = self.bank(g, row, probe)
            after, after_evidence = self.utility(b.h, probe, after_bank, before_evidence["actions"])
            del g, ga
        gain = after-before
        b.h_probe_cursor = getattr(b, "h_probe_cursor", 0)+1
        return gain, {"performed": True, "support": row, "probe": probe, "role": "train",
             "before": before_evidence, "after": after_evidence, "gain": gain, "isolation": proof,
             "virtual_G_steps": 1, "virtual_H_steps": 0, "Global_only": True,
             "new_JVPs": 0, "historical_JVP_relabelled": False,
             "counts": {s: b.virtual_counts[s]-count_before[s] for s in count_before},
             "IRRA_images": b.counts["IRRA_image"]-image_before,
             "IRRA_texts": b.counts["IRRA_text"]-text_before}
