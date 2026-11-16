"""IRRA生产接线：保留原G(I)/H(T)推理和完整当前G三视图库。"""
from contextlib import contextmanager
from dataclasses import asdict
import torch
from tbps.artifacts import read_json, json_bytes, sha_file
from tbps.alternating.state import clone_tree, tree_sha
from tbps.alternating.native_sources import native_session
from tbps.augmented_attraction.backend import AttractionBackend
from tbps.augmented_attraction.views import ViewSettings, recipe, apply_views
from .settings import Settings, require
from .teacher import ResponseTeacher
from .memory import semantic_key
from .probes import ResponseProber

class AttributeResponseBackend(AttractionBackend):
    def __init__(self, *args):
        super().__init__(*args)
        self.response_settings = Settings(**self.c["attribute_response"])
        self.teacher = ResponseTeacher(self.response_settings)
        self.virtual_counts = {"G": 0, "H": 0}
        self.probe_cursor = 0
        self.h_probe_cursor = 0
        self.response_visits = {"G": 0, "H": 0}
        self.prober = ResponseProber(self)
        self.anchors = {}
        for row in self.p["samples"]:
            md = read_json(self.gentries[row]["metadata_path"])
            self.anchors[row] = md["source_selection"]["selected_attributes"]

    def load_reward_inputs(self):
        super().load_reward_inputs()
        self.current_view_features = self.clean[:, None].expand(-1, 3, -1).clone()
        self.view_bank_sha = tree_sha(self.current_view_features)

    def example(self, row):
        return self.native.get("formal_training.data").example_from(self.entries[row])

    def make_views(self, row):
        image = self.image(row)
        plan = recipe(self.view_settings, row, self.round_index, *image.shape[-2:])
        return apply_views(image.detach().cpu(), plan).to(image.device)

    def attribute_states(self, row, payload, attributes):
        index = self.p["train_ids"].index(row)
        source = self.anchors[row]
        mapping = {semantic_key(a): i for i, a in enumerate(source)}
        states = []; keys = []
        clean = self.clean[index]; current = self.current_features[index]
        for attr in attributes:
            key = semantic_key(attr); keys.append(key); j = mapping.get(key)
            score = float(payload["S_raw"][j]) if j is not None else 0.
            prior = float(payload["score_weights"][j]) if j is not None else 1/max(1, len(attributes))
            norm = float(payload["attribute_jvps"][j].norm()) if j is not None else 0.
            unit = payload["attribute_jvps"][j] / max(norm, 1e-12) if j is not None else torch.zeros_like(clean)
            events = self.teacher.memory.read(key)
            last = events[-1]["response"] if events else [0., 0.]
            states.append([score/(1+score), prior, len(attributes)/32., norm/(1+norm),
                float(clean @ unit), float(current @ unit), float(clean @ current),
                float(clean @ payload["z_t"]), self.round_index/60., float(j is not None),
                last[0], last[1]])
        return keys, states

    def g_update(self, row, condition):
        from .g_step import train_single_sample
        self.grant.validate(self.c, "smoke"); self.g.requires_grad_(True)
        index = self.p["train_ids"].index(row)
        views = self.make_views(row)
        with torch.no_grad():
            clean = self.encoder(views).detach().clone()
        clean[0] = self.clean[index].to(clean)
        keys, states = self.attribute_states(row, condition.payload, self.anchors[row])
        require(len(keys) == condition.payload["K"], "original Top-K attribute order preserved")
        teacher_before = tree_sha(self.teacher.state_dict())
        weights, metadata = self.teacher.weights(keys, states, condition.payload["score_weights"],
                                                  version=self.round_index)
        self.response_visits["G"] += 1
        probe = {"performed": False, "reason": "scheduled sparse lookahead"}
        if keys and (self.probe_cursor == 0 or (self.response_visits["G"]-1) % self.response_settings.probe_interval == 0):
            probe = self.prober.g_to_h(row, condition, keys, states, clean, views)
        gradients = []; before = self.g_fingerprint(); teacher_h = self.h_fingerprint()
        def observe(event):
            gradients.append({"record": event["record"], "gradient_sha256": tree_sha(
                {n: p.grad for n, p in self.g.named_parameters() if p.grad is not None})})
        raw = train_single_sample(self.g, self.encoder, self.ga, views, condition.payload,
            observer=observe, profile=self.profile_g, clean_features=clean, attribute_weights=weights,
            coordinate_gradients=self.response_settings.coordinate_gradients)
        result = {k: v for k, v in raw.items() if k not in ("serial_adversarial", "joint_adversarial")}
        result.update(G_before=before, G_after=self.g_fingerprint(), gradient_observations=gradients,
                      G_input_view_samples=3*result["optimizer_steps"],
                      encoder_adv_view_samples=3*result["optimizer_steps"], encoder_clean_view_samples=3)
        result["attraction"] = {"teacher_H_state": teacher_h, "condition_identity": condition.identity.key,
            "anchor_sha256": self.anchor_sha, "candidate_count": len(self.clean),
            "settings": dict(self.c["attraction"]), "partner_text_feature": condition.payload["z_t"].tolist(),
            "view_recipe": recipe(self.view_settings, row, self.round_index, *views.shape[-2:]),
            "clean_view_sha256": tree_sha(views.detach().cpu())}
        result["attribute_response"] = {"weights": metadata, "lookahead": probe,
            "teacher_before": teacher_before, "teacher_after": tree_sha(self.teacher.state_dict()),
            "original_TopK_unchanged": True, "four_mode_reward_optimized": False}
        self.cursor["g_updates"] += result["optimizer_steps"]
        return result

    def h_update(self, row, bank):
        from .h_step import train_step
        self.grant.validate(self.c, "smoke")
        require(bank == self.last_bank and bank.g_state_sha256 == self.g_fingerprint(),
                "H requires full current G bank")
        self.cursor["h_visits"] += 1
        if not self.entries[row]["eligible"]:
            return {"row_id": row, "fixed_NOOP": True, "optimizer_steps_this_step": 0, "IRRA_text_samples": 0}
        self.h.requires_grad_(True); example = self.example(row)
        query = self.catalog.get_train(row); index = self.p["train_ids"].index(row)
        def bind(k):
            return self.reward.bind(query, self.runtime, example, k)
        _, states = self.attribute_states(row, self.original[row].payload, example.attributes)
        self.response_visits["H"] += 1
        callback = None
        if self.h_probe_cursor == 0 or (self.response_visits["H"]-1) % self.response_settings.probe_interval == 0:
            callback = lambda text, edit, i: self.prober.h_to_g(row, text, i)
        def save_features(texts, features):
            from tbps.alternating.formal_state import tensor_bytes
            name = "h_features_" + str(index).zfill(4)
            self.store.publish(name, {"features.pt": tensor_bytes(features), "texts.json": json_bytes({"texts": texts})},
                               {"configuration": self.c["fresh_config_sha256"], "row": row})
            return {"artifact": str(self.store.root/name), "feature_sha256": sha_file(self.store.root/name/"features.pt"),
                    "texts_sha256": sha_file(self.store.root/name/"texts.json")}
        log, _ = train_step(self.native.get("phase4_training"), self.native.get("policy"),
            self.native.get("constraints"), self.h, self.ha, example, self.sampler, bind,
            self.cursor["h_visits"], row, self.profile_cpu, self.current_view_features[index],
            self.teacher, states, version=self.round_index, response_probe=callback, feature_sink=save_features)
        self.cursor["h_updates"] += 1; log["bank_sha256"] = bank.adversarial_sha256
        log["attraction"].update(teacher_G_state=bank.g_state_sha256, anchor_sha256=self.anchor_sha,
            settings=dict(self.c["attraction"]), view_bank_sha256=self.view_bank_sha)
        return log

    def complete_state(self, index):
        state = super().complete_state(index)
        state["attribute_teacher"] = self.teacher.state_dict()
        state["attribute_runtime"] = {"virtual_counts": dict(self.virtual_counts),
            "probe_cursor": self.probe_cursor, "h_probe_cursor": self.h_probe_cursor, "response_visits": dict(self.response_visits)}
        return state

    def restore_complete(self, value):
        self.teacher.load_state_dict(value["attribute_teacher"])
        runtime = value["attribute_runtime"]
        require(set(runtime) == {"virtual_counts", "probe_cursor", "h_probe_cursor", "response_visits"},
                "complete attribute runtime")
        self.virtual_counts = dict(runtime["virtual_counts"]); self.probe_cursor = runtime["probe_cursor"]
        self.h_probe_cursor = runtime["h_probe_cursor"]
        require(all(type(x) is int and x >= 0 for x in (*self.virtual_counts.values(), self.probe_cursor, self.h_probe_cursor, *runtime["response_visits"].values())), "nonnegative absolute auxiliary counters")
        self.response_visits = dict(runtime["response_visits"])
        super().restore_complete({k: v for k, v in value.items()
                                  if k not in ("attribute_teacher", "attribute_runtime")})

@contextmanager
def loaded(c, bundle, grant):
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    torch.cuda.manual_seed_all(42); cuda = clone_tree(torch.cuda.get_rng_state_all())
    for side in ("G", "H"):
        bundle.state[side+"_RNG"]["cuda"] = clone_tree(cuda)
    source = IRRASource.from_dict(read_json(c["sources"]["dataset_config"]["path"])["irra"])
    adapter = IRRAAdapter.from_source(source, device="cuda:0", allow_model_load=True)
    with native_session(c, reuse=bundle.modules) as session:
        backend = AttributeResponseBackend(c, bundle, grant, adapter); backend.native_session = session
        try:
            yield backend
        finally:
            backend.close()
