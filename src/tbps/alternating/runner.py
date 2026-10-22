"""两轮A1/B控制流。当前仅开放CPU Fixture；真实GPU执行等待新的明确授权。"""
from .contracts import require
from .state import clone_tree, tree_sha
from tbps.artifacts import digest

class Prototype:
    def __init__(self, protocol, backend, journal):
        require(protocol["arms"] == ["A1", "B"] and protocol["rounds"] == 2, "fixed A1/B design")
        require(len(protocol["samples"]) == 8 and len(set(protocol["samples"])) == 8,
                "fixed eight update members")
        require(len(protocol["source_space_sha256"]) == 64 and
                len(protocol["clean_gallery_sha256"]) == 64, "fixed ModelSpace/Clean bindings")
        ids = protocol["train_ids"]
        require(len(ids) == 2048 and len(set(ids)) == 2048
                and set(protocol["samples"]) <= set(ids), "full Train reward identities")
        require(backend.execution_kind == "synthetic_cpu_fixture", "GPU execution not authorized in this revision")
        self.p, self.b, self.j = protocol, backend, journal

    def run(self, parent):
        """backend各函数持有模型权限；H rollout只收到公开Caption对象，无bank/PID。"""
        required={"G","H","G_Adam","H_Adam","G_RNG","H_RNG","cursor","parents"}
        require(required <= set(parent), "complete G/H/dual Adam/RNG parent required")
        require(parent["parents"]["protocol"] == digest(self.p), "parent/new protocol mismatch")
        results = {}
        for arm in self.p["arms"]:
            state = clone_tree(parent)
            for index in (1, 2):
                self.b.restore_complete(state)
                self.j.begin(arm, index, tree_sha(state))
                self.b.enter_phase_rng("H")
                public = [self.b.public_text(row_id) for row_id in self.p["samples"]]
                require(all(set(x) == {"caption", "attributes"} for x in public), "H public input leak")
                traces = [self.b.h_rollout(x) for x in public]
                self.b.leave_phase_rng("H")
                conditions = [
                    self.b.condition(arm, row_id, trace)
                    for row_id, trace in zip(self.p["samples"], traces)
                ]
                # A1/B的相同8条、K/选择顺序来自已冻结原G条件，不随H选择改变。
                self.b.enter_phase_rng("G")
                before_h = self.b.h_fingerprint()
                g_log = []
                for row_id, condition in zip(self.p["samples"], conditions):
                    result = self.b.g_update(row_id, condition)
                    require(result["optimizer_steps"] == condition.payload["K"] + 1,
                            "original G K+1 accounting")
                    g_log.append(result)
                require(before_h == self.b.h_fingerprint(), "H changed during G phase")
                self.b.leave_phase_rng("G")
                g_sha = self.b.g_fingerprint()
                # 完整顺序、单一G版本；不接受仅8图刷新或2040旧图混合。
                bank = self.b.refresh_bank(self.p["train_ids"], arm, index)
                bank.validate(self.p["train_ids"], g_sha, index, arm,
                              space=self.p["source_space_sha256"], clean=self.p["clean_gallery_sha256"])
                self.b.install_new_reward_bank(bank)
                self.b.enter_phase_rng("H")
                h_log = [self.b.h_update(row_id, bank) for row_id in self.p["samples"]]
                require(g_sha == self.b.g_fingerprint(), "G changed during H phase")
                self.b.leave_phase_rng("H")
                self.b.assert_frozen_sources()
                state = self.b.complete_state(index)
                ledger = {"arm": arm, "round": index, "G": g_log, "H": h_log,
                          "bank": bank.__dict__, "all_stages_verified": True,
                          "conditions": [c.identity.key for c in conditions]}
                self.j.commit(arm, index, state, ledger)
                results[(arm, index)] = ledger
        return results
