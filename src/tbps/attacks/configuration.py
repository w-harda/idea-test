"""冻结攻击器生成阶段的纯配置校验；与未来victim评分执行分开。"""
from dataclasses import asdict
from ..config import IRRA
from ..contracts import AttackBudget, ModelSpec, sha256
from .base import need
from .gv2 import BUDGET as G
from .hv1 import BUDGET as H
from .provenance import IRRA_SHA

def validate_generation(registry, name, *, budget, dataset_adapter_id, permissions,
                        attack_mode, surrogate, surrogate_sha256, victim, victim_sha256,
                        victim_adapter_ready=False):
    info=registry.describe(name)
    sha256(surrogate_sha256);sha256(victim_sha256)
    need(type(budget) is AttackBudget and type(surrogate) is ModelSpec and type(victim) is ModelSpec,
         "typed budget/ModelSpec required")
    need(dataset_adapter_id=="cuhk_pedes", "other Dataset Adapters are not implemented")
    need(type(permissions) is frozenset and permissions==info.online_access,
         "only declared public generation access is allowed; no PID/pairs/target feedback")
    need(not info.spec.attack_time_optimization and not info.gradients_for_generation and
         not info.rewards_for_generation and not info.surrogate_for_generation,
         "this configuration entry supports frozen public-input generation only")
    need(surrogate==IRRA and surrogate_sha256==IRRA_SHA, "frozen IRRA training/constraint provenance required")
    need("score" in victim.capabilities, "victim must declare native scoring")
    need(attack_mode in ("white_box","cross_model"), "cross-dataset execution not implemented")
    different=(victim.id!=surrogate.id or victim_sha256!=surrogate_sha256)
    need(different==(attack_mode=="cross_model"), "model/checkpoint axis conflicts with attack mode")
    if name in ("gv2","hv1","joint"):
        expected={"gv2":G,"hv1":H,"joint":AttackBudget(G.image,H.text)}[name]
        need(budget==expected, "cannot change frozen attack budget/constraints")
    need(type(victim_adapter_ready) is bool, "explicit victim readiness")
    # 声明可兼容原生fusion/任意维度，并不注册不存在的RaSa/APTM。
    need(not victim_adapter_ready or victim==IRRA, "target model adapter is absent")
    return {"plugin":name,"generation_api_available":True,"attack_budget":asdict(budget),
            "victim_model_id":victim.id,"victim_scoring":victim.scoring,
            "victim_adapter_ready":victim_adapter_ready,
            "victim_feedback_to_attack":False,"evaluation_engine_available":False,
            "real_G_H_equivalence":"pending Stage 3B"}
