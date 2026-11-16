"""独立算法配置；控制真实更新与前瞻计算的边界。"""
from dataclasses import dataclass, asdict
import math
from tbps.artifacts import ArtifactError

def require(ok, message):
    if not ok:
        raise ArtifactError(message)

@dataclass(frozen=True)
class Settings:
    width: int = 48
    history: int = 8
    memory_capacity: int = 1024
    probe_interval: int = 32
    probe_attributes: int = 2
    replay_capacity: int = 512
    teacher_batch: int = 16
    teacher_lr: float = 0.0003
    weight_mix: float = 0.5
    weight_temperature: float = 1.0
    response_weight: float = 0.1
    credit_weight: float = 0.1
    attribute_distill_weight: float = 0.1
    rank_weight: float = 1.0
    max_text_batch: int = 32
    coordinate_gradients: bool = False
    seed: int = 42

    def __post_init__(self):
        ints = ("width", "history", "memory_capacity", "probe_interval",
                "probe_attributes", "replay_capacity", "teacher_batch", "max_text_batch", "seed")
        for name in ints:
            v = getattr(self, name)
            require(type(v) is int and v > 0, name + " must be a positive integer")
        for name in ("teacher_lr", "weight_mix", "weight_temperature", "response_weight",
                     "credit_weight", "attribute_distill_weight", "rank_weight"):
            v = getattr(self, name)
            require(type(v) in (int, float) and math.isfinite(v) and v >= 0, "invalid " + name)
        require(0 <= self.weight_mix <= 1 and self.weight_temperature > 0 and self.teacher_lr > 0,
                "positive temperature/LR and bounded weight mixture")
        require(type(self.coordinate_gradients) is bool, "boolean gradient coordination")
        require(self.memory_capacity >= self.history and self.replay_capacity >= self.teacher_batch,
                "bounded history/replay capacities")
    def record(self):
        return asdict(self)
