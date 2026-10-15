"""训练专用冻结特征奖励接口；政策推理模块不导入此模块。"""
from dataclasses import dataclass, asdict
import math
import torch

def frozen_features(x, shape=None):
    if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 512
            or x.device.type != "cpu" or x.dtype != torch.float32
            or x.requires_grad or not torch.isfinite(x).all()
            or not torch.allclose(x.norm(dim=1), torch.ones(len(x)), atol=2e-4, rtol=0)):
        raise ValueError("detached finite unit CPU retrieval features [N,512] required")
    if shape is not None and x.shape != shape:
        raise ValueError("feature shape mismatch")

@dataclass(frozen=True)
class TrainGallery:
    clean: torch.Tensor
    adversarial: torch.Tensor
    positive_mask: torch.Tensor
    paired_index: int
    query_split: str
    gallery_splits: tuple[str, ...]
    query_role: str
    gallery_roles: tuple[str, ...]

    def __post_init__(self):
        if type(self.gallery_splits) is not tuple or self.query_split != "train" or len(self.gallery_splits) != len(self.clean) or any(s != "train" for s in self.gallery_splits):
            raise ValueError("reward feedback permits training data only")
        if (self.query_role != "train" or type(self.gallery_roles) is not tuple
                or len(self.gallery_roles) != len(self.clean)
                or any(role != "train" for role in self.gallery_roles)):
            raise ValueError("internal validation cannot supply reward feedback")
        frozen_features(self.clean)
        frozen_features(self.adversarial, self.clean.shape)
        m = self.positive_mask
        if (not isinstance(m, torch.Tensor) or m.dtype != torch.bool or m.device.type != "cpu" or m.shape != (len(self.clean),)
                or not m.any() or m.all()):
            raise ValueError("training positives and negatives required")
        if type(self.paired_index) is not int or not 0 <= self.paired_index < len(m) or not m[self.paired_index]:
            raise ValueError("paired image must be a training positive")

@dataclass(frozen=True)
class Reward:
    total: float
    R_sim: float
    R_rank: float
    clean_sim_drop: float
    adv_sim_drop: float
    synergy_increment: float
    clean_margin_drop: float

class RewardEvaluator:
    def __init__(self, gallery, encode_text, lambda_sim=1., lambda_rank=1.):
        if not isinstance(gallery, TrainGallery):
            raise TypeError("explicit train-only reward context required")
        if any(not math.isfinite(v) or v < 0 for v in (lambda_sim, lambda_rank)) or lambda_sim+lambda_rank == 0:
            raise ValueError("finite nonnegative reward weights required")
        self.gallery = gallery
        self.encode_text = encode_text
        self.weights = lambda_sim, lambda_rank

    @torch.no_grad()
    def evaluate_many(self, original, variants):
        # callback只拿文本；真实模型冻结和split来源由下一阶段接线核验。
        features = self.encode_text((original, *variants))
        frozen_features(features, torch.Size((1+len(variants), 512)))
        ctx = self.gallery
        adv_scores = features @ ctx.adversarial.T
        clean_scores = features @ ctx.clean.T
        # 明确冻结Margin定义：最佳同PID正例分数−最大负例分数。
        def margins(scores):
            return scores[:, ctx.positive_mask].max(1).values - scores[:, ~ctx.positive_mask].max(1).values
        am, cm = margins(adv_scores), margins(clean_scores)
        result = []
        for i in range(1, len(features)):
            adv_drop = float(adv_scores[0, ctx.paired_index]-adv_scores[i, ctx.paired_index])
            clean_drop = float(clean_scores[0, ctx.paired_index]-clean_scores[i, ctx.paired_index])
            rank_drop = float(am[0]-am[i])
            result.append(Reward(self.weights[0]*adv_drop+self.weights[1]*rank_drop,
                                 adv_drop, rank_drop, clean_drop, adv_drop,
                                 adv_drop-clean_drop, float(cm[0]-cm[i])))
        return tuple(result)
