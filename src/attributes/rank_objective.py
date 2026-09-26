"""图库级首次正确 ID 图片名次的可微目标。"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _masks(scores: Tensor, query_ids: Sequence[int | str], gallery_ids: Sequence[int | str]) -> tuple[Tensor, Tensor]:
    if scores.ndim != 2 or scores.shape != (len(query_ids), len(gallery_ids)):
        raise ValueError("scores shape must be [queries, gallery]")
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positive = torch.tensor(
        [[qid == gid for gid in gallery_ids] for qid in query_ids],
        device=scores.device, dtype=torch.bool,
    )
    if not positive.any(dim=1).all() or positive.all(dim=1).any():
        raise ValueError("each query needs positive and negative gallery images")
    return positive, ~positive


def soft_first_hit_rank(scores: Tensor, query_ids: Sequence[int | str],
                        gallery_ids: Sequence[int | str], tau: float) -> Tensor:
    """1 + 逐张错误 ID 图片对最高分正确图片的软超越。"""
    if not 0 < tau < float("inf"):
        raise ValueError("tau must be finite and positive")
    positive, negative = _masks(scores, query_ids, gallery_ids)
    best_positive = scores.masked_fill(~positive, -torch.inf).max(dim=1).values
    outrank = torch.sigmoid((scores - best_positive[:, None]) / tau)
    return 1 + (outrank * negative).sum(dim=1)


def query_utility(delta: Tensor) -> Tensor:
    """正收益 log1p，负收益线性。"""
    return torch.where(delta > 0, torch.log1p(delta.clamp_min(0)), delta)


def rank_loss(clean_scores: Tensor, adv_scores: Tensor,
              query_ids: Sequence[int | str], gallery_ids: Sequence[int | str],
              tau: float) -> Tensor:
    if clean_scores.shape != adv_scores.shape:
        raise ValueError("clean and adversarial gallery must have the same members")
    clean = soft_first_hit_rank(clean_scores.detach(), query_ids, gallery_ids, tau)
    adv = soft_first_hit_rank(adv_scores, query_ids, gallery_ids, tau)
    return -query_utility(adv - clean).mean()


def exact_first_hit_rank(scores: Tensor, query_ids: Sequence[int | str],
                         gallery_ids: Sequence[int | str]) -> Tensor:
    """评测使用真实图片列表名次；并列按图库原始顺序。"""
    positive, _ = _masks(scores, query_ids, gallery_ids)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    hits = positive.gather(1, order)
    return hits.to(torch.int64).argmax(dim=1) + 1


def calibrate_tau(clean_scores: Tensor, query_ids: Sequence[int | str],
                  gallery_ids: Sequence[int | str]) -> float:
    """训练集干净分数差 |negative - best positive| 的中位数。"""
    positive, negative = _masks(clean_scores, query_ids, gallery_ids)
    best = clean_scores.masked_fill(~positive, -torch.inf).max(dim=1).values
    differences = (clean_scores - best[:, None]).abs()[negative]
    value = float(differences.median().item())
    if value <= 0:
        raise ValueError("clean score gaps do not yield a positive tau")
    return value
