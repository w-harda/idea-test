"""IRRA 官方 Text -> Image 的 CMC、mAP、mINP 与逐查询诊断。"""
from __future__ import annotations

import numpy as np
import torch

IRRA_METRICS_SHA = "94910252a9e42f0820d49c11203b31d3b67951f58624f66abff4bf662500d5fb"


@torch.no_grad()
def text_to_image_metrics(similarity, query_ids, gallery_ids):
    scores = torch.as_tensor(similarity)
    qids = torch.as_tensor(query_ids, device=scores.device)
    gids = torch.as_tensor(gallery_ids, device=scores.device)
    if (scores.ndim != 2 or scores.shape != (len(qids), len(gids))
            or not len(qids) or not len(gids) or not torch.isfinite(scores).all()):
        raise ValueError("Text -> Image 相似度矩阵或 ID 无效")
    # 与 IRRA utils/metrics.py::rank 一致，全排序；无 ReID camera 排除。
    order = torch.argsort(scores, dim=1, descending=True)
    matches = gids[order].eq(qids[:, None])
    relevant = matches.sum(1)
    if not (relevant > 0).all():
        raise ValueError("某条文本 query 的 person ID 不在图库中")
    positions = torch.arange(1, scores.shape[1] + 1, device=scores.device)
    cumulative = matches.cumsum(1)
    ap = (cumulative / positions * matches).sum(1) / relevant
    last = (matches * positions).amax(1)
    inp = relevant.float() / last
    first = torch.where(matches, positions, scores.shape[1] + 1).amin(1)
    rows = [
        {"first_hit_rank": int(rank), "positive_images": int(rel),
         "AP": float(a), "INP": float(i)}
        for rank, rel, a, i in zip(first.cpu(), relevant.cpu(), ap.cpu(), inp.cpu())
    ]
    return aggregate_rows(rows), rows


def aggregate_rows(rows):
    if not rows:
        raise ValueError("没有可汇总的查询")
    ranks = np.array([r["first_hit_rank"] for r in rows])
    return {
        "query_count": len(rows), "metric_unit": "percent",
        "R@1": float((ranks <= 1).mean() * 100),
        "R@5": float((ranks <= 5).mean() * 100),
        "R@10": float((ranks <= 10).mean() * 100),
        "mAP": float(np.mean([r["AP"] for r in rows]) * 100),
        "mINP": float(np.mean([r["INP"] for r in rows]) * 100),
        "mean_first_hit_rank": float(ranks.mean()),
        "median_first_hit_rank": float(np.median(ranks)),
    }


def paired_summary(clean_rows, adv_rows):
    if ([r["row_id"] for r in clean_rows] != [r["row_id"] for r in adv_rows]
            or any(c["person_id"] != a["person_id"]
                   for c, a in zip(clean_rows, adv_rows))):
        raise ValueError("Clean/adv 的 query 顺序或 ID 不一致")
    clean = aggregate_rows(clean_rows)
    adv = aggregate_rows(adv_rows)
    delta = np.array([a["first_hit_rank"] - c["first_hit_rank"]
                      for c, a in zip(clean_rows, adv_rows)])
    return {
        "clean": clean, "adversarial": adv,
        "DR_mAP": (clean["mAP"] - adv["mAP"]) / clean["mAP"]
                  if clean["mAP"] else None,
        "DR_mAP_definition": "单一 TBPS victim IRRA 的 mAP Drop Rate；不是多 ReID victim 的 aAP/mDR",
        "diagnostics": {
            "mean_delta_rank": float(delta.mean()),
            "median_delta_rank": float(np.median(delta)),
            **{name: {"count": int(mask.sum()), "fraction": float(mask.mean())}
               for name, mask in [
                   ("delta_rank_positive", delta > 0),
                   ("delta_rank_zero", delta == 0),
                   ("delta_rank_negative", delta < 0)]},
        },
    }
