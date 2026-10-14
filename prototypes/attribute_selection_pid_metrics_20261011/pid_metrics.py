"""PID relevance 指标及跨 victim 聚合；不含攻击或评分逻辑。"""
import math
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from attributes.tbps_retrieval_metrics import text_to_image_metrics, aggregate_rows

MODES = ("S_TOPK", "RANDOM_K", "S_BOTTOMK")

def replace_paired(scores, index, paired_cosine):
    if scores.ndim != 1 or not 0 <= index < len(scores):
        raise ValueError("invalid paired index/scores")
    if not torch.isfinite(scores).all() or not math.isfinite(paired_cosine) or abs(paired_cosine) > 1.00001:
        raise ValueError("nonfinite/invalid cosine")
    result = scores.clone()
    result[index] = paired_cosine
    return result

def check_saved(scores, index, saved, tau=0.02):
    """核对原诊断的同一 soft-rank、严格 greater-than rank 和 tie count。"""
    others = torch.cat((scores[:index], scores[index + 1:]))
    paired = scores[index]
    rank = 1 + int((others > paired).sum())
    ties = int((others == paired).sum())
    soft = float(1 + torch.sigmoid((others - paired) / tau).sum())
    error = abs(soft - saved["paired_soft_rank_F"])
    # 原始数值是 fp32 GPU reduction；仅允许8个该数值处的ULP，并保留实际误差。
    tolerance = max(2e-6, 8 * abs(float(np.spacing(np.float32(saved["paired_soft_rank_F"])))))
    if rank != saved["positive_image_rank"] or ties != saved["positive_image_rank_tie_count"]:
        raise ValueError(f"paired rank/tie reconstruction failed: {rank}/{ties} vs saved {saved}")
    if error > tolerance:
        raise ValueError(f"soft-rank reconstruction failed: abs={error}, tolerance={tolerance}")
    return {"paired_rank_exact": True, "paired_ties_exact": True,
            "soft_rank_abs_error": error, "soft_rank_atol": tolerance,
            "soft_rank_F": soft}

def pid_row(scores, query_pid, gallery_pids):
    _, rows = text_to_image_metrics(scores.unsqueeze(0), [query_pid], gallery_pids)
    return rows[0]

def drop_rate(clean, adversarial):
    if not math.isfinite(clean) or not math.isfinite(adversarial) or not 0 <= clean <= 100 or not 0 <= adversarial <= 100:
        raise ValueError("mAP must be finite percent in [0,100]")
    return None if clean == 0 else (clean - adversarial) / clean * 100

def aggregate_victims(reports):
    """论文口径：先跨 victim 平均mAP，再算mDR；另列mean-per-victim-DR。"""
    if not reports:
        raise ValueError("no victim reports")
    contract = reports[0]["aggregation_contract"]
    victim_ids = [r["victim_id"] for r in reports]
    if len(set(victim_ids)) != len(victim_ids):
        raise ValueError("duplicate victim identity; cannot average repeated IRRA as multiple victims")
    if any(r["aggregation_contract"] != contract for r in reports):
        raise ValueError("different query/gallery/attack protocol cannot be aggregated")
    clean = float(np.mean([r["clean"]["mAP"] for r in reports]))
    out = {"victim_count": len(reports), "victim_ids": victim_ids,
           "single_victim": len(reports) == 1, "metric_unit": "percent",
           "aAP_clean": clean, "modes": {},
           "mDR_definition": "(aAP_clean-aAP_adversarial)/aAP_clean*100",
           "mean_per_victim_DR_definition": "mean((clean_mAP_v-adv_mAP_v)/clean_mAP_v*100); auxiliary, not mDR",
           "aggregation_contract": contract}
    for mode in MODES:
        maps = {r["victim_id"]: r["adversarial"][mode]["mAP"] for r in reports}
        adv = float(np.mean(list(maps.values())))
        drs = [drop_rate(r["clean"]["mAP"], r["adversarial"][mode]["mAP"]) for r in reports]
        out["modes"][mode] = {"mAP_by_victim": maps, "aAP_adversarial": adv,
            "mDR": drop_rate(clean, adv),
            "mean_per_victim_DR": None if any(d is None for d in drs) else float(np.mean(drs))}
    return out
