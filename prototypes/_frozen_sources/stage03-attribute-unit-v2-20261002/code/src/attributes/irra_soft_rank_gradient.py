"""Stage03-v2候选：paired soft-rank的Gradient×Input，以provenance聚合属性。

完整caption仅用于本阶段估计/审计；Top-K后图像侧仍须attribute-only guidance。
输出是结构化属性键下的、原文token证据上的局部归因，不保证共享证据可独立化。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch

from .irra_paired_rank_drop import _normalized_features, paired_image_ranks
from .irra_token_alignment import align_irra_tokens, map_attribute_tokens
from .text_extractor import extract_with_provenance


def paired_soft_rank(similarity: torch.Tensor, paired_index: int, tau: float) -> torch.Tensor:
    """1+sum(j!=paired)sigmoid((s_j-s_plus)/tau)，包含同PID的其他图片。

    若将paired自身也纳入用户公式的求和，只多常数0.5，梯度完全相同。
    """
    if (similarity.ndim != 1 or not similarity.is_floating_point() or similarity.numel() < 1
            or not torch.isfinite(similarity).all()):
        raise ValueError("similarity must be a finite float [G] tensor")
    if (isinstance(paired_index, bool) or not isinstance(paired_index, int)
            or not 0 <= paired_index < similarity.numel()):
        raise ValueError("paired index out of range")
    if isinstance(tau, bool) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    competitors = torch.arange(similarity.numel(), device=similarity.device) != paired_index
    return 1 + torch.sigmoid((similarity[competitors] - similarity[paired_index]) / tau).sum()


def gradient_times_input(rank: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """负号与移除方向一致：e->(1-epsilon)e，rank差的一阶项为epsilon*(-grad·e)。"""
    gradient, = torch.autograd.grad(rank, embedding, create_graph=False, retain_graph=False)
    scores = -(gradient.float() * embedding.detach().float()).sum(dim=-1)
    if not torch.isfinite(gradient).all() or not torch.isfinite(scores).all():
        raise RuntimeError("nonfinite embedding gradient or attribution")
    return scores, gradient


def within_caption_spearman(gradient_scores: Sequence[float], rank_scores: Sequence[float]) -> dict:
    """平均秩处理ties；常量/不足两个样本返回未定义，不填0。"""
    if len(gradient_scores) != len(rank_scores):
        raise ValueError("correlation score lengths differ")
    n = len(rank_scores)
    result = {"n": n, "rho": None, "undefined_reason": None,
              "gradient_distinct": len(set(gradient_scores)), "rank_distinct": len(set(rank_scores))}
    if any(not math.isfinite(value) for value in [*gradient_scores, *rank_scores]):
        raise ValueError("correlation requires finite scores")
    if n < 2:
        result["undefined_reason"] = "fewer_than_two_attributes"
        return result
    if len(set(rank_scores)) == 1:
        result["undefined_reason"] = "constant_rank_scores"
        return result
    if len(set(gradient_scores)) == 1:
        result["undefined_reason"] = "constant_gradient_scores"
        return result

    def ranks(values):
        order = sorted(range(n), key=lambda index: values[index])
        output = [0.] * n
        start = 0
        while start < n:
            end = start + 1
            while end < n and values[order[end]] == values[order[start]]:
                end += 1
            average = (start + 1 + end) / 2
            for index in order[start:end]:
                output[index] = average
            start = end
        tensor = torch.tensor(output, dtype=torch.float64)
        return tensor - tensor.mean()

    left, right = ranks(gradient_scores), ranks(rank_scores)
    result["rho"] = float((left @ right / (left.norm() * right.norm())).item())
    return result


class IRRAPairedSoftRankGradientScorer:
    """hook固定模型的词嵌入输出，完整原文单条编码；不修改权重、文本或图库。"""

    def __init__(self, victim, gallery_paths: Sequence[str], gallery_features: torch.Tensor):
        self.victim = victim
        self.paths = tuple(gallery_paths)
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("gallery paths must be nonempty and unique")
        _normalized_features(gallery_features, len(self.paths), "gallery")
        self.gallery = gallery_features.detach().clone()
        self.indices = {path: index for index, path in enumerate(self.paths)}
        self.embedding_layer = victim.model.base_model.token_embedding
        self._assert_frozen()

    def _assert_frozen(self):
        if self.victim.training or any(p.requires_grad or p.grad is not None for p in self.victim.parameters()):
            raise ValueError("victim must be frozen/eval with no parameter gradients")

    def _encode(self, caption: str, expected_ids: Sequence[int], scales: Mapping[int, float] | None = None):
        captured = []

        def hook(_module, inputs, output):
            actual = inputs[0].detach().cpu().tolist()
            if actual != [list(expected_ids)] or len(captured) != 0:
                raise RuntimeError("official token IDs differ or embedding hook called repeatedly")
            leaf = output.detach().clone()
            if scales:
                for index, scale in scales.items():
                    leaf[:, index] *= scale
            leaf.requires_grad_(True)
            captured.append(leaf)
            return leaf

        handle = self.embedding_layer.register_forward_hook(hook)
        try:
            features = self.victim.encode_texts([caption])
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("embedding hook must capture one text branch forward")
        _normalized_features(features, 1, "text")
        if features.device != self.gallery.device or features.shape[1] != self.gallery.shape[1]:
            raise ValueError("text/gallery feature mismatch")
        return features, captured[0]

    @torch.enable_grad()
    def score(self, caption: str, paired_image: str, attributes: Mapping[str, str],
              provenance: Mapping[str, object], *, tau: float) -> dict:
        self._assert_frozen()
        if paired_image not in self.indices:
            raise ValueError("paired image absent from fixed gallery")
        if extract_with_provenance(caption) != {"attributes": dict(attributes), "provenance": dict(provenance)}:
            raise ValueError("attributes/provenance differ from frozen Stage01 mapping")
        alignment = align_irra_tokens(caption, self.victim.tokenizer, self.victim.text_length)
        mapping = map_attribute_tokens(caption, attributes, provenance, alignment)
        with torch.no_grad():
            reference = self.victim.encode_texts([caption])
        features, embedding = self._encode(caption, alignment["token_ids"])
        difference = float((features.detach() - reference).abs().max().item())
        if not torch.equal(features.detach(), reference):
            raise RuntimeError(f"embedding hook changed official baseline features: {difference}")
        paired = self.indices[paired_image]
        similarity = (features @ self.gallery.T)[0]
        rank = paired_soft_rank(similarity, paired, tau)
        token_scores, gradient = gradient_times_input(rank, embedding)
        self._assert_frozen()
        token_values = token_scores[0].detach().cpu().tolist()
        rows = []
        for row in mapping:
            value = (sum(token_values[index] for index in row["token_indices"])
                     if row["mapping_status"] == "valid" else None)
            rows.append({**row, "S_grad": value})
        tokens = [{**row, "token_score": token_values[row["token_index"]]}
                  for row in alignment["content_tokens"]]
        eot = alignment["eot_index"]
        after_eot = gradient[0, eot + 1:]
        return {
            "method": "stage03-v2-paired-soft-rank-gradient-times-input-prototype",
            "caption": caption, "paired_image": paired_image, "paired_gallery_index": paired,
            "gallery_count": len(self.paths), "tau": tau,
            "r0": int(paired_image_ranks(similarity.detach()[None], paired)[0].item()),
            "r_soft": float(rank.item()), "r_soft_with_paired_self": float(rank.item()) + 0.5,
            "soft_rank_policy": "all gallery j except paired index; same-PID other images included",
            "paired_similarity": float(similarity[paired].detach().item()),
            "aggregation": "sum of unique mapped semantic-evidence token indices per slot; signed -grad dot input",
            "attribution_scope": "original-caption token evidence anchored to (slot,canonical); shared tokens not separable",
            "attributes": dict(attributes), "attribute_results": rows,
            "scores_by_attribute": {f"{row['slot']}={row['canonical']}": row["S_grad"]
                                    for row in rows if row["S_grad"] is not None},
            "token_alignment": alignment, "token_attributions": tokens,
            "diagnostics": {
                "baseline_feature_max_abs_difference": difference,
                "embedding_dtype": str(embedding.dtype), "gradient_dtype": str(gradient.dtype),
                "gradient_norm": float(gradient.float().norm().item()),
                "content_tokens_with_nonzero_gradient": sum(bool(gradient[0, row["token_index"]].count_nonzero())
                                                             for row in alignment["content_tokens"]),
                "after_eot_gradient_max_abs": float(after_eot.abs().max().item()) if after_eot.numel() else 0.,
                "parameter_gradients_absent": True,
            },
        }

    @torch.enable_grad()
    def directional_check(self, caption: str, paired_image: str, token_indices: Sequence[int],
                          *, tau: float, epsilon: float = 0.02) -> dict:
        """仅检查局部导数负号/数值，不用于评分，不是Integrated Gradients。"""
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be between 0 and 1")
        alignment = align_irra_tokens(caption, self.victim.tokenizer, self.victim.text_length)
        content = {row["token_index"] for row in alignment["content_tokens"]}
        selected = sorted(set(token_indices))
        if not selected or not set(selected) <= content:
            raise ValueError("directional check needs content token indices")
        paired = self.indices[paired_image]
        features, embedding = self._encode(caption, alignment["token_ids"])
        rank = paired_soft_rank((features @ self.gallery.T)[0], paired, tau)
        token_scores, _ = gradient_times_input(rank, embedding)
        analytic = float(token_scores[0, selected].sum().item())
        values = []
        for scale in (1 - epsilon, 1 + epsilon):
            features, _ = self._encode(caption, alignment["token_ids"], {index: scale for index in selected})
            values.append(float(paired_soft_rank((features @ self.gallery.T)[0], paired, tau).item()))
        numeric = (values[0] - values[1]) / (2 * epsilon)
        self._assert_frozen()
        return {"token_indices": selected, "epsilon": epsilon, "analytic_negative_grad_dot_input": analytic,
                "central_difference_removal_direction": numeric,
                "relative_difference": abs(analytic-numeric) / max(abs(analytic), abs(numeric), 1e-12),
                "scope": "local directional derivative only; official fp16 forward may quantize finite difference"}
