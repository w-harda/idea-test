"""独立 contextual support 敏感性验证，不接入任何正式 scorer/pipeline。"""
from __future__ import annotations

from copy import deepcopy
import math

import torch

from .irra_token_alignment import align_irra_tokens, map_attribute_tokens, _span_token_indices
from .irra_soft_rank_gradient import paired_soft_rank
from .text_extractor import extract_with_provenance


def contextual_supports(caption, traced, alignment):
    """已记录最小 evidence 与 linked_object 的二值并集；不推断缺失对象。"""
    rows = deepcopy(map_attribute_tokens(caption, **traced, alignment=alignment))
    for row in rows:
        evidence = list(row['token_indices'])
        objects = set()
        reasons = list(row['mapping_reasons'])
        for span in row['linked_objects']:
            indices, errors = _span_token_indices(caption, span, alignment['all_content_tokens'])
            objects.update(indices)
            reasons.extend(errors)
        row['evidence_token_indices'] = evidence
        row['object_token_indices'] = sorted(objects)
        row['support_token_indices'] = sorted(set(evidence) | objects) if not reasons else []
        row['support_status'] = 'valid' if row['support_token_indices'] else 'unresolved'
        row['support_reasons'] = sorted(set(reasons))
        row['support_scope'] = 'recorded evidence union recorded linked_object; binary mask'
        if row['slot'] == 'hair_length' and not objects:
            row['warnings'].append('hair_object_not_recorded_support_is_evidence_only')
    pairs = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left['support_status'] != 'valid' or right['support_status'] != 'valid':
                continue
            a, b = set(left['support_token_indices']), set(right['support_token_indices'])
            pairs.append({'left': left['slot'], 'right': right['slot'],
                          'identical_support': a == b, 'overlap': sorted(a & b),
                          'shared_evidence': sorted(set(left['evidence_token_indices']) &
                                                    set(right['evidence_token_indices']))})
    return rows, pairs


def projected_gradient(gradient, indices, epsilon):
    """S=||M G||F，delta=epsilon*M G/S；没有方向/长度/槽位权重。"""
    if gradient.ndim != 3 or gradient.shape[1] != 1 or not torch.isfinite(gradient).all():
        raise ValueError('gradient must be finite [L,1,D]')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    indices = sorted(set(indices))
    if not indices or min(indices) < 0 or max(indices) >= gradient.shape[0]:
        raise ValueError('support indices out of range or empty')
    masked = torch.zeros_like(gradient, dtype=torch.float32)
    masked[indices] = gradient[indices].float()
    score = float(masked.double().norm().item())
    if score == 0:
        return score, None
    delta = masked * (epsilon / score)
    return score, delta


class AttributeUnitProbe:
    def __init__(self, victim, gallery, paired_index, tau=.02):
        self.victim = victim
        self.gallery = gallery.detach()
        self.paired = paired_index
        self.tau = tau
        self.assert_frozen()

    def assert_frozen(self):
        if self.victim.training or any(p.requires_grad or p.grad is not None
                                       for p in self.victim.parameters()):
            raise RuntimeError('IRRA parameters must be frozen/eval and have no gradients')

    def encode(self, caption, layer, delta=None):
        blocks = self.victim.model.base_model.transformer.resblocks
        if not 1 <= layer <= len(blocks):
            raise ValueError('block layer out of range')
        captured = []

        def hook(_module, _inputs, output):
            if captured or output.ndim != 3 or output.shape[1] != 1:
                raise RuntimeError('one singleton text block output required')
            original = output.detach()
            changed = original.clone() if delta is None else (original.float() + delta).to(original.dtype)
            if changed.shape != original.shape:
                raise ValueError('delta shape differs from hidden states')
            # effective_delta 是进入下一层的实际改变量，包含cast/add的数值误差。
            effective = changed.double() - original.double()
            leaf = changed.detach().requires_grad_(True)
            captured.append((leaf, effective))
            return leaf

        handle = blocks[layer - 1].register_forward_hook(hook)
        try:
            features = self.victim.encode_texts([caption])
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('text block hook not executed exactly once')
        return features, *captured[0]

    def objective(self, features):
        similarities = (features @ self.gallery.T)[0]
        return paired_soft_rank(similarities, self.paired, self.tau), similarities[self.paired]

    @torch.enable_grad()
    def run(self, caption, layer=8, epsilons=(.01, .05, .1)):
        self.assert_frozen()
        traced = extract_with_provenance(caption)
        alignment = align_irra_tokens(caption, self.victim.tokenizer, self.victim.text_length)
        supports, pairs = contextual_supports(caption, traced, alignment)
        with torch.no_grad():
            reference = self.victim.encode_texts([caption])
        feature, hidden, zero_effective = self.encode(caption, layer)
        if not torch.equal(reference, feature.detach()) or zero_effective.count_nonzero():
            raise RuntimeError('zero intervention changes original encoder')
        with torch.no_grad():
            explicit_zero, _, zero_effective = self.encode(caption, layer, torch.zeros_like(hidden, dtype=torch.float32))
        if not torch.equal(reference, explicit_zero) or zero_effective.count_nonzero():
            raise RuntimeError('explicit zero delta changes original encoder')
        value, cosine = self.objective(feature)
        gradient, = torch.autograd.grad(value, hidden)
        baseline = float(value.detach().item())
        cosine0 = float(cosine.detach().item())
        g = gradient.detach().double()
        token_lookup = {r['token_index']: r for r in alignment['content_tokens']}
        for row in supports:
            row['support_tokens'] = [token_lookup[i] for i in row['support_token_indices']]
            row['S'] = None
            row['interventions'] = []
            if row['support_status'] != 'valid':
                continue
            for epsilon in epsilons:
                score, delta = projected_gradient(gradient, row['support_token_indices'], epsilon)
                row['S'] = score
                if delta is None:
                    row['support_status'] = 'zero_gradient'
                    break
                with torch.no_grad():
                    plus_feat, _, plus_effective = self.encode(caption, layer, delta)
                    minus_feat, _, minus_effective = self.encode(caption, layer, -delta)
                    plus_f, plus_cos = self.objective(plus_feat)
                    minus_f, minus_cos = self.objective(minus_feat)
                predicted = epsilon * score
                actual = float(plus_f.item()) - baseline
                actual_minus = float(minus_f.item()) - baseline
                effective_norm = float(plus_effective.norm().item())
                effective_prediction = float((g * plus_effective).sum().item())
                outside = plus_effective.clone()
                outside[row['support_token_indices']] = 0
                if outside.count_nonzero():
                    raise RuntimeError('effective delta escaped support')
                row['interventions'].append({
                    'epsilon': epsilon, 'constructed_delta_norm': float(delta.double().norm().item()),
                    'effective_delta_norm': effective_norm,
                    'effective_budget_relative_error': abs(effective_norm - epsilon) / epsilon,
                    'minus_effective_delta_norm': float(minus_effective.norm().item()),
                    'F_before': baseline, 'F_after': float(plus_f.item()),
                    'F_change': actual, 'F_minus_change': actual_minus,
                    'first_order_prediction': predicted,
                    'first_order_effective_prediction': effective_prediction,
                    'relative_prediction_error': abs(actual - predicted) / predicted,
                    'central_change': (actual - actual_minus) / 2,
                    'central_relative_error': abs((actual - actual_minus) / 2 - predicted) / predicted,
                    'paired_cosine_before': cosine0, 'paired_cosine_after': float(plus_cos.item()),
                    'paired_cosine_minus': float(minus_cos.item()),
                    'direction': 'normalized masked gradient; no manual direction',
                })
        self.assert_frozen()
        # 所有masked方向均仅来自同一次baseline的G；每次从完整原文独立干预。
        return {'caption': caption, 'layer': layer, 'tau': self.tau,
                'F': baseline, 'paired_cosine': cosine0, 'attributes': traced['attributes'],
                'provenance': traced['provenance'], 'token_alignment': alignment,
                'attribute_results': supports, 'support_pairs': pairs,
                'diagnostics': {'zero_intervention_exact': True, 'zero_feature_max_difference': 0.,
                    'parameters_frozen': True, 'parameter_gradients_absent': True,
                    'hidden_dtype': str(hidden.dtype), 'gradient_dtype': str(gradient.dtype),
                    'gradient_norm': float(g.norm().item()),
                    'after_eot_gradient_max_abs': float(g[alignment['eot_index']+1:].abs().max().item())
                        if alignment['eot_index'] + 1 < g.shape[0] else 0.}}
