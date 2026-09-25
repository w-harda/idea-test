"""Select query-specific shared attributes from Stage 03 discrimination scores."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .attribute_scoring import SUPPORTED_SLOTS


SLOT_ORDER = {slot: index for index, slot in enumerate(SUPPORTED_SLOTS)}


def select_attributes(
    shared_attributes: Mapping[str, str],
    scores: Mapping[str, int],
    beta: float = 1.0,
) -> dict[str, object]:
    """Compute weights, entropy, effective count, and ordered Dynamic Top-K."""
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise ValueError("beta must satisfy 0 < beta <= 1")
    beta = float(beta)
    if not math.isfinite(beta) or not 0 < beta <= 1:
        raise ValueError("beta must satisfy 0 < beta <= 1")
    if not isinstance(shared_attributes, Mapping) or not isinstance(scores, Mapping):
        raise ValueError("shared_attributes and scores must be objects")
    if any(
        slot not in SLOT_ORDER or not isinstance(value, str) or value == "null"
        for slot, value in shared_attributes.items()
    ):
        raise ValueError("shared_attributes contains an invalid slot or value")
    if set(scores) != set(shared_attributes) or any(
        not isinstance(score, int) or isinstance(score, bool) or score < 0
        for score in scores.values()
    ):
        raise ValueError("scores must match shared_attributes with nonnegative integers")

    slots = tuple(slot for slot in SUPPORTED_SLOTS if slot in shared_attributes)
    total = sum(scores.values())
    if total == 0:
        weights = {slot: 0.0 for slot in slots}
        entropy = 0.0
        n_eff = 0.0
        k_star = 0
        selected_slots: list[str] = []
    else:
        weights = {slot: scores[slot] / total for slot in slots}
        entropy = -math.fsum(
            weight * math.log(weight) for weight in weights.values() if weight > 0
        )
        n_eff = math.exp(entropy)
        k_star = min(len(slots), math.ceil(beta * n_eff))
        selected_slots = sorted(
            slots, key=lambda slot: (-scores[slot], SLOT_ORDER[slot])
        )[:k_star]

    return {
        "beta": beta,
        "weights": weights,
        "entropy": entropy,
        "n_eff": n_eff,
        "k_star": k_star,
        "selected_slots": selected_slots,
        "selected_attributes": {
            slot: shared_attributes[slot] for slot in selected_slots
        },
    }
