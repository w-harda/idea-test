"""Query-specific matching and attribute discrimination for canonical TBPS attributes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .visual_canonicalizer import SLOTS


SUPPORTED_SLOTS = tuple(slot for slot in SLOTS if slot != "upper_clothing_type")


def _attributes(value: object, label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} attributes must be an object")
    if set(value) != set(SLOTS) or any(
        not isinstance(value[slot], str) for slot in SLOTS
    ):
        raise ValueError(f"{label} attributes must contain exactly 13 string slots")
    return value


class AttributeGallery:
    """Index one image per path using canonical slot/value bit masks."""

    def __init__(self, records: Iterable[Mapping[str, object]]):
        self.value_masks: dict[str, dict[str, int]] = {
            slot: {} for slot in SUPPORTED_SLOTS
        }
        self.null_masks: dict[str, int] = dict.fromkeys(SUPPORTED_SLOTS, 0)
        seen: set[str] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"gallery record {index} must be an object")
            image = record.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError(f"gallery record {index} needs an image path")
            if image in seen:
                raise ValueError(f"duplicate gallery image: {image}")
            seen.add(image)
            attributes = _attributes(record.get("attributes"), f"gallery record {index}")
            bit = 1 << index
            for slot in SUPPORTED_SLOTS:
                value = attributes[slot]
                if value == "null":
                    self.null_masks[slot] |= bit
                else:
                    masks = self.value_masks[slot]
                    masks[value] = masks.get(value, 0) | bit
        self.size = len(seen)
        self.all_mask = (1 << self.size) - 1

    def score(self, text_attributes: Mapping[str, str]) -> dict[str, object]:
        """Return S(a) = |C(A without a) minus C(A)| for one query."""
        attributes = _attributes(text_attributes, "query")
        shared = {
            slot: attributes[slot]
            for slot in SUPPORTED_SLOTS
            if attributes[slot] != "null"
        }
        slots = tuple(shared)
        # Exact null-count masks keep G_valid fixed for every deletion.
        exact = [self.all_mask] + [0] * len(slots)
        for processed, slot in enumerate(slots):
            unknown = self.null_masks[slot]
            known = self.all_mask ^ unknown
            for count in range(processed + 1, 0, -1):
                exact[count] = (exact[count] & known) | (exact[count - 1] & unknown)
            exact[0] &= known
        valid = 0
        for mask in exact[: len(slots) // 2 + 1]:
            valid |= mask

        matches = [self.value_masks[slot].get(shared[slot], 0) for slot in slots]
        conflicts = [
            self.all_mask ^ (matches[index] | self.null_masks[slot])
            for index, slot in enumerate(slots)
        ]
        all_conflicts = 0
        all_matches = 0
        for match, conflict in zip(matches, conflicts, strict=True):
            all_matches |= match
            all_conflicts |= conflict
        baseline = valid & ~all_conflicts & all_matches
        scores: dict[str, int] = {}
        for index, slot in enumerate(slots):
            other_conflicts = 0
            other_matches = 0
            for other_index in range(len(slots)):
                if other_index != index:
                    other_conflicts |= conflicts[other_index]
                    other_matches |= matches[other_index]
            without = valid & ~other_conflicts & other_matches
            scores[slot] = (without & ~baseline).bit_count()
        valid_count = valid.bit_count()
        return {
            "shared_attributes": shared,
            "gallery_count": self.size,
            "valid_gallery_count": valid_count,
            "excluded_gallery_count": self.size - valid_count,
            "candidate_count": baseline.bit_count(),
            "scores": scores,
        }
