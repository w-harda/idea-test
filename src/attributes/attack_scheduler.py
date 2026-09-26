"""Schedule one Image -> Text round per ordered Stage 04 attribute."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from collections.abc import Mapping, Sequence

from .attribute_scoring import SUPPORTED_SLOTS
from .text_extractor import extract_with_provenance


@dataclass(frozen=True)
class SelectedAttribute:
    slot: str
    value: str
    score: int


@dataclass(frozen=True)
class AttackState:
    image: Any
    text: str
    targets: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ImageAttackRequest:
    row_id: str
    round_index: int  # 1-based
    attribute: SelectedAttribute
    state: AttackState


@dataclass(frozen=True)
class ImageAttackResult:
    image: Any
    guidance: Any = None


@dataclass(frozen=True)
class TextAttackRequest:
    row_id: str
    round_index: int  # 1-based
    attribute: SelectedAttribute
    state: AttackState  # contains the image produced in this round
    target: dict[str, Any]  # current M(attribute), indexed into state.text
    image_result: ImageAttackResult


@dataclass(frozen=True)
class TextAttackResult:
    text: str
    targets: Mapping[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class AttackResult:
    state: AttackState
    completed: tuple[SelectedAttribute, ...]


class ImageAttack(Protocol):
    def __call__(self, request: ImageAttackRequest) -> ImageAttackResult: ...


class TextAttack(Protocol):
    def __call__(self, request: TextAttackRequest) -> TextAttackResult: ...


def plan_attack(record: Mapping[str, Any]) -> tuple[SelectedAttribute, ...]:
    """Read A* in stored order without scoring or sorting again."""
    if not isinstance(record, Mapping):
        raise ValueError("Stage 04 record must be an object")
    slots = record.get("selected_slots")
    selected = record.get("selected_attributes")
    shared = record.get("shared_attributes")
    attributes = record.get("attributes")
    scores = record.get("scores")
    k_star = record.get("k_star")
    if (
        not isinstance(record.get("row_id"), str)
        or not isinstance(record.get("caption"), str)
        or not isinstance(slots, Sequence)
        or isinstance(slots, (str, bytes))
        or not all(isinstance(value, Mapping) for value in
                   (selected, shared, attributes, scores))
        or not isinstance(k_star, int)
        or isinstance(k_star, bool)
        or k_star < 0
        or len(slots) != k_star
        or any(not isinstance(slot, str) for slot in slots)
        or len(set(slots)) != len(slots)
        or set(selected) != set(slots)
    ):
        raise ValueError("invalid Stage 04 attack selection")

    plan = []
    for slot in slots:
        value = selected[slot]
        score = scores.get(slot)
        if (
            slot not in SUPPORTED_SLOTS
            or not isinstance(value, str)
            or not value
            or value == "null"
            or shared.get(slot) != value
            or attributes.get(slot) != value
            or not isinstance(score, int)
            or isinstance(score, bool)
            or score < 0
        ):
            raise ValueError(f"invalid selected attribute: {slot}")
        plan.append(SelectedAttribute(slot, value, score))
    return tuple(plan)


def _validate_span(text: str, span: Any) -> None:
    if not isinstance(span, Mapping):
        raise ValueError("attribute target span must be an object")
    start, end, raw = span.get("start"), span.get("end"), span.get("raw")
    if (
        not isinstance(start, int) or isinstance(start, bool)
        or not isinstance(end, int) or isinstance(end, bool)
        or not isinstance(raw, str) or not raw
        or not 0 <= start < end <= len(text)
        or text[start:end] != raw
    ):
        raise ValueError("attribute target span does not match current text")


def _validate_targets(
    text: str, targets: Mapping[str, Any], remaining: Sequence[SelectedAttribute]
) -> None:
    if not isinstance(targets, Mapping):
        raise ValueError("attribute targets must be an object")
    for attribute in remaining:
        target = targets.get(attribute.slot)
        if not isinstance(target, Mapping) or target.get("canonical") != attribute.value:
            raise ValueError(f"missing current target for {attribute.slot}")
        mentions = target.get("mentions")
        if not isinstance(mentions, list) or not mentions:
            raise ValueError(f"missing mentions for {attribute.slot}")
        for mention in mentions:
            _validate_span(text, mention)
            if "linked_object" in mention:
                _validate_span(text, mention["linked_object"])


def run_attack(
    record: Mapping[str, Any],
    *,
    image: Any,
    image_attack: ImageAttack,
    text_attack: TextAttack,
) -> AttackResult:
    """Run every selected attribute in A* order with fixed Image -> Text calls.

    TextAttackResult.targets may omit a new mapping if the old remaining spans
    still match the returned text. When text edits shift spans, the text callback
    must return updated targets for every remaining attribute.
    """
    plan = plan_attack(record)
    text = record["caption"]
    if not plan:
        return AttackResult(AttackState(image, text, {}), ())

    traced = extract_with_provenance(text)
    if traced["attributes"] != record["attributes"]:
        raise ValueError("Stage 04 attributes differ from Stage 01 extraction")
    targets = {
        attribute.slot: deepcopy(traced["provenance"][attribute.slot])
        for attribute in plan
    }
    _validate_targets(text, targets, plan)
    state = AttackState(image, text, targets)
    row_id = record["row_id"]

    for round_index, attribute in enumerate(plan, 1):
        image_result = image_attack(
            ImageAttackRequest(row_id, round_index, attribute, state)
        )
        if not isinstance(image_result, ImageAttackResult):
            raise TypeError("image_attack must return ImageAttackResult")
        after_image = AttackState(image_result.image, state.text, state.targets)
        text_result = text_attack(
            TextAttackRequest(
                row_id, round_index, attribute, after_image,
                deepcopy(after_image.targets[attribute.slot]), image_result,
            )
        )
        if not isinstance(text_result, TextAttackResult):
            raise TypeError("text_attack must return TextAttackResult")
        if not isinstance(text_result.text, str):
            raise ValueError("text_attack must return text as a string")
        remaining = plan[round_index:]
        candidate_targets = (
            state.targets if text_result.targets is None else text_result.targets
        )
        _validate_targets(text_result.text, candidate_targets, remaining)
        state = AttackState(
            image_result.image,
            text_result.text,
            {
                item.slot: deepcopy(candidate_targets[item.slot])
                for item in remaining
            },
        )
    return AttackResult(state, plan)
