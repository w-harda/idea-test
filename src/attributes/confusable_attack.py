"""Stage 05 文本回调：仅在 M 定位的属性词内替换一个形近字符。"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

import torch

from .attack_scheduler import TextAttackRequest, TextAttackResult

# 限定为同形或近形的单个 Unicode 字符；不使用可增减长度的替换。
CONFUSABLES: dict[str, tuple[str, ...]] = {
    "a": ("а", "ɑ"), "c": ("с",), "d": ("ԁ",), "e": ("е",),
    "g": ("ɡ",), "h": ("һ",), "i": ("і",), "j": ("ј",),
    "l": ("ӏ",), "n": ("ո",), "o": ("о", "ο"),
    "p": ("р",), "s": ("ѕ",), "u": ("υ",),
    "v": ("ν",), "w": ("ԝ",), "x": ("х",), "y": ("у",),
    "A": ("А",), "B": ("В",), "C": ("С",), "E": ("Е",),
    "H": ("Н",), "K": ("К",), "M": ("М",),
    "O": ("О",), "P": ("Р",), "S": ("Ѕ",), "T": ("Т",),
    "X": ("Х",), "Y": ("Ү",),
}


def character_candidates(text: str, target: dict, used_offsets: set[int] | None = None):
    """返回 (绝对 offset, 替换字符, 新文本)，每项只改一个原文位置。"""
    used_offsets = used_offsets or set()
    seen = set()
    for mention in target["mentions"]:
        for offset in range(mention["start"], mention["end"]):
            if offset in used_offsets:
                continue
            for replacement in CONFUSABLES.get(text[offset], ()):
                if len(replacement) != 1 or (offset, replacement) in seen:
                    continue
                seen.add((offset, replacement))
                yield offset, replacement, text[:offset] + replacement + text[offset + 1:]


def _refresh_targets(targets: dict, text: str) -> dict:
    updated = deepcopy(targets)
    for target in updated.values():
        for mention in target["mentions"]:
            mention["raw"] = text[mention["start"]:mention["end"]]
            if "linked_object" in mention:
                obj = mention["linked_object"]
                obj["raw"] = text[obj["start"]:obj["end"]]
    return updated


class ConfusableTextCallback:
    """开发阶段以当前图库 source soft rank 选候选；跳过为合法选项。"""

    def __init__(self, score_candidates: Callable[[torch.Tensor, list[str]], torch.Tensor]):
        self.score_candidates = score_candidates
        self.used_offsets: set[int] = set()
        self.events: list[dict] = []

    def __call__(self, request: TextAttackRequest) -> TextAttackResult:
        text = request.state.text
        options = list(character_candidates(text, request.target, self.used_offsets))
        texts = [text] + [candidate[2] for candidate in options]
        with torch.no_grad():
            scores = self.score_candidates(request.state.image, texts)
        if scores.shape != (len(texts),) or not torch.isfinite(scores).all():
            raise ValueError("candidate scorer must return finite one-dimensional scores")
        # 下标 0 是跳过；同分时保持原文。
        best = int(scores.argmax().item())
        event = {
            "round": request.round_index, "slot": request.attribute.slot,
            "candidate_count": len(options), "selected": best != 0,
            "score_before": float(scores[0]), "score_after": float(scores[best]),
        }
        if best == 0:
            self.events.append(event)
            return TextAttackResult(text)
        offset, replacement, changed = options[best - 1]
        self.used_offsets.add(offset)
        event.update({"offset": offset, "from": text[offset], "to": replacement})
        self.events.append(event)
        return TextAttackResult(changed, _refresh_targets(request.state.targets, changed))
