"""将官方 UPAR40 的独立 sigmoid 概率保守映射到固定的 13 个槽位。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


# 顺序来自官方 infer_upar_test_phase.py 的 predictions.csv 表头。
UPAR40_NAMES = (
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short",
    *(f"UpperBody-Color-{color}" for color in (
        "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
        "Purple", "Red", "White", "Yellow", "Other")),
    "LowerBody-Length-Short",
    *(f"LowerBody-Color-{color}" for color in (
        "Black", "Blue", "Brown", "Green", "Grey", "Orange", "Pink",
        "Purple", "Red", "White", "Yellow", "Other")),
    "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal",
    "Accessory-Glasses-Sun", "Accessory-Hat",
)

SLOTS = (
    "age", "gender", "hair_length", "upper_clothing_length",
    "upper_clothing_color", "upper_clothing_type", "lower_clothing_length",
    "lower_clothing_color", "lower_clothing_type", "backpack", "bag",
    "glasses", "hat",
)


class VisualCanonicalizer:
    """独立标签仅在明显胜出时落槽；模糊、冲突和 Other 均为 null。"""

    def __init__(self, positive: float = 0.7, negative: float = 0.3, margin: float = 0.2):
        if not (0 <= negative < 0.5 < positive <= 1 and 0 <= margin <= 1):
            raise ValueError("阈值必须满足 0 <= negative < 0.5 < positive <= 1")
        self.positive = positive
        self.negative = negative
        self.margin = margin

    def _choice(self, scores: Mapping[str, float], *, other: float | None = None) -> str:
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        winner, first = ranked[0]
        second = max((score for _, score in ranked[1:]), default=0.0)
        if other is not None:
            second = max(second, other)
        if first < self.positive or first - second < self.margin:
            return "null"
        return winner

    def _binary(self, probability: float) -> str:
        if probability >= self.positive:
            return "yes"
        if probability <= self.negative:
            return "no"
        return "null"

    def _length(self, probability: float) -> str:
        if probability >= self.positive:
            return "short"
        if probability <= self.negative:
            return "long"
        return "null"

    def canonicalize(self, probabilities: Sequence[float]) -> dict[str, str]:
        if len(probabilities) != 40:
            raise ValueError(f"UPAR40 必须有 40 个概率，收到 {len(probabilities)} 个")
        values = [float(value) for value in probabilities]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("UPAR40 概率必须是 [0, 1] 内的有限数")
        p = dict(zip(UPAR40_NAMES, values, strict=True))
        result = dict.fromkeys(SLOTS, "null")
        result["age"] = self._choice({
            "young": p["Age-Young"], "adult": p["Age-Adult"],
            "elderly": p["Age-Old"],
        })
        female = p["Gender-Female"]
        result["gender"] = "female" if female >= self.positive else (
            "male" if female <= self.negative else "null")
        result["hair_length"] = self._choice({
            "short": p["Hair-Length-Short"],
            "long": p["Hair-Length-Long"],
            "bald": p["Hair-Length-Bald"],
        })
        for side, slot in (("UpperBody", "upper"), ("LowerBody", "lower")):
            result[f"{slot}_clothing_length"] = self._length(p[f"{side}-Length-Short"])
            colors = {
                color.lower(): p[f"{side}-Color-{color}"]
                for color in ("Black", "Blue", "Brown", "Green", "Grey", "Orange",
                              "Pink", "Purple", "Red", "White", "Yellow")
            }
            result[f"{slot}_clothing_color"] = self._choice(
                colors, other=p[f"{side}-Color-Other"])
        result["lower_clothing_type"] = self._choice({
            "trousers_shorts": p["LowerBody-Type-Trousers&Shorts"],
            "skirt_dress": p["LowerBody-Type-Skirt&Dress"],
        })
        for accessory in ("Backpack", "Bag", "Hat"):
            result[accessory.lower()] = self._binary(p[f"Accessory-{accessory}"])
        normal = p["Accessory-Glasses-Normal"]
        sun = p["Accessory-Glasses-Sun"]
        if normal <= self.negative and sun <= self.negative:
            result["glasses"] = "no_glasses"
        else:
            result["glasses"] = self._choice({
                "normal_glasses": normal, "sunglasses": sun,
            })
        return result
