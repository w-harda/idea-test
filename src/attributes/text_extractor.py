"""与数据集格式无关的、基于显式短语的文本属性提取器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parents[2]
_ONTOLOGY = _ROOT / "ontology" / "upar_attribute_space.yaml"
_ALIASES = _ROOT / "ontology" / "alias_map.yaml"
_CLAUSE_SPLIT = re.compile(r"[,;.]|\b(?:wearing|wears|dressed in|with|without|under|over|beneath|underneath|in|but|that|which|who|whose)\b", re.I)
_AND = re.compile(r"\band\b", re.I)
_NEGATIVE = re.compile(r"\b(?:without|no|not|never|isn't|doesn't)\b", re.I)
_HAIR_LENGTH = re.compile(r"\b(short|long|shoulder[\s-]+length|medium(?:[\s-]+length)?)\s+(?:\w+\s+){0,2}hair\b", re.I)
_CARRY = re.compile(r"\b(?:carrying|carry|carries|carried|holding|holds|held)\b", re.I)
_WEAR = re.compile(r"\b(?:wearing|wears|wear|worn)\b", re.I)
_ACTION_BOUNDARY = re.compile(r"[,;.]|\b(?:but|with|in|under|over|beneath|underneath|that|which|who|whose)\b", re.I)
_POSTPOSED_CARRY = re.compile(r"^\s+(?:held|carried)\s+(?:over|in|on|under|beneath|by|across|around)\b", re.I)
_HAS_ON = re.compile(r"\b(?:has|have|had)\s+on\b", re.I)
_HAS = re.compile(r"\b(?:has|have|had)\b", re.I)
_ON_AFTER_GARMENT = re.compile(r"^\s+on\b", re.I)
_DRESS_COMPOUND = re.compile(r"\bdress[\s-]+(shirts?|shoes?|pants?)\b", re.I)
_SUIT_LOWER = re.compile(r"suit[\s-]+(?:pants|trousers?)\b", re.I)
_GARMENT_PART = re.compile(
    r"\b(?:sleeves?|hood|fur|collar|cuffs?|hat|cap|lining|zip|zipper|pockets?)"
    r"\s+(?:of|on)\s+(?:(?:his|her|the|its|a|an|my|your|their|our)\s+)?"
    r"(?P<modifiers>(?:[a-z-]+\s+){0,2})$",
    re.I,
)
_LEADING_PRONOUN = re.compile(r"^\s*(she|he)\b", re.I)


@dataclass(frozen=True)
class _Match:
    start: int
    end: int
    value: str


@dataclass(frozen=True)
class _Garment:
    side: str
    kind: str | None
    rank: int
    colors: frozenset[str]
    lengths: frozenset[str]
    object_match: _Match
    color_matches: tuple[_Match, ...]
    length_matches: tuple[_Match, ...]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是映射: {path}")
    return data


def _pattern(phrase: str) -> re.Pattern[str]:
    words = re.split(r"[\s-]+", phrase.strip())
    if not words or not all(words):
        raise ValueError(f"空 alias: {phrase!r}")
    return re.compile(r"(?<!\w)" + r"[\s-]+".join(map(re.escape, words)) + r"(?!\w)", re.I)


class _Matcher:
    def __init__(self, aliases: dict[str, list[str]]):
        self.patterns = [
            (_pattern(phrase), canonical)
            for canonical, phrases in aliases.items()
            for phrase in phrases
        ]

    def find(self, text: str) -> list[_Match]:
        found = [
            _Match(match.start(), match.end(), canonical)
            for pattern, canonical in self.patterns
            for match in pattern.finditer(text)
        ]
        # 长短 alias 重叠时优先完整短语，例如 t-shirt 优先于 shirt。
        found.sort(key=lambda item: (item.start, -(item.end - item.start)))
        selected: list[_Match] = []
        for item in found:
            if not selected or item.start >= selected[-1].end:
                selected.append(item)
        return selected


def _single(values: set[str] | frozenset[str]) -> str:
    return next(iter(values)) if len(values) == 1 else "null"


def _located_matches(
    matches: list[_Match], pieces: list[tuple[str, int | None]], matcher: _Matcher
) -> tuple[_Match, ...]:
    """还原拼接文本中的原文位置；跨边界命中时改用真实片段中的同值 alias。"""
    segments = []
    cursor = 0
    for content, source_start in pieces:
        segments.append((cursor, cursor + len(content), source_start))
        cursor += len(content)
    located = [
        _Match(source_start + match.start - left, source_start + match.end - left, match.value)
        for match in matches
        for left, right, source_start in segments
        if source_start is not None and left <= match.start and match.end <= right
    ]
    for canonical in {match.value for match in matches} - {match.value for match in located}:
        located.extend(
            _Match(source_start + item.start, source_start + item.end, item.value)
            for content, source_start in pieces if source_start is not None
            for item in matcher.find(content) if item.value == canonical
        )
    return tuple(located)


def _source_span(caption: str, source_index: list[int], match: _Match) -> dict[str, Any]:
    """lower() 可能扩展 Unicode 字符，使用逐字符索引返回原文位置。"""
    start = source_index[match.start]
    end = source_index[match.end - 1] + 1
    return {"raw": caption[start:end], "start": start, "end": end}


class TextAttributeExtractor:
    """将 caption 映射到 ontology 中固定的 13 个槽位。"""

    def __init__(self, ontology_path: str | Path = _ONTOLOGY, alias_path: str | Path = _ALIASES):
        ontology = _read_yaml(Path(ontology_path))
        slots = ontology.get("slots")
        if not isinstance(slots, list) or len(slots) != ontology.get("slot_count"):
            raise ValueError("ontology slots 与 slot_count 不一致")
        self.allowed = {
            slot["key"]: {value["key"] for value in slot["values"]}
            for slot in slots
        }
        if len(self.allowed) != len(slots) or any("null" not in values for values in self.allowed.values()):
            raise ValueError("ontology slot 重复或缺少 null")
        self.slot_order = tuple(self.allowed)
        aliases = _read_yaml(Path(alias_path))
        for slot in ("age", "gender", "hair_length", "upper_clothing_type", "lower_clothing_type"):
            self._check_aliases(aliases[slot], slot)
        for slot in ("upper_clothing_color", "lower_clothing_color"):
            self._check_aliases(aliases["colors"], slot)
        for slot in ("upper_clothing_length", "lower_clothing_length"):
            self._check_aliases(aliases["lengths"], slot)
        for slot in ("backpack", "bag", "hat"):
            if slot not in self.allowed or not isinstance(aliases["accessories"][slot], list):
                raise ValueError(f"非法配饰 alias: {slot}")
        self._check_aliases(aliases["accessories"]["glasses"], "glasses")

        self.direct = {slot: _Matcher(aliases[slot]) for slot in ("age", "gender", "hair_length")}
        self.colors = _Matcher(aliases["colors"])
        self.lengths = _Matcher(aliases["lengths"])
        self.upper = _Matcher(aliases["upper_clothing_type"])
        self.lower = _Matcher(aliases["lower_clothing_type"])
        self.generic_upper = _Matcher({"generic": aliases["generic_garments"]["upper"]})
        self.generic_lower = _Matcher({"generic": aliases["generic_garments"]["lower"]})
        self.accessories = {
            slot: _Matcher({"yes": aliases["accessories"][slot]})
            for slot in ("backpack", "bag", "hat")
        }
        self.accessories["glasses"] = _Matcher(aliases["accessories"]["glasses"])

    def _check_aliases(self, mapping: dict[str, list[str]], slot: str) -> None:
        if slot not in self.allowed or not isinstance(mapping, dict):
            raise ValueError(f"非法 alias 映射: {slot}")
        for canonical, phrases in mapping.items():
            if canonical not in self.allowed[slot] or not isinstance(phrases, list) or not phrases:
                raise ValueError(f"{slot} 的 alias 指向 ontology 外的值: {canonical}")
            if not all(isinstance(phrase, str) and phrase.strip() for phrase in phrases):
                raise ValueError(f"{slot} 的 alias 必须是非空字符串")

    def extract(self, caption: str) -> dict[str, str]:
        """仅提取明示属性；未提及、矛盾或无法绑定的槽位返回 null。"""
        return self._extract(caption, with_provenance=False)[0]

    def extract_with_provenance(self, caption: str) -> dict[str, Any]:
        """同时返回属性及其在原始 caption 中的真实匹配位置。"""
        attributes, provenance = self._extract(caption, with_provenance=True)
        return {"attributes": attributes, "provenance": provenance}

    def _extract(
        self, caption: str, with_provenance: bool
    ) -> tuple[dict[str, str], dict[str, Any] | None]:
        if not isinstance(caption, str):
            raise TypeError("caption 必须是 str")
        result = dict.fromkeys(self.slot_order, "null")
        text = caption.lower()
        supporting: dict[str, list[tuple[_Match, _Match | None]]] = {
            slot: [] for slot in self.slot_order
        }

        for slot, matcher in self.direct.items():
            matches = [item for item in matcher.find(text) if not _negated(text, item.start)]
            values = {item.value for item in matches}
            if slot == "hair_length":
                for match in _HAIR_LENGTH.finditer(text):
                    if _negated(text, match.start()):
                        continue
                    canonical = "short" if match.group(1) == "short" else "long"
                    values.add(canonical)
                    if not any(
                        item.value == canonical and item.start <= match.start(1)
                        and match.end(1) <= item.end for item in matches
                    ):
                        matches.append(_Match(match.start(1), match.end(1), canonical))
            if slot == "gender":
                pronoun = _LEADING_PRONOUN.match(text)
                if pronoun:
                    canonical = "female" if pronoun.group(1) == "she" else "male"
                    values.add(canonical)
                    matches.append(_Match(pronoun.start(1), pronoun.end(1), canonical))
            result[slot] = _single(values)
            supporting[slot].extend((item, None) for item in matches)

        garments = [
            garment
            for phrase, offset in self._phrases(text)
            for garment in self._garments(phrase, offset, text)
        ]
        for side in ("upper", "lower"):
            candidates = [garment for garment in garments if garment.side == side]
            if not candidates:
                continue
            outer_rank = max(garment.rank for garment in candidates)
            primary = [garment for garment in candidates if garment.rank == outer_rank]
            type_slot = f"{side}_clothing_type"
            color_slot = f"{side}_clothing_color"
            length_slot = f"{side}_clothing_length"
            result[type_slot] = _single({garment.kind for garment in primary if garment.kind is not None})
            result[color_slot] = _single(set().union(*(garment.colors for garment in primary)))
            result[length_slot] = _single(set().union(*(garment.lengths for garment in primary)))
            for garment in primary:
                if garment.kind is not None:
                    supporting[type_slot].append((garment.object_match, None))
                supporting[color_slot].extend(
                    (match, garment.object_match) for match in garment.color_matches
                )
                supporting[length_slot].extend(
                    (match, garment.object_match) for match in garment.length_matches
                )

        backpack_mentions = self.accessories["backpack"].find(text)
        for slot, matcher in self.accessories.items():
            values = set()
            for item in matcher.find(text):
                if slot == "bag" and any(
                    item.start < backpack.end and backpack.start < item.end
                    for backpack in backpack_mentions
                ):
                    continue
                negative = _negated(text, item.start)
                canonical = ("no_glasses" if slot == "glasses" else "no") if negative else item.value
                values.add(canonical)
                supporting[slot].append((_Match(item.start, item.end, canonical), None))
            result[slot] = _single(values)

        if not with_provenance:
            return result, None

        source_index = [index for index, char in enumerate(caption) for _ in char.lower()]
        provenance = {}
        for slot, canonical in result.items():
            if canonical == "null":
                provenance[slot] = None
                continue
            mentions = []
            seen = set()
            for match, linked in sorted(
                supporting[slot], key=lambda pair: (pair[0].start, pair[0].end)
            ):
                if match.value != canonical:
                    continue
                key = (match.start, match.end, linked.start if linked else None,
                       linked.end if linked else None)
                if key in seen:
                    continue
                seen.add(key)
                mention = _source_span(caption, source_index, match)
                if linked is not None:
                    mention["linked_object"] = _source_span(caption, source_index, linked)
                mentions.append(mention)
            provenance[slot] = {"canonical": canonical, "mentions": mentions}
        return result, provenance

    def _phrases(self, text: str) -> list[tuple[str, int]]:
        phrases = []
        boundaries = [(match.start(), match.end()) for match in _CLAUSE_SPLIT.finditer(text)]
        start = 0
        for end, next_start in boundaries + [(len(text), len(text))]:
            clause = text[start:end]
            part_start = 0
            for match in _AND.finditer(clause):
                if self._has_entity(clause[part_start:match.start()]):
                    phrase = clause[part_start:match.start()]
                    if phrase.strip():
                        phrases.append((phrase, start + part_start))
                    part_start = match.end()
            phrase = clause[part_start:]
            if phrase.strip():
                phrases.append((phrase, start + part_start))
            start = next_start
        return phrases

    def _has_entity(self, phrase: str) -> bool:
        return bool(
            self.upper.find(phrase) or self.lower.find(phrase)
            or self.generic_upper.find(phrase) or self.generic_lower.find(phrase)
            or any(matcher.find(phrase) for matcher in self.accessories.values())
            or re.search(r"\bhair\b", phrase)
        )

    def _garments(self, phrase: str, offset: int, full_text: str) -> list[_Garment]:
        compounds = list(_DRESS_COMPOUND.finditer(phrase))
        matches = [
            (item, side, rank)
            for matcher, side, rank in (
                (self.upper, "upper", 0), (self.lower, "lower", 0),
                (self.generic_upper, "upper", -1), (self.generic_lower, "lower", -1),
            )
            for item in matcher.find(phrase)
            if not any(item.start < compound.end() and compound.start() < item.end
                       for compound in compounds)
            and not (side == "upper" and phrase[item.start:item.end] == "suit"
                     and _SUIT_LOWER.match(phrase, item.start))
        ]
        for compound in compounds:
            noun = compound.group(1)
            if noun.startswith("shirt"):
                matches.append((_Match(compound.start(), compound.end(), "t_shirt_shirt"), "upper", 0))
            elif noun.startswith("pant"):
                matches.append((_Match(compound.start(), compound.end(), "trousers_shorts"), "lower", 0))
        matches.sort(key=lambda entry: (entry[0].start, -(entry[0].end - entry[0].start)))
        garments = []
        previous_end = 0
        for position, (item, side, rank) in enumerate(matches):
            if item.start < previous_end:
                continue
            next_start = next(
                (other.start for other, _, _ in matches[position + 1:] if other.start >= item.end),
                len(phrase),
            )
            tail = phrase[item.end:next_start]
            if (_negated(full_text, offset + item.start)
                    or _carried(full_text, offset + item.start, offset + item.end)
                    or _POSTPOSED_CARRY.match(full_text[offset + item.end:])):
                previous_end = item.end
                continue
            descriptor = phrase[previous_end:item.start]
            color_text = descriptor
            color_pieces: list[tuple[str, int | None]] = [(descriptor, offset + previous_end)]
            # 后置颜色仅在当前衣物与下一件衣物之间查找。
            post_color = re.match(r"\s+(?:in|of|colored|coloured)\s+([\w\s-]+)", tail)
            if post_color:
                color_text += " " + post_color.group(1)
                color_pieces.extend([
                    (" ", None),
                    (post_color.group(1), offset + item.end + post_color.start(1)),
                ])
            copula = re.match(r"\s+(?:is|are|was|were)\b", tail, re.I)
            if copula:
                part = _GARMENT_PART.search(descriptor)
                if part:
                    # 部件的谓词颜色不属于整件衣物；保留衣物名词前的修饰色。
                    color_text = part.group("modifiers")
                    color_pieces = [(color_text, offset + previous_end + part.start("modifiers"))]
                else:
                    color_text += " " + tail[copula.end():]
                    color_pieces.extend([
                        (" ", None),
                        (tail[copula.end():], offset + item.end + copula.end()),
                    ])
            color_matches = self.colors.find(color_text)
            length_matches = self.lengths.find(descriptor)
            colors = frozenset(match.value for match in color_matches)
            lengths = frozenset(match.value for match in length_matches)
            kind = None if item.value == "generic" else item.value
            if side == "upper":
                rank = {"t_shirt_shirt": 0, "hoodie_sweater": 1,
                        "vest_sleeveless": 1, "jacket_coat": 2}.get(kind, rank)
            garments.append(_Garment(
                side, kind, rank, colors, lengths,
                _Match(offset + item.start, offset + item.end, item.value),
                _located_matches(color_matches, color_pieces, self.colors),
                tuple(_Match(offset + previous_end + match.start,
                             offset + previous_end + match.end, match.value)
                      for match in length_matches),
            ))
            previous_end = item.end
        return garments


def _negated(text: str, start: int) -> bool:
    context = re.split(r"[,;.]|\bbut\b", text[:start])[-1]
    recent = " ".join(context.split()[-5:])
    negation = list(_NEGATIVE.finditer(recent))
    if not negation:
        return False
    last = negation[-1]
    # "no bag and a hat" 中，冠词引入新的肯定实体；without 则可辖及并列实体。
    if last.group().lower() != "without" and re.search(
        r"\band\s+(?:a|an|the)\b", recent[last.end():]
    ):
        return False
    return True


def _carried(text: str, start: int, end: int) -> bool:
    """最近的明确动作是 carry/hold 时，该衣物不是穿着实体。"""
    context = _ACTION_BOUNDARY.split(text[:start])[-1]
    carrying = list(_CARRY.finditer(context))
    if not carrying:
        return False
    wearing = [match.start() for match in _WEAR.finditer(context)]
    wearing.extend(match.start() for match in _HAS_ON.finditer(context))
    if _ON_AFTER_GARMENT.match(text[end:]):
        wearing.extend(match.start() for match in _HAS.finditer(context))
    return carrying[-1].start() > max(wearing, default=-1)


@lru_cache(maxsize=1)
def _default_extractor() -> TextAttributeExtractor:
    return TextAttributeExtractor()


def extract(caption: str) -> dict[str, str]:
    """使用项目默认 ontology 与 alias 映射提取固定属性字典。"""
    return _default_extractor().extract(caption)


def extract_with_provenance(caption: str) -> dict[str, Any]:
    """使用默认 ontology 返回固定属性字典和对应原文位置。"""
    return _default_extractor().extract_with_provenance(caption)
