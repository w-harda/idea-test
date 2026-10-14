"""Stage03-v2：冻结 IRRA 的明确配对图像排名差；不改变历史 v1 scorer。

Downstream contract：完整 caption 仅用于本阶段的属性重要性估计和审计。
Top-K 之后，图像侧攻击/generator 必须使用 attribute-only guidance，条件和文本监督
只能来自选中的 attribute representation；不得额外使用完整 caption、其编码或由其
构造的 image-side supervision。属性表示形式尚未冻结，此模块不实现下游攻击。
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import torch

from .text_extractor import extract_with_provenance


LITERAL_REMOVAL_POLICY = "historical prototype: all provenance mention spans, literal deletion"
REMOVAL_POLICY = "prototype slot ablation: minimal evidence, cross-slot/binding guards, Stage01 validation"
RANK_POLICY = "torch.argsort(dim=1, descending=True), stable=False (PyTorch default), 1-based"


class IRRATextEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor: ...


@dataclass(frozen=True)
class CuhkPairedCaption:
    row_id: str
    annotation_row_index: int
    caption_index: int
    caption: str
    image: str
    person_id: int | str
    paired_gallery_index: int


@dataclass(frozen=True)
class CuhkPairedSplit:
    gallery_paths: tuple[str, ...]
    gallery_ids: tuple[int | str, ...]
    captions: tuple[CuhkPairedCaption, ...]


def load_cuhk_paired_split(annotation: str | Path, *, split: str) -> CuhkPairedSplit:
    """显式选择 split，按 annotation 图像行展开 captions；不按 PID 选择配对图。"""
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val or test")
    rows = json.loads(Path(annotation).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("CUHK annotation must be a JSON array")
    paths, ids, captions = [], [], []
    seen = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"invalid annotation row {row_index}")
        if row.get("split") != split:
            continue
        image, pid, texts = row.get("file_path"), row.get("id"), row.get("captions")
        if not isinstance(image, str) or not image.strip():
            raise ValueError(f"invalid image path at row {row_index}")
        relative = PurePosixPath(image.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in image or not relative.parts:
            raise ValueError(f"invalid image path at row {row_index}")
        image = relative.as_posix()
        if image in seen:
            raise ValueError(f"duplicate gallery image: {image}")
        if (isinstance(pid, bool) or not isinstance(pid, (int, str))
                or not isinstance(texts, list) or not texts
                or any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError(f"invalid caption/PID at row {row_index}")
        index = len(paths)
        seen.add(image)
        paths.append(image)
        ids.append(pid)
        captions.extend(CuhkPairedCaption(
            f"cuhk:{row_index}:{caption_index}", row_index, caption_index,
            text, image, pid, index,
        ) for caption_index, text in enumerate(texts))
    if not paths:
        raise ValueError(f"empty CUHK {split} split")
    return CuhkPairedSplit(tuple(paths), tuple(ids), tuple(captions))


def paired_image_ranks(similarity: torch.Tensor, paired_index: int) -> torch.Tensor:
    """与官方 IRRA full 排序相同；并列值不额外定义稳定顺序或平均排名。"""
    if (not isinstance(similarity, torch.Tensor) or similarity.ndim != 2
            or not similarity.is_floating_point() or 0 in similarity.shape
            or not torch.isfinite(similarity).all()):
        raise ValueError("similarity must be a nonempty finite float [N,G] tensor")
    if (isinstance(paired_index, bool) or not isinstance(paired_index, int)
            or not 0 <= paired_index < similarity.shape[1]):
        raise ValueError("paired gallery index out of range")
    order = torch.argsort(similarity, dim=1, descending=True)
    return (order == paired_index).to(torch.int64).argmax(dim=1) + 1


def remove_provenance_mentions(caption: str, target: Mapping[str, object]) -> dict:
    """历史 literal helper，仅用于诊断；不能据其结果直接生成有效 Stage03-v2 分数。

    linked_object 仅保留作诊断，不扩展删除范围；否定词、冠词和语法均不重写。
    同一属性内重复/重叠 spans 取区间并集；其他属性始终从完整 caption 独立消融。
    """
    if not isinstance(caption, str) or not isinstance(target, Mapping):
        raise ValueError("caption and provenance target are required")
    mentions = target.get("mentions")
    if not isinstance(mentions, list) or not mentions:
        raise ValueError("provenance mentions must be a nonempty list")
    spans = []
    for mention in mentions:
        if not isinstance(mention, Mapping):
            raise ValueError("invalid provenance mention")
        start, end, raw = mention.get("start"), mention.get("end"), mention.get("raw")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end <= len(caption)
                or not isinstance(raw, str) or caption[start:end] != raw):
            raise ValueError("provenance span does not match original caption")
        spans.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    pieces = []
    cursor = 0
    for start, end in merged:
        pieces.extend((caption[cursor:start], " "))
        cursor = end
    pieces.append(caption[cursor:])
    removed = re.sub(r"\s+", " ", "".join(pieces)).strip()
    removed = re.sub(r"\s+([,.;:!?])", r"\1", removed)
    return {
        "removed_text": removed,
        "removed_spans": [{"start": start, "end": end, "raw": caption[start:end]}
                          for start, end in merged],
        "mentions": deepcopy(mentions),
        "removal_policy": LITERAL_REMOVAL_POLICY,
    }


def _minimal_evidence(slot: str, canonical: str, mention: Mapping) -> dict:
    """只分解有限的明确复合形式，其他情况保留原 span 交给保守检查。"""
    patterns = {
        ("age", "young"): r"(?P<cue>young)\s+(?:man|woman|person|male|female)",
        ("age", "elderly"): r"(?P<cue>old|older)\s+(?:man|woman)",
        ("hair_length", "short"): r"(?P<cue>short)\s+hair",
        ("hair_length", "long"): r"(?P<cue>long|shoulder[\s-]+length|medium(?:[\s-]+length)?)\s+hair",
        ("hair_length", "bald"): r"(?P<cue>bald)\s+head",
        ("upper_clothing_length", "short"): r"(?P<cue>short)[\s-]+sleeved",
        ("upper_clothing_length", "long"): r"(?P<cue>long)[\s-]+sleeved",
    }
    match = re.fullmatch(patterns.get((slot, canonical), r"(?!)"), mention["raw"], re.I)
    if match:
        start = mention["start"] + match.start("cue")
        end = mention["start"] + match.end("cue")
        # 去掉 short-sleeved 的连接符，留下原有 sleeved；不生成替换词。
        if mention["raw"][match.end("cue"):].startswith("-"):
            end += 1
        return {"start": start, "end": end,
                "raw": mention["raw"][match.start("cue"):end - mention["start"]]}
    return {key: mention[key] for key in ("start", "end", "raw")}


def _intersects(left: Mapping, right: Mapping) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def _source_offsets_after_removal(caption: str, spans: Sequence[Mapping]) -> list[int | None]:
    """与 literal helper 相同的空白清理，保留输出字符到原文的映射以核对实体身份。"""
    removed = {index for span in spans for index in range(span["start"], span["end"])}
    chars: list[tuple[str, int | None]] = []
    for index, char in enumerate(caption):
        if index in removed:
            if index == 0 or index - 1 not in removed:
                chars.append((" ", None))
        else:
            chars.append((char, index))
    collapsed: list[tuple[str, int | None]] = []
    for char, index in chars:
        if char.isspace():
            if collapsed and not collapsed[-1][0].isspace():
                collapsed.append((" ", index))
        else:
            if char in ",.;:!?" and collapsed and collapsed[-1][0] == " ":
                collapsed.pop()
            collapsed.append((char, index))
    if collapsed and collapsed[-1][0] == " ":
        collapsed.pop()
    return [index for _, index in collapsed]


def ablate_slot_attribute(caption: str, attributes: Mapping[str, str],
                          provenance: Mapping[str, object], slot: str) -> dict:
    """单槽保守消融；不能隔离时返回 invalid/unresolved，不返回可评分文本。

    valid 的范围是 Stage01 显式槽证据及实体绑定，不宣称 NLP/IRRA 完整语义解耦。
    """
    before = extract_with_provenance(caption)
    if slot not in before["attributes"] or attributes.get(slot) in (None, "null"):
        raise ValueError(f"invalid attribute slot: {slot}")
    base = {
        "ablation_status": "invalid/unresolved", "invalid_reasons": [],
        "removed_text": None, "removed_spans": [], "attempted_spans": [],
        "mentions": deepcopy(provenance.get(slot, {}).get("mentions", []))
                    if isinstance(provenance.get(slot), Mapping) else [],
        "removal_policy": REMOVAL_POLICY,
        "validation_scope": "Stage01 explicit slot evidence and linked-object identity",
    }

    def invalid(code: str, **details) -> dict:
        base["invalid_reasons"].append({"code": code, **details})
        return base

    if dict(attributes) != before["attributes"] or dict(provenance) != before["provenance"]:
        return invalid("stage01_mapping_mismatch")
    target = provenance[slot]
    if not target or not target["mentions"]:
        return invalid("missing_target_evidence")
    if attributes[slot] in ("no", "no_glasses"):
        return invalid("negative_polarity_scope_unresolved")
    spans = [_minimal_evidence(slot, attributes[slot], mention) for mention in target["mentions"]]
    base["attempted_spans"] = deepcopy(spans)
    for other, other_target in provenance.items():
        if other == slot or other_target is None:
            continue
        for mention in other_target["mentions"]:
            evidence = _minimal_evidence(other, attributes[other], mention)
            if any(_intersects(span, evidence) for span in spans):
                base["invalid_reasons"].append({"code": "shared_slot_evidence", "other_slot": other})
            # 复合 mention 的其余字符也是已知语义载体：删 man 后只剩 young，
            # 即使 Stage01 仍提取 age，也不能证明年龄还绑定同一个人物。
            if any(_intersects(span, mention) and not _intersects(span, evidence) for span in spans):
                base["invalid_reasons"].append({"code": "other_slot_context_removed", "other_slot": other})
            if "linked_object" in mention and any(
                    _intersects(span, mention["linked_object"]) for span in spans):
                base["invalid_reasons"].append({"code": "other_slot_binding_removed", "other_slot": other})
    if base["invalid_reasons"]:
        return base
    candidate = remove_provenance_mentions(caption, {"mentions": spans})
    if not candidate["removed_text"] or not re.search(r"\w", candidate["removed_text"]):
        return invalid("empty_semantic_text")
    after = extract_with_provenance(candidate["removed_text"])
    # Stage01 只捕捉开头 he/she，未记录 his/her 等线索；不能把“复提取null”当作
    # 已消除全部显式性别信息。仅拒绝这类不充分证据，不修改Stage01性别定义。
    if slot == "gender" and re.search(r"\b(?:he|him|his|she|her|hers)\b", candidate["removed_text"], re.I):
        return invalid("unmapped_gender_cue_remaining")
    if after["attributes"][slot] != "null":
        return invalid("target_slot_not_removed", after=after["attributes"][slot])
    for other, value in attributes.items():
        if other != slot and after["attributes"][other] != value:
            base["invalid_reasons"].append({"code": "other_slot_value_changed", "other_slot": other,
                                           "before": value, "after": after["attributes"][other]})
    if base["invalid_reasons"]:
        return base
    positions = _source_offsets_after_removal(caption, candidate["removed_spans"])
    source_to_output = {source: out for out, source in enumerate(positions) if source is not None}

    def projected(span):
        # 非目标证据与实体边界均需保留原字符；重复同名衣物不能代替原实体。
        return (source_to_output[span["start"]], source_to_output[span["end"] - 1] + 1)

    def identity(mention, other, value, *, project):
        evidence = _minimal_evidence(other, value, mention)
        evidence_range = projected(evidence) if project else (evidence["start"], evidence["end"])
        obj = mention.get("linked_object")
        object_range = (projected(obj) if project else (obj["start"], obj["end"])) if obj else None
        return evidence_range, object_range

    for other, other_target in provenance.items():
        if other == slot or other_target is None:
            continue
        expected = {identity(m, other, attributes[other], project=True) for m in other_target["mentions"]}
        actual = {identity(m, other, attributes[other], project=False)
                  for m in after["provenance"][other]["mentions"]}
        if expected != actual:
            base["invalid_reasons"].append({"code": "other_slot_evidence_or_binding_changed", "other_slot": other})
    if base["invalid_reasons"]:
        return base
    base.update({"ablation_status": "valid", "removed_text": candidate["removed_text"],
                 "removed_spans": candidate["removed_spans"], "target_after": "null"})
    return base


def _normalized_features(features: torch.Tensor, rows: int, label: str) -> None:
    if (not isinstance(features, torch.Tensor) or features.ndim != 2
            or features.shape[0] != rows or features.shape[1] == 0
            or features.dtype != torch.float32 or not torch.isfinite(features).all()
            or not torch.allclose(features.norm(dim=1), torch.ones(
                rows, device=features.device), atol=1e-4, rtol=1e-4)):
        raise ValueError(f"{label} must be normalized finite float32 [N,D] features")


class IRRAPairedImageRankDropScorer:
    """消费一次预计算的 clean gallery；不编码图片、不接线 Top-K、保留有符号分数。

    caption/removed_text 是 paired-rank 估计与审计字段，不能作为 Top-K 后图像侧
    攻击/G 的额外条件或文本监督；该阶段只允许选中属性表示提供 attribute-only guidance。
    """

    def __init__(self, encoder: IRRATextEncoder, gallery_paths: Sequence[str],
                 gallery_features: torch.Tensor):
        paths = tuple(gallery_paths)
        if (not paths or any(not isinstance(path, str) or not path for path in paths)
                or len(set(paths)) != len(paths)):
            raise ValueError("gallery paths must be nonempty and unique")
        _normalized_features(gallery_features, len(paths), "gallery")
        self.encoder = encoder
        self.gallery_paths = paths
        self._indices = {path: index for index, path in enumerate(paths)}
        # 与调用方后续原地修改隔离；所有属性及后续 caption 复用同一份快照。
        self._gallery_features = gallery_features.detach().clone()

    @torch.no_grad()
    def score(self, caption: str, paired_image: str, attributes: Mapping[str, str],
              provenance: Mapping[str, object], *, slots: Sequence[str]) -> dict:
        """slots 由调用方显式给定，避免将 prototype 候选池冻结为研究方法。

        返回的完整 caption、消融文本与完整 provenance 用于评分可追溯性；下游必须
        根据 Top-K 结果构造选中属性表示，不能把整条结果当作图像侧监督输入。
        """
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("caption must be nonempty")
        if paired_image not in self._indices:
            raise ValueError("paired image is absent from the fixed gallery")
        if not isinstance(attributes, Mapping) or not isinstance(provenance, Mapping):
            raise ValueError("attributes and provenance must be mappings")
        if isinstance(slots, str) or any(not isinstance(slot, str) for slot in slots):
            raise ValueError("slots must be an explicit sequence of attribute names")
        slots = tuple(slots)
        if len(set(slots)) != len(slots):
            raise ValueError("duplicate scoring slots")
        removals = []
        for slot in slots:
            value, target = attributes.get(slot), provenance.get(slot)
            if (not isinstance(value, str) or value == "null" or not isinstance(target, Mapping)
                    or target.get("canonical") != value):
                raise ValueError(f"invalid attribute/provenance: {slot}")
            removals.append(ablate_slot_attribute(caption, attributes, provenance, slot))
        paired = self._indices[paired_image]

        def rank_text(text: str) -> tuple[int, float, int]:
            # 单条编码和单行 matmul 固定计算形状，避免原文与各消融 batch 精度不同。
            text_features = self.encoder.encode_texts([text])
            _normalized_features(text_features, 1, "text")
            if (text_features.device != self._gallery_features.device
                    or text_features.shape[1] != self._gallery_features.shape[1]):
                raise ValueError("text/gallery feature device or dimension mismatch")
            similarity = text_features @ self._gallery_features.T
            rank = int(paired_image_ranks(similarity, paired)[0].item())
            paired_similarity = float(similarity[0, paired].item())
            ties = int((similarity[0] == similarity[0, paired]).sum().item()) - 1
            return rank, paired_similarity, ties

        r0, sim0, ties0 = rank_text(caption)
        results = []
        for slot, removal in zip(slots, removals):
            if removal["ablation_status"] != "valid":
                results.append({"slot": slot, "canonical": attributes[slot], **removal,
                                "rank_after_removal": None, "score": None,
                                "paired_similarity_after_removal": None,
                                "paired_similarity_ties_after_removal": None})
                continue
            ri, simi, tiesi = rank_text(removal["removed_text"])
            results.append({
                "slot": slot, "canonical": attributes[slot], **removal,
                "rank_after_removal": ri, "score": ri - r0,
                "paired_similarity_after_removal": simi,
                "paired_similarity_ties_after_removal": tiesi,
            })
        return {
            "method": "stage03-v2-irra-paired-image-rank-drop",
            "score_schema": "stage03-v2-slot-ablation-1",
            "caption": caption, "paired_image": paired_image, "paired_gallery_index": paired,
            "gallery_count": len(self.gallery_paths), "r0": r0,
            "paired_similarity_original": sim0, "paired_similarity_ties_original": ties0,
            "rank_policy": RANK_POLICY, "removal_policy": REMOVAL_POLICY,
            "attributes": deepcopy(dict(attributes)), "provenance": deepcopy(dict(provenance)),
            "scoring_slots": list(slots), "attribute_results": results,
            "scores": {row["slot"]: row["score"] for row in results if row["ablation_status"] == "valid"},
            "valid_slots": [row["slot"] for row in results if row["ablation_status"] == "valid"],
            "invalid_slots": [row["slot"] for row in results if row["ablation_status"] != "valid"],
        }
