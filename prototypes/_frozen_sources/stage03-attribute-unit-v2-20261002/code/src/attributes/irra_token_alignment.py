"""Stage03-v2 prototype：官方 IRRA BPE 到 Stage01 原文证据的可审计映射。

此处只做 provenance 锚定的 token 归因，不修改文本或 Stage01 定义。不同槽可
共用 token；这类分数不能解释为已经分离的结构化属性梯度。linked_object 只作
诊断，不并入目标属性 token。清洗改变字符、UTF-8 字符被 BPE 切开或属性截断时
保守标记 unresolved，而不通过重复词搜索猜测位置。
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import sys

from .irra_paired_rank_drop import _minimal_evidence


ALIGNMENT_POLICY = "official IRRA cleaning/regex/byte-BPE replay with exact source offsets"
ATTRIBUTE_MAPPING_POLICY = "Stage01 minimal explicit evidence; linked_object diagnostic only"


def _normalized_source(caption: str) -> tuple[str, list[tuple[int, int]], bool]:
    """只对可保留位置的 whitespace/lower 变换建立字符映射。"""
    chars, offsets = [], []
    expansion = False
    for index, char in enumerate(caption):
        if char.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                offsets.append((index, index + 1))
            elif chars:
                offsets[-1] = (offsets[-1][0], index + 1)
            continue
        lowered = char.lower()
        expansion = expansion or len(lowered) != 1
        for output in lowered:
            chars.append(output)
            offsets.append((index, index + 1))
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets, expansion


def align_irra_tokens(caption: str, tokenizer, text_length: int = 77) -> dict:
    """复现 supplied 官方 SimpleTokenizer，输出与官方 tokenize 相同的 SOT/EOT/pad。

    content_tokens 只包含模型保留的内容，all_content_tokens 另保留截断内容用于
    属性诊断。token_index 是模型位置（首个内容为1）；SOT/EOT/pad不在这两张表。
    status 表示整条文本映射是否完全可靠；某个未涉及异常的属性仍可独立映射。
    """
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must be nonempty")
    if isinstance(text_length, bool) or not isinstance(text_length, int) or text_length < 2:
        raise ValueError("text_length must be an integer >= 2")
    module = sys.modules.get(type(tokenizer).__module__)
    basic_clean = getattr(module, "basic_clean", None)
    whitespace_clean = getattr(module, "whitespace_clean", None)
    if (not callable(basic_clean) or not callable(whitespace_clean)
            or not hasattr(tokenizer, "pat") or not callable(getattr(tokenizer, "bpe", None))
            or not isinstance(getattr(tokenizer, "byte_encoder", None), Mapping)
            or not isinstance(getattr(tokenizer, "encoder", None), Mapping)):
        raise ValueError("tokenizer must expose official IRRA SimpleTokenizer API and cleaners")
    full_ids = list(tokenizer.encode(caption))
    if any(isinstance(item, bool) or not isinstance(item, int) for item in full_ids):
        raise ValueError("tokenizer.encode must return integer token ids")
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    retained_ids = full_ids[:text_length - 2]
    token_ids = [sot, *retained_ids, eot]
    token_ids += [0] * (text_length - len(token_ids))
    cleaned = whitespace_clean(basic_clean(caption)).lower()
    normalized, offsets, lower_expansion = _normalized_source(caption)
    reasons = []
    if cleaned != normalized:
        reasons.append("cleaning_changes_source_characters")
    if lower_expansion:
        reasons.append("unicode_lower_expansion")
    can_map_source = not reasons
    byte_decoder = {value: key for key, value in tokenizer.byte_encoder.items()}
    special_ids = {tokenizer.encoder[name] for name in
                   ("<|startoftext|>", "<|endoftext|>", "<|mask|>")
                   if name in tokenizer.encoder}
    rows, replay_ids = [], []
    for word_index, match in enumerate(tokenizer.pat.finditer(cleaned)):
        word = match.group()
        encoded = "".join(tokenizer.byte_encoder[byte] for byte in word.encode("utf-8"))
        pieces = tokenizer.bpe(encoded).split(" ")
        replay_ids.extend(tokenizer.encoder[piece] for piece in pieces)
        # BPE符号可逆地对应byte；只有完整Unicode字符边界才可映射原文span。
        boundaries = {0: 0}
        byte_count = 0
        for index, char in enumerate(word):
            byte_count += len(char.encode("utf-8"))
            boundaries[byte_count] = index + 1
        reconstructed = "".join(piece[:-4] if piece.endswith("</w>") else piece
                                for piece in pieces)
        representation_ok = reconstructed == encoded
        cursor = 0
        for piece in pieces:
            token_id = tokenizer.encoder[piece]
            symbols = piece[:-4] if piece.endswith("</w>") else piece
            next_cursor = cursor + len(symbols)
            token_reasons = list(reasons)
            start = end = raw = None
            word_start = word_end = None
            if can_map_source:
                word_start = offsets[match.start()][0]
                word_end = offsets[match.end() - 1][1]
            if not representation_ok or any(symbol not in byte_decoder for symbol in symbols):
                token_reasons.append("bpe_byte_reconstruction_mismatch")
            elif cursor not in boundaries or next_cursor not in boundaries:
                token_reasons.append("utf8_character_split")
            elif can_map_source:
                normalized_start = match.start() + boundaries[cursor]
                normalized_end = match.start() + boundaries[next_cursor]
                start = offsets[normalized_start][0]
                end = offsets[normalized_end - 1][1]
                raw = caption[start:end]
            if token_id in special_ids:
                token_reasons.append("special_token_in_caption")
            rows.append({
                "token_index": len(rows) + 1, "token_id": token_id,
                "bpe_piece": piece, "word_index": word_index,
                "word_start": word_start, "word_end": word_end,
                "byte_start": cursor, "byte_end": next_cursor,
                "start": start, "end": end, "raw": raw,
                "span_status": "unresolved" if token_reasons else "valid",
                "reasons": token_reasons,
                "included": len(rows) < text_length - 2,
            })
            cursor = next_cursor
    replay_matches = replay_ids == full_ids
    if not replay_matches:
        reasons.append("tokenizer_encode_replay_mismatch")
        for row in rows:
            row["span_status"] = "unresolved"
            row["reasons"].append("tokenizer_encode_replay_mismatch")
    token_reasons = sorted({reason for row in rows for reason in row["reasons"]})
    reasons = sorted(set(reasons + token_reasons))
    truncated = len(full_ids) > len(retained_ids)
    retained_rows = rows[:text_length - 2]
    cut_word = bool(truncated and retained_rows and
                    retained_rows[-1]["word_index"] == rows[len(retained_rows)]["word_index"])
    return {
        "status": "unresolved" if reasons else "valid", "reasons": reasons,
        "caption": caption,
        "alignment_policy": ALIGNMENT_POLICY,
        "token_ids": token_ids, "content_tokens": retained_rows, "all_content_tokens": rows,
        "text_length": text_length, "sot_index": 0, "eot_index": len(retained_ids) + 1,
        "content_token_count": len(full_ids), "retained_content_token_count": len(retained_ids),
        "truncated": truncated, "truncated_content_token_count": len(full_ids) - len(retained_ids),
        "truncation_splits_word": cut_word, "tokenizer_encode_matches_replay": replay_matches,
        "normalized_text": cleaned,
    }


def _checked_span(caption: str, span: Mapping) -> dict:
    if not isinstance(span, Mapping):
        raise ValueError("provenance span must be a mapping")
    start, end, raw = span.get("start"), span.get("end"), span.get("raw")
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or not 0 <= start < end <= len(caption) or caption[start:end] != raw):
        raise ValueError("provenance span does not match original caption")
    return {"start": start, "end": end, "raw": raw}


def _semantic_span(caption: str, slot: str, canonical: str, mention: Mapping) -> dict:
    span = _minimal_evidence(slot, canonical, mention)
    # removal helper删除连接符是语法清理；梯度归因只取short/long本身。
    if slot == "upper_clothing_length" and span["raw"].endswith("-"):
        span["end"] -= 1
        span["raw"] = caption[span["start"]:span["end"]]
    return _checked_span(caption, span)


def _overlap(left: Mapping, right: Mapping) -> bool:
    return left["start"] < right["end"] and right["start"] < left["end"]


def _span_token_indices(caption: str, span: Mapping, tokens: list[dict]) -> tuple[set[int], list[str]]:
    """只检查可见性与完整BPE边界，供目标证据和绑定实体共用。"""
    expected = {index for index in range(span["start"], span["end"])
                if not caption[index].isspace()}
    covered, selected, reasons = set(), set(), []
    for token in tokens:
        if token["start"] is None:
            word = {"start": token["word_start"], "end": token["word_end"]}
            if word["start"] is None or _overlap(span, word):
                reasons.extend(token["reasons"] or ["unresolved_token_span"])
            continue
        if not _overlap(span, token):
            continue
        if token["span_status"] != "valid":
            reasons.extend(token["reasons"] or ["unresolved_token_span"])
        if not token["included"]:
            reasons.append("attribute_evidence_truncated")
        if token["start"] < span["start"] or token["end"] > span["end"]:
            reasons.append("attribute_boundary_splits_bpe_token")
        if token["span_status"] == "valid" and token["included"]:
            selected.add(token["token_index"])
            covered.update(range(token["start"], token["end"]))
    if not expected.issubset(covered):
        reasons.append("attribute_evidence_not_fully_covered")
    return selected, sorted(set(reasons))


def map_attribute_tokens(caption: str, attributes: Mapping[str, str],
                         provenance: Mapping[str, object], alignment: Mapping) -> list[dict]:
    """所有非null槽按原文位置映射；同槽重复token去重，跨槽共享仅显式标注。

    valid只表示Stage01记录证据可精确投影到输入token，并非canonical的所有隐含
    信息已经记录；尤其否定极性和未记录代词不自动扩展。属性边界切开一个BPE
    token、证据/linked_object被截断或无法可靠映射时不生成部分属性分数。
    linked_object必须可见但不加入归因token；原文含特殊token时所有属性保守拒绝。
    """
    if not isinstance(attributes, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("attributes and provenance must be mappings")
    if alignment.get("caption") != caption:
        raise ValueError("alignment does not belong to this original caption")
    tokens = alignment["all_content_tokens"]
    rows = []
    for slot, canonical in attributes.items():
        if canonical in (None, "null"):
            continue
        row = {
            "slot": slot, "canonical": canonical,
            "mapping_status": "unresolved", "mapping_reasons": [],
            "token_indices": [], "semantic_spans": [], "mentions": [],
            "linked_objects": [], "shared_token_indices": [], "shared_slots": [],
            "binding_diagnostics": [],
            "warnings": [], "mapping_policy": ATTRIBUTE_MAPPING_POLICY,
            "attribution_scope": "Stage01 recorded explicit evidence only",
        }
        target = provenance.get(slot)
        if (not isinstance(canonical, str) or not isinstance(target, Mapping)
                or target.get("canonical") != canonical
                or not isinstance(target.get("mentions"), list) or not target["mentions"]):
            row["mapping_reasons"].append("missing_or_mismatched_attribute_provenance")
            rows.append(row)
            continue
        row["mentions"] = deepcopy(target["mentions"])
        try:
            spans = []
            for mention in target["mentions"]:
                _checked_span(caption, mention)
                spans.append(_semantic_span(caption, slot, canonical, mention))
                if "linked_object" in mention:
                    row["linked_objects"].append(_checked_span(caption, mention["linked_object"]))
            spans = list({(span["start"], span["end"]): span for span in spans}.values())
        except (ValueError, TypeError, KeyError):
            row["mapping_reasons"].append("invalid_provenance_span")
            rows.append(row)
            continue
        row["semantic_spans"] = spans
        if canonical in ("no", "no_glasses"):
            row["warnings"].append("negative_polarity_not_expanded_beyond_provenance")
        if slot == "gender":
            row["warnings"].append("unrecorded_gender_cues_not_added")
        if "special_token_in_caption" in alignment.get("reasons", []):
            row["mapping_reasons"].append("special_token_in_caption")
        selected = set()
        for span in spans:
            indices, reasons = _span_token_indices(caption, span, tokens)
            selected.update(indices)
            row["mapping_reasons"].extend(reasons)
        for binding in row["linked_objects"]:
            _, reasons = _span_token_indices(caption, binding, tokens)
            row["binding_diagnostics"].append({
                **binding, "status": "unresolved" if reasons else "valid",
                "reasons": [reason.replace("attribute_", "binding_") for reason in reasons],
            })
            if reasons:
                row["mapping_reasons"].append("binding_not_visible_or_unresolved")
        if not alignment["tokenizer_encode_matches_replay"]:
            row["mapping_reasons"].append("tokenizer_encode_replay_mismatch")
        row["mapping_reasons"] = sorted(set(row["mapping_reasons"]))
        # unresolved不留下能被误用的部分token集合。
        if not row["mapping_reasons"] and selected:
            row["mapping_status"] = "valid"
            row["token_indices"] = sorted(selected)
        elif not row["mapping_reasons"]:
            row["mapping_reasons"].append("no_attribute_content_tokens")
        rows.append(row)
    for row in rows:
        owners = {}
        for other in rows:
            if other is row:
                continue
            for token_index in set(row["token_indices"]) & set(other["token_indices"]):
                owners.setdefault(token_index, []).append(other["slot"])
        row["shared_token_indices"] = sorted(owners)
        row["shared_slots"] = sorted({slot for slots in owners.values() for slot in slots})
        row["shared_token_owners"] = {str(index): sorted(slots) for index, slots in owners.items()}
        if owners:
            row["warnings"].append("shared_token_attribution_is_not_independent_slot_gradient")
    return rows
