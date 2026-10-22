"""新交替身份与原Canonical锚点；不重新定义K、编辑规则或Loss。"""
from dataclasses import dataclass, asdict
import math
from tbps.artifacts import digest

def require(ok, message):
    if not ok:
        raise ValueError(message)

def text_sha(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class Anchor:
    semantic_axis: str
    canonical_value: str
    region: str
    raw_text: str
    char_start: int
    char_end: int

    def check(self, caption):
        require(0 <= self.char_start < self.char_end <= len(caption)
                and caption[self.char_start:self.char_end] == self.raw_text, "original Span mismatch")
        return self

    @property
    def h_id(self):
        # 原H使用带默认空格的json，不等同于tbps.artifacts.digest。
        import hashlib, json
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                              ensure_ascii=False).encode()).hexdigest()[:24]

    def stable_id(self, caption):
        return digest({"original_caption_sha256": text_sha(caption), "anchor": asdict(self)})

@dataclass(frozen=True)
class Edit:
    attribute_id: str
    position: int
    original: str
    replacement: str

@dataclass(frozen=True)
class TextTrace:
    original: str
    adversarial: str
    attributes: tuple
    edits: tuple
    budget_k: int
    action_sha256: str

    def validate(self, legal_pairs):
        require(type(self.attributes) is tuple and type(self.edits) is tuple, "immutable trace required")
        require(type(self.budget_k) is int and 0 <= len(self.edits) <= self.budget_k, "H budget")
        require(len(self.action_sha256) == 64, "full original action digest required")
        attrs = {}
        for a in self.attributes:
            a.check(self.original)
            require(a.h_id not in attrs, "AMBIGUOUS H anchor")
            attrs[a.h_id] = a
        import re
        words = list(re.finditer(r"\w+", self.original))
        used_attributes, used_words, positions = set(), set(), set()
        chars = list(self.original)
        for e in self.edits:
            require(e.attribute_id in attrs, "unknown H attribute")
            a = attrs[e.attribute_id]
            require(a.char_start <= e.position < a.char_end and e.position not in positions, "edit outside anchor")
            require(len(e.original) == len(e.replacement) == 1
                    and self.original[e.position] == e.original
                    and (e.original, e.replacement) in legal_pairs, "Unicode mapping mismatch")
            word = next(((m.start(), m.end()) for m in words if m.start() <= e.position < m.end()), None)
            require(word is not None and word not in used_words
                    and e.attribute_id not in used_attributes, "shared word/attribute budget")
            used_words.add(word); used_attributes.add(e.attribute_id); positions.add(e.position)
            chars[e.position] = e.replacement
        require("".join(chars) == self.adversarial, "trace replay mismatch")
        return self

    def mapping(self, g_anchor):
        g_anchor.check(self.original)
        # G/H IDs不参与匹配，严格原文Span及全部语义字段；重叠不猜测。
        matches = [a for a in self.attributes if a == g_anchor]
        require(len(matches) == 1, "missing/AMBIGUOUS canonical alignment")
        s, e = g_anchor.char_start, g_anchor.char_end
        return {
            "anchor_id": g_anchor.stable_id(self.original),
            "original": asdict(g_anchor), "edited_span": [s, e],
            "edited_raw_text": self.adversarial[s:e],
            "original_utf8_span": [len(self.original[:s].encode()), len(self.original[:e].encode())],
            "edited_utf8_span": [len(self.adversarial[:s].encode()), len(self.adversarial[:e].encode())],
        }

@dataclass(frozen=True)
class ConditionIdentity:
    original_caption_sha256: str
    context_caption_sha256: str
    selected_g_ids: tuple
    scores: tuple
    source_space_sha256: str
    anchor_mapping_sha256: str

    def __post_init__(self):
        require(type(self.selected_g_ids) is tuple and type(self.scores) is tuple
                and len(self.selected_g_ids) == len(self.scores)
                and len(set(self.selected_g_ids)) == len(self.selected_g_ids), "fixed G selection")
        require(all(math.isfinite(s) and s > 0 for s in self.scores)
                and list(self.scores) == sorted(self.scores, reverse=True), "original S_raw order")
        for h in (self.original_caption_sha256, self.context_caption_sha256,
                  self.source_space_sha256, self.anchor_mapping_sha256):
            require(len(h) == 64 and all(c in "0123456789abcdef" for c in h), "SHA required")

    @property
    def key(self):
        return digest(asdict(self))

@dataclass(frozen=True)
class BankIdentity:
    g_state_sha256: str
    ordered_train_ids: tuple
    clean_sha256: str
    adversarial_sha256: str
    source_space_sha256: str
    round: int
    arm: str

    def validate(self, expected_ids, g_sha, round_index, arm, *, space=None, clean=None):
        require(self.ordered_train_ids == tuple(expected_ids)
                and len(set(self.ordered_train_ids)) == len(expected_ids), "full ordered Train bank required")
        require(self.g_state_sha256 == g_sha and self.round == round_index and self.arm == arm,
                "stale/cross-arm bank")
        if space is not None:
            require(self.source_space_sha256 == space, "wrong IRRA bank ModelSpace")
        if clean is not None:
            require(self.clean_sha256 == clean, "wrong fixed Clean bank binding")
        return self

def cache_key(kind, *, g=None, text=None, space, ordered_ids=(), reward_context=None):
    """内容/模型空间/有序成员是依赖；H版本本身不使固定G图像失效。"""
    require(kind in ("clean", "adversarial", "text", "reward"), "use ConditionIdentity.key for conditions; unknown cache kind")
    require((kind == "clean" or text is not None or g is not None), "missing cache dependencies")
    if kind in ("adversarial", "reward"):
        require(g is not None, "G identity required")
    if kind in ("text", "reward"):
        require(text is not None, "context caption required")
    if kind == "reward":
        require(reward_context is not None, "paired index/positive-mask context required")
    return digest({"kind": kind, "G": g if kind in ("adversarial", "reward") else None,
                   "text": text if kind in ("text", "reward") else None,
                   "reward_context": reward_context if kind == "reward" else None,
                   "space": space, "ordered_ids": list(ordered_ids)})
