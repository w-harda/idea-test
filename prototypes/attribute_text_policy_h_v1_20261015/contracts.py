"""H-v1仅文本输入与Canonical→原始Unicode字符Span契约。"""
from dataclasses import dataclass, asdict
import hashlib, json, re

AXES = ("age", "gender", "hair_length", "color", "garment_type",
        "sleeve_length", "lower_garment_length", "hem_extent",
        "hat", "glasses", "backpack", "bag")
REGIONS = ("person", "head", "upper", "lower", "full_body", "carried", "other")

@dataclass(frozen=True)
class AttributeSpan:
    semantic_axis: str
    canonical_value: str
    region: str
    raw_text: str
    char_start: int
    char_end: int

    def validate(self, caption):
        if self.semantic_axis not in AXES or self.region not in REGIONS:
            raise ValueError("unsupported attribute type/region")
        if not isinstance(self.canonical_value, str) or not self.canonical_value:
            raise ValueError("missing canonical value")
        if (type(self.char_start) is not int or type(self.char_end) is not int
                or not 0 <= self.char_start < self.char_end <= len(caption)
                or caption[self.char_start:self.char_end] != self.raw_text):
            raise ValueError("original source span mismatch; relocation forbidden")

    @property
    def attribute_id(self):
        # 不编码row_id、PID、图像路径或配对信息。
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True,
                              ensure_ascii=False).encode()).hexdigest()[:24]

@dataclass(frozen=True)
class TextExample:
    caption: str
    attributes: tuple[AttributeSpan, ...]

    def __post_init__(self):
        if not isinstance(self.caption, str) or not self.caption.strip():
            raise ValueError("nonempty original caption required")
        if type(self.attributes) is not tuple:
            raise ValueError("immutable text-only attribute tuple required")
        for attr in self.attributes:
            if not isinstance(attr, AttributeSpan):
                raise TypeError("only explicit text-source attributes accepted")
            attr.validate(self.caption)
        ids = [a.attribute_id for a in self.attributes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate attribute source")

def from_text_spans(caption, attributes):
    """只接受明确的文本字段；不静默接收或过滤带PID/评分的记录。"""
    allowed = set(AttributeSpan.__dataclass_fields__)
    result = []
    for row in attributes:
        if not isinstance(row, dict) or set(row) != allowed:
            raise ValueError("only canonical/type/source-span fields accepted")
        result.append(AttributeSpan(**row))
    return TextExample(caption, tuple(result))

def word_at(caption, position):
    for match in re.finditer(r"\w+", caption, re.UNICODE):
        if match.start() <= position < match.end():
            return (match.start(), match.end())
    return None
