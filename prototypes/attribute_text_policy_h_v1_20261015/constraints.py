"""受控形近Unicode白名单与可回放的单字符替换；不做自然语言生成。"""
from dataclasses import dataclass, asdict
import unicodedata
from contracts import TextExample, word_at

# 人工限制的Greek/Cyrillic形近字；非通用语义等价承诺，显示依赖字体。
CONFUSABLES = {
    "a": ("а",), "c": ("с", "ϲ"), "e": ("е",), "h": ("һ",),
    "i": ("і",), "j": ("ј",), "l": ("ӏ",), "o": ("о", "ο"),
    "p": ("р", "ρ"), "s": ("ѕ",), "u": ("υ",), "x": ("х", "χ"),
    "y": ("у",),
    "A": ("А",), "B": ("В",), "C": ("С",), "E": ("Е",),
    "H": ("Н",), "I": ("І",), "J": ("Ј",), "K": ("К",),
    "M": ("М",), "O": ("О",), "P": ("Р",), "S": ("Ѕ",),
    "T": ("Т",), "X": ("Х",), "Y": ("Υ",),
}
REPLACEMENTS = tuple(sorted({r for rs in CONFUSABLES.values() for r in rs}))

@dataclass(frozen=True)
class EditRecord:
    attribute_id: str
    semantic_axis: str
    canonical_value: str
    region: str
    source_span: tuple[int, int]
    source_raw: str
    position: int
    position_in_attribute: int
    original_utf8_offset: int
    original_char: str
    replacement_char: str
    original_codepoint: str
    replacement_codepoint: str
    replacement_unicode_name: str

def make_edit(example, attribute_index, position, replacement):
    if type(attribute_index) is not int or not 0 <= attribute_index < len(example.attributes):
        raise ValueError("invalid attribute index")
    a = example.attributes[attribute_index]
    if type(position) is not int or not a.char_start <= position < a.char_end:
        raise ValueError("edit outside original attribute span")
    original = example.caption[position]
    if replacement not in CONFUSABLES.get(original, ()):
        raise ValueError("replacement outside visual whitelist")
    return EditRecord(a.attribute_id, a.semantic_axis, a.canonical_value, a.region,
                      (a.char_start, a.char_end), a.raw_text, position,
                      position-a.char_start,
                      len(example.caption[:position].encode("utf-8")),
                      original, replacement, f"U+{ord(original):04X}",
                      f"U+{ord(replacement):04X}", unicodedata.name(replacement))

def replay(example, edits, budget):
    if type(budget) is not int or budget < 0 or len(edits) > budget:
        raise ValueError("edit budget exceeded")
    chars = list(example.caption)
    seen_attrs, seen_words = set(), set()
    attrs = {a.attribute_id: i for i, a in enumerate(example.attributes)}
    for edit in edits:
        if edit.attribute_id not in attrs:
            raise ValueError("unknown canonical attribute")
        expected = make_edit(example, attrs[edit.attribute_id],
                             edit.position, edit.replacement_char)
        if edit != expected:
            raise ValueError("edit trace/source/codepoint mismatch")
        word = word_at(example.caption, edit.position)
        if word is None or word in seen_words or edit.attribute_id in seen_attrs:
            raise ValueError("one edit per selected attribute and shared word")
        seen_attrs.add(edit.attribute_id)
        seen_words.add(word)
        chars[edit.position] = edit.replacement_char
    return "".join(chars)

def candidates(example, current, attribute_index, used_words, probe):
    a = example.attributes[attribute_index]
    result = {}
    if probe.inspect(current).truncated:
        return result
    for pos in range(a.char_start, a.char_end):
        word = word_at(example.caption, pos)
        if word is None or word in used_words:
            continue
        legal = []
        for replacement in CONFUSABLES.get(example.caption[pos], ()):
            modified = current[:pos]+replacement+current[pos+1:]
            if probe.legal_change(current, modified):
                legal.append(replacement)
        if legal:
            result[pos] = tuple(legal)
    return result
