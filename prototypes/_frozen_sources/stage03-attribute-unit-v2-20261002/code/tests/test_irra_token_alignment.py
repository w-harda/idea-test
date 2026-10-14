"""轻量官方形态tokenizer验证：原文位置、BPE边界、截断与跨槽共用。"""
import html

import ftfy
import pytest
import regex

from attributes.irra_token_alignment import align_irra_tokens, map_attribute_tokens
from attributes.irra_paired_rank_drop import ablate_slot_attribute
from attributes.text_extractor import extract_with_provenance


def basic_clean(text):
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def whitespace_clean(text):
    return regex.sub(r"\s+", " ", text).strip()


class FakeOfficialTokenizer:
    """测试清洗/regex/byte-BPE接口，不需要官方大型merge词表。"""

    def __init__(self, split_words=None):
        self.byte_encoder = {byte: chr(byte + 256) for byte in range(256)}
        self.encoder = {"<|mask|>": 49405, "<|startoftext|>": 49406,
                        "<|endoftext|>": 49407}
        self.pat = regex.compile(
            r"<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )
        self.split_words = split_words or {}

    def bpe(self, encoded):
        decoder = {value: key for key, value in self.byte_encoder.items()}
        word = bytes(decoder[char] for char in encoded).decode("utf-8")
        if word in self.encoder:
            return word
        points = [0, *self.split_words.get(word, []), len(encoded)]
        pieces = [encoded[start:end] for start, end in zip(points[:-1], points[1:])]
        pieces[-1] += "</w>"
        for piece in pieces:
            if piece not in self.encoder:
                self.encoder[piece] = len(self.encoder) + 100
        return " ".join(pieces)

    def encode(self, text):
        tokens = []
        for match in self.pat.finditer(whitespace_clean(basic_clean(text)).lower()):
            encoded = "".join(self.byte_encoder[byte] for byte in match.group().encode("utf-8"))
            tokens.extend(self.encoder[piece] for piece in self.bpe(encoded).split(" "))
        return tokens


def mapped(text, tokenizer=None, text_length=77):
    tokenizer = tokenizer or FakeOfficialTokenizer()
    alignment = align_irra_tokens(text, tokenizer, text_length)
    rows = map_attribute_tokens(text, **extract_with_provenance(text), alignment=alignment)
    return alignment, {row["slot"]: row for row in rows}


def test_replays_full_sequence_and_original_whitespace_case_offsets():
    text = " \tA  BLACK\nshirt.  "
    tokenizer = FakeOfficialTokenizer()
    alignment = align_irra_tokens(text, tokenizer)
    assert alignment["status"] == "valid"
    assert alignment["token_ids"][:6] == [49406, *tokenizer.encode(text), 49407]
    assert alignment["eot_index"] == 5
    assert len(alignment["token_ids"]) == 77
    assert all(value == 0 for value in alignment["token_ids"][6:])
    assert [(row["start"], row["end"], row["raw"]) for row in alignment["content_tokens"]] == [
        (2, 3, "A"), (5, 10, "BLACK"), (11, 16, "shirt"), (16, 17, ".")]


def test_repeated_words_resolve_original_positions_not_find():
    alignment, rows = mapped("A black shirt and black pants.")
    assert rows["upper_clothing_color"]["token_indices"] == [2]
    assert rows["lower_clothing_color"]["token_indices"] == [5]
    assert rows["upper_clothing_color"]["shared_token_indices"] == []
    assert alignment["content_tokens"][1]["raw"] == "black"


def test_age_gender_compound_maps_only_respective_evidence():
    _, rows = mapped("A young man wearing a black shirt.")
    assert rows["age"]["semantic_spans"] == [{"start": 2, "end": 7, "raw": "young"}]
    assert rows["age"]["token_indices"] == [2]
    assert rows["gender"]["token_indices"] == [3]
    assert rows["age"]["shared_token_indices"] == []


def test_removal_invalid_type_still_maps_without_linked_color_tokens():
    text = "A woman wearing a black shirt, a pair of black pants and black shoes."
    _, rows = mapped(text)
    traced = extract_with_provenance(text)
    assert ablate_slot_attribute(text, **traced, slot="upper_clothing_type")["ablation_status"] != "valid"
    assert rows["upper_clothing_type"]["mapping_status"] == "valid"
    assert rows["upper_clothing_type"]["token_indices"] == [6]
    assert rows["upper_clothing_color"]["token_indices"] == [5]
    assert rows["upper_clothing_color"]["linked_objects"] == [{"start": 24, "end": 29, "raw": "shirt"}]


def test_real_shared_short_is_available_but_explicitly_shared():
    text = ("short dark hair blue shirt with bright green on front wearing a backpack, "
            "with greenish type shorts on and slip on shoes")
    _, rows = mapped(text)
    for slot in ("hair_length", "upper_clothing_length"):
        assert rows[slot]["mapping_status"] == "valid"
        assert rows[slot]["token_indices"] == [1]
        assert rows[slot]["shared_token_indices"] == [1]
        assert "shared_token_attribution_is_not_independent_slot_gradient" in rows[slot]["warnings"]


def test_real_shared_black_maps_same_original_token_for_two_slots():
    text = ("the man is dressed in a Business suit.  The jacket is solid black as well as "
            "the pants and tie.  His shirt is gray or light blue.")
    _, rows = mapped(text)
    upper, lower = rows["upper_clothing_color"], rows["lower_clothing_color"]
    assert upper["token_indices"] == lower["token_indices"]
    assert upper["shared_slots"] == ["lower_clothing_color"]
    assert upper["semantic_spans"] == [{"start": 60, "end": 65, "raw": "black"}]


def test_split_bpe_pieces_are_aggregated_within_one_exact_attribute():
    text = "A black sweatshirt."
    _, rows = mapped(text, FakeOfficialTokenizer({"sweatshirt": [5]}))
    assert rows["upper_clothing_type"]["mapping_status"] == "valid"
    assert rows["upper_clothing_type"]["token_indices"] == [3, 4]


def test_short_sleeved_connector_is_not_added_to_attribute_tokens():
    _, rows = mapped("A man in a black short-sleeved shirt.")
    length = rows["upper_clothing_length"]
    assert length["mapping_status"] == "valid"
    assert length["semantic_spans"] == [{"start": 17, "end": 22, "raw": "short"}]
    assert length["token_indices"] == [6]


def test_duplicate_mentions_do_not_double_count_token():
    text = "a black shirt"
    traced = extract_with_provenance(text)
    traced["provenance"]["upper_clothing_color"]["mentions"] *= 2
    alignment = align_irra_tokens(text, FakeOfficialTokenizer())
    rows = map_attribute_tokens(text, **traced, alignment=alignment)
    color = next(row for row in rows if row["slot"] == "upper_clothing_color")
    assert color["token_indices"] == [2]
    assert len(color["semantic_spans"]) == 1


def test_partial_bpe_token_attribute_boundary_is_unresolved():
    text = "backpack"
    alignment = align_irra_tokens(text, FakeOfficialTokenizer())
    rows = map_attribute_tokens(text, {"bag": "yes"}, {
        "bag": {"canonical": "yes", "mentions": [{"start": 4, "end": 8, "raw": "pack"}]}
    }, alignment)
    assert rows[0]["mapping_status"] == "unresolved"
    assert rows[0]["token_indices"] == []
    assert "attribute_boundary_splits_bpe_token" in rows[0]["mapping_reasons"]


def test_truncated_attribute_is_not_scored_from_partial_evidence():
    text = "A man is wearing a black sweatshirt."
    alignment, rows = mapped(text, text_length=5)
    assert alignment["truncated"]
    assert alignment["eot_index"] == 4
    assert len(alignment["content_tokens"]) == 3
    assert len(alignment["all_content_tokens"]) == 8
    assert rows["gender"]["mapping_status"] == "valid"
    assert rows["upper_clothing_type"]["mapping_status"] == "unresolved"
    assert "attribute_evidence_truncated" in rows["upper_clothing_type"]["mapping_reasons"]


def test_truncation_through_bpe_word_is_diagnosed():
    alignment, rows = mapped("A black sweatshirt.", FakeOfficialTokenizer({"sweatshirt": [5]}), 5)
    assert alignment["truncation_splits_word"]
    assert rows["upper_clothing_type"]["mapping_status"] == "unresolved"
    assert rows["upper_clothing_color"]["mapping_status"] == "unresolved"
    assert "binding_not_visible_or_unresolved" in rows["upper_clothing_color"]["mapping_reasons"]


@pytest.mark.parametrize("text", ["A black shirt &amp; pants.", "A black shirt â pants."])
def test_html_or_ftfy_character_changes_never_guess_offsets(text):
    alignment, rows = mapped(text)
    assert alignment["status"] == "unresolved"
    assert "cleaning_changes_source_characters" in alignment["reasons"]
    assert alignment["tokenizer_encode_matches_replay"]
    assert all(row["mapping_status"] == "unresolved" for row in rows.values())


def test_unicode_complete_char_bpe_boundaries_can_be_mapped():
    text = "CAFÉ black shirt"
    alignment, rows = mapped(text)
    assert alignment["status"] == "valid"
    assert alignment["content_tokens"][0]["raw"] == "CAFÉ"
    assert alignment["content_tokens"][0]["end"] == 4
    assert rows["upper_clothing_color"]["mapping_status"] == "valid"


def test_utf8_character_split_is_unresolved_only_for_involved_tokens():
    alignment, rows = mapped("café black shirt", FakeOfficialTokenizer({"café": [4]}))
    assert alignment["status"] == "unresolved"
    assert "utf8_character_split" in alignment["reasons"]
    assert alignment["content_tokens"][0]["start"] is None
    assert rows["upper_clothing_color"]["mapping_status"] == "valid"


def test_unicode_lower_expansion_is_explicit_unresolved():
    alignment = align_irra_tokens("İ in a black shirt", FakeOfficialTokenizer())
    assert "unicode_lower_expansion" in alignment["reasons"]


def test_invalid_provenance_cannot_map_even_if_word_exists_elsewhere():
    text = "a black shirt and black pants"
    traced = extract_with_provenance(text)
    traced["provenance"]["upper_clothing_color"]["mentions"][0]["start"] += 1
    rows = map_attribute_tokens(text, **traced, alignment=align_irra_tokens(text, FakeOfficialTokenizer()))
    assert next(row for row in rows if row["slot"] == "upper_clothing_color")["mapping_status"] == "unresolved"


def test_stale_caption_alignment_is_rejected():
    alignment = align_irra_tokens("a black shirt", FakeOfficialTokenizer())
    with pytest.raises(ValueError, match="does not belong"):
        map_attribute_tokens("a white shirt", **extract_with_provenance("a white shirt"), alignment=alignment)


def test_malformed_mention_is_unresolved_instead_of_crashing():
    text = "a black shirt"
    traced = extract_with_provenance(text)
    traced["provenance"]["upper_clothing_color"]["mentions"] = ["black"]
    rows = map_attribute_tokens(text, **traced, alignment=align_irra_tokens(text, FakeOfficialTokenizer()))
    assert next(row for row in rows if row["slot"] == "upper_clothing_color")["mapping_reasons"] == ["invalid_provenance_span"]


def test_negative_value_preserves_provenance_scope_warning():
    _, rows = mapped("A man without a hat.")
    assert rows["hat"]["mapping_status"] == "valid"
    assert rows["hat"]["canonical"] == "no"
    assert "negative_polarity_not_expanded_beyond_provenance" in rows["hat"]["warnings"]


def test_special_caption_token_is_flagged_not_ordinary_content_attribution():
    alignment = align_irra_tokens("black <|mask|> shirt", FakeOfficialTokenizer())
    assert "special_token_in_caption" in alignment["reasons"]
    special = alignment["content_tokens"][1]
    assert special["span_status"] == "unresolved"


def test_retained_color_with_truncated_linked_shirt_is_unresolved():
    alignment, rows = mapped("A black shirt.", text_length=4)
    assert alignment["content_tokens"][-1]["raw"] == "black"
    color = rows["upper_clothing_color"]
    assert color["mapping_status"] == "unresolved"
    assert color["token_indices"] == []
    assert "binding_not_visible_or_unresolved" in color["mapping_reasons"]
    assert "binding_evidence_truncated" in color["binding_diagnostics"][0]["reasons"]


def test_caption_eot_special_token_rejects_all_ordinary_attributes():
    _, rows = mapped("A black shirt <|endoftext|> and black pants.")
    assert len(rows) == 4
    assert all(row["mapping_status"] == "unresolved" for row in rows.values())
    assert all(row["token_indices"] == [] for row in rows.values())
    assert all("special_token_in_caption" in row["mapping_reasons"] for row in rows.values())


def test_binding_boundary_inside_bpe_token_is_unresolved_without_scoring_binding():
    text = "black tshirt"
    alignment = align_irra_tokens(text, FakeOfficialTokenizer())
    rows = map_attribute_tokens(text, {"upper_clothing_color": "black"}, {
        "upper_clothing_color": {"canonical": "black", "mentions": [{
            "start": 0, "end": 5, "raw": "black",
            "linked_object": {"start": 7, "end": 12, "raw": "shirt"},
        }]}
    }, alignment)
    assert rows[0]["mapping_status"] == "unresolved"
    assert rows[0]["mapping_reasons"] == ["binding_not_visible_or_unresolved"]
    assert rows[0]["binding_diagnostics"][0]["reasons"] == ["binding_boundary_splits_bpe_token"]
    assert rows[0]["token_indices"] == []


def test_encode_replay_mismatch_is_explicit_and_blocks_mapping():
    tokenizer = FakeOfficialTokenizer()
    encode = tokenizer.encode
    tokenizer.encode = lambda text: [*encode(text), 12345]
    alignment, rows = mapped("a black shirt", tokenizer)
    assert not alignment["tokenizer_encode_matches_replay"]
    assert all(row["mapping_status"] == "unresolved" for row in rows.values())


@pytest.mark.parametrize("text_length", [1, True, 2.5])
def test_invalid_model_sequence_length_is_rejected(text_length):
    with pytest.raises(ValueError):
        align_irra_tokens("black shirt", FakeOfficialTokenizer(), text_length)
