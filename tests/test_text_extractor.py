import json
import sys
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from attributes import TextAttributeExtractor, extract, extract_with_provenance
from attributes.dataset_adapter import CaptionRecord, JsonCaptionAdapter
from scripts.batch_extract import main as batch_main


def test_required_caption_maps_to_fixed_slots() -> None:
    assert extract("a man with black hair wearing a white shirt and black pants") == {
        "age": "null",
        "gender": "male",
        "hair_length": "null",
        "upper_clothing_length": "null",
        "upper_clothing_color": "white",
        "upper_clothing_type": "t_shirt_shirt",
        "lower_clothing_length": "null",
        "lower_clothing_color": "black",
        "lower_clothing_type": "trousers_shorts",
        "backpack": "null",
        "bag": "null",
        "glasses": "null",
        "hat": "null",
    }


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("without a hat", {"hat": "no", "backpack": "null"}),
        ("a woman with a hat and a backpack", {"gender": "female", "hat": "yes", "backpack": "yes"}),
        ("no bag and a hat", {"bag": "no", "hat": "yes"}),
        ("a lady wearing navy trousers", {"gender": "female", "lower_clothing_color": "blue", "lower_clothing_length": "null"}),
        ("a white tee under a black jacket", {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "black"}),
        ("a red and white shirt", {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "null"}),
        ("a shirt and a jacket", {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "null"}),
        ("a red shirt without a black jacket", {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "red"}),
        ("black hair and a blue skirt", {"hair_length": "null", "lower_clothing_color": "blue", "lower_clothing_type": "skirt_dress"}),
        ("long black hair in a white shirt", {"hair_length": "long", "upper_clothing_color": "white"}),
        ("a man and a woman with no glasses", {"gender": "null", "glasses": "no_glasses"}),
        ("a person in jeans and a shirt", {"lower_clothing_length": "null", "upper_clothing_length": "null"}),
    ],
)
def test_explicit_attributes_binding_and_conflicts(caption: str, expected: dict[str, str]) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected
    assert len(result) == 13


def test_repeated_aliases_keep_one_canonical_value() -> None:
    result = extract("a man, male, wearing a navy blue t-shirt")
    assert result["gender"] == "male"
    assert result["upper_clothing_color"] == "blue"
    assert result["upper_clothing_type"] == "t_shirt_shirt"


def test_non_string_caption_is_rejected() -> None:
    with pytest.raises(TypeError):
        TextAttributeExtractor().extract(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        (
            "A boy wearing a blue tee shirt with black shorts and carrying a black jacket over his shoulder.",
            {"gender": "male", "upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "blue",
             "lower_clothing_type": "trousers_shorts", "lower_clothing_color": "black"},
        ),
        (
            "A woman wearing a white shirt and holding a red coat.",
            {"gender": "female", "upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "white"},
        ),
        (
            "A man wearing a black jacket and carrying a bag.",
            {"gender": "male", "upper_clothing_type": "jacket_coat", "upper_clothing_color": "black", "bag": "yes"},
        ),
        (
            "A woman in a red coat holding an umbrella.",
            {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "red"},
        ),
        (
            "A man carrying a red coat and wearing a black jacket.",
            {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "black"},
        ),
        (
            "A woman carrying a black jacket and a red coat.",
            {"upper_clothing_type": "null", "upper_clothing_color": "null", "upper_clothing_length": "null"},
        ),
        (
            "A woman wearing a blue skirt and holding black pants.",
            {"lower_clothing_type": "skirt_dress", "lower_clothing_color": "blue"},
        ),
        (
            "A person wearing a white shirt and holding a long red coat.",
            {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "white", "upper_clothing_length": "null"},
        ),
        (
            "A person wearing a white shirt and a red coat held in one hand.",
            {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "white"},
        ),
    ],
)
def test_carried_garments_do_not_override_worn_garments(caption: str, expected: dict[str, str]) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected


@pytest.mark.parametrize("verb", ["carry", "carries", "carrying", "holding", "holds", "held"])
def test_carry_hold_forms_exclude_coat(verb: str) -> None:
    result = extract(f"A person wearing a white shirt and {verb} a red coat.")
    assert result["upper_clothing_type"] == "t_shirt_shirt"
    assert result["upper_clothing_color"] == "white"


@pytest.mark.parametrize(
    ("word", "gender"),
    [("girl", "female"), ("boy", "male"), ("guy", "male"),
     ("women", "female"), ("men", "male")],
)
def test_gender_aliases(word: str, gender: str) -> None:
    assert extract(word)["gender"] == gender


@pytest.mark.parametrize("word", ["bookbag", "book bag", "knapsack", "school bag", "schoolbag"])
def test_backpack_aliases(word: str) -> None:
    result = extract(f"A girl wearing a {word}.")
    assert result["gender"] == "female"
    assert result["backpack"] == "yes"
    assert result["bag"] == "null"


@pytest.mark.parametrize("word", ["leggings", "slacks", "capris", "tights", "capri", "khakis"])
def test_lower_clothing_aliases(word: str) -> None:
    assert extract(f"A woman wearing {word}.")["lower_clothing_type"] == "trousers_shorts"


@pytest.mark.parametrize("word", ["briefcase", "satchel", "tote"])
def test_bag_aliases(word: str) -> None:
    assert extract(f"A man holding a {word}.")["bag"] == "yes"


@pytest.mark.parametrize(
    ("caption", "slot", "color"),
    [
        ("navy shirt", "upper_clothing_color", "blue"),
        ("maroon shirt", "upper_clothing_color", "red"),
        ("burgundy jacket", "upper_clothing_color", "red"),
        ("silver jacket", "upper_clothing_color", "grey"),
        ("cream shirt", "upper_clothing_color", "white"),
        ("off-white shirt", "upper_clothing_color", "white"),
        ("tan pants", "lower_clothing_color", "brown"),
        ("beige pants", "lower_clothing_color", "brown"),
        ("khaki pants", "lower_clothing_color", "brown"),
        ("teal shirt", "upper_clothing_color", "blue"),
        ("turquoise shirt", "upper_clothing_color", "blue"),
        ("cyan shirt", "upper_clothing_color", "blue"),
        ("gold coat", "upper_clothing_color", "yellow"),
    ],
)
def test_color_aliases(caption: str, slot: str, color: str) -> None:
    assert extract(caption)[slot] == color


def test_multiple_canonical_colors_remain_unresolved() -> None:
    assert extract("red and black shirt")["upper_clothing_color"] == "null"


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        (
            "A man carrying a black jacket.",
            {"upper_clothing_type": "null", "upper_clothing_color": "null", "upper_clothing_length": "null"},
        ),
        (
            "A man carrying a bag and wearing a black jacket.",
            {"bag": "yes", "upper_clothing_type": "jacket_coat", "upper_clothing_color": "black"},
        ),
        (
            "She carries a backpack and has a red shirt on.",
            {"gender": "female", "backpack": "yes", "upper_clothing_type": "t_shirt_shirt",
             "upper_clothing_color": "red"},
        ),
        (
            "The person is carrying a teal bag and has on a pink shirt.",
            {"bag": "yes", "upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "pink"},
        ),
        (
            "A woman holding a red coat and wearing a white shirt.",
            {"gender": "female", "upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "white"},
        ),
        (
            "A man carrying a bag and has black pants on.",
            {"bag": "yes", "lower_clothing_type": "trousers_shorts", "lower_clothing_color": "black"},
        ),
        (
            "A woman holding a coat and had on a blue shirt.",
            {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "blue"},
        ),
    ],
)
def test_carried_scope_stops_at_new_wearing_cue(caption: str, expected: dict[str, str]) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        (
            "A man wearing a white dress shirt.",
            {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "white",
             "lower_clothing_type": "null"},
        ),
        (
            "A man wearing black dress shoes.",
            {"lower_clothing_type": "null"},
        ),
        (
            "A man wearing black dress pants.",
            {"lower_clothing_type": "trousers_shorts", "lower_clothing_color": "black"},
        ),
        (
            "A woman wearing a red dress.",
            {"lower_clothing_type": "skirt_dress", "lower_clothing_color": "red"},
        ),
        (
            "A man wearing white dress shirts.",
            {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "white",
             "lower_clothing_type": "null"},
        ),
        (
            "A man wearing black dress pant.",
            {"lower_clothing_type": "trousers_shorts", "lower_clothing_color": "black"},
        ),
        (
            "A man wearing dress shoe.",
            {"lower_clothing_type": "null"},
        ),
    ],
)
def test_dress_compounds_do_not_mean_a_dress(caption: str, expected: dict[str, str]) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("His trousers are black.", {"lower_clothing_type": "trousers_shorts", "lower_clothing_color": "black"}),
        ("His shirt is blue.", {"upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "blue"}),
        ("Her jacket was grey.", {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "grey"}),
        ("His shorts were purple.", {"lower_clothing_type": "trousers_shorts", "lower_clothing_color": "purple"}),
        ("The top is black.", {"upper_clothing_type": "null", "upper_clothing_color": "black"}),
        ("The pants are purple.", {"lower_clothing_type": "trousers_shorts", "lower_clothing_color": "purple"}),
        ("His shirt is blue and his trousers are black.",
         {"upper_clothing_color": "blue", "lower_clothing_color": "black"}),
        ("The man is wearing a grey jacket and a blue shirt.",
         {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "grey"}),
    ],
)
def test_copula_color_binds_to_its_garment(caption: str, expected: dict[str, str]) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected


@pytest.mark.parametrize(
    "caption",
    ["A young girl.", "A young boy.", "He is young.", "A teenage woman."],
)
def test_young_age_aliases(caption: str) -> None:
    assert extract(caption)["age"] == "young"


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("A woman with shoulder-length black hair.", "long"),
        ("A woman with shoulder length hair.", "long"),
        ("A man with medium-length hair.", "long"),
        ("A man with medium length brown hair.", "long"),
        ("A man with medium hair.", "long"),
        ("A man carrying a medium bag.", "null"),
        ("A woman with short black hair.", "short"),
        ("A man with long brown hair.", "long"),
    ],
)
def test_hair_length_requires_hair_context(caption: str, expected: str) -> None:
    assert extract(caption)["hair_length"] == expected


@pytest.mark.parametrize(
    ("word", "expected_type"),
    [
        ("parka", "jacket_coat"), ("windbreaker", "jacket_coat"), ("suit", "jacket_coat"),
        ("cardigan", "hoodie_sweater"), ("pullover", "hoodie_sweater"),
        ("polo", "t_shirt_shirt"), ("jersey", "t_shirt_shirt"),
    ],
)
def test_cross_dataset_clothing_aliases(word: str, expected_type: str) -> None:
    result = extract(f"A person wearing a black {word}.")
    assert result["upper_clothing_type"] == expected_type
    assert result["upper_clothing_color"] == "black"


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("The man in the suit, tie, and overcoat carried a black bag in his right hand.",
         {"upper_clothing_type": "jacket_coat", "bag": "yes"}),
        ("A man with a coat carried over his left arm.",
         {"upper_clothing_type": "null"}),
        ("A woman with a jacket held in her hand.",
         {"upper_clothing_type": "null"}),
        ("She is carrying a large purse that matches her grey shirt.",
         {"bag": "yes", "upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "grey"}),
        ("She is carrying a black purse which matches her grey shirt.",
         {"bag": "yes", "upper_clothing_type": "t_shirt_shirt", "upper_clothing_color": "grey"}),
    ],
)
def test_carried_scope_and_postposed_relation(caption: str, expected: dict[str, str]) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("A young adult with short black hair.", "adult"),
        ("The young adults are walking.", "adult"),
        ("A young middle-aged man.", "adult"),
        ("A young boy.", "young"),
        ("A young woman.", "young"),
        ("He is young.", "young"),
    ],
)
def test_specific_age_phrase_takes_precedence(caption: str, expected: str) -> None:
    assert extract(caption)["age"] == expected


@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        ("He is wearing black suit pants.",
         {"upper_clothing_type": "null", "lower_clothing_type": "trousers_shorts",
          "lower_clothing_color": "black"}),
        ("He wears grey suit trousers.",
         {"upper_clothing_type": "null", "lower_clothing_type": "trousers_shorts",
          "lower_clothing_color": "grey"}),
        ("He wears grey suit trouser.",
         {"upper_clothing_type": "null", "lower_clothing_type": "trousers_shorts",
          "lower_clothing_color": "grey"}),
        ("He wears a black suit.",
         {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "black"}),
        ("He is wearing a suit jacket.",
         {"upper_clothing_type": "jacket_coat"}),
        ("He wears a grey jacket and black suit trousers.",
         {"upper_clothing_type": "jacket_coat", "upper_clothing_color": "grey",
          "lower_clothing_type": "trousers_shorts", "lower_clothing_color": "black"}),
    ],
)
def test_suit_lower_compound_does_not_create_upper_garment(
    caption: str, expected: dict[str, str]
) -> None:
    result = extract(caption)
    assert {slot: result[slot] for slot in expected} == expected


@pytest.mark.parametrize(
    ("caption", "expected_color"),
    [
        ("He is wearing a black jacket. The sleeves of his jacket are white.", "black"),
        ("She wears a red coat. The hood of the coat is grey.", "red"),
        ("He is wearing a blue jacket. The fur on his jacket is white.", "blue"),
        ("The sleeves of his jacket are white.", "null"),
        ("The pockets of the jacket are red.", "null"),
        ("The white cuffs of his black jacket are red.", "black"),
        ("Her jacket is black.", "black"),
    ],
)
def test_garment_part_color_does_not_override_main_color(
    caption: str, expected_color: str
) -> None:
    result = extract(caption)
    assert result["upper_clothing_type"] == "jacket_coat"
    assert result["upper_clothing_color"] == expected_color


def _mentions(caption: str, slot: str) -> list[dict]:
    result = extract_with_provenance(caption)
    assert result["attributes"] == extract(caption)
    assert len(result["provenance"]) == 13
    for key, value in result["provenance"].items():
        if result["attributes"][key] == "null":
            assert value is None
            continue
        assert value["canonical"] == result["attributes"][key]
        for mention in value["mentions"]:
            assert caption[mention["start"]:mention["end"]] == mention["raw"]
            if "linked_object" in mention:
                linked = mention["linked_object"]
                assert caption[linked["start"]:linked["end"]] == linked["raw"]
    return result["provenance"][slot]["mentions"]


def test_provenance_preserves_alias_and_original_case() -> None:
    caption = "A woman in a Navy Blue jacket."
    mentions = _mentions(caption, "upper_clothing_color")
    assert [item["raw"] for item in mentions] == ["Navy Blue"]
    assert mentions[0]["linked_object"]["raw"] == "jacket"
    assert extract_with_provenance(caption)["provenance"]["upper_clothing_color"]["canonical"] == "blue"


def test_repeated_color_spans_are_bound_to_each_garment() -> None:
    caption = "A man wears a black jacket and black trousers."
    upper = _mentions(caption, "upper_clothing_color")
    lower = _mentions(caption, "lower_clothing_color")
    assert [(item["start"], item["raw"], item["linked_object"]["raw"]) for item in upper] == [
        (caption.index("black"), "black", "jacket")
    ]
    assert [(item["start"], item["raw"], item["linked_object"]["raw"]) for item in lower] == [
        (caption.rindex("black"), "black", "trousers")
    ]


def test_outer_garment_provenance_excludes_inner_shirt() -> None:
    caption = "A man wears a black jacket over a white shirt."
    color = _mentions(caption, "upper_clothing_color")
    kind = _mentions(caption, "upper_clothing_type")
    assert [(item["raw"], item["linked_object"]["raw"]) for item in color] == [
        ("black", "jacket")
    ]
    assert [item["raw"] for item in kind] == ["jacket"]


def test_postposed_color_and_repeated_same_canonical_mentions() -> None:
    caption = "He wears a black jacket. The jacket is black. His trousers are black."
    upper = _mentions(caption, "upper_clothing_color")
    lower = _mentions(caption, "lower_clothing_color")
    assert [item["start"] for item in upper] == [caption.index("black"), caption.index("black", 20)]
    assert [item["linked_object"]["raw"] for item in upper] == ["jacket", "jacket"]
    assert [item["raw"] for item in lower] == ["black"]
    assert lower[0]["linked_object"]["raw"] == "trousers"
    assert [item["raw"] for item in _mentions(caption, "lower_clothing_type")] == ["trousers"]


def test_age_pronoun_length_and_conflict_provenance() -> None:
    assert [item["raw"] for item in _mentions("A teenage woman.", "age")] == ["teenage"]
    assert [item["raw"] for item in _mentions("She is wearing a grey coat.", "gender")] == ["She"]
    caption = "A man in a long grey coat."
    assert _mentions(caption, "upper_clothing_length")[0]["linked_object"]["raw"] == "coat"
    assert _mentions(caption, "upper_clothing_color")[0]["linked_object"]["raw"] == "coat"
    result = extract_with_provenance("red and black shirt")
    assert result["attributes"]["upper_clothing_color"] == "null"
    assert result["provenance"]["upper_clothing_color"] is None


def test_unicode_lowercase_expansion_keeps_original_offsets() -> None:
    caption = "İ woman wearing a Navy Blue jacket."
    assert _mentions(caption, "gender")[0]["raw"] == "woman"
    color = _mentions(caption, "upper_clothing_color")[0]
    assert color["raw"] == "Navy Blue"
    assert color["start"] == caption.index("Navy Blue")


def test_synthetic_color_phrase_uses_real_source_mention() -> None:
    caption = "The dark shirt is blue."
    assert [item["raw"] for item in _mentions(caption, "upper_clothing_color")] == ["blue"]


@pytest.mark.parametrize(("caption", "slot", "canonical", "raw"), [
    ("A woman with medium-length black hair.", "hair_length", "long", "medium-length"),
    ("A man without a hat.", "hat", "no", "hat"),
    ("A woman with sunglasses and a backpack.", "glasses", "sunglasses", "sunglasses"),
    ("A woman with sunglasses and a backpack.", "backpack", "yes", "backpack"),
])
def test_direct_attribute_mentions_keep_raw_alias(
    caption: str, slot: str, canonical: str, raw: str
) -> None:
    result = extract_with_provenance(caption)
    assert result["provenance"][slot]["canonical"] == canonical
    assert [item["raw"] for item in _mentions(caption, slot)] == [raw]


@pytest.mark.parametrize("caption", [
    "without a hat",
    "A young adult with short black hair.",
    "He is wearing black suit pants.",
    "He wears a grey jacket and black suit trousers.",
    "He is wearing a black jacket. The sleeves of his jacket are white.",
    "A boy wearing a blue tee shirt with black shorts and carrying a black jacket.",
    "A woman carrying a black jacket and a red coat.",
    "A man with shoulder-length black hair and a navy blue parka.",
])
def test_provenance_api_preserves_frozen_attributes(caption: str) -> None:
    assert extract_with_provenance(caption)["attributes"] == extract(caption)


@pytest.mark.parametrize("extension", ["json", "jsonl"])
def test_batch_provenance_flag_keeps_default_output(extension: str) -> None:
    caption = "A woman in a Navy Blue jacket."
    extractor = TextAttributeExtractor()

    def run(with_provenance: bool) -> dict:
        argv = ["batch_extract.py", "--input", f"input.{extension}",
                "--output", f"output.{extension}"]
        if with_provenance:
            argv.append("--with-provenance")
        writer = mock_open()
        with (patch.object(sys, "argv", argv),
              patch("scripts.batch_extract.TextAttributeExtractor", return_value=extractor),
              patch.object(JsonCaptionAdapter, "iter_records", return_value=iter([CaptionRecord("p1", caption)])),
              patch.object(Path, "open", writer)):
            batch_main()
        data = json.loads("".join(call.args[0] for call in writer().write.call_args_list))
        return data[0] if extension == "json" else data

    plain = run(False)
    traced = run(True)
    assert list(plain) == ["id", "caption", "attributes"]
    assert {key: traced[key] for key in plain} == plain
    assert traced["provenance"]["upper_clothing_color"]["mentions"][0]["raw"] == "Navy Blue"
