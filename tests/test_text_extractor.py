import pytest

from attributes import TextAttributeExtractor, extract


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
