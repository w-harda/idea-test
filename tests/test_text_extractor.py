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
