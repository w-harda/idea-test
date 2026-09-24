import pytest

from attributes.visual_canonicalizer import SLOTS, UPAR40_NAMES, VisualCanonicalizer


def test_upar40_order_and_slot_shape() -> None:
    assert len(UPAR40_NAMES) == 40
    assert len(set(UPAR40_NAMES)) == 40
    result = VisualCanonicalizer().canonicalize([0.5] * 40)
    assert tuple(result) == SLOTS
    assert set(result.values()) == {"null"}


def test_confident_predictions_map_without_inventing_upper_type() -> None:
    probabilities = dict.fromkeys(UPAR40_NAMES, 0.1)
    probabilities.update({
        "Age-Adult": 0.92,
        "Gender-Female": 0.9,
        "Hair-Length-Long": 0.88,
        "UpperBody-Length-Short": 0.85,
        "UpperBody-Color-Red": 0.91,
        "LowerBody-Length-Short": 0.1,
        "LowerBody-Color-Blue": 0.84,
        "LowerBody-Type-Skirt&Dress": 0.9,
        "Accessory-Backpack": 0.82,
        "Accessory-Bag": 0.12,
        "Accessory-Glasses-Normal": 0.86,
        "Accessory-Hat": 0.15,
    })
    result = VisualCanonicalizer().canonicalize(
        [probabilities[name] for name in UPAR40_NAMES])
    assert result == {
        "age": "adult", "gender": "female", "hair_length": "long",
        "upper_clothing_length": "short", "upper_clothing_color": "red",
        "upper_clothing_type": "null", "lower_clothing_length": "long",
        "lower_clothing_color": "blue", "lower_clothing_type": "skirt_dress",
        "backpack": "yes", "bag": "no", "glasses": "normal_glasses", "hat": "no",
    }


def test_conflicts_and_other_abstain() -> None:
    probabilities = dict.fromkeys(UPAR40_NAMES, 0.5)
    probabilities.update({
        "Age-Young": 0.85, "Age-Adult": 0.81,
        "Hair-Length-Short": 0.9, "Hair-Length-Long": 0.8,
        "UpperBody-Color-Red": 0.9, "UpperBody-Color-Blue": 0.8,
        "LowerBody-Color-Black": 0.88, "LowerBody-Color-Other": 0.9,
        "LowerBody-Type-Trousers&Shorts": 0.9,
        "LowerBody-Type-Skirt&Dress": 0.8,
        "Accessory-Glasses-Normal": 0.9, "Accessory-Glasses-Sun": 0.85,
    })
    result = VisualCanonicalizer().canonicalize(
        [probabilities[name] for name in UPAR40_NAMES])
    for slot in ("age", "hair_length", "upper_clothing_color",
                 "lower_clothing_color", "lower_clothing_type", "glasses"):
        assert result[slot] == "null"


def test_glasses_absent_only_when_both_probs_low() -> None:
    values = dict.fromkeys(UPAR40_NAMES, 0.5)
    values["Accessory-Glasses-Normal"] = 0.1
    values["Accessory-Glasses-Sun"] = 0.2
    result = VisualCanonicalizer().canonicalize([values[name] for name in UPAR40_NAMES])
    assert result["glasses"] == "no_glasses"


@pytest.mark.parametrize("invalid", [[0.5] * 39, [0.5] * 39 + [float("nan")],
                                     [0.5] * 39 + [1.1]])
def test_invalid_probabilities_rejected(invalid: list[float]) -> None:
    with pytest.raises(ValueError):
        VisualCanonicalizer().canonicalize(invalid)
