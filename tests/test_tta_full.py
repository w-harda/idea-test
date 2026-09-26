"""完整 Vanilla TTA 的独立入口与六项对照校验。"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from attributes.tta_full import FullVanillaTTA


def test_full_tta_passes_original_pair_and_projects_budget():
    class Source:
        resolution = 2

    class Attacker:
        def attack(self, image, texts, ids, *, device, max_length, scales):
            assert texts == ["black pants"]
            assert ids == [0]
            assert device == image.device
            assert max_length == 77
            assert scales == (0.5, 0.75, 1.25, 1.5)
            return image + 0.1, ["black trousers"]

    attack = object.__new__(FullVanillaTTA)
    attack.source = Source()
    attack.attacker = Attacker()
    original = torch.full((1, 3, 2, 2), 0.5)
    image, text = attack(original, "black pants")
    assert text == "black trousers"
    assert (image - original).abs().max().item() == pytest.approx(8 / 255)
    assert original.unique().tolist() == [0.5]
    with pytest.raises(ValueError):
        attack(original, "")


def test_six_way_comparator_checks_the_exact_sample():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_tta_baselines.py"
    spec = spec_from_file_location("compare_tta_baselines", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    def row(index):
        return {
            "row_id": f"cuhk:{index}:0", "image": f"{index}.jpg",
            "caption": f"caption {index}", "person_id": index,
            "clean_rank": index + 1, "clean_soft_rank": float(index + 1),
        }

    attribute = {
        "queries": 3, "gallery_images": 29, "seed": 20260926, "tau": 0.077,
        "results": [
            {**row(i), "arms": {
                "tta_only": {"rank": i + 2, "soft_rank": float(i + 2)},
                "text_only": {"rank": i + 3, "soft_rank": float(i + 3)},
                "tta_text": {"rank": i + 4, "soft_rank": float(i + 4)},
            }} for i in range(3)
        ],
    }
    common = {
        "gallery_images": 29, "seed": 20260926, "tau": 0.077,
        "gallery_sha256": module.EXPECTED_GALLERY_SHA256,
    }
    vanilla = {**common, "results": [
        {**row(i), "vanilla_rank": i + 2, "vanilla_soft_rank": float(i + 2)}
        for i in range(3)]}
    full = {**common, "results": [
        {**row(i), "full_rank": i + 5, "full_soft_rank": float(i + 5),
         "linf": 8 / 255} for i in range(3)]}
    comparison = module.compare(attribute, vanilla, full)
    assert len(comparison["mean_ranks"]) == 6
    assert comparison["mean_ranks"]["vanilla_tta_full"] == 6
    assert comparison["results"][0]["ranks"]["vanilla_tta_full"] == 5

    full["results"][1]["caption"] = "different caption"
    with pytest.raises(ValueError, match="caption"):
        module.compare(attribute, vanilla, full)
