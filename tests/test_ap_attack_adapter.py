"""AP-Attack 生成器预算、空间映射与逐属性回调。"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("kornia")

from attributes.ap_attack_adapter import (
    APAttackGenerator, APImageCallback, map_native_delta_to_clip,
)


class ConstantGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, normalized):
        self.seen.append(normalized.detach().clone())
        return torch.full_like(normalized, 0.1)


def test_ap_generator_uses_official_normalization_and_native_budget():
    attack = object.__new__(APAttackGenerator)
    attack.generator = ConstantGenerator()
    attack.device = torch.device("cpu")
    original = torch.full((1, 3, 256, 128), 0.25)
    attacked = attack.attack_native(original, original)
    assert torch.all(attack.generator.seen[0] == -0.5)
    assert (attacked - original).abs().max().item() == pytest.approx(8 / 255)
    assert torch.all(original == 0.25)


def test_ap_delta_maps_to_same_clip_clean_image_with_budget():
    native = torch.full((1, 3, 256, 128), 0.5)
    attacked = native + 8 / 255
    clean_clip = torch.full((1, 3, 224, 224), 0.5)
    mapped = map_native_delta_to_clip(
        native, attacked, clean_clip, (302, 113), 224)
    assert mapped.shape == clean_clip.shape
    assert (mapped - clean_clip).abs().max().item() <= 8 / 255 + 1e-6
    assert torch.all(clean_clip == 0.5)


def test_guided_ap_callback_repeats_image_only_generator_in_order():
    class FakeAttack:
        def __init__(self):
            self.inputs = []

        def attack_native(self, current, original):
            self.inputs.append(current.detach().clone())
            return (current + 4 / 255).clamp(
                min=original - 8 / 255, max=original + 8 / 255)

    attack = FakeAttack()
    native = torch.full((1, 3, 256, 128), 0.5)
    clip = torch.full((1, 3, 224, 224), 0.5)
    callback = APImageCallback(attack, native, clip, (302, 113), 224)
    state = SimpleNamespace(image=clip)
    for slot in ("hat", "lower_clothing_color"):
        request = SimpleNamespace(
            state=state, attribute=SimpleNamespace(slot=slot))
        result = callback(request)
        assert result.guidance["attribute_conditioned"] is False
        state = SimpleNamespace(image=result.image)
    assert callback.calls == 2
    assert not torch.equal(attack.inputs[0], attack.inputs[1])
    assert (state.image - clip).abs().max().item() <= 8 / 255 + 1e-6
