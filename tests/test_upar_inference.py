from types import SimpleNamespace

from attributes.upar_inference import _build_backbone_without_extra_weights


def test_backbone_construction_disables_only_extra_pretrained_download() -> None:
    calls = []

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return "constructed EVA"

    backbone = SimpleNamespace(timm=SimpleNamespace(create_model=create_model))
    original_timm = backbone.timm

    def construct(*, pretrained):
        assert pretrained is None
        return backbone.timm.create_model("eva_large_patch14_196.in22k_ft_in1k",
                                          pretrained=True, num_classes=0)

    backbone.swin_base_patch4_window7_224 = construct
    assert _build_backbone_without_extra_weights(backbone) == "constructed EVA"
    assert calls == [("eva_large_patch14_196.in22k_ft_in1k",
                      {"pretrained": False, "num_classes": 0})]
    assert backbone.timm is original_timm
