"""检验support重叠、未解析边界与等预算一阶预测，不加载真实模型。"""
import pytest
import torch

from attributes.irra_attribute_unit_prototype import AttributeUnitProbe, contextual_supports, projected_gradient
from attributes.irra_token_alignment import align_irra_tokens
from attributes.text_extractor import extract_with_provenance
from test_irra_token_alignment import FakeOfficialTokenizer


def supports(text, length=77, tokenizer=None):
    alignment = align_irra_tokens(text, tokenizer or FakeOfficialTokenizer(), length)
    rows, pairs = contextual_supports(text, extract_with_provenance(text), alignment)
    return {r['slot']: r for r in rows}, pairs


def test_two_short_occurrences_include_different_clothing_binding():
    rows, _ = supports('A man with short hair wearing a short-sleeved black shirt.')
    hair, clothes = rows['hair_length'], rows['upper_clothing_length']
    assert hair['support_token_indices'] != clothes['support_token_indices']
    assert set(hair['evidence_token_indices']).isdisjoint(clothes['evidence_token_indices'])
    assert clothes['object_token_indices']


def test_shared_color_evidence_is_distinguished_by_original_objects():
    text = 'The jacket is solid black as well as the pants.'
    rows, pairs = supports(text)
    a, b = rows['upper_clothing_color'], rows['lower_clothing_color']
    assert a['evidence_token_indices'] == b['evidence_token_indices']
    assert a['support_token_indices'] != b['support_token_indices']
    pair = next(p for p in pairs if {p['left'], p['right']} ==
                {'upper_clothing_color', 'lower_clothing_color'})
    assert pair['shared_evidence'] and not pair['identical_support']


def test_object_token_is_binary_union_not_counted_twice():
    rows, _ = supports('A man wearing a black shirt.')
    color, kind = rows['upper_clothing_color'], rows['upper_clothing_type']
    assert kind['support_token_indices'] == color['object_token_indices']
    assert len(color['support_token_indices']) == len(set(color['support_token_indices']))


def test_truncated_object_keeps_no_partial_support():
    rows, _ = supports('person person person black shirt', length=6)
    assert rows['upper_clothing_color']['support_status'] == 'unresolved'
    assert rows['upper_clothing_color']['support_token_indices'] == []


def test_true_multi_subword_evidence_keeps_all_pieces():
    rows, _ = supports('A man carrying a schoolbag.',
                       tokenizer=FakeOfficialTokenizer({'schoolbag': [6]}))
    assert len(rows['backpack']['evidence_token_indices']) == 2


def test_same_budget_for_different_support_sizes_and_quadratic_prediction():
    h = torch.arange(1., 16., dtype=torch.float64).reshape(5, 1, 3).requires_grad_(True)
    objective = (h.square()).sum()
    g, = torch.autograd.grad(objective, h)
    epsilon = .03
    for indices in ([0], [1, 2, 3]):
        score, delta = projected_gradient(g, indices, epsilon)
        assert delta.norm().item() == pytest.approx(epsilon, rel=1e-6)
        actual = ((h.detach() + delta).square().sum() - objective.detach()).item()
        # 二次目标的余项应为epsilon²，检查方向/预算/一阶预测同时正确。
        assert actual == pytest.approx(epsilon * score + epsilon**2, abs=1e-6)


def test_zero_gradient_has_no_invented_direction():
    score, delta = projected_gradient(torch.zeros(5, 1, 3), [1, 2], .01)
    assert score == 0 and delta is None


def test_fp16_effective_budget_can_differ_from_constructed_budget():
    # 真实加法必须审计；数学上规范化delta不保证低精度H发生同样大小的变化。
    hidden = torch.full((2, 1, 3), 100., dtype=torch.float16)
    _, delta = projected_gradient(torch.ones_like(hidden), [0], .001)
    assert delta.norm().item() == pytest.approx(.001)
    effective = (hidden.float() + delta).half().double() - hidden.double()
    assert effective.norm().item() == 0


def test_probe_continues_original_graph_preserves_zero_and_frozen_parameters():
    class MixingBlock(torch.nn.Module):
        def forward(self, x):
            return x + .1 * torch.cumsum(x, dim=0)

    class ToyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer = FakeOfficialTokenizer()
            self.text_length = 77
            self.model = torch.nn.Module()
            self.model.base_model = torch.nn.Module()
            self.model.base_model.token_embedding = torch.nn.Embedding(49408, 3)
            self.model.base_model.transformer = torch.nn.Module()
            self.model.base_model.transformer.resblocks = torch.nn.Sequential(*[MixingBlock() for _ in range(12)])
            self.eval().requires_grad_(False)

        def encode_texts(self, texts):
            aligned = align_irra_tokens(texts[0], self.tokenizer)
            ids = torch.tensor([aligned['token_ids']])
            hidden = self.model.base_model.token_embedding(ids).permute(1, 0, 2)
            hidden = self.model.base_model.transformer.resblocks(hidden)
            return torch.nn.functional.normalize(hidden[aligned['eot_index']], dim=-1)

    torch.manual_seed(4)
    victim = ToyVictim()
    gallery = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    probe = AttributeUnitProbe(victim, gallery, 0)
    result = probe.run('A man wearing a black shirt.', epsilons=(.001,))
    assert result['diagnostics']['zero_intervention_exact']
    assert result['diagnostics']['parameters_frozen']
    assert all(p.grad is None and not p.requires_grad for p in victim.parameters())
    assert all(not b._forward_hooks for b in victim.model.base_model.transformer.resblocks)
    for row in result['attribute_results']:
        check = row['interventions'][0]
        assert check['effective_delta_norm'] == pytest.approx(.001, rel=3e-3)
        assert check['central_relative_error'] < .02
