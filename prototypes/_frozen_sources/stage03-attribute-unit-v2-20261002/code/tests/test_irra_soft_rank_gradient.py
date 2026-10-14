"""paired soft-rank符号、图连接、冻结参数与相关性口径。"""
import pytest
import torch

from attributes.irra_soft_rank_gradient import (
    IRRAPairedSoftRankGradientScorer, gradient_times_input, paired_soft_rank, within_caption_spearman,
)
from attributes.text_extractor import extract_with_provenance


def test_paired_soft_rank_uses_every_other_image_and_self_constant():
    scores = torch.tensor([0.3, 0.5, 0.1], requires_grad=True)
    rank = paired_soft_rank(scores, 0, .2)
    expected = 1 + torch.sigmoid(torch.tensor(1.)) + torch.sigmoid(torch.tensor(-1.))
    assert torch.allclose(rank, expected)
    full_sum = 1 + torch.sigmoid((scores - scores[0]) / .2).sum()
    assert torch.allclose(full_sum, rank + .5)
    gradient, = torch.autograd.grad(rank, scores, retain_graph=True)
    full_gradient, = torch.autograd.grad(full_sum, scores)
    assert torch.allclose(gradient, full_gradient)
    assert gradient[0] < 0 and gradient[1] > 0 and gradient[2] > 0
    assert torch.allclose(gradient.sum(), torch.tensor(0.))


def test_single_image_soft_rank_is_one_and_gradient_zero():
    scores = torch.tensor([.4], requires_grad=True)
    rank = paired_soft_rank(scores, 0, .02)
    assert rank.item() == 1
    assert torch.autograd.grad(rank, scores)[0].item() == 0


def test_signed_gradient_times_input_matches_local_removal_direction():
    embedding = torch.tensor([[[2., -1.], [1., 3.]]], requires_grad=True)
    coefficients = torch.tensor([[[3., 1.], [-2., 1.]]])
    rank = (embedding * coefficients).sum()
    token_scores, gradient = gradient_times_input(rank, embedding)
    assert torch.equal(gradient, coefficients)
    assert torch.equal(token_scores, torch.tensor([[-5., -1.]]))
    epsilon = .01
    altered = ((embedding.detach() * (1-epsilon)) * coefficients).sum()
    assert torch.allclose((altered-rank.detach()) / epsilon, token_scores.sum(), atol=1e-4)


def test_paired_similarity_must_remain_in_gradient_graph():
    # 提高paired分数会降低soft rank，不允许把s_plus detach导致导数丢失。
    embedding = torch.tensor([[[1., 0.]]], requires_grad=True)
    gallery = torch.tensor([[1., 0.], [0., 1.]])
    rank = paired_soft_rank((embedding[0] @ gallery.T)[0], 0, .5)
    scores, gradient = gradient_times_input(rank, embedding)
    assert gradient[0, 0, 0] < 0 and scores[0, 0] > 0


@pytest.mark.parametrize("tau", [0., -1., float('inf'), float('nan'), True])
def test_invalid_temperature_rejected(tau):
    with pytest.raises(ValueError):
        paired_soft_rank(torch.tensor([1., 0.]), 0, tau)


@pytest.mark.parametrize("gradient,rank,expected", [
    ([1., 2.], [0., 1.], 1.), ([2., 1.], [0., 1.], -1.),
    ([1., 2., 3.], [0., 0., 1.], 3**.5/2),
])
def test_within_caption_spearman_uses_average_tie_ranks(gradient, rank, expected):
    assert within_caption_spearman(gradient, rank)["rho"] == pytest.approx(expected)


def test_constant_rank_differences_are_undefined_not_zero():
    result = within_caption_spearman([1., 2., 3., 4.], [0., 0., 0., 0.])
    assert result["rho"] is None and result["undefined_reason"] == "constant_rank_scores"


def test_constant_gradient_scores_and_insufficient_samples_are_undefined():
    assert within_caption_spearman([1., 1.], [0., 1.])["undefined_reason"] == "constant_gradient_scores"
    assert within_caption_spearman([1.], [0.])["undefined_reason"] == "fewer_than_two_attributes"


def test_correlation_rejects_bad_lengths_or_nan():
    with pytest.raises(ValueError):
        within_caption_spearman([1.], [])
    with pytest.raises(ValueError):
        within_caption_spearman([float('nan'), 1.], [0., 1.])


class ToyVictim(torch.nn.Module):
    def __init__(self):
        from test_irra_token_alignment import FakeOfficialTokenizer
        super().__init__()
        self.tokenizer = FakeOfficialTokenizer()
        self.text_length = 77
        self.model = torch.nn.Module()
        self.model.base_model = torch.nn.Module()
        torch.manual_seed(5)
        self.model.base_model.token_embedding = torch.nn.Embedding(49408, 3)
        self.eval().requires_grad_(False)

    def encode_texts(self, texts):
        from attributes.irra_token_alignment import align_irra_tokens
        alignment = align_irra_tokens(texts[0], self.tokenizer)
        ids = torch.tensor([alignment["token_ids"]])
        embedding = self.model.base_model.token_embedding(ids)
        value = embedding[:, :alignment["eot_index"] + 1].sum(dim=1)
        return torch.nn.functional.normalize(value, dim=-1)


def test_frozen_embedding_hook_scores_invalid_removal_type_and_cleans_up():
    victim = ToyVictim()
    gallery = torch.eye(3)
    scorer = IRRAPairedSoftRankGradientScorer(victim, ["paired", "other", "third"], gallery)
    gallery.zero_()  # scorer必须保留自己的固定图库快照
    text = "A woman wearing a black shirt."
    result = scorer.score(text, "paired", **extract_with_provenance(text), tau=.2)
    scores = result["scores_by_attribute"]
    assert set(scores) == {"gender=female", "upper_clothing_color=black", "upper_clothing_type=t_shirt_shirt"}
    assert result["diagnostics"]["gradient_norm"] > 0
    assert result["diagnostics"]["baseline_feature_max_abs_difference"] == 0
    assert result["diagnostics"]["after_eot_gradient_max_abs"] == 0
    assert all(p.grad is None and not p.requires_grad for p in victim.parameters())
    assert not victim.model.base_model.token_embedding._forward_hooks


def test_shared_evidence_gets_same_attribution_with_structured_keys():
    victim = ToyVictim()
    scorer = IRRAPairedSoftRankGradientScorer(victim, ["paired", "other", "third"], torch.eye(3))
    text = "short dark hair blue shirt"
    result = scorer.score(text, "paired", **extract_with_provenance(text), tau=.2)
    rows = {row["slot"]: row for row in result["attribute_results"]}
    assert rows["hair_length"]["S_grad"] == rows["upper_clothing_length"]["S_grad"]
    assert rows["hair_length"]["shared_token_indices"]


def test_hook_is_removed_when_official_ids_do_not_match():
    victim = ToyVictim()
    scorer = IRRAPairedSoftRankGradientScorer(victim, ["paired"], torch.tensor([[1., 0., 0.]]))
    with pytest.raises(RuntimeError, match="official token IDs"):
        scorer._encode("A man.", [0] * 77)
    assert not victim.model.base_model.token_embedding._forward_hooks
