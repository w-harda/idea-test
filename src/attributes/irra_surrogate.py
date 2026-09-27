"""IRRA checkpoint as the frozen, differentiable attack source."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .irra_victim import FrozenIRRA, IMAGE_SIZE
from .rank_objective import soft_first_hit_rank


class IRRASurrogate(nn.Module):
    """Share one official IRRA model with the evaluator; keep input gradients."""

    def __init__(self, victim: FrozenIRRA, *, require_cuda: bool = True):
        super().__init__()
        self.victim = victim
        self.victim.eval().requires_grad_(False)
        self.image_size = IMAGE_SIZE
        if require_cuda and not next(self.victim.model.parameters()).is_cuda:
            raise RuntimeError("IRRA attack source requires CUDA")

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def pixels(self, image):
        return self.victim.pixels(image)

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        """Official IRRA test preprocessing followed by its image encoder."""
        return self.victim.encode_images(pixels)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        return self.victim.encode_texts(texts)

    def encode_normalized_images(self, normalized: torch.Tensor) -> torch.Tensor:
        """TTA already applies the same mean/std; do not normalize twice."""
        if normalized.ndim != 4 or normalized.shape[1:] != (3, *IMAGE_SIZE):
            raise ValueError("TTA IRRA input must be [N,3,384,128]")
        return F.normalize(self.victim.model.encode_image(normalized).float(), dim=-1)

    def inference_image(self, normalized_images: torch.Tensor):
        return {"image_feat": self.encode_normalized_images(normalized_images)}

    def inference_text(self, text_batch):
        if not hasattr(text_batch, "texts"):
            raise TypeError("TTA text batch must provide original strings")
        return {"text_feat": self.encode_texts(text_batch.texts)}

    @staticmethod
    def similarity(image_features: torch.Tensor,
                   text_features: torch.Tensor) -> torch.Tensor:
        """Official retrieval orientation: captions by gallery images."""
        return text_features @ image_features.T

    def retrieval_scores(self, texts: Sequence[str],
                         gallery_features: torch.Tensor) -> torch.Tensor:
        return self.similarity(gallery_features, self.encode_texts(texts))

    def soft_rank(self, texts: Sequence[str], gallery_features: torch.Tensor,
                  person_ids, gallery_ids, tau: float) -> torch.Tensor:
        return soft_first_hit_rank(
            self.retrieval_scores(texts, gallery_features),
            person_ids, gallery_ids, tau,
        )

    def attack_loss(self, pixels: torch.Tensor,
                    texts: Sequence[str]) -> torch.Tensor:
        """Differentiable negative paired similarity for gradient checks/future G."""
        images = self.encode_images(pixels)
        captions = self.encode_texts(texts)
        if images.shape != captions.shape:
            raise ValueError("paired image/text batch sizes differ")
        return -(images * captions).sum(dim=-1).mean()
