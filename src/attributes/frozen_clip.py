"""冻结 OpenAI CLIP source surrogate，输入图像保持像素梯度。"""
from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FrozenCLIP(nn.Module):
    """直接从本地权重加载；不会下载或更新 source。"""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu"):
        super().__init__()
        import clip

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        model, _ = clip.load(str(path), device=device, jit=False)
        self.model = model.float().eval()
        self.model.requires_grad_(False)
        self._tokenize = clip.tokenize
        self.resolution = int(self.model.visual.input_resolution)
        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenCLIP":
        super().train(False)
        return self

    def _preprocess(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [N,3,H,W] RGB tensors")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("images must have finite [0,1] pixel values")
        height, width = images.shape[-2:]
        if min(height, width) < 1:
            raise ValueError("images must have nonempty spatial dimensions")
        scale = self.resolution / min(height, width)
        new_height, new_width = round(height * scale), round(width * scale)
        resized = F.interpolate(images, size=(new_height, new_width), mode="bicubic", align_corners=False)
        top = (new_height - self.resolution) // 2
        left = (new_width - self.resolution) // 2
        cropped = resized[:, :, top:top + self.resolution, left:left + self.resolution]
        return (cropped.clamp(0, 1) - self.mean) / self.std

    def encode_images(self, images: Tensor) -> Tensor:
        encoded = self.model.encode_image(self._preprocess(images))
        return F.normalize(encoded.float(), dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be a nonempty string sequence")
        tokens = self._tokenize(list(texts), truncate=True).to(self.mean.device)
        encoded = self.model.encode_text(tokens)
        return F.normalize(encoded.float(), dim=-1)
