"""Frozen official CUHK-PEDES IRRA retrieval evaluator."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Sequence

import torch
import yaml
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

OFFICIAL_CUHK_SHA256 = "9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"
IMAGE_SIZE = (384, 128)
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenIRRA(nn.Module):
    """Use the untouched official model code and exact test preprocessing."""

    def __init__(self, repo: str | Path, checkpoint: str | Path,
                 config: str | Path, device: str = "cuda"):
        super().__init__()
        root = Path(repo).resolve()
        path = Path(checkpoint).resolve()
        cfg_path = Path(config).resolve()
        if file_sha256(path) != OFFICIAL_CUHK_SHA256:
            raise ValueError("IRRA checkpoint SHA-256 differs from fixed official CUHK model")
        with cfg_path.open(encoding="utf-8") as handle:
            settings = yaml.load(handle, Loader=yaml.FullLoader)
        expected = {
            "dataset_name": "CUHK-PEDES",
            "pretrain_choice": "ViT-B/16",
            "img_size": IMAGE_SIZE,
            "text_length": 77,
            "stride_size": 16,
            "cmt_depth": 4,
            "vocab_size": 49408,
            "loss_names": "sdm+mlm+id",
        }
        for key, value in expected.items():
            if settings.get(key) != value:
                raise ValueError(f"IRRA config {key} differs from verified checkpoint")
        if not (root / "model" / "build.py").is_file():
            raise FileNotFoundError(root / "model" / "build.py")
        sys.path.insert(0, str(root))
        from model import build_model
        from utils.simple_tokenizer import SimpleTokenizer

        self.tokenizer = SimpleTokenizer()
        self.text_length = 77
        model = build_model(SimpleNamespace(**settings), num_classes=11003)
        state = torch.load(path, map_location="cpu", weights_only=True)["model"]
        result = model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("IRRA state_dict mismatch")
        self.model = model.eval().to(device)
        self.model.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        self.pil_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.to(device)
        if next(self.model.parameters()).device.type != "cuda" or not self.mean.is_cuda:
            raise RuntimeError("IRRA victim must be on CUDA")

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def pixels(self, image: Image.Image) -> torch.Tensor:
        return self.pil_transform(image.convert("RGB"))

    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1:] != (3, *IMAGE_SIZE):
            raise ValueError("IRRA images must have shape [N,3,384,128]")
        normalized = (pixels - self.mean) / self.std
        return F.normalize(self.model.encode_image(normalized).float(), dim=-1)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("IRRA needs nonempty texts")
        sot = self.tokenizer.encoder["<|startoftext|>"]
        eot = self.tokenizer.encoder["<|endoftext|>"]
        rows = []
        for caption in texts:
            ids = [sot] + self.tokenizer.encode(caption) + [eot]
            if len(ids) > self.text_length:
                ids = ids[:self.text_length]
                ids[-1] = eot
            ids += [0] * (self.text_length - len(ids))
            rows.append(ids)
        tokens = torch.tensor(rows, device=self.mean.device, dtype=torch.long)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)
