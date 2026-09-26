"""AP-Attack 生成器的图像推理与 Stage 05 最小回调。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision import transforms

from .attack_scheduler import ImageAttackRequest, ImageAttackResult
from .generator_framework import IMAGE_EPSILON, project_image

# Duke stage2 ReID + attribute-semantic 10/10 reproduction checkpoint.
AP_SEMANTIC_CHECKPOINT_SHA256 = "9dd2afe66afedc7ad17b43ebd14bfc2349e519bd74dbc5addacf5fad13434692"
AP_GENERATOR_SOURCE_SHA256 = "a5b5a8a5f3df8bf4137b95ed6e98a1ee2667a6f2b51cbcb4f4c82759546c9342"
AP_INPUT_SIZE = (256, 128)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class APAttackGenerator:
    """加载原项目 Generator；属性语义已编码于训练权重，推理输入仅是图像。"""

    def __init__(self, source_root: str | Path, checkpoint: str | Path,
                 device: torch.device):
        root = Path(source_root).resolve()
        path = Path(checkpoint).resolve()
        source_file = root / "advers" / "GD.py"
        if not source_file.is_file() or _sha256(source_file) != AP_GENERATOR_SOURCE_SHA256:
            raise ValueError("AP-Attack advers/GD.py missing or mismatched")
        if not path.is_file() or _sha256(path) != AP_SEMANTIC_CHECKPOINT_SHA256:
            raise ValueError("AP-Attack 10/10 semantic checkpoint missing or mismatched")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from advers.GD import Generator

        generator = Generator(3, 3, 32, norm="bn", beta=0.1)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("module."): value for key, value in state.items()}
        generator.load_state_dict(state, strict=True)
        self.generator = generator.to(device).eval()
        self.device = next(self.generator.parameters()).device

    def attack_native(self, current: torch.Tensor,
                      original: torch.Tensor) -> torch.Tensor:
        """原项目 [0.5,0.5,0.5] 归一化和像素 8/255 截断。"""
        if (current.shape != (1, 3, *AP_INPUT_SIZE)
                or original.shape != current.shape
                or current.device != self.device or original.device != self.device
                or not current.is_floating_point() or not original.is_floating_point()
                or not torch.isfinite(current).all() or not torch.isfinite(original).all()
                or current.min() < 0 or current.max() > 1
                or original.min() < 0 or original.max() > 1):
            raise ValueError("AP-Attack expects finite [0,1] 256x128 RGB image")
        with torch.no_grad():
            delta = self.generator((current - 0.5) / 0.5)
            if delta.shape != current.shape or not torch.isfinite(delta).all():
                raise ValueError("AP-Attack generator returned invalid delta")
            proposed = current + delta.clamp(-IMAGE_EPSILON, IMAGE_EPSILON)
            return project_image(original, proposed).detach()


def map_native_delta_to_clip(
    native_original: torch.Tensor, native_attacked: torch.Tensor,
    clean_clip: torch.Tensor, raw_size: tuple[int, int],
    clip_resolution: int,
) -> torch.Tensor:
    """把原图坐标的扰动送入同一 CLIP crop，再加到原先的干净 CLIP 图。"""
    height, width = raw_size
    if height < 1 or width < 1 or clean_clip.shape != (1, 3, clip_resolution, clip_resolution):
        raise ValueError("invalid raw image size or CLIP image")
    if native_attacked.shape != native_original.shape:
        raise ValueError("native attack shape mismatch")
    native_delta = native_attacked - native_original
    if native_delta.abs().max() > IMAGE_EPSILON + 1e-6:
        raise ValueError("native AP-Attack exceeds 8/255")
    raw_delta = F.interpolate(native_delta, size=raw_size, mode="bicubic",
                              align_corners=False, antialias=True)
    clip_transform = transforms.Compose([
        transforms.Resize(clip_resolution,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(clip_resolution),
    ])
    clip_delta = clip_transform(raw_delta).clamp(-IMAGE_EPSILON, IMAGE_EPSILON)
    return project_image(clean_clip, clean_clip + clip_delta).detach()


class APImageCallback:
    """每个 A* 属性调用一次图像生成器；生成器本身不接收属性。"""

    def __init__(self, attack: APAttackGenerator, native_original: torch.Tensor,
                 clean_clip: torch.Tensor, raw_size: tuple[int, int],
                 clip_resolution: int):
        self.attack = attack
        self.native_original = native_original.detach()
        self.native_current = native_original.detach().clone()
        self.clean_clip = clean_clip.detach()
        self.expected_clip = clean_clip.detach()
        self.raw_size = raw_size
        self.clip_resolution = clip_resolution
        self.calls = 0

    def __call__(self, request: ImageAttackRequest) -> ImageAttackResult:
        if (not isinstance(request.state.image, torch.Tensor)
                or request.state.image.shape != self.expected_clip.shape
                or not torch.allclose(request.state.image, self.expected_clip, atol=1e-6, rtol=0)):
            raise ValueError("AP-Attack callback image state out of sync")
        self.native_current = self.attack.attack_native(
            self.native_current, self.native_original)
        self.expected_clip = map_native_delta_to_clip(
            self.native_original, self.native_current, self.clean_clip,
            self.raw_size, self.clip_resolution)
        self.calls += 1
        return ImageAttackResult(
            self.expected_clip,
            guidance={"ap_generator_calls": self.calls,
                      "attribute_conditioned": False},
        )
