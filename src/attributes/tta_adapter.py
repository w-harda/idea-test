"""调用固定版本 TTA 官方 img_attack 的 Stage 05 图像回调。"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from math import isfinite

import kornia.augmentation as K
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .attack_scheduler import ImageAttackRequest, ImageAttackResult
from .generator_framework import IMAGE_EPSILON, project_image

# 官方仓库 YanGGGL/Transform_to_Transfer_Attack @ fe4f1ec
TTA_SOURCE_SHA256 = "1b28ecb608cfcf7d400c0c4616854a0af5f053fb3509a861899bdd518cacca78"


def load_official_attack(source_root: str | Path):
    """从官方文件装载原样 Attack 类及图像路径辅助函数。

    官方模块顶层强制读取 GloVe，虽然 img_attack 完全不用它；只编译
    需要的定义，避免要求文本攻击的 GloVe/BERT 资源。
    """
    path = Path(source_root) / "attacker_TTA.py"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TTA_SOURCE_SHA256:
        raise ValueError("TTA attacker_TTA.py 与已验证官方版本不一致")
    tree = ast.parse(data, filename=str(path))
    names = {"RWAug_Search", "select_op", "trace_prob", "Attack"}
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    if {node.name for node in selected} != names:
        raise ValueError("TTA 官方攻击定义不完整")
    scope = {
        "np": np, "torch": torch, "nn": nn, "F": F, "K": K,
        "transforms": transforms, "softmax": nn.Softmax(dim=0),
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["Attack"]


class _CLIPSourceBridge(nn.Module):
    """把 FrozenCLIP 映射到 TTA 的 inference_image/text 两个方法。"""

    def __init__(self, source):
        super().__init__()
        self.source = source

    def inference_image(self, normalized_images):
        features = self.source.model.encode_image(normalized_images)
        return {"image_feat": F.normalize(features.float(), dim=-1)}

    def inference_text(self, token_batch):
        return {"text_feat": self.source.encode_texts(token_batch.texts)}


class _TextBatch:
    def __init__(self, texts):
        self.texts = tuple(texts)

    def to(self, _device):
        return self


def _tokenize(texts, **_kwargs):
    return _TextBatch(texts)


class TTAImageCallback:
    """逐属性调用官方 img_attack，并约束相对同一原图的累计预算。"""

    def __init__(self, source, official_root: str | Path, original: torch.Tensor,
                 *, steps: int = 10, transforms_per_scale: int = 6,
                 step_size: float = 2 / 255,
                 scales: tuple[float, ...] | None = (0.5, 0.75, 1.25, 1.5)):
        if (steps < 1 or transforms_per_scale < 1 or not 0 < step_size <= IMAGE_EPSILON
                or (scales is not None and any(not isfinite(scale) or scale <= 0
                                               for scale in scales))):
            raise ValueError("invalid TTA attack parameters")
        if original.ndim != 4 or original.shape[0] != 1 or original.shape[1] != 3:
            raise ValueError("TTA callback expects one [1,3,H,W] image")
        if (not original.is_floating_point() or not torch.isfinite(original).all()
                or original.min() < 0 or original.max() > 1):
            raise ValueError("original image must have finite [0,1] pixels")
        if original.shape[-2:] != (source.resolution, source.resolution):
            raise ValueError("TTA callback image must match source CLIP resolution")
        self.original = original.detach().clone()
        self.bridge = _CLIPSourceBridge(source).eval()
        attack_type = load_official_attack(official_root)
        self.attacker = attack_type(None, _tokenize, imgs_eps=IMAGE_EPSILON,
                                    step_size=step_size)
        self.attacker.N_trans = transforms_per_scale
        self.momentum = torch.zeros_like(original)
        self.steps = steps
        self.scales = scales

    def __call__(self, request: ImageAttackRequest) -> ImageAttackResult:
        current = request.state.image
        if not isinstance(current, torch.Tensor) or current.shape != self.original.shape:
            raise ValueError("current image does not match original")
        if (current - self.original).abs().max() > IMAGE_EPSILON + 1e-6:
            raise ValueError("incoming image exceeds cumulative budget")
        target = request.state.targets[request.attribute.slot]["mentions"][0]
        word = target["raw"]
        with torch.enable_grad():
            proposed, momentum = self.attacker.img_attack(
                self.bridge, [request.state.text, word], current.detach(), self.original,
                [0, 0], self.steps, self.momentum, current.device,
                scales=self.scales,
            )
        image = project_image(self.original, proposed.detach()).detach()
        self.momentum = momentum.detach()
        return ImageAttackResult(image, guidance={"tta_steps": self.steps,
                                                  "attribute_word": word})


def vanilla_tta_image_attack(
    source, official_root: str | Path, original: torch.Tensor, caption: str,
) -> torch.Tensor:
    """官方两次图像更新，跳过其文本攻击；全程只用完整原始 caption。

    不接收 Stage 04 记录或 Stage 05 request。两次 img_attack 对应官方
    TTAttacker 的 Image_1 / Image_2，动量跨次传递，预算始终相对原图。
    """
    if not isinstance(caption, str) or not caption:
        raise ValueError("vanilla TTA needs a nonempty original caption")
    if (original.ndim != 4 or original.shape[0] != 1 or original.shape[1] != 3
            or original.shape[-2:] != (source.resolution, source.resolution)
            or not original.is_floating_point() or not torch.isfinite(original).all()
            or original.min() < 0 or original.max() > 1):
        raise ValueError("vanilla TTA expects one finite [0,1] CLIP-size image")
    attack_type = load_official_attack(official_root)
    attacker = attack_type(None, _tokenize, imgs_eps=IMAGE_EPSILON,
                           step_size=2 / 255)
    attacker.N_trans = 6
    bridge = _CLIPSourceBridge(source).eval()
    momentum = torch.zeros_like(original)
    current = original.detach().clone()
    for _ in range(2):
        with torch.enable_grad():
            proposed, momentum = attacker.img_attack(
                bridge, [caption], current, original, [0], 10, momentum,
                original.device, scales=(0.5, 0.75, 1.25, 1.5),
            )
        current = project_image(original, proposed.detach()).detach()
        momentum = momentum.detach()
    return current
