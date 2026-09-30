"""AP-Attack CUHK G：只将 IDE 特征分支替换为冻结的 IRRA 图像特征。"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .ap_cuhk_training import (
    EPSILON, TrainImages, image_path, make_loader,
)
from .irra_victim import IMAGE_SIZE


class DualTrainImages(TrainImages):
    """同一 train 图像的 AP/Baseline 2 与 IRRA 官方 clean 像素视图。"""

    def __init__(self, records, root, ap_transform):
        super().__init__(records, root, ap_transform)
        self.irra_transform = T.Compose([
            T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(image_path(self.root, row.path)) as image:
            rgb = image.convert("RGB")
            return self.transform(rgb), self.irra_transform(rgb), row.person_id


def make_dual_loader(records, image_root: Path, ap_source: Path, base_config,
                     *, batch_size=None, workers=None):
    """复用 Baseline 2 的 native transform 和 P×K sampler。"""
    native_loader = make_loader(
        records, image_root, ap_source, "generator", base_config,
        batch_size=batch_size, workers=workers)
    return DataLoader(
        DualTrainImages(records, image_root, native_loader.dataset.transform),
        batch_size=native_loader.batch_size, sampler=native_loader.sampler,
        shuffle=False, num_workers=native_loader.num_workers,
        drop_last=native_loader.drop_last, pin_memory=native_loader.pin_memory,
        generator=native_loader.generator,
        worker_init_fn=native_loader.worker_init_fn,
    )


def transport_for_training(clean_ap: torch.Tensor, adv_ap: torch.Tensor,
                           clean_irra: torch.Tensor):
    """Baseline 1 像素映射的可微形式；绝不 detach 对抗图或 resize 结果。"""
    if (clean_ap.ndim != 4 or clean_ap.shape != adv_ap.shape
            or clean_ap.shape[1:] != (3, 256, 128)
            or clean_irra.shape != (len(clean_ap), 3, *IMAGE_SIZE)):
        raise ValueError("AP/IRRA 训练像素尺寸不匹配")
    native_clean = (clean_ap + 1.) * .5
    native_adv = (adv_ap + 1.) * .5
    delta = native_adv - native_clean
    if not all(torch.isfinite(x).all() for x in (native_clean, native_adv, clean_irra, delta)):
        raise ValueError("AP/IRRA 训练像素非有限")
    if float(delta.abs().amax()) > EPSILON + 1e-6:
        raise ValueError("AP native 扰动超过 8/255")
    mapped = F.interpolate(delta, size=IMAGE_SIZE, mode="bilinear",
                           align_corners=False)
    if float(mapped.abs().amax()) > EPSILON + 1e-6:
        raise ValueError("AP→IRRA resize 放大扰动")
    adv_irra = (clean_irra + mapped).clamp(0, 1)
    return native_clean, native_adv, adv_irra


def ap_irra_losses(clean_ap: torch.Tensor, adv_ap: torch.Tensor,
                   clean_irra: torch.Tensor, ids: torch.Tensor,
                   irra: nn.Module, semantic: nn.Module, original):
    """固定同一 clean/adv/PID，semantic 精确复用 Baseline 2 公式。"""
    native_clean, native_adv, adv_irra = transport_for_training(
        clean_ap, adv_ap, clean_irra)
    triplet = original["adv_TripletLoss"](.3)
    with torch.no_grad():
        clean_feature = irra.encode_images(clean_irra)
    adv_feature = irra.encode_images(adv_irra)
    irra_loss = 10 * triplet(clean_feature.detach(), adv_feature, ids)
    clean_tokens = semantic(clean_ap, ids)[-1]
    adv_tokens = semantic(adv_ap, ids)[-1]
    semantic_loss = 10 * torch.stack([
        triplet(c.detach(), a, ids)
        for c, a in zip(clean_tokens.chunk(5), adv_tokens.chunk(5))
    ]).sum()
    return irra_loss + semantic_loss, irra_loss, semantic_loss, {
        "clean_irra_feature": clean_feature,
        "adv_irra_feature": adv_feature,
        "native_clean": native_clean,
        "native_adv": native_adv,
        "adv_irra": adv_irra,
    }


def ide_loss_on_same_images(clean_ap, adv_ap, ids, ide, original):
    """仅用于 smoke：与 IRRA 分支共用同一 AP clean/adv/PID。"""
    transform = T.Compose([
        T.Normalize([-1.] * 3, [2.] * 3),
        T.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    clean_feature = ide(transform(clean_ap))
    adv_feature = ide(transform(adv_ap))
    return 10 * original["adv_TripletLoss"](.3)(
        clean_feature.detach(), adv_feature, ids)
