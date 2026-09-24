"""调用仓库外的官方 C2T-Net 模型，输出按官方顺序排列的 UPAR40 概率。"""

from __future__ import annotations

import sys
from types import ModuleType
from pathlib import Path
from typing import Sequence
from unittest.mock import patch


class Upar40Predictor:
    def __init__(self, source: str | Path, checkpoint: str | Path, device: str = "cuda"):
        import torch
        from PIL import Image  # noqa: F401 - 提前检查推理环境
        from torchvision import transforms

        source = Path(source).resolve()
        checkpoint = Path(checkpoint).resolve()
        if not (source / "models" / "backbone" / "swin_transformer2.py").is_file():
            raise FileNotFoundError(f"找不到官方 C2T-Net 模型源码: {source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 UPAR checkpoint: {checkpoint}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("PyTorch 未检测到 CUDA，无法执行 GPU 推理")
        # 官方模型源码留在独立目录，不进入本仓库。mmcv.runner 的导入只服务于
        # Swin 预初始化权重；本接口直接严格加载最终 UPAR checkpoint。
        sys.path.insert(0, str(source))
        mmcv = ModuleType("mmcv")
        mmcv.runner = ModuleType("mmcv.runner")
        def unused_pretrain_loader(*_args, **_kwargs):
            raise RuntimeError("推理不应加载额外的 Swin 预训练权重")
        mmcv.runner.load_checkpoint = unused_pretrain_loader
        with patch.dict(sys.modules, {"mmcv": mmcv, "mmcv.runner": mmcv.runner}):
            from models.backbone import swin_transformer2
        from models.base_block import FeatClassifier, LinearClassifier
        import timm

        original_create_model = timm.create_model
        def create_without_download(*args, **kwargs):
            kwargs["pretrained"] = False
            return original_create_model(*args, **kwargs)
        with patch.object(timm, "create_model", create_without_download):
            backbone = swin_transformer2.swin_base_patch4_window7_224(pretrained=None)
        # 官方测试阶段推理脚本显式设为 2048，而非 model_factory 中的 1024。
        classifier = LinearClassifier(
            nattr=40, c_in=2048, bn=False, pool="avg", scale=1)
        model = FeatClassifier(backbone, classifier)
        # 官方 checkpoint 的 metric 等元数据含 NumPy scalar；只允许这些
        # 数值类型参与反序列化，仍保持 weights_only 的受限加载。
        import numpy as np
        numeric_dtypes = ("float16", "float32", "float64", "int32", "int64")
        safe_types = [np.core.multiarray.scalar, np.dtype]
        safe_types.extend({type(np.dtype(name)) for name in numeric_dtypes})
        with torch.serialization.safe_globals(safe_types):
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = saved.get("state_dicts", saved) if isinstance(saved, dict) else saved
        if not isinstance(state, dict) or not state:
            raise ValueError("checkpoint 缺少模型 state_dicts")
        if all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.torch = torch
        # 与官方 upar.yaml 的 HEIGHT/WIDTH 一致；无裁剪，保留行人全身。
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def predict_paths(self, paths: Sequence[Path]) -> list[list[float]]:
        from PIL import Image

        if not paths:
            return []
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        batch = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(batch)
            logits = outputs[0][0]
            if logits.ndim != 2 or logits.shape != (len(paths), 40):
                raise ValueError(f"官方模型输出形状不是 [batch, 40]: {tuple(logits.shape)}")
            probabilities = self.torch.sigmoid(logits)
        return probabilities.cpu().tolist()
