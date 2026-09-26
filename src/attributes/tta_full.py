"""官方 TTA 图文完整流程在 Frozen CLIP 上的最小接口适配。"""
from __future__ import annotations

import ast
import copy
import hashlib
import time
from pathlib import Path

import kornia.augmentation as K
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from .generator_framework import IMAGE_EPSILON, project_image
from .tta_adapter import TTA_SOURCE_SHA256, _CLIPSourceBridge


def load_official_full_classes(source_root: str | Path, glove_model):
    """按固定哈希装载官方 TTAttacker、Attack 和文本辅助定义。"""
    path = Path(source_root) / "attacker_TTA.py"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TTA_SOURCE_SHA256:
        raise ValueError("TTA attacker_TTA.py 与已验证官方版本不一致")
    tree = ast.parse(data, filename=str(path))
    names = {
        "TTAttacker", "Attack", "RWAug_Search", "select_op", "trace_prob",
        "get_substitues", "get_bpe_substitues",
    }
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    if {node.name for node in selected} != names:
        raise ValueError("TTA 官方图文攻击定义不完整")
    filter_words = next(
        ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "filter_words"
                for target in node.targets)
        and isinstance(node.value, ast.List)
    )
    scope = {
        "np": np, "torch": torch, "nn": nn, "F": F, "K": K,
        "transforms": transforms, "softmax": nn.Softmax(dim=0),
        "time": time, "copy": copy, "filter_words": set(filter_words),
        "glovemodel": glove_model,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["TTAttacker"], scope["Attack"]


class _FullCLIPBridge(_CLIPSourceBridge):
    """复现官方 CLIP 包装器：BERT IDs 解码后交给 CLIP tokenizer。"""

    def __init__(self, source, bert_tokenizer):
        super().__init__(source)
        self.bert_tokenizer = bert_tokenizer

    def inference_text(self, text_input):
        texts = [
            self.bert_tokenizer.decode(ids.tolist())
            .replace("[PAD]", "").replace("[CLS]", "")
            .replace("[SEP]", "").strip()
            for ids in text_input.input_ids
        ]
        return {"text_feat": self.source.encode_texts(texts)}


class FullVanillaTTA:
    """官方 Image_1 → Text_1 → Image_2；不接收任何属性记录。"""

    def __init__(self, source, official_root: str | Path,
                 bert_path: str | Path, glove_path: str | Path):
        from gensim.models import KeyedVectors
        from transformers import BertForMaskedLM, BertTokenizer

        bert_path = Path(bert_path)
        glove_path = Path(glove_path)
        if not bert_path.is_dir() or not glove_path.is_file():
            raise FileNotFoundError("local BERT and GloVe resources are required")
        self.source = source
        self.tokenizer = BertTokenizer.from_pretrained(
            str(bert_path), local_files_only=True)
        device = next(source.model.parameters()).device
        ref_net = BertForMaskedLM.from_pretrained(
            str(bert_path), local_files_only=True).to(device).eval()
        ref_net.requires_grad_(False)
        self.ref_net = ref_net
        glove_model = KeyedVectors.load(str(glove_path), mmap="r")
        self.glove_model = glove_model
        tta_class, attack_class = load_official_full_classes(official_root, glove_model)
        bridge = _FullCLIPBridge(source, self.tokenizer).eval()
        multi_attack = attack_class(
            ref_net, self.tokenizer, cls=False, max_length=77,
            number_perturbation=1, topk=10, threshold_pred_score=0.3,
            imgs_eps=IMAGE_EPSILON, step_size=2 / 255,
        )
        self.attacker = tta_class(bridge, multi_attack)

    def __call__(self, original: torch.Tensor, caption: str) -> tuple[torch.Tensor, str]:
        if not isinstance(caption, str) or not caption:
            raise ValueError("full TTA needs a nonempty original caption")
        if (original.ndim != 4 or original.shape[0] != 1 or original.shape[1] != 3
                or original.shape[-2:] != (self.source.resolution, self.source.resolution)
                or not original.is_floating_point() or not torch.isfinite(original).all()
                or original.min() < 0 or original.max() > 1):
            raise ValueError("full TTA expects one finite [0,1] CLIP-size image")
        with torch.enable_grad():
            image, texts = self.attacker.attack(
                original, [caption], [0], device=original.device,
                max_length=77, scales=(0.5, 0.75, 1.25, 1.5),
            )
        if not isinstance(texts, list) or len(texts) != 1 or not isinstance(texts[0], str):
            raise ValueError("official TTA returned invalid text")
        return project_image(original, image.detach()).detach(), texts[0]
