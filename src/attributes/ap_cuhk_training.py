"""AP-Attack CUHK train-only 适配；训练图中没有 TBPS victim 或 caption。"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .ap_gallery_baseline import (
    EPSILON, SOURCE_HASHES, atomic_json, atomic_torch, fingerprint, image_path,
    sha256, write_path,
)

AP_SOURCE = Path("/home/lzf/ldx/projects/AP-Attack")
OUTPUT = Path("/home/lzf/ldx/outputs/AP-Attack/cuhk_reid_semantic_10_10")
ANNOTATION = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGE_ROOT = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")
CLIP_REID = Path("/home/lzf/ldx/checkpoints/AP-Attack/clipreid/Duke_clipreid_ViT-B-16_60.pth")
CLIP_REID_SHA = "d61a156497e73e451dba43697eaedd5e4cf6b714e0ce9e34b4ababc1313e3c53"
RESNET_SHA = "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a"
TRAIN_HASHES = {
    **{f"advers/{name}": value for name, value in SOURCE_HASHES.items()},
    "train_ap_attack.py": "bc9ff2071f45f5b2de66b4f20686140a0fce51a0d74ba6b1e3bb628a5ff69cae",
    "tools/train_ide_duke.py": "e37f4c353c2dda97c07660864ebed9c093e38a5cc17700449360b44f58b429e5",
    "model/make_model_clipreid.py": "a2b3f8e36ffd9673e4d602e4b9684fabd6e256e056d4385ab4c6bc5523d1e82f",
    "model/clip/model.py": "957887b17b8861108c96bcffc794a056af061da57e31eee8d8ce802e0181844c",
    "model/clip/clip.py": "cf21f9727141ec3c970e4064edd3edd3701a1276a9b1a2c141fd222a83c368ac",
    "model/clip/simple_tokenizer.py": "d1b09f10ed3d1e2c343619cb98cbfacb92363dd8607526a43aa7fe2d60d15bad",
    "model/clip/bpe_simple_vocab_16e6.txt.gz": "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",
    "datasets/sampler.py": "9616ea253ce9e3c691c1abc198921e33733a0d59c0a34eee97e2e7e7d433e6bb",
    "loss/supcontrast.py": "a0288ebdecbe65d193cc808fde7b2b4c839a278bf2b6369bf1a7f235b447bfa6",
    "solver/scheduler.py": "b7517bb0f79a3334e2a3ac5ecb14bdcf69cb28c62dd3c734c4e3d212d0955228",
    "solver/cosine_lr.py": "d3db6e30c1666c5af71200086f33c644d476310b6c5dc7ed98097cc989bdad4d",
    "solver/scheduler_factory.py": "8c18fd8c468205ed929ee6c841f88e43647ad9d9bc068b5b9c3f313648284b2b",
    "solver/make_optimizer_prompt.py": "fa0bba8c38b87d273cedb94a6b9383c240cfe676c705b024f9c693a437c81bf3",
    "config/defaults.py": "74c913dfd041e113cd3630745fa3b5a4d04c0235c137658480fcc268efa4d32f",
    "configs/att/vit_clipreid_att_reid_semantic_10_10.yml": "81bacc57b8063c8faacddbf8c635bbec048ac7ac99e7d84ced167771fa68654d",
}
TRAIN_HASHES.update({'datasets/make_dataloader_clipreid.py': '835f643d03789d6544c89141ffcb7386a43b4ba0b1d4e5ba10e1a76b2780a887', 'processor/processor_clipreid_att.py': 'a4c14aab9408eaf6995f04c55a026a94142be8fcc495f40fc402b7b18fef64f4', 'train_att_model.py': '0b3902f116580bf3695943d07e0abdc456fd908f91bffa02602fbd3b944667d5'})
UNUSED_CLASS_KEYS = {"classifier.weight", "classifier_proj.weight", "prompt_learner.cls_ctx"}


@dataclass(frozen=True)
class TrainImage:
    path: str
    person_id: int
    original_id: int


def train_index(annotation: Path, image_root: Path, *, expected_counts=True):
    """非 train 的 ID/路径仅用于隔离审计；caption 永不提取，返回数据仅有 train。"""
    data = json.loads(annotation.read_text(encoding="utf-8"))
    split_ids = {name: set() for name in ("train", "val", "test")}
    split_paths = {name: set() for name in split_ids}
    rows = []
    counts = {name: 0 for name in split_ids}
    for row in data:
        split = row["split"]
        if split not in split_ids:
            raise ValueError(f"未知 split: {split}")
        pid, relative = row["id"], row["file_path"]
        if not isinstance(pid, int) or not isinstance(relative, str):
            raise ValueError("ID 必须为整数，路径必须为字符串")
        if relative in split_paths[split]:
            raise ValueError("同一 split 有重复图像")
        split_ids[split].add(pid)
        split_paths[split].add(relative)
        counts[split] += 1
        if split == "train":
            image_path(image_root, relative)
            rows.append((relative, pid))
    for other in ("val", "test"):
        if split_ids["train"] & split_ids[other]:
            raise ValueError(f"train/{other} identity 泄漏")
        if split_paths["train"] & split_paths[other]:
            raise ValueError(f"train/{other} 图像泄漏")
    if not rows:
        raise ValueError("train split 为空")
    if expected_counts and (
        counts != {"train": 34054, "val": 3078, "test": 3074}
        or {k: len(v) for k, v in split_ids.items()} !=
        {"train": 11003, "val": 1000, "test": 1000}
    ):
        raise ValueError("CUHK 官方 split 数量不匹配")
    mapping = {pid: i for i, pid in enumerate(sorted(split_ids["train"]))}
    records = tuple(TrainImage(path, mapping[pid], pid) for path, pid in rows)
    metadata = {
        "annotation_sha256": sha256(annotation),
        "training_split": "train", "split_image_counts": counts,
        "split_identity_counts": {k: len(v) for k, v in split_ids.items()},
        "train_test_identity_overlap": 0, "train_val_identity_overlap": 0,
        "train_test_image_overlap": 0, "train_val_image_overlap": 0,
        "training_records_sha256": fingerprint(rows),
        "id_mapping": "sorted train-only original IDs -> contiguous [0,11002]",
        "training_fields": ["file_path", "id"], "captions_used": False,
        "camera_annotation": None, "attribute_annotation_required": False,
        "non_train_metadata_use": "counts/disjointness audit only; not returned to training",
    }
    return records, metadata


def require_selection(config):
    """缺少原 camera-aware 选取协议时，默认阻止正式训练。"""
    selection = config["checkpoint_selection"]
    if (selection.get("policy") != "fixed_final_epoch"
            or selection.get("user_accepted_adaptation") is not True):
        raise RuntimeError(
            "正式训练已阻止：CUHK 没有原 camera-aware ReID checkpoint 选取协议；"
            "需先确定并明确接受选取 adaptation，不能使用 IRRA/test 或伪造 camera。")


def check_sources(root: Path):
    hashes = {name: sha256(root / name) for name in TRAIN_HASHES}
    if hashes != TRAIN_HASHES:
        mismatches = [name for name in hashes if hashes[name] != TRAIN_HASHES[name]]
        raise ValueError(f"已审计 AP recipe 源码改变: {mismatches}")
    return hashes


def source_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(root: Path, name: str):
    if name not in sys.modules:
        module = ModuleType(name)
        module.__path__ = [str(root)]
        sys.modules[name] = module
    elif Path(sys.modules[name].__path__[0]).resolve() != root.resolve():
        raise ValueError("AP 包已从其他路径加载")
    return name


def official_attack_functions(root: Path):
    """逐字加载官方函数 AST，避免导入原 main/其他数据集 evaluator。"""
    tree = ast.parse((root / "train_ap_attack.py").read_text())
    names = {"L_norm", "perturb_train", "adv_TripletLoss"}
    nodes = [n for n in tree.body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError("AP 原损失函数缺失")
    namespace = {
        "torch": torch, "nn": nn, "F": F,
        "Imagenet_mean": [.5] * 3, "Imagenet_stddev": [.5] * 3,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]),
                 str(root / "train_ap_attack.py"), "exec"), namespace)
    return namespace


def official_generator(root: Path):
    # GD.py 中绝对 advers 导入；训练进程独立于 IRRA。
    package(root / "advers", "advers")
    module = importlib.import_module("advers.GD")
    return module.Generator(3, 3, 32, norm="bn", n_blocks=6, beta=.1).apply(
        module.weights_init)


def official_ide(root: Path, classes: int):
    cache = Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/resnet50-0676ba61.pth"
    if not cache.is_file() or sha256(cache) != RESNET_SHA:
        raise ValueError("本地 ImageNet ResNet50 V1 缓存缺失/不匹配；禁止静默下载")
    module = source_file(root / "tools/train_ide_duke.py", "_ap_cuhk_ide")
    return module.IDE(num_classes=classes, pretrained=True)


def official_semantic(root: Path, classes: int):
    if sha256(CLIP_REID) != CLIP_REID_SHA:
        raise ValueError("固定 CLIP-ReID 初始化 checkpoint 不匹配")
    clip_cache = Path(os.environ["AP_ATTACK_CLIP_CACHE"]) / "ViT-B-16.pt"
    if not clip_cache.is_file() or sha256(clip_cache) != (
        "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
    ):
        raise ValueError("原 OpenAI CLIP JIT 缓存缺失/不匹配；禁止静默下载")
    defaults = source_file(root / "config/defaults.py", "_ap_cuhk_defaults")
    cfg = defaults._C.clone()
    cfg.merge_from_file(str(root / "configs/att/vit_clipreid_att_reid_semantic_10_10.yml"))
    cfg.DATASETS.NAMES = "CUHK-PEDES"
    package(root / "model", "_ap_cuhk_model")
    module = importlib.import_module("_ap_cuhk_model.make_model_clipreid")
    model = module.build_transformer_att_LAST(classes, 0, 0, cfg)
    state = torch.load(CLIP_REID, map_location="cpu", weights_only=False)
    target = model.state_dict()
    skipped = []
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key not in target:
            raise ValueError(f"CLIP-ReID 未知键: {key}")
        if target[key].shape != value.shape:
            if key not in UNUSED_CLASS_KEYS:
                raise ValueError(f"CLIP-ReID 活跃参数形状不匹配: {key}")
            skipped.append(key)
        else:
            target[key].copy_(value)
    missing = set(target) - {k.removeprefix("module.") for k in state}
    allowed = {k for k in target if k.startswith("prompt_text_")} | {
        "token_embbdings.weight",  # 原 PR checkpoint 不含此未使用的 token embedding 别名。
        "prompt_learner.token_prefix_2", "prompt_learner.token_prefix_3",
        "prompt_learner.token_prefix_4", "prompt_learner.token_prefix_5",
        "prompt_learner.token_prefix_6", "prompt_learner.token_suffix_my",
    }
    if not missing <= allowed or set(skipped) != UNUSED_CLASS_KEYS:
        raise ValueError(f"非预期 CLIP-ReID 初始化差异: {missing - allowed}, {skipped}")
    for name, value in model.named_parameters():
        value.requires_grad_("prompt_text_" in name)
    return model, cfg, {"unused_class_head_reinitialized": sorted(skipped),
                         "initialization_only_keys": sorted(missing),
                         "frozen_backbone": "existing Duke CLIP-ReID; no CUHK/IRRA pretraining",
                         "checkpoint_sha256": CLIP_REID_SHA}


class TrainImages(Dataset):
    def __init__(self, records, root, transform):
        self.records, self.root, self.transform = records, root, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(image_path(self.root, row.path)) as image:
            pixels = self.transform(image.convert("RGB"))
        return pixels, row.person_id


@dataclass
class IDEWorkerSeed:
    seed: int

    def __call__(self, worker_id):
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)


def make_loader(records, image_root, root, stage, config, *, batch_size=None, workers=None):
    batch = batch_size or config[stage]["batch_size"]
    worker_count = config["workers"] if workers is None else workers
    sampler = None
    generator = None
    if stage == "ide":
        helper = source_file(root / "tools/train_ide_duke.py", "_ap_cuhk_ide")
        transform = T.Compose([
            helper.RandomSizedRectCrop(256, 128), T.RandomHorizontalFlip(),
            T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        generator = torch.Generator().manual_seed(config["seed"])
    else:
        interpolation = T.InterpolationMode.BILINEAR if stage == "inversion" else T.InterpolationMode.BICUBIC
        transform = T.Compose([
            T.Resize((256, 128), interpolation=interpolation), T.ToTensor(),
            T.Normalize([.5] * 3, [.5] * 3),
        ])
        if stage == "generator":
            if batch % 4 or batch < 8:
                raise ValueError("G 的 P×K batch 至少 2 IDs × 4 images")
            helper = source_file(root / "datasets/sampler.py", "_ap_cuhk_sampler")
            # None 仅满足原 sampler 四元组接口；不是 camera/view 标注，训练不返回这些字段。
            sampler = helper.RandomIdentitySampler(
                [(r.path, r.person_id, None, None) for r in records], batch, 4)
    return DataLoader(
        TrainImages(records, image_root, transform), batch_size=batch,
        shuffle=sampler is None, sampler=sampler, num_workers=worker_count,
        drop_last=stage == "ide", pin_memory=stage == "ide", generator=generator,
        worker_init_fn=IDEWorkerSeed(config["seed"]) if stage == "ide" else None,
    )


def seed_all(seed, stage):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = stage != "ide"
    torch.backends.cuda.matmul.allow_tf32 = stage == "generator"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_resume(path, model, optimizer, scaler, scheduler, loader, spec, epoch, *, mode="formal"):
    atomic_torch(path, {
        "schema": 1, "spec": spec, "epoch": epoch, "mode": mode,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
        "rng": rng_state(),
        "loader_generator": loader.generator.get_state() if loader.generator else None,
        "boundary": "epoch_end" if mode == "formal" else "smoke_single_step",
    })


def load_resume(path, model, optimizer, scaler, scheduler, loader, spec, *, mode="formal"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["schema"] != 1 or state["spec"] != spec or state["mode"] != mode:
        raise ValueError("续跑来源、数据、配置或 smoke/formal 模式不匹配")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    if scheduler:
        scheduler.load_state_dict(state["scheduler"])
    if loader.generator:
        loader.generator.set_state(state["loader_generator"])
    restore_rng(state["rng"])
    return state["epoch"]


def generator_loss(pixels, ids, generator, ide, semantic, original):
    adv, _ = original["perturb_train"](pixels, generator, "train")
    transform = T.Compose([
        T.Normalize([-1.] * 3, [2.] * 3),
        T.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    clean_tokens = semantic(pixels, ids)[-1]
    adv_tokens = semantic(adv, ids)[-1]
    clean_ide, adv_ide = ide(transform(pixels)), ide(transform(adv))
    loss = original["adv_TripletLoss"](.3)
    reid = 10 * loss(clean_ide.detach(), adv_ide, ids)
    semantic_loss = 10 * torch.stack([
        loss(c.detach(), a, ids)
        for c, a in zip(clean_tokens.chunk(5), adv_tokens.chunk(5))
    ]).sum()
    return reid + semantic_loss, reid, semantic_loss, adv
