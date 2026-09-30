#!/usr/bin/env python3
"""AP-Attack Baseline 3：仅用冻结 IRRA 图像特征替换 IDE 分支。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for key, value in {
    "TORCH_HOME": "/home/lzf/ldx/cache/torch",
    "AP_ATTACK_CLIP_CACHE": "/home/lzf/ldx/cache/clip",
    "HF_HOME": "/home/lzf/ldx/cache/huggingface",
    "TMPDIR": "/home/lzf/ldx/tmp",
}.items():
    os.environ[key] = value

import torch
import yaml
from torch.cuda import amp

from attributes.ap_cuhk_training import (
    ANNOTATION, AP_SOURCE, CLIP_REID, CLIP_REID_SHA, IMAGE_ROOT,
    atomic_json, atomic_torch, check_sources, official_attack_functions,
    official_semantic, save_resume, load_resume, seed_all, sha256, train_index,
    write_path,
)
from attributes.ap_cuhk_irra_training import (
    ap_irra_losses, ide_loss_on_same_images, make_dual_loader,
)
from attributes.irra_victim import FrozenIRRA, OFFICIAL_CUHK_SHA256
from scripts import train_ap_cuhk_baseline as base

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs/ap_cuhk_irra_surrogate.yaml"
BASE_OUTPUT = Path("/home/lzf/ldx/outputs/AP-Attack/cuhk_reid_semantic_10_10")
OUTPUT = Path("/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10")
IRRA_REPO = Path("/home/lzf/Attack/IRRA-main/IRRA-main")
IRRA_CHECKPOINT = Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth")
IRRA_CONFIG = REPO / "configs/irra_cuhk_official.yaml"
IRRA_SOURCE_FILES = (
    "model/build.py", "model/clip_model.py", "utils/simple_tokenizer.py",
)
FINAL_NAME = "G_CUHK_IRRA_AP.pth.tar"


def read_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    baseline = base.read_config(base.DEFAULT_CONFIG)
    g = baseline["generator"]
    expected = {
        "schema": 1, "training_dataset": "CUHK-PEDES/train",
        "training_surrogate": "IRRA image encoder",
        "semantic_source": "Baseline 2 Inversion_CUHK epoch 20",
        "seed": baseline["seed"], "workers": baseline["workers"],
        "generator": {
            "epochs": g["epochs"], "batch_size": g["batch_size"],
            "instances_per_identity": g["instances_per_identity"],
            "architecture": g["architecture"],
            "initialization": g["initialization"],
            "optimizer": g["optimizer"], "lr": g["lr"], "betas": g["betas"],
            "weight_decay": g["weight_decay"], "epsilon": baseline["epsilon"],
            "loss": g["loss"], "margin": g["margin"],
            "irra_weight": g["reid_weight"],
            "semantic_weight": g["semantic_weight"],
            "semantic_reduction": g["semantic_reduction"],
            "native_resize": "PIL bicubic 256x128",
            "victim_resize": "PIL bilinear clean 384x128 + differentiable bilinear native delta",
            "validation": "none", "checkpoint_selection": "fixed_final_epoch_60",
        },
    }
    if config != expected:
        raise ValueError("Baseline 3 必须保持 Baseline 2 G recipe，仅替换 IDE 分支")
    return config, baseline


def git_commit(path):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path,
                                   text=True).strip()


def verify_baseline2(data, baseline_config, *, include_ide=False):
    stored = json.loads((BASE_OUTPUT / "run-metadata.json").read_text())
    if (stored["dataset"] != data or stored["config"] != baseline_config
            or stored["training_uses_irra"] or stored["training_uses_captions"]):
        raise ValueError("Baseline 2 train-only 来源不匹配")
    path = BASE_OUTPUT / "inversion/epoch_20.pth"
    inv_sha = base.checkpoint_provenance(path, "inversion", 20, stored)
    dependencies = {"inversion": {
        "path": str(path), "sha256": inv_sha, "epoch": 20,
        "training_dataset": "CUHK-PEDES/train",
    }}
    if include_ide:
        path = BASE_OUTPUT / "ide/epoch_50.pth"
        ide_sha = base.checkpoint_provenance(path, "ide", 50, stored)
        dependencies["ide_comparison_only"] = {
            "path": str(path), "sha256": ide_sha, "epoch": 50,
            "training_dataset": "CUHK-PEDES/train",
        }
    return dependencies


def prepare(args, config, baseline_config, *, include_ide=False):
    records, data = train_index(args.annotation, args.image_root)
    deps = verify_baseline2(data, baseline_config, include_ide=include_ide)
    ap_hashes = check_sources(args.ap_source)
    if sha256(CLIP_REID) != CLIP_REID_SHA:
        raise ValueError("冻结 CLIP-ReID checkpoint 不匹配")
    irra_sha = sha256(args.irra_checkpoint)
    if irra_sha != OFFICIAL_CUHK_SHA256:
        raise ValueError("IRRA 官方 checkpoint SHA-256 不匹配")
    irra = {
        "checkpoint": str(args.irra_checkpoint.resolve()),
        "checkpoint_sha256": irra_sha,
        "config": str(args.irra_config.resolve()),
        "config_sha256": sha256(args.irra_config),
        "source_root": str(args.irra_repo.resolve()),
        "source_hashes": {
            name: sha256(args.irra_repo / name) for name in IRRA_SOURCE_FILES
        },
        "feature": "model/build.py::encode_image -> ViT CLS x[:,0,:] -> float -> L2 normalize",
        "feature_dim": 512, "frozen": True,
    }
    spec = {
        "baseline": "AP-Attack G_CUHK_IRRA_AP",
        "training_dataset": "CUHK-PEDES/train",
        "training_surrogate": "IRRA image encoder",
        "config": config, "baseline2_generator_recipe": baseline_config["generator"],
        "dataset": data, "ap_source": str(args.ap_source.resolve()),
        "ap_source_hashes": ap_hashes,
        "semantic_backbone": {
            "checkpoint": str(CLIP_REID), "sha256": CLIP_REID_SHA,
            "training_dataset": "DukeMTMC-reID", "frozen": True,
        },
        "inversion_source": deps["inversion"], "irra_source": irra,
        "training_uses_captions": False, "training_uses_irra_text": False,
        "checkpoint_selection": "fixed_final_epoch_60; no validation",
        "seed": config["seed"], "project_commit": git_commit(REPO),
        "adapter_hashes": {
            "module": sha256(REPO / "src/attributes/ap_cuhk_irra_training.py"),
            "runner": sha256(Path(__file__).resolve()),
            "baseline2_module": sha256(REPO / "src/attributes/ap_cuhk_training.py"),
        },
        "torch_version": str(torch.__version__),
    }
    return records, spec, deps


def formal_artifacts(output):
    paths = list((output / "generator").glob("epoch_*.pth"))
    for name in ("generator/last.pt", "progress.json", FINAL_NAME):
        path = output / name
        if path.exists():
            paths.append(path)
    return paths


def record_setup(output, spec):
    path = write_path(output / "run-metadata.json")
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("现有训练输出来源不同；不能覆盖或静默 resume")
    atomic_json(path, spec)
    atomic_json(output / "config.json", spec["config"])
    atomic_json(output / "dataset-metadata.json", spec["dataset"])


def load_sources(args, spec):
    semantic, _, _ = official_semantic(args.ap_source, 11003)
    semantic.load_state_dict(
        torch.load(spec["inversion_source"]["path"], map_location="cpu",
                   weights_only=False), strict=True)
    semantic = semantic.cuda().eval().requires_grad_(False)
    irra = FrozenIRRA(args.irra_repo, args.irra_checkpoint, args.irra_config,
                      device="cuda").eval().requires_grad_(False)
    original = official_attack_functions(args.ap_source)
    return semantic, irra, original


def append_log(output, row):
    path = write_path(output / "loss.jsonl")
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def export_state(path, model, spec, epoch):
    atomic_torch(path, model.state_dict())
    atomic_json(path.with_suffix(path.suffix + ".json"), {
        "mode": "formal", "stage": "generator", "epoch": epoch,
        "checkpoint": str(path), "checkpoint_sha256": sha256(path), "spec": spec,
    })


def selected_checkpoint(output):
    return output / "generator/epoch_60.pth"


def verify_selected(output, spec):
    path = selected_checkpoint(output)
    metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    actual = sha256(path)
    if (metadata.get("mode") != "formal"
            or metadata.get("stage") != "generator"
            or metadata.get("epoch") != 60
            or metadata.get("checkpoint_sha256") != actual
            or metadata.get("spec") != spec):
        raise ValueError("Baseline 3 epoch 60 checkpoint 来源不匹配")
    return path


def export_generator(output, model_state, spec):
    path = write_path(output / FINAL_NAME)
    atomic_torch(path, model_state)
    metadata = {
        "mode": "formal", "training_dataset": "CUHK-PEDES/train",
        "train_images": spec["dataset"]["split_image_counts"]["train"],
        "train_identities": spec["dataset"]["split_identity_counts"]["train"],
        "training_surrogate": "IRRA image encoder",
        "surrogate": "official_IRRA_image_encoder",
        "training_uses_captions": False,
        "irra_checkpoint": spec["irra_source"]["checkpoint"],
        "irra_checkpoint_sha256": spec["irra_source"]["checkpoint_sha256"],
        "semantic_backbone": spec["semantic_backbone"],
        "inversion_source": spec["inversion_source"],
        "captions_used": False, "training_uses_irra_text": False,
        "epsilon": spec["config"]["generator"]["epsilon"],
        "generator_config": spec["config"]["generator"],
        "optimizer_config": {
            key: spec["config"]["generator"][key]
            for key in ("optimizer", "lr", "betas", "weight_decay")
        },
        "seed": spec["seed"], "epoch": 60,
        "project_commit": spec["project_commit"],
        "selection": "fixed_final_epoch_60; no source mAP selection",
        "checkpoint_sha256": sha256(path), "spec": spec,
    }
    atomic_json(path.with_suffix(path.suffix + ".json"), metadata)
    atomic_json(output / "progress.json", {
        "state": "complete", "epoch": 60, "formal_training_started": True,
        "generator": str(path), "sha256": sha256(path),
    })


def train(args, config, baseline_config):
    if formal_artifacts(args.output_dir) and not args.resume:
        raise ValueError("已有正式训练输出；显式使用 --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline 3 正式训练需要 CUDA")
    records, spec, _ = prepare(args, config, baseline_config)
    record_setup(args.output_dir, spec)
    seed_all(config["seed"], "generator")
    loader = make_dual_loader(records, args.image_root, args.ap_source,
                              baseline_config, workers=config["workers"])
    semantic, irra, original = load_sources(args, spec)
    generator, optimizer, scaler, scheduler, _ = base.stage_objects(
        "generator", args.ap_source, 11003, baseline_config)
    output = write_path(args.output_dir / "generator")
    output.mkdir(parents=True, exist_ok=True)
    last = output / "last.pt"
    start = 1
    if last.exists():
        if not args.resume:
            raise ValueError("已有 G checkpoint；显式使用 --resume")
        start = load_resume(last, generator, optimizer, scaler, scheduler,
                            loader, spec) + 1
    elif args.resume and formal_artifacts(args.output_dir):
        raise ValueError("正式输出已存在但 generator/last.pt 缺失")
    for epoch in range(start, 61):
        generator.train()
        seen, total = 0, 0.
        for step, (ap_pixels, irra_clean, ids) in enumerate(loader, 1):
            ap_pixels = ap_pixels.cuda(non_blocking=True)
            irra_clean = irra_clean.cuda(non_blocking=True)
            ids = ids.cuda(non_blocking=True)
            optimizer.zero_grad()
            with amp.autocast():
                adv, _ = original["perturb_train"](ap_pixels, generator, "train")
                loss, irra_loss, semantic_loss, _ = ap_irra_losses(
                    ap_pixels, adv, irra_clean, ids, irra, semantic, original)
            if not all(bool(torch.isfinite(x)) for x in
                       (loss, irra_loss, semantic_loss)):
                raise RuntimeError("Baseline 3 G loss 非有限")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += len(ids)
            total += float(loss.detach()) * len(ids)
            if step == 1 or step % 100 == 0:
                append_log(args.output_dir, {
                    "stage": "generator", "epoch": epoch, "step": step,
                    "loss": float(loss.detach()),
                    "irra": float(irra_loss.detach()),
                    "semantic": float(semantic_loss.detach()),
                    "lr": optimizer.param_groups[0]["lr"],
                })
        if not seen:
            raise RuntimeError("Baseline 3 train dataloader 为空")
        if epoch % 10 == 0 or epoch == 60:
            export_state(output / f"epoch_{epoch}.pth", generator, spec, epoch)
        save_resume(last, generator, optimizer, scaler, scheduler, loader,
                    spec, epoch)
        atomic_json(args.output_dir / "progress.json", {
            "stage": "generator", "epoch": epoch, "max_epochs": 60,
            "mean_loss": total / seen, "epoch_samples": seen,
            "last_checkpoint": str(last), "last_sha256": sha256(last),
            "state": "stage_complete" if epoch == 60 else "training",
            "formal_training_started": True,
        })
    path = verify_selected(args.output_dir, spec)
    export_generator(args.output_dir,
                     torch.load(path, map_location="cpu", weights_only=False),
                     spec)


def smoke(args, config, baseline_config):
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline 3 smoke 需要 CUDA")
    output = write_path(args.output_dir / "smoke")
    if any((output / name).exists() for name in
           ("result.json", "optimizer-step-intent.json", "resume.pt")):
        raise ValueError("该目录已执行单步 smoke，不可重复 optimizer step")
    records, spec, dependencies = prepare(
        args, config, baseline_config, include_ide=True)
    record_setup(output, spec)
    seed_all(config["seed"], "generator")
    loader = make_dual_loader(records, args.image_root, args.ap_source,
                              baseline_config, batch_size=8, workers=0)
    ap_pixels, irra_clean, ids = next(iter(loader))
    ap_pixels, irra_clean, ids = (
        ap_pixels.cuda(), irra_clean.cuda(), ids.cuda())
    semantic, irra, original = load_sources(args, spec)
    ide = base.official_ide(args.ap_source, 11003).cuda()
    ide.load_state_dict(torch.load(
        dependencies["ide_comparison_only"]["path"], map_location="cpu",
        weights_only=False), strict=True)
    ide.eval().requires_grad_(False)
    generator, optimizer, _, scheduler, _ = base.stage_objects(
        "generator", args.ap_source, 11003, baseline_config)
    scaler = amp.GradScaler(init_scale=1.)  # 仅诊断；正式训练使用原默认 scale。
    before = {
        key: value.detach().cpu().clone()
        for key, value in generator.named_parameters() if value.requires_grad
    }
    with amp.autocast():
        adv, _ = original["perturb_train"](ap_pixels, generator, "train")
        loss, irra_loss, semantic_loss, info = ap_irra_losses(
            ap_pixels, adv, irra_clean, ids, irra, semantic, original)
        with torch.no_grad():
            ide_loss = ide_loss_on_same_images(
                ap_pixels, adv, ids, ide, original)
    if not all(bool(torch.isfinite(x)) for x in
               (loss, irra_loss, semantic_loss, ide_loss)):
        raise RuntimeError("单步 smoke loss 非有限")
    g_parameters = tuple(p for p in generator.parameters() if p.requires_grad)
    branch_grads = torch.autograd.grad(
        irra_loss, (adv,) + g_parameters,
        retain_graph=True, allow_unused=True)
    adv_grad = branch_grads[0]
    g_irra_grad_l1 = sum(float(g.abs().sum()) for g in branch_grads[1:]
                         if g is not None)
    if (adv_grad is None or not torch.isfinite(adv_grad).all()
            or float(adv_grad.abs().sum()) <= 0
            or not all(g is None or torch.isfinite(g).all()
                       for g in branch_grads[1:])
            or g_irra_grad_l1 <= 0):
        raise RuntimeError("IRRA image feature 到 G 的梯度链路失败")
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grads = [p.grad for p in g_parameters if p.grad is not None]
    if (not grads or not all(torch.isfinite(g).all() for g in grads)
            or sum(float(g.abs().sum()) for g in grads) <= 0):
        raise RuntimeError("G 梯度缺失、非有限或为零")
    if any(p.requires_grad or p.grad is not None for p in irra.parameters()):
        raise RuntimeError("IRRA 参数未完全冻结")
    if any(p.requires_grad or p.grad is not None for p in semantic.parameters()):
        raise RuntimeError("semantic 参数未完全冻结")
    native_linf = (info["native_adv"] - info["native_clean"]).abs().flatten(1).amax(1)
    victim_linf = (info["adv_irra"] - irra_clean).abs().flatten(1).amax(1)
    if max(float(native_linf.max()), float(victim_linf.max())) > 8/255 + 1e-6:
        raise RuntimeError("单步 smoke 超出 8/255 像素预算")
    atomic_json(output / "optimizer-step-intent.json", {
        "max_real_steps": 1, "retry_forbidden": True,
    })
    scaler.step(optimizer)
    scaler.update()
    changed = sum(
        not torch.equal(before[k], p.detach().cpu())
        for k, p in generator.named_parameters() if p.requires_grad)
    if changed == 0:
        raise RuntimeError("单步 smoke 后 G 参数未更新")
    resume = output / "resume.pt"
    save_resume(resume, generator, optimizer, scaler, scheduler, loader,
                spec, 0, mode="smoke")
    expected = {k: v.detach().cpu().clone()
                for k, v in generator.state_dict().items()}
    lr = optimizer.param_groups[0]["lr"]
    draws = torch.rand(4).cpu()
    with torch.no_grad():
        next(p for p in generator.parameters() if p.requires_grad).add_(1)
    optimizer.param_groups[0]["lr"] = 9
    epoch = load_resume(resume, generator, optimizer, scaler, scheduler,
                        loader, spec, mode="smoke")
    resume_ok = (epoch == 0 and optimizer.param_groups[0]["lr"] == lr
                 and all(torch.equal(v, generator.state_dict()[k].detach().cpu())
                         for k, v in expected.items())
                 and torch.equal(draws, torch.rand(4).cpu()))
    if not resume_ok:
        raise RuntimeError("单步 smoke resume 不一致")
    result = {
        "scope": "one batch and one real G optimizer step; no formal training",
        "real_optimizer_steps": 1, "batch_shape": list(ap_pixels.shape),
        "batch_id_count": len(ids.unique()),
        "same_batch_clean_adv_ids_for_comparison": True,
        "irra_feature_location": spec["irra_source"]["feature"],
        "irra_clean_feature_shape": list(info["clean_irra_feature"].shape),
        "irra_adv_feature_shape": list(info["adv_irra_feature"].shape),
        "irra_feature_l2_normalized": True,
        "irra_checkpoint_sha256": spec["irra_source"]["checkpoint_sha256"],
        "irra_parameters_frozen": True, "irra_parameter_grads_none": True,
        "semantic_backbone_frozen": True,
        "inversion_source": spec["inversion_source"],
        "baseline2_ide_comparison_source": dependencies["ide_comparison_only"],
        "baseline2_ide_loss_same_images": float(ide_loss),
        "irra_loss": float(irra_loss), "semantic_loss": float(semantic_loss),
        "total_loss": float(loss), "irra_to_adv_grad_l1": float(adv_grad.abs().sum()),
        "irra_to_g_grad_l1": g_irra_grad_l1,
        "g_gradient_finite": True, "g_gradient_nonzero": True,
        "changed_trainable_tensors": changed,
        "native_linf_mean": float(native_linf.mean()),
        "native_linf_max": float(native_linf.max()),
        "victim_linf_mean": float(victim_linf.mean()),
        "victim_linf_max": float(victim_linf.max()),
        "epsilon": 8/255, "training_uses_captions": False,
        "training_uses_irra_text": False,
        "smoke_amp_init_scale": 1., "resume_ok": resume_ok,
        "resume_sha256": sha256(resume), "formal_training_started": False,
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "status", "smoke", "train"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--ap-source", type=Path, default=AP_SOURCE)
    parser.add_argument("--irra-repo", type=Path, default=IRRA_REPO)
    parser.add_argument("--irra-checkpoint", type=Path, default=IRRA_CHECKPOINT)
    parser.add_argument("--irra-config", type=Path, default=IRRA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = write_path(args.output_dir)
    config, baseline_config = read_config(args.config)
    if args.command == "train":
        train(args, config, baseline_config)
    elif args.command == "smoke":
        smoke(args, config, baseline_config)
    else:
        plan = {
            "training_dataset": config["training_dataset"],
            "training_surrogate": config["training_surrogate"],
            "semantic_source": str(BASE_OUTPUT / "inversion/epoch_20.pth"),
            "irra_checkpoint": str(args.irra_checkpoint),
            "generator_fresh": True, "generator_epochs": 60,
            "checkpoint_selection": "fixed_final_epoch_60",
            "final_generator": str(args.output_dir / FINAL_NAME),
            "formal_artifacts": [str(p) for p in formal_artifacts(args.output_dir)],
        }
        if args.command == "plan":
            _, spec, _ = prepare(args, config, baseline_config)
            plan["dataset"] = spec["dataset"]
            plan["inversion_source"] = spec["inversion_source"]
        else:
            path = args.output_dir / "progress.json"
            plan["progress"] = json.loads(path.read_text()) if path.exists() else None
            smoke_path = args.output_dir / "smoke/result.json"
            plan["smoke"] = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
