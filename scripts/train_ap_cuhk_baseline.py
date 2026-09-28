#!/usr/bin/env python3
"""AP-Attack CUHK：audit/status/smoke/train；本阶段默认阻止正式训练。"""
from __future__ import annotations

import argparse
import hashlib
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
    # 所有缓存强制位于授权目录，避免受外部环境影响。
    os.environ[key] = value

import torch
import yaml
from torch.cuda import amp

from attributes.ap_cuhk_training import (
    ANNOTATION, AP_SOURCE, CLIP_REID_SHA, IMAGE_ROOT, OUTPUT,
    atomic_json, atomic_torch, check_sources, fingerprint, generator_loss,
    load_resume, make_loader, official_attack_functions, official_generator,
    official_ide, official_semantic, package, require_selection,
    rng_state, save_resume, seed_all, sha256, source_file, train_index, write_path,
)
import importlib

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/ap_cuhk_original_training.yaml"
FROZEN_RECIPE_SHA = "e2b7842da25ee46aeb1370f0e85e3a27a3515b1ba2cafbd23bc056322cc49570"


def read_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    # 本入口仅支持已审计 recipe；配置不是调参接口。
    original = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if fingerprint({k: v for k, v in config.items() if k != "checkpoint_selection"}) != FROZEN_RECIPE_SHA:
        raise ValueError("禁止改变已冻结 AP recipe")
    for key in original:
        if key != "checkpoint_selection" and config.get(key) != original[key]:
            raise ValueError(f"禁止改变已冻结 AP recipe: {key}")
    if set(config) != set(original):
        raise ValueError("训练配置字段不匹配")
    return config


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def prepare(args, config):
    records, data = train_index(args.annotation, args.image_root)
    hashes = check_sources(args.ap_source)
    repo = Path(__file__).resolve().parents[1]
    spec = {
        "baseline": "AP-Attack G_CUHK^AP", "config": config, "dataset": data,
        "ap_source": str(args.ap_source.resolve()), "ap_source_hashes": hashes,
        "ap_upstream_commit": "f7d906eae675aabccdd388a2b06d01d51d73d89d",
        "ap_reproduction_commit": git_commit(args.ap_source),
        "project_commit": git_commit(repo),
        "adapter_hashes": {
            "module": sha256(repo / "src/attributes/ap_cuhk_training.py"),
            "runner": sha256(Path(__file__).resolve()),
        },
        "torch_version": str(torch.__version__),
        "clip_reid_initialization_sha256": CLIP_REID_SHA,
        "checkpoint_selection": config["checkpoint_selection"],
        "training_uses_irra": False, "training_uses_captions": False,
        "formal_resume": "epoch boundaries; RNG/optimizer/AMP/scheduler/loader state",
    }
    return records, spec


def record_setup(output, config, spec):
    path = write_path(output / "config.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != config:
        raise ValueError("已有训练输出配置不同；使用独立目录")
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    atomic_json(output / "dataset-metadata.json", spec["dataset"])
    atomic_json(output / "run-metadata.json", spec)


def append_log(output, row):
    path = write_path(output / "loss.jsonl")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    path = write_path(output / "train.log")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def export_state(path, model, metadata):
    atomic_torch(path, model.state_dict())
    atomic_json(path.with_suffix(path.suffix + ".json"), {
        **metadata, "checkpoint_sha256": sha256(path),
        "checkpoint": str(path), "state_format": "raw_state_dict",
    })


def stage_objects(stage, root, classes):
    scheduler = None
    extra = {}
    if stage == "ide":
        model = official_ide(root, classes).cuda()
        optimizer = torch.optim.SGD([
            {"params": model.base.parameters(), "lr": .01},
            {"params": list(model.feat.parameters()) + list(model.feat_bn.parameters())
                       + list(model.classifier.parameters()), "lr": .1},
        ], momentum=.9, weight_decay=5e-4, nesterov=True)
    elif stage == "inversion":
        model, cfg, extra = official_semantic(root, classes)
        model.cuda()
        package(root / "solver", "_ap_cuhk_solver")
        helper = importlib.import_module("_ap_cuhk_solver.make_optimizer_prompt")
        optimizer = helper.make_optimizer_textInverse(cfg, model)
        factory = importlib.import_module("_ap_cuhk_solver.scheduler_factory")
        scheduler = factory.create_scheduler(optimizer, 40, 2e-4, 2e-4, 10)
    else:
        model = official_generator(root).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.5, .999))
    # 原 IDE 为 FP32；inversion/G 使用 CUDA AMP。
    return model, optimizer, amp.GradScaler(enabled=stage != "ide"), scheduler, extra


def train_stage(args, config, records, spec, stage, dependencies=None):
    seed_all(config["seed"], stage)
    classes = spec["dataset"]["split_identity_counts"]["train"]
    loader = make_loader(records, args.image_root, args.ap_source, stage, config)
    # 原 G main 先构造/加载 semantic，再构造 IDE，最后初始化 G。
    if stage == "generator":
        semantic, _, _ = official_semantic(args.ap_source, classes)
        semantic.load_state_dict(torch.load(Path(dependencies["inversion"]["path"]),
                                           map_location="cpu", weights_only=False), strict=True)
        semantic = semantic.cuda().eval().requires_grad_(False)
        ide = official_ide(args.ap_source, classes).cuda()
        ide.load_state_dict(torch.load(Path(dependencies["ide"]["path"]),
                                      map_location="cpu", weights_only=False), strict=True)
        ide.eval().requires_grad_(False)
        original = official_attack_functions(args.ap_source)
    model, optimizer, scaler, scheduler, extra = stage_objects(stage, args.ap_source, classes)
    output = write_path(args.output_dir / stage)
    output.mkdir(parents=True, exist_ok=True)
    stage_spec = {**spec, "stage": stage, "initialization": extra,
                  "dependencies": dependencies or {}}
    last = output / "last.pt"
    start = 1
    if last.exists():
        if not args.resume:
            raise ValueError("已有训练 checkpoint；显式使用 --resume")
        start = load_resume(last, model, optimizer, scaler, scheduler, loader, stage_spec) + 1
    epochs = config[stage]["epochs"]
    if start > epochs:
        return model
    if stage == "inversion":
        helper = source_file(args.ap_source / "loss/supcontrast.py", "_ap_cuhk_supcon")
        contrast = helper.SupConLoss("cuda")

    for epoch in range(start, epochs + 1):
        model.train()
        if stage == "ide":
            factor = .1 if epoch >= 40 else 1.
            for group, lr in zip(optimizer.param_groups, (.01, .1)):
                group["lr"] = lr * factor
        elif scheduler:
            scheduler.step(epoch)
        seen, total = 0, 0.
        for step, (pixels, ids) in enumerate(loader, 1):
            pixels, ids = pixels.cuda(), ids.cuda()
            optimizer.zero_grad()
            with amp.autocast(enabled=stage != "ide"):
                if stage == "ide":
                    logits, _ = model(pixels, is_training=True)
                    loss = torch.nn.functional.cross_entropy(logits, ids)
                    components = {}
                elif stage == "inversion":
                    image_features, text_features, *_ = model(pixels, ids)
                    # 原 processor 在 autocast 退出后计算 SupCon；下方重算。
                    loss = None
                    components = {}
                else:
                    loss, reid, semantic_loss, _ = generator_loss(
                        pixels, ids, model, ide, semantic, original)
                    components = {"reid": float(reid.detach()), "semantic": float(semantic_loss.detach())}
            if stage == "inversion":
                loss = .1 * (contrast(image_features, text_features, ids, ids) +
                             contrast(text_features, image_features, ids, ids))
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += len(ids)
            total += float(loss.detach()) * len(ids)
            if step == 1 or step % (100 if stage == "generator" else 50) == 0:
                append_log(args.output_dir, {
                    "stage": stage, "epoch": epoch, "step": step,
                    "loss": float(loss.detach()), **components,
                    "lr": [g["lr"] for g in optimizer.param_groups[:2]],
                })
        if not seen:
            raise RuntimeError("训练 dataloader 为空")
        # 只在完整 epoch 后写可续跑状态；中断会从上一个 epoch 重跑。
        if epoch % config[stage]["save_period"] == 0 or epoch == epochs:
            export_state(output / f"epoch_{epoch}.pth", model, {
                "mode": "formal", "stage": stage, "epoch": epoch, "spec": stage_spec})
        # 先完成阶段权重导出，再推进可续跑 epoch，避免断电丢失 epoch 20 等里程碑。
        save_resume(last, model, optimizer, scaler, scheduler, loader, stage_spec, epoch)
        atomic_json(args.output_dir / "progress.json", {
            "stage": stage, "epoch": epoch, "max_epochs": epochs,
            "mean_loss": total / seen, "epoch_samples": seen,
            "last_checkpoint": str(last), "last_sha256": sha256(last),
            "state": "stage_complete" if epoch == epochs else "training",
            "formal_training_started": True,
        })
    return model


def train(args, config):
    require_selection(config)  # 先检查，再读取数据、分配 GPU 或创建训练输出。
    if not torch.cuda.is_available():
        raise RuntimeError("AP training 需要 CUDA")
    records, spec = prepare(args, config)
    record_setup(args.output_dir, config, spec)
    atomic_json(args.output_dir / "progress.json", {"state": "initializing",
                "formal_training_started": True})
    for stage in ("ide", "inversion", "generator"):
        dependencies = None
        if stage == "generator":
            paths = {
                "ide": args.output_dir / "ide/epoch_50.pth",
                "inversion": args.output_dir / "inversion/epoch_20.pth",
            }
            dependencies = {name: {"path": str(p), "sha256": sha256(p)}
                            for name, p in paths.items()}
        model = train_stage(args, config, records, spec, stage, dependencies)
        del model
        torch.cuda.empty_cache()
    g = torch.load(args.output_dir / "generator/epoch_60.pth", map_location="cpu", weights_only=False)
    path = args.output_dir / "G_CUHK_AP.pth.tar"
    atomic_torch(path, g)
    metadata = {
        "mode": "formal", "training_dataset": "CUHK-PEDES/train",
        "surrogate": "IDE (ResNet50), retrained on CUHK train",
        "generator_epoch": 60, "inversion_epoch": 20, "ide_epoch": 50,
        "selection": "fixed_final_epoch; no validation mAP best claim",
        "checkpoint_selection": config["checkpoint_selection"],
        "checkpoint_sha256": sha256(path), "spec": spec,
    }
    atomic_json(path.with_suffix(path.suffix + ".json"), metadata)
    # 兼容原 AP 命名；明确是固定最终轮别名，绝非验证集 best。
    alias = write_path(args.output_dir / "best_G_V.pth.tar")
    temp = write_path(alias.with_suffix(alias.suffix + ".tmp"))
    temp.write_bytes(path.read_bytes())
    temp.replace(alias)
    atomic_json(args.output_dir / "best_G_V.pth.tar.json", metadata)
    atomic_json(args.output_dir / "progress.json", {
        "state": "complete", "formal_training_started": True,
        "generator": str(path), "sha256": sha256(path),
        "selection": metadata["selection"]})


def smoke(args, config):
    if not torch.cuda.is_available():
        raise RuntimeError("smoke 需要 CUDA")
    output = write_path(args.output_dir / "smoke")
    if any((output / name).exists() for name in ("result.json", "resume.pt", "optimizer-step-intent.json")):
        raise ValueError("该目录已有单步 smoke；不可重复执行 optimizer step，请查看 result.json")
    records, spec = prepare(args, config)
    record_setup(output, config, spec)
    seed_all(config["seed"], "generator")
    # 用原 P×K sampler 抽 2 IDs×4 图；完整 train index 不被截成前缀。
    loader = make_loader(records, args.image_root, args.ap_source, "generator",
                         config, batch_size=8, workers=0)
    pixels, ids = next(iter(loader))
    pixels, ids = pixels.cuda(), ids.cuda()
    ide = official_ide(args.ap_source, 11003).cuda().eval().requires_grad_(False)
    semantic, _, initialization = official_semantic(args.ap_source, 11003)
    semantic = semantic.cuda().eval().requires_grad_(False)
    with torch.no_grad(), amp.autocast():
        ide_features = ide((pixels * .5 + .5 -
                            pixels.new_tensor([.485, .456, .406])[None, :, None, None]) /
                           pixels.new_tensor([.229, .224, .225])[None, :, None, None])
        image_features, text_features, _, _, tokens = semantic(pixels, ids)
    contrast = source_file(args.ap_source / "loss/supcontrast.py", "_ap_cuhk_supcon").SupConLoss("cuda")
    inversion_forward_loss = .1 * (
        contrast(image_features, text_features, ids, ids) +
        contrast(text_features, image_features, ids, ids))
    g, optimizer, scaler, scheduler, _ = stage_objects("generator", args.ap_source, 11003)
    # 诊断只做一次实际 step；避免默认 65536 scale 的首次溢出。正式训练不变。
    scaler = amp.GradScaler(init_scale=1.)
    original = official_attack_functions(args.ap_source)
    before = {k: p.detach().cpu().clone() for k, p in g.named_parameters() if p.requires_grad}
    with amp.autocast():
        loss, reid, semantic_loss, adversarial = generator_loss(pixels, ids, g, ide, semantic, original)
    if not torch.isfinite(loss) or not torch.isfinite(inversion_forward_loss):
        raise RuntimeError("smoke loss 非有限")
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in g.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(x).all() for x in gradients):
        raise RuntimeError("G gradient 缺失或非有限；未执行 optimizer step")
    grad_l1 = sum(float(x.abs().sum()) for x in gradients)
    if grad_l1 <= 0:
        raise RuntimeError("G gradient 为零；未执行 optimizer step")
    atomic_json(output / "optimizer-step-intent.json", {"max_real_steps": 1, "retry_forbidden": True})
    scaler.step(optimizer)  # 本次任务唯一真实训练 step。
    scaler.update()
    changed = sum(not torch.equal(before[k], p.detach().cpu()) for k, p in g.named_parameters() if p.requires_grad)
    linf = float(((adversarial - pixels) * .5).abs().max())
    forbidden = [name for name in sys.modules if "irra" in name.lower()]
    frozen_clean = all(not p.requires_grad and p.grad is None
                       for m in (ide, semantic) for p in m.parameters())
    if not changed or linf > 8 / 255 + 1e-6 or forbidden or not frozen_clean:
        raise RuntimeError("smoke 参数/预算/冻结/训练图隔离失败")
    path = output / "resume.pt"
    save_resume(path, g, optimizer, scaler, scheduler, loader, spec, 0, mode="smoke")
    expected = {k: v.detach().cpu().clone() for k, v in g.state_dict().items()}
    expected_lr = optimizer.param_groups[0]["lr"]
    draws = torch.rand(4).cpu()
    with torch.no_grad():
        next(p for p in g.parameters() if p.requires_grad).add_(1)
    optimizer.param_groups[0]["lr"] = 9
    epoch = load_resume(path, g, optimizer, scaler, scheduler, loader, spec, mode="smoke")
    resume_ok = (epoch == 0 and optimizer.param_groups[0]["lr"] == expected_lr
                 and all(torch.equal(v, g.state_dict()[k].detach().cpu()) for k, v in expected.items())
                 and torch.equal(draws, torch.rand(4).cpu()))
    if not resume_ok:
        raise RuntimeError("smoke resume 不一致")
    result = {
        "scope": "one real G batch only; not trained G_CUHK",
        "real_optimizer_steps": 1, "smoke_amp_init_scale": 1., "dataset": spec["dataset"],
        "batch_shape": list(pixels.shape), "batch_id_count": len(ids.unique()),
        "ide_feature_shape": list(ide_features.shape),
        "pseudo_token_shape": list(tokens.shape),
        "inversion_forward_loss": float(inversion_forward_loss),
        "g_loss": float(loss.detach()), "reid_loss": float(reid.detach()),
        "semantic_loss": float(semantic_loss.detach()),
        "gradient_l1": grad_l1, "changed_trainable_tensors": changed,
        "max_linf": linf, "epsilon": 8 / 255,
        "frozen_models_have_no_grad": frozen_clean,
        "irra_imported_modules": forbidden, "resume_ok": resume_ok,
        "resume_sha256": sha256(path), "semantic_initialization": initialization,
        "smoke_ide": "ImageNet initialized; CUHK IDE not trained",
        "smoke_inversions": "fresh networks; CUHK inversion not trained",
        "formal_training_started": False,
    }
    atomic_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "smoke", "train", "status"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ap-source", type=Path, default=AP_SOURCE)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true", help="从完整 epoch checkpoint 恢复三个阶段")
    args = parser.parse_args()
    args.output_dir = write_path(args.output_dir)
    config = read_config(args.config)
    if args.command == "train":
        train(args, config)
    elif args.command == "smoke":
        smoke(args, config)
    elif args.command == "audit":
        _, spec = prepare(args, config)
        record_setup(args.output_dir, config, spec)
        print(json.dumps(spec, ensure_ascii=False, indent=2))
    else:
        state = {"formal_training_started": False,
                 "checkpoint_selection": config["checkpoint_selection"]}
        path = args.output_dir / "progress.json"
        if path.exists():
            state.update(json.loads(path.read_text()))
        smoke_path = args.output_dir / "smoke/result.json"
        state["smoke"] = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
        print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
