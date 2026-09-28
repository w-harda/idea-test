"""AP CUHK train-only、原损失语义及续跑边界测试。"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.cuda import amp

from attributes.ap_cuhk_training import (
    AP_SOURCE, TRAIN_HASHES, TrainImages, check_sources, fingerprint,
    load_resume, make_loader, official_attack_functions, validate_fixed_protocol,
    save_resume, train_index,
)
from attributes.ap_gallery_baseline import sha256
from scripts.run_ap_cuhk_irra_baseline import cuhk_provenance
from scripts.train_ap_cuhk_baseline import DEFAULT_CONFIG, read_config


def dataset(tmp_path, rows=None):
    root = tmp_path / "imgs"
    root.mkdir()
    for name in ("a.png", "b.png", "c.png", "d.png"):
        Image.new("RGB", (31, 63), (30, 50, 80)).save(root / name)
    if rows is None:
        rows = [
            {"split": "train", "id": 91, "file_path": "a.png", "captions": {"never": "read"}},
            {"split": "train", "id": 2, "file_path": "b.png"},
            {"split": "val", "id": 92, "file_path": "nonexistent-val-image"},
            {"split": "test", "id": 93, "file_path": "nonexistent-test-image"},
        ]
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps(rows), encoding="utf-8")
    return annotation, root


def test_index_returns_train_only_without_caption_or_camera(tmp_path):
    path, root = dataset(tmp_path)
    records, meta = train_index(path, root, expected_counts=False)
    assert [(r.path, r.person_id, r.original_id) for r in records] == [
        ("a.png", 1, 91), ("b.png", 0, 2)]
    assert meta["training_fields"] == ["file_path", "id"]
    assert meta["captions_used"] is False and meta["camera_annotation"] is None
    assert meta["split_image_counts"] == {"train": 2, "val": 1, "test": 1}
    assert meta["train_test_identity_overlap"] == 0


@pytest.mark.parametrize("change,reason", [
    (lambda rows: rows[-1].update(id=91), "identity"),
    (lambda rows: rows[-1].update(file_path="a.png"), "图像"),
    (lambda rows: rows.append(rows[0].copy()), "重复"),
    (lambda rows: rows[0].update(file_path="../escape.png"), "路径"),
])
def test_leakage_and_bad_path_rejected(tmp_path, change, reason):
    path, root = dataset(tmp_path)
    rows = json.loads(path.read_text())
    change(rows)
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        train_index(path, root, expected_counts=False)


def test_official_count_guard(tmp_path):
    path, root = dataset(tmp_path)
    with pytest.raises(ValueError, match="数量"):
        train_index(path, root)


def test_default_fixed_protocol_allows_training_without_camera():
    config = read_config(DEFAULT_CONFIG)
    validate_fixed_protocol(config)
    assert config["checkpoint_selection"]["epochs"] == {
        "ide": 50, "inversion": 20, "generator": 60}
    assert config["checkpoint_selection"]["validation"] == "none"
    assert "user_accepted_adaptation" not in config["checkpoint_selection"]


@pytest.mark.parametrize("field,value", [
    ("validation", "IRRA/test"), ("policy", "camera_aware_best"),
    ("epochs", {"ide": 50, "inversion": 40, "generator": 60}),
])
def test_fixed_protocol_rejects_feedback_selection_and_wrong_epochs(field, value):
    config = read_config(DEFAULT_CONFIG)
    config["checkpoint_selection"][field] = value
    with pytest.raises(ValueError):
        validate_fixed_protocol(config)


def test_cannot_tune_frozen_recipe(tmp_path):
    config = read_config(DEFAULT_CONFIG)
    config["generator"]["reid_weight"] = 11
    import yaml
    path = tmp_path / "different.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="recipe"):
        read_config(path)


@pytest.mark.skipif(not AP_SOURCE.exists(), reason="真实 AP 源码仅在服务器可用")
def test_pinned_recipe_sources():
    assert check_sources(AP_SOURCE) == TRAIN_HASHES


@pytest.mark.skipif(not AP_SOURCE.exists(), reason="真实 AP 源码仅在服务器可用")
def test_original_loss_prefers_farthest_negative_and_keeps_gradient_semantics():
    original = official_attack_functions(AP_SOURCE)
    clean = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    ids = torch.tensor([0, 1, 2])
    unchanged = original["adv_TripletLoss"]()(clean, clean, ids)
    attacking = clean[[2, 0, 0]].clone().requires_grad_()
    moved = original["adv_TripletLoss"]()(clean, attacking, ids)
    assert moved < unchanged
    moved.backward()
    assert attacking.grad is not None and torch.isfinite(attacking.grad).all()
    delta = torch.tensor([[[[.01]], [[.01]], [[.01]]]], requires_grad=True)
    scaled = original["L_norm"](delta)
    assert torch.allclose(scaled.detach(), torch.full_like(delta, .02))
    scaled.sum().backward()
    # 原 .data 除 std 的值/梯度语义不能被悄悄改成功能性除法。
    assert torch.equal(delta.grad, torch.ones_like(delta))


@pytest.mark.skipif(not AP_SOURCE.exists(), reason="真实 AP 源码仅在服务器可用")
def test_pk_loader_uses_train_ids_and_original_tensor_shape(tmp_path):
    path, root = dataset(tmp_path)
    records, _ = train_index(path, root, expected_counts=False)
    loader = make_loader(records, root, AP_SOURCE, "generator",
                         read_config(DEFAULT_CONFIG), batch_size=8, workers=0)
    pixels, ids = next(iter(loader))
    assert pixels.shape == (8, 3, 256, 128)
    assert sorted(torch.bincount(ids).tolist()) == [4, 4]
    assert pixels.min() >= -1 and pixels.max() <= 1


def test_resume_restores_optimizer_scheduler_rng_and_rejects_wrong_mode(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.2, momentum=.9)
    scaler = amp.GradScaler(enabled=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    loader = torch.utils.data.DataLoader(torch.zeros(2, 2), batch_size=2,
                                        generator=torch.Generator().manual_seed(1234))
    # 小型 mock step 用于验证 optimizer 状态，不是 AP/G 真实训练。
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    spec = {"stage": "unit", "data": "train-only"}
    path = tmp_path / "resume.pt"
    save_resume(path, model, optimizer, scaler, scheduler, loader, spec, 7)
    expected = {k: x.clone() for k, x in model.state_dict().items()}
    lr = optimizer.param_groups[0]["lr"]
    next_values = (random.random(), np.random.rand(), torch.rand(3))
    with torch.no_grad():
        model.weight.zero_()
    optimizer.param_groups[0]["lr"] = 9.
    assert load_resume(path, model, optimizer, scaler, scheduler, loader, spec) == 7
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())
    assert optimizer.param_groups[0]["lr"] == lr
    assert scheduler.last_epoch == 1
    assert random.random() == next_values[0]
    assert np.random.rand() == next_values[1]
    assert torch.equal(torch.rand(3), next_values[2])
    with pytest.raises(ValueError, match="模式"):
        load_resume(path, model, optimizer, scaler, scheduler, loader, spec, mode="smoke")
    with pytest.raises(ValueError):
        load_resume(path, model, optimizer, scaler, scheduler, loader, {"data": "test"})


def test_cuhk_evaluator_rejects_smoke_and_hash_mismatch(tmp_path):
    path = tmp_path / "generator.pth.tar"
    torch.save({"fake": torch.zeros(1)}, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = {"mode": "smoke", "training_dataset": "CUHK-PEDES/train",
                "checkpoint_sha256": sha256(path)}
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="smoke"):
        cuhk_provenance(path)
    metadata["mode"] = "formal"
    sidecar.write_text(json.dumps(metadata))
    assert cuhk_provenance(path) == metadata
    with pytest.raises(ValueError):
        cuhk_provenance(path, "0" * 64)

def test_epoch_checkpoint_export_precedes_resume_and_completed_stage_skips_steps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import train_ap_cuhk_baseline as runner
    import copy

    class TinyIDE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Linear(3, 2)

        def forward(self, pixels, is_training=False):
            features = pixels.mean((2, 3))
            return self.head(features), features

    def objects(*_):
        model = TinyIDE()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        return model, optimizer, amp.GradScaler(enabled=False), None, {}

    original_save = runner.save_resume

    def save_after_export(path, *args, **kwargs):
        assert (path.parent / "epoch_1.pth").is_file()
        return original_save(path, *args, **kwargs)

    monkeypatch.setattr(runner, "save_resume", save_after_export)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.randn(2, 3, 8, 8), torch.tensor([0, 1])), batch_size=2)
    monkeypatch.setattr(runner, "make_loader", lambda *_args, **_kwargs: loader)
    monkeypatch.setattr(runner, "stage_objects", objects)
    monkeypatch.setattr(runner, "seed_all", lambda *_: None)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *_a, **_k: self)
    config = copy.deepcopy(read_config(DEFAULT_CONFIG))
    config["ide"]["epochs"] = config["ide"]["save_period"] = 1
    args = SimpleNamespace(output_dir=tmp_path / "successful", image_root=tmp_path,
                           ap_source=AP_SOURCE, resume=False)
    spec = {"dataset": {"split_identity_counts": {"train": 2}}}
    model = runner.train_stage(args, config, [], spec, "ide")
    assert (args.output_dir / "ide/epoch_1.pth.json").is_file()
    assert json.loads((args.output_dir / "progress.json").read_text())["epoch"] == 1
    args.resume = True
    monkeypatch.setattr(torch.optim.SGD, "step", lambda *_a, **_k: pytest.fail("已完成阶段不应再训练"))
    restored = runner.train_stage(args, config, [], spec, "ide")
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())

@pytest.mark.skipif(not AP_SOURCE.exists(), reason="真实 AP scheduler 仅在服务器可用")
def test_inversion_epoch20_keeps_original_cosine_schedule():
    from scripts.train_ap_cuhk_baseline import inversion_scheduler
    config = read_config(DEFAULT_CONFIG)
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=.005)
    scheduler = inversion_scheduler(optimizer, AP_SOURCE, config)
    assert config["inversion"]["epochs"] == 20
    assert scheduler.t_initial == 40 and scheduler.warmup_t == 10
    assert scheduler._get_lr(20)[0] == pytest.approx(.0026)


def test_setup_migrates_audit_only_metadata_but_protects_formal_training(tmp_path):
    import copy
    import yaml
    from scripts.train_ap_cuhk_baseline import record_setup
    config = read_config(DEFAULT_CONFIG)
    previous = copy.deepcopy(config)
    previous["checkpoint_selection"] = {"policy": "unresolved_camera_aware_reid"}
    previous["inversion"]["epochs"] = 40
    previous_spec = {"dataset": {"training_split": "train"}, "version": "audit-only"}
    record_setup(tmp_path, previous, previous_spec)
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke/resume.pt").write_bytes(b"historical smoke")
    spec = {"dataset": {"training_split": "train"}, "version": "fixed epochs"}
    record_setup(tmp_path, config, spec)
    archived = list((tmp_path / "setup-history").glob("*/config.yaml"))
    assert len(archived) == 1 and yaml.safe_load(archived[0].read_text()) == previous
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == config
    assert (tmp_path / "smoke/resume.pt").read_bytes() == b"historical smoke"
    (tmp_path / "ide").mkdir()
    (tmp_path / "ide/last.pt").write_bytes(b"formal checkpoint")
    with pytest.raises(ValueError, match="正式训练"):
        record_setup(tmp_path, previous, previous_spec)
    assert yaml.safe_load((tmp_path / "config.yaml").read_text()) == config


def test_generator_dependencies_reject_wrong_epoch_hash_or_source(tmp_path):
    from scripts.train_ap_cuhk_baseline import (
        export_state, generator_dependencies, selected_checkpoints)
    config = read_config(DEFAULT_CONFIG)
    spec = {"dataset": {"training_split": "train"}, "project_commit": "unit"}
    model = torch.nn.Linear(2, 2)
    paths = selected_checkpoints(tmp_path, config)
    for stage in ("ide", "inversion"):
        export_state(paths[stage], model, {
            "mode": "formal", "stage": stage,
            "epoch": config["checkpoint_selection"]["epochs"][stage], "spec": spec})
    dependencies = generator_dependencies(tmp_path, config, spec)
    assert dependencies["ide"]["epoch"] == 50
    assert dependencies["inversion"]["epoch"] == 20
    sidecar = paths["inversion"].with_suffix(".pth.json")
    original = json.loads(sidecar.read_text())
    for key, value in (("epoch", 19), ("checkpoint_sha256", "0" * 64),
                       ("spec", {"dataset": {"training_split": "test"}}),
                       ("mode", "smoke")):
        changed = {**original, key: value}
        sidecar.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="依赖"):
            generator_dependencies(tmp_path, config, spec)
    sidecar.write_text(json.dumps(original))


def test_three_stage_entry_selects_50_20_60_and_exports_complete_provenance(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import train_ap_cuhk_baseline as runner
    config = read_config(DEFAULT_CONFIG)
    spec = {
        "config": config, "project_commit": "unit_mock",
        "dataset": {"split_image_counts": {"train": 34054},
                    "split_identity_counts": {"train": 11003}},
        "unit_mock": True,
    }
    calls = []
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "prepare", lambda *_: ([], spec))

    def fake_stage(args, cfg, records, current_spec, stage, dependencies=None):
        epoch = cfg[stage]["epochs"]
        path = runner.selected_checkpoints(args.output_dir, cfg)[stage]
        model = torch.nn.Linear(2, 2)
        if args.resume:
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        else:
            runner.export_state(path, model, {
                "mode": "formal", "stage": stage, "epoch": epoch,
                "spec": {**current_spec, "stage": stage, "dependencies": dependencies or {}}})
        calls.append((stage, epoch, dependencies))
        return model

    monkeypatch.setattr(runner, "train_stage", fake_stage)
    args = SimpleNamespace(output_dir=tmp_path, resume=False)
    runner.train(args, config)
    assert [(stage, epoch) for stage, epoch, _ in calls] == [
        ("ide", 50), ("inversion", 20), ("generator", 60)]
    assert calls[-1][2]["ide"]["epoch"] == 50
    assert calls[-1][2]["inversion"]["epoch"] == 20
    metadata = json.loads((tmp_path / "G_CUHK_AP.pth.tar.json").read_text())
    assert metadata["generator_epoch"] == 60 and metadata["seed"] == 1234
    assert metadata["train_images"] == 34054 and metadata["train_identities"] == 11003
    assert metadata["project_commit"] == "unit_mock"
    assert metadata["surrogate_source"]["epoch"] == 50
    assert metadata["inversion_source"]["epoch"] == 20
    assert metadata["semantic_backbone"]["training_dataset"] == "DukeMTMC-reID"
    assert metadata["training_uses_irra"] is False
    assert metadata["checkpoint_sha256"] == sha256(tmp_path / "G_CUHK_AP.pth.tar")
    assert (tmp_path / "G_CUHK_AP.pth.tar").read_bytes() == (tmp_path / "best_G_V.pth.tar").read_bytes()
    assert runner.training_plan(tmp_path, config)["training_allowed"] is True
    with pytest.raises(ValueError, match="--resume"):
        runner.train(args, config)
    args.resume = True
    runner.train(args, config)
    assert len(calls) == 6
