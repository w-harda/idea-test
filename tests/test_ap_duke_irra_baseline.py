"""AP 整图库冻结生成与 IRRA TBPS 协议的行为测试。"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch
from PIL import Image

from attributes import ap_gallery_baseline as ap
from attributes.tbps_retrieval_metrics import (
    IRRA_METRICS_SHA, paired_summary, text_to_image_metrics,
)


def test_generator_only_accepts_pixels_and_exact_official_l_norm():
    class Delta(torch.nn.Module):
        def forward(self, pixels):
            self.seen = pixels.clone()
            return torch.full_like(pixels, 0.1)

    generator = ap.FrozenAPGenerator.__new__(ap.FrozenAPGenerator)
    generator.generator = Delta()
    generator.device = torch.device("cpu")
    clean = torch.full((2, 3, 256, 128), 0.5)
    clean[0, :, 0, 0] = 0.99
    adv = generator(clean)
    assert torch.equal(generator.generator.seen, (clean - 0.5) / 0.5)
    delta_normalized = torch.full_like(clean, 0.1).clamp(-8 / 255, 8 / 255) / 0.5
    official_normalized = ((clean - 0.5) / 0.5 + delta_normalized).clamp(-1, 1)
    assert torch.allclose(adv, official_normalized * 0.5 + 0.5, atol=1e-7, rtol=0)
    assert float((adv - clean).abs().max()) <= ap.EPSILON + 1e-6
    assert list(inspect.signature(ap.FrozenAPGenerator.__call__).parameters) == ["self", "pixels"]


@pytest.mark.parametrize("invalid", [
    torch.ones(1, 3, 384, 128), torch.full((1, 3, 256, 128), 1.1),
    torch.full((1, 3, 256, 128), float("nan")),
])
def test_generator_rejects_wrong_space_or_pixels(invalid):
    generator = ap.FrozenAPGenerator.__new__(ap.FrozenAPGenerator)
    with pytest.raises(ValueError, match="AP 输入"):
        generator(invalid)


def test_delta_transport_preserves_clean_and_does_not_amplify():
    clean = torch.full((2, 3, 256, 128), 0.5)
    victim = torch.rand(2, 3, 384, 128)
    zero, linf = ap.transport_delta(clean, clean, victim)
    assert torch.equal(zero, victim)
    assert linf == [0, 0]
    delta = torch.full_like(clean, ap.EPSILON)
    delta[:, :, ::2] *= -1
    adv, linf = ap.transport_delta(clean, clean + delta, victim)
    assert adv.shape == victim.shape
    assert max(linf) <= ap.EPSILON + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1
    with pytest.raises(ValueError, match="超过"):
        ap.transport_delta(clean, clean + 9 / 255, victim)


class StatefulGenerator:
    """模拟官方 eval 仍更新 SpectralNorm 推理状态。"""
    def __init__(self):
        self.counter = 0

    def __call__(self, pixels):
        self.counter += 1
        return pixels + ap.EPSILON * self.counter / 10

    def state(self):
        return {"counter": self.counter}

    def restore(self, state):
        self.counter = state["counter"]


@pytest.fixture
def cache_inputs(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    paths = tuple(f"{i}.png" for i in range(4))
    for path in paths:
        Image.new("RGB", (12, 24), (128, 128, 128)).save(images / path)
    spec = {"batch_size": 2, "gallery_paths": list(paths)}
    return images, paths, spec


def cache_tensor(directory, spec, count=4):
    return torch.cat([row[2] for row in ap.cached_batches(directory, spec, count)])


def test_gallery_once_shared_by_queries_and_resume_preserves_state(tmp_path, cache_inputs):
    images, paths, spec = cache_inputs
    calls = []
    def factory():
        calls.append(True)
        return StatefulGenerator()
    cache = tmp_path / "cache"
    first = ap.build_cache(cache, spec, paths, images, factory, max_images=2)
    assert first["done"] == 2 and not first["complete"]
    with pytest.raises(ValueError, match="尚未覆盖"):
        list(ap.cached_batches(cache, spec, 4))
    completed = ap.build_cache(cache, spec, paths, images, factory)
    assert completed["done"] == 4 and completed["complete"]
    assert ap.read_progress(cache, spec)["state"]["counter"] == 2
    full = tmp_path / "full"
    ap.build_cache(full, spec, paths, images, StatefulGenerator)
    assert torch.equal(cache_tensor(cache, spec), cache_tensor(full, spec))
    ap.build_cache(cache, spec, paths, images, factory)
    assert len(calls) == 2  # 缓存完成后不创建 G；caption 数量不会影响生成。


def test_interrupted_commit_restores_last_state(tmp_path, cache_inputs, monkeypatch):
    images, paths, spec = cache_inputs
    cache = tmp_path / "interrupted"
    real_save = ap.atomic_torch
    def interrupted_save(path, value):
        if path.name == "progress.pt" and value["done"] == 4:
            raise RuntimeError("模拟写入中断")
        real_save(path, value)
    monkeypatch.setattr(ap, "atomic_torch", interrupted_save)
    with pytest.raises(RuntimeError, match="模拟"):
        ap.build_cache(cache, spec, paths, images, StatefulGenerator)
    assert ap.read_progress(cache, spec)["done"] == 2
    monkeypatch.setattr(ap, "atomic_torch", real_save)
    ap.build_cache(cache, spec, paths, images, StatefulGenerator)
    reference = tmp_path / "reference"
    ap.build_cache(reference, spec, paths, images, StatefulGenerator)
    assert torch.equal(cache_tensor(cache, spec), cache_tensor(reference, spec))


def test_cache_rejects_changed_spec_or_corruption(tmp_path, cache_inputs):
    images, paths, spec = cache_inputs
    cache = tmp_path / "cache"
    ap.build_cache(cache, spec, paths, images, StatefulGenerator)
    with pytest.raises(ValueError, match="配置不匹配"):
        ap.read_progress(cache, {**spec, "checkpoint_sha256": "different"})
    chunk = next(cache.glob("gallery_*.pt"))
    chunk.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="损坏"):
        ap.read_progress(cache, spec)


def test_writes_outside_ldx_rejected():
    with pytest.raises(ValueError, match="所有写入"):
        ap.atomic_json(Path("/home/lzf/TBPS/forbidden.json"), {})
    with pytest.raises(ValueError, match="所有写入"):
        ap.write_path(Path("/home/lzf/ldx/../Attack/forbidden.json"))


def test_tbps_uses_all_same_id_gallery_images_and_last_positive_for_inp():
    scores = torch.tensor([[12., 11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1.]])
    # 同 ID 正样本在第 2、8 位；不排除 query 原配对图或 camera。
    ids = [9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9]
    metrics, rows = text_to_image_metrics(scores, [1], ids)
    assert rows[0]["first_hit_rank"] == 2
    assert rows[0]["positive_images"] == 2
    assert rows[0]["AP"] == pytest.approx((1 / 2 + 2 / 8) / 2)
    assert rows[0]["INP"] == pytest.approx(2 / 8)
    assert metrics["R@1"] == 0 and metrics["R@5"] == 100
    assert metrics["mAP"] == pytest.approx(37.5)
    assert metrics["mINP"] == pytest.approx(25)


def test_metrics_match_untouched_irra_rank():
    path = Path("/home/lzf/Attack/IRRA-main/IRRA-main/utils/metrics.py")
    if not path.exists():
        pytest.skip("服务器外没有官方 IRRA 源码")
    assert ap.sha256(path) == IRRA_METRICS_SHA
    # 提取未经改动的 rank AST，避免与指标无关的 PrettyTable 展示依赖。
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rank")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    official_rank = namespace["rank"]
    torch.manual_seed(42)
    scores = torch.randn(9, 15)
    query_ids = torch.tensor([0, 1, 2] * 3)
    gallery_ids = torch.tensor([0, 1, 2] * 5)
    cmc, map_, minp, _ = official_rank(scores, query_ids, gallery_ids)
    ours, _ = text_to_image_metrics(scores, query_ids, gallery_ids)
    for k in (1, 5, 10):
        assert ours[f"R@{k}"] == pytest.approx(float(cmc[k - 1]), abs=1e-5)
    assert ours["mAP"] == pytest.approx(float(map_), abs=1e-5)
    assert ours["mINP"] == pytest.approx(float(minp), abs=1e-5)


def test_missing_positive_and_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match="不在图库"):
        text_to_image_metrics(torch.zeros(1, 2), [3], [1, 2])
    with pytest.raises(ValueError, match="无效"):
        text_to_image_metrics(torch.tensor([[float("nan")]]), [1], [1])


def test_single_victim_drop_rate_and_delta_signs():
    clean = [{"row_id": "a", "person_id": 1, "first_hit_rank": 1, "AP": 0.8, "INP": 0.5},
             {"row_id": "b", "person_id": 2, "first_hit_rank": 4, "AP": 0.4, "INP": 0.2}]
    adv = [{**clean[0], "first_hit_rank": 3, "AP": 0.4},
           {**clean[1], "first_hit_rank": 2, "AP": 0.2}]
    result = paired_summary(clean, adv)
    assert result["DR_mAP"] == pytest.approx(0.5)
    assert result["diagnostics"]["mean_delta_rank"] == 0
    assert result["diagnostics"]["delta_rank_positive"]["count"] == 1
    assert result["diagnostics"]["delta_rank_negative"]["count"] == 1
    assert "aAP" not in result and "mDR" not in result
    with pytest.raises(ValueError, match="顺序"):
        paired_summary(clean, adv[::-1])


def test_formal_evaluation_requires_full_adversarial_gallery(monkeypatch):
    from scripts import run_ap_duke_irra_baseline as runner
    paths = tuple(str(i) for i in range(3074))
    monkeypatch.setattr(runner, "generator_spec", lambda args: (paths, {}))
    monkeypatch.setattr(runner, "selected_protocol", lambda *args: ([], paths, [1] * 3074))
    monkeypatch.setattr(runner, "read_progress", lambda *args: {"done": 4})
    with pytest.raises(ValueError, match="整个评测图库"):
        runner.evaluate(None, Path("/home/lzf/ldx/not-created"), "full")


def test_partial_generation_stops_only_on_fixed_batch_boundary(tmp_path, cache_inputs):
    images, paths, spec = cache_inputs
    with pytest.raises(ValueError, match="固定 batch 边界"):
        ap.build_cache(tmp_path / "partial", spec, paths, images,
                       StatefulGenerator, max_images=1)


def test_real_protocol_manifest_counts_without_running_models():
    from scripts import run_ap_duke_irra_baseline as runner
    from types import SimpleNamespace

    if not runner.ANNOTATION.exists():
        pytest.skip("服务器外没有 CUHK 标注")
    args = SimpleNamespace(annotation=runner.ANNOTATION,
                           query_manifest=runner.MANIFEST)
    gallery = ap.gallery_paths(args.annotation)
    full, paths, ids = runner.selected_protocol(args, "full", gallery)
    fixed, same_paths, same_ids = runner.selected_protocol(args, "500", gallery)
    assert len(full) == 6156 and len(fixed) == 500
    assert len(paths) == 3074 and len(set(ids)) == 1000
    assert same_paths == paths and same_ids == ids
    assert len({q.row_id for q in fixed}) == 500
    assert [q.row_id for q in fixed[:20]] == [q.row_id for q in full[:20]]
