"""一次性真实IRRA等价Smoke；默认仅CPU来源检查，不包含攻击/训练/排名。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

from .artifacts import ArtifactError, ArtifactStore, checked, digest, json_bytes, read_json, sha_file
from .contracts import FileRef
from .datasets.cuhk_pedes import CuhkPedes
from .datasets.historical_membership import load_historical_partitions
from .models.irra_source import IRRASource

LIMITS = {"image": 5, "text": 4}
ROW_IDS = ("cuhk:9955:0", "cuhk:29253:0")
THRESHOLD = 1e-6
BASE = "/home/lzf/ldx/outputs/tbps-framework-stage25"


def require(ok, message):
    if not ok:
        raise ArtifactError(message)


def resolve_config(path):
    ref = FileRef(str(Path(path).resolve()), sha_file(path))
    config = read_json(path)
    require(set(config) == {"schema", "sources", "row_ids", "limits", "max_abs_error",
                            "minimum_free_mib", "maximum_utilization_percent", "precision",
                            "physical_gpu", "membership_sha256"}, "Smoke config fields")
    require(config["schema"] == "tbps-irra-smoke-1", "Smoke schema")
    require(tuple(config["row_ids"]) == ROW_IDS and config["limits"] == LIMITS,
            "fixed two Train samples and 5/4 limits required")
    require(config["max_abs_error"] == THRESHOLD and config["precision"] == "fp32/math/no_tf32",
            "fixed FP32 and 1e-6 threshold required")
    require(type(config["minimum_free_mib"]) is int and config["minimum_free_mib"] >= 8192,
            "minimum free memory cannot be lowered")
    require(type(config["maximum_utilization_percent"]) is int and
            0 <= config["maximum_utilization_percent"] <= 10, "GPU utilization guard")
    require(type(config["physical_gpu"]) is int and config["physical_gpu"] == 0, "GPU 0 only")
    source_ref = FileRef(**config["sources"])
    checked(source_ref)
    sources = read_json(source_ref.path, "tbps-stage2-cpu-sources-1")
    source = IRRASource.from_dict(sources["irra"])
    source_report = source.verify()
    dataset = CuhkPedes(FileRef(**sources["annotation"]), sources["image_root"])
    partitions = load_historical_partitions(dataset, {
        k: FileRef(**v) for k, v in sources["membership"].items()})
    require(partitions.fingerprint == config["membership_sha256"], "membership digest changed")
    queries = {q.query_id: q for q in partitions.train.queries}
    gallery = {g.image_id: g for g in partitions.train.gallery}
    samples = []
    inputs = []
    for row_id in ROW_IDS:
        q = partitions.require_member(queries[row_id], "attack_train")
        g = partitions.require_member(gallery[q.paired_image_id], "attack_train")
        require(q.split == g.split == "train", "only official Train allowed")
        public = dataset.public_image(partitions.train, g.image_id)
        image_ref = FileRef(str(dataset.image_path(g.image_rel_path)), g.image_sha256)
        checked(image_ref)
        inputs.append(image_ref)
        samples.append({"row_id": q.query_id, "image_rel_path": g.image_rel_path,
                        "image": asdict(image_ref), "caption": q.caption,
                        "caption_sha256": hashlib.sha256(q.caption.encode("utf-8")).hexdigest(),
                        "caption_index": q.caption_index, "pid": asdict(q.pid),
                        "official_split": q.split, "role": q.role})
    require(len({s["image_rel_path"] for s in samples}) == 2, "two distinct images required")
    root = Path(__file__).resolve().parents[2]
    producers = tuple(FileRef(str(root / rel), sha_file(root / rel)) for rel in (
        "src/tbps/smoke_irra.py", "src/tbps/artifacts.py", "src/tbps/models/irra.py",
        "src/tbps/models/irra_io.py", "src/tbps/models/irra_source.py",
        "src/tbps/models/irra_cpu_checks.py", "src/tbps/models/base.py",
        "src/tbps/datasets/cuhk_pedes.py", "src/tbps/datasets/historical_membership.py",
        "src/tbps/contracts.py"))
    refs = (ref, source_ref, dataset.annotation, *partitions.sources, *partitions.metadata_refs,
            source.checkpoint, source.config, source.clip_bootstrap, source.bpe,
            *(r for _, r in source.code), *inputs, *producers)
    refs = tuple({r.path: r for r in refs}.values())
    before = {r.path: checked(r) for r in refs}
    report = {"schema": "tbps-irra-smoke-preflight-1", "samples": samples,
              "membership_sha256": partitions.fingerprint,
              "source_fingerprint": source.fingerprint,
              "source": source_report, "config_sha256": ref.sha256,
              "limits": LIMITS, "max_abs_error": THRESHOLD,
              "model_loaded": False, "model_forwards": 0, "official_test_used": False}
    return config, source, refs, before, report


def gpu_available(config):
    """在Torch/CUDA加载前检查物理GPU；不终止或修改其他进程。"""
    command = ["nvidia-smi", "-i", "0", "--query-gpu=name,memory.free,memory.used,utilization.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = [r.strip() for r in result.stdout.splitlines() if r.strip()]
    require(len(rows) == 1, "single GPU resource report required")
    name, free, used, util = [s.strip() for s in rows[0].split(",")]
    require(name == "NVIDIA GeForce RTX 4090", "authorized RTX 4090 required")
    free, used, util = int(free), int(used), int(util)
    processes = subprocess.run(["nvidia-smi", "-i", "0",
        "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout.strip()
    require(not processes, "GPU has existing compute processes; safe stop")
    require(free >= config["minimum_free_mib"] and util <= config["maximum_utilization_percent"],
            "GPU resources insufficient; safe stop")
    return {"name": name, "free_mib": free, "used_mib": used, "utilization_percent": util,
            "compute_processes": []}


class ForwardBudget:
    """只保留标量，不保存Tensor参数/图；包括独立官方路径的所有样本调用。"""
    def __init__(self):
        self.started = dict.fromkeys(LIMITS, 0)
        self.completed = dict.fromkeys(LIMITS, 0)
        self.calls = []
        self.phase = "unset"

    def wrap(self, kind, function):
        def call(tensor):
            import torch
            n = len(tensor)
            require(kind in LIMITS and 0 < n <= 2, "only small nonempty batches allowed")
            require(self.started[kind] + n <= LIMITS[kind], "forward sample budget exceeded")
            self.started[kind] += n
            record = {"kind": kind, "phase": self.phase, "samples": n,
                      "grad_enabled": torch.is_grad_enabled(),
                      "inference_mode": torch.is_inference_mode_enabled(), "returned": False}
            self.calls.append(record)
            result = function(tensor)
            self.completed[kind] += n
            record["returned"] = True
            return result
        return call

    def report(self):
        return {"sample_forwards_started": dict(self.started),
                "sample_forwards_completed": dict(self.completed), "calls": list(self.calls)}


def compare(a, b, *, exact=False):
    import torch
    require(a.shape == b.shape and a.dtype == b.dtype, "shape/dtype mismatch")
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "nonfinite comparison")
    error = float((a.detach().cpu() - b.detach().cpu()).abs().max())
    if exact:
        require(torch.equal(a.detach().cpu(), b.detach().cpu()), "exact comparison failed")
    require(error <= THRESHOLD, "max_abs_error exceeds fixed 1e-6 threshold: " + str(error))
    return {"shape": list(a.shape), "dtype": str(a.dtype), "finite": True,
            "max_abs_error": error, "exact": bool(torch.equal(a.detach().cpu(), b.detach().cpu()))}


def feature_comparison(a, b):
    import torch
    report = compare(a, b)
    require(a.ndim == 2 and a.shape == (2, 512) and a.dtype == torch.float32,
            "expected two FP32 native retrieval features")
    report["official_l2_norms"] = a.norm(dim=-1).detach().cpu().tolist()
    report["adapter_l2_norms"] = b.norm(dim=-1).detach().cpu().tolist()
    require(all(abs(n - 1) <= THRESHOLD for n in
                report["official_l2_norms"] + report["adapter_l2_norms"]), "feature norm mismatch")
    return report


def model_snapshot(model):
    """逐Tensor流式hash；不保存整个GPU模型副本。"""
    import torch
    hashes = {}
    for kind, sequence in (("parameters", model.named_parameters()), ("buffers", model.named_buffers())):
        h = hashlib.sha256()
        count = 0
        for name, tensor in sequence:
            require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),
                    "nonfinite model state")
            h.update(json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
            cpu = tensor.detach().cpu().contiguous()
            h.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
            del cpu
        hashes[kind] = {"sha256": h.hexdigest(), "tensor_count": count}
    require(all(not m.training for m in model.modules()), "IRRA left eval mode")
    require(all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "IRRA parameter gradient/update forbidden")
    hashes["eval"] = True
    hashes["requires_grad"] = False
    hashes["parameter_grads_none"] = True
    return hashes


def gradient_report(pixels):
    import torch
    grad = pixels.grad
    require(grad is not None and bool(torch.isfinite(grad).all()) and bool((grad != 0).any()),
            "input gradient missing/nonfinite/zero")
    return {"shape": list(grad.shape), "finite": True, "nonzero": True,
            "l2_norm": float(grad.norm()), "max_abs": float(grad.abs().max()),
            "probe": "normalized image feature [0,0]; generic derivative, no attack objective"}


def independent_tokenizer(source):
    # 独立实例和官方tokenize函数；不用Adapter包装器生成reference IDs。
    ref = source.ref("utils/simple_tokenizer.py")
    checked(ref)
    spec = importlib.util.spec_from_file_location("_tbps_smoke_native_tokenizer", ref.path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
        result = module.SimpleTokenizer(source.bpe.path)
    finally:
        sys.dont_write_bytecode = old
    checked(ref)
    return result


def run_once(config, source, refs, before, preflight, output):
    store = ArtifactStore(output, new_base=BASE)
    # 任一intent/pending/结果存在都拒绝第二次GPU运行，不自动恢复或重试。
    require(not store.root.exists() or not any(store.root.iterdir()), "one-attempt output already used")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "set CUDA_VISIBLE_DEVICES=0 explicitly")
    resources = gpu_available(config)
    store.publish("attempt", {"preflight.json": json_bytes(preflight),
                            "resources_before.json": json_bytes(resources)},
                  {"purpose": "one authorized equivalence smoke", "config": preflight["config_sha256"]},
                  sources=refs)
    counters = ForwardBudget()
    report = {"schema": "tbps-irra-smoke-result-1", "status": "RUNNING", "models_loaded": 0,
              "model_load_attempts": 0, "preflight": preflight,
              "resources_before": resources, "optimizer_steps": 0, "attack_forwards": 0}
    backend = None
    started = time.perf_counter()
    original_signal = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise InterruptedError("SIGTERM: preserve one-attempt evidence; no automatic retry")
    signal.signal(signal.SIGTERM, stop)
    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image
        from torch.nn import functional as F
        from unittest.mock import patch
        from .models import irra as adapter_module
        from .models.irra import IRRAAdapter, fp32_runtime
        from .models.irra_io import pixels_from_pil, normalize_pixels
        from .models.irra_cpu_checks import official_function
        require(not torch.cuda.is_initialized(), "isolated process with uninitialized CUDA required")
        torch.manual_seed(42)
        native_transform = official_function(source, "datasets/build.py", "build_transforms",
                                              {"T": T})((384, 128), is_train=False)
        native_tokenize = official_function(source, "datasets/bases.py", "tokenize", {"torch": torch})
        native_tokenizer = independent_tokenizer(source)
        raw, official_raw, normalized = [], [], []
        captions = tuple(s["caption"] for s in preflight["samples"])
        for sample in preflight["samples"]:
            checked(FileRef(**sample["image"]))
            with Image.open(sample["image"]["path"]) as image:
                rgb = image.convert("RGB")
                raw.append(pixels_from_pil(rgb))
                official_raw.append(T.Compose(native_transform.transforms[:2])(rgb))
                normalized.append(native_transform(rgb))
        pixels = torch.stack(raw)
        native_pixels = torch.stack(official_raw)
        native_normalized = torch.stack(normalized)
        report["raw_preprocessing"] = compare(native_pixels, pixels, exact=True)
        report["normalized_preprocessing"] = compare(native_normalized, normalize_pixels(pixels), exact=True)
        native_ids = torch.stack([native_tokenize(c, native_tokenizer, text_length=77, truncate=True)
                                  for c in captions])
        report["resources_at_load"] = gpu_available(config)
        original_load = adapter_module.load_official_model
        def load_one(*args, **kwargs):
            require(report["model_load_attempts"] == 0, "only one backend load allowed")
            report["model_load_attempts"] += 1
            torch.cuda.reset_peak_memory_stats()
            model = original_load(*args, **kwargs)
            report["models_loaded"] += 1
            return model
        print("IRRA_SMOKE_STAGE load_one_frozen_backend", flush=True)
        with patch.object(adapter_module, "load_official_model", new=load_one):
            adapter = IRRAAdapter.from_source(source, device="cuda:0", allow_model_load=True)
        backend = adapter._backend
        report["space"] = asdict(adapter.space)
        adapter_ids, token_reports = adapter._tokenizer.batch(captions)
        report["tokenizer"] = compare(native_ids, adapter_ids, exact=True)
        report["token_ids"] = native_ids.tolist()
        report["token_lengths"] = [r.untruncated_length for r in token_reports]
        report["truncated"] = [r.truncated for r in token_reports]
        report["model_before"] = model_snapshot(backend)
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        with patch.object(backend, "encode_image", new=counters.wrap("image", backend.encode_image)), \
             patch.object(backend, "encode_text", new=counters.wrap("text", backend.encode_text)):
            print("IRRA_SMOKE_STAGE official_native_2_image_2_text", flush=True)
            counters.phase = "official_native"
            # A独立官方预处理/Tokenizer/原生编码；仅共用已冻结的同一个Backend和运行设置。
            with fp32_runtime(), torch.inference_mode(), torch.autocast("cuda", enabled=False):
                a_image = F.normalize(backend.encode_image(native_normalized.cuda()).float(), dim=-1)
                a_text = F.normalize(backend.encode_text(native_ids.cuda()).float(), dim=-1)
                a_score = a_text @ a_image.T
            print("IRRA_SMOKE_STAGE adapter_2_image_2_text", flush=True)
            counters.phase = "adapter_inference"
            b_image = adapter.inference().encode_images(pixels.cuda())
            b_text, _ = adapter.inference().encode_texts(captions)
            b_score = adapter.score(b_text, b_image).payload
            report["image_features"] = feature_comparison(a_image, b_image.payload)
            report["text_features"] = feature_comparison(a_text, b_text.payload)
            report["scores"] = compare(a_score, b_score)
            report["scores"]["official"] = a_score.cpu().tolist()
            report["scores"]["adapter"] = b_score.cpu().tolist()
            require(not b_image.payload.requires_grad and not b_text.payload.requires_grad,
                    "inference view leaked gradients")
            print("IRRA_SMOKE_STAGE surrogate_1_image_gradient", flush=True)
            counters.phase = "surrogate_input_gradient"
            probe_pixels = pixels[:1].detach().clone().cuda().requires_grad_(True)
            feature = adapter.surrogate().encode_images(probe_pixels).payload
            feature[0, 0].backward()
            report["input_gradient"] = gradient_report(probe_pixels)
            del feature, probe_pixels
        torch.cuda.synchronize()
        report["forward_and_gradient_seconds"] = time.perf_counter() - forward_start
        report["model_after"] = model_snapshot(backend)
        require(report["model_before"] == report["model_after"], "model parameters/buffers changed")
        require(counters.started == counters.completed == LIMITS, "unexpected exact 5/4 forward counts")
        require(before == {r.path: checked(r) for r in refs}, "source SHA/mtime changed")
        report["source_files_unchanged"] = len(refs)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["exception_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        # 失败不再调用模型/梯度；仅尽可能记录资源和冻结摘要。
        if backend is not None:
            try:
                report["model_after_failure"] = model_snapshot(backend)
            except BaseException as state_error:
                report["state_check_error"] = str(state_error)
    finally:
        signal.signal(signal.SIGTERM, original_signal)
        report["elapsed_seconds"] = time.perf_counter() - started
        report["forwards"] = counters.report()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            report["gpu_peak_allocated_mib"] = sys.modules["torch"].cuda.max_memory_allocated() / 2**20
            report["gpu_peak_reserved_mib"] = sys.modules["torch"].cuda.max_memory_reserved() / 2**20
        # 失败时不要求来源仍匹配才能保存错误证据；原始SHA由attempt回执永久保存。
        store.publish("result", {"report.json": json_bytes(report)},
                      {"attempt_manifest_sha256": sha_file(store.root / "attempt" / "manifest.json")})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("IRRA_GPU_EQUIVALENCE_" + report["status"], flush=True)
    return 0 if report["status"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Once-only 2 Train image/caption IRRA equivalence; default CPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=BASE + "/irra_equivalence_v1")
    parser.add_argument("--allow-gpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        config, source, refs, before, report = resolve_config(args.config)
        if not args.allow_gpu_smoke:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print("IRRA_SMOKE_CPU_PREFLIGHT_PASS")
            return 0
        return run_once(config, source, refs, before, report, args.output)
    except (ValueError, TypeError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print("IRRA_SMOKE_REJECTED " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
