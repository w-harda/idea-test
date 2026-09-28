"""冻结 AP-Attack 的整图库生成；此模块不读取文本、ID 或 victim。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from types import ModuleType
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

ROOT = Path("/home/lzf/ldx")
EPSILON = 8 / 255
NATIVE_SIZE = (256, 128)
SOURCE_HASHES = {
    "GD.py": "a5b5a8a5f3df8bf4137b95ed6e98a1ee2667a6f2b51cbcb4f4c82759546c9342",
    "spectral.py": "8f9c814042306504a35d6f42f9a6649cbe98d62e445db69c75bd7bd1f3cd97b5",
    "gumbel.py": "80f57dedc9223028d5a2e8b88a6dfb00103567fd1fcd3cf33833a80050467628",
}
DEFAULT_G_SHA = "9dd2afe66afedc7ad17b43ebd14bfc2349e519bd74dbc5addacf5fad13434692"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def write_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("所有写入必须位于 /home/lzf/ldx")
    return path


def atomic_json(path: Path, value):
    path = write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = write_path(path.with_suffix(path.suffix + ".tmp"))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def atomic_torch(path: Path, value):
    path = write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = write_path(path.with_suffix(path.suffix + ".tmp"))
    torch.save(value, temp)
    temp.replace(path)


def gallery_paths(annotation: Path) -> tuple[str, ...]:
    """只提取 test 图库有序路径，不读取 caption、person ID 或配对 query。"""
    data = json.loads(annotation.read_text(encoding="utf-8"))
    paths = tuple(row["file_path"] for row in data if row["split"] == "test")
    if len(paths) != 3074 or len(set(paths)) != len(paths):
        raise ValueError("CUHK test gallery 必须恰好包含 3074 张唯一图像")
    return paths


def image_path(image_root: Path, relative: str) -> Path:
    root = image_root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("图库图像缺失或路径越界")
    return path


def native_pixels(image: Image.Image) -> torch.Tensor:
    # 官方 val_transforms 的 PIL 双线性 Resize + ToTensor。
    return transforms.functional.to_tensor(
        transforms.functional.resize(image.convert("RGB"), NATIVE_SIZE,
                                     interpolation=transforms.InterpolationMode.BILINEAR)
    )


class FrozenAPGenerator:
    """原样加载官方网络与 L_norm 等价推理，只接收 [0,1] 图像。"""

    def __init__(self, source_root: Path, checkpoint: Path,
                 expected_sha: str = DEFAULT_G_SHA, device: str = "cuda"):
        if sha256(checkpoint) != expected_sha:
            raise ValueError("G checkpoint SHA-256 不匹配")
        root = source_root.resolve()
        for name, expected in SOURCE_HASHES.items():
            if sha256(root / "advers" / name) != expected:
                raise ValueError(f"官方 AP 推理源码不匹配: {name}")
        # 隔离包名，避免 AP 的 model/utils 与 IRRA 发生导入冲突。
        package = "_tbps_official_ap"
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = [str(root / "advers")]
            sys.modules[package] = module
        # 官方 GD.py 使用绝对 advers 导入，临时挂接同一官方包。
        old = {k: v for k, v in sys.modules.items()
               if k == "advers" or k.startswith("advers.")}
        try:
            for key in old:
                del sys.modules[key]
            sys.modules["advers"] = sys.modules[package]
            spec = importlib.util.spec_from_file_location(
                package + ".GD", root / "advers" / "GD.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.generator = module.Generator(3, 3, 32, norm="bn", beta=0.1)
        finally:
            for key in list(sys.modules):
                if key == "advers" or key.startswith("advers."):
                    del sys.modules[key]
            sys.modules.update(old)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.generator.load_state_dict(state, strict=True)
        self.generator.eval().requires_grad_(False).to(device)
        self.device = torch.device(device)

    def state(self):
        # 官方 SpectralNorm 即使 eval 仍更新 u/v；续跑需恢复这些值。
        return {k: v.detach().cpu().clone()
                for k, v in self.generator.state_dict().items()}

    def restore(self, state):
        self.generator.load_state_dict(state, strict=True)

    @torch.no_grad()
    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        if (pixels.ndim != 4 or pixels.shape[1:] != (3, *NATIVE_SIZE)
                or not pixels.is_floating_point() or not torch.isfinite(pixels).all()
                or pixels.min() < 0 or pixels.max() > 1):
            raise ValueError("AP 输入必须是有限的 [N,3,256,128] RGB [0,1] 像素")
        pixels = pixels.to(self.device)
        # 官方验证循环对生成器使用 AMP；CPU 单元测试关闭 AMP。
        with torch.autocast(device_type=self.device.type,
                            enabled=self.device.type == "cuda"):
            delta = self.generator((pixels - 0.5) / 0.5)
        if delta.shape != pixels.shape or not torch.isfinite(delta).all():
            raise ValueError("AP 生成器输出无效")
        # 官方 L_norm 在像素空间 clamp 8/255，然后除以 std=0.5。
        # 将归一化图像还原为像素后，严格等价于以下加法。
        return (pixels + delta.clamp(-EPSILON, EPSILON)).clamp(0, 1).detach()


def transport_delta(clean_native: torch.Tensor, adv_native: torch.Tensor,
                    clean_victim: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    """仅双线性传递扰动；保持 victim 官方干净预处理的像素基准。"""
    if (clean_native.shape != adv_native.shape or clean_native.ndim != 4
            or clean_native.shape[1:] != (3, *NATIVE_SIZE)
            or clean_victim.shape != (len(clean_native), 3, 384, 128)
            or not all(torch.isfinite(x).all() for x in
                       (clean_native, adv_native, clean_victim))
            or any(x.min() < 0 or x.max() > 1 for x in
                   (clean_native, adv_native, clean_victim))):
        raise ValueError("AP/IRRA 图像空间或像素无效")
    delta = adv_native - clean_native
    if float(delta.abs().max()) > EPSILON + 1e-6:
        raise ValueError("AP 原生扰动超过 8/255")
    mapped = F.interpolate(delta, size=(384, 128), mode="bilinear",
                           align_corners=False)
    if float(mapped.abs().max()) > EPSILON + 1e-6:
        raise ValueError("Resize 放大了扰动")
    adv = (clean_victim + mapped).clamp(0, 1)
    linf = (adv - clean_victim).abs().flatten(1).amax(1).tolist()
    return adv.detach(), linf


def cache_spec(annotation: Path, image_root: Path, paths: tuple[str, ...],
               source_root: Path, checkpoint: Path, expected_sha: str,
               origin: str, batch_size: int):
    if sha256(checkpoint) != expected_sha:
        raise ValueError("G checkpoint SHA-256 不匹配")
    hashes = {name: sha256(source_root / "advers" / name)
              for name in SOURCE_HASHES}
    if hashes != SOURCE_HASHES:
        raise ValueError("AP 官方推理源码发生变化")
    if origin not in {"local_reproduction", "author_pretrained_verified"}:
        raise ValueError("G 来源必须明确标记")
    return {
        "schema": 1, "annotation_sha256": sha256(annotation),
        "image_root": str(image_root.resolve()),
        "gallery_paths": list(paths), "gallery_sha256": fingerprint(paths),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": expected_sha,
        "checkpoint_origin": origin, "source_hashes": hashes,
        "training_dataset": "DukeMTMC-reID", "surrogate": "IDE (ResNet50)",
        "architecture": "Generator(3,3,32,norm=bn,n_blocks=6,beta=0.1)",
        "epsilon": EPSILON, "native_size": list(NATIVE_SIZE),
        "preprocess": "official PIL bilinear resize, (pixel-0.5)/0.5",
        "cache_precision": "float32", "generator_precision": "official CUDA AMP",
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "torch_version": str(torch.__version__), "batch_size": batch_size,
        "spectral_state": "official u/v updates retained and checkpointed",
        "victim_information_used": "none", "query_information_used": "none",
    }


def read_progress(cache: Path, spec):
    path = cache / "progress.pt"
    if not path.exists():
        return {"spec": spec, "done": 0, "chunks": [], "linf": [], "state": None}
    progress = torch.load(path, map_location="cpu", weights_only=True)
    if progress["spec"] != spec:
        raise ValueError("图库缓存配置不匹配，必须使用独立目录")
    cursor = 0
    for chunk in progress["chunks"]:
        path = cache / chunk["file"]
        if chunk["start"] != cursor or sha256(path) != chunk["sha256"]:
            raise ValueError("图库缓存分块缺失、损坏或次序不匹配")
        cursor += chunk["count"]
    if cursor != progress["done"] or len(progress["linf"]) != cursor:
        raise ValueError("图库缓存进度不一致")
    return progress


def build_cache(cache: Path, spec, paths, image_root: Path, factory,
                max_images: int | None = None):
    """事务式保存分块及推理状态；缓存完成后不再调用生成器。"""
    cache = write_path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    progress = read_progress(cache, spec)
    target = len(paths) if max_images is None else min(
        len(paths), progress["done"] + max_images)
    if max_images is not None and max_images < 1:
        raise ValueError("max_images 必须为正数")
    if target != len(paths) and target % spec["batch_size"]:
        raise ValueError("部分生成必须停在固定 batch 边界，以保持 SpectralNorm 续跑一致")
    generator = None
    if progress["done"] < target:
        generator = factory()
        if progress["state"] is not None:
            generator.restore(progress["state"])
    while progress["done"] < target:
        start = progress["done"]
        stop = min(start + spec["batch_size"], target)
        pixels, raw_hashes = [], []
        for relative in paths[start:stop]:
            path = image_path(image_root, relative)
            raw_hashes.append(sha256(path))
            with Image.open(path) as image:
                pixels.append(native_pixels(image))
        clean = torch.stack(pixels)
        attacked = generator(clean).cpu()
        linf = (attacked - clean).abs().flatten(1).amax(1).tolist()
        if max(linf) > EPSILON + 1e-6:
            raise ValueError("AP 实测扰动超过 8/255")
        name = f"gallery_{start:04d}_{stop:04d}.pt"
        chunk_path = cache / name
        atomic_torch(chunk_path, {
            "start": start, "paths": list(paths[start:stop]),
            "raw_sha256": raw_hashes, "adv_pixels": attacked.float(),
        })
        progress["chunks"].append({
            "file": name, "sha256": sha256(chunk_path),
            "start": start, "count": stop - start})
        progress["done"] = stop
        progress["linf"].extend(linf)
        progress["state"] = generator.state()
        atomic_torch(cache / "progress.pt", progress)
        print(json.dumps({"gallery_cached": stop, "target": len(paths)}), flush=True)
    manifest = {k: v for k, v in progress.items() if k != "state"}
    manifest["complete"] = progress["done"] == len(paths)
    manifest["mean_linf_native"] = (
        sum(progress["linf"]) / progress["done"] if progress["done"] else None)
    manifest["max_linf_native"] = max(progress["linf"], default=None)
    atomic_json(cache / "gallery.json", manifest)
    return manifest


def cached_batches(cache: Path, spec, count: int):
    progress = read_progress(cache, spec)
    if progress["done"] < count:
        raise ValueError("对抗图库缓存尚未覆盖所需全部图像")
    for chunk in progress["chunks"]:
        if chunk["start"] >= count:
            break
        payload = torch.load(cache / chunk["file"], map_location="cpu",
                             weights_only=True)
        expected = spec["gallery_paths"][chunk["start"]:
                                         chunk["start"] + chunk["count"]]
        if payload["paths"] != expected or payload["start"] != chunk["start"]:
            raise ValueError("图库分块路径顺序不匹配")
        n = min(chunk["count"], count - chunk["start"])
        if (payload["adv_pixels"].shape != (chunk["count"], 3, *NATIVE_SIZE)
                or not torch.isfinite(payload["adv_pixels"]).all()):
            raise ValueError("缓存 adversarial image 无效")
        yield payload["paths"][:n], payload["raw_sha256"][:n], payload["adv_pixels"][:n]
