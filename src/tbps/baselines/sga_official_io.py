"""官方SGA文件及CUHK必要接线；无模型加载或攻击迭代副作用。"""
from pathlib import Path
import importlib.util
import sys
from dataclasses import replace
from tbps.artifacts import FileRef, checked, ArtifactError
from tbps.datasets.cuhk_pedes import CuhkPedes

OFFICIAL_COMMIT = "f49e6926cd13148e49bdbc2c3e537660e096230c"
SAMPLE_IDS = ("cuhk:37132:0", "cuhk:37147:0")

def load_file(ref, name):
    ref = FileRef(**ref) if isinstance(ref, dict) else ref
    checked(ref)
    spec = importlib.util.spec_from_file_location(name + "_" + ref.sha256[:12], ref.path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    checked(ref)
    return module

def official_modules(config):
    return tuple(load_file(config["official"][name], "_sga_" + name)
                 for name in ("attacker", "tokenizer", "dataset"))

def dataset(config, scope="smoke"):
    ds = CuhkPedes(FileRef(**config["annotation"]), config["image_root"])
    full = ds.manifest("test", expected_counts=(3074, 6156))
    if scope == "smoke":
        rows = tuple(g for g in full.gallery if g.image_id in SAMPLE_IDS)
        queries = tuple(q for q in full.queries if q.paired_image_id in SAMPLE_IDS)
        if tuple(g.image_id for g in rows) != SAMPLE_IDS or len(queries) != 4:
            raise ArtifactError("fixed 2 Gallery / 4 Query identity")
        manifest = replace(full, gallery=rows, queries=queries)
    elif scope == "full":
        manifest = full
    else:
        raise ArtifactError("scope")
    _, _, native_dataset = official_modules(config)
    matched = replace(manifest, queries=tuple(replace(q, caption=native_dataset.pre_caption(q.caption, 30))
                                             for q in manifest.queries))
    return ds, manifest, matched

def source_pixels(path):
    from PIL import Image
    from torchvision import transforms
    with Image.open(path) as image:
        return transforms.Compose([transforms.Resize((384, 128)), transforms.ToTensor()])(image.convert("RGB"))

def official_projection_check(clean, adv):
    import torch
    eps = 2 / 255
    if not (torch.isfinite(adv).all() and adv.dtype == torch.float32 and adv.shape == clean.shape
            and (adv >= (clean - eps).clamp(0, 1)).all()
            and (adv <= (clean + eps).clamp(0, 1)).all()):
        raise ArtifactError("official FP32 2/255 projection bounds")
    exact = float((adv.double() - clean.double()).abs().max())
    return {"official_FP32_bounds_exact": True, "real_linf": exact,
            "mathematical_epsilon": eps, "roundoff_above_real_epsilon": max(0., exact - eps),
            "extra_projection": False}
