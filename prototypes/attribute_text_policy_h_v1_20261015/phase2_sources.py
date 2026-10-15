"""Phase 2固定来源和小型IO工具；不加载任何模型。"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
G2 = REPO / "prototypes/attribute_image_generator_v2_20261014"
RAW = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGES = RAW.parent / "imgs"
SPLIT = Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_unit_v4_learned_reader_20261003/outputs/split.json")
MANIFEST_SHA = "506caf30b8f82b8a29fc35e06df7bb04a5b2bcddc58a1f159b524bb107e7511e"
PROTOCOL_SHA = "9ad4903a6de6dc221df1a15af48d9e63e6332d7fcd3a9a62ecd3f11a2ed91c9d"
RAW_SHA = "fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0"
SPLIT_SHA = "33ec828fdf45de86d77a3fcaa23918953085fd2257d825150e4600d430687891"
EPOCH10_SHA = "f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e"
IRRA_SHA = "9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def checked(path, expected):
    if sha(path) != expected:
        raise ValueError("source SHA mismatch: " + str(path))
    return Path(path)

def within(root, path):
    root = Path(root).resolve()
    p = Path(path).resolve()
    if not p.is_relative_to(root) or p == root:
        raise ValueError("path escapes protected root")
    return p

def new_json(path, value):
    """只发布新文件；已有证据绝不覆盖。"""
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
