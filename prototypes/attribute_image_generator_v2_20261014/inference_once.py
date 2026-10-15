"""独立image-only推理模块；不读取监督/cache/IRRA。"""
import hashlib,json
from pathlib import Path
import torch
from image_only_generator import ImageOnlyGenerator
from budget_projection import check_pixels,add_projected_delta

@torch.inference_mode()
def generate_once(generator,image):
    check_pixels(image)
    generator.eval()
    delta=generator(image)
    adversarial=add_projected_delta(image,image,delta)
    return {"delta":delta,"adversarial":adversarial,"effective_delta":adversarial-image}

def load_generator(checkpoint,device="cpu"):
    # 仅读取G权重与架构，不导入Epoch/条件/IRRA模块。
    checkpoint=Path(checkpoint)
    side=json.loads(checkpoint.with_suffix(".json").read_text())
    h=hashlib.sha256()
    with checkpoint.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    if h.hexdigest()!=side["sha256"]: raise ValueError("inference checkpoint SHA mismatch")
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if payload["schema"]!="gv2-epoch-checkpoint-1":raise ValueError("unsupported G-v2 checkpoint")
    config=payload["contract"]["config"]
    g=ImageOnlyGenerator(**config["architecture"]).to(device)
    g.load_state_dict(payload["generator"],strict=True)
    return g.eval()
