"""G-v2独立IRRA图像适配器；只读官方源码/权重，无文本前向。"""
import importlib, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from budget_projection import check_pixels
import cache_io as io

OFFICIAL=Path("/home/lzf/Attack/IRRA-main/IRRA-main")
CHECKPOINT=Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth")
CONFIG=Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/configs/irra_cuhk_official.yaml")
CLIP_CACHE=Path("/home/lzf/ldx/cache/clip")
CHECKPOINT_SHA="9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"
SIZE=(384,128)
MEAN=(.48145466,.4578275,.40821073)
STD=(.26862954,.26130258,.27577711)

def pixels_from_pil(image):
    return transforms.Compose([transforms.Resize(SIZE),transforms.ToTensor()])(image.convert("RGB"))

class FrozenIRRAImageEncoder(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model.eval().requires_grad_(False)
        self.register_buffer("mean",torch.tensor(MEAN).reshape(1,3,1,1))
        self.register_buffer("std",torch.tensor(STD).reshape(1,3,1,1))
        self.eval()
    def train(self,mode=True):
        super().train(False)
        return self
    def forward(self,pixels):
        check_pixels(pixels)
        if pixels.shape[1:]!=(3,*SIZE) or pixels.dtype!=torch.float32:
            raise ValueError("IRRA fp32 RGB [B,3,384,128] required")
        feature=self.model.encode_image((pixels-self.mean)/self.std).float()
        if feature.shape!=(len(pixels),512) or not torch.isfinite(feature).all() or (feature.norm(dim=1)==0).any():
            raise ValueError("invalid official IRRA retrieval features")
        return F.normalize(feature,dim=-1)

def load_official(device="cuda"):
    import yaml
    sys.dont_write_bytecode=True
    io.verify({str(CHECKPOINT):CHECKPOINT_SHA})
    settings=yaml.load(CONFIG.read_text(),Loader=yaml.FullLoader)
    expected={"dataset_name":"CUHK-PEDES","pretrain_choice":"ViT-B/16","img_size":SIZE,
              "text_length":77,"stride_size":16,"cmt_depth":4,"vocab_size":49408,"loss_names":"sdm+mlm+id"}
    if any(settings.get(k)!=v for k,v in expected.items()): raise ValueError("official config mismatch")
    sys.path.insert(0,str(OFFICIAL))
    build=importlib.import_module("model.build")
    clip=importlib.import_module("model.clip_model")
    if Path(build.__file__).resolve()!=OFFICIAL/"model/build.py": raise ValueError("official module path collision")
    def read_only_clip(url,root):
        path=CLIP_CACHE/Path(url).name
        if io.sha(path)!=url.split("/")[-2]: raise ValueError("missing/modified cached CLIP; download forbidden")
        return str(path)
    # 仅替换下载器为只读cache读取，不改变模型/编码函数，也不修改官方文件。
    with patch.object(clip,"_download",side_effect=read_only_clip):
        model=build.build_model(SimpleNamespace(**settings),num_classes=11003)
    state=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)["model"]
    model.load_state_dict(state,strict=True)
    del state
    model.float().eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return FrozenIRRAImageEncoder(model).to(device)
