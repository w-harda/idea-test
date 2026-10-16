"""官方IRRA像素/Tokenizer接口；惰性依赖，仅显式CPU预处理或tokenization。"""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from importlib.metadata import version
import unicodedata
from ..artifacts import ArtifactError, checked, digest, sha_file
from ..contracts import PublicTextInput

SIZE=(384,128)
MEAN=(.48145466,.4578275,.40821073)
STD=(.26862954,.26130258,.27577711)
PREPROCESS={"schema":"tbps-irra-preprocess-1","size_hw":list(SIZE),"pixels":"RGB float32 [0,1]",
            "resize":"PIL RGB, torchvision Resize bilinear, no augmentation",
            "mean":list(MEAN),"std":list(STD),"normalization_location":"differentiable encoder forward",
            "feature_normalization":"torch F.normalize p=2 dim=-1 eps=1e-12"}
METRICS={"schema":"tbps-irra-ranking-1","orientation":"Query x Gallery","score":"L2(text) @ L2(image).T",
         "sort":"torch.argsort descending=True, default stable=False; no new tie breaking",
         "relevance":"all same namespaced PID positives; no camera filter",
         "mAP":"mean over all Query AP; all positives retained",
         "CMC":"first positive hit; R@1/5/10","mINP":"mean(num positives / last positive rank)",
         "unit":"percentage","missing_positive":"reject in DatasetManifest before ranking",
         "official_rank":"utils/metrics.py::rank(get_mAP=True,max_rank=10)"}

@dataclass(frozen=True, slots=True)
class TokenReport:
    ids: tuple[int,...]
    untruncated_length: int
    truncated: bool
    text_length: int = 77

class IRRATokenizer:
    def __init__(self, tokenizer, *,fingerprint):
        self._tokenizer=tokenizer;self.fingerprint=fingerprint
        self.sot=tokenizer.encoder["<|startoftext|>"];self.eot=tokenizer.encoder["<|endoftext|>"]
        if (self.sot,self.eot)!=(49406,49407):raise ArtifactError("official IRRA vocabulary markers mismatch")
    def inspect(self,text,*,truncate=True):
        if type(text) is PublicTextInput:text=text.caption
        if type(text) is not str:raise TypeError("Caption string or PublicTextInput required")
        if type(truncate) is not bool:raise TypeError("explicit truncation policy")
        ids=[self.sot]+self._tokenizer.encode(text)+[self.eot]
        if any(type(i) is not int or not 0<=i<49408 for i in ids):raise ArtifactError("invalid IRRA token ID")
        length=len(ids);truncated=length>77
        if truncated:
            if not truncate:raise ArtifactError("Caption exceeds official 77-token context")
            ids=ids[:77];ids[-1]=self.eot
        return TokenReport(tuple(ids+[0]*(77-len(ids))),length,truncated)
    def batch(self,texts,*,truncate=True):
        import torch
        if type(texts) not in (tuple,list) or not texts:raise ArtifactError("nonempty Caption sequence")
        reports=tuple(self.inspect(t,truncate=truncate) for t in texts)
        return torch.tensor([r.ids for r in reports],dtype=torch.long),reports

def official_tokenizer(source):
    """只加载独立官方SimpleTokenizer文件；不import datasets/model或加载IRRA。"""
    ref=source.ref("utils/simple_tokenizer.py")
    checked(ref);checked(source.bpe)
    name="_tbps_irra_tokenizer_"+ref.sha256[:16]
    spec=importlib.util.spec_from_file_location(name,ref.path)
    module=importlib.util.module_from_spec(spec)
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
    try:spec.loader.exec_module(module)
    finally:sys.dont_write_bytecode=previous
    tokenizer=module.SimpleTokenizer(source.bpe.path)
    checked(ref);checked(source.bpe)
    return IRRATokenizer(tokenizer,fingerprint=digest({"code":ref.sha256,"bpe":source.bpe.sha256,
                                 "context":77,"truncate":"head then final EOT","vocab":49408,
                                 "adapter":sha_file(__file__),"runtime":{k:version(k) for k in ("ftfy","regex")},
                                 "python":sys.version.split()[0],"unicode":unicodedata.unidata_version}))

def pixels_from_pil(image):
    from PIL import Image
    from torchvision import transforms
    if not isinstance(image,Image.Image):raise TypeError("PIL image required")
    return transforms.Compose([transforms.Resize(SIZE),transforms.ToTensor()])(image.convert("RGB"))

def normalize_pixels(pixels):
    import torch
    if not isinstance(pixels,torch.Tensor) or pixels.dtype!=torch.float32 or pixels.ndim!=4 or pixels.shape[1:]!=(3,*SIZE):
        raise ArtifactError("raw float32 RGB [B,3,384,128] required")
    if not len(pixels) or not torch.isfinite(pixels).all() or bool((pixels<0).any()) or bool((pixels>1).any()):
        raise ArtifactError("pixels must be finite [0,1]")
    # 不detach/转换PIL/重复Resize；攻击与投影必须在此函数之前的原像素空间。
    mean=pixels.new_tensor(MEAN).reshape(1,3,1,1);std=pixels.new_tensor(STD).reshape(1,3,1,1)
    return (pixels-mean)/std
