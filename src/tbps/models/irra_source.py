"""官方IRRA来源与离线构造；import/describe/verify均不加载模型。"""
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
from contextlib import contextmanager
from ..artifacts import ArtifactError, checked, digest, read_json
from ..contracts import FileRef

OFFICIAL_IRRA_SHA="9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"
REQUIRED=("model/__init__.py","model/build.py","model/clip_model.py","model/objectives.py",
          "utils/simple_tokenizer.py","datasets/bases.py","datasets/build.py","datasets/cuhkpedes.py","utils/metrics.py")

@dataclass(frozen=True, slots=True)
class IRRASource:
    root: str
    checkpoint: FileRef
    config: FileRef
    clip_bootstrap: FileRef
    bpe: FileRef
    code: tuple[tuple[str,FileRef],...]
    weight_training_dataset: str
    def __post_init__(self):
        if type(self.root) is not str or not Path(self.root).is_absolute():raise ArtifactError("absolute official root required")
        for ref in (self.checkpoint,self.config,self.clip_bootstrap,self.bpe):
            if type(ref) is not FileRef:raise TypeError("pinned FileRef required")
        if type(self.code) is not tuple or any(type(x) is not tuple or len(x)!=2 or type(x[0]) is not str or type(x[1]) is not FileRef for x in self.code):raise ArtifactError("immutable code source mapping")
        names=[n for n,r in self.code]
        if len(set(names))!=len(names) or not set(REQUIRED)<=set(names):raise ArtifactError("incomplete official source closure")
        root=Path(self.root).resolve()
        for n,r in self.code:
            if not n.endswith(".py") or Path(n).is_absolute() or ".." in Path(n).parts or Path(r.path).resolve()!=root/n:
                raise ArtifactError("official code path collision")
        if self.checkpoint.sha256!=OFFICIAL_IRRA_SHA:raise ArtifactError("only locked official IRRA checkpoint supported")
        if self.weight_training_dataset!="CUHK-PEDES official train":raise ArtifactError("explicit official model weight training domain")
    @property
    def fingerprint(self):return digest(asdict(self))
    def ref(self,name):return dict(self.code)[name]
    def settings(self):
        checked(self.config)
        import yaml
        settings=yaml.load(Path(self.config.path).read_text(),Loader=yaml.FullLoader)
        expected={"dataset_name":"CUHK-PEDES","pretrain_choice":"ViT-B/16","img_size":(384,128),"text_length":77,
                  "stride_size":16,"cmt_depth":4,"vocab_size":49408,"loss_names":"sdm+mlm+id"}
        if any(settings.get(k)!=v for k,v in expected.items()):raise ArtifactError("official config mismatch")
        checked(self.config)
        return settings
    def verify(self):
        refs=(self.checkpoint,self.config,self.clip_bootstrap,self.bpe,*(r for n,r in self.code))
        before=tuple(checked(r) for r in refs)
        self.settings()
        # 不导入官方模块或Torch；只读SHA。forward能力不由describe自动授权。
        return {"schema":"tbps-irra-source-1","model_id":"irra","fingerprint":self.fingerprint,
                "files":list(before),"weight_training_dataset":self.weight_training_dataset,
                "model_loaded":False,"artifact_space_compatible_with_legacy":False}
    @classmethod
    def from_dict(cls,data):
        if type(data) is not dict or set(data)!={"root","checkpoint","config","clip_bootstrap","bpe","code","weight_training_dataset"}:raise ArtifactError("IRRA source fields")
        if type(data["code"]) is not dict:raise ArtifactError("source code mapping required")
        return cls(data["root"],*(FileRef(**data[k]) for k in ("checkpoint","config","clip_bootstrap","bpe")),
                   tuple((k,FileRef(**v)) for k,v in sorted(data["code"].items())),data["weight_training_dataset"])

@contextmanager
def official_model_modules(source):
    """官方绝对model包需要显式隔离；不污染调用方已有同名包，不写pyc。"""
    if any(k=="model" or k.startswith("model.") for k in sys.modules):raise ArtifactError("model package namespace already occupied; use an isolated process")
    old_path=list(sys.path);old_bytecode=sys.dont_write_bytecode;before=set(sys.modules)
    sys.dont_write_bytecode=True;sys.path.insert(0,source.root)
    try:
        build=importlib.import_module("model.build")
        clip=importlib.import_module("model.clip_model")
        if Path(build.__file__).resolve()!=Path(source.root).resolve()/"model/build.py":raise ArtifactError("wrong official model module")
        yield build,clip
    finally:
        sys.path[:]=old_path;sys.dont_write_bytecode=old_bytecode
        for name in set(sys.modules)-before:
            if name=="model" or name.startswith("model."):sys.modules.pop(name,None)

def load_official_model(source,*,device,allow_model_load=False):
    """仅显式授权时调用；本阶段CPU检查不会调用此函数。不下载、不回退权重。"""
    if allow_model_load is not True:raise ArtifactError("explicit model-load authorization required")
    if device!="cpu" and not (type(device) is str and device.startswith("cuda:") and device[5:].isdigit()):
        raise ArtifactError("explicit cpu or cuda:index required")
    source.verify()
    import yaml
    import torch
    from unittest.mock import patch
    settings=source.settings()
    def read_only_clip(url,root):
        if Path(url).name!=Path(source.clip_bootstrap.path).name or url.split("/")[-2]!=source.clip_bootstrap.sha256:
            raise ArtifactError("unapproved CLIP bootstrap")
        checked(source.clip_bootstrap);return source.clip_bootstrap.path
    with official_model_modules(source) as (build,clip):
        with patch.object(clip,"_download",new=read_only_clip):
            model=build.build_model(SimpleNamespace(**settings),num_classes=11003)
        state=torch.load(source.checkpoint.path,map_location="cpu",weights_only=False)
        if type(state) is not dict or "model" not in state:raise ArtifactError("official checkpoint payload")
        model.load_state_dict(state["model"],strict=True)
        del state
    model.float().eval().requires_grad_(False)
    checked(source.checkpoint)
    return model.to(device)
