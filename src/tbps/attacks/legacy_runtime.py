"""显式旧实现加载会话；默认拒绝，Stage 3A只测试导入和CPU Mock。"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
from ..artifacts import checked, read_json
from .base import need, frozen_state
from .gv2 import GV2Plugin
from .hv1 import HV1Plugin
from .joint import JointPlugin

_LOCK=threading.Lock()
G_REL="prototypes/attribute_image_generator_v2_20261014"
H_REL="prototypes/attribute_text_policy_h_v1_20261015"

class _PinnedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, pins): self.root,self.pins=root,pins
    def find_spec(self, fullname, path=None, target=None):
        need(fullname.split(".")[0] not in ("model","datasets"), "victim/model loading forbidden in attack import session")
        spec=importlib.machinery.PathFinder.find_spec(fullname,path)
        if spec and spec.origin and spec.origin not in ("built-in","frozen"):
            p=Path(spec.origin).resolve()
            if p.is_relative_to(self.root):
                need(str(p) in self.pins, "unpinned legacy import: "+str(p))
                from ..contracts import FileRef
                checked(FileRef(str(p),self.pins[str(p)]))
        return spec

@contextmanager
def legacy_imports(sources, *, allow_import=False):
    """绝对import会话，拒绝同名碰撞；退出恢复sys.path、模块注册和pyc标志。"""
    need(allow_import is True, "explicit legacy-import authorization required")
    need(_LOCK.acquire(blocking=False), "legacy import session is not reentrant/thread-safe")
    old_path=list(sys.path);old_bytecode=sys.dont_write_bytecode
    before=dict(sys.modules);finder=None
    try:
        root=sources.root
        h,g=root/H_REL,root/G_REL
        owned={"src","formal_training","phase5_joint"}
        for p in sources.config["runtime_sources"]:
            path=Path(p)
            for parent in (h,g):
                if path.is_relative_to(parent):
                    owned.add(path.relative_to(parent).parts[0].removesuffix(".py"))
        need(not any(k.split(".")[0] in owned for k in before), "legacy top-level module collision; use an isolated process")
        finder=_PinnedFinder(root,sources.config["runtime_sources"])
        sys.meta_path.insert(0,finder)
        sys.path[:0]=[str(h),str(g),str(root)]
        sys.dont_write_bytecode=True
        yield importlib.import_module
    finally:
        if finder in sys.meta_path: sys.meta_path.remove(finder)
        sys.path[:]=old_path;sys.dont_write_bytecode=old_bytecode
        # 移除该会话新注册的旧源码模块（第三方包保留）；不改已有模块。
        for name,module in tuple(sys.modules.items()):
            filename=getattr(module,"__file__",None)
            if name not in before and (name.split(".")[0] in locals().get("owned",set()) or
               (filename and Path(filename).resolve().is_relative_to(sources.root))):
                sys.modules.pop(name,None)
        _LOCK.release()

class _LegacyG:
    execution_kind="legacy_unverified"
    preprocess_id="PIL RGB Resize bilinear(384,128) + ToTensor; no mean/std before G"
    def __init__(self, model, inference, pixel_reader, device):
        self.model,self.generate_once=model,inference.generate_once
        self.pixel_reader,self.device=pixel_reader,device
    def load_pixels(self, path):
        from PIL import Image
        with Image.open(path) as image: return self.pixel_reader(image)[None].to(self.device)

class _LegacyH:
    execution_kind="legacy_unverified"
    def __init__(self, attacker, module):
        self.attacker,self.module=attacker,module
        self.model=attacker.policy
    def attack(self, caption):
        # 使用与正式Validation/Phase5一致的math/无fastpath设置；不调用IRRA。
        from ..models.irra import fp32_runtime
        with fp32_runtime(): return self.attacker.attack(caption)
    def validate(self, caption, record):
        return self.module.validate_action(caption,record,self.attacker.probe,self.attacker.frontend)

@contextmanager
def load_legacy_plugins(sources, *, components=("gv2","hv1"), device="cpu", allow_model_load=False):
    """供后续独立授权使用；本阶段不调用。无victim/奖励/条件准备参数。"""
    need(allow_model_load is True, "explicit G/H model-load authorization required")
    need(type(components) is tuple and components and len(set(components))==len(components) and
         set(components)<={"gv2","hv1"}, "explicit unique G/H components")
    need(device=="cpu" or (type(device) is str and device.startswith("cuda:") and device[5:].isdigit()),
         "explicit device required")
    sources.verify()
    with legacy_imports(sources,allow_import=True) as old:
        import torch
        result={}
        if "gv2" in components:
            inference=old("inference_once")
            # 原load_generator只读不可变文件，不调用load_latest/current/rollback。
            with torch.inference_mode(False):
                model=inference.load_generator(sources.ref("gv2","checkpoint").path,device)
                model.eval().requires_grad_(False)
            backend=_LegacyG(model,inference,old("irra_image_adapter").pixels_from_pil,device)
            result["gv2"]=GV2Plugin(backend=backend,provenance=sources.provenance("gv2"))
        if "hv1" in components:
            module=old("phase5_joint.backend")
            probe=old("tokenizer_probe").official_probe()
            frontend=old("text_frontend").CaptionFrontend()
            ref=sources.ref("hv1","checkpoint");checked(ref)
            payload=torch.load(ref.path,map_location="cpu",weights_only=False)
            need(payload["schema"]=="hv1-formal-periodic-checkpoint-1" and
                 payload["protocol_sha256"]==sources.descriptions["hv1"]["protocol_digest"] and
                 payload["cursor"]==14336, "selected H payload/source/cursor mismatch")
            architecture=sources.protocol("hv1")["config"]["architecture"]
            with torch.inference_mode(False):
                policy=old("policy").CharacterPolicy(probe,**architecture)
                policy.load_state_dict(payload["policy"],strict=True)
                policy.eval().requires_grad_(False)
            attacker=module.TextAttacker(policy,frontend,probe)
            state_sha_from=old("formal_training.validation").state_sha_from
            need(attacker.before==state_sha_from(payload["policy"]), "H checkpoint values changed on loading")
            del payload
            result["hv1"]=HV1Plugin(backend=_LegacyH(attacker,module),provenance=sources.provenance("hv1"))
        if set(result)=={"gv2","hv1"}:
            result["joint"]=JointPlugin(image_plugin=result["gv2"],text_plugin=result["hv1"])
        snapshots={name:frozen_state(plugin.backend.model) for name,plugin in result.items() if name!="joint"}
        try: yield result
        finally:
            for name,state in snapshots.items():
                need(frozen_state(result[name].backend.model)==state, "attack model changed during session")
            sources.verify()
