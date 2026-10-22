"""生产后端来源与隔离导入；默认只读，不执行模型/训练。"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
import importlib
import importlib.machinery
import sys
import os
import threading
from unittest.mock import patch
from tbps.artifacts import checked, FileRef, read_json, digest
from .contracts import require

_LOCK = threading.Lock()
ENTRIES = (
    "image_only_generator", "budget_projection", "condition_schema", "serial_training",
    "inference_once", "losses", "irra_image_adapter", "epoch_checkpoint",
    "contracts", "constraints", "policy", "tokenizer_probe", "text_frontend",
    "phase4_training", "formal_training.checkpoint", "formal_training.data",
    "formal_training.runtime", "training_inputs", "importance", "attribute_interface",
    "attribute_tta", "paired_prefix", "learned_reader",
    "attributes.irra_token_alignment", "utils.simple_tokenizer",
)

def verify_config(path):
    c = read_json(path, "gh-native-backend-1")
    require(c["base_commit"] == "8ae623d3b6cb01f96c86e15e167d7dd7b1f56734",
            "fixed CPU prototype parent")
    require(c["gpu_execution_default"] is False and c["probe"]["optimizer_updates"] == 0,
            "default deny/probe no updates")
    require(c["probe"]["rows"] == ["cuhk:26530:0", "cuhk:26747:0"]
            and c["probe"]["max_G_forwards"] == c["probe"]["max_IRRA_images"] == 4
            and c["probe"]["max_JVPs"] == 1, "fixed numerical probe")
    require(c["byte_alignment_policy"] == "exact-byte-span-v1"
            and c["runtime"]["torch"] == "2.5.1+cu124", "fixed native mapping/runtime")
    root = Path(__file__).resolve().parents[3]
    require(set(c["producers"]) == {str(p) for p in (root/"src/tbps").rglob("*.py")},
            "complete backend producer set")
    for p,h in c["producers"].items():
        checked(FileRef(p,h))
    for ref in (c["prototype_parent"], *c["sources"].values()):
        checked(FileRef(**ref))
    for p,h in c["runtime_pins"].items():
        checked(FileRef(p,h))
    expected_condition=digest({"IRRA":c["prototype"]["source_space_sha256"],"reader":c["sources"]["reader"],
        "basis":c["sources"]["basis"],"byte_mapping":c["byte_alignment_policy"],
        "producer_sources":{p:h for p,h in c["producers"].items() if "/alternating/" in p}})
    require(c["condition_space_sha256"]==expected_condition,"condition reader/basis/producer identity")
    c["_config_path"] = str(Path(path).resolve())
    return c

@contextmanager
def native_session(c, *, reuse=None):
    """会话内所有旧代码执行均SHA核验；退出恢复路径、模块注册、缓存环境和锁。
    reuse只重装同一已核验模块对象，避免第二次导入产生不同TextExample类身份。
    """
    require(_LOCK.acquire(False), "native import session not reentrant")
    before = dict(sys.modules)
    paths,meta,bytecode = list(sys.path), list(sys.meta_path), sys.dont_write_bytecode
    env = {k:os.environ.get(k) for k in ("TORCH_HOME","HF_HOME","XDG_CACHE_HOME","CUDA_CACHE_PATH")}
    roots = tuple(Path(x).resolve() for x in c["managed_roots"])
    pins = {**c["runtime_pins"], **c["producers"]}
    observed = {}
    def managed(p):
        return any(p.is_relative_to(root) for root in roots)
    original = importlib.machinery.SourceFileLoader.get_code
    def get_code(loader, name):
        p = Path(loader.path).resolve()
        if managed(p):
            require(str(p) in pins, "unpinned production import: "+str(p))
            observed[str(p)] = checked(FileRef(str(p),pins[str(p)]))
            content=p.read_bytes()
            import hashlib
            require(hashlib.sha256(content).hexdigest()==pins[str(p)], "import bytes changed")
            return compile(content,str(p),"exec",dont_inherit=True)
        return original(loader,name)
    try:
        # 真实IRRA需先在此会话外加载；现存旧命名空间不得被覆盖。
        for name,module in before.items():
            filename=getattr(module,"__file__",None)
            if name in ("__main__","__mp_main__") and filename and Path(filename).resolve()==Path(__file__).with_name("native_cli.py").resolve():
                checked(FileRef(str(Path(filename).resolve()),pins[str(Path(filename).resolve())]))
                continue
            if filename and managed(Path(filename).resolve()) and not (name=="tbps" or name.startswith("tbps.")):
                raise ValueError("legacy namespace collision: "+name)
        if reuse:
            for name,module in reuse.items():
                p=Path(module.__file__).resolve()
                require(managed(p) and str(p) in pins, "reused module outside source pins")
                checked(FileRef(str(p),pins[str(p)]))
                require(name not in sys.modules, "reused namespace collision")
                sys.modules[name]=module
        sys.path[:0]=c["import_paths"]
        sys.dont_write_bytecode=True
        with patch.object(importlib.machinery.SourceFileLoader,"get_code",get_code):
            session=SimpleNamespace(import_module=importlib.import_module,observed=observed,modules={})
            yield session
            session.modules.update({name:module for name,module in sys.modules.items()
                if name not in before and getattr(module,"__file__",None)
                and managed(Path(module.__file__).resolve()) and not (name=="tbps" or name.startswith("tbps."))})
    finally:
        sys.path[:]=paths;sys.meta_path[:]=meta;sys.dont_write_bytecode=bytecode
        for k,v in env.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
        for name,module in tuple(sys.modules.items()):
            p=getattr(module,"__file__",None)
            if name not in before and p and managed(Path(p).resolve()) and not (name=="tbps" or name.startswith("tbps.")):
                sys.modules.pop(name,None)
        try:
            require(all(sys.modules.get(k) is v for k,v in before.items()), "pre-existing module changed")
        finally:
            _LOCK.release()

def bindings(session):
    modules={name:session.import_module(name) for name in ENTRIES}
    return SimpleNamespace(modules=modules, get=modules.__getitem__)

def strict_import_preflight(c):
    from tbps.runtime_closure import cpu_only
    import torch
    # kornia导入会构建无参数ScalePyramid默认参数；仅CPU构造，不允许权重、前向或CUDA。
    from .parents import cpu_restore_only
    with cpu_restore_only(()) as vendor_events:
        importlib.import_module("kornia")
    from importlib.metadata import version
    require(torch.__version__==c["runtime"]["torch"] and version("numpy")==c["runtime"]["numpy"]
            and list(sys.version_info[:2])==c["runtime"]["python_major_minor"],"frozen native CPU/GPU runtime mismatch")
    before=(list(sys.path),list(sys.meta_path),sys.dont_write_bytecode)
    with cpu_only() as attempts:
        with native_session(c) as s:
            b=bindings(s)
            # 真实BPE及前端纯CPU初始化，捕获真实动态导入；不构建reader/模型。
            p=Path(c["sources"]["bpe"]["path"])
            tokenizer=b.get("utils.simple_tokenizer").SimpleTokenizer(bpe_path=str(p))
            probe=b.get("tokenizer_probe").TokenizerProbe(tokenizer)
            frontend=b.get("text_frontend").CaptionFrontend()
            require(probe.inspect("blue shirt").model_tokens[0]==probe.sot, "native BPE")
            require(callable(frontend.convert), "native Canonical frontend")
            executed=dict(s.observed)
    require(before==(list(sys.path),list(sys.meta_path),sys.dont_write_bytecode)
            and not torch.cuda.is_initialized(), "CPU import environment not restored")
    return {"status":"GH_NATIVE_IMPORT_CLOSURE_PASS","entries":list(ENTRIES),
            "executed_sources":executed,"forbidden_attempts":dict(attempts),
            "vendor_import_guard":vendor_events,"namespace_restored":True,"model_loads":0,"model_forwards":0,"CUDA":False}

def source_key(c):
    return digest({k:v for k,v in c.items() if not k.startswith("_")})
