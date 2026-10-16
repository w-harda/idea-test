"""只读G/H真实加载导入闭包；禁止权重、神经实例/前向、导数及CUDA。"""
from contextlib import contextmanager, ExitStack
import ast
import importlib.machinery
import inspect
from pathlib import Path
import sys
from unittest.mock import patch
from .artifacts import ArtifactError, checked, read_json, digest
from .contracts import FileRef
from .attacks.base import need
from .attacks import legacy_runtime
from .attacks.legacy_runtime import legacy_imports

# 与真实工厂中的old("...")入口逐项核对；新增入口必须同时补齐可解析符号。
ENTRY_SYMBOLS = {
    "inference_once": ("load_generator", "generate_once", "ImageOnlyGenerator"),
    "irra_image_adapter": ("pixels_from_pil",),
    "phase5_joint.backend": ("TextAttacker", "validate_action"),
    "tokenizer_probe": ("official_probe", "TokenizerProbe"),
    "text_frontend": ("CaptionFrontend",),
    "policy": ("CharacterPolicy",),
    "formal_training.validation": ("state_sha_from",),
}

def factory_entries():
    tree = ast.parse(inspect.getsource(legacy_runtime.load_legacy_plugins))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "old"]
    need(all(len(n.args) == 1 and isinstance(n.args[0], ast.Constant)
             and type(n.args[0].value) is str for n in calls),
         "runtime factory contains uninspectable dynamic import")
    names = {n.args[0].value for n in calls}
    need(names == set(ENTRY_SYMBOLS), "runtime factory import entries changed; extend CPU closure explicitly")
    return sorted(names)

def verify_derivation(path, config):
    d = read_json(path, "tbps-runtime-source-derivation-1")
    need(set(d) == {"schema", "base_commit", "parent", "derived", "added_runtime_sources", "reason"},
         "derivation fields")
    need(d["base_commit"] == "0fb90cd1b280da886aec08834dc13d28eacd5848",
         "derivation base mismatch")
    for key in ("parent", "derived"):
        checked(FileRef(**d[key]))
    parent, child = (read_json(d[k]["path"], "tbps-legacy-attack-plugins-1")
                     for k in ("parent", "derived"))
    p, c = dict(parent), dict(child)
    pp, cp = p.pop("runtime_sources"), c.pop("runtime_sources")
    additions = d["added_runtime_sources"]
    need(p == c and not set(pp).intersection(additions)
         and cp == {**pp, **additions}, "only explicit additive runtime pins permitted")
    need(len(additions) == 1, "Stage 3B-R allows exactly one verified addition")
    source, sha = next(iter(additions.items()))
    need(Path(source) == Path(parent["legacy_root"]) /
         "prototypes/attribute_image_generator_v2_20261014/condition_schema.py"
         and sha == "f13932c2688cab4e135d063eed6f066016f6ba26e2316cbc201f19b246b298b4",
         "unexpected Stage 3B-R source addition")
    checked(FileRef(source, sha))
    cfg = read_json(config, "tbps-gh-equivalence-smoke-1")
    need(cfg["attack_sources"] == d["derived"], "CPU config does not bind derived sources")
    return d

@contextmanager
def cpu_only():
    import torch
    need(not torch.cuda.is_initialized(), "isolated CPU process required")
    attempts = {}
    def forbidden(name):
        def stop(*args, **kwargs):
            attempts[name] = attempts.get(name, 0) + 1
            raise ArtifactError("CPU closure forbids " + name)
        return stop
    with ExitStack() as stack:
        for owner, name, label in (
            (torch, "load", "torch.load"), (torch.jit, "load", "torch.jit.load"),
            (torch.nn.Module, "__init__", "neural construction"),
            (torch.nn.Module, "_call_impl", "neural forward"),
            (torch.cuda, "init", "CUDA init"), (torch.cuda, "_lazy_init", "CUDA lazy init"),
            (torch.Tensor, "cuda", "Tensor.cuda"), (torch.nn.Module, "cuda", "Module.cuda"),
            (torch.Tensor, "backward", "backward"),
            (torch.autograd, "backward", "autograd.backward"),
            (torch.autograd, "grad", "autograd.grad"),
        ):
            stack.enter_context(patch.object(owner, name, new=forbidden(label)))
        if hasattr(torch._C, "_cuda_init"):
            stack.enter_context(patch.object(torch._C, "_cuda_init", new=forbidden("CUDA native init")))
        yield attempts
        need(not attempts and not torch.cuda.is_initialized(), "forbidden CPU operation was attempted")

@contextmanager
def source_execution_audit(sources):
    """覆盖MetaPath导入及spec_from_file_location直接执行，均在get_code之前核验。"""
    root = sources.root.resolve()
    pins = sources.config["runtime_sources"]
    tokenizer = sources.config.get("text_constraint_tokenizer", {}).get("source")
    external = {str(Path(tokenizer["path"]).resolve()): tokenizer["sha256"]} if tokenizer else {}
    observed = {}
    original = importlib.machinery.SourceFileLoader.get_code
    def get_code(loader, fullname):
        path = Path(loader.path).resolve()
        if path.is_relative_to(root) or str(path) in external:
            expected = pins.get(str(path), external.get(str(path)))
            need(expected is not None, "unpinned direct legacy import: " + str(path))
            observed[str(path)] = checked(FileRef(str(path), expected))
        return original(loader, fullname)
    with patch.object(importlib.machinery.SourceFileLoader, "get_code", new=get_code):
        yield observed
        need(observed == {p: checked(FileRef(p, v["sha256"])) for p, v in observed.items()},
             "runtime import source changed")

def inspect_imports(sources):
    """只解析工厂全部入口及CPU前端；不构建G/H，不执行attack/generate/metrics。"""
    entries = factory_entries()
    before_path, before_meta = list(sys.path), list(sys.meta_path)
    before_bytecode = sys.dont_write_bytecode
    before_modules = dict(sys.modules)
    symbols = {}
    with legacy_imports(sources, allow_import=True) as old:
        for name in entries:
            module = old(name)
            for symbol in ENTRY_SYMBOLS[name]:
                value = getattr(module, symbol)
                need(callable(value), "non-callable runtime symbol: " + name + "." + symbol)
                origin = str(Path(inspect.getsourcefile(inspect.unwrap(value))).resolve())
                need(origin in sources.config["runtime_sources"], "runtime symbol outside pinned sources: " + origin)
                symbols[name + "." + symbol] = {
                    "module": value.__module__, "source": origin,
                    "sha256": sources.config["runtime_sources"][origin],
                }
        # 构造仅CPU规则/BPE对象，触发真实延迟文件导入，绝不构造神经策略。
        frontend = old("text_frontend").CaptionFrontend()
        probe = old("tokenizer_probe").official_probe()
        need(callable(frontend.convert) and callable(probe.inspect), "frontend/probe interface mismatch")
        # 检查原前端声明的动态依赖（含ontology），不能只锁定Python普通import。
        for relative, sha in frontend.sources.items():
            path = str((sources.root / relative).resolve())
            need(sources.config["runtime_sources"].get(path) == sha, "frontend source pin mismatch: " + path)
            checked(FileRef(path, sha))
    need(sys.path == before_path and sys.meta_path == before_meta
         and sys.dont_write_bytecode == before_bytecode, "legacy namespace settings not restored")
    need(all(sys.modules.get(k) is v for k, v in before_modules.items()), "pre-existing module changed")
    leaked = [k for k, v in sys.modules.items() if k not in before_modules
              and getattr(v, "__file__", None)
              and Path(v.__file__).resolve().is_relative_to(sources.root)]
    need(not leaked, "legacy modules leaked: " + repr(leaked))
    return {"factory_entries": entries, "symbols": symbols, "namespace_restored": True,
            "frontend_sources_checked": len(frontend.sources),
            "tokenizer_context": 77}

def check_derived(config, derivation, preflight_fn):
    # 整个旧preflight也处于硬性CPU闸门内，而非仅保护最后两个import。
    with cpu_only() as attempts:
        control_paths = (Path(__file__), Path(__file__).with_name("check_attack_runtime.py"),
                         Path(derivation).resolve(), Path(config).resolve())
        from .artifacts import sha_file
        controls = {str(p.resolve()): checked(FileRef(str(p.resolve()), sha_file(p))) for p in control_paths}
        before_namespace = (list(sys.path), list(sys.meta_path), sys.dont_write_bytecode)
        d = verify_derivation(derivation, config)
        from .attacks.provenance import LegacyAttackSources
        sources = LegacyAttackSources(d["derived"]["path"])
        pins_before = {r.path: checked(r) for r in sources.runtime_refs}
        with source_execution_audit(sources) as executed:
            result = preflight_fn(config)
            need(result[1].fingerprint == sources.fingerprint, "preflight source mismatch")
            closure = inspect_imports(sources)
        sources.verify()
        need(pins_before == {r.path: checked(r) for r in sources.runtime_refs}, "runtime sources changed")
        need(before_namespace == (sys.path, sys.meta_path, sys.dont_write_bytecode),
             "preflight namespace settings not restored")
        need(controls == {p: checked(FileRef(p, v["sha256"])) for p, v in controls.items()},
             "CPU preflight controls changed")
        report = dict(result[-1])
        report.update(schema="tbps-gh-runtime-closure-1", status="PASS", closure=closure,
                      derivation=d, executed_sources=executed, runtime_source_count=len(pins_before),
                      forbidden_attempts=dict(attempts), weights_restored=0, neural_forwards=0,
                      neural_models_constructed=0, CUDA_initialized=False, backward=0,
                      import_audit_sha256=digest(executed), control_sources=controls,
                      preflight_namespace_restored=True)
        return report
