"""Stage 3B-R：完整加载入口、直接文件导入、CPU硬闸门及派生来源回归。"""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactError, sha_file
from tbps.attacks.legacy_runtime import G_REL, H_REL, legacy_imports
from tbps.runtime_closure import (cpu_only, inspect_imports, source_execution_audit,
                                factory_entries, verify_derivation, ENTRY_SYMBOLS)

@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = {
            G_REL+"/inference_once.py": "from image_only_generator import ImageOnlyGenerator\ndef load_generator(): pass\ndef generate_once(): pass\n",
            G_REL+"/image_only_generator.py": "from budget_projection import check_pixels\nclass ImageOnlyGenerator: pass\n",
            G_REL+"/budget_projection.py": "def check_pixels(): pass\n",
            G_REL+"/irra_image_adapter.py": "import cache_io\ndef pixels_from_pil(): pass\n",
            G_REL+"/cache_io.py": "import condition_schema\n",
            G_REL+"/condition_schema.py": "SCHEMA = 'fixture'\n",
            H_REL+"/phase5_joint/__init__.py": "",
            H_REL+"/phase5_joint/backend.py": "from policy import CharacterPolicy\nclass TextAttacker: pass\ndef validate_action(): pass\n",
            H_REL+"/policy.py": "from constraints import legal\nclass CharacterPolicy: pass\n",
            H_REL+"/constraints.py": "def legal(): pass\n",
            H_REL+"/tokenizer_probe.py": "class TokenizerProbe:\n def inspect(self): pass\ndef official_probe(): return TokenizerProbe()\n",
            H_REL+"/text_frontend.py": "class CaptionFrontend:\n sources = {}\n def convert(self): pass\n",
            H_REL+"/formal_training/__init__.py": "",
            H_REL+"/formal_training/validation.py": "from helper import stable\ndef state_sha_from(): pass\n",
            H_REL+"/helper.py": "def stable(): pass\n",
        }
        for rel, code in files.items():
            p = root/rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(code, encoding="utf-8")
        pins = {str(root/r): sha_file(root/r) for r in files}
        yield SimpleNamespace(root=root, config={"runtime_sources":pins})

def run_fixture(sources):
    with cpu_only(), source_execution_audit(sources) as executed:
        report = inspect_imports(sources)
    return report, executed

class ClosureTests(unittest.TestCase):
    def test_all_factory_entries_have_symbol_resolution(self):
        self.assertEqual(factory_entries(), sorted(ENTRY_SYMBOLS))

    def test_new_factory_entry_cannot_silently_escape_check(self):
        with patch("tbps.runtime_closure.inspect.getsource", return_value='def factory(): old("new_dependency")'):
            with self.assertRaisesRegex(ArtifactError, "entries changed"): factory_entries()

    def test_nonliteral_factory_import_is_rejected(self):
        with patch("tbps.runtime_closure.inspect.getsource", return_value='def factory(): old(variable)'):
            with self.assertRaisesRegex(ArtifactError, "uninspectable"): factory_entries()

    def test_positive_complete_closure_without_weights_forward_or_cuda(self):
        with fixture() as src:
            report, executed = run_fixture(src)
            self.assertTrue(report["namespace_restored"])
            self.assertEqual(len(executed), len(src.config["runtime_sources"]))
            self.assertIn(str(src.root/G_REL/"condition_schema.py"), executed)
            self.assertFalse(torch.cuda.is_initialized())

    def test_each_transitive_source_missing_is_rejected(self):
        with fixture() as src:
            original = dict(src.config["runtime_sources"])
            for path in original:
                with self.subTest(path=path):
                    src.config["runtime_sources"] = {p:h for p,h in original.items() if p != path}
                    with self.assertRaisesRegex(ArtifactError, "unpinned"): run_fixture(src)
                    self.assertNotIn("inference_once", sys.modules)

    def test_each_transitive_source_sha_tamper_is_rejected(self):
        with fixture() as src:
            original = dict(src.config["runtime_sources"])
            for path in original:
                with self.subTest(path=path):
                    src.config["runtime_sources"] = {**original, path:"0"*64}
                    with self.assertRaises(ArtifactError): run_fixture(src)

    def test_namespace_restored_after_success_and_failure(self):
        with fixture() as src:
            before = (list(sys.path),list(sys.meta_path),sys.dont_write_bytecode)
            self.assertNotIn("policy",sys.modules)
            run_fixture(src)
            self.assertEqual(before,(sys.path,sys.meta_path,sys.dont_write_bytecode))
            src.config["runtime_sources"].pop(str(src.root/H_REL/"helper.py"))
            with self.assertRaises(ArtifactError): run_fixture(src)
            self.assertEqual(before,(sys.path,sys.meta_path,sys.dont_write_bytecode))
            self.assertNotIn("policy",sys.modules)
            self.assertNotIn("formal_training",sys.modules)
            # 异常后锁也必须释放。
            src.config["runtime_sources"][str(src.root/H_REL/"helper.py")] = sha_file(src.root/H_REL/"helper.py")
            run_fixture(src)

    def test_existing_owned_namespace_is_not_replaced(self):
        with fixture() as src:
            sentinel = SimpleNamespace()
            with patch.dict(sys.modules, {"policy":sentinel}):
                with self.assertRaisesRegex(ArtifactError, "collision"): run_fixture(src)
                self.assertIs(sys.modules["policy"],sentinel)

    def test_direct_file_loader_missing_and_tampered_pins(self):
        with fixture() as src:
            path = src.root/"direct_helper.py"
            path.write_text("VALUE=1\n")
            def direct():
                with cpu_only(), source_execution_audit(src):
                    spec=importlib.util.spec_from_file_location("direct_unregistered",path)
                    module=importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module.VALUE
            with self.assertRaisesRegex(ArtifactError,"unpinned direct"): direct()
            src.config["runtime_sources"][str(path)]="0"*64
            with self.assertRaises(ArtifactError): direct()
            src.config["runtime_sources"][str(path)]=sha_file(path)
            self.assertEqual(direct(),1)


    def test_success_without_derivation_cannot_be_claimed(self):
        from tbps.check_attack_runtime import check
        with fixture() as src:
            with patch("tbps.check_attack_runtime.preflight",return_value=({},src,(),(),(),{},{})):
                with self.assertRaisesRegex(ArtifactError,"derivation manifest"):check("fixture")

    def test_source_change_during_import_audit_is_rejected(self):
        with fixture() as src:
            p=src.root/"direct.py";p.write_text("VALUE=1\n")
            src.config["runtime_sources"][str(p)]=sha_file(p)
            with self.assertRaises(ArtifactError):
                with source_execution_audit(src):
                    spec=importlib.util.spec_from_file_location("direct_change",p)
                    spec.loader.exec_module(importlib.util.module_from_spec(spec))
                    p.write_text("VALUE=2\n")

    def test_missing_runtime_symbol_is_rejected(self):
        with fixture() as src:
            # 入口助手必须真正可解析，不能仅导入同名空模块。
            p=src.root/H_REL/"formal_training/validation.py";p.write_text("VALUE=1\n")
            src.config["runtime_sources"][str(p)]=sha_file(p)
            with self.assertRaises(AttributeError):run_fixture(src)

class CpuGuardTests(unittest.TestCase):
    def test_weight_restore_forbidden(self):
        with self.assertRaisesRegex(ArtifactError,"torch.load"):
            with cpu_only(): torch.load("never-opened.pt")

    def test_neural_constructor_forbidden(self):
        with self.assertRaisesRegex(ArtifactError,"neural construction"):
            with cpu_only(): torch.nn.Linear(1,1)

    def test_neural_forward_forbidden(self):
        module=torch.nn.Identity()
        with self.assertRaisesRegex(ArtifactError,"neural forward"):
            with cpu_only(): module(torch.zeros(1))

    def test_cuda_entrypoints_forbidden_without_initializing(self):
        for name in ("init","_lazy_init"):
            with self.subTest(name=name), self.assertRaisesRegex(ArtifactError,"CUDA"):
                with cpu_only(): getattr(torch.cuda,name)()
        self.assertFalse(torch.cuda.is_initialized())

    def test_caught_forbidden_attempt_still_fails(self):
        with self.assertRaisesRegex(ArtifactError,"attempted"):
            with cpu_only():
                try: torch.load("never-opened.pt")
                except ArtifactError: pass

    def test_guards_restore_original_functions_on_failure(self):
        original=torch.load
        with self.assertRaises(ArtifactError):
            with cpu_only(): torch.load("never-opened.pt")
        self.assertIs(torch.load,original)

    def test_backward_forbidden(self):
        with self.assertRaisesRegex(ArtifactError,"backward"):
            with cpu_only(): torch.zeros(1,requires_grad=True).sum().backward()

    def test_already_initialized_cuda_is_rejected(self):
        with patch.object(torch.cuda,"is_initialized",return_value=True):
            with self.assertRaisesRegex(ArtifactError,"isolated CPU"):
                with cpu_only(): pass

class DerivationTests(unittest.TestCase):
    def setup_files(self,tmp):
        root=Path(tmp)
        # 固定原源码只作为小型只读Fixture复制；不加载历史资产。
        source=Path(__file__).resolve().parents[2]/"prototypes/attribute_image_generator_v2_20261014/condition_schema.py"
        dst=root/"legacy"/G_REL/"condition_schema.py";dst.parent.mkdir(parents=True)
        dst.write_bytes(source.read_bytes())
        parent={"schema":"tbps-legacy-attack-plugins-1","legacy_root":str(root/"legacy"),
                "runtime_sources":{},"unchanged_field":"fixed"}
        child={**parent,"runtime_sources":{str(dst):sha_file(dst)}}
        def write(name,value):
            p=root/name;p.write_text(json.dumps(value))
            return {"path":str(p),"sha256":sha_file(p)}
        pref=write("parent.json",parent);cref=write("child.json",child)
        config=write("cpu.json",{"schema":"tbps-gh-equivalence-smoke-1","attack_sources":cref})
        deriv={"schema":"tbps-runtime-source-derivation-1","base_commit":"0fb90cd1b280da886aec08834dc13d28eacd5848",
               "parent":pref,"derived":cref,"added_runtime_sources":child["runtime_sources"],"reason":"fixture"}
        dref=write("derivation.json",deriv)
        return root,config,dref,deriv,child

    def test_exact_addition_passes_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,c,d,deriv,child=self.setup_files(tmp)
            before=(root/"parent.json").stat().st_mtime_ns,sha_file(root/"parent.json")
            verify_derivation(d["path"],c["path"])
            self.assertEqual(before,((root/"parent.json").stat().st_mtime_ns,sha_file(root/"parent.json")))

    def test_parent_sha_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,c,d,deriv,child=self.setup_files(tmp)
            (root/"parent.json").write_text("{}")
            with self.assertRaises(ArtifactError):verify_derivation(d["path"],c["path"])

    def test_changing_existing_fields_rejected_even_with_updated_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,c,d,deriv,child=self.setup_files(tmp)
            child["unchanged_field"]="relaxed"
            (root/"child.json").write_text(json.dumps(child))
            deriv["derived"]["sha256"]=sha_file(root/"child.json")
            (root/"derivation.json").write_text(json.dumps(deriv))
            with self.assertRaisesRegex(ArtifactError,"additive"):verify_derivation(d["path"],c["path"])

    def test_cpu_config_cannot_bind_parent_or_other_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,c,d,deriv,child=self.setup_files(tmp)
            (root/"cpu.json").write_text(json.dumps({"schema":"tbps-gh-equivalence-smoke-1","attack_sources":deriv["parent"]}))
            with self.assertRaisesRegex(ArtifactError,"does not bind"):verify_derivation(d["path"],c["path"])

    def test_added_file_sha_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,c,d,deriv,child=self.setup_files(tmp)
            Path(next(iter(child["runtime_sources"]))).write_text("tampered")
            with self.assertRaises(ArtifactError):verify_derivation(d["path"],c["path"])

if __name__=="__main__": unittest.main()
