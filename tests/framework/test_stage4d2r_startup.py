"""执行真实child控制流/门禁/包装/流式事务；设备与权重入口均为CPU Mock。"""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import os
import sys
import tempfile
import hashlib
import time
import unittest
import torch
import tbps.materialization.full_gpu as runner
import tbps.materialization.gpu_safety as safety
import tbps.materialization.legacy_text  # 提前完成公共模块导入，避免Fixture执行审计误判
from tbps.artifacts import ArtifactError, ArtifactStore, json_bytes, read_json, digest
from tbps.materialization.throughput import fixture, text_result, validate
from tbps.materialization.full_runner import publish_texts, stream_images
from tbps.materialization.irra_startup import CPUStartupGate, isolation_snapshot
from tbps.runtime_closure import cpu_only
from tbps.materialization.bulk_session import StreamingMaterializedStore
from tbps.attacks.legacy_runtime import legacy_imports, _LOCK
from tbps.attacks.gv2 import GV2Plugin
from stage3a_fixtures import GBackend, support


class ChildFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tbps-framework-startup-repair-")
        self.root = Path(self.tmp.name)
        base = fixture(self.root, gallery=2, queries=4, sources=1, checkpoint_bytes=128)
        self.store = StreamingMaterializedStore(base.store, base.plan)
        self.output = self.root / "attempt-v2"
        self.records = ArtifactStore(self.output, new_base=self.root)
        self.config = self.root / "control.json"
        self.config.write_text("{}")
        self.c = {"frozen_config": {"path": str(self.config)}, "max_seconds": 7200,
                  "minimum_available_ram_bytes": 1, "system_ram_reserve_bytes": 1,
                  "runtime_disk_reserve_bytes": 1}
        self.events = []
        self.gates = []
        self.observers = []
        self.token = "fixture-token"
        self.records.publish("attempt", {
            "preflight.json": json_bytes({"status": "G_FULL_CPU_PREFLIGHT_PASS",
                "plan_id": self.store.plan.id, "CPU_contexts_restored": True,
                "GH_status": "PASS", "old_pixel_reference": {}}),
            "control.json": json_bytes({"token_sha256": digest(self.token),
                "config": asdict(runner.ref(self.config)), "started_unix": time.time()}),
            "old_files.json": b"{}"}, {"fixture": True})
        self.native_load = torch.load
        self.before = isolation_snapshot()
        self.optim_before = (torch.Tensor.backward, torch.autograd.functional.jvp,
                             torch.func.jvp, torch.optim.Optimizer.__init__)
        self.modules_before = dict(sys.modules)
        self.fault = None
        self.forbidden = None

    def tearDown(self):
        self.assertEqual(isolation_snapshot(), self.before)
        self.assertEqual(self.optim_before, (torch.Tensor.backward, torch.autograd.functional.jvp,
                                            torch.func.jvp, torch.optim.Optimizer.__init__))
        self.assertFalse(_LOCK.locked())
        self.assertFalse(torch.cuda.is_initialized())
        self.assertTrue(all(sys.modules.get(k) is v for k, v in self.modules_before.items()))
        self.tmp.cleanup()

    def fail(self, name):
        if self.fault == name:
            raise ArtifactError("fixture failure: " + name)

    @contextmanager
    def cpu(self):
        self.events.append("CPU_read_start")
        with cpu_only():
            self.assertFalse(torch.cuda.is_initialized())
            self.assertFalse(self.gates[-1].exited)
            yield
        self.events.append("CPU_context_exit")

    def gate(self):
        gate = CPUStartupGate(self.events.append)
        self.gates.append(gate)
        if self.fault == "old_order":
            native = gate.initialize_cuda
            def old_order():
                with safety.no_optimization():
                    native()
            gate.initialize_cuda = old_order
        return gate

    def prepare(self, path):
        self.events.append("CPU_sources_checked")
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(self.gates[-1].exited)
        self.fail("source")
        if self.fault == "early_cuda":
            self.gates[-1].initialize_cuda()
        if self.fault == "namespace":
            sys.path.append("/fixture/leaked-import-path")
        if self.fault == "lock":
            _LOCK.acquire()
        if self.fault == "plan":
            from types import SimpleNamespace
            return {}, SimpleNamespace(id="wrong"), support(), None, None
        return {}, self.store.plan, support(), None, None

    def cuda_reset(self):
        self.events.append("CUDA_mock")
        self.assertTrue(self.gates[-1].completed and self.gates[-1].exited)
        self.assertEqual(isolation_snapshot(), self.gates[-1].before)
        self.assertFalse(_LOCK.locked())
        self.assertFalse(torch.cuda.is_initialized())
        self.fail("cuda")

    @contextmanager
    def no_opt(self):
        self.assertIn("CUDA_mock", self.events)
        self.events.append("no_opt_install")
        self.fail("no_opt")
        with safety.no_optimization() as blocked:
            self.events.append("no_opt_ready")
            yield blocked
        self.events.append("no_opt_restored")

    @contextmanager
    def only_g(self, *args):
        self.assertIn("CUDA_mock", self.events)
        self.assertIn("no_opt_ready", self.events)
        self.events.append("load_guard_install")
        self.fail("load_guard")
        with safety.g_load_only(*args):
            self.events.append("load_guard_ready")
            yield
        self.events.append("load_guard_restored")

    def load(self, path, *args, **kwargs):
        if Path(path) == Path(self.store.plan.g.checkpoint.path):
            # 不反序列化任何权重；只是保留实际torch.load包装的调用路径。
            self.events.append("G_weight_mock")
            return {"fixture": True}
        return self.native_load(path, *args, **kwargs)

    def observer(self, *args):
        value = safety.GObservation(*args)
        self.observers.append(value)
        return value

    @contextmanager
    def plugins(self, legacy, **kwargs):
        self.events.append("G_load_entry")
        self.assertEqual(kwargs, {"components": ("gv2",), "device": "cuda:0",
                                  "allow_model_load": True})
        self.assertIn("load_guard_ready", self.events)
        self.assertIn("no_opt_ready", self.events)
        self.assertIsNot(torch.load, self.gates[-1].before[0][0])
        self.assertIsNot(torch.autograd.grad, self.gates[-1].before[0][6])
        self.fail("factory")
        if self.forbidden == "other_weight":
            torch.load(self.root / "IRRA-forbidden.pt", weights_only=True)
        if self.forbidden == "backward":
            torch.ones(1).backward()
        if self.forbidden == "grad":
            torch.autograd.grad(torch.ones(1), torch.ones(1))
        if self.forbidden == "autograd_backward":
            torch.autograd.backward(torch.ones(1))
        if self.forbidden == "jvp":
            torch.autograd.functional.jvp(lambda x: x, torch.ones(1), torch.ones(1))
        if self.forbidden == "func_jvp":
            torch.func.jvp(lambda x: x, (torch.ones(1),), (torch.ones(1),))
        if self.forbidden == "optimizer":
            torch.optim.Optimizer([], {})
        torch.load(self.store.plan.g.checkpoint.path, weights_only=True)
        with legacy_imports(legacy, allow_import=True) as old:
            backend = GBackend(old("inference_once"), pixels=torch.full((1, 3, 384, 128), .5))
            plugin = GV2Plugin(backend=backend, provenance=self.store.plan.g)
            def before_forward(module, args):
                key = self.observers[-1].active
                req = self.store.image_request(key)
                uid = self.store.unit_id(req.binding)
                claim = "generation-claim-" + uid
                self.store.store.verify(claim)
                self.assertFalse((self.store.store.root / ("generation-proof-" + uid)).exists())
                self.events.append("Claim_before_G:" + key)
                self.fail("forward")
                if self.forbidden == "non_G_forward":
                    torch.nn.Identity()(args[0])
            hook = backend.model.register_forward_pre_hook(before_forward)
            try:
                yield {"gv2": plugin}
            finally:
                hook.remove()
                self.assertEqual(len(backend.model._forward_pre_hooks), 0)
                self.assertEqual(len(backend.model._forward_hooks), 0)

    def run_child(self):
        # 所有真实CUDA入口硬拒绝；唯一启动调用reset被Mock，无设备初始化。
        with ExitStack() as stack:
            def p(owner, name, **kw): return stack.enter_context(patch.object(owner, name, **kw))
            p(runner, "OUTPUT", new=str(self.output))
            p(runner, "FULL", new=str(self.root))
            p(runner, "OLD_BASE", new=str(self.root))
            p(runner, "PLAN", new=self.store.plan.id)
            p(runner, "control", return_value=self.c)
            p(runner, "frozen_identity", return_value={})
            p(runner, "records", return_value=self.records)
            p(runner, "CPUStartupGate", side_effect=self.gate)
            p(runner, "cpu_only", new=self.cpu)
            p(runner, "prepare", side_effect=self.prepare)
            p(runner, "StreamingMaterializedStore", return_value=self.store)
            p(runner, "gpu_available", side_effect=lambda c: (
                self.fail("resource") or {"fixture": True}))
            p(runner, "own_gpu_only", return_value=[])
            p(runner, "ram", return_value={"MemAvailable": 0 if self.fault == "ram" else 2**40})
            p(runner, "no_optimization", new=self.no_opt)
            p(runner, "g_load_only", new=self.only_g)
            p(runner, "load_legacy_plugins", new=self.plugins)
            p(runner, "GObservation", side_effect=self.observer)
            p(torch, "load", new=self.load)
            p(sys, "path", new=list(sys.path))
            if self.fault == "lock":
                stack.callback(lambda: _LOCK.release() if _LOCK.locked() else None)
            p(torch.cuda, "reset_peak_memory_stats", side_effect=self.cuda_reset)
            p(torch.cuda, "manual_seed_all", return_value=None)
            for name in ("init", "_lazy_init"):
                p(torch.cuda, name, side_effect=AssertionError("real CUDA forbidden"))
            if hasattr(torch._C, "_cuda_init"):
                p(torch._C, "_cuda_init", side_effect=AssertionError("native CUDA forbidden"))
            for name in ("memory_allocated", "memory_reserved"):
                p(torch.cuda, name, return_value=0)
            stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}))
            stack.enter_context(patch("builtins.print"))
            result = runner.child(str(self.config), self.token)
        return result, read_json(self.output / "gpu-result/report.json")

    def test_actual_child_correct_order_real_hook_claim_and_readback(self):
        exitcode, report = self.run_child()
        # 2图Fixture不可冒充3074正式完成：保留生产固定总数拒绝。
        self.assertEqual(exitcode, 1)
        self.assertEqual(report["error"], "exact 3074 generation / one restoration")
        names = ["CPU_read_start", "CPU_sources_checked", "CPU_context_exit",
                 "cpu_sources_verified", "cpu_isolation_exited", "cuda_initialization_call",
                 "CUDA_mock", "no_opt_ready", "load_guard_ready", "G_load_entry", "G_weight_mock",
                 "Claim_before_G:g0", "Claim_before_G:g1"]
        self.assertEqual(sorted(names, key=self.events.index), names)
        self.assertEqual(report["actual_G_forwards"]["started"], 2)
        self.assertEqual(report["actual_G_forwards"]["completed"], 2)
        self.assertEqual(report["counts"]["G_load_attempts"], 1)
        self.assertEqual(report["counts"]["G_loads"], 1)
        self.assertEqual(report["model_before"], report["model_after"])
        self.assertEqual(report["stream"]["generated"], 2)
        self.store.store.verify("image-full")
        for key in ("g0", "g1"):
            uid = self.store.unit_id(self.store.image_request(key).binding)
            self.store.store.verify("generation-proof-" + uid)
        self.assertIn("load_guard_restored", self.events)
        self.assertIn("no_opt_restored", self.events)

    def test_old_combined_order_is_rejected_by_actual_gate(self):
        self.fault = "old_order"
        _, report = self.run_child()
        self.assertIn("CPU contexts must exit", report["error"])
        self.assertNotIn("CUDA_mock", self.events)
        self.assertNotIn("G_load_entry", self.events)

    def test_early_cuda_in_actual_cpu_phase_is_rejected(self):
        self.fault = "early_cuda"
        with self.assertRaisesRegex(ArtifactError, "CPU contexts must exit"):
            self.run_child()
        self.assertNotIn("CUDA_mock", self.events)
        self.assertNotIn("G_load_entry", self.events)

    def test_cpu_source_failure_never_loads_model(self):
        self.fault = "source"
        with self.assertRaisesRegex(ArtifactError, "fixture failure: source"):
            self.run_child()
        self.assertNotIn("G_load_entry", self.events)
        self.assertFalse((self.output / "gpu-start").exists())

    def test_cpu_pending_failure_never_loads_model(self):
        (self.store.store.root / ".pending-fixture").mkdir()
        with self.assertRaisesRegex(ArtifactError, "pending store"):
            self.run_child()
        self.assertNotIn("CUDA_mock", self.events)
        self.assertNotIn("G_load_entry", self.events)

    def test_failures_in_each_gpu_startup_stage_block_generation(self):
        # 每个参数化子案例各自独立临时Attempt，未恢复真实模型。
        for fault in ("cuda", "no_opt", "load_guard", "factory", "forward"):
            with self.subTest(fault=fault):
                other = ChildFlowTests()
                other.setUp()
                try:
                    other.fault = fault
                    _, report = other.run_child()
                    self.assertIn("fixture failure: " + fault, report["error"])
                    self.assertEqual(report["actual_G_forwards"]["completed"], 0)
                    if fault in ("cuda", "no_opt", "load_guard"):
                        self.assertNotIn("G_load_entry", other.events)
                    self.assertFalse((other.store.store.root / "image-full").exists())
                finally:
                    other.tearDown()

    def test_all_prohibitions_apply_inside_actual_child_factory(self):
        for operation in ("other_weight", "backward", "grad", "autograd_backward",
                          "jvp", "func_jvp", "optimizer"):
            with self.subTest(operation=operation):
                other = ChildFlowTests()
                other.setUp()
                try:
                    other.forbidden = operation
                    _, report = other.run_child()
                    self.assertEqual(report["actual_G_forwards"]["completed"], 0)
                    self.assertEqual(report["counts"]["G_loads"], 0)
                    self.assertEqual(report["counts"]["G_load_attempts"], 0)
                    self.assertTrue("forbids" in report["error"] or "forbidden" in report["error"])
                    self.assertFalse((other.store.store.root / "image-full").exists())
                finally:
                    other.tearDown()

    def test_second_private_child_attempt_refused_before_cuda(self):
        self.run_child()
        self.events.clear()
        with self.assertRaisesRegex(ArtifactError, "already consumed"):
            self.run_child()
        self.assertNotIn("CUDA_mock", self.events)

    def test_wrong_launch_token_refused_before_cpu_and_gpu(self):
        self.token = "wrong"
        with self.assertRaisesRegex(ArtifactError, "immutable CPU launch gate"):
            self.run_child()
        self.assertEqual(self.events, [])

    def test_cpu_plan_namespace_lock_ram_and_resource_fail_before_model(self):
        for fault in ("plan", "namespace", "lock", "ram", "resource"):
            with self.subTest(fault=fault):
                other = ChildFlowTests()
                other.setUp()
                try:
                    other.fault = fault
                    with self.assertRaises(ArtifactError):
                        other.run_child()
                    self.assertNotIn("CUDA_mock", other.events)
                    self.assertNotIn("G_load_entry", other.events)
                finally:
                    other.tearDown()

    def test_non_G_forward_is_rejected_under_actual_child_observer(self):
        self.forbidden = "non_G_forward"
        _, report = self.run_child()
        self.assertIn("non-G neural forward forbidden", report["error"])
        self.assertEqual(report["actual_G_forwards"]["completed"], 0)
        self.assertFalse((self.store.store.root / "image-full").exists())

    def test_actual_joint_seal_path_reuses_fixture_images_and_texts(self):
        from dataclasses import replace
        from tbps.materialization.full_audit import verify_text_units
        from tbps.materialization.full_gpu_audit import collection_units, verify_images
        from tbps.materialization.audit_retained import snapshot
        from tbps.materialization import full_audit, legacy_text
        self.run_child()
        _, _, images, _ = collection_units(self.store, "image-full")
        verify_images(self.store, images, self.observers[-1].records, {})
        pairs = []
        for i in range(4):
            req, result = text_result(self.store, i)
            trace = result.trace()
            trace["original_utf8_sha256"] = hashlib.sha256(req.public.caption.encode()).hexdigest()
            trace["adversarial_utf8_sha256"] = hashlib.sha256(result.payload.encode()).hexdigest()
            pairs.append((req, replace(result, trace_json=json_bytes(trace))))
        class Source:
            def iter_outputs(inner): return iter(pairs)
            def report(inner): return {"status": "TEXT_SOURCE_VERIFIED"}
        source = Source()
        publish_texts(self.store, source, validate, expected={"edited":4,"NOOP":0,"edits":4})
        original_texts = snapshot(self.store.store.root / "text-full")
        audited = verify_text_units(self.store, source, validate, {"edited":4,"NOOP":0,"edits":4})
        @contextmanager
        def native(_): yield validate
        def verify(store, historical, validator, expected):
            self.assertEqual(expected, {"edited":6116,"NOOP":40,"edits":13537})
            # 固定生产计数仅在Fixture来源适配层映射为4条；原审计/集合算法实际执行。
            return verify_text_units(store, historical, validator, {"edited":4,"NOOP":0,"edits":4})
        with patch.object(runner, "OUTPUT", str(self.output)), patch.object(runner, "FULL", str(self.root)), \
             patch.object(runner, "OLD_BASE", str(self.root)), patch.object(runner, "PLAN", self.store.plan.id), \
             patch.object(runner, "control", return_value=self.c), \
             patch.object(runner, "prepare", return_value=({}, self.store.plan, type("L",(),{"config_ref":None})(), None, None)), \
             patch.object(runner, "StreamingMaterializedStore", return_value=self.store), \
             patch.object(runner, "native_action_validator", new=native), \
             patch.object(full_audit, "verify_text_units", side_effect=verify), \
             patch.object(full_audit, "EXPECTED_ACTION_DIGEST", audited["ordered_text_action_digest"]), \
             patch.object(legacy_text, "HistoricalTextSource", return_value=source), \
             patch.object(torch.cuda, "init", side_effect=AssertionError("CUDA forbidden")), \
             patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")):
            body = runner.seal_joint(str(self.config), {})
        self.assertEqual(body["G_forwards_during_composition"], 0)
        _, _, joint_images, joint_texts = collection_units(self.store, "joint-full")
        self.assertEqual(len(joint_images), 2)
        self.assertEqual(len(joint_texts), 4)
        self.assertEqual(images, joint_images)
        self.assertEqual(original_texts, snapshot(self.store.store.root / "text-full"))
        self.assertFalse(torch.cuda.is_initialized())


class DerivedControlTests(unittest.TestCase):
    def parent_control(self,path):
        # 历史v2配置必须继续由其原入口验证；新v3入口另有独立测试。
        import importlib.util
        from tbps.artifacts import FileRef,checked
        cfg=read_json(path)
        sources=cfg["entry_sources"]
        def load(name,filename):
            source=next(p for p in sources if Path(p).name==filename)
            checked(FileRef(source,sources[source]))
            spec=importlib.util.spec_from_file_location(name,source)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module
        safety=load("_frozen_stage4d2r_safety","gpu_safety.py")
        with patch.dict(sys.modules,{"gpu_safety":safety}):
            entry=load("_frozen_stage4d2r_entry","full_gpu.py")
        return entry.control(path)
    def test_derived_config_preserves_scope_plan_and_resources(self):
        root = Path(__file__).resolve().parents[2]
        parent = read_json(root / "configs/framework/stage4d2_generation.server.json")
        child = read_json(root / "configs/framework/stage4d2r_generation.server.json")
        self.assertEqual({k:v for k,v in parent.items() if k not in ("output","entry_sources")},
                         {k:v for k,v in child.items() if k not in ("output","entry_sources")})
        self.assertTrue(child["output"].endswith("/generation_v2"))
        self.assertEqual(self.parent_control(root / "configs/framework/stage4d2r_generation.server.json"), child)
        self.assertNotEqual(parent["output"], child["output"])

    def test_parent_v1_config_cannot_start_repaired_v2_entry(self):
        root = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(ArtifactError, "fixed Stage4D2"):
            runner.control(root / "configs/framework/stage4d2_generation.server.json")


    def test_derivation_sha_and_exact_allowed_delta(self):
        from tbps.artifacts import FileRef, checked
        root = Path(__file__).resolve().parents[2]
        derivation = read_json(root / "configs/framework/stage4d2r_derivation.server.json")
        self.assertFalse(derivation["GPU_execution_authorized"])
        for key in ("parent", "derived"):
            checked(FileRef(**derivation[key]))
        parent, child = (read_json(derivation[k]["path"]) for k in ("parent", "derived"))
        self.assertEqual([k for k in parent if parent[k] != child[k]],
                         derivation["changed_fields"])
        for path, sha in child["entry_sources"].items():
            checked(FileRef(path, sha))
