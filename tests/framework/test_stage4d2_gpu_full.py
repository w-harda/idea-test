"""仅合成CPU Fixture与控制流程Mock，不加载真实G/H/IRRA或CUDA。"""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch,Mock
import tempfile
import unittest
import torch
from tbps.artifacts import ArtifactError,json_bytes
from tbps.materialization.throughput import fixture
from tbps.materialization.bulk_session import StreamingMaterializedStore
from tbps.materialization.full_runner import stream_images
from tbps.materialization.full_gpu import fresh_images,verify_subset,main,wait_owned_child,frozen_identity
from tbps.materialization.full_gpu_audit import verify_images,collection_units
from tbps.materialization.gpu_safety import GObservation,g_load_only,no_optimization,FullGuard,own_gpu_only
from tbps.engines.replay_control import ResourceLimit
from tbps.materialization.audit_retained import snapshot
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.attacks.gv2 import GV2Plugin
from stage3a_fixtures import GBackend,support

class GPUControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-g-full-")
        base=fixture(Path(self.temp.name),queries=4,gallery=2,sources=1,checkpoint_bytes=128)
        self.store=StreamingMaterializedStore(base.store,base.plan)
    def tearDown(self):self.temp.cleanup()
    def factory(self,observer=None):
        @contextmanager
        def load():
            with legacy_imports(support(),allow_import=True) as old:
                backend=GBackend(old("inference_once"),pixels=torch.full((1,3,384,128),.5))
                plugin=GV2Plugin(backend=backend,provenance=self.store.plan.g)
                if observer:
                    with observer.attach(plugin):yield plugin
                else:yield plugin
        return load
    def test_fresh_store_before_model(self):
        fresh_images(self.store)
    def test_claim_rejects_before_loading(self):
        b=self.store.image_request("g0").binding;uid=self.store.unit_id(b)
        self.store.store.publish("generation-claim-"+uid,{"claim.json":b"{}"},{"fixture":True})
        with self.assertRaises(ArtifactError):fresh_images(self.store)
    def test_pending_rejects_before_loading(self):
        (self.store.store.root/".pending-unit").mkdir()
        with self.assertRaises(ArtifactError):fresh_images(self.store)
    def test_default_cli_is_only_CPU_preflight(self):
        with patch("tbps.materialization.full_gpu.preflight",return_value=({},None,None,None,
                  {"status":"G_FULL_CPU_PREFLIGHT_PASS"},{})),patch("subprocess.Popen",side_effect=AssertionError("GPU child")):
            self.assertEqual(main(["--config","fixture"]),0)
    def test_actual_G_hooks_order_and_count(self):
        observer=GObservation(("g0","g1"))
        report=stream_images(self.store,self.factory(observer))
        self.assertEqual(observer.started,2);self.assertEqual(observer.completed,2)
        self.assertEqual(report["generated"],2);self.assertEqual(report["skipped"],0)
        self.assertEqual([r["image_id"] for r in observer.records],["g0","g1"])
    def test_wrong_forward_order_stops_with_claim(self):
        observer=GObservation(("g1","g0"))
        with self.assertRaises(ArtifactError):stream_images(self.store,self.factory(observer))
        self.assertEqual(observer.started,0)
        uid=self.store.unit_id(self.store.image_request("g0").binding)
        self.assertTrue((self.store.store.root/("generation-claim-"+uid)).exists())
        self.assertFalse((self.store.store.root/"image-full").exists())
    def test_historical_pixel_mismatch_preserved_and_stops(self):
        old={"g0":dict.fromkeys(("original_pixels_sha256","adversarial_pixels_sha256","effective_delta_sha256"),"0"*64)}
        observer=GObservation(("g0","g1"),old)
        with self.assertRaises(ArtifactError):stream_images(self.store,self.factory(observer))
        self.assertEqual(observer.completed,1)
        self.assertFalse(observer.comparisons["g0"]["exact"])
        self.assertFalse((self.store.store.root/"image-full").exists())
    def test_G_load_only_once_and_other_checkpoint_refused(self):
        p=Path(self.temp.name)/"test.pt";torch.save({"x":torch.ones(1)},p)
        counts={"G_load_attempts":0,"pixel_tensor_loads":0}
        with g_load_only(p,self.store.store.root,counts):
            torch.load(p,weights_only=True)
            with self.assertRaises(ArtifactError):torch.load(p,weights_only=True)
            with self.assertRaises(ArtifactError):torch.load(p.with_name("other.pt"))
        self.assertEqual(counts["G_load_attempts"],1)
    def test_optimization_blocked_and_namespace_restored(self):
        original=torch.autograd.grad
        with self.assertRaises(ArtifactError):
            with no_optimization():torch.autograd.grad(torch.ones(1),torch.ones(1))
        self.assertIs(torch.autograd.grad,original)
    def test_observer_rejects_non_G_neural_forward(self):
        with self.factory()() as plugin:
            other=torch.nn.Identity()
            with GObservation(("g0","g1")).attach(plugin):
                with self.assertRaises(ArtifactError):other(torch.ones(1))
    def test_no_artificial_RSS_cap(self):
        c={"max_seconds":7200,"system_ram_reserve_bytes":1024,"runtime_disk_reserve_bytes":1}
        guard=FullGuard(c,self.temp.name,memory=lambda:{"MemAvailable":1024**4},sample=lambda:8*1024**3)
        guard.check()
        self.assertIsNone(guard.snapshot()["rss_hard_limit_bytes"])
        self.assertEqual(guard.peak,8*1024**3)
    def test_low_actual_RAM_stops(self):
        c={"max_seconds":7200,"system_ram_reserve_bytes":1024,"runtime_disk_reserve_bytes":1}
        guard=FullGuard(c,self.temp.name,memory=lambda:{"MemAvailable":100},sample=lambda:100)
        with self.assertRaises(ResourceLimit):guard.check()
    def test_resource_time_limit(self):
        tick=[0.]
        c={"max_seconds":7200,"system_ram_reserve_bytes":1,"runtime_disk_reserve_bytes":1}
        guard=FullGuard(c,self.temp.name,memory=lambda:{"MemAvailable":10000},sample=lambda:100,clock=lambda:tick[0])
        tick[0]=7200
        with self.assertRaises(ResourceLimit):guard.check()
    def test_other_GPU_process_conflict(self):
        with patch("subprocess.run",return_value=Mock(stdout=str(__import__("os").getpid()+1))):
            with self.assertRaises(ArtifactError):own_gpu_only()
    def test_old_files_preserved_when_only_append(self):
        before=snapshot(self.store.store.root)
        (self.store.store.root/"extra").write_bytes(b"append")
        self.assertEqual(verify_subset(before),len(before))
        p=next(iter(before));Path(p).write_bytes(b"changed")
        with self.assertRaises(ArtifactError):verify_subset(before)
    def test_independent_image_audit_reads_claims_and_pixels(self):
        observer=GObservation(("g0","g1"))
        stream_images(self.store,self.factory(observer))
        body,core,images,texts=collection_units(self.store,"image-full")
        before=snapshot(self.store.store.root)
        report=verify_images(self.store,images,observer.records,{})
        self.assertEqual(report["count"],2);self.assertEqual(texts,())
        self.assertLessEqual(report["maximum_linf"],8/255)
        self.assertEqual(before,snapshot(self.store.store.root))
    def test_image_audit_rejects_actual_count_mismatch(self):
        observer=GObservation(("g0","g1"))
        stream_images(self.store,self.factory(observer))
        _,_,images,_=collection_units(self.store,"image-full")
        with self.assertRaises(ArtifactError):verify_images(self.store,images,observer.records[:1],{})


class LaunchProtectionTests(unittest.TestCase):
    def test_accidental_new_library_production_import_refused(self):
        with self.assertRaises(ArtifactError):frozen_identity()
    def test_interrupt_stops_only_created_process(self):
        process=Mock();process.wait.side_effect=[KeyboardInterrupt(),0];process.poll.return_value=None
        with self.assertRaises(KeyboardInterrupt):wait_owned_child([],{},None,10,spawn=lambda *a,**kw:process)
        process.terminate.assert_called_once();process.kill.assert_not_called()
    def test_unresponsive_created_child_is_killed_without_retry(self):
        import subprocess
        process=Mock();process.poll.return_value=None
        process.wait.side_effect=[KeyboardInterrupt(),subprocess.TimeoutExpired("owned",30),0]
        factory=Mock(return_value=process)
        with self.assertRaises(KeyboardInterrupt):wait_owned_child([],{},None,10,spawn=factory)
        factory.assert_called_once();process.terminate.assert_called_once();process.kill.assert_called_once()
