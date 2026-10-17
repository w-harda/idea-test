"""真实PIL/Resize/ToTensor + 合成CPU G + 原Claim事务；无真实权重/CUDA。"""
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch
import tempfile
import unittest
import torch
from PIL import Image
from torchvision import transforms
import tbps.materialization.generation as generation
import tbps.materialization.full_runner as full_runner
from tbps.artifacts import ArtifactStore, FileRef, ArtifactError, checked, read_json, json_bytes, sha_file
from tbps.materialization.throughput import fixture
from tbps.materialization.bulk_session import StreamingMaterializedStore
from tbps.materialization.audit_retained import snapshot
from tbps.materialization.claim_recovery import FirstClaimRecovery,stream_with_recovery,verify_recovery_completion
from tbps.materialization.full_gpu import fresh_images
from tbps.materialization.full_gpu_audit import collection_units,verify_images
from tbps.materialization.gpu_safety import GObservation,no_optimization,g_load_only
from tbps.attacks.legacy_runtime import legacy_imports,_LegacyG
from tbps.attacks.gv2 import GV2Plugin
from stage3a_fixtures import support,MockG

def ref(path): return FileRef(str(Path(path).resolve()),sha_file(path))

class World:
    def __init__(self,root,count=2):
        self.root=root
        base=fixture(root,gallery=count,queries=4,sources=1,checkpoint_bytes=128)
        gallery=[]
        for item in base.plan.manifest.gallery:
            path=Path(base.plan.image_root)/item.image_rel_path
            Image.new("RGB",(11,19),(50,100,150)).save(path,format="PNG")
            gallery.append(replace(item,image_sha256=sha_file(path)))
        plan=replace(base.plan,manifest=replace(base.plan.manifest,gallery=tuple(gallery)))
        self.store=StreamingMaterializedStore(ArtifactStore(root/"full"/"artifacts",new_base=root),plan)
        self.store.publish_plan()
        self.old=ArtifactStore(root/"failed-generation-v2",new_base=root)
        baseline=snapshot(self.store.store.root.parent)
        self.old.publish("attempt",{"old_files.json":json_bytes(baseline),"preflight.json":json_bytes({
            "status":"G_FULL_CPU_PREFLIGHT_PASS","plan_id":plan.id,
            "CPU_contexts_restored":True,"GH_status":"PASS"})},{"fixture":True})
        self.req=self.store.image_request("g0");uid=self.store.unit_id(self.req.binding)
        generation._prepare(self.store,self.req)
        self.old.publish("gpu-start",{"claim.json":b"{}"},{"fixture":True})
        self.report={"status":"FAIL","plan_id":plan.id,
            "counts":{"G_load_attempts":1,"G_loads":1,"pixel_tensor_loads":0},
            "actual_G_forwards":{"started":0,"completed":0,"records":[],"historical_two_image_comparisons":{}},
            "error":"non-G neural forward forbidden","traceback":"clean_reader pixels_from_pil",
            **dict.fromkeys(("H_loads","H_forwards","IRRA_loads","IRRA_forwards","backward","optimizer","JVP"),0)}
        self.old.publish("gpu-result",{"report.json":json_bytes(self.report)},{"fixture":True})
        self.old.publish("result",{"report.json":json_bytes({"status":"FAIL","plan_id":plan.id})},{"fixture":True})
        self.spec={"schema":"tbps-zero-forward-first-claim-1","plan_id":plan.id,"image_id":"g0","unit_id":uid,
            "failed_attempt":str(self.old.root),"failed_files":snapshot(self.old.root),
            "claim_files":snapshot(self.store.store.root/("generation-claim-"+uid)),
            "baseline_files":asdict(ref(self.old.root/"attempt/old_files.json")),"prior_failed_files":{}}
        self.path=root/"recovery.json";self.write_spec()
        self.attempt=ArtifactStore(root/"generation-v3-fixture",new_base=root)
        self.attempt.publish("gpu-start",{"claim.json":b"{}"},{"fixture":True})
    def write_spec(self):
        self.path.write_bytes(json_bytes(self.spec))
        self.recovery=FirstClaimRecovery(ref(self.path))
    @contextmanager
    def plugin(self, *, guarded=True):
        import inspect
        source=support()
        missing=source.root/"prototypes/attribute_image_generator_v2_20261014/condition_schema.py"
        source.config["runtime_sources"][str(missing)]=sha_file(missing)
        with legacy_imports(source,allow_import=True) as old:
            reader=old("irra_image_adapter").pixels_from_pil
            backend=_LegacyG(MockG().eval().requires_grad_(False),old("inference_once"),reader,"cpu")
            backend.execution_kind="cpu_fixture"
            plugin=GV2Plugin(backend=backend,provenance=self.store.plan.g)
            pins={k:asdict(ref(p)) for k,p in {
                "backend":inspect.getsourcefile(_LegacyG),"pixel_reader":inspect.getsourcefile(reader),
                "torchvision":inspect.getsourcefile(transforms.Resize)}.items()}
            observer=GObservation([g.image_id for g in self.store.plan.manifest.gallery],
                image_sources={g.image_id:self.store.image_request(g.image_id).public.content
                               for g in self.store.plan.manifest.gallery},preprocess_sources=pins)
            if guarded:
                with observer.attach(plugin):yield plugin,observer
            else:yield plugin,observer

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-v3-recovery-")
        self.w=World(Path(self.tmp.name));self.load=torch.load
        self.call=torch.nn.Module._call_impl;self.prepare=generation._prepare
    def tearDown(self):
        self.assertIs(torch.load,self.load);self.assertIs(torch.nn.Module._call_impl,self.call)
        self.assertIs(generation._prepare,self.prepare);self.assertFalse(torch.cuda.is_initialized())
        self.tmp.cleanup()
    def test_incomplete_snapshot_descriptor_rejected_before_runtime_preflight(self):
        path=next(iter(self.w.spec["failed_files"]))
        self.w.spec["failed_files"][path].pop("path")
        self.w.path.write_bytes(json_bytes(self.w.spec))
        with self.assertRaisesRegex(ArtifactError,"canonical snapshot"):
            FirstClaimRecovery(ref(self.w.path))
    def test_strict_preflight_accepts_only_exact_zero_forward_first_claim(self):
        report=fresh_images(self.w.store,asdict(ref(self.w.path)))
        self.assertEqual(report["prior_G_forwards"],0)
        with self.assertRaises(ArtifactError):fresh_images(self.w.store)
    def run_stream(self):
        w=self.w;events=[];holders=[];loads=[]
        @contextmanager
        def factory():
            loads.append(True)
            with w.plugin() as (plugin,observer):
                holders.append(observer)
                def before(module,args):
                    key=observer.active;uid=w.store.unit_id(w.store.image_request(key).binding)
                    w.store.store.verify("generation-claim-"+uid)
                    events.append(key)
                hook=plugin.backend.model.register_forward_pre_hook(before)
                try:yield plugin
                finally:hook.remove()
        before_claim=dict(w.spec["claim_files"])
        old=snapshot(w.old.root);progress=[]
        counts={"G_load_attempts":0,"pixel_tensor_loads":0}
        with patch.object(torch.cuda,"init",side_effect=AssertionError("CUDA forbidden")), \
             patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")), \
             no_optimization(),g_load_only(w.store.plan.g.checkpoint.path,w.store.store.root,counts):
            report=stream_with_recovery(w.store,factory,w.recovery,w.attempt,
                resource_check=lambda:None,progress=lambda n,t,s:progress.append((n,s)))
        self.assertEqual(before_claim,snapshot(w.store.store.root/("generation-claim-"+w.spec["unit_id"])))
        self.assertEqual(old,snapshot(w.old.root))
        return report,holders[0],events,progress,loads
    def test_real_resize_tensor_G_hook_recovery_normal_proof_and_image_seal(self):
        report,observer,events,progress,loads=self.run_stream()
        self.assertEqual(events,["g0","g1"])
        self.assertEqual(len(loads),1)
        self.assertEqual(observer.started,2);self.assertEqual(observer.completed,2)
        # 每图原pipeline读取一次、原GV2Plugin再读取一次，都是原既定路径。
        self.assertEqual(observer.preprocess_calls,4)
        self.assertEqual(report["generated"],2);self.assertEqual(report["skipped"],0)
        self.assertEqual(report["recovered_first"],1)
        self.assertEqual(report["legacy_stream_generated"],1)
        self.assertEqual(report["legacy_stream_skipped"],1)
        self.assertEqual(progress,[(1,{"generated":1,"skipped":0}),(2,{"generated":2,"skipped":0})])
        _,_,images,_=collection_units(self.w.store,"image-full")
        audit=verify_images(self.w.store,images,observer.records,{})
        self.assertEqual(audit["count"],2)
        proof=verify_recovery_completion(self.w.store,self.w.recovery,self.w.attempt)
        self.assertTrue(proof["old_claim_preserved"])
        self.assertEqual(proof["new_G_forwards"],1)
    def test_completed_recovery_cannot_be_repeated(self):
        self.run_stream()
        with self.assertRaises(ArtifactError):self.w.recovery.verify(self.w.store)
    def test_other_claim_is_rejected_before_any_factory(self):
        generation._prepare(self.w.store,self.w.store.image_request("g1"))
        with self.assertRaises(ArtifactError):
            stream_with_recovery(self.w.store,lambda:(_ for _ in ()).throw(AssertionError("model")),
                self.w.recovery,self.w.attempt,resource_check=lambda:None,progress=lambda *a:None)
    def test_pending_proof_or_image_prevents_recovery(self):
        for kind in ("pending","proof","image"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(prefix="tbps-framework-v3-reject-") as tmp:
                w=World(Path(tmp))
                if kind=="pending":(w.store.store.root/".pending-fixture").mkdir()
                elif kind=="proof":
                    w.store.store.publish("generation-proof-"+w.spec["unit_id"],{"generation.json":b"{}"},{"fixture":True})
                else:
                    with w.plugin(guarded=False) as (plugin,_):
                        w.store.publish_image(w.req,plugin.backend.load_pixels(w.req.public.content.path),plugin.generate(w.req))
                with self.assertRaises(ArtifactError):w.recovery.verify(w.store)
    def test_failure_report_or_original_claim_tampering_rejected(self):
        for target in ("failure","claim"):
            with self.subTest(target=target),tempfile.TemporaryDirectory(prefix="tbps-framework-v3-tamper-") as tmp:
                w=World(Path(tmp))
                p=w.old.root/"gpu-result/report.json" if target=="failure" else Path(next(iter(w.spec["claim_files"])))
                p.write_bytes(p.read_bytes()+b" ")
                with self.assertRaises(ArtifactError):w.recovery.verify(w.store)
    def test_nonzero_or_incomplete_forward_evidence_rejected_even_with_resealed_fixture(self):
        # 使用新的合法Fixture Manifest验证语义拒绝，而不是仅检查文件SHA。
        for value in (1,False):
            with self.subTest(value=value),tempfile.TemporaryDirectory(prefix="tbps-framework-v3-count-") as tmp:
                w=World(Path(tmp))
                report=dict(w.report,actual_G_forwards={**w.report["actual_G_forwards"],"started":value})
                # 保持真实Fixture文件/Manifest不变，仅在语义读取边界注入坏计数。
                from tbps.materialization import claim_recovery as cr
                original=cr.read_json
                def read(path,*args):
                    if Path(path)==w.old.root/"gpu-result/report.json":return report
                    return original(path,*args)
                with patch.object(cr,"read_json",side_effect=read),self.assertRaises(ArtifactError):
                    w.recovery.verify(w.store)
    def test_wrong_identity_or_plan_rejected(self):
        for field,value in (("image_id","g1"),("unit_id","0"*64),("plan_id","0"*64)):
            with self.subTest(field=field),tempfile.TemporaryDirectory(prefix="tbps-framework-v3-identity-") as tmp:
                w=World(Path(tmp));w.spec[field]=value;w.write_spec()
                with self.assertRaises(ArtifactError):w.recovery.verify(w.store)
    def test_unrelated_store_file_is_not_hidden_by_recovery(self):
        (self.w.store.store.root/"unexpected").write_bytes(b"unknown")
        with self.assertRaisesRegex(ArtifactError,"inventory"):self.w.recovery.verify(self.w.store)
    def test_recovery_scope_does_not_allow_other_claim_or_second_first_call(self):
        w=self.w
        with self.assertRaises(ArtifactError),w.recovery.resumed_prepare(w.store):
            generation._prepare(w.store,w.req);generation._prepare(w.store,w.req)
        with self.assertRaises(ArtifactError):
            generation._prepare(w.store,w.req)
    def test_failed_first_read_retains_old_claim_new_recovery_start_and_no_proof(self):
        w=self.w
        @contextmanager
        def factory():
            with w.plugin() as (plugin,observer):
                with patch.object(plugin.backend,"load_pixels",side_effect=ArtifactError("fixture read failed")):
                    yield plugin
        with self.assertRaisesRegex(ArtifactError,"fixture read failed"):
            stream_with_recovery(w.store,factory,w.recovery,w.attempt,
                resource_check=lambda:None,progress=lambda *a:None)
        self.assertTrue((w.attempt.root/"claim-recovery-start").exists())
        self.assertFalse((w.attempt.root/"claim-recovery-proof").exists())
        self.assertFalse((w.store.store.root/("generation-proof-"+w.spec["unit_id"])).exists())
        with self.assertRaisesRegex(ArtifactError,"consumed"):
            stream_with_recovery(w.store,factory,w.recovery,w.attempt,
                resource_check=lambda:None,progress=lambda *a:None)

class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-v3-preprocess-")
        self.w=World(Path(self.tmp.name))
    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized());self.tmp.cleanup()
    def test_resize_outside_verified_reader_still_forbidden(self):
        with self.w.plugin():
            with self.assertRaisesRegex(ArtifactError,"non-G"):
                transforms.Resize((384,128))(Image.new("RGB",(5,9)))
    def test_other_neural_network_still_forbidden(self):
        with self.w.plugin():
            with self.assertRaisesRegex(ArtifactError,"non-G"):torch.nn.Identity()(torch.ones(1))
    def test_unbound_image_path_refused_before_resize(self):
        with self.w.plugin() as (plugin,observer):
            with self.assertRaisesRegex(ArtifactError,"bound Gallery"):
                plugin.backend.load_pixels(str(Path(self.tmp.name)/"other.png"))
            self.assertEqual(observer.preprocess_calls,0)
    def test_unpinned_reader_or_transform_source_refused(self):
        with self.w.plugin(guarded=False) as (plugin,observer):
            observer.preprocess_sources["torchvision"]["sha256"]="0"*64
            with self.assertRaises(ArtifactError),observer.attach(plugin):pass
    def test_changed_backend_reader_refused(self):
        with self.w.plugin(guarded=False) as (plugin,observer):
            plugin.backend.pixel_reader=lambda image:torch.ones(3,384,128)
            with self.assertRaisesRegex(ArtifactError,"pixels_from_pil"),observer.attach(plugin):pass
    def test_wrong_resize_settings_and_hooks_refused_inside_reader(self):
        native=transforms.Resize.__init__
        for mode in ("size","hook"):
            with self.subTest(mode=mode),self.w.plugin() as (plugin,observer):
                def init(module,*args,**kwargs):
                    native(module,*args,**kwargs)
                    if mode=="size":module.size=(128,384)
                    else:module.register_forward_pre_hook(lambda *a:None)
                with patch.object(transforms.Resize,"__init__",new=init),self.assertRaisesRegex(ArtifactError,"non-G"):
                    plugin.backend.load_pixels(self.w.req.public.content.path)

class Order3074Tests(unittest.TestCase):
    def test_original_stream_3074_order_under_recovery_accounting_without_tensor_IO(self):
        # 仅此调度压力案例Mock像素事务/封存，避免写入5GiB；
        # 两图测试已经独立执行真实预处理、Hook、Claim/Proof和像素读回。
        from types import SimpleNamespace
        from tbps.materialization.contracts import MaterializedUnit
        with tempfile.TemporaryDirectory(prefix="tbps-framework-v3-order3074-") as tmp:
            w=World(Path(tmp),count=3074);memory={};order=[];progress=[]
            @contextmanager
            def factory():
                yield SimpleNamespace(backend=SimpleNamespace(model=torch.nn.Identity().eval().requires_grad_(False),
                                                             load_pixels=lambda p:None))
            protocol_ref=ref(w.store.store.root/"protocol/protocol.json")
            manifest_ref=ref(w.store.store.root/"protocol/manifest.json")
            def unit(key):
                req=w.store.image_request(key)
                return MaterializedUnit(w.store.unit_id(req.binding),w.store.plan.id,req.binding,"image",
                    req.binding.input_sha256,protocol_ref,protocol_ref,manifest_ref)
            def generate(store,key,plugin,reader,**kwargs):
                order.append(key);result=unit(key)
                if key=="g0":
                    _,claim,proof=generation._prepare(store,store.image_request(key))
                    memory[key]=result;generation._finish(store,result,claim,proof)
                else:memory[key]=result
                return result
            with patch.object(w.store,"available",side_effect=lambda b:b.sample_id in memory), \
                 patch.object(w.store,"reference",side_effect=lambda b:memory[b.sample_id]), \
                 patch.object(w.store,"load",return_value=({},{})), \
                 patch.object(generation,"generate_image_once",side_effect=generate), \
                 patch.object(full_runner,"generate_image_once",side_effect=generate), \
                 patch.object(full_runner,"seal_generated_collection",return_value=ref(w.store.store.root/"protocol/protocol.json")):
                result=stream_with_recovery(w.store,factory,w.recovery,w.attempt,
                    resource_check=lambda:None,progress=lambda n,t,s:progress.append((n,s["generated"],s["skipped"])))
            self.assertEqual(order,["g"+str(i) for i in range(3074)])
            self.assertEqual(progress,[(i,i,0) for i in range(1,3075)])
            self.assertEqual(result["generated"],3074)
            self.assertEqual(result["legacy_stream_generated"],3073)
            self.assertEqual(result["legacy_stream_skipped"],1)
            self.assertEqual(result["skipped"],0)
