"""六节点接线CPU回归；不加载受害者、不初始化CUDA、不排名真实Full。"""
from contextlib import nullcontext
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch,Mock
import copy,hashlib,io,json,tempfile,unittest
import torch
from tbps.artifacts import FileRef,ArtifactError,sha_file,json_bytes,digest,read_json
from tbps.ablations import gv2_sweep as s
from tbps.ablations import gv2_sweep_target as t
from tbps.engines.provenance import BoundRepresentation,payload_digest
from tbps.models.base import RepresentationBatch,InputBinding
from test_stage4a_engine import Fixture

class SweepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-sweep-");self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.src=self.base/"source.json";self.src.write_text("{}");self.ref=s.ref(self.src)
        self.parent=asdict(self.ref)
        self.original={"config":{"architecture":s.ARCH,"epochs":10,"optimizer":{"lr":2e-4,"betas":[.5,.999]}}}
        self.ext={"original_training_contract":self.original,"parent_checkpoint":self.parent,
            "original_contract_sha256":digest(self.original),"start_epoch":11,"target_epochs":60,
            "effective_training_configuration":{**self.original["config"],"epochs":60}}
    def cp(self,e):
        contract=self.original if e==10 else self.ext
        return {"schema":"gv2-epoch-checkpoint-1" if e==10 else "gv2-epoch-extension-checkpoint-1",
            "completed_epoch":e,"next_epoch":e+1,"contract":copy.deepcopy(contract),
            "contract_sha256":digest(contract),"generator":{"w":torch.ones(1)}}
    def v(self,e=60,n=1,root="pixels"):
        return s.VariantStore(self.base/root,self.base,{"epoch":e,"checkpoint":self.parent,
            "ordered_gallery":[f"g{i}" for i in range(n)],"no_target_feedback":True})
    def arrays(self):
        x=torch.full((1,3,384,128),.5);adv=x+.01
        return {"original":x,"adversarial":adv,"effective_delta":adv-x}
    def ready(self):
        v=self.v();v.publish_protocol((self.ref,));v.claim(0);return v
    def test_six_fixed_checkpoint_nodes(self):
        for e in s.EPOCHS:self.assertTrue(torch.equal(s.generator_state(self.cp(e),e,self.original,self.parent)["w"],torch.ones(1)))
        with self.assertRaises(ValueError):s.generator_state(self.cp(60),15,self.original,self.parent)
    def test_wrong_epoch_rejected(self):
        d=self.cp(60);d["next_epoch"]=60
        with self.assertRaises(ValueError):s.generator_state(d,60,self.original,self.parent)
    def test_contract_digest_rejected(self):
        d=self.cp(60);d["contract_sha256"]="0"*64
        with self.assertRaises(ValueError):s.generator_state(d,60,self.original,self.parent)
    def test_original_parent_not_silently_reconfigured(self):
        for field,value in (("parent_checkpoint",{"path":"wrong","sha256":"0"*64}),("start_epoch",10),("target_epochs",61)):
            d=self.cp(60);d["contract"][field]=value;d["contract_sha256"]=digest(d["contract"])
            with self.subTest(field=field),self.assertRaises(ValueError):s.generator_state(d,60,self.original,self.parent)
    def test_optimizer_or_architecture_change_refused(self):
        d=self.cp(60);d["contract"]["effective_training_configuration"]["optimizer"]={"lr":1e-3}
        d["contract_sha256"]=digest(d["contract"])
        with self.assertRaises(ValueError):s.generator_state(d,60,self.original,self.parent)
        orig=copy.deepcopy(self.original);orig["config"]["architecture"]["blocks"]=6
        d=self.cp(10);d["contract"]=orig;d["contract_sha256"]=digest(orig)
        with self.assertRaises(ValueError):s.generator_state(d,10,orig,self.parent)
    def test_checkpoint_state_no_optimizer_use(self):
        d=self.cp(60);d["optimizer"]=object();d["rng"]=object()
        self.assertIs(s.generator_state(d,60,self.original,self.parent),d["generator"])
    def test_wrong_schema_rejected(self):
        d=self.cp(60);d["schema"]="gv2-epoch-checkpoint-1"
        with self.assertRaises(ValueError):s.generator_state(d,60,self.original,self.parent)
    def test_protocol_epoch_isolation(self):
        with self.assertRaises(ValueError):self.v(e=11)
        v=self.ready();other=self.v(e=20);self.assertNotEqual(v.protocol,other.protocol)
        with self.assertRaises(ValueError):other.verify()
    def test_duplicate_gallery_identity_rejected(self):
        with self.assertRaises(ValueError):s.VariantStore(self.base/"x",self.base,
            {"epoch":60,"checkpoint":self.parent,"ordered_gallery":["g","g"],"no_target_feedback":True})
    def test_target_feedback_claim_rejected(self):
        with self.assertRaises(ValueError):s.VariantStore(self.base/"x",self.base,
            {"epoch":60,"checkpoint":self.parent,"ordered_gallery":["g"],"no_target_feedback":False})
    def test_publish_requires_pre_forward_claim(self):
        v=self.v();v.publish_protocol((self.ref,))
        with self.assertRaises(ArtifactError):v.publish(0,self.arrays(),self.ref)
    def test_unique_claim_and_immutable_unit(self):
        v=self.ready();v.publish(0,self.arrays(),self.ref)
        with self.assertRaises(ValueError):v.claim(0)
        with self.assertRaises(ArtifactError):v.publish(0,self.arrays(),self.ref)
    def test_pixel_readback_budget_and_exact_values(self):
        v=self.ready();arrays=self.arrays();v.publish(0,arrays,self.ref);read,md=v.load(0)
        self.assertLessEqual(md["pixels"]["linf"],8/255)
        self.assertTrue(all(torch.equal(arrays[k],read[k]) for k in arrays))
    def test_pixel_over_budget_refused(self):
        v=self.ready();a=self.arrays();a["adversarial"]=a["original"]+.04;a["effective_delta"]=a["adversarial"]-a["original"]
        with self.assertRaises(ValueError):v.publish(0,a,self.ref)
    def test_effective_delta_not_invented(self):
        v=self.ready();a=self.arrays();a["effective_delta"].zero_()
        with self.assertRaises(ValueError):v.publish(0,a,self.ref)
    def test_fp32_range_shape_finite(self):
        for bad in (torch.ones((1,3,384,128))*float("nan"),torch.ones((1,3,384,128))*2,
                    torch.ones((1,3,384,128),dtype=torch.float64),torch.ones(1,512)):
            v=self.v(root=str(id(bad)));v.publish_protocol((self.ref,));v.claim(0);a=self.arrays();a["adversarial"]=bad
            with self.subTest(shape=bad.shape),self.assertRaises((ValueError,RuntimeError)):v.publish(0,a,self.ref)
    def test_storage_compact_real_clone(self):
        v=self.ready();a=self.arrays();big=torch.stack([a["adversarial"]]*5)
        a["adversarial"]=big[2];v.publish(0,a,self.ref);read,_=v.load(0)
        self.assertEqual(read["adversarial"].untyped_storage().nbytes(),read["adversarial"].numel()*4)
        self.assertLess((v.store.root/v.key(0)/"pixels.pt").stat().st_size,1800000)
    def test_source_tamper_rejected(self):
        v=self.ready();v.publish(0,self.arrays(),self.ref);self.src.write_text("changed")
        with self.assertRaises(ArtifactError):v.load(0)
    def test_payload_tamper_rejected(self):
        v=self.ready();v.publish(0,self.arrays(),self.ref);(v.store.root/v.key(0)/"pixels.pt").write_bytes(b"bad")
        with self.assertRaises(ArtifactError):v.load(0)
    def test_fault_keeps_claim_and_pending(self):
        v=self.ready()
        def fail(step):
            if step=="before_publish":raise RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):v.publish(0,self.arrays(),self.ref,fault=fail)
        self.assertTrue((v.store.root/"claim-000000").exists());self.assertTrue(v.store.pending())
        with self.assertRaises(ValueError):v.verify()
        with self.assertRaises(ValueError):v.claim(0)
    def test_out_of_order_index_rejected(self):
        v=self.v()
        for index in (-1,1,True):
            with self.subTest(index=index),self.assertRaises(ValueError):v.claim(index)
    def test_public_csv_four_metrics_no_crlf(self):
        groups={m:{k:23.456789 for k in s.KEYS} for m in t.METHODS}
        raw=s.csv_metrics(groups);self.assertNotIn(b"\r",raw);self.assertNotIn(b"DR",raw);self.assertNotIn(b"mINP",raw)
        self.assertEqual(raw.decode().splitlines()[1].split(",")[0],t.METHODS[0])
        self.assertIn(b"23.46",s.csv_metrics(groups,True))
    def test_dataset_subset_preserves_official_order(self):
        f=Fixture(self.base);m=f.manifest
        # Fixture四Query不足10，必须拒绝，不自动补样本/改变规模。
        with self.assertRaises(ValueError):s.select(m,{"smoke":{"gallery_ids":[g.image_id for g in m.gallery[:10]],"query_ids":[q.query_id for q in m.queries]}},"smoke")
        self.assertIs(s.select(m,{},"full"),m)
    def test_query_text_origin_not_gallery_hash(self):
        f=Fixture(self.base);ctx={"manifest":f.manifest,"plan":NS(h=NS(checkpoint=f.ref,sources_sha256=digest("h"))),"core":f.ref}
        reference=Mock();reference.read.side_effect=[f.inputs["clean_text"],f.inputs["adv_text"]]
        clean,adv=t.texts(reference,ctx)
        self.assertEqual(adv.original_sha256,tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in f.manifest.queries))
        adv.verify();self.assertEqual(len(adv.original_sha256),4);self.assertEqual(clean,f.inputs["clean_text"])
    def test_two_mode_native_engine_same_protocol(self):
        for fusion in (False,True):
            f=Fixture(self.base,fusion=fusion);inputs={k:f.inputs[k] for k in ("clean_text","adv_text","adv_gallery")}
            _,result,scores=f.engine().run(inputs,modes=("Image-only","Joint"),query_chunk=128)
            self.assertEqual(set(result["groups"]),{"Image-only","Joint"})
            self.assertEqual(scores["Joint"].shape,(4,12))
    def test_forward_claim_control_flow_cpu_mock(self):
        events=[];self.src.write_bytes(json_bytes(self.original));rr=s.ref(self.src)
        checkpoint=self.base/"checkpoint.pt";checkpoint.write_bytes(b"fixture")
        cfg=self.base/"fixture-config.json";cfg.write_text("{}")
        core=self.base/"core.json";core.write_text("{}")
        point=s.ref(checkpoint);core_ref=s.ref(core)
        self.ref=point;self.parent=asdict(point);self.ext["parent_checkpoint"]=self.parent
        v=self.v(n=2)
        rows=[NS(image_id=f"g{i}") for i in range(2)];arrays=self.arrays()
        ctx={"manifest":NS(gallery=rows),"legacy":object(),"core":core_ref,
            "images":{x.image_id:NS(manifest=core_ref) for x in rows},"store":NS(load=lambda u:(arrays,{})),
            "gate":NS(initialize_cuda=lambda:events.append("CUDA_init")),"plan":NS(h=NS(checkpoint=rr))}
        c={"sources":{"original_contract":asdict(rr)},"points":{"10":asdict(point),"60":asdict(point)},"path":str(cfg)}
        class G(torch.nn.Module):
            def __init__(self,**kw):super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
            def to(self,*a,**k):return self
            def forward(self,x):events.append("G_forward");return x*0+self.w*.01
        def once(g,x):
            delta=g(x);adv=x+delta
            return {"adversarial":adv,"effective_delta":adv-x}
        def old(name):return NS(ImageOnlyGenerator=G) if name=="image_only_generator" else NS(generate_once=once)
        real_to=torch.Tensor.to
        def cpu_to(x,*a,**k):
            if a and a[0]=="cuda:0":return x
            return real_to(x,*a,**k)
        claim=v.claim
        def recorded(i):events.append("claim");return claim(i)
        with patch.object(s,"prepare",return_value=ctx),patch.object(s,"variant",return_value=v),patch.object(s,"resources",return_value={}),\
             patch.object(s,"legacy_imports",return_value=nullcontext(old)),patch.object(s,"BulkSession",return_value=nullcontext()),\
             patch.object(torch,"load",side_effect=lambda *a,**k:self.cp(60) if str(a[0])==point.path else real_load(*a,**k)),\
             patch.object(torch.Tensor,"to",cpu_to),patch.object(torch.cuda,"max_memory_allocated",return_value=0),\
             patch.object(torch.cuda,"max_memory_reserved",return_value=0),patch.object(v,"claim",side_effect=recorded),\
             patch("tbps.models.irra.fp32_runtime",return_value=nullcontext()),patch.object(torch.optim,"Adam",side_effect=AssertionError("optimizer forbidden")):
            s.generate(c,"smoke",60)
        self.assertEqual(events,["CUDA_init","claim","G_forward","claim","G_forward"])
        report=read_json(v.store.root/"execution/report.json");self.assertEqual(report["counts"]["G_forwards"],2)
        self.assertEqual(report["status"],"PASS");self.assertEqual(report["model_before"],report["model_after"])
    def test_unfinished_claim_stops_before_model_load(self):
        v=self.ready();ctx={"c":{}}
        with patch.object(s,"prepare",return_value=ctx),patch.object(s,"variant",return_value=v),\
             patch.object(s,"resources",side_effect=AssertionError("must stop before resources/CUDA")):
            with self.assertRaisesRegex(ValueError,"unfinished Claim"):s.generate({},"smoke",60,resume=True)
    def test_full_epoch10_generation_forbidden(self):
        with self.assertRaisesRegex(ValueError,"Epoch10"):s.generate({},"full",10)
    def test_configuration_fixed_epochs_order_and_sha(self):
        root=Path(__file__).resolve().parents[2]
        c=json.loads((root/"configs/framework/gv2_epoch_sweep.server.json").read_text())
        p=self.base/"config.json"
        c["points"]={k:c["points"][k] for k in reversed(c["points"])};p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,"six fixed"):s.config(p)
        c=json.loads((root/"configs/framework/gv2_epoch_sweep.server.json").read_text())
        c["points"]["60"]["sha256"]="0"*64;p.write_text(json.dumps(c))
        with self.assertRaises(ArtifactError):s.config(p)
    def test_missing_runtime_producer_rejected(self):
        root=Path(__file__).resolve().parents[2]
        c=json.loads((root/"configs/framework/gv2_epoch_sweep.server.json").read_text())
        c["producers"].pop(str(root/"src/tbps/gv2_epoch_eval.py"));p=self.base/"config.json";p.write_text(json.dumps(c))
        with self.assertRaisesRegex(ValueError,"complete fixed"):s.config(p)
    def resource_c(self):
        return {"minimum_free_mib":6144,"gpu_policy":"shared_observe_only",
            "minimum_ram_bytes":32*2**30,"minimum_disk_bytes":{"smoke":10*2**30,"full":100*2**30}}
    def resource_fixture(self,*,free=18433,util=74,ram_kib=64*2**20,disk_bytes=200*2**30):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch("tbps.engines.materialized_irra.gpu_available",return_value={
            "free_mib":free,"utilization_percent":util,"compute_processes":"1402779, python, 5610"}))
        stack.enter_context(patch.object(s.Path,"read_text",return_value=f"MemAvailable: {ram_kib} kB\n"))
        stack.enter_context(patch.object(s.os,"statvfs",return_value=NS(f_bavail=disk_bytes,f_frsize=1,f_favail=10000)))
        return stack
    def test_shared_gpu_74percent_18433mib_allowed(self):
        with self.resource_fixture():
            report=s.resources(self.resource_c())
        self.assertEqual(report["GPU"]["utilization_percent"],74)
        self.assertIn("1402779",report["GPU"]["compute_processes"])
    def test_shared_gpu_90percent_allowed(self):
        with self.resource_fixture(util=90):
            self.assertEqual(s.resources(self.resource_c())["GPU"]["utilization_percent"],90)
    def test_shared_gpu_exact_6144_allowed(self):
        with self.resource_fixture(free=6144,util=100):
            self.assertEqual(s.resources(self.resource_c())["GPU"]["free_mib"],6144)
    def test_shared_gpu_below_6144_rejected(self):
        with self.resource_fixture(free=6143),self.assertRaisesRegex(ValueError,"memory insufficient"):
            s.resources(self.resource_c())
    def test_shared_gpu_ram_insufficient_rejected(self):
        with self.resource_fixture(ram_kib=1024),self.assertRaisesRegex(ValueError,"RAM/disk"):
            s.resources(self.resource_c())
    def test_shared_gpu_disk_insufficient_rejected(self):
        with self.resource_fixture(disk_bytes=1024),self.assertRaisesRegex(ValueError,"RAM/disk"):
            s.resources(self.resource_c())
    def test_gpu_status_failure_stops_before_attempt(self):
        c=self.resource_c();c["output_base"]=str(self.base/"output")
        with patch.object(t,"prepare",return_value={}),\
             patch("tbps.engines.materialized_irra.gpu_available",side_effect=RuntimeError("nvidia-smi failed")):
            with self.assertRaisesRegex(RuntimeError,"nvidia-smi failed"):t.smoke(c)
        self.assertFalse((self.base/"output").exists())
    def test_resource_check_never_interferes_or_initializes_cuda(self):
        with self.resource_fixture(),patch.object(s.os,"kill",side_effect=AssertionError("no signal")),\
             patch.object(s.os,"killpg",side_effect=AssertionError("no process-group signal")),\
             patch.object(s.subprocess,"run",side_effect=AssertionError("no process control")),\
             patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("no CUDA")):
            s.resources(self.resource_c())
        self.assertFalse(torch.cuda.is_initialized())
    def test_shared_policy_required(self):
        c=self.resource_c();c["gpu_policy"]="exclusive"
        with self.resource_fixture(),self.assertRaisesRegex(ValueError,"shared GPU"):
            s.resources(c)
    def test_full_generation_requires_explicit_flag(self):
        with patch("sys.argv",["entry","--config","unused","--action","generate","--scope","full","--epoch","20"]),\
             patch.object(s,"config",return_value={}),patch.object(s,"generate",side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError,"authorization"):s.main()
    def test_full_target_requires_explicit_flag(self):
        with patch("sys.argv",["entry","--config","unused","--action","target","--scope","full","--epoch","20","--model","rasa"]),\
             patch.object(s,"config",return_value={}),patch.object(t,"target",side_effect=AssertionError("must not run")):
            with self.assertRaisesRegex(ValueError,"authorization"):s.main()
    def test_independent_audit_reads_result_without_scoring(self):
        from tbps.artifacts import ArtifactStore
        from tbps.engines.evaluation import publish_result,EvaluationEngine
        f=Fixture(self.base)
        inputs={k:f.inputs[k] for k in ("clean_text","adv_text","adv_gallery")}
        protocol,result,scores=f.engine().run(inputs,modes=("Image-only","Joint"),query_chunk=128)
        eroot=self.base/"epoch_0060";store=ArtifactStore(eroot/"irra",new_base=self.base)
        publish_result(store,"result",protocol,result,scores)
        groups={t.METHODS[i]:{k:result["groups"][m][k] for k in s.KEYS} for i,m in enumerate(("Image-only","Joint"))}
        store.publish("completed",{"summary.json":json_bytes({"groups":groups,"gallery_count":12,"query_count":4}),
            "metrics.csv":s.csv_metrics(groups),"metrics_2dp.csv":s.csv_metrics(groups,True)},{})
        execution={"status":"PASS","model_before":"same","model_after":"same","counts":{"text_samples":0},
            **{k:0 for k in ("G_H_loads","G_H_forwards","backward","optimizer","JVP")}}
        store.publish("execution",{"report.json":json_bytes(execution)},{})
        with patch.object(t,"prepare",return_value={}),patch.object(t,"paths",return_value=(eroot,self.base)),\
             patch.object(t,"audit_variant",return_value={}),patch.object(EvaluationEngine,"run",side_effect=AssertionError("no re-scoring")):
            t.audit({},"smoke",60,"irra")
        self.assertEqual(read_json(store.root/"cpu_audit/report.json")["status"],"PASS")
    def test_source_and_module_import_has_no_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())

real_load=torch.load
if __name__=="__main__":unittest.main()
