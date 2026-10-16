"""正式调度CPU测试：合成特征/模拟编码器，绝不加载真实G/IRRA。"""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from test_phase2 import fixture, GMock
from test_phase3 import EncoderFixture
from training_inputs import TrainingCatalog
from phase2_sources import digest, read_json, sha
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from constraints import candidates
from real_feedback import InferenceRuntime
from phase3_inputs import import_gv2
from phase4_training import train_step, optimizer_for
from feedback_backend import state_sha
from .common import immutable_json, run_lock, Stop, safe_root, BASE
from .data import qualify_caption, example_from, summarize, members
from .storage import GalleryStore
from .runtime import RewardBank, FormalRuntime
from . import checkpoint as ck
from . import training, validation, cli

class Fixture:
    def __init__(self,path,val_count=12):
        original,_,_,_=fixture()
        vals=tuple(replace(original[3],row_id=f"cuhk:{20+i}:0",image_rel_path=f"v{i}.jpg",pid=50+i,
                           caption="A woman wears a blue coat.") for i in range(val_count))
        self.samples=original[:3]+vals
        self.catalog=TrainingCatalog(self.samples,{"fixture":"CPU formal, not real"},Path("."),
                          {"counts":{"train":3,"validation":val_count}})
        self.probe,self.frontend=official_probe(),CaptionFrontend()
        self.root=Path(path)
        self.stores={}
        rng=torch.Generator().manual_seed(14)
        for role in ("train","validation"):
            ss=members(self.catalog,role)
            p={"schema":"hv1-formal-gallery-1","role":role,"catalog_sha256":self.catalog.fingerprint,
                "execution_backend":"cpu_test_fixture","identities":[asdict(s) for s in ss],"source_sha256":{}}
            store=GalleryStore(self.root/role,self.catalog,p,True)
            for s in ss:
                data={k:F.normalize(torch.randn(1,512,generator=rng),dim=1) for k in ("clean","adversarial")}
                store.prepare(s,lambda data=data:(data,{"linf":.01,"execution":{"G":1,"clean":1,"adv":1,"text":0}}))
                if role=="validation":
                    text=F.normalize(torch.randn(1,512,generator=rng),dim=1)
                    store.prepare(s,lambda text=text:({"text":text},{"execution":{"G":0,"clean":0,"adv":0,"text":1}}),"original_text")
            self.stores[role]=store
        self.ledger={}
        for role in ("train","validation"):
            self.ledger[role]={"entries":[dict(qualify_caption(s.caption,self.frontend,self.probe),identity=asdict(s))
                                         for s in members(self.catalog,role)]}
        self.cfg=training.config(2,2)
        self.cfg["architecture"]={"width":16,"layers":1,"heads":4,"max_bytes":1024}
        self.protocol=training.protocol_for(self.catalog,self.stores["train"],self.ledger["train"],self.cfg)
        self.bank=RewardBank(self.catalog,self.stores["train"],self.frontend)
    def runtime(self,total=99):
        inference,_=import_gv2()
        return FormalRuntime(GMock(),EncoderFixture(),inference.generate_once,self.probe,"cpu",
                      {"G":0,"clean":0,"adv":0,"text":total},"cpu_test_fixture")
    def run(self,**kwargs):
        runtime=self.runtime()
        try:
            return training.execute(self.root/"run",self.protocol,self.catalog,self.ledger["train"],
                    self.stores["train"],self.probe,self.bank,runtime,SimpleNamespace(requested=False),**kwargs)
        finally:runtime.close()

class QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.frontend=CaptionFrontend();cls.probe=official_probe()
    def check(self,text):return qualify_caption(text,self.frontend,self.probe)
    def test_real_caption_has_exact_spans(self):
        e=self.check("A woman wears a black coat and black pants.")
        self.assertTrue(e["eligible"])
        ex=example_from(e)
        for a in ex.attributes:self.assertEqual(ex.caption[a.char_start:a.char_end],a.raw_text)
    def test_no_attrs_fixed_noop(self):
        e=self.check("xyzzy")
        self.assertFalse(e["eligible"]);self.assertTrue(e["structural_K0"])
        self.assertIn("no_canonical_attributes",e["reasons"])
    def test_original_truncation_not_filtered(self):
        e=self.check(("A man wears black pants. "*40))
        self.assertFalse(e["eligible"]);self.assertTrue(e["original_truncated"])
        self.assertIn("original_BPE_truncated",e["reasons"])
    def test_oversize_records_reason(self):
        e=self.check("A man "+("x"*1200))
        self.assertIn("H_bytes_over_1024",e["reasons"])
    def test_repeated_attribute_spans(self):
        e=self.check("A man wears a black shirt and black pants.")
        spans=[a for a in e["attributes"] if a["raw_text"]=="black"]
        self.assertGreaterEqual(len(spans),2)
        self.assertGreater(len({a["char_start"] for a in spans}),1)
    def test_caption_only_frontend(self):
        with self.assertRaises(TypeError):self.frontend("A man",pid=1)
    def test_summary_keeps_all_entries(self):
        rows=[self.check("xyzzy"),self.check("A man in a blue coat.")]
        s=summarize(rows)
        self.assertEqual(s["samples"],2);self.assertEqual(s["eligible"],1);self.assertEqual(s["structural_K0"],1)

    def test_qualification_content_receipt(self):
        from . import data
        _,_,_,catalog=fixture()
        entries=[dict(self.check(s.caption),identity=asdict(s)) for s in catalog.train]
        protocol={"cpu":"qualification fixture"}
        value={"protocol":protocol,"protocol_sha256":digest(protocol),"entries":entries,
               "summary":summarize(entries)}
        with tempfile.TemporaryDirectory() as path:
            root=Path(path);p=root/"qualification"/"train.json";immutable_json(p,value)
            immutable_json(root/"qualification"/"completed.json",{"train_sha256":sha(p)})
            with patch.object(data,"base_protocol",return_value=protocol):
                data.read_qualification(root,catalog,"train")
                changed=read_json(p);changed["entries"][0]["eligible"]=False
                p.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):data.read_qualification(root,catalog,"train")

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name);self.store=self.f.stores["train"];self.s=self.store.samples[0]
    def test_train_validation_isolation(self):
        with self.assertRaises(ValueError):self.store.load(self.f.catalog.validation[0])
        with self.assertRaises(ValueError):RewardBank(self.f.catalog,self.f.stores["validation"],self.f.frontend)
    def test_complete_audit(self):
        self.assertEqual(self.store.audit()["image"],3)
        self.assertEqual(self.f.stores["validation"].audit()["original_text"],12)
    def test_reuse_no_forward(self):
        with patch.object(self.f,"runtime",side_effect=AssertionError("GPU forbidden")):
            self.store.prepare(self.s,lambda:(_ for _ in ()).throw(AssertionError("no recomputation")))
    def test_feature_corruption_rejected(self):
        (self.store.path(self.s)/"features.pt").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):self.store.load(self.s)
    def test_receipt_corruption_rejected(self):
        p=self.store.path(self.s)/"receipt.json";value=read_json(p);value["features_sha256"]="f"*64
        p.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.store.load(self.s)
    def test_identity_mutation_rejected(self):
        with self.assertRaises(ValueError):self.store.load(replace(self.s,pid=987))
    def test_pending_blocks_audit(self):
        (self.store.root/"image"/"unknown.pending").mkdir()
        with self.assertRaises(ValueError):self.store.audit()
    def test_partial_audit_explicit(self):
        fresh=GalleryStore(Path(self.tmp.name)/"empty",self.f.catalog,self.store.protocol,True)
        self.assertEqual(fresh.audit(False)["image"],0)
        with self.assertRaises(ValueError):fresh.audit()
    def test_compact_shared_storage(self):
        fresh=GalleryStore(Path(self.tmp.name)/"compact",self.f.catalog,self.store.protocol,True)
        big=F.normalize(torch.randn(500,512),dim=1)
        fresh.prepare(self.s,lambda:({"clean":big[:1],"adversarial":big[1:2]},
                                     {"linf":0.,"execution":{"G":1,"clean":1,"adv":1,"text":0}}))
        data,_=fresh.load(self.s)
        self.assertTrue(torch.equal(data["clean"],big[:1]))
        self.assertEqual(data["clean"].untyped_storage().nbytes(),512*4)
        self.assertLess((fresh.path(self.s)/"features.pt").stat().st_size,10000)
    def test_pending_never_repeats_G(self):
        fresh=GalleryStore(Path(self.tmp.name)/"pending",self.f.catalog,self.store.protocol,True)
        unit=fresh.path(self.s);unit.with_name(unit.name+".pending").mkdir(parents=True)
        with self.assertRaises(FileExistsError):fresh.prepare(self.s,lambda:(_ for _ in ()).throw(AssertionError("G")))
    def test_incorrect_budget_unpublished(self):
        fresh=GalleryStore(Path(self.tmp.name)/"badbudget",self.f.catalog,self.store.protocol,True)
        data=self.store.load(self.s)[0]
        with self.assertRaises(ValueError):
            fresh.prepare(self.s,lambda:(data,{"linf":.04,"execution":{"G":1,"clean":1,"adv":1,"text":0}}))
        self.assertFalse(fresh.path(self.s).exists())

    def test_gallery_completion_tampering_rejected(self):
        immutable_json(self.store.root/"completed.json",self.store.audit())
        (self.store.root/"completed.json").write_text('{"image":0}')
        with self.assertRaises(ValueError):self.store.audit()

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)
    def test_order_deterministic_and_excludes_val(self):
        ss=self.f.catalog.train
        a=training.order(ss,3)
        self.assertEqual(a,training.order(ss,3))
        self.assertEqual(len(a),9)
        self.assertFalse(set(a)&{s.row_id for s in self.f.catalog.validation})
    def test_periodic_not_per_step_full_checkpoints(self):
        result=self.f.run()
        payload,audit=ck.audit(self.f.root/"run",self.f.protocol)
        self.assertEqual(result["committed_updates"],6)
        self.assertEqual([x["cursor"] for x in audit["published"]],[0,2,3,5,6])
        self.assertEqual(payload["cursor"],6)
        self.assertTrue(result["restored_H_Adam_RNG_exact"])
    def test_boundary_resume_equals_uninterrupted(self):
        self.f.run()
        expected,_=ck.audit(self.f.root/"run",self.f.protocol)
        with tempfile.TemporaryDirectory() as path:
            other=Fixture(path);other.run(max_visits=3);other.run(resume=True)
            actual,_=ck.audit(other.root/"run",other.protocol)
            for key in ("policy","optimizer","rng"):self.assertTrue(ck.same(expected[key],actual[key]),key)
    def test_unknown_pending_refused_before_training(self):
        self.f.run(max_visits=2)
        ck.begin(self.f.root/"run",self.f.protocol,"wrong")
        with self.assertRaises(ValueError):self.f.run(resume=True,rollback_uncommitted=True)
    def test_uncommitted_tail_requires_explicit_ack(self):
        self.f.run(max_visits=2)
        payload,audit=ck.audit(self.f.root/"run",self.f.protocol)
        work=ck.begin(self.f.root/"run",self.f.protocol,audit["published"][-1]["sha256"])
        ck.append(work,{"tentative":"no committed count"})
        with self.assertRaises(ValueError):self.f.run(resume=True)
        self.f.run(resume=True,rollback_uncommitted=True)
        _,result=ck.audit(self.f.root/"run",self.f.protocol)
        self.assertEqual(result["committed_updates"],6)
        archives=list((self.f.root/"run"/"abandoned").glob("*/rollback.json"))
        self.assertEqual(len(archives),1)
    def test_partial_tail_preserved_on_rollback(self):
        self.f.run(max_visits=2)
        _,a=ck.audit(self.f.root/"run",self.f.protocol)
        work=ck.begin(self.f.root/"run",self.f.protocol,a["published"][-1]["sha256"])
        (work/"steps.jsonl").write_bytes(b'{"partial":')
        self.f.run(resume=True,rollback_uncommitted=True)
        files=list((self.f.root/"run"/"abandoned").glob("*/steps.jsonl"))
        self.assertEqual(files[0].read_bytes(),b'{"partial":')
    def test_log_tampering_rejected(self):
        self.f.run(max_visits=2)
        p=self.f.root/"run"/"checkpoints"/"visit_00000002"/"steps.jsonl"
        p.write_bytes(p.read_bytes()+b"\n")
        with self.assertRaises(ValueError):ck.audit(self.f.root/"run",self.f.protocol)
    def test_checkpoint_corruption_rejected(self):
        self.f.run(max_visits=2)
        (self.f.root/"run"/"checkpoints"/"visit_00000002"/"checkpoint.pt").write_bytes(b"bad")
        with self.assertRaises(ValueError):ck.audit(self.f.root/"run",self.f.protocol)
    def test_config_mismatch_rejected(self):
        self.f.run(max_visits=2)
        changed=copy.deepcopy(self.f.protocol);changed["config"]["optimizer"]["lr"]=.002
        with self.assertRaises(ValueError):ck.audit(self.f.root/"run",changed)
    def test_H_fresh_seed_no32_checkpoint(self):
        h=training.fresh_policy(self.f.probe,self.f.cfg)
        h2=training.fresh_policy(self.f.probe,self.f.cfg)
        self.assertEqual(state_sha(h),state_sha(h2))
        with patch.object(torch,"load",side_effect=AssertionError("must not load old weights")):
            h3=training.fresh_policy(self.f.probe,self.f.cfg)
        self.assertEqual(state_sha(h3),state_sha(h))
    def test_runtime_frozen_and_H_only_updates(self):
        r=self.f.run(max_visits=2)
        self.assertTrue(r["runtime"]["parameters_buffers_unchanged"])
        self.assertEqual(r["runtime"]["samples"]["G"],0)
        self.assertEqual(r["runtime"]["samples"]["clean"],0)
        self.assertEqual(r["runtime"]["samples"]["text"],6)
    def test_ineligible_kept_without_optimizer(self):
        first=self.f.ledger["train"]["entries"][0]
        first["eligible"]=False;first["reasons"]=["fixture_no_legal_edit"]
        self.f.protocol=training.protocol_for(self.f.catalog,self.f.stores["train"],self.f.ledger["train"],self.f.cfg)
        r=self.f.run()
        self.assertEqual(r["committed_visits"],6);self.assertEqual(r["committed_updates"],4)
    def test_val_query_cannot_get_reward(self):
        runtime=self.f.runtime()
        self.addCleanup(runtime.close)
        s=self.f.catalog.validation[0]
        with self.assertRaises(ValueError):
            self.f.bank.bind(s,runtime,self.f.frontend(s.caption),1)
    def test_changed_query_example_rejected(self):
        runtime=self.f.runtime();self.addCleanup(runtime.close)
        with self.assertRaises(ValueError):
            self.f.bank.bind(self.f.catalog.train[0],runtime,self.f.frontend("A woman in red."),1)
    def test_safe_stop_commits_current_boundary(self):
        r=self.f.run(max_visits=1)
        self.assertEqual(r["committed_visits"],1)
        self.assertFalse((self.f.root/"run"/"pending").exists())
    def test_duplicate_start_refused(self):
        self.f.run(max_visits=1)
        with self.assertRaises(ValueError):self.f.run()
    def test_zero_visit_stop_initial_checkpoint_valid(self):
        runtime=self.f.runtime();self.addCleanup(runtime.close)
        r=training.execute(self.f.root/"run",self.f.protocol,self.f.catalog,self.f.ledger["train"],
            self.f.stores["train"],self.f.probe,self.f.bank,runtime,SimpleNamespace(requested=True))
        self.assertEqual(r["committed_visits"],0)

    def test_executed_unlogged_tail_replay_exact(self):
        with tempfile.TemporaryDirectory() as path:
            clean=Fixture(path);clean.run()
            expected,_=ck.audit(clean.root/"run",clean.protocol)
        original=ck.append
        def interrupt(work,record):
            if record["visit"]==3: raise RuntimeError("CPU simulated hard interruption after optimizer.step")
            return original(work,record)
        with patch.object(ck,"append",side_effect=interrupt),self.assertRaises(RuntimeError):
            self.f.run()
        _,a=ck.audit(self.f.root/"run",self.f.protocol)
        self.assertEqual(a["committed_updates"],2)
        self.f.run(resume=True,rollback_uncommitted=True)
        actual,_=ck.audit(self.f.root/"run",self.f.protocol)
        for key in ("policy","optimizer","rng"):self.assertTrue(ck.same(expected[key],actual[key]),key)
    def test_cpu_audit_readonly(self):
        self.f.run(max_visits=2)
        files=[p for p in (self.f.root/"run").rglob("*") if p.is_file()]
        before={str(p):(sha(p),p.stat().st_mtime_ns) for p in files}
        ck.audit(self.f.root/"run",self.f.protocol)
        self.assertEqual(before,{str(p):(sha(p),p.stat().st_mtime_ns) for p in files})
    def test_log_statistics_keeps_components_separate(self):
        self.f.run()
        stats=training.log_statistics(self.f.root/"run",self.f.protocol)
        self.assertEqual(stats["1"]["updates"],3)
        for key in ("sampled_R_sim","sampled_R_rank","sampled_synergy_increment"):
            self.assertIn(key,stats["1"]["scalars"])
    def test_foreign_source_refuses_before_step(self):
        self.f.protocol["source_sha256"][__file__]="f"*64
        with self.assertRaises(ValueError):self.f.run()

    def test_long_runtime_does_not_retain_forward_tensor_arguments(self):
        import gc,weakref
        inference,_=import_gv2()
        encoder=EncoderFixture()
        refs=[];original=encoder.model.encode_image
        def capture(x):
            refs.append(weakref.ref(x))
            return original(x)
        encoder.model.encode_image=capture
        runtime=FormalRuntime(GMock(),encoder,inference.generate_once,self.f.probe,"cpu",
                    {"G":2,"clean":2,"adv":2,"text":6},"cpu_test_fixture")
        self.addCleanup(runtime.close)
        for _ in range(2):runtime.image_pair(torch.full((1,3,8,4),.5))
        runtime.encode_text(("A man in black.","A woman in blue.","A red coat."))
        gc.collect()
        self.assertEqual(len(refs),4)
        self.assertTrue(all(r() is None for r in refs))
        self.assertFalse(hasattr(runtime.encoder.model.encode_image,"call_args_list"))
        self.assertFalse(hasattr(runtime.encoder.model.encode_text,"call_args_list"))
        self.assertEqual(runtime.samples,{"G":2,"clean":2,"adv":2,"text":3})
        runtime.finish()
    def test_plain_hooks_removed_on_finish(self):
        runtime=self.f.runtime()
        runtime.finish()
        self.assertFalse(runtime.generator._forward_pre_hooks)
        self.assertTrue(hasattr(runtime.encoder.model.encode_text,"__self__"))

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)
    def test_metrics_official_multi_positive(self):
        x=torch.eye(512)[:12];ids=torch.tensor([1,1,2,3,4,5,6,7,8,9,10,11])
        x[1]=F.normalize(x[0]+x[1],dim=0)
        x[2]=F.normalize(x[0]+.1*x[2],dim=0)
        r=validation.metrics(torch.eye(512)[:1],x,torch.tensor([1]),ids)
        self.assertEqual(r["R@1"],100.)
        self.assertAlmostEqual(r["mAP"],(1.+2/3)/2*100,places=4)
    def test_four_groups_identical_if_no_edits(self):
        x=F.normalize(torch.randn(12,512),dim=1);ids=torch.arange(12)
        r=validation.four_groups(x,x,{"clean":x,"adversarial":x},ids)
        self.assertTrue(all(m["mAP"]==100. for m in r.values()))
    def test_missing_positive_rejected(self):
        x=F.normalize(torch.randn(12,512),dim=1)
        with self.assertRaises(ValueError):validation.metrics(x,x,torch.arange(12)+50,torch.arange(12))
    def test_best_joint_and_earliest_tie(self):
        def row(sha_,cursor,score):return {"checkpoint_sha256":sha_,"cursor":cursor,
            "groups":{"Joint":{"mAP":score},"Image-only":{"mAP":40.}},
            "Joint_minus_ImageOnly_mAP_pp":score-40.}
        best=validation.select_best([row("a",5,30),row("b",2,30),row("c",1,31)])
        self.assertEqual(best["selected_checkpoint_sha256"],"b")
        self.assertFalse(best["uses_official_Test"])
    def test_full_validation_noop_keeps_all(self):
        # 显式全部NOOP fixture，仅验证Validation全部Query保留，无GPU。
        self.f.run(max_visits=3)
        payload,audit=ck.audit(self.f.root/"run",self.f.protocol)
        ledger=copy.deepcopy(self.f.ledger["validation"])
        for e in ledger["entries"]:e["eligible"]=False;e["reasons"]=["fixture_NOOP"]
        policy=training.fresh_policy(self.f.probe,self.f.cfg);policy.load_state_dict(payload["policy"])
        runtime=self.f.runtime();self.addCleanup(runtime.close)
        result=validation.evaluate(self.f.root/"val",self.f.protocol,payload,
            audit["published"][-1]["sha256"],self.f.stores["validation"],ledger,policy,runtime,self.f.probe,
            SimpleNamespace(requested=False))
        self.assertEqual(result["queries"],12);self.assertTrue(result["all_queries_retained"])
        self.assertEqual(result["groups"]["Image-only"],result["groups"]["Joint"])
        self.assertEqual(runtime.samples["text"],0)
        validation.audit_result(self.f.root/"val"/audit["published"][-1]["sha256"],
            self.f.stores["validation"],ledger,self.f.protocol)
    def test_validation_text_inference_and_frozen_H(self):
        self.f.run(max_visits=1)
        payload,audit=ck.audit(self.f.root/"run",self.f.protocol)
        policy=training.fresh_policy(self.f.probe,self.f.cfg);policy.load_state_dict(payload["policy"])
        before=state_sha(policy);runtime=self.f.runtime();self.addCleanup(runtime.close)
        result=validation.evaluate(self.f.root/"val",self.f.protocol,payload,audit["published"][-1]["sha256"],
            self.f.stores["validation"],self.f.ledger["validation"],policy,runtime,self.f.probe,
            SimpleNamespace(requested=False))
        self.assertEqual(state_sha(policy),before);self.assertTrue(result["H_unchanged"])
        self.assertLessEqual(runtime.samples["text"],12)
    def test_original_truncation_allowed_only_diagnostic(self):
        runtime=self.f.runtime();self.addCleanup(runtime.close)
        text="A man wears a black coat. "*40
        with self.assertRaises(ValueError):runtime.encode_text((text,))
        x=runtime.encode_original((text,))
        self.assertEqual(x.shape,(1,512))

class CLISafetyTests(unittest.TestCase):
    def test_default_is_CPU(self):self.assertEqual(cli.parser().parse_args([]).action,"preflight")
    def test_gpu_actions_require_confirmation_before_catalog(self):
        for argv in (["--action","prepare","--role","train"],["--action","train","--epochs","1"],
                     ["--action","validate","--all-epoch-checkpoints"]):
            with patch.object(cli,"load_training_catalog",side_effect=AssertionError("must not load")):
                with self.assertRaises(ValueError):cli.main(argv)
    def test_epochs_not_silently_authorized(self):
        self.assertIsNone(cli.parser().parse_args(["--action","train"]).epochs)
        self.assertEqual(training.config()["epochs"],10)
    def test_path_cannot_overwrite_old_experiment(self):
        with self.assertRaises(ValueError):safe_root(BASE.parent/"phase4")
    def test_lock_excludes_concurrent_writer(self):
        with tempfile.TemporaryDirectory() as path:
            with run_lock(path):
                with self.assertRaises(ValueError):
                    with run_lock(path):pass
    def test_immutable_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as path:
            p=Path(path)/"p.json";immutable_json(p,{"a":1});immutable_json(p,{"a":1})
            with self.assertRaises(ValueError):immutable_json(p,{"a":2})
    def test_no_real_model_loader_in_test(self):
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
