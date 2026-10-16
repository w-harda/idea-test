"""Phase5 CPU：合成Gallery/文本特征、受约束H动作与事务恢复，零真实GPU。"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import copy,json,tempfile,unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from phase2_sources import digest,sha,read_json
from policy import PolicyOutput
from constraints import candidates,make_edit,replay
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from .common import legacy,immutable_json,source_pins,OUTPUT
from .protocol import validate_selection_record,test_layout_matches
from .storage import Store
from .backend import TextAttacker,Backend,validate_action,action_statistics
from . import evaluate as ev

class ScriptPolicy(nn.Module):
    """明确合成CPU动作策略，仅用于代码测试，不是训练H或真实IRRA。"""
    def __init__(self,probe):
        super().__init__();self.anchor=nn.Parameter(torch.tensor(1.));self.probe=probe;self.seen=[]
    def attack(self,example):
        self.seen.append(example)
        edit=None;chosen=None
        for i,a in enumerate(example.attributes):
            valid=candidates(example,example.caption,i,set(),self.probe)
            if valid:
                pos=next(iter(valid));edit=make_edit(example,i,pos,valid[pos][0]);chosen=a;break
        edits=(edit,) if edit else ()
        adv=replay(example,edits,1)
        return PolicyOutput(example.caption,adv,1,tuple(1. for _ in example.attributes),
            (chosen.attribute_id,) if chosen else (),edits,(),torch.tensor(0.),
            self.probe.pair_report(example.caption,adv),tuple(asdict(a) for a in example.attributes))

class CPUEncoder:
    def __init__(self,probe,limit):
        self.samples=0;self.limit=limit;self.probe=probe
    def encode(self,texts):
        self.samples+=len(texts)
        return F.normalize(torch.stack([torch.arange(1.,513.)+len(t)*.01 for t in texts]),dim=1)
    def finish(self):return {"CPU_fixture":True,"IRRA_text_samples_this_attempt":self.samples}

class CPUBackend(Backend):
    def __init__(self,protocol):
        self.protocol=protocol;self.probe=official_probe();self.frontend=CaptionFrontend()
        self.attacker=TextAttacker(ScriptPolicy(self.probe),self.frontend,self.probe)
        self.encoder_factory=CPUEncoder;self.encoder=None

class Fixture:
    def __init__(self,root):
        self.root=Path(root);rng=torch.Generator().manual_seed(4)
        self.images=F.normalize(torch.randn(12,512,generator=rng),dim=1)
        self.adv=F.normalize(self.images+.01*torch.randn(12,512,generator=rng),dim=1)
        self.text=self.images.repeat_interleave(2,dim=0)
        gallery=[{"index":i,"image":str(i)+".jpg","pid":i+1,"row_id":f"cuhk:{i}:0"} for i in range(12)]
        queries=[{"index":i,"pid":i//2+1,"paired_index":i//2,"caption_index":i%2,
                  "row_id":f"cuhk:{i//2}:{i%2}","caption":"A man wears a black shirt."} for i in range(24)]
        qids=[q["pid"] for q in queries];gids=[g["pid"] for g in gallery]
        metric=legacy("eval_metrics").metrics
        self.protocol={"schema":"hv1-joint-full-1","execution_backend":"cpu_fixture","gallery":gallery,"queries":queries,
            "gallery_count":12,"query_count":24,"text_batch":5,"source_sha256":{},"evidence_sha256":{},
            "model_sha256":{"G":"f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e",
                           "IRRA":"9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009","H":"f"*64},
            "H_selection":{"sha256":"a"*64,"checkpoint_sha256":"f"*64},
            "expected_baselines":{"Clean":metric(self.text,self.images,qids,gids)[0],
                                  "Image-only":metric(self.text,self.adv,qids,gids)[0]}}
        self.evidence={"G_historical_samples":12,"IRRA_image_historical_samples":12,"maximum_linf":.01,
                       "source_files_sha256":{},"G_new_samples":0,"IRRA_image_new_samples":0}
    def refs(self,p):return self.images.clone(),self.text.clone()
    def imgs(self,p,check=None):return self.adv.clone(),self.evidence
    def run(self,**kwargs):
        return ev.run(self.protocol,self.root,CPUBackend,self.refs,self.imgs,**kwargs)
    def audit(self):
        with patch.object(ev,"references",side_effect=self.refs),patch.object(ev,"adversarial_gallery",side_effect=self.imgs):
            return ev.audit(self.root,self.protocol)

class SelectionTests(unittest.TestCase):
    def setUp(self):
        def row(h,c,m):return {"checkpoint_sha256":h,"cursor":c,"groups":{"Joint":{"mAP":m},"Image-only":{"mAP":40.}},
                             "Joint_minus_ImageOnly_mAP_pp":m-40.}
        self.results={"a":row("a",2048,25.),"b":row("b",4096,30.)}
        from formal_training.validation import select_best
        self.record=select_best(list(self.results.values()))
    def test_valid_actual_validation_minimum(self):
        self.assertEqual(validate_selection_record(self.record,self.results,{"a","b"}),"a")
    def test_Test_driven_selection_rejected(self):
        self.record["uses_official_Test"]=True
        with self.assertRaises(ValueError):validate_selection_record(self.record,self.results,{"a","b"})
    def test_wrong_winner_rejected(self):
        self.record["selected_checkpoint_sha256"]="b"
        with self.assertRaises(ValueError):validate_selection_record(self.record,self.results,{"a","b"})
    def test_missing_validation_epoch_rejected(self):
        with self.assertRaises(ValueError):validate_selection_record(self.record,self.results,{"a","b","c"})
    def test_modified_validation_metric_rejected(self):
        self.record["candidates"][0]["groups"]["Joint"]["mAP"]=0.
        with self.assertRaises(ValueError):validate_selection_record(self.record,copy.deepcopy({
            "a":{"checkpoint_sha256":"a","cursor":2048,"groups":{"Joint":{"mAP":25.},"Image-only":{"mAP":40.}},
                 "Joint_minus_ImageOnly_mAP_pp":-15.},
            "b":self.results["b"]}),{"a","b"})
    def test_duplicate_selection_candidates_rejected(self):
        self.record["candidates"].append(self.record["candidates"][0])
        with self.assertRaises(ValueError):validate_selection_record(self.record,self.results,{"a","b"})
    def test_greedy_tie_selects_earliest_Validation_only(self):
        from formal_training.validation import select_best
        self.results["b"]["groups"]["Joint"]["mAP"]=25.;self.results["b"]["Joint_minus_ImageOnly_mAP_pp"]=-15.
        r=select_best(list(self.results.values()))
        self.assertEqual(validate_selection_record(r,self.results,{"a","b"}),"a")

class ActionTests(unittest.TestCase):
    def setUp(self):
        self.probe=official_probe();self.frontend=CaptionFrontend()
        self.policy=ScriptPolicy(self.probe);self.h=TextAttacker(self.policy,self.frontend,self.probe)
    def test_H_accepts_only_Caption(self):
        with self.assertRaises(TypeError):self.h.attack({"caption":"A man.","pid":1})
        with self.assertRaises(TypeError):self.h.attack("A man.",pid=1)
    def test_policy_never_receives_labels_or_features(self):
        self.h.attack("A man wears black pants.")
        ex=self.policy.seen[0]
        self.assertEqual(set(asdict(ex)),{"caption","attributes"})
    def test_trace_and_unicode_exact(self):
        caption="A man wears a black shirt."
        record=self.h.attack(caption)
        self.assertEqual(validate_action(caption,record,self.probe,self.frontend),record)
        self.assertTrue(record["edits"]);self.assertLessEqual(len(record["edits"]),record["budget_k"])
        self.assertIn("replacement_codepoint",record["edits"][0])
    def test_no_attributes_NOOP_retained(self):
        r=self.h.attack("xyzzy")
        self.assertFalse(r["eligible"]);self.assertEqual(r["adversarial"],"xyzzy");self.assertEqual(r["H_forward"],0)
    def test_truncated_original_NOOP(self):
        text="A man wears black pants. "*40;r=self.h.attack(text)
        self.assertEqual(r["adversarial"],text);self.assertFalse(r["eligible"])
        self.assertTrue(r["tokenizer_audit"]["before"]["truncated"])
    def test_oversize_NOOP_retained(self):
        text="A man "+("x"*1200);r=self.h.attack(text)
        self.assertIn("H_bytes_over_1024",r["reasons"]);self.assertEqual(r["adversarial"],text)
    def test_wrong_caption_binding_rejected(self):
        r=self.h.attack("A man wears a black shirt.")
        with self.assertRaises(ValueError):validate_action("A woman.",r,self.probe)
    def test_illegal_unicode_trace_rejected(self):
        text="A man wears black pants.";r=self.h.attack(text)
        r["edits"][0]["replacement_char"]="Z"
        with self.assertRaises(ValueError):validate_action(text,r,self.probe)
    def test_fake_attribute_span_rejected(self):
        text="A man wears black pants.";r=self.h.attack(text)
        r["attributes"][0]["char_start"]+=1
        with self.assertRaises(ValueError):validate_action(text,r,self.probe)
    def test_tokenizer_metadata_tampering_rejected(self):
        text="A man wears black pants.";r=self.h.attack(text)
        r["tokenizer_audit"]["model_tokens_changed"]=False
        with self.assertRaises(ValueError):validate_action(text,r,self.probe)
    def test_stats_include_NOOP_all_queries(self):
        rows=[self.h.attack("xyzzy"),self.h.attack("A man in a black coat.")]
        stats=action_statistics(rows)
        self.assertEqual(stats["query_count"],2);self.assertEqual(stats["NOOP_queries"],1)
        self.assertEqual(stats["H_forward_samples"],1)
    def test_H_frozen_state_unchanged(self):
        self.h.attack("A woman in blue jeans.")
        self.assertTrue(self.h.finish()["H_unchanged"]);self.assertIsNone(self.policy.anchor.grad)
    def test_JSON_roundtrip_edit_trace(self):
        text="A man wears a black coat.";r=self.h.attack(text)
        validate_action(text,json.loads(json.dumps(r)),self.probe,self.frontend)

class SelectedHLoadTests(unittest.TestCase):
    def setUp(self):
        from policy import CharacterPolicy
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/"checkpoint.pt"
        self.arch={"width":16,"layers":1,"heads":4,"max_bytes":1024}
        self.policy=CharacterPolicy(official_probe(),**self.arch)
        self.payload={"schema":"hv1-formal-periodic-checkpoint-1","protocol_sha256":"b"*64,
                      "policy":self.policy.state_dict()}
    def save(self):
        torch.save(self.payload,self.path)
        return {"query_count":1,"H_selection":{"checkpoint":str(self.path),"checkpoint_sha256":sha(self.path),
                "training_protocol_sha256":"b"*64,"architecture":self.arch}}
    def test_selected_H_weights_load_exact_CPU(self):
        from feedback_backend import state_sha
        b=Backend(self.save(),CPUEncoder)
        self.assertEqual(b.attacker.before,state_sha(self.policy))
        self.assertTrue(b.finish()["H_unchanged"]);self.assertFalse(torch.cuda.is_initialized())
    def test_checkpoint_file_SHA_checked_before_load(self):
        p=self.save();self.path.write_bytes(b"bad")
        with self.assertRaises(ValueError):Backend(p,CPUEncoder)
    def test_wrong_training_protocol_rejected(self):
        self.payload["protocol_sha256"]="c"*64
        with self.assertRaises(ValueError):Backend(self.save(),CPUEncoder)
    def test_H_first_batch_uses_Validation_attention_settings(self):
        previous=torch.backends.mha.get_fastpath_enabled()
        try:
            torch.backends.mha.set_fastpath_enabled(True)
            b=Backend(self.save(),CPUEncoder)
            self.assertFalse(torch.backends.mha.get_fastpath_enabled())
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.cuda.is_initialized())
            b.finish()
        finally:torch.backends.mha.set_fastpath_enabled(previous)
    def test_wrong_training_schema_rejected(self):
        self.payload["schema"]="legacy_step32"
        with self.assertRaises(ValueError):Backend(self.save(),CPUEncoder)

class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name);self.store=Store(self.f.root,self.f.protocol,True)
    def publish(self):
        i,rows=self.store.parts("clean_text")[0]
        return self.store.prepare("clean_text",i,rows,lambda:(self.f.text[:len(rows)],
                      {"G_new_samples":0,"IRRA_image_new_samples":0,"IRRA_text_samples":0,"reference_reused":True}))
    def test_storage_compact_and_values_exact(self):
        self.publish();x,_=self.store.load("clean_text",0,self.f.protocol["queries"][:5])
        self.assertTrue(torch.equal(x,self.f.text[:5]))
        self.assertEqual(x.untyped_storage().nbytes(),5*512*4)
    def test_reuse_does_not_call_producer(self):
        self.publish();i,rows=self.store.parts("clean_text")[0]
        self.store.prepare("clean_text",i,rows,lambda:(_ for _ in ()).throw(AssertionError("recompute")))
    def test_corrupt_feature_refused(self):
        self.publish();(self.store.path("clean_text",0)/"features.pt").write_bytes(b"bad")
        with self.assertRaises(ValueError):self.store.load("clean_text",0,self.f.protocol["queries"][:5])
    def test_mismatched_identity_refused(self):
        self.publish();rows=copy.deepcopy(self.f.protocol["queries"][:5]);rows[0]["pid"]=900
        with self.assertRaises(ValueError):self.store.load("clean_text",0,rows)
    def test_pending_requires_user_ack_and_preserves_evidence(self):
        i,rows=self.store.parts("adv_text")[0]
        with self.assertRaises(RuntimeError):
            self.store.prepare("adv_text",i,rows,lambda:(_ for _ in ()).throw(RuntimeError("CPU interruption")))
        with self.assertRaises(ValueError):self.store.check_pending()
        self.store.check_pending(True)
        self.assertEqual(len(list((self.f.root/"abandoned").glob("*/recovery.json"))),1)
    def test_unknown_pending_not_recoverable(self):
        p=self.f.root/"features/unknown/000000.pending";p.mkdir(parents=True)
        with self.assertRaises(ValueError):self.store.check_pending(True)
    def test_image_generation_in_Phase5_forbidden(self):
        i,rows=self.store.parts("clean_text")[0]
        with self.assertRaises(ValueError):self.store.prepare("clean_text",i,rows,lambda:(
            self.f.text[:len(rows)],{"G_new_samples":1,"IRRA_image_new_samples":0}))
    def test_partial_audit_and_full_refusal(self):
        self.publish();self.assertEqual(self.store.audit(False)["clean_text"],1)
        with self.assertRaises(ValueError):self.store.audit()

    def test_receipt_corruption_rejected(self):
        self.publish();p=self.store.path("clean_text",0)/"receipt.json"
        data=read_json(p);data["feature_sha256"]="0"*64;p.write_text(json.dumps(data))
        with self.assertRaises(ValueError):self.store.load("clean_text",0,self.f.protocol["queries"][:5])

    def test_published_intent_tampering_rejected(self):
        self.publish();p=self.store.path("clean_text",0)/"intent.json"
        value=read_json(p);value["index"]=99;p.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.store.load("clean_text",0,self.f.protocol["queries"][:5])

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(self.tmp.name)
    def test_four_groups_and_full_precision_csv(self):
        r=self.f.run();self.assertEqual(set(r["groups"]),set(ev.GROUPS))
        self.assertEqual((self.f.root/"metrics.csv").read_bytes(),ev.csv_bytes(r))
        self.assertEqual(r["counts"]["G_new_samples"],0);self.assertEqual(r["query_count"],24)
    def test_CPU_readonly_audit_exact_and_no_model(self):
        self.f.run()
        files=[p for p in self.f.root.rglob("*") if p.is_file()]
        before={str(p):(sha(p),p.stat().st_mtime_ns) for p in files}
        with patch.object(ev,"Backend",side_effect=AssertionError("no models")):self.f.audit()
        self.assertEqual(before,{str(p):(sha(p),p.stat().st_mtime_ns) for p in files})
    def test_reentry_complete_no_extra_backend(self):
        self.f.run()
        with patch.object(CPUBackend,"__init__",side_effect=AssertionError("no new H/GPU")):self.f.run()
    def test_baseline_mismatch_stops_before_H(self):
        self.f.protocol["expected_baselines"]["Clean"]["mAP"]-=1
        with patch.object(CPUBackend,"__init__",side_effect=AssertionError("no GPU")),self.assertRaises(ValueError):
            self.f.run()
        self.assertTrue(list((self.f.root/"attempts").glob("*/failure.json")))
    def test_safe_stop_then_resume_identical_metrics(self):
        flag=SimpleNamespace(requested=False)
        original=CPUBackend.text
        def stop_after_first(owner,*args):
            result=original(owner,*args);flag.requested=True;return result
        with patch.object(CPUBackend,"text",stop_after_first),self.assertRaises(ev.SafeStop):self.f.run(stop=flag)
        self.assertTrue(list((self.f.root/"attempts").glob("*/interruption.json")))
        r=self.f.run()
        self.assertEqual(r,self.f.audit())
    def test_failed_uncommitted_text_requires_explicit_recovery(self):
        original=CPUBackend.text
        def interrupted(owner,*args):
            original(owner,*args);raise RuntimeError("CPU simulated text inference failure")
        with patch.object(CPUBackend,"text",interrupted),self.assertRaises(RuntimeError):self.f.run()
        with self.assertRaises(ValueError):self.f.run()
        result=self.f.run(recover_pending=True)
        self.assertEqual(result["query_count"],24);self.assertTrue(list((self.f.root/"abandoned").glob("*/recovery.json")))
    def test_missing_completed_result_evidence_refused(self):
        self.f.run();(self.f.root/"metrics.csv").write_text("bad")
        with self.assertRaises(ValueError):self.f.audit()
    def test_query_count_twice_gallery_not_regenerated(self):
        r=self.f.run()
        self.assertEqual(r["gallery_count"],12);self.assertEqual(r["query_count"],24)
        self.assertEqual(r["counts"]["G_historical_samples"],12)
    def test_joint_comparison_negative_allowed(self):
        d=ev.comparisons({"Image-only":{"mAP":30.},"Joint":{"mAP":40.}})
        self.assertEqual(d["Joint_minus_ImageOnly_mAP_pp"],10.)
        self.assertFalse(d["Joint_better_than_ImageOnly"])
    def test_drop_rate_definition_exact(self):
        r=self.f.run()
        for k in ev.GROUPS:
            expected=(r["groups"]["Clean"]["mAP"]-r["groups"][k]["mAP"])/r["groups"]["Clean"]["mAP"]*100
            self.assertEqual(r["DR_mAP_percent"][k],expected)
    def test_NOOPs_kept_in_all_6156_protocol_positions(self):
        self.f.protocol["queries"][0]["caption"]="xyzzy"
        self.f.protocol["queries"][1]["caption"]="A man wears black pants. "*40
        r=self.f.run()
        self.assertEqual(r["query_statistics"]["NOOP_queries"],2)
        self.assertEqual(r["query_count"],24)
    def test_changed_H_protocol_refuses_existing_cache(self):
        self.f.run();changed=copy.deepcopy(self.f.protocol);changed["model_sha256"]["H"]="e"*64
        with self.assertRaises(ValueError):Store(self.f.root,changed)

    def test_attempt_record_tampering_rejected(self):
        self.f.run();done=read_json(self.f.root/"completed.json")
        (self.f.root/done["execution_path"]).write_text("{}")
        with self.assertRaises(ValueError):self.f.audit()
    def test_NOOP_text_does_not_construct_encoder(self):
        b=CPUBackend(self.f.protocol)
        b.encoder_factory=lambda *_:(_ for _ in ()).throw(AssertionError("no GPU"))
        x,stats=b.text(("xyzzy",),self.f.text[:1])
        self.assertTrue(torch.equal(x,self.f.text[:1]));self.assertEqual(stats["IRRA_text_samples"],0)
        b.finish()

class SafetyTests(unittest.TestCase):
    def test_default_never_resolves_or_loads_models(self):
        with patch.object(ev,"source_pins",return_value={}),patch.object(ev,"resolve_selection",side_effect=AssertionError("not ready")):
            ev.main([])
    def test_run_confirmation_before_any_preflight(self):
        with patch.object(ev,"preflight",side_effect=AssertionError("no source/GPU")):
            with self.assertRaises(ValueError):ev.main(["--action","run"])
    def test_output_cannot_point_to_Phase4(self):
        with self.assertRaises(ValueError):ev.main(["--output",str(OUTPUT.parents[1]/"formal_v1")])
    def test_recovery_only_user_run(self):
        with self.assertRaises(ValueError):ev.main(["--action","audit","--recover-pending"])
    def test_no_smoke_or_100_mode(self):
        self.assertNotIn("mode",{x.dest for x in ev.parser()._actions})
    def test_frozen_sources_still_match(self):
        self.assertGreater(len(source_pins()),62)
    def test_layout_all_Query_identities_and_positives(self):
        with tempfile.TemporaryDirectory() as p:
            f=Fixture(p);test_layout_matches(f.protocol["gallery"],f.protocol["queries"],f.protocol)
            bad=copy.deepcopy(f.protocol["queries"]);bad[0]["paired_index"]=11
            with self.assertRaises(ValueError):test_layout_matches(f.protocol["gallery"],bad,f.protocol)
    def test_no_GPU_initialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
