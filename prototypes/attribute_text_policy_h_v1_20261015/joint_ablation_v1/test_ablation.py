"""联合消融CPU tests：合成图文特征、真实Span/BPE规则，零真实GPU。"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import copy,json,tempfile,unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
from phase2_sources import digest,sha,read_json
from constraints import candidates,make_edit,replay
from contracts import word_at
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from feedback_backend import state_sha
from phase5_joint.backend import validate_action
from .common import trials,trial_protocol,legacy,pins
from .strategies import (RandomStrategy,UntrainedStrategy,build_untrained,rng_for,validate_reference,
                         validate_trial_action,edit_statistics,reference_statistics)
from .backend import Engine
from .storage import TrialStore
from . import evaluate as ev

PROBE=official_probe()
FRONT=CaptionFrontend()
def learned_record(caption,target=1):
    from formal_training.data import qualify_caption,example_from
    entry=qualify_caption(caption,FRONT,PROBE);ex=example_from(entry)
    used=set();edits=[];text=caption
    indices=sorted(range(len(ex.attributes)),key=lambda i:(ex.attributes[i].semantic_axis!="color",i))
    for i in indices:
        if len(edits)>=target:break
        options=candidates(ex,text,i,used,PROBE)
        if options:
            pos=next(iter(options));edit=make_edit(ex,i,pos,options[pos][0])
            edits.append(edit);text=text[:pos]+edit.replacement_char+text[pos+1:]
            used.add(word_at(caption,pos))
    if not entry["eligible"]:edits=[];text=caption
    budget=min(len(ex.attributes),max(len(edits)+1,2)) if entry["eligible"] else 0
    selected=[e.attribute_id for e in edits]
    for a in ex.attributes:
        if len(selected)>=budget:break
        if a.attribute_id not in selected:selected.append(a.attribute_id)
    record={"original":caption,"adversarial":text,"budget_k":budget,
        "predicted_importance":[1.]*len(ex.attributes),"attributes":[asdict(a) for a in ex.attributes],
        "selected_attribute_ids":selected,"edits":[asdict(e) for e in edits],"decisions":[],
        "tokenizer_audit":PROBE.pair_report(caption,text),"eligible":entry["eligible"],
        "reasons":entry["reasons"],"H_forward":int(entry["eligible"])}
    validate_reference(caption,record,PROBE,FRONT)
    return record

class CPUEncoder:
    def __init__(self,probe,limit):self.probe=probe;self.limit=limit;self.samples=0
    def encode(self,texts):
        self.samples+=len(texts)
        if self.samples>self.limit:raise ValueError("CPU fixture limit")
        return F.normalize(torch.stack([torch.arange(1.,513.)+len(t)*.01 for t in texts]),dim=1)
    def finish(self):return {"CPU_fixture":True,"IRRA_text_samples_this_attempt":self.samples}
class CPUEngine(Engine):
    def __init__(self,p):super().__init__(p,CPUEncoder)

class Fixture:
    def __init__(self,root):
        self.root=Path(root)
        rng=torch.Generator().manual_seed(8)
        self.arrays={"clean_gallery":F.normalize(torch.randn(12,512,generator=rng),dim=1)}
        self.arrays["image_gallery"]=F.normalize(self.arrays["clean_gallery"]+.05,dim=1)
        self.arrays["clean_text"]=self.arrays["clean_gallery"].repeat_interleave(2,dim=0)
        self.arrays["adv_text"]=self.arrays["clean_text"].clone()
        self.queries=[{"index":i,"row_id":f"cuhk:{i//2}:{i%2}","caption_index":i%2,
                      "pid":i//2+1,"paired_index":i//2,"caption":"A man wears a black shirt."} for i in range(24)]
        self.queries[0]["caption"]="xyzzy"
        self.gallery=[{"index":i,"row_id":f"cuhk:{i}:0","image":str(i)+".jpg","pid":i+1} for i in range(12)]
        self.learned=[learned_record(q["caption"]) for q in self.queries]
        ids=[q["pid"] for q in self.queries];gids=[g["pid"] for g in self.gallery]
        groups={name:legacy("eval_metrics").metrics(self.arrays[text],self.arrays[image],ids,gids)[0]
                for name,text,image in (("Clean","clean_text","clean_gallery"),
                 ("Image-only","clean_text","image_gallery"),("Text-only","adv_text","clean_gallery"),
                 ("Joint","adv_text","image_gallery"))}
        cfg={"random_seeds":[42,43],"untrained_seed":42,"search_max_nodes":256}
        arch={"width":16,"layers":1,"heads":4,"max_bytes":1024}
        self.parent={"schema":"hv1-joint-ablation-1","execution_backend":"cpu_fixture","config":cfg,
            "trials":trials(cfg),"gallery":self.gallery,"queries":self.queries,"text_batch":5,
            "architecture":arch,"source_sha256":{},"input_files_sha256":{},"Test_selection":False,
            "model_sha256":{"G":"a"*64,"IRRA":"b"*64,"H":"c"*64},
            "fixed_baseline_groups":groups,"learned_edit_statistics":reference_statistics(self.learned),
            "untrained_initial_state_sha256":state_sha(build_untrained(arch,PROBE))}
    def run(self,**kw):return ev.run(self.parent,self.root,self.arrays,self.learned,CPUEngine,**kw)
    def audit(self):return ev.audit(self.root,self.parent,self.arrays,self.learned)

class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.caption="A man wears a black shirt and blue pants."
        self.ref=learned_record(self.caption,2)
    def strategy(self,kind="random_attribute",seed=42,max_nodes=256):
        return RandomStrategy(kind,seed,PROBE,FRONT,max_nodes)
    def test_random_exact_actual_edits_not_K(self):
        ref=learned_record(self.caption,1)
        self.assertGreater(ref["budget_k"],len(ref["edits"]))
        a=self.strategy().attack(self.caption,ref)
        self.assertEqual(len(a["edits"]),1);self.assertEqual(a["budget_k"],ref["budget_k"])
    def test_random_all_legal_attributes_coverage(self):
        ids=set()
        ref=learned_record(self.caption,1)
        for seed in range(12):
            a=self.strategy(seed=seed).attack(self.caption,ref);ids.update(e["attribute_id"] for e in a["edits"])
        self.assertGreater(len(ids),1)
    def test_H_selected_preserves_full_selection_and_actual_support(self):
        a=self.strategy("h_selected").attack(self.caption,self.ref)
        self.assertEqual(a["selected_attribute_ids"],self.ref["selected_attribute_ids"])
        self.assertEqual([e["attribute_id"] for e in a["edits"]],[e["attribute_id"] for e in self.ref["edits"]])
    def test_H_selected_randomizes_position_or_Unicode(self):
        ref=learned_record(self.caption,1)
        strings={self.strategy("h_selected",seed).attack(self.caption,ref)["adversarial"] for seed in range(10)}
        self.assertGreater(len(strings),1)
    def test_fixed_seed_repeat_and_JSON_roundtrip(self):
        a=self.strategy().attack(self.caption,self.ref);b=self.strategy().attack(self.caption,self.ref)
        self.assertEqual(a,b)
        trial={"kind":"random_attribute","seed":42}
        validate_trial_action(self.caption,json.loads(json.dumps(a)),json.loads(json.dumps(self.ref)),trial,PROBE,FRONT)
    def test_no_global_rng_consumption(self):
        import random
        random.seed(6);before=random.getstate()
        self.strategy().attack(self.caption,self.ref)
        self.assertEqual(random.getstate(),before)
    def test_Caption_only_rejects_PID(self):
        ref=copy.deepcopy(self.ref);ref["pid"]=1
        with self.assertRaises(ValueError):self.strategy().attack(self.caption,ref)
        with self.assertRaises(TypeError):self.strategy().attack(self.caption,self.ref,pid=1)
    def test_no_image_or_reward_in_rng(self):
        self.assertEqual(rng_for(self.caption,42,"random_attribute").random(),
                         rng_for(self.caption,42,"random_attribute").random())
        with self.assertRaises(TypeError):rng_for(self.caption,42,"random_attribute",pid=1)
    def test_source_NOOP_is_exact_random_NOOP(self):
        ref=learned_record(self.caption,0)
        self.assertEqual(len(ref["edits"]),0)
        for kind in ("random_attribute","h_selected"):
            a=self.strategy(kind).attack(self.caption,ref)
            self.assertEqual(a["adversarial"],self.caption);self.assertTrue(a["ablation"]["matched"])
    def test_no_attributes_retained(self):
        ref=learned_record("xyzzy")
        a=self.strategy().attack("xyzzy",ref)
        self.assertEqual(a["budget_k"],0);self.assertEqual(a["adversarial"],"xyzzy")
    def test_original_truncation_retained(self):
        caption="A man wears a black shirt. "*40
        a=self.strategy().attack(caption,learned_record(caption))
        self.assertEqual(a["adversarial"],caption);self.assertEqual(a["H_forward"],0)
    def test_per_word_single_edit_and_valid_trace(self):
        for kind in ("random_attribute","h_selected"):
            for seed in (42,43,44):
                a=self.strategy(kind,seed).attack(self.caption,self.ref)
                words=[word_at(self.caption,e["position"]) for e in a["edits"]]
                self.assertEqual(len(words),len(set(words)))
                validate_action(self.caption,a,PROBE,FRONT)
    def test_unmatched_finite_search_explicit_not_impossible(self):
        self.assertEqual(len(self.ref["edits"]),2)
        a=self.strategy(max_nodes=1).attack(self.caption,self.ref)
        self.assertLess(len(a["edits"]),2);self.assertFalse(a["ablation"]["matched"])
        self.assertEqual(a["ablation"]["failure_reason"],"search_node_limit_no_complete_match")
        self.assertTrue(a["ablation"]["search_exhausted"]);self.assertEqual(a["budget_k"],self.ref["budget_k"])
    def test_matching_statistics_no_silent_drop(self):
        a=self.strategy(max_nodes=1).attack(self.caption,self.ref)
        s=edit_statistics([a])
        self.assertEqual(s["queries"],1);self.assertEqual(s["under_target_queries"],1)
        self.assertEqual(s["over_target_queries"],0)
    def test_tampered_count_accounting_refused(self):
        a=self.strategy().attack(self.caption,self.ref);a["ablation"]["actual_edits"]=999
        with self.assertRaises(ValueError):validate_trial_action(self.caption,a,self.ref,{"kind":"random_attribute","seed":42},PROBE,FRONT)
    def test_tampered_H_support_refused(self):
        a=self.strategy("h_selected").attack(self.caption,self.ref);a["selected_attribute_ids"].reverse()
        with self.assertRaises(ValueError):validate_trial_action(self.caption,a,self.ref,{"kind":"h_selected","seed":42},PROBE,FRONT)
    def test_tampered_source_budget_and_search_limit_refused(self):
        a=self.strategy().attack(self.caption,self.ref);a["ablation"]["source_H_budget_k"]+=1
        with self.assertRaises(ValueError):validate_trial_action(self.caption,a,self.ref,{"kind":"random_attribute","seed":42},PROBE,FRONT)
        a=self.strategy().attack(self.caption,self.ref);a["ablation"]["search_limit"]=999
        with self.assertRaises(ValueError):validate_trial_action(self.caption,a,self.ref,{"kind":"random_attribute","seed":42},PROBE,FRONT)
    def test_illegal_Unicode_rejected(self):
        a=self.strategy().attack(self.caption,self.ref);a["edits"][0]["replacement_char"]="Z"
        with self.assertRaises(ValueError):validate_action(self.caption,a,PROBE,FRONT)
    def test_changed_Caption_reference_rejected(self):
        with self.assertRaises(ValueError):self.strategy().attack("A woman.",self.ref)

class UntrainedTests(unittest.TestCase):
    def test_seed42_reproducible_and_CPU_rng_preserved(self):
        arch={"width":16,"layers":1,"heads":4,"max_bytes":1024}
        before=torch.get_rng_state().clone()
        p=build_untrained(arch,PROBE);q=build_untrained(arch,PROBE)
        self.assertTrue(torch.equal(torch.get_rng_state(),before));self.assertEqual(state_sha(p),state_sha(q))
    def test_default_initial_H_matches_original_seed42(self):
        p=build_untrained({"width":64,"layers":2,"heads":4,"max_bytes":1024},PROBE)
        self.assertEqual(state_sha(p),"9d607ba8b6b348a8783b3dad9b79d0206dd23147edec5fa794b7e932c5a2853f")
    def test_untrained_no_checkpoint_loading_or_SCST(self):
        with patch.object(torch,"load",side_effect=AssertionError("no trained weights")):
            s=UntrainedStrategy({"width":16,"layers":1,"heads":4,"max_bytes":1024},PROBE,FRONT)
            s.attack("A man in a black coat.",learned_record("A man in a black coat."))
            result=s.finish()
        self.assertEqual(result["SCST_training_steps"],0);self.assertTrue(result["H_unchanged"])
    def test_untrained_keeps_own_natural_budget(self):
        caption="A man wears black pants.";ref=learned_record(caption,0)
        s=UntrainedStrategy({"width":16,"layers":1,"heads":4,"max_bytes":1024},PROBE,FRONT)
        a=s.attack(caption,ref)
        self.assertIn("not forced",a["ablation"]["count_matching"])
        self.assertEqual(a["ablation"]["target_edits"],0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in s.attacker.policy.parameters()))
    def test_untrained_policy_input_no_labels(self):
        s=UntrainedStrategy({"width":16,"layers":1,"heads":4,"max_bytes":1024},PROBE,FRONT)
        with self.assertRaises(TypeError):s.attack("A man.",learned_record("A man."),pid=1)
        self.assertFalse(torch.cuda.is_initialized())

class StatisticsTests(unittest.TestCase):
    def test_sample_std_ddof1(self):
        r=ev.mean_std([1.,2.,3.,4.,5.])
        self.assertEqual(r["n"],5);self.assertEqual(r["mean"],3.)
        self.assertAlmostEqual(r["std"],(2.5)**.5)
    def test_single_seed_std_NA(self):self.assertIsNone(ev.mean_std([42.])["std"])
    def test_config_fixed_multiple_seeds_and_no_override(self):
        from .common import config
        self.assertEqual(config()["random_seeds"],[42,43,44,45,46])
        self.assertNotIn("seed",{x.dest for x in ev.parser()._actions})
    def test_csv_order_independent_of_JSON_key_sort(self):
        with tempfile.TemporaryDirectory() as path:
            f=Fixture(path)
            # 无真实前向；只合成统计结果，JSON排序不应改变CSV顺序。
            results=[]
            for trial in f.parent["trials"]:
                results.append({"trial":trial,"groups":f.parent["fixed_baseline_groups"],
                    "edit_statistics":{"NOOP_ratio":.5,"exact_count_match_ratio":1.},
                    "Joint_minus_LearnedJoint_mAP_pp":0.,"DR_mAP_percent":{k:0. for k in f.parent["fixed_baseline_groups"]}})
            summary=ev.aggregate(f.parent,results)
            self.assertEqual(ev.aggregate_csv(summary),ev.aggregate_csv(json.loads(json.dumps(summary,sort_keys=True))))
    def test_reference_edit_distribution(self):
        rows=[learned_record("xyzzy"),learned_record("A man in a black coat.",1)]
        s=reference_statistics(rows)
        self.assertEqual(s["NOOP_ratio"],.5);self.assertEqual(s["actual_edit_count_distribution"],{"0":1,"1":1})

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.f=Fixture(self.tmp.name)
    def test_resume_after_trial_execution_before_completion(self):
        original=ev.immutable_json
        failed=[]
        def interrupt(path,payload):
            path=Path(path)
            if path.name=="completed.json" and path.parent.parent.name=="variants" and not failed:
                failed.append(path)
                raise RuntimeError("fixture interruption at result receipt")
            return original(path,payload)
        with patch.object(ev,"immutable_json",side_effect=interrupt):
            with self.assertRaises(RuntimeError):self.f.run()
        execution=failed[0].parent/"execution.json"
        original_sha=sha(execution)
        self.f.run()
        self.assertEqual(sha(execution),original_sha)
        self.f.audit()
    def test_changed_uncommitted_execution_refused(self):
        original=ev.immutable_json
        failed=[]
        def interrupt(path,payload):
            path=Path(path)
            if path.name=="completed.json" and path.parent.parent.name=="variants" and not failed:
                failed.append(path)
                raise RuntimeError("fixture interruption at result receipt")
            return original(path,payload)
        with patch.object(ev,"immutable_json",side_effect=interrupt):
            with self.assertRaises(RuntimeError):self.f.run()
        execution=failed[0].parent/"execution.json"
        value=read_json(execution);value["result_digest"]="f"*64
        execution.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,"execution/result mismatch"):self.f.run()
    def test_all_seed_trials_and_original_baselines_reused(self):
        s=self.f.run()
        self.assertEqual(len(s["trials"]),5)
        self.assertEqual(s["Learned_H_v1"]["groups"],self.f.parent["fixed_baseline_groups"])
        self.assertEqual(s["G_new_samples"],0);self.assertEqual(s["IRRA_image_new_samples"],0)
        self.assertIn("learned_h_v1",s["aggregate"])
    def test_aggregate_and_complete_CSV(self):
        s=self.f.run()
        self.assertEqual((self.f.root/"metrics.csv").read_bytes(),ev.detailed_csv(s))
        self.assertEqual((self.f.root/"aggregate.csv").read_bytes(),ev.aggregate_csv(s))
        self.assertEqual(s["aggregate"]["random_attribute"]["groups"]["Joint"]["mAP"]["n"],2)
        self.assertIsNone(s["aggregate"]["untrained"]["groups"]["Joint"]["mAP"]["std"])
    def test_random_edit_count_distribution_exact_and_NOOP(self):
        s=self.f.run()
        for row in s["trials"]:
            if row["trial"]["kind"]!="untrained":
                e=row["edit_statistics"]
                self.assertEqual(e["target_edit_count_distribution"],e["actual_edit_count_distribution"])
                self.assertEqual(e["NOOP_queries"],1)
    def test_independent_CPU_readonly_audit_exact(self):
        result=self.f.run()
        paths=[p for p in self.f.root.rglob("*") if p.is_file()]
        old={str(p):(sha(p),p.stat().st_mtime_ns) for p in paths}
        self.assertEqual(result,self.f.audit())
        self.assertEqual(old,{str(p):(sha(p),p.stat().st_mtime_ns) for p in paths})
    def test_completed_reentry_no_new_model(self):
        self.f.run()
        with patch.object(CPUEngine,"__init__",side_effect=AssertionError("no new models")):self.f.run()
    def test_source_models_selection_unchanged_in_result(self):
        s=self.f.run()
        self.assertEqual(s["model_sha256"],self.f.parent["model_sha256"])
        self.assertFalse(s["Test_model_selection"])
    def test_Trial_text_cache_only_no_Gallery_generation(self):
        self.f.run()
        for p in (self.f.root/"variants").glob("*/features"):
            self.assertEqual([x.name for x in p.iterdir()],["adv_text"])
    def test_interrupted_resume_matches_uninterrupted(self):
        flag=SimpleNamespace(requested=False);original=CPUEngine.batch
        def stop(owner,*args):
            result=original(owner,*args);flag.requested=True;return result
        with patch.object(CPUEngine,"batch",stop),self.assertRaises(ev.SafeStop):self.f.run(stop=flag)
        result=self.f.run()
        self.assertEqual(result,self.f.audit())
    def test_hard_failure_pending_preserved_recovery(self):
        original=CPUEngine.batch
        def fail(owner,*args):original(owner,*args);raise RuntimeError("CPU fixture hard failure")
        with patch.object(CPUEngine,"batch",fail),self.assertRaises(RuntimeError):self.f.run()
        with self.assertRaises(ValueError):self.f.run()
        result=self.f.run(recover_pending=True)
        self.assertTrue(list((self.f.root/"variants").glob("*/abandoned/*/recovery.json")))
        self.assertEqual(result,self.f.audit())
    def test_cache_feature_corruption_rejected(self):
        self.f.run();p=next((self.f.root/"variants").glob("*/features/adv_text/*/features.pt"));p.write_bytes(b"bad")
        with self.assertRaises(ValueError):self.f.audit()
    def test_changed_config_refuses_reuse(self):
        self.f.run();self.f.parent["config"]["search_max_nodes"]=1
        with self.assertRaises(ValueError):self.f.run()
    def test_trial_summary_tampering_rejected(self):
        self.f.run();p=next((self.f.root/"variants").glob("*/summary.json"));p.write_text("{}")
        with self.assertRaises(ValueError):self.f.audit()
    def test_missing_trial_cannot_aggregate(self):
        s=self.f.run()
        with self.assertRaises(ValueError):ev.aggregate(self.f.parent,s["trials"][:-1])
    def test_corrupt_attempt_rejected(self):
        self.f.run();p=next((self.f.root/"attempts").glob("*/execution.json"));p.write_text("{}")
        with self.assertRaises(ValueError):self.f.audit()

class SafetyTests(unittest.TestCase):
    def test_default_CPU_only(self):self.assertEqual(ev.parser().parse_args([]).action,"preflight")
    def test_real_run_requires_user_before_preflight(self):
        with patch.object(ev,"preflight",side_effect=AssertionError("no preflight/model")):
            with self.assertRaises(ValueError):ev.main(["--action","run"])
    def test_no_unfrozen_random_seed_CLI(self):
        self.assertNotIn("search_limit",{x.dest for x in ev.parser()._actions})
    def test_history_output_forbidden(self):
        from .common import FULL
        with self.assertRaises(ValueError):ev.main(["--output",str(FULL)])
    def test_recovery_only_manual_run(self):
        with self.assertRaises(ValueError):ev.main(["--action","audit","--recover-pending"])
    def test_frozen_sources_unchanged(self):self.assertGreater(len(pins()),70)
    def test_no_GPU_initialized(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
