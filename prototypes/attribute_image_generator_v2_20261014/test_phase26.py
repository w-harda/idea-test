"""Phase 2.6 CPU：恢复日志完整性、固定单Epoch入口与统计。"""
import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import cache_io as io
import epoch_checkpoint as ck
import phase26_train as runner
from phase26_support import FIELDS,stats,summarize,validation_stats
from test_phase24 import Base,tree_equal

class LogRecoveryTests(Base):
    def setup_checkpoint(self):
        g,enc,opt=self.components()
        row={"epoch":1,"sample_position":1,"row_id":"cuhk:10:0","role":"train","K":0,"optimizer_steps":1,"rounds":[{"total":.1}]}
        log=self.root/"samples.jsonl";log.write_text(json.dumps(row)+"\n")
        progress={"epoch":1,"samples":1,"sample_order":[row["row_id"]],"optimizer_steps":1,"sample_logs":[row],
            "sample_log_path":str(log),"sample_log_sha256":io.sha(log)}
        contract={"config":{"scope":"formal"},"data":"fixed"}
        p=ck.save_epoch(self.root,g,opt,1,contract,progress)
        return g,opt,contract,p,log,progress
    def test_valid_associated_log_restores(self):
        g,o,c,p,log,progress=self.setup_checkpoint()
        self.assertEqual(ck.load_latest(self.root,c,g,o)["next_epoch"],2)
    def test_modified_log_SHA_rejected_before_state_mutation(self):
        g,o,c,p,log,progress=self.setup_checkpoint();before=copy.deepcopy(g.state_dict())
        log.write_text("corrupted")
        with self.assertRaisesRegex(ValueError,"SHA"):ck.load_latest(self.root,c,g,o)
        self.assertTrue(tree_equal(before,g.state_dict()))
    def test_deleted_log_rejected(self):
        g,o,c,p,log,progress=self.setup_checkpoint();log.unlink()
        with self.assertRaisesRegex(ValueError,"sample log missing"):ck.load_latest(self.root,c,g,o)
    def test_truncated_log_rejected(self):
        g,o,c,p,log,progress=self.setup_checkpoint();log.write_bytes(log.read_bytes()[:10])
        with self.assertRaisesRegex(ValueError,"SHA"):ck.load_latest(self.root,c,g,o)
    def test_matching_updated_SHA_but_wrong_content_rejected(self):
        g,o,c,p,log,progress=self.setup_checkpoint()
        row=copy.deepcopy(progress["sample_logs"][0]);row["row_id"]="cuhk:11:0"
        log.write_text(json.dumps(row)+"\n");progress["sample_log_sha256"]=io.sha(log)
        with self.assertRaisesRegex(ValueError,"progress mismatch"):ck.verify_associated_logs(progress,True)
    def test_formal_requires_log_at_save(self):
        g,e,o=self.components()
        with self.assertRaisesRegex(ValueError,"missing associated"):ck.save_epoch(self.root,g,o,1,{"config":{"scope":"formal"}},{})
        self.assertFalse((self.root/"checkpoints/epoch_0001").exists())
    def test_partial_path_or_SHA_pair_rejected(self):
        for progress in ({"sample_log_path":"a"},{"sample_log_sha256":"b"},{"samples":1}):
            with self.subTest(progress=progress),self.assertRaisesRegex(ValueError,"missing associated"):
                ck.verify_associated_logs(progress)
    def test_legacy_generic_logless_snapshot_still_supported(self):
        g,e,o=self.components();ck.save_epoch(self.root,g,o,1,{}, {})
        self.assertEqual(ck.load_latest(self.root,{},g,o)["next_epoch"],2)
    def test_path_escape_rejected(self):
        g,o,c,p,log,progress=self.setup_checkpoint();progress["sample_log_path"]="/etc/hostname"
        with self.assertRaisesRegex(ValueError,"G-v2 outputs"):ck.verify_associated_logs(progress,True)
    def test_wrong_order_counts_or_rounds_rejected(self):
        _,_,_,_,_,p=self.setup_checkpoint()
        for field,value in (("sample_order",["wrong"]),("samples",2),("optimizer_steps",2),("epoch",2)):
            bad=copy.deepcopy(p);bad[field]=value
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"progress mismatch"):ck.verify_associated_logs(bad,True)
    def test_supplementary_log_SHA_checked(self):
        _,_,_,_,_,p=self.setup_checkpoint();extra=self.root/"rounds.jsonl";extra.write_text("finite")
        p["supplementary_log_sha256"]={str(extra):io.sha(extra)}
        ck.verify_associated_logs(p,True);extra.write_text("tampered")
        with self.assertRaisesRegex(ValueError,"SHA"):ck.verify_associated_logs(p,True)
    def test_missing_supplementary_log_rejected(self):
        _,_,_,_,_,p=self.setup_checkpoint();p["supplementary_log_sha256"]={str(self.root/"missing"):"x"}
        with self.assertRaisesRegex(ValueError,"supplementary log missing"):ck.verify_associated_logs(p,True)

class ControlledEntryTests(unittest.TestCase):
    def test_persistent_target10_execution_limit1(self):
        cfg=runner.formal_configuration();seen=[]
        def fake(cache,out,config,**kwargs):seen.append((config["epochs"],kwargs["stop_after_epochs"]));return "one"
        self.assertEqual(runner.launch_once(fake,"cache","out",cfg),"one")
        self.assertEqual(seen,[(10,1)]);self.assertEqual(cfg["epochs"],10)
    def test_frozen_hyperparameters(self):
        c=runner.formal_configuration()
        self.assertEqual(c["architecture"],{"width":32,"blocks":4,"epsilon":8/255})
        self.assertEqual(c["optimizer"],{"lr":2e-4,"betas":[.5,.999]})
        self.assertEqual(c["weights"],{"lambda_global":1.,"lambda_local":1.,"lambda_retain":1.})
        self.assertEqual((c["seed"],c["epochs"],c["scope"]),(42,10,"formal"))
    def test_finite_statistics_and_zero_gradient_allowed(self):
        rows=[]
        for i,phase in enumerate(("attribute","joint")):
            r={key:.1 for key in FIELDS}
            r.update(phase=phase,K=1,sample_position=1,G_gradient_l2=0.,changed_parameter_count=0)
            rows.append(r)
        s=summarize(rows)
        self.assertEqual(s["all"]["rounds"],2);self.assertEqual(s["all"]["zero_G_gradient_rounds"],2)
        self.assertEqual(s["K1_joint"]["rounds"],1);self.assertIsNone(s["K7"]["global"])
    def test_nonfinite_metrics_rejected(self):
        r={key:1. for key in FIELDS};r["global"]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):stats([r])
    def test_validation_updates_rejected(self):
        with self.assertRaises(ValueError):validation_stats([{"optimizer_steps":1}])
    def test_finite_Adam_state(self):
        class Opt:state={0:{"exp_avg":torch.tensor(float("inf"))}}
        with self.assertRaisesRegex(ValueError,"Adam"):runner.finite_optimizer(Opt())
    def test_CPU_not_initialized_CUDA(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
