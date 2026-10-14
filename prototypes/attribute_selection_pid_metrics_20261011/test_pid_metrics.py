"""仅CPU：多正例AP、score回填、历史数值核验、跨victim口径/协议门禁。"""
import copy
import math
import unittest
from unittest.mock import patch
import torch
from pid_metrics import MODES, replace_paired, check_saved, pid_row, drop_rate, aggregate_victims
from inputs import verify, canonical_sha

def report(victim="a", clean=80, adv=40):
    return {"victim_id": victim, "aggregation_contract": {"query_set_sha256":"same",
        "gallery_layout_sha256":"same","attack_set_id":"same","protocol_kind":"paired_replacement_PID_diagnostic"},
        "clean":{"mAP":clean},"adversarial":{m:{"mAP":adv} for m in MODES}}

def saved(scores, index):
    other=torch.cat((scores[:index],scores[index+1:]))
    return {"paired_soft_rank_F":float(1+torch.sigmoid((other-scores[index])/.02).sum()),
        "positive_image_rank":1+int((other>scores[index]).sum()),
        "positive_image_rank_tie_count":int((other==scores[index]).sum())}

class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.s=torch.tensor([.9,.8,.7,.6],dtype=torch.float32)
        self.pid=torch.tensor([1,2,1,3])

    def test_all_pid_positives(self):
        r=pid_row(self.s,1,self.pid)
        self.assertAlmostEqual(r["AP"],5/6,places=6)
        self.assertEqual(r["positive_images"],2)

    def test_lower_paired_not_whole_pid(self):
        s=replace_paired(self.s,0,.1)
        r=pid_row(s,1,self.pid)
        self.assertAlmostEqual(r["AP"],.5)
        self.assertEqual(r["first_hit_rank"],2)
        self.assertEqual(saved(s,0)["positive_image_rank"],4)

    def test_single_positive(self):
        self.assertAlmostEqual(pid_row(self.s,2,self.pid)["AP"],.5)

    def test_no_camera_exclusion(self):
        self.assertEqual(pid_row(self.s,1,self.pid)["positive_images"],2)

    def test_absent_pid_rejected(self):
        with self.assertRaises(ValueError):pid_row(self.s,99,self.pid)

    def test_nonpaired_unchanged(self):
        t=replace_paired(self.s,1,.01)
        self.assertTrue(torch.equal(t[[0,2,3]],self.s[[0,2,3]]))
        self.assertTrue(torch.equal(self.s,torch.tensor([.9,.8,.7,.6])))

    def test_noop_replacement_exact(self):
        self.assertTrue(torch.equal(replace_paired(self.s,0,float(self.s[0])),self.s))

    def test_invalid_index(self):
        with self.assertRaises(ValueError):replace_paired(self.s,4,.1)

    def test_invalid_cosine(self):
        for v in [float("nan"),float("inf"),1.2]:
            with self.subTest(v=v),self.assertRaises(ValueError):replace_paired(self.s,0,v)

    def test_nonfinite_scores(self):
        s=self.s.clone();s[0]=float("nan")
        with self.assertRaises(ValueError):replace_paired(s,0,.1)

    def test_reconstruction_exact(self):
        self.assertEqual(check_saved(self.s,0,saved(self.s,0))["soft_rank_abs_error"],0)

    def test_rank_mismatch_rejected(self):
        x=saved(self.s,0);x["positive_image_rank"]+=1
        with self.assertRaises(ValueError):check_saved(self.s,0,x)

    def test_soft_rank_mismatch_rejected(self):
        x=saved(self.s,0);x["paired_soft_rank_F"]+=.1
        with self.assertRaises(ValueError):check_saved(self.s,0,x)

    def test_tie_count_exact(self):
        s=torch.tensor([.5,.5,.2])
        x=saved(s,0)
        self.assertEqual(x["positive_image_rank"],1)
        self.assertEqual(x["positive_image_rank_tie_count"],1)
        self.assertTrue(check_saved(s,0,x)["paired_ties_exact"])

    def test_single_model_drop(self):
        self.assertEqual(drop_rate(80,40),50)

    def test_negative_drop_not_clamped(self):
        self.assertEqual(drop_rate(40,80),-100)

    def test_zero_clean_undefined(self):
        self.assertIsNone(drop_rate(0,0))

    def test_invalid_map(self):
        with self.assertRaises(ValueError):drop_rate(float("nan"),2)
        with self.assertRaises(ValueError):drop_rate(101,2)

    def test_single_victim_aap(self):
        a=aggregate_victims([report()])
        self.assertEqual(a["aAP_clean"],80)
        self.assertEqual(a["modes"][MODES[0]]["aAP_adversarial"],40)
        self.assertEqual(a["modes"][MODES[0]]["mDR"],50)
        self.assertTrue(a["single_victim"])

    def test_multivictim_paper_formula_not_mean_dr(self):
        a=aggregate_victims([report("a",80,40),report("b",20,20)])
        self.assertEqual(a["aAP_clean"],50)
        self.assertEqual(a["modes"][MODES[0]]["aAP_adversarial"],30)
        self.assertEqual(a["modes"][MODES[0]]["mDR"],40)
        self.assertEqual(a["modes"][MODES[0]]["mean_per_victim_DR"],25)

    def test_duplicate_victim_rejected(self):
        with self.assertRaises(ValueError):aggregate_victims([report(),report()])

    def test_protocol_changes_rejected(self):
        for key in ["query_set_sha256","gallery_layout_sha256","attack_set_id","protocol_kind"]:
            b=report("b");b["aggregation_contract"][key]="different"
            with self.subTest(key=key),self.assertRaises(ValueError):aggregate_victims([report(),b])

    def test_no_victims_rejected(self):
        with self.assertRaises(ValueError):aggregate_victims([])

    def test_source_sha_change_rejected(self):
        with patch("inputs.digest",return_value="actual"):
            with self.assertRaises(ValueError):verify({"source":"expected"})

    def test_source_sha_exact(self):
        with patch("inputs.digest",return_value="actual"):verify({"source":"actual"})

    def test_query_identity_hash_order_sensitive(self):
        self.assertNotEqual(canonical_sha(["a","b"]),canonical_sha(["b","a"]))
        self.assertEqual(canonical_sha({"a":1,"b":2}),canonical_sha({"b":2,"a":1}))

if __name__=="__main__":unittest.main()
