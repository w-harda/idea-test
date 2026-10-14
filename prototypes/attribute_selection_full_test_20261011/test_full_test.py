"""纯CPU测试：完整test覆盖、同K选择、真实共享图库PID指标与安全门禁。"""
import unittest,tempfile,copy,math,sys,random
from pathlib import Path
from unittest.mock import patch
import core as c
import run
def raw():
 return [{"split":"train","id":1,"file_path":"t.jpg","captions":["train"]},
 {"split":"test","id":2,"file_path":"a.jpg","captions":["first","second"]},
 {"split":"test","id":2,"file_path":"b.jpg","captions":["third","fourth"]}]
def payload(k=2,m=4):
 return {"caption_id":"cuhk:1:0","natural_k":k,"candidate_count":m,
 "ranked_attributes":[{"instance_id":str(i),"S_raw":float(m-i)} for i in range(m)]}
class ProtocolTests(unittest.TestCase):
 def test_all_images_queries(self):
  m=c.manifest(raw());self.assertEqual((m["gallery_count"],m["query_count"]),(2,4))
 def test_first_caption_only_condition(self):
  m=c.manifest(raw());self.assertEqual(m["gallery"][0]["caption"],"first");self.assertEqual(m["queries"][1]["caption"],"second")
 def test_annotation_order(self):
  m=c.manifest(raw());self.assertEqual([q["row_id"] for q in m["queries"]],["cuhk:1:0","cuhk:1:1","cuhk:2:0","cuhk:2:1"])
 def test_same_pid_distinct_images(self):
  m=c.manifest(raw());self.assertEqual([g["pid"] for g in m["gallery"]],[2,2])
 def test_duplicate_image_rejected(self):
  r=raw();r[2]["file_path"]="a.jpg"
  with self.assertRaises(ValueError):c.manifest(r)
 def test_empty_caption_rejected(self):
  r=raw();r[1]["captions"]=[""]
  with self.assertRaises(ValueError):c.manifest(r)
 def test_no_test_rejected(self):
  with self.assertRaises(ValueError):c.manifest(raw()[:1])
 def test_escape_image_rejected(self):
  r=raw();r[1]["file_path"]="../../../../other"
  with self.assertRaises(ValueError):c.manifest(r)
 def test_all_modes_same_K(self):
  for k in range(1,8):
   r=c.selected(payload(k,8));self.assertEqual({len(x) for x in r["selection_ids"].values()},{k})
 def test_modes_sorted_by_raw(self):
  r=c.selected(payload());ids=r["selection_ids"]
  self.assertEqual(ids["S_TOPK"],["0","1"]);self.assertEqual(ids["S_BOTTOMK"],["2","3"])
  self.assertEqual(ids["RANDOM_K"],sorted(ids["RANDOM_K"],key=int))
 def test_without_replacement(self):
  r=c.selected(payload(6,8));self.assertEqual(len(set(r["selection_ids"]["RANDOM_K"])),6)
 def test_random_reproducible(self):
  self.assertEqual(c.selected(payload()),c.selected(payload()))
 def test_selection_no_attack_rng(self):
  random.seed(123);before=random.getstate();c.selected(payload());self.assertEqual(before,random.getstate())
 def test_m_eq_K_kept(self):
  r=c.selected(payload(4,4));self.assertEqual(len({tuple(x) for x in r["selection_ids"].values()}),1)
 def test_K0_passthrough(self):
  r=c.selected(payload(0,0));self.assertTrue(all(v==[] for v in r["selection_ids"].values()))
 def test_rank_rejected(self):
  r=payload();r["ranked_attributes"].reverse()
  with self.assertRaises(ValueError):c.selected(r)
 def test_duplicates_rejected(self):
  r=payload();r["ranked_attributes"][1]["instance_id"]="0"
  with self.assertRaises(ValueError):c.selected(r)
 def test_invalid_K(self):
  with self.assertRaises(ValueError):c.selected(payload(5,4))
 def test_inventory_not_drop_all_selected(self):
  rows=[c.selected(payload(0,0)),c.selected(payload(4,4)),c.selected(payload())]
  x=c.inventory(rows);self.assertEqual(x["image_count"],3);self.assertEqual(x["K0_clean_passthrough"],1);self.assertEqual(x["m_gt_K"],1)
 def test_smoke_scope_no_formal(self):
  m=c.manifest(raw());s=run.scope_manifest(m,"smoke");self.assertFalse(s["all_test_captions_included"])
 def test_full_scope_exact(self):
  m=c.manifest(raw());self.assertIs(run.scope_manifest(m,"full"),m)
 def test_incomplete_gate(self):
  m=c.manifest(raw())
  with self.assertRaises(ValueError):c.require_complete({"scope":"full","method_audit_passed":True,"gallery_count":1,"query_count":4},m,"full")
 def test_full_flag_gate(self):
  with self.assertRaises(SystemExit):run.main(["--stage","all","--scope","full"])
 def test_default_preflight(self):
  args=run.parser().parse_args([]);self.assertEqual(args.stage,"preflight");self.assertEqual(args.scope,"smoke")
 def test_identity_sensitive(self):
  self.assertNotEqual(c.identity(["a","b"]),c.identity(["b","a"]))
 def test_sha_guard(self):
  with tempfile.TemporaryDirectory(dir="/home/lzf/ldx") as d:
   p=Path(d)/"x";p.write_text("a");locked={str(p):c.sha(p)};c.verify(locked);p.write_text("b")
   with self.assertRaises(ValueError):c.verify(locked)
 def test_safe_write(self):
  with self.assertRaises(ValueError):c.safe("/tmp/not-allowed")
 def test_budget_config_static(self):
  src=Path(run.__file__).read_text()
  self.assertIn('"epsilon":8/255',src);self.assertIn('"steps_per_round":10',src);self.assertIn('"score_weighting":False',src)
class MetricTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  sys.path.insert(0,str(c.REPO/"src"));sys.path.insert(0,str(c.REPO/"prototypes/attribute_selection_pid_metrics_20261011"))
  import torch
  from attributes.tbps_retrieval_metrics import text_to_image_metrics
  from pid_metrics import aggregate_victims
  cls.torch=torch;cls.metrics=staticmethod(text_to_image_metrics);cls.aggregate=staticmethod(aggregate_victims)
 def test_all_same_pid_positive(self):
  out,rows=self.metrics([[.9,.1,.8]],[1],[1,1,2])
  self.assertEqual(rows[0]["positive_images"],2);self.assertAlmostEqual(out["mAP"],(1+2/3)/2*100,places=5)
 def test_shared_gallery_for_all_queries(self):
  q=self.torch.tensor([[1.,0.],[0.,1.]]);g=self.torch.tensor([[0.,1.],[1.,0.]])
  out,_=self.metrics(q@g.T,[1,2],[1,2]);self.assertAlmostEqual(out["mAP"],50.)
 def test_no_positive_fail(self):
  with self.assertRaises(ValueError):self.metrics([[1.,0.]],[3],[1,2])
 def test_single_victim_aAP_mDR(self):
  r={"victim_id":"IRRA","aggregation_contract":{"same":True},"clean":{"mAP":80.},"adversarial":{m:{"mAP":40.} for m in c.MODES}}
  out=self.aggregate([r]);self.assertEqual(out["modes"]["S_TOPK"]["aAP_adversarial"],40.);self.assertEqual(out["modes"]["S_TOPK"]["mDR"],50.)
 def test_multivictim_ratio_of_mean(self):
  a={"victim_id":"a","aggregation_contract":{},"clean":{"mAP":80.},"adversarial":{m:{"mAP":40.} for m in c.MODES}}
  b={"victim_id":"b","aggregation_contract":{},"clean":{"mAP":40.},"adversarial":{m:{"mAP":30.} for m in c.MODES}}
  x=self.aggregate([a,b]);self.assertAlmostEqual(x["modes"]["S_TOPK"]["mDR"],(60-35)/60*100)
 def test_quantization_not_used(self):
  source=(c.ROOT/"runtime.py").read_text()+(c.ROOT/"run.py").read_text();self.assertNotIn("Image.fromarray",source);self.assertNotIn("save_image",source)
 def test_float_archive_exact(self):
  with tempfile.TemporaryDirectory(dir="/home/lzf/ldx") as d:
   x=self.torch.tensor([.1234567],dtype=self.torch.float32);p=Path(d)/"x.pt";self.torch.save(x,p)
   self.assertTrue(self.torch.equal(x,self.torch.load(p,weights_only=True)))
if __name__=="__main__":unittest.main()
