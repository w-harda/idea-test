"""仅CPU合成数据；真实GPU只由显式Smoke命令执行。"""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from eval_common import *
import eval_protocol as protocol
import eval_storage as storage
import eval_metrics as em
import evaluate as runner
from eval_backend import ForwardBudget,RealBackend
torch.set_num_threads(2)

def dataset():
 gs=[{"index":i,"row_id":f"cuhk:{i}:0","pid":i//2,"image":f"{i}.jpg","image_sha256":"fake"} for i in range(10)]
 qs=[{"index":i,"row_id":f"cuhk:{i}:0","pid":i//2,"caption":f"Original caption {i}.","caption_index":0,"paired_index":i} for i in range(10)]
 return gs,qs
def fake_protocol():
 gs,qs=dataset()
 return {"mode":"smoke","gallery":gs,"queries":qs,"gallery_count":10,"query_count":10,
 "image_batch":2,"text_batch":2,"reuse_clean_reference":False,"source_sha256":{},
 "checkpoint_sha256":{"1":"a","3":"b","10":"c"},"expected_full_clean":{}}
def reference():
 x=torch.zeros(10,512)
 for i in range(10):x[i,i//2]=1
 return x,x.clone()

class FakeBackend:
 totals=[]
 def __init__(self,p,b):self.b=b;self.gs,self.qs=dataset();self.closed=False
 def text(self,rows):
  self.b.consume("text",len(rows));FakeBackend.totals.extend(("text",r["row_id"]) for r in rows)
  return reference()[1][[r["index"] for r in rows]],{"text_samples":len(rows),"G_samples":0,"IRRA_image_samples":0}
 def image(self,family,rows):
  self.b.consume(family,len(rows))
  if family!="clean":self.b.consume("G_"+family,len(rows))
  FakeBackend.totals.extend((family,r["row_id"]) for r in rows)
  return reference()[0][[r["index"] for r in rows]],{"text_samples":0,"G_samples":0 if family=="clean" else len(rows),
   "IRRA_image_samples":len(rows),"linfs":[0. if family=="clean" else .01 for r in rows]}
 def finish(self):self.closed=True;return {"fake_CPU":True}

class Base(unittest.TestCase):
 def setUp(self):
  EVAL_OUTPUTS.mkdir(parents=True,exist_ok=True);self.tmp=tempfile.TemporaryDirectory(prefix="cpu_",dir=EVAL_OUTPUTS)
  self.out=Path(self.tmp.name);FakeBackend.totals=[]
 def tearDown(self):self.tmp.cleanup()
 def run_fake(self,p=None,factory=FakeBackend,stop=None):
  return runner.run_pipeline(p or fake_protocol(),self.out,backend_factory=factory,reference=reference(),stop=stop)

class PermissionTests(unittest.TestCase):
 def test_smoke_fixed10(self):
  self.assertEqual(protocol.authorize("smoke","run",10,10),(10,10))
 def test_oversize_smoke_rejected(self):
  for g,q in ((11,10),(10,11),(100,100),(3074,6156),(9,10)):
   with self.subTest(g=g,q=q),self.assertRaises(ValueError):protocol.authorize("smoke","run",g,q)
 def test_user_modes_need_separate_explicit_flags(self):
  with self.assertRaises(ValueError):protocol.authorize("diagnostic100","run",100,100)
  with self.assertRaises(ValueError):protocol.authorize("full","run",3074,6156,True,False)
  self.assertEqual(protocol.authorize("diagnostic100","run",100,100,True),(100,100))
  self.assertEqual(protocol.authorize("full","run",3074,6156,False,True),(3074,6156))
 def test_large_audit_also_manual_only(self):
  with self.assertRaises(ValueError):protocol.authorize("full","audit",3074,6156)
 def test_CPU_preflight_not_authorized_gpu(self):
  protocol.authorize("full","preflight",3074,6156)
 def test_large_main_without_flag_rejected_before_preflight(self):
  with patch.object(runner,"preflight",side_effect=AssertionError("must not enter")):
   with self.assertRaises(ValueError):runner.main(["--mode","full","--action","run","--gallery-count","3074","--query-count","6156"])
 def test_forward_sample_caps_not_only_batch_calls(self):
  b=ForwardBudget(10,10)
  with torch.inference_mode():
   b.consume("epoch1",10)
   with self.assertRaises(ValueError):b.consume("epoch1",1)
   with self.assertRaises(ValueError):b.consume("text",11)
  self.assertEqual(b.calls["epoch1"],1)
 def test_forward_guard_requires_inference(self):
  with self.assertRaises(ValueError):ForwardBudget(10,10).consume("clean",1)
 def test_real_backend_guard_before_model_loading(self):
  p=fake_protocol();p["gallery_count"]=11
  with patch("eval_backend.adapter.load_official",side_effect=AssertionError("GPU forbidden")):
   with self.assertRaises(ValueError):RealBackend(p,ForwardBudget(11,10))
 def test_no_CUDA(self):self.assertFalse(torch.cuda.is_initialized())
 def test_output_cannot_target_training(self):
  with self.assertRaises(ValueError):safe(TRAIN/"bad_eval")

class MetricTests(unittest.TestCase):
 def test_analytic_multi_positive_AP_INP_CMC(self):
  q=torch.zeros(1,512);q[0,0]=1
  g=torch.zeros(10,512)
  scores=torch.tensor([.9,.8,.7,.6,.5,.4,.3,.2,.1,.0])
  g[:,0]=scores;g[:,1]=(1-scores.square()).sqrt()
  result,_=em.metrics(q,g,[5],[0,5,0,5,0,0,0,0,0,0])
  self.assertAlmostEqual(result["mAP"],50.,places=5);self.assertAlmostEqual(result["mINP"],50.,places=5)
  self.assertEqual((result["R@1"],result["R@5"],result["R@10"]),(0.,100.,100.))
 def test_official_rank_exact_same_scores_including_ties(self):
  q,g=reference();ids=[i//2 for i in range(10)]
  result,sim=em.metrics(q,g,ids,ids)
  cmc,ap,inp,_=em.official_rank()(sim,torch.tensor(ids),torch.tensor(ids))
  self.assertEqual(result["mAP"],float(ap));self.assertEqual(result["mINP"],float(inp))
  self.assertEqual(result["R@1"],float(cmc[0]))
 def test_missing_positive_rejected(self):
  q,g=reference()
  with self.assertRaises(ValueError):em.metrics(q,g,[99]*10,list(range(10)))
 def test_nonunit_features_rejected(self):
  with self.assertRaises(ValueError):em.normalized(torch.ones(1,512),1)
 def test_nonfinite_feature_rejected(self):
  t=reference()[0];t[0,0]=float("nan")
  with self.assertRaises(ValueError):em.normalized(t,10)
 def test_drop_zero_null_and_negative(self):
  self.assertIsNone(em.drops({k:0 for k in METRICS},{k:1 for k in METRICS})["mAP"])
  self.assertEqual(em.drops({k:10 for k in METRICS},{k:11 for k in METRICS})["mAP"],-10)
 def test_clean_mismatch_rejected_no_relaxation(self):
  with self.assertRaises(ValueError):em.check_clean({k:1 for k in METRICS},{k:2 for k in METRICS})

class IdentityTests(unittest.TestCase):
 def test_official_pixel_preprocessing_exact(self):
  from PIL import Image
  import irra_image_adapter as adapter
  _,_,build=official_imports()
  im=Image.new("RGB",(170,390),(123,78,211))
  x=adapter.pixels_from_pil(im)
  official=build(img_size=(384,128),is_train=False)(im)
  actual=(x-torch.tensor(adapter.MEAN)[:,None,None])/torch.tensor(adapter.STD)[:,None,None]
  self.assertTrue(torch.equal(actual,official))

 def test_subset_10_legal_pairs_deterministic(self):
  gs,qs=dataset()
  a,b=protocol.subset(gs,qs,10,42)
  self.assertEqual((len(a),len(b)),(10,10));self.assertEqual((a,b),protocol.subset(gs[::-1],qs,10,42))
  self.assertEqual([q["caption"] for q in b],[q["caption"] for q in qs])
  self.assertTrue(all(sum(g["pid"]==q["pid"] for g in a)==2 for q in b))
 def test_duplicate_and_unsafe_images_rejected(self):
  raw=[{"split":"test","file_path":"a.jpg","id":1,"captions":["x"]}]*2
  with self.assertRaises(ValueError):protocol.layout(raw)
  raw[0]={**raw[0],"file_path":"../../escape"}
  with self.assertRaises(ValueError):protocol.layout(raw[:1])
 def test_all_captions_pid_and_order_preserved(self):
  raw=[{"split":"train"},{"split":"test","file_path":"a.jpg","id":12,"captions":["A!","B?","C."]}]
  gs,qs=protocol.layout(raw)
  self.assertEqual([q["row_id"] for q in qs],["cuhk:1:0","cuhk:1:1","cuhk:1:2"])
  self.assertEqual([q["caption"] for q in qs],raw[1]["captions"]);self.assertEqual(gs[0]["pid"],12)
 def test_no_caption_silent_skip(self):
  with self.assertRaises(ValueError):protocol.layout([{"split":"test","file_path":"a.jpg","id":1,"captions":[]}])

class CacheTests(Base):
 def test_round_trip_atomic_sha(self):
  p=fake_protocol();rows=p["gallery"][:2];feature=reference()[0][:2]
  storage.begin_unit(self.out,"clean",0);storage.publish(self.out,"clean",0,p,rows,feature,{"linfs":[0,0],"text_samples":0,"G_samples":0,"IRRA_image_samples":2})
  v=storage.load_unit(self.out,"clean",0,p,rows);self.assertTrue(torch.equal(v["features"],feature))
 def test_corrupt_feature_refused(self):
  p=fake_protocol();rows=p["gallery"][:2]
  storage.publish(self.out,"clean",0,p,rows,reference()[0][:2],{"linfs":[0,0],"text_samples":0,"G_samples":0,"IRRA_image_samples":2})
  (storage.unit_path(self.out,"clean",0)/"features.pt").write_bytes(b"corrupt")
  with self.assertRaisesRegex(ValueError,"SHA"):storage.load_unit(self.out,"clean",0,p,rows)
 def test_protocol_or_identity_mismatch_rejected(self):
  p=fake_protocol();rows=p["gallery"][:2];storage.publish(self.out,"clean",0,p,rows,reference()[0][:2],{"linfs":[0,0],"text_samples":0,"G_samples":0,"IRRA_image_samples":2})
  bad=copy.deepcopy(p);bad["checkpoint_sha256"]["1"]="wrong"
  with self.assertRaises(ValueError):storage.load_unit(self.out,"clean",0,bad,rows)
  with self.assertRaises(ValueError):storage.load_unit(self.out,"clean",0,p,rows[::-1])
 def test_immutable_no_overwrite(self):
  p=fake_protocol();rows=p["gallery"][:2];storage.publish(self.out,"clean",0,p,rows,reference()[0][:2],{"linfs":[0,0],"text_samples":0,"G_samples":0,"IRRA_image_samples":2})
  with self.assertRaises(ValueError):storage.publish(self.out,"clean",0,p,rows,reference()[0][:2],{"linfs":[0,0],"text_samples":0,"G_samples":0,"IRRA_image_samples":2})
 def test_uncommitted_claim_prevents_automatic_duplicate_forward(self):
  storage.begin_unit(self.out,"epoch1",0)
  with self.assertRaisesRegex(ValueError,"uncommitted"):storage.begin_unit(self.out,"epoch1",0)
 def test_pending_not_read_as_published(self):
  parent=storage.unit_path(self.out,"clean",0).parent;parent.mkdir(parents=True);(parent/".pending-crash").mkdir()
  self.assertIsNone(storage.load_unit(self.out,"clean",0,fake_protocol(),[]))
 def test_cached_budget_and_forward_count_validation(self):
  stats={"linfs":[.04],"text_samples":0,"G_samples":1,"IRRA_image_samples":1}
  with self.assertRaises(ValueError):storage.validate_stats("epoch1",stats,1)
  stats["linfs"]=[.01];stats["G_samples"]=0
  with self.assertRaises(ValueError):storage.validate_stats("epoch1",stats,1)
 def test_concurrent_writer_rejected(self):
  with io.output_lock(self.out):
   with self.assertRaises(ValueError):
    with io.output_lock(self.out):pass

class PipelineTests(Base):
 def test_four_groups_clean_text_once_and_exact_counts(self):
  s=self.run_fake()
  self.assertEqual((s["gallery_count"],s["query_count"]),(10,10));self.assertFalse(s["formal_test"])
  for family in ("text",*GROUPS):self.assertEqual(sum(k==family for k,r in FakeBackend.totals),10)
  for family in GROUPS[1:]:self.assertEqual(s["group_stats"][family]["G_samples"],10)
  self.assertEqual(s["optimizer_steps"],0)
 def test_complete_reentry_no_backend_or_rewriting(self):
  self.run_fake();files={str(p):io.sha(p) for p in (self.out/"features").rglob("*") if p.is_file()}
  self.run_fake(factory=lambda *a:self.fail("model loaded"));io.verify(files)
 def test_graceful_stop_resume_no_completed_sample_repeated(self):
  stop=runner.SignalBoundary()
  class StopBackend(FakeBackend):
   def image(self,family,rows):
    result=super().image(family,rows)
    if family=="epoch1":stop.request()
    return result
  with self.assertRaises(runner.SafeStop):self.run_fake(factory=StopBackend,stop=stop)
  before=list(FakeBackend.totals);self.run_fake()
  self.assertEqual(len(FakeBackend.totals),50);self.assertEqual(len(set(FakeBackend.totals)),50)
  self.assertTrue(list((self.out/"attempts").glob("*/interruption.json")))
 def test_clean_gate_precedes_any_G(self):
  p=fake_protocol();p["mode"]="full";p["expected_full_clean"]={k:0. for k in METRICS}
  with self.assertRaises(ValueError):self.run_fake(p)
  self.assertFalse(any(k.startswith("epoch") for k,r in FakeBackend.totals));self.assertTrue((self.out/"clean_mismatch.json").exists())
 def test_feature_reference_mismatch_stops(self):
  class Bad(FakeBackend):
   def text(self,rows):
    v,s=super().text(rows);return v.roll(1,1),s
  with self.assertRaises(ValueError):self.run_fake(factory=Bad)
  self.assertFalse(any(k=="clean" for k,r in FakeBackend.totals))
 def test_changed_protocol_rejected_without_new_calls(self):
  self.run_fake();before=len(FakeBackend.totals);p=fake_protocol();p["image_batch"]=1
  with self.assertRaises(ValueError):self.run_fake(p)
  self.assertEqual(len(FakeBackend.totals),before)
 def test_reused_reference_no_clean_or_text_encode(self):
  p=fake_protocol();p["reuse_clean_reference"]=True
  self.run_fake(p)
  self.assertFalse(any(k in ("text","clean") for k,r in FakeBackend.totals))
 def test_readonly_metric_audit_keeps_all_files(self):
  p=fake_protocol();self.run_fake(p)
  hashes={str(x):io.sha(x) for x in self.out.rglob("*") if x.is_file()}
  runner.evaluate_cached(p,self.out,reference(),write=False);io.verify(hashes)
 def test_cached_metrics_reproduce(self):
  p=fake_protocol();s=self.run_fake(p)
  again=runner.evaluate_cached(p,self.out,reference());self.assertEqual(s,again)

if __name__=="__main__":unittest.main()
