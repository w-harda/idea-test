"""CPU测试：official PID指标、Test正例/缓存绑定、共享图库、单前向及门禁。"""
import copy
import json
import unittest
from unittest.mock import patch
import torch
import g_test_core as g
import g_test_evaluate as run
from generator_model import ConditionalGenerator,generate_once
torch.set_num_threads(2)

class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.row={"index":0,"row_id":"cuhk:10:0","pid":3,"image":"a.jpg","caption":"a"}
        self.protocol={"scope":"smoke"}
        self.source={"score_sha256":"score","score_path":"score.json","image_sha256":"image",
                     "natural_k":1,"selection_ids":{"S_TOPK":["id"]}}
        self.entry={**self.row,"tensor_sha256":"tensor","score_sha256":"score","image_sha256":"image","K":1}
        self.meta={**self.row,"tensor_sha256":"tensor","score_sha256":"score","score_path":"score.json",
                   "image_sha256":"image","protocol_sha256":g.c.identity(self.protocol),
                   "selected_instance_ids":["id"],"score_normalized_weights":[1.],"source_scope":"official test score cache"}
        self.condition={"has_local":True}
    def bind(self):g.validate_condition_binding(self.entry,self.meta,self.protocol,self.row,self.source,self.condition)
    def test_valid_test_binding(self):self.bind()
    def test_wrong_PID_rejected(self):
        self.entry["pid"]=99
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_caption_rejected(self):
        self.meta["caption"]="another"
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_source_image_rejected(self):
        self.entry["image"]="wrong.jpg"
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_score_gallery_cache_rejected(self):
        self.meta["score_sha256"]="train score"
        with self.assertRaises(ValueError):self.bind()
    def test_train_condition_rejected(self):
        self.meta["source_scope"]="train"
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_TopK_rejected(self):
        self.meta["selected_instance_ids"]=["another"]
        with self.assertRaises(ValueError):self.bind()
    def test_wrong_K_rejected(self):
        self.entry["K"]=2
        with self.assertRaises(ValueError):self.bind()
    def test_weight_sum_rejected(self):
        self.meta["score_normalized_weights"]=[.5]
        with self.assertRaises(ValueError):self.bind()
    def test_tensor_binding_rejected(self):
        self.meta["tensor_sha256"]="modified"
        with self.assertRaises(ValueError):self.bind()
    def test_K0_global_only_binding(self):
        self.entry["K"]=0;self.source["natural_k"]=0
        self.source["selection_ids"]["S_TOPK"]=[];self.meta["selected_instance_ids"]=[]
        self.meta["score_normalized_weights"]=[];self.condition["has_local"]=False;self.bind()
    def test_K0_nonzero_local_rejected(self):
        self.entry["K"]=0;self.source["natural_k"]=0
        self.source["selection_ids"]["S_TOPK"]=[];self.meta["selected_instance_ids"]=[]
        self.meta["score_normalized_weights"]=[]
        with self.assertRaises(ValueError):self.bind()
    def test_smoke_deterministic_two_PID(self):
        gallery=[{"index":i,"pid":pid} for i,pid in enumerate([5,5,6,7])]
        m={"gallery":gallery,"queries":[{"paired_index":i} for i in range(4)]}
        x=g.scoped(m,"smoke")
        self.assertEqual([r["index"] for r in x["gallery"]],[0,2]);self.assertEqual(x["query_count"],2)
    def test_full_keeps_every_image(self):
        m={"gallery":[{"index":i} for i in range(5)]}
        self.assertIs(g.scoped(m,"full"),m)
    def test_missing_cached_test_caption_rejected(self):
        with self.assertRaises(ValueError):g.condition_entry({"entries":[]},"cuhk:1:0")
    def test_full_requires_manual_flag_before_audit(self):
        with patch.object(g,"audit",side_effect=AssertionError("must not start")):
            with self.assertRaises(SystemExit):run.main(["--scope","full","--stage","all"])
    def test_default_preflight_no_GPU_or_output(self):
        with patch.object(g,"audit",return_value=({},None,None,None)),patch.object(run,"prepare",side_effect=AssertionError),patch.object(run.ck,"training_lock",side_effect=AssertionError):
            run.main([])
    def test_clean_match(self):
        x={k:1. for k in ("mAP","R@1","R@5","R@10","mINP")}
        self.assertTrue(all(v==0 for v in g.assert_clean_match(x,x).values()))
    def test_clean_mismatch_rejected(self):
        x={k:1. for k in ("mAP","R@1","R@5","R@10","mINP")};y={**x,"mAP":1.01}
        with self.assertRaises(ValueError):g.assert_clean_match(x,y)

class TestRetrieval(unittest.TestCase):
    def test_PID_multi_positive_AP_INP(self):
        summary,rows=g.text_to_image_metrics(torch.tensor([[.9,.8,.7]]),[1],[1,2,1])
        self.assertAlmostEqual(rows[0]["AP"],(1+2/3)/2,places=6)
        self.assertEqual(rows[0]["positive_images"],2);self.assertAlmostEqual(rows[0]["INP"],2/3,places=6)
    def test_missing_PID_positive_rejected(self):
        with self.assertRaises(ValueError):g.text_to_image_metrics(torch.tensor([[.9,.8]]),[3],[1,2])
    def test_gallery_PID_order_matters(self):
        _,a=g.text_to_image_metrics(torch.tensor([[.9,.8]]),[1],[1,2])
        _,b=g.text_to_image_metrics(torch.tensor([[.9,.8]]),[1],[2,1])
        self.assertEqual(a[0]["AP"],1);self.assertEqual(b[0]["AP"],.5)
    def test_mDR_percent(self):self.assertEqual(g.drop_rate(60,30),50)
    def test_shared_gallery_two_queries(self):
        q=torch.zeros(2,512);q[0,0]=1;q[1,1]=1
        f=q.clone()
        summary,rows=g.metrics_for(q,f,[{"pid":1},{"pid":2}],[{"pid":1},{"pid":2}])
        self.assertEqual(summary["mAP"],100);self.assertEqual(len(rows),2)

class TestGeneration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.image=torch.rand(1,3,32,16)
        self.model=ConditionalGenerator(width=4,blocks=1,condition_hidden=8).eval().requires_grad_(False)
        self.z=torch.nn.functional.normalize(torch.randn(1,512),dim=1);self.a=torch.randn(1,512)
    def archive(self):
        r=generate_once(self.model,self.image,self.z,self.a)
        feature=torch.zeros(1,512);feature[0,0]=1
        return {"pixels":r["adversarial"],"delta":r["delta"],"effective_delta":r["effective_delta"],"IRRA_feature":feature}
    def test_one_forward_no_backward_or_parameter_update(self):
        before={k:v.clone() for k,v in self.model.state_dict().items()};calls=[]
        h=self.model.register_forward_hook(lambda *args:calls.append(1))
        with patch("torch.autograd.grad",side_effect=AssertionError):self.archive()
        h.remove();self.assertEqual(len(calls),1)
        self.assertTrue(all(torch.equal(v,self.model.state_dict()[k]) for k,v in before.items()))
    def test_archive_valid_and_budget(self):self.assertLessEqual(g.validate_archive(self.archive(),self.image,{}),8/255+1e-6)
    def test_archive_effective_delta_rejected(self):
        x=self.archive();x["effective_delta"]=torch.ones_like(self.image)
        with self.assertRaises(ValueError):g.validate_archive(x,self.image,{})
    def test_archive_feature_rejected(self):
        x=self.archive();x["IRRA_feature"]=torch.zeros(1,512)
        with self.assertRaises(ValueError):g.validate_archive(x,self.image,{})
    def test_K0_still_runs_G(self):
        calls=[];h=self.model.register_forward_hook(lambda *args:calls.append(1))
        r=generate_once(self.model,self.image,self.z,torch.zeros_like(self.a));h.remove()
        self.assertEqual(len(calls),1);self.assertGreater(float(r["effective_delta"].abs().max()),0)
    def test_old_standalone_source_not_changed(self):
        old=json.loads((g.c.ROOT/"outputs/checkpoint_repair/before_sources.json").read_text())["generate.py"]
        self.assertEqual(old,(g.c.ROOT/"generate.py").read_text())

if __name__=="__main__":unittest.main()
