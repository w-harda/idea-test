"""H-v1 Phase 3 CPU测试：真实接口核心用fixture验证，不调用真实模型/GPU。"""
from dataclasses import replace,asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from test_phase2 import fixture,GMock
from training_inputs import TrainingCatalog
from tokenizer_probe import official_probe
from text_frontend import CaptionFrontend
from phase2_sources import sha,digest
from phase3_inputs import select_smoke,validate_selection,import_gv2
from real_feedback import InferenceRuntime
from real_feature_cache import RealFeatureCache
from feedback_backend import state_sha
from constraints import candidates,make_edit,replay
import smoke_phase3

class RetrievalFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.linspace(.1,1.,512))
        self.register_buffer("offset",torch.tensor(.1))
    def encode_image(self,x):
        return self.weight[None]+x.mean((1,2,3))[:,None]*self.weight.flip(0)[None]
    def encode_text(self,tokens):
        return self.weight[None]+((tokens.float().sum(1)%97)/100)[:,None]*self.weight.flip(0)[None]

class EncoderFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=RetrievalFixture()
        self.register_buffer("mean",torch.tensor(.4))
        self.register_buffer("std",torch.tensor(.2))
    def forward(self,x):
        return F.normalize(self.model.encode_image((x-self.mean)/self.std),dim=1)

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe=official_probe()
        cls.inference,_=import_gv2()
    def setUp(self):
        self.runtime=InferenceRuntime(GMock(),EncoderFixture(),self.inference.generate_once,
                        self.probe,"cpu",{"G":2,"clean":2,"adv":2,"text":6},"cpu_test_fixture")
        self.addCleanup(self.runtime.close)

    def test_image_exactly_once(self):
        data,md=self.runtime.image_pair(torch.full((1,3,8,4),.5))
        self.assertEqual(self.runtime.samples,{"G":1,"clean":1,"adv":1,"text":0})
        self.assertEqual(self.runtime.calls,self.runtime.samples)
        self.assertLessEqual(md["linf"],8/255)
        for x in data.values():self.assertEqual(x.shape,(1,512))

    def test_text_counts_and_unit_features(self):
        f=self.runtime.encode_text(("A black shirt.","A blаck shirt.","A blue shirt."))
        self.assertEqual(self.runtime.samples["text"],3)
        self.assertEqual(self.runtime.calls["text"],1)
        self.assertTrue(torch.allclose(f.norm(dim=1),torch.ones(3),atol=2e-4))
        self.assertFalse(f.requires_grad)

    def test_raw_unicode_reaches_tokenizer(self):
        texts=("black","blаck")
        reports=[self.probe.inspect(t) for t in texts]
        self.assertNotEqual(reports[0].model_tokens,reports[1].model_tokens)
        self.runtime.encode_text(texts)

    def test_text_truncation_rejected_before_forward(self):
        with self.assertRaises(ValueError):self.runtime.encode_text(("black "*100,))
        self.assertEqual(self.runtime.calls["text"],0)

    def test_forward_limit(self):
        self.runtime.limits["G"]=1
        self.runtime.image_pair(torch.zeros(1,3,8,4))
        with self.assertRaises(ValueError):self.runtime.image_pair(torch.zeros(1,3,8,4))
        self.assertEqual(self.runtime.samples["G"],1)

    def test_plain_grad_context_rejected(self):
        with self.assertRaises(ValueError):self.runtime.consume("text",1)

    def test_model_train_flag_rejected(self):
        self.runtime.encoder.model.train()
        with self.assertRaises(ValueError):self.runtime.encode_text(("A man",))

    def test_model_requires_grad_rejected(self):
        self.runtime.generator.scale.requires_grad_(True)
        with self.assertRaises(ValueError):self.runtime.image_pair(torch.zeros(1,3,8,4))

    def test_buffer_version_rejected(self):
        self.runtime.encoder.model.offset.add_(1)
        with self.assertRaises(ValueError):self.runtime.assert_frozen()

    def test_data_mutation_caught_by_final_sha(self):
        self.runtime.generator.scale.data.add_(.001)
        with self.assertRaises(ValueError):self.runtime.finish()

    def test_finish_sha_exact(self):
        self.runtime.image_pair(torch.zeros(1,3,8,4))
        self.runtime.encode_text(("A man",))
        final=self.runtime.finish()
        self.assertEqual(final["state_sha256_before"],final["state_sha256_after"])
        self.assertTrue(final["gradients_absent"])
        self.assertIs(self.runtime.finish(),final)

    def test_no_input_or_model_gradients(self):
        x=torch.full((1,3,8,4),.3,requires_grad=True)
        self.runtime.image_pair(x)
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in self.runtime.models for p in m.parameters()))

    def test_out_of_range_input(self):
        with self.assertRaises(ValueError):self.runtime.image_pair(torch.full((1,3,8,4),2.))

    def test_single_image_required(self):
        with self.assertRaises(ValueError):self.runtime.image_pair(torch.zeros(2,3,8,4))

    def test_invalid_generated_budget(self):
        self.runtime.generator.scale.data.fill_(.1)
        with self.assertRaises(ValueError):self.runtime.image_pair(torch.zeros(1,3,8,4))

    def test_zero_or_wrong_features_rejected(self):
        for f in (torch.zeros(1,512),torch.ones(1,513),torch.full((1,512),float("nan"))):
            with self.assertRaises(ValueError):self.runtime.checked_features(f,1)

    def test_real_runtime_cannot_use_cpu(self):
        with self.assertRaises(ValueError):
            InferenceRuntime(GMock(),EncoderFixture(),self.inference.generate_once,self.probe,"cpu",{},"official_irra_real")

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend=CaptionFrontend();cls.probe=official_probe();cls.inference,_=import_gv2()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        samples,_,_,_=fixture()
        for s in samples:(self.root/s.image_rel_path).write_bytes(s.row_id.encode())
        self.samples=tuple(replace(s,image_sha256=sha(self.root/s.image_rel_path)) for s in samples)
        self.catalog=TrainingCatalog(self.samples,{"fixture":"phase3 CPU not official"},self.root,{})
        self.protocol={"execution_backend":"cpu_test_fixture","catalog_sha256":self.catalog.fingerprint,
                       "gallery":[asdict(s) for s in self.samples[:3]],"queries":[asdict(s) for s in self.samples[:3]]}
        self.cache=RealFeatureCache(self.root/"cache",self.catalog,self.protocol,self.frontend,self.probe)
        self.runtime=InferenceRuntime(GMock(),EncoderFixture(),self.inference.generate_once,self.probe,"cpu",
                                     {"G":3,"clean":3,"adv":3,"text":9},"cpu_test_fixture")
        self.addCleanup(self.runtime.close)
    def prepare(self):
        for i,s in enumerate(self.samples[:3]):
            self.cache.image(s,self.runtime,loader=lambda p:torch.full((1,3,8,4),.3))
    def variant(self):
        e=self.frontend(self.samples[0].caption)
        for i in range(len(e.attributes)):
            c=candidates(e,e.caption,i,set(),self.probe)
            if c:
                pos=next(iter(c));edit=make_edit(e,i,pos,c[pos][0])
                return replay(e,(edit,),1)
        self.fail("no candidate")

    def test_image_resume_no_new_forward(self):
        self.prepare();before=dict(self.runtime.samples)
        self.cache.image(self.samples[0],self.runtime,loader=lambda p:torch.zeros(1,3,8,4))
        self.assertEqual(before,self.runtime.samples)

    def test_compact_storage(self):
        self.prepare()
        data,_=self.cache.image(self.samples[0])
        for x in data.values():self.assertEqual(x.untyped_storage().nbytes(),x.numel()*4)

    def test_corrupt_feature_rejected(self):
        self.prepare()
        p=next((self.root/"cache/image").glob("*/features.pt"));p.write_bytes(b"bad")
        with self.assertRaises(ValueError):self.cache.gather(self.samples[:3])

    def test_missing_receipt_not_recomputed(self):
        self.prepare();identity=asdict(self.samples[0])
        p=self.cache.path("image",identity);p.with_name(p.name+".receipt.json").unlink()
        before=dict(self.runtime.samples)
        with self.assertRaises(ValueError):self.cache.image(self.samples[0],self.runtime,loader=lambda p:torch.zeros(1,3,8,4))
        self.assertEqual(before,self.runtime.samples)

    def test_pending_refuses_new_forward(self):
        p=self.cache.path("image",asdict(self.samples[0]))
        p.with_name(p.name+".pending").mkdir()
        with self.assertRaises(FileExistsError):self.cache.image(self.samples[0],self.runtime,loader=lambda p:torch.zeros(1,3,8,4))
        self.assertEqual(self.runtime.samples["G"],0)

    def test_wrong_image_source_sha(self):
        (self.root/self.samples[0].image_rel_path).write_bytes(b"changed")
        with self.assertRaises(ValueError):self.cache.image(self.samples[0],self.runtime,loader=lambda p:torch.zeros(1,3,8,4))
        self.assertEqual(self.runtime.samples["G"],0)

    def test_validation_forbidden(self):
        with self.assertRaises(ValueError):self.cache.image(self.samples[3],self.runtime)

    def test_selection_identity_rejected(self):
        forged=replace(self.samples[0],pid=99)
        with self.assertRaises(ValueError):self.cache.image(forged,self.runtime)

    def test_mock_cache_protocol_rejected(self):
        protocol={**self.protocol,"execution_backend":"cpu_mock"}
        with self.assertRaises(ValueError):RealFeatureCache(self.root/"other",self.catalog,protocol,self.frontend,self.probe)

    def test_changed_protocol_rejected(self):
        protocol={**self.protocol,"new_setting":1}
        with self.assertRaises(ValueError):RealFeatureCache(self.root/"cache",self.catalog,protocol,self.frontend,self.probe)

    def test_catalog_fingerprint_rejected(self):
        protocol={**self.protocol,"catalog_sha256":"bad"}
        with self.assertRaises(ValueError):RealFeatureCache(self.root/"x",self.catalog,protocol,self.frontend,self.probe)

    def test_text_bundle_reuses_features(self):
        q=self.samples[0];texts=(q.caption,self.variant(),q.caption)
        a,_=self.cache.text(q,texts,self.runtime);before=dict(self.runtime.samples)
        b,_=self.cache.text(q,texts,self.runtime)
        self.assertTrue(torch.equal(a,b));self.assertEqual(before,self.runtime.samples)

    def test_text_wrong_original_rejected(self):
        with self.assertRaises(ValueError):self.cache.text(self.samples[0],("wrong",),self.runtime)
        self.assertEqual(self.runtime.samples["text"],0)

    def test_illegal_unicode_rejected(self):
        q=self.samples[0]
        with self.assertRaises(ValueError):self.cache.text(q,(q.caption,q.caption+"!"),self.runtime)

    def test_more_than_three_variants_rejected(self):
        q=self.samples[0]
        with self.assertRaises(ValueError):self.cache.text(q,(q.caption,)*4,self.runtime)

    def test_multipoistive_reward_binding(self):
        self.prepare()
        bound=self.cache.bind_reward(self.samples[0],self.samples[:3],self.runtime,3)
        self.assertEqual(bound.evaluator.gallery.positive_mask.tolist(),[True,True,False])
        sr,gr=bound.evaluate_many(self.samples[0].caption,(self.variant(),self.samples[0].caption))
        self.assertEqual(sr.total,sr.R_sim+sr.R_rank)
        self.assertEqual(sr.synergy_increment,sr.adv_sim_drop-sr.clean_sim_drop)
        self.assertLess(abs(gr.total),1e-6)
        self.runtime.finish()

    def test_reward_budget_rejected_before_encoding(self):
        self.prepare();bound=self.cache.bind_reward(self.samples[0],self.samples[:3],self.runtime,0)
        with self.assertRaises(ValueError):bound.evaluate_many(self.samples[0].caption,(self.variant(),))
        self.assertEqual(self.runtime.samples["text"],0)

    def test_query_validation_rejected(self):
        self.prepare()
        with self.assertRaises(ValueError):self.cache.bind_reward(self.samples[3],self.samples[:3],self.runtime,1)

class SelectionAndCLITests(unittest.TestCase):
    def setUp(self):
        samples,_,_,_=fixture()
        base=samples[0]
        self.samples=tuple(replace(base,row_id=f"cuhk:{i}:0",image_rel_path=f"{i}.jpg",pid=10+i//2)
                           for i in range(10))
        self.catalog=TrainingCatalog(self.samples,{"fixture":"selection"},".",{})

    def test_selection_deterministic_and_train_only(self):
        a,b=select_smoke(self.catalog);c,d=select_smoke(self.catalog)
        self.assertEqual(a,c);self.assertEqual(b,d);self.assertEqual(len(a),10)
        for q in b:self.assertEqual(sum(s.pid==q.pid for s in a),2)

    def test_over_limit_rejected(self):
        with self.assertRaises(ValueError):validate_selection(self.catalog,self.samples*2,self.samples)

    def test_negative_missing_rejected(self):
        with self.assertRaises(ValueError):validate_selection(self.catalog,self.samples[:2],self.samples[:1])

    def test_query_pair_missing_rejected(self):
        with self.assertRaises(ValueError):validate_selection(self.catalog,self.samples[1:],self.samples[:1])

    def test_cli_default_no_gpu(self):
        protocol={}
        with patch.object(smoke_phase3,"preflight",return_value=(self.catalog,self.samples,self.samples,None,protocol)), \
             patch.object(smoke_phase3,"run_smoke",side_effect=AssertionError("GPU")):
            smoke_phase3.main([])

    def test_cli_gpu_permission_required(self):
        with patch.object(smoke_phase3,"preflight",side_effect=AssertionError("must not reach")):
            with self.assertRaises(ValueError):smoke_phase3.main(["--action","smoke"])

    def test_existing_output_no_gpu_retry(self):
        with tempfile.TemporaryDirectory() as d, \
             patch.object(smoke_phase3,"safe_output",return_value=Path(d)), \
             patch.object(smoke_phase3,"load_real_runtime",side_effect=AssertionError("GPU")):
            with self.assertRaises(FileExistsError):
                smoke_phase3.run_smoke(self.catalog,self.samples,self.samples,None,
                                      {"execution_backend":"official_irra_real"},d)

if __name__=="__main__":
    unittest.main()
