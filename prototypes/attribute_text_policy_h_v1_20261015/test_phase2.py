"""Phase 2合成CPU契约测试；真实前端/官方Tokenizer只读取文本，不加载网络。"""
import copy
from dataclasses import replace, asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from contracts import TextExample
from text_frontend import CaptionFrontend
from training_inputs import SampleIdentity, TrainingCatalog, validate_identities
from feedback_backend import FrozenMockOps, BoundReward, validate_variant, state_sha
from feature_cache import GalleryFeatureCache
from tokenizer_probe import official_probe
from constraints import make_edit, replay, candidates
from phase2_sources import digest, sha, new_json, checked, within
from policy import CharacterPolicy
from scst import scst_loss

class GMock(nn.Module):
    h_cpu_mock=True
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(.01))
    def forward(self,x):
        return torch.ones_like(x)*self.scale

class ImageMock(nn.Module):
    h_cpu_mock=True
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.arange(1.,513.)/512)
    def forward(self,x):
        avg=x.mean((1,2,3))
        return self.weight[None]+avg[:,None]*torch.flip(self.weight,(0,))[None]

class TextMock(nn.Module):
    h_cpu_mock=True
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.arange(1.,513.)/512)
    def forward(self,tokens):
        return self.weight[None]+(tokens.float().sum(1)%97)[:,None]*.001*torch.flip(self.weight,(0,))[None]

def fixture():
    raw=[
      {"id":11,"split":"train","file_path":"a.jpg","captions":["A man wears a black shirt."]},
      {"id":11,"split":"train","file_path":"b.jpg","captions":["A man wears black pants."]},
      {"id":22,"split":"train","file_path":"c.jpg","captions":["A woman wears a blue shirt."]},
      {"id":33,"split":"train","file_path":"d.jpg","captions":["A woman wears a coat."]}]
    samples=tuple(SampleIdentity("cuhk:"+str(i)+":0",r["file_path"],0,"train",
                   "validation" if i==3 else "train",r["id"],r["captions"][0],"a"*64,"b"*64,"c"*64)
                  for i,r in enumerate(raw))
    records=[{"row_id":s.row_id,"pid":s.pid,"caption":s.caption} for s in samples]
    audit=validate_identities(samples,raw,records,[0,1,2],[3],{"train":3,"validation":1})
    return samples,raw,records,TrainingCatalog(samples,{"fixture":"synthetic-not-official"},Path("."),audit)

class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend=CaptionFrontend()
        cls.probe=official_probe()

    def test_caption_only(self):
        example=self.frontend("A man wears a black shirt and black pants.")
        self.assertIsInstance(example,TextExample)
        self.assertEqual(set(asdict(example)),{"caption","attributes"})

    def test_record_rejected(self):
        with self.assertRaises(TypeError):
            self.frontend({"caption":"A man","pid":1})

    def test_extra_argument_rejected(self):
        with self.assertRaises(TypeError):
            self.frontend("A man",pid=1)

    def test_spans_exact(self):
        example,audit=self.frontend.convert("A woman wears a red coat and blue jeans.")
        self.assertEqual(len(example.attributes),5)
        self.assertEqual(audit["exact_span_count"],5)
        for a in example.attributes:
            self.assertEqual(example.caption[a.char_start:a.char_end],a.raw_text)

    def test_repeated_spans(self):
        e=self.frontend("He wears a blue shirt, a blue shirt.")
        blues=[a for a in e.attributes if a.raw_text=="blue"]
        self.assertEqual([(a.char_start,a.char_end) for a in blues],[(11,15),(25,29)])
        self.assertNotEqual(blues[0].attribute_id,blues[1].attribute_id)

    def test_unicode_prefix_positions(self):
        e=self.frontend("🙂 A man wears a black shirt.")
        for a in e.attributes:
            self.assertEqual(e.caption[a.char_start:a.char_end],a.raw_text)
        self.assertTrue(any(a.raw_text=="black" for a in e.attributes))

    def test_empty_candidates(self):
        self.assertEqual(self.frontend("xyz qqq").attributes,())

    def test_no_hidden_load(self):
        with patch.object(torch,"load",side_effect=AssertionError("forbidden")), \
             patch.object(nn.Module,"_call_impl",side_effect=AssertionError("forbidden")):
            self.assertTrue(self.frontend("A man wears a black shirt.").attributes)

    def test_policy_input_text_only(self):
        e=self.frontend("A man wears a black shirt.")
        torch.manual_seed(42)
        p=CharacterPolicy(self.probe,width=16,layers=1,heads=4)
        self.assertIsNotNone(p.attack(e))
        with self.assertRaises(TypeError):
            p({"caption":e.caption,"pid":22})

    def test_frontend_source_tamper(self):
        with patch("text_frontend.checked",side_effect=ValueError("SHA")):
            with self.assertRaises(ValueError):
                CaptionFrontend()

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.samples,self.raw,self.records,self.catalog=fixture()

    def validate(self,samples=None,raw=None,records=None):
        return validate_identities(samples or self.samples,raw or self.raw,records or self.records,
                                   [0,1,2],[3],{"train":3,"validation":1})

    def test_valid_sources(self):
        self.assertEqual(self.validate()["pid_intersection"],0)

    def test_wrong_caption(self):
        for field,value in [("caption","different"),("pid",99),("image_rel_path","wrong.jpg"),
                            ("caption_index",1),("row_id","cuhk:1:0"),("split","test"),("role","test")]:
            with self.subTest(field=field),self.assertRaises((ValueError,IndexError)):
                self.validate((replace(self.samples[0],**{field:value}),*self.samples[1:]))

    def test_official_test_annotation(self):
        raw=copy.deepcopy(self.raw); raw[0]["split"]="test"
        with self.assertRaises(ValueError): self.validate(raw=raw)

    def test_pid_leakage(self):
        raw=copy.deepcopy(self.raw); raw[3]["id"]=11
        samples=(*self.samples[:3],replace(self.samples[3],pid=11))
        records=copy.deepcopy(self.records);records[3]["pid"]=11
        with self.assertRaises(ValueError): self.validate(samples,raw,records)

    def test_duplicate(self):
        with self.assertRaises(ValueError): self.validate((*self.samples[:3],self.samples[0]))

    def test_split_record_forged(self):
        records=copy.deepcopy(self.records);records[0]["caption"]="fake"
        with self.assertRaises(ValueError): self.validate(records=records)

    def test_validation_feedback_rejected(self):
        with self.assertRaises(ValueError): self.catalog.require_train(self.samples[3])

    def test_forged_sample_rejected(self):
        with self.assertRaises(ValueError): self.catalog.require_train(replace(self.samples[0],pid=99))

    def test_unknown_query_rejected(self):
        with self.assertRaises(ValueError): self.catalog.get_train("official-test:1")

    def test_missing_sha(self):
        with self.assertRaises(ValueError):
            self.validate((replace(self.samples[0],image_sha256=""),*self.samples[1:]))

    def test_protocol_digest_changes(self):
        c=TrainingCatalog(self.samples,{"fixture":"changed"},".",{})
        self.assertNotEqual(c.fingerprint,self.catalog.fingerprint)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):within(d,Path(d)/"../escape")

    def test_image_sha_check(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.jpg";p.write_bytes(b"fixture")
            s=replace(self.samples[0],image_sha256=sha(p))
            c=TrainingCatalog((s,),{"fixture":"image"},d,{})
            self.assertEqual(c.verify_image(s),p)
            p.write_bytes(b"changed")
            with self.assertRaises(ValueError): c.verify_image(s)

class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frontend=CaptionFrontend();cls.probe=official_probe()
    def setUp(self):
        self.samples,_,_,self.catalog=fixture()
        self.ops=FrozenMockOps(GMock(),ImageMock(),TextMock(),self.probe)
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache=GalleryFeatureCache(self.temp.name,self.catalog,self.ops.protocol)

    def prepare(self):
        for i,s in enumerate(self.samples[:3]):
            self.cache.prepare_mock(s,torch.full((1,3,8,4),.2+i*.2),self.ops)
        return self.cache.bind_reward(self.samples[0],self.samples[:3],self.ops,self.frontend,3)

    def variant(self):
        e=self.frontend(self.samples[0].caption)
        for i in range(len(e.attributes)):
            supports=candidates(e,e.caption,i,set(),self.probe)
            if supports:
                pos=next(iter(supports))
                edit=make_edit(e,i,pos,supports[pos][0])
                return replay(e,(edit,),1)
        self.fail("no legal edit")

    def test_frozen_image_budget(self):
        data,info=self.ops.image_pair(torch.full((1,3,8,4),.99,requires_grad=True))
        self.assertLessEqual(info["linf"],8/255)
        self.assertEqual(info["G_forward"],1)
        self.assertEqual(info["IRRA_image_forward"],2)
        self.assertFalse(data["clean"].requires_grad)
        self.ops.assert_frozen()

    def test_no_real_loader(self):
        with self.assertRaises(ValueError): FrozenMockOps(nn.Identity(),ImageMock(),TextMock(),self.probe)

    def test_no_input_grad(self):
        pixels=torch.full((1,3,8,4),.3,requires_grad=True)
        self.ops.image_pair(pixels)
        self.assertIsNone(pixels.grad)
        for m in self.ops.modules:
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.parameters()))

    def test_buffer_change_rejected(self):
        self.ops.generator.register_buffer("changed",torch.tensor(1))
        with self.assertRaises(ValueError): self.ops.assert_frozen()

    def test_training_flag_rejected(self):
        self.ops.image_encoder.train()
        with self.assertRaises(ValueError):self.ops.image_pair(torch.zeros(1,3,8,4))

    def test_invalid_delta(self):
        self.ops.generator.scale.data.fill_(.5)
        self.ops.initial=tuple(state_sha(m) for m in self.ops.modules)
        with self.assertRaises(ValueError): self.ops.image_pair(torch.zeros(1,3,8,4))

    def test_cache_resume_no_forward(self):
        self.prepare();counts=dict(self.ops.counts)
        self.cache.prepare_mock(self.samples[0],torch.zeros(1,3,8,4),self.ops)
        self.assertEqual(self.ops.counts,counts)

    def test_compact_storage(self):
        self.prepare()
        for x in self.cache.load(self.samples[0]).values():
            self.assertEqual(x.untyped_storage().nbytes(),x.numel()*x.element_size())

    def test_model_protocol_mismatch(self):
        self.prepare()
        protocol=copy.deepcopy(self.ops.protocol);protocol["mock_model_sha256"]=("wrong",)
        with self.assertRaises(ValueError):GalleryFeatureCache(self.temp.name,self.catalog,protocol)

    def test_fake_tokenizer_rejected(self):
        with self.assertRaises(ValueError):
            FrozenMockOps(GMock(),ImageMock(),TextMock(),object())

    def test_catalog_mutation_rejected(self):
        self.catalog.source_contract["tampered"]=True
        with self.assertRaises(ValueError):self.catalog.require_train(self.samples[0])

    def test_source_protocol_mismatch(self):
        changed=TrainingCatalog(self.samples,{"changed":1},".",{})
        with self.assertRaises(ValueError):GalleryFeatureCache(self.temp.name,changed,self.ops.protocol)

    def test_real_cache_rejected(self):
        with self.assertRaises(ValueError):
            GalleryFeatureCache(self.temp.name,self.catalog,{"execution_backend":"real"})

    def test_validation_cache_rejected(self):
        with self.assertRaises(ValueError):self.cache.prepare_mock(self.samples[3],torch.zeros(1,3,8,4),self.ops)

    def test_corrupt_cache(self):
        self.prepare();unit,_=self.cache.paths(self.samples[0])
        (unit/"features.pt").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):self.cache.load(self.samples[0])

    def test_receipt_tamper(self):
        self.prepare();_,receipt=self.cache.paths(self.samples[0]);receipt.write_text("{}")
        with self.assertRaises(ValueError):self.cache.load(self.samples[0])

    def test_missing_receipt_no_regeneration(self):
        self.prepare();_,receipt=self.cache.paths(self.samples[0]);receipt.unlink()
        counts=dict(self.ops.counts)
        with self.assertRaises(ValueError):self.cache.prepare_mock(self.samples[0],torch.zeros(1,3,8,4),self.ops)
        self.assertEqual(self.ops.counts,counts)

    def test_pending_no_repeated_generation(self):
        unit,_=self.cache.paths(self.samples[0])
        self.ops.generator.scale.data.fill_(1.)
        self.ops.initial=tuple(state_sha(m) for m in self.ops.modules)
        # 模型来源变化先拒绝，不创建意图。
        with self.assertRaises(ValueError):self.cache.prepare_mock(self.samples[0],torch.zeros(1,3,8,4),self.ops)
        (Path(self.temp.name)/(unit.name+".pending")).mkdir()
        with self.assertRaises(FileExistsError):self.cache.prepare_mock(self.samples[0],torch.zeros(1,3,8,4),
            FrozenMockOps(GMock(),ImageMock(),TextMock(),self.probe))

    def test_all_same_pid_positives(self):
        bound=self.prepare()
        self.assertEqual(bound.evaluator.gallery.positive_mask.tolist(),[True,True,False])
        self.assertEqual(bound.evaluator.gallery.paired_index,0)

    def test_query_wrong_caption(self):
        bound=self.prepare()
        with self.assertRaises(ValueError):bound.evaluate_many("wrong",(self.variant(),))

    def test_reward_and_synergy(self):
        bound=self.prepare();r=bound.evaluate_many(self.samples[0].caption,(self.variant(),))[0]
        self.assertEqual(r.total,r.R_sim+r.R_rank)
        self.assertEqual(r.synergy_increment,r.adv_sim_drop-r.clean_sim_drop)
        self.ops.assert_frozen()

    def test_noop_reward(self):
        bound=self.prepare();r=bound.evaluate_many(self.samples[0].caption,(self.samples[0].caption,))[0]
        self.assertTrue(all(abs(v)<1e-6 for v in asdict(r).values()))

    def test_unicode_tokens_real(self):
        before=self.probe.inspect(self.samples[0].caption)
        after=self.probe.inspect(self.variant())
        self.assertNotEqual(before.model_tokens,after.model_tokens)
        self.assertFalse(after.truncated)

    def test_illegal_unicode(self):
        e=self.frontend(self.samples[0].caption)
        for variant in ("X"+e.caption[1:],e.caption+"!",e.caption.replace("black","blа\u200bck"),
                        e.caption.replace("black","bl💥ck"),e.caption.replace("man","mаn").replace("black","blаck")):
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                validate_variant(e,variant,1,self.probe)

    def test_legal_unicode(self):
        e=self.frontend(self.samples[0].caption)
        edits=validate_variant(e,self.variant(),1,self.probe)
        self.assertEqual(len(edits),1)
        self.assertTrue(edits[0].replacement_codepoint.startswith("U+"))

    def test_validation_query_rejected(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.cache.bind_reward(self.samples[3],self.samples[:3],self.ops,self.frontend,1)

    def test_missing_pair_rejected(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.cache.bind_reward(self.samples[0],self.samples[1:3],self.ops,self.frontend,1)

    def test_negative_required(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.cache.bind_reward(self.samples[0],self.samples[:2],self.ops,self.frontend,1)

    def test_duplicate_gallery_rejected(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.cache.bind_reward(self.samples[0],(self.samples[0],)*2,self.ops,self.frontend,1)

    def test_reward_stop_gradient(self):
        bound=self.prepare()
        r=bound.evaluate_many(self.samples[0].caption,(self.variant(),self.samples[0].caption))
        lp=torch.tensor(.2,requires_grad=True)
        loss,_=scst_loss(lp,r[0].total,r[1].total)
        loss.backward()
        self.assertIsNotNone(lp.grad)
        for m in self.ops.modules:
            self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_text_feature_no_grad(self):
        f=self.ops.encode_text((self.samples[0].caption,self.variant()))
        self.assertFalse(f.requires_grad)
        self.assertTrue(torch.allclose(f.norm(dim=1),torch.ones(2),atol=2e-4))

    def test_truncated_text_rejected(self):
        with self.assertRaises(ValueError):self.ops.encode_text(("black "*100,))

    def test_no_metrics_coupling(self):
        # 两个正例从rank1/2变为rank1/3：best-positive Margin仍相同，但AP从1降到5/6。
        before=torch.tensor([.9,.8,.7]);after=torch.tensor([.9,.6,.7])
        positive=torch.tensor([True,True,False])
        margin=lambda x:x[positive].max()-x[~positive].max()
        self.assertEqual(float(margin(before)),float(margin(after)))
        self.assertAlmostEqual((1+2/3)/2,5/6)

if __name__=="__main__":
    unittest.main()
