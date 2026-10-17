"""真实模型禁止；合成CPU像素、Mock插件、Unicode规则与不可变存储测试。"""
from dataclasses import asdict,replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file,json_bytes,digest,PendingPublication,read_json
from tbps.contracts import DatasetManifest,GalleryItem,QueryItem,PersonID
from tbps.attacks.base import output,tensor_sha,TextRequest,ImageRequest
from tbps.attacks.gv2 import GV2Plugin,BUDGET as G_BUDGET
from tbps.attacks.hv1 import HV1Plugin,BUDGET as H_BUDGET
from tbps.attacks.provenance import AttackProvenance
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.models.base import ModelSpace
from tbps.materialization.contracts import MaterializationPlan
from tbps.materialization.store import MaterializedStore,image_checks,utf8_sha
from tbps.materialization.pipeline import materialize_image,materialize_text,seal_collection,target_input_binding
from tbps.materialization.legacy_text import HistoricalTextSource
from tbps.materialization.generation import generate_image_once,generate_text_once,seal_generated_collection
from stage3a_fixtures import support,GBackend,HBackend
from stage4b_fixtures import ReplayFixture

class MaterializationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage4c-")
        self.root=Path(self.temp.name);images=self.root/"images";images.mkdir()
        ann=self.root/"annotation.json";ann.write_bytes(b'{"fixture":true}')
        self.ref=lambda p:FileRef(str(p),sha_file(p))
        self.galleries=[];self.queries=[]
        for i in range(12):
            p=images/(str(i)+".bin");p.write_bytes(str(i).encode())
            pid=PersonID("cuhk_pedes_fixture",str(i//2))
            self.galleries.append(GalleryItem("g"+str(i),p.name,sha_file(p),pid,"test"))
            for j,caption in enumerate(("red shirt","plain")):
                self.queries.append(QueryItem("q"+str(2*i+j),caption,j,"g"+str(i),pid,"test"))
        self.manifest=DatasetManifest("cuhk_pedes_fixture","test",self.ref(ann),tuple(self.galleries),tuple(self.queries))
        weights=[]
        for name in ("g","h"):
            p=self.root/(name+".pt");p.write_bytes(name.encode());weights.append(self.ref(p))
        shared=digest("fixture source")
        prov=lambda name,ck:AttackProvenance(name,ck,shared,shared,shared,"cuhk_pedes:attack_train:2048","cpu_fixture",shared if name=="hv1" else None)
        self.g,self.h=prov("gv2",weights[0]),prov("hv1",weights[1])
        self.plan=MaterializationPlan(self.manifest,self.g,self.h,(self.ref(ann),*weights),str(images),pixel_size=(8,8),fixture=True)
        self.store=MaterializedStore(ArtifactStore(self.root/"saved",new_base=self.root),self.plan)
        self.store.publish_plan()
        self.session=legacy_imports(support(),allow_import=True);self.old=self.session.__enter__()
        self.gb=GBackend(self.old("inference_once"));self.hb=HBackend(self.old)
        self.gp=GV2Plugin(backend=self.gb,provenance=self.g);self.hp=HV1Plugin(backend=self.hb,provenance=self.h)
        self.validate=lambda caption,action:self.hb.module.validate_action(caption,action,self.hb.probe,self.hb.frontend)["adversarial"]
    def tearDown(self):
        self.session.__exit__(None,None,None);self.temp.cleanup()
    def image(self,i=0):
        return materialize_image(self.store,"g"+str(i),self.gp,lambda ref:self.gb.pixels.clone())
    def text(self,i=0):
        return materialize_text(self.store,"q"+str(i),self.hp,self.validate)
    def test_plan_digests_and_binding_do_not_rehash_full_manifest_per_query(self):
        expected=self.plan.manifest_sha256;pid=self.plan.id
        with patch("tbps.materialization.contracts.asdict",side_effect=AssertionError("full Manifest rehash")):
            self.assertEqual(self.plan.binding("q0","text").dataset_manifest_sha256,expected)
            self.assertEqual(self.plan.id,pid)
    def test_source_stat_guard_refuses_changes_even_if_mtime_restored(self):
        import os
        self.store.verify_plan();p=Path(self.g.checkpoint.path);stat=p.stat()
        p.write_bytes(b"x");os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        with self.assertRaises(ArtifactError):self.store.verify_plan()
    def test_verified_plan_cache_and_explicit_boundary_full_SHA(self):
        self.store.verify_plan()
        with patch.object(self.store.store,"verify",wraps=self.store.store.verify) as verify:
            self.store.verify_plan();self.assertEqual(verify.call_count,0)
            self.store.verify_plan(force=True);self.assertEqual(verify.call_count,1)
    def test_UTF8_content_is_not_silently_normalized(self):
        self.assertNotEqual(utf8_sha("é"),utf8_sha("e\u0301"))
    def test_guarded_generation_claim_precedes_G_and_resume_does_not_repeat(self):
        from tbps.materialization.generation import generate_image_once
        original=self.gp.generate
        def guarded(req):
            claim=self.store.store.root/("generation-claim-"+self.store.unit_id(req.binding))
            self.assertTrue((claim/"completed.json").exists())
            return original(req)
        with patch.object(self.gp,"generate",side_effect=guarded) as g:
            ref=generate_image_once(self.store,"g0",self.gp,lambda x:self.gb.pixels)
            again=generate_image_once(self.store,"g0",self.gp,lambda x:self.gb.pixels)
            self.assertEqual(ref,again);self.assertEqual(g.call_count,1)
    def test_failure_after_G_before_pixel_publication_refuses_automatic_forward_retry(self):
        from tbps.materialization.generation import generate_image_once
        original=self.gp.generate
        def crash(req):original(req);raise InterruptedError("after G, before publish")
        with patch.object(self.gp,"generate",side_effect=crash):
            with self.assertRaises(InterruptedError):generate_image_once(self.store,"g0",self.gp,lambda x:self.gb.pixels)
        with patch.object(self.gp,"generate",side_effect=AssertionError("must not repeat")) as g:
            with self.assertRaises(PendingPublication):generate_image_once(self.store,"g0",self.gp,lambda x:self.gb.pixels)
            g.assert_not_called()
    def test_guarded_text_resume_and_joint_references_only(self):
        from tbps.materialization.generation import generate_text_once,seal_generated_collection
        refs=[]
        with patch.object(self.hp,"generate",wraps=self.hp.generate) as h:
            for i in range(24):refs.append(generate_text_once(self.store,"q"+str(i),self.hp,self.validate))
            self.assertEqual(generate_text_once(self.store,"q0",self.hp,self.validate),refs[0])
            self.assertEqual(h.call_count,24)
            seal_generated_collection(self.store,"text_only",texts=tuple(refs),validate=self.validate)
            self.assertEqual(h.call_count,24)
    def test_real_pixel_roundtrip_SHA_and_effective_delta(self):
        ref=self.image();x,md=self.store.load(ref)
        self.assertEqual(ref.content_sha256,tensor_sha(x["adversarial"]))
        self.assertTrue(torch.equal(x["effective_delta"],x["adversarial"]-x["original"]))
        self.assertLessEqual(md["pixels"]["linf"],8/255)
        self.assertFalse(md["pixels"]["preview"])
        self.assertEqual(self.store.attack_artifact(ref).attack_id,"gv2")
    def test_content_SHA_differs_from_serialized_file_SHA(self):
        ref=self.image();self.assertNotEqual(ref.content_sha256,ref.data.sha256)
        self.assertEqual(self.store.load(ref)[1]["original_file"]["sha256"],self.galleries[0].image_sha256)
    def test_shared_storage_is_compact_after_save_and_load(self):
        ref=self.image()
        original=torch.load
        large=torch.zeros(1000,3,8,8);base=original(ref.data.path,map_location="cpu",weights_only=True)
        views={}
        for k,x in base.items():
            large=torch.zeros(1000,3,8,8);large[:1]=x;views[k]=large[:1]
        with patch("torch.load",return_value=views):x,_=self.store.load(ref)
        for k,v in x.items():
            self.assertTrue(torch.equal(v,base[k]));self.assertEqual(v.untyped_storage().nbytes(),1*3*8*8*4)
    def test_unicode_roundtrip_content_and_full_action(self):
        ref=self.text();adv,md=self.store.load(ref,validate=self.validate)
        self.assertNotEqual(adv,"red shirt");self.assertEqual(utf8_sha(adv),ref.content_sha256)
        self.assertEqual(md["action"]["original"],"red shirt");self.assertTrue(md["action"]["edits"])
        self.assertEqual(self.store.attack_artifact(ref,validate=self.validate).attack_id,"hv1")
    def test_NOOP_query_retained(self):
        ref=self.text(1);adv,md=self.store.load(ref,validate=self.validate)
        self.assertEqual(adv,"plain");self.assertEqual(md["action"]["edits"],[])
        self.assertEqual(md["action"]["H_forward"],0)
    def test_unsafe_Unicode_encoding_rejected(self):
        with self.assertRaises(UnicodeError):utf8_sha("\ud800")
    def test_invalid_Unicode_action_rejected(self):
        req=self.store.text_request("q0");result=self.hp.generate(req);trace=result.trace()
        trace["action"]["adversarial"]="arbitrary"
        bad=output("hv1",req,H_BUDGET,"arbitrary",None,trace,result.provenance())
        with self.assertRaises(ValueError):self.store.publish_text(req,bad,self.validate)
    def test_invalid_delta_budget_range_dtype_or_shape_rejected(self):
        c=torch.full((1,3,8,8),.5)
        for a,d in ((c+.04,torch.full_like(c,.04)),(c+1,torch.ones_like(c)),(c.double(),torch.zeros_like(c).double()),
                    (c[:,:,:4],torch.zeros_like(c[:,:,:4])),(c,torch.ones_like(c)),(torch.full_like(c,float("nan")),torch.zeros_like(c))):
            with self.subTest(shape=a.shape,dtype=a.dtype),self.assertRaises(ArtifactError):image_checks(c,a,d,(8,8))
    def test_not_a_pixel_feature_embedding(self):
        req=self.store.image_request("g0");result=self.gp.generate(req)
        bad=replace(result,payload=torch.zeros(1,512),effective_delta=torch.zeros(1,512))
        with self.assertRaises(ArtifactError):self.store.publish_image(req,self.gb.pixels,bad)
    def test_wrong_input_identity_rejected(self):
        req=self.store.image_request("g0");result=self.gp.generate(req)
        with self.assertRaises(ArtifactError):self.store.publish_image(self.store.image_request("g1"),self.gb.pixels,result)
    def test_wrong_checkpoint_source_rejected(self):
        req=self.store.text_request("q0");result=self.hp.generate(req);p=result.provenance();p["checkpoint"]["sha256"]="0"*64
        bad=output("hv1",req,H_BUDGET,result.payload,None,result.trace(),p)
        with self.assertRaises(ArtifactError):self.store.publish_text(req,bad,self.validate)
    def test_wrong_caption_identity_rejected(self):
        req=self.store.text_request("q0");result=self.hp.generate(req)
        with self.assertRaises(ArtifactError):self.store.publish_text(self.store.text_request("q1"),result,self.validate)
    def test_trace_pixel_mutation_rejected(self):
        req=self.store.image_request("g0");result=self.gp.generate(req)
        changed=result.payload.clone();changed[...,3,3]-=.001
        result=replace(result,payload=changed,effective_delta=changed-self.gb.pixels)
        with self.assertRaises(ArtifactError):self.store.publish_image(req,self.gb.pixels,result)
    def test_existing_valid_unit_skips_second_G_forward(self):
        ref=self.image()
        with patch.object(self.gp,"generate",side_effect=AssertionError("second G forward")):
            self.assertEqual(self.image(),ref)
    def test_existing_valid_unit_skips_second_H_forward(self):
        ref=self.text()
        with patch.object(self.hp,"generate",side_effect=AssertionError("second H forward")):
            self.assertEqual(self.text(),ref)
    def test_immutable_double_publication_rejected(self):
        req=self.store.image_request("g0");result=self.gp.generate(req)
        self.store.publish_image(req,self.gb.pixels,result)
        with self.assertRaises(ArtifactError):self.store.publish_image(req,self.gb.pixels,result)
    def test_interruption_preserves_pending_and_prevents_regeneration(self):
        req=self.store.image_request("g0");result=self.gp.generate(req)
        def stop(phase):
            if phase=="before_receipt":raise InterruptedError("synthetic interruption")
        with self.assertRaises(InterruptedError):self.store.publish_image(req,self.gb.pixels,result,fault=stop)
        with patch.object(self.gp,"generate",side_effect=AssertionError("repeat forbidden")):
            with self.assertRaises(PendingPublication):self.image()
        self.assertTrue(self.store.store.pending())
    def test_missing_or_tampered_unit_rejected(self):
        ref=self.image();Path(ref.data.path).write_bytes(b"corrupt")
        with self.assertRaises(ArtifactError):self.store.load(ref)
    def test_original_file_changed_rejected(self):
        req=self.store.image_request("g0");result=self.gp.generate(req)
        Path(req.public.content.path).write_bytes(b"wrong")
        with self.assertRaises(ArtifactError):self.store.publish_image(req,self.gb.pixels,result)
    def test_source_plan_tamper_rejected(self):
        (self.store.store.root/"protocol/protocol.json").write_text("{}")
        with self.assertRaises(ArtifactError):self.image()
    def test_text_requires_frozen_action_validation(self):
        ref=self.text()
        with self.assertRaises(ArtifactError):self.store.load(ref)
    def test_full_independent_gallery_and_query_seal_without_regeneration(self):
        images=tuple(self.image(i) for i in range(12));texts=tuple(self.text(i) for i in range(24))
        with patch.object(self.gp,"generate",side_effect=AssertionError("no G in Joint")),patch.object(self.hp,"generate",side_effect=AssertionError("no H in Joint")):
            ref=seal_collection(self.store,"joint",images=images,texts=texts,validate=self.validate)
        body=read_json(ref.path);self.assertEqual(len(body["images"]),12);self.assertEqual(len(body["texts"]),24)
        self.assertEqual(body["new_G_forwards"],0);self.assertEqual(body["new_H_forwards"],0)
    def test_reversed_or_incomplete_order_refused(self):
        images=tuple(self.image(i) for i in range(12))
        for seq in (images[::-1],images[:-1]):
            with self.assertRaises(ArtifactError):seal_collection(self.store,"image_only",images=seq)
    def test_target_model_spaces_share_samples_but_not_cache_identity(self):
        refs=tuple(self.text(i) for i in range(24))
        def space(name):return ModelSpace(name,*([digest(name)]*5),"fp32")
        a=target_input_binding(self.store,space("irra_fixture"),"text",refs,validate=self.validate)
        b=target_input_binding(self.store,space("fusion_fixture"),"text",refs,validate=self.validate)
        self.assertEqual(a.ordered_sha256,b.ordered_sha256);self.assertNotEqual(a.cache_id,b.cache_id)
    def test_target_model_or_feedback_cannot_enter_generation_API(self):
        with self.assertRaises(TypeError):materialize_image(self.store,"g0",self.gp,lambda ref:self.gb.pixels,victim="irra")
        with self.assertRaises(TypeError):materialize_text(self.store,"q0",self.hp,self.validate,reward=1)
        self.assertFalse(set(TextRequest.__dataclass_fields__)&{"pid","paired_index","reward"})
    def test_default_real_generation_gate_before_any_model_call(self):
        with patch.object(MaterializationPlan,"fixture",False):
            with patch.object(self.gp,"generate") as g,self.assertRaisesRegex(ArtifactError,"not authorized"):self.image()
            g.assert_not_called()
    def test_materialization_plan_has_no_victim_or_target_score(self):
        names=set(MaterializationPlan.__dataclass_fields__)
        self.assertFalse(names&{"victim","victim_model","target_model","score","reward","tokenizer"})
    def test_historical_Embedding_cannot_masquerade_as_materialized_unit(self):
        with self.assertRaises(ArtifactError):self.store.load(object())
    def test_import_has_no_Torch_model_loading_side_effect(self):
        import subprocess,sys,os
        result=subprocess.run([sys.executable,"-B","-c","import sys; import tbps.materialization.pipeline; import tbps.materialization.generation; assert 'torch' not in sys.modules"],env=os.environ,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

class HistoricalTextTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-history-text-");self.root=Path(self.temp.name)
        self.f=ReplayFixture(self.root);self.s=self.f.sources()
        h=dict(self.s.models)["H"];sha=digest("fixture")
        self.p=AttackProvenance("hv1",h,sha,sha,sha,"cuhk_pedes:attack_train:2048","cpu_fixture",sha)
    def tearDown(self):self.temp.cleanup()
    def historical_store(self,source):
        models=dict(self.s.models);g=replace(self.p,attack_id="gv2",checkpoint=models["G"],selection_sha256=None)
        refs={r.path:r for r in self.s.refs.values()}
        refs.update({p:FileRef(p,v["sha256"]) for p,v in source.file_snapshot.items()})
        plan=MaterializationPlan(self.s.manifest,g,self.p,tuple(refs.values()),str(self.root/"images"),fixture=True)
        store=MaterializedStore(ArtifactStore(self.root/"saved",new_base=self.root),plan);store.publish_plan()
        return store
    def test_historical_text_publishes_as_real_string_with_original_evidence_pointer(self):
        source=HistoricalTextSource(self.s,self.p,self.f.validate)
        values=list(source.iter_outputs());store=self.historical_store(source)
        request,result=values[0];ref=store.publish_text(request,result,self.f.validate)
        adv,md=store.load(ref,validate=self.f.validate)
        self.assertEqual(adv,result.payload);self.assertEqual(md["trace"]["H_forward_samples_this_run"],0)
        self.assertEqual(md["trace"]["stage"],"historical_text_reuse")
    def test_historical_action_pointer_mismatch_is_rejected(self):
        source=HistoricalTextSource(self.s,self.p,self.f.validate)
        values=list(source.iter_outputs());store=self.historical_store(source)
        request,result=values[0];trace=result.trace();trace["historical_query_index"]+=1
        bad=output("hv1",request,H_BUDGET,result.payload,None,trace,result.provenance())
        with self.assertRaises(ArtifactError):store.publish_text(request,bad,self.f.validate)
    def test_all_old_Unicode_outputs_reusable_without_H_or_feature_load(self):
        source=HistoricalTextSource(self.s,self.p,self.f.validate)
        with patch("torch.load",side_effect=AssertionError("no Tensor/model reading")):
            values=list(source.iter_outputs())
        self.assertEqual(len(values),24);self.assertEqual(source.report()["status"],"TEXT_SOURCE_VERIFIED")
        self.assertEqual(source.report()["H_forwards"],0)
    def test_partial_iteration_cannot_claim_complete(self):
        source=HistoricalTextSource(self.s,self.p,self.f.validate);iterator=source.iter_outputs();next(iterator)
        self.assertEqual(source.report()["status"],"TEXT_SOURCE_PENDING");iterator.close()
    def test_historical_text_wrong_H_source_rejected(self):
        bad=replace(self.p,checkpoint=FileRef(self.p.checkpoint.path,"0"*64))
        with self.assertRaises(ArtifactError):HistoricalTextSource(self.s,bad,self.f.validate)
    def test_historical_text_tamper_refused_and_not_rewritten(self):
        p=self.f.store/"features/adv_text/000000/metadata.json";p.write_text("{}")
        with self.assertRaises(ArtifactError):list(HistoricalTextSource(self.s,self.p,self.f.validate).iter_outputs())
        self.assertEqual(p.read_text(),"{}")
    def test_historical_text_keeps_exact_query_order(self):
        values=list(HistoricalTextSource(self.s,self.p,self.f.validate).iter_outputs())
        self.assertEqual([r.binding.sample_id for r,_ in values],[q.query_id for q in self.s.manifest.queries])

if __name__=="__main__":unittest.main()
