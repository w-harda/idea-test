"""历史分型/只读Loader/统一Engine合成等价，不计算真实Full矩阵。"""
from dataclasses import replace,asdict
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file,read_json,json_bytes,digest
from tbps.contracts import DatasetManifest,PersonID,AttackBudget,ImageBudget,TextBudget
from tbps.models.base import InputBinding,RepresentationBatch
from tbps.engines.provenance import BoundRepresentation,AttackOrigin,payload_digest
from tbps.engines.historical_contracts import HistoricalReplayIdentity,HistoricalFeatureBinding,MODEL_SHAS
from tbps.engines.historical_sources import JointFullSources
from tbps.engines.historical_loader import JointFullLoader,feature_only_cpu
from tbps.models.irra_cached import IRRACachedScorer
from tbps.engines.evaluation import EvaluationEngine,publish_result,load_result
from stage4b_fixtures import ReplayFixture
from test_stage4a_engine import NativeMock

class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage4b1-test-");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.f=ReplayFixture(self.root)
    def engine_inputs(self):
        s,r,v=self.f.load();scorer=IRRACachedScorer(s.space,s.identity)
        e=EvaluationEngine(s.manifest,scorer,direction="higher_better",execution="historical_fixture_replay")
        inputs={"clean_gallery":v["clean_gallery"],"adv_gallery":v["image_gallery"],
                "clean_text":v["clean_text"],"adv_text":v["adv_text"]}
        return s,r,scorer,e,inputs

    def test_real_cached_scorer_default_refuses_even_small_score_call(self):
        s=self.f.sources()
        refs=tuple((name,FileRef(ref.path,MODEL_SHAS[name])) for name,ref in s.identity.models)
        identity=replace(s.identity,models=refs,fixture=False)
        space=replace(s.space,model_id="irra",checkpoint_sha256=MODEL_SHAS["IRRA"])
        # 仅测试执行闸门；来源核验已有独立篡改/真实文件检查，不读取真实模型。
        with patch.object(HistoricalReplayIdentity,"verify"):
            scorer=IRRACachedScorer(space,identity)
        with self.assertRaisesRegex(ArtifactError,"Stage 4B-2"):scorer.score(None,None)
        self.assertEqual(scorer.score_calls,0)

    def test_acceptance_thresholds_cannot_be_relaxed(self):
        from tbps.engines.replay_acceptance import validate_plan,FIXED
        p=self.root/"plan.json";p.write_bytes(json_bytes(FIXED));validate_plan(p)
        for field,value in (("max_rss_bytes",4*1024**3),("max_seconds",3600),
                            ("metric_atol_percentage_points",0.001),("execution_authorized",True),
                            ("query_chunk",64),("torch","other")):
            p.write_bytes(json_bytes({**FIXED,field:value}))
            with self.subTest(field=field),self.assertRaises(ArtifactError):validate_plan(p)

    def test_missing_original_source_or_execution_rejected(self):
        s=self.f.sources()
        Path(s.models[0][1].path).unlink()
        with self.assertRaises((ArtifactError,OSError)):self.f.sources()

    def test_verification_does_not_change_SHA_or_mtime(self):
        before={str(p):(sha_file(p),p.stat().st_mtime_ns) for p in self.f.root.rglob("*") if p.is_file()}
        self.f.load()
        self.assertEqual(before,{str(p):(sha_file(p),p.stat().st_mtime_ns) for p in self.f.root.rglob("*") if p.is_file()})

    def test_member_binding_reversal_rejected(self):
        s,r,scorer,e,inputs=self.engine_inputs()
        b=inputs["adv_text"];binding=replace(b.binding,ordered_ids=b.binding.ordered_ids[::-1])
        inputs["adv_text"]=replace(b,binding=binding)
        with self.assertRaisesRegex(ArtifactError,"order"):e.run(inputs,query_chunk=128)

    def test_mismatched_histories_rejected(self):
        s,r,scorer,e,inputs=self.engine_inputs();b=inputs["adv_text"]
        other=replace(b.identity,legacy_space_sha256=digest("other"))
        binding=replace(b.binding,replay_id=other.id)
        inputs["adv_text"]=replace(b,binding=binding,identity=other)
        with self.assertRaisesRegex(ArtifactError,"mixed historical"):e.run(inputs,query_chunk=128)

    def test_metadata_features_metrics_status_separate(self):
        s=self.f.sources();self.assertEqual(s.report()["status"],"METADATA_VERIFIED")
        _,r,_=self.f.load();self.assertEqual(r["status"],"FEATURES_VERIFIED");self.assertEqual(r["metrics_replay_status"],"PENDING")
        self.assertEqual(r["scores_computed"],0)

    def test_unknown_pixel_SHA_is_required_not_feature_SHA(self):
        s=self.f.sources();self.assertEqual(s.identity.adversarial_pixel_sha256,"UNKNOWN")
        with self.assertRaises(ArtifactError):replace(s.identity,adversarial_pixel_sha256="0"*64)
        with self.assertRaises(ArtifactError):replace(s.identity,artifact_kind="adversarial_rgb")

    def test_cross_victim_or_dataset_rejected(self):
        s=self.f.sources()
        for field,value in (("victim_model","rasa"),("victim_model","aptm"),("dataset","icfg"),("official_split","train")):
            with self.subTest(field=field),self.assertRaises(ArtifactError):replace(s.identity,**{field:value})

    def test_real_profile_frozen_models_rejected(self):
        s=self.f.sources()
        with self.assertRaises(ArtifactError):replace(s.identity,fixture=False)

    def test_not_an_InputBinding_or_BoundRepresentation(self):
        _,_,values=self.f.load();b=values["image_gallery"]
        self.assertNotIsInstance(b,BoundRepresentation);self.assertNotIsInstance(b.binding,InputBinding)
        self.assertNotIn("inputs",b.descriptor());self.assertEqual(b.descriptor()["adversarial_pixel_sha256"],"UNKNOWN")

    def test_all_families_order_norm_finite_and_text_identity(self):
        s,r,v=self.f.load()
        for family,x in v.items():
            self.assertTrue(torch.allclose(x.batch.payload.norm(dim=1),torch.ones(len(x.batch.payload))))
            self.assertTrue(torch.equal(x.batch.payload,self.f.values[family]))
            if family.endswith("text"):self.assertEqual(len(x.binding.text_content_sha256),24)
            else:self.assertEqual(x.binding.text_content_sha256,())

    def test_replay_equals_normal_materialized_fixture_score_metrics(self):
        s,r,scorer,e,inputs=self.engine_inputs()
        p,a,scores=e.run(inputs,query_chunk=128)
        # 正常路径使用独立NativeMock.score和严格BoundRepresentation，不走历史验证分支。
        manifest=replace(s.manifest,dataset_id="cuhk_pedes_fixture",
            gallery=tuple(replace(g,pid=PersonID("cuhk_pedes_fixture",g.pid.value)) for g in s.manifest.gallery),
            queries=tuple(replace(q,pid=PersonID("cuhk_pedes_fixture",q.pid.value)) for q in s.manifest.queries))
        mock=NativeMock(dim=512);mock.space=scorer.space;normal={}
        models=dict(s.identity.models)
        for key,b in inputs.items():
            modality=b.batch.modality;origin=None
            if key.startswith("adv"):
                budget=AttackBudget(image=ImageBudget("linf",8/255,"rgb_0_1")) if modality=="image" else AttackBudget(
                    text=TextBudget("unicode_character",None,"natural_k","fixture_unicode",1,1))
                origin=AttackOrigin("gv2" if modality=="image" else "hv1",models["G" if modality=="image" else "H"],
                                   s.headers["protocol"],budget,digest("surrogate"))
            content=b.binding.text_content_sha256 if modality=="text" else tuple(
                digest(["synthetic materialized pixels",i]) for i in range(len(b.binding.ordered_ids))) if origin else b.binding.original_content_sha256
            binding=InputBinding(mock.space.id,digest(asdict(manifest)),content,modality,b.binding.ordered_ids)
            normal[key]=BoundRepresentation(b.batch,binding,b.payload_sha256,b.cache_sources,origin,
                                           b.binding.original_content_sha256 if origin else ())
        _,expected,other=EvaluationEngine(manifest,mock,direction="higher_better").run(normal,query_chunk=128)
        self.assertEqual(a["groups"],expected["groups"]);self.assertEqual(a["drop_rate_percent"],expected["drop_rate_percent"])
        for mode in scores:self.assertTrue(torch.equal(scores[mode],other[mode]))
        self.assertEqual(scorer.score_calls,4)

    def test_engine_replay_mixed_materialized_is_rejected(self):
        s,r,scorer,e,inputs=self.engine_inputs();inputs["adv_text"]=object()
        with self.assertRaisesRegex(ArtifactError,"mix materialized"):e.run(inputs,query_chunk=128)

    def test_engine_strict_path_does_not_accept_replay(self):
        s,r,scorer,e,inputs=self.engine_inputs()
        strict=EvaluationEngine(replace(s.manifest,dataset_id="cuhk_pedes_fixture",
            gallery=tuple(replace(g,pid=PersonID("cuhk_pedes_fixture",g.pid.value)) for g in s.manifest.gallery),
            queries=tuple(replace(q,pid=PersonID("cuhk_pedes_fixture",q.pid.value)) for q in s.manifest.queries)),
            scorer,direction="higher_better")
        with self.assertRaisesRegex(ArtifactError,"bound representation"):strict.run(inputs)

    def test_score_direction_and_space_conflicts_rejected(self):
        s,r,scorer,e,inputs=self.engine_inputs()
        with self.assertRaises(ArtifactError):IRRACachedScorer(replace(s.space,model_id="aptm"),s.identity)
        with self.assertRaises(ArtifactError):EvaluationEngine(s.manifest,scorer,direction="lower_better",execution="historical_fixture_replay")
        b=inputs["clean_text"]
        with self.assertRaises(ArtifactError):replace(b,binding=replace(b.binding,model_space_id=digest("wrong")))
        # 在representation构造阶段已拒绝foreign space。

    def test_missing_feature_file_or_extra_file_refused(self):
        p=self.f.store/"features/clean_text/000000/features.pt";p.unlink()
        with self.assertRaisesRegex(ArtifactError,"inventory"):self.f.sources()

    def test_pending_never_deleted_or_recovered(self):
        pending=self.f.store/"features/adv_text/000003.pending";pending.mkdir()
        with self.assertRaisesRegex(ArtifactError,"pending"):self.f.sources()
        self.assertTrue(pending.exists())

    def test_feature_sha_tamper_rejected(self):
        p=self.f.store/"features/clean_text/000000/features.pt";p.write_bytes(b"corrupt")
        s=self.f.sources()
        with self.assertRaises(ArtifactError):JointFullLoader(s,self.f.validate).verify_features()

    def test_receipt_conflict_even_with_updated_declared_sha(self):
        p=self.f.store/"features/clean_text/000000/receipt.json";v=read_json(p);v["feature_sha256"]="0"*64
        p.write_bytes(json_bytes(v));self.f.summary["feature_files_sha256"][str(p.relative_to(self.f.store))]=sha_file(p);self.f.resign()
        with self.assertRaisesRegex(ArtifactError,"receipt"):JointFullLoader(self.f.sources(),self.f.validate).verify_features()

    def test_batch_order_mismatch_even_with_resigned_metadata(self):
        p=self.f.store/"features/clean_text/000000/metadata.json";v=read_json(p);v["rows"]=v["rows"][::-1]
        p.write_bytes(json_bytes(v));self.f.summary["feature_files_sha256"][str(p.relative_to(self.f.store))]=sha_file(p);self.f.resign()
        with self.assertRaisesRegex(ArtifactError,"order"):JointFullLoader(self.f.sources(),self.f.validate).verify_features()

    def test_Unicode_caption_action_conflict_rejected(self):
        p=self.f.store/"features/adv_text/000000/metadata.json";md=read_json(p);md["stats"]["actions"][0]["original"]="wrong"
        p.write_bytes(json_bytes(md))
        self.f.summary["feature_files_sha256"][str(p.relative_to(self.f.store))]=sha_file(p)
        rp=p.with_name("receipt.json");receipt=read_json(rp);receipt["metadata_sha256"]=sha_file(p);rp.write_bytes(json_bytes(receipt))
        self.f.summary["feature_files_sha256"][str(rp.relative_to(self.f.store))]=sha_file(rp);self.f.resign()
        with self.assertRaisesRegex(ValueError,"Caption"):JointFullLoader(self.f.sources(),self.f.validate).verify_features()

    def test_source_selection_or_checkpoint_tamper_rejected(self):
        s=self.f.sources();s.selection.path
        Path(s.selection.path).write_text("{}")
        with self.assertRaises(ArtifactError):self.f.sources()

    def test_wrong_dtype_normalization_and_nonfinite(self):
        for x in (torch.zeros(16,512),torch.ones(16,512,dtype=torch.float64),torch.full((16,512),float("nan"))):
            with self.subTest(dtype=x.dtype),patch("torch.load",return_value=x):
                loader=JointFullLoader(self.f.sources(),self.f.validate)
                with self.assertRaises(ArtifactError):loader.verify_features()

    def test_clone_detaches_large_underlying_storage(self):
        s=self.f.sources();loader=JointFullLoader(s,self.f.validate)
        big=torch.zeros(1000,512);big[:12]=self.f.values["clean_gallery"];view=big[:12]
        index,rows=next(loader.parts("clean_gallery"))
        with patch("torch.load",return_value=view):
            x,_,_=loader.load_batch("clean_gallery",index,rows)
        self.assertTrue(torch.equal(x,view));self.assertNotEqual(x.data_ptr(),view.data_ptr())
        self.assertEqual(x.untyped_storage().nbytes(),12*512*4)
        self.assertGreater(view.untyped_storage().nbytes(),x.untyped_storage().nbytes())

    def test_safe_interruption_no_returned_completion_or_writes(self):
        s=self.f.sources();before={str(p):sha_file(p) for p in self.f.store.rglob("*") if p.is_file()}
        def stop():raise InterruptedError("controlled CPU stop")
        with self.assertRaises(InterruptedError):JointFullLoader(s,self.f.validate).verify_features(check_stop=stop)
        self.assertEqual(before,{str(p):sha_file(p) for p in self.f.store.rglob("*") if p.is_file()})

    def test_model_load_and_forward_CUDA_forbidden(self):
        s=self.f.sources();allowed={str(p.resolve()) for p in self.f.store.rglob("features.pt")}
        with feature_only_cpu(allowed):
            with self.assertRaises(ArtifactError):torch.load(dict(s.models)["G"].path,map_location="cpu",weights_only=True)
        for op in (lambda:torch.nn.Linear(1,1),lambda:torch.cuda.init()):
            with self.assertRaises(ArtifactError):
                with feature_only_cpu(allowed):op()
        self.assertFalse(torch.cuda.is_initialized())

    def test_publish_replay_score_identity_does_not_claim_new_pixels(self):
        s,r,scorer,e,inputs=self.engine_inputs();p,report,scores=e.run(inputs,query_chunk=128)
        store=ArtifactStore(self.root/"tbps-framework-results",new_base=self.root)
        publish_result(store,"replay",p,report,scores)
        saved=load_result(store,"replay",expected_protocol_sha256=report["protocol_sha256"])
        self.assertEqual(saved[1],report)
        records=read_json(store.root/"replay/score_records.json")
        self.assertEqual(records["Joint"]["adversarial_pixel_sha256"],"UNKNOWN")
        self.assertTrue(records["Joint"]["not_a_new_attack"])



class NativeActionRulesTests(unittest.TestCase):
    def test_frozen_native_Unicode_span_and_tokenizer_audit_refuses_illegal_action(self):
        import copy,json
        from tbps.engines.historical_loader import native_action_validator
        root=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014")
        path=root/"prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/features/adv_text/000000/metadata.json"
        if not path.exists():self.skipTest("original native CPU rule assets unavailable")
        cfg=read_json(Path(__file__).resolve().parents[2]/"configs/framework/stage4b1_replay.server.json")
        action=read_json(path)["stats"]["actions"][0]
        with feature_only_cpu(set()),native_action_validator(FileRef(**cfg["attack_sources"])) as validate:
            self.assertEqual(validate(action["original"],action),action["adversarial"])
            changed=copy.deepcopy(action);changed["edits"][0]["replacement_codepoint"]="U+0000"
            with self.assertRaises(ValueError):validate(action["original"],changed)
            changed=copy.deepcopy(action);changed["tokenizer_audit"]["model_tokens_changed"]=False
            with self.assertRaises(ValueError):validate(action["original"],changed)
        self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
