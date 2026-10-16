"""合成CPU评测；官方rank仅接收小矩阵，不读取历史特征或模型。"""
from dataclasses import replace,asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactError,ArtifactStore,digest,sha_file,json_bytes
from tbps.contracts import (FileRef,PersonID,GalleryItem,QueryItem,DatasetManifest,ModelSpec,
                           AttackBudget,ImageBudget,TextBudget,EvaluationIdentity)
from tbps.models.base import ModelSpace,RepresentationBatch,ScoreBatch,InputBinding
from tbps.engines.metrics import ranking,drops
from tbps.engines.provenance import BoundRepresentation,AttackOrigin,payload_digest
from tbps.engines.evaluation import EvaluationEngine,publish_result,load_result,MODES
from tbps.engines.legacy import LegacySpace,describe_joint_full,require_replay

def h(value):return digest(value)

class NativeMock:
    query_independent_scores=True
    def __init__(self,dim=3,fusion=False,lower=False):
        self.fusion,self.lower=fusion,lower;self.calls=[];self.score_direction="lower_better" if lower else "higher_better"
        self.spec=ModelSpec("fusion_fixture" if fusion else "irra_fixture","1",(5,7),None,
                            None if fusion else dim,"fusion" if fusion else "dot","native",
                            frozenset({"score","fusion"} if fusion else {"score"}))
        self.space=ModelSpace(self.spec.id,h("ckpt"),h("code"),h("pixels"),h("tokenizer"),h("score"),"fp32")
    def slice_query(self,batch,start,end):
        x=batch.payload
        if self.fusion:x={"tokens":x["tokens"][start:end],"captions":x["captions"][start:end]}
        else:x=x[start:end]
        return RepresentationBatch(x,self.space.id,"text")
    def score(self,q,g):
        self.calls.append((q,g))
        assert isinstance(q,RepresentationBatch) and isinstance(g,RepresentationBatch)
        a,b=q.payload,g.payload
        if self.fusion:a,b=a["tokens"].sum(1),b["tokens"].sum(1)
        out=a@b.T
        return ScoreBatch(-out if self.lower else out,self.space.id,self.space.score_id)

class Fixture:
    def __init__(self,root,*,dim=3,fusion=False,lower=False):
        self.root=root;self.adapter=NativeMock(dim,fusion,lower)
        self.file=root/"source.json";self.file.write_bytes(b'{"fixture":true}')
        ref=FileRef(str(self.file),sha_file(self.file));self.ref=ref
        dataset="cuhk_pedes_fixture"
        gs=tuple(GalleryItem("g"+str(i),"images/"+str(i)+".jpg",h(["image",i]),
                 PersonID(dataset,str(i//2)),"test") for i in range(12))
        qs=tuple(QueryItem("q"+str(i),"Caption "+str(i),i%2,"g"+str((i//2)*2),
                 PersonID(dataset,str(i//2)),"test") for i in range(4))
        self.manifest=DatasetManifest(dataset,"test",ref,gs,qs);self.mhash=digest(asdict(self.manifest))
        self.inputs={}
        for fam,n,mod in (("clean_text",4,"text"),("adv_text",4,"text"),("clean_gallery",12,"image"),("adv_gallery",12,"image")):
            # 小整数可精确表示，保证分块/整块score数值相同，而非依赖BLAS舍入。
            values=torch.arange(n*dim,dtype=torch.float32).reshape(n,dim)%7
            if fam.startswith("adv"):values=values.flip(0).clone()
            payload={"tokens":values[:,None,:].repeat(1,2,1),"captions":tuple("public "+str(i) for i in range(n))} if fusion else values
            ids=tuple(x.query_id for x in qs) if mod=="text" else tuple(x.image_id for x in gs)
            originals=tuple(hh(x.caption) for x in qs) if mod=="text" else tuple(x.image_sha256 for x in gs)
            content=tuple(h(["edited",i,mod]) for i in range(n)) if fam.startswith("adv") else originals
            origin=None
            if fam.startswith("adv"):
                budget=AttackBudget(image=ImageBudget("linf",8/255,"rgb_0_1")) if mod=="image" else AttackBudget(
                    text=TextBudget("unicode_character",None,"natural_k","original_unicode",1,1))
                origin=AttackOrigin("gv2" if mod=="image" else "hv1",ref,ref,budget,h("training surrogate"))
            batch=RepresentationBatch(payload,self.adapter.space.id,mod)
            inp=InputBinding(self.adapter.space.id,self.mhash,content,mod,ids)
            self.inputs[fam]=BoundRepresentation(batch,inp,payload_digest(payload),(ref,),origin,originals if origin else ())
    def engine(self,**kwargs):return EvaluationEngine(self.manifest,self.adapter,direction="lower_better" if self.adapter.lower else "higher_better",**kwargs)

def hh(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()

class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage4a-test-")
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.f=Fixture(self.root)

    def test_four_modes_separate_gallery_queries(self):
        p,r,s=self.f.engine().run(self.f.inputs)
        self.assertEqual(tuple(r["groups"]),tuple(MODES))
        self.assertEqual((r["gallery_count"],r["query_count"]),(12,4))
        self.assertEqual(len(self.f.adapter.calls),4)
        for q,g in self.f.adapter.calls:self.assertEqual((len(q.payload),len(g.payload)),(4,12))
        self.assertFalse(p["score_feedback_to_attacks"])

    def test_multi_caption_multi_positive_without_camera_filter(self):
        identity=self.f.manifest.evaluation_identity()
        self.assertEqual(identity.positives(0),(0,1));self.assertEqual(identity.positives(1),(0,1))
        self.assertEqual(self.f.manifest.queries[0].paired_image_id,self.f.manifest.queries[1].paired_image_id)
        score=torch.zeros(4,12);score[:2,0]=10;score[:2,1]=9;score[2:,2]=10;score[2:,3]=9
        values,_=ranking(score,identity);self.assertTrue(all(v==100 for v in values.values()))

    def test_known_ap_inp_and_recall(self):
        identity=EvaluationIdentity(("q",),tuple("g"+str(i) for i in range(10)),
                                    (PersonID("d","p"),),tuple(PersonID("d","p" if i in (1,3) else str(i)) for i in range(10)))
        values,_=ranking(torch.arange(10,0,-1,dtype=torch.float32)[None],identity)
        self.assertEqual(values,{"mAP":50.0,"R@1":0.0,"R@5":100.0,"R@10":100.0,"mINP":50.0})

    def test_dr_formula_zero_clean_and_negative_drop(self):
        clean={k:50. for k in ("mAP","R@1","R@5","R@10","mINP")}
        self.assertEqual(drops(clean,{k:25. for k in clean})["mAP"],50.)
        self.assertEqual(drops(clean,{k:75. for k in clean})["mAP"],-50.)
        clean["mAP"]=0.;self.assertIsNone(drops(clean,clean)["mAP"])

    def test_arbitrary_dimensions(self):
        for dim in (1,7,33):
            with self.subTest(dim=dim):
                f=Fixture(self.root,dim=dim);self.assertEqual(len(f.engine().run(f.inputs)[1]["groups"]),4)

    def test_native_fusion_structures_no_embedding_requirement(self):
        f=Fixture(self.root,dim=7,fusion=True);p,r,s=f.engine().run(f.inputs,query_chunk=3)
        self.assertIsNone(f.adapter.spec.representation_dim);self.assertEqual(s["Joint"].shape,(4,12))

    def test_chunk_scores_and_metrics_exact_including_ties(self):
        _,a,x=self.f.engine().run(self.f.inputs)
        for chunk in (1,3,9):
            _,b,y=self.f.engine().run(self.f.inputs,query_chunk=chunk)
            self.assertEqual(a["groups"],b["groups"])
            for mode in MODES:self.assertTrue(torch.equal(x[mode],y[mode]))

    def test_lower_better_native_scores(self):
        _,a,_=self.f.engine().run(self.f.inputs)
        f=Fixture(self.root,lower=True);_,b,_=f.engine().run(f.inputs)
        self.assertEqual(a["groups"],b["groups"])

    def test_native_direction_conflict_rejected(self):
        with self.assertRaisesRegex(ArtifactError,"direction conflict"):
            EvaluationEngine(self.f.manifest,self.f.adapter,direction="lower_better")

    def test_wrong_result_protocol_rejected(self):
        p,r,s=self.f.engine().run(self.f.inputs);store=ArtifactStore(self.root/"results",new_base=self.root)
        publish_result(store,"one",p,r,s)
        with self.assertRaises(ArtifactError):load_result(store,"one",expected_protocol_sha256=h("other"))

    def test_unknown_direction_rejected(self):
        with self.assertRaises(ArtifactError):EvaluationEngine(self.f.manifest,self.f.adapter,direction="guess")

    def test_wrong_order_rejected(self):
        b=self.f.inputs["clean_text"]
        self.f.inputs["clean_text"]=replace(b,inputs=replace(b.inputs,ordered_ids=b.inputs.ordered_ids[::-1]))
        with self.assertRaisesRegex(ArtifactError,"order"):self.f.engine().run(self.f.inputs)

    def test_missing_member_rejected(self):
        self.f.inputs.pop("adv_text")
        with self.assertRaisesRegex(ArtifactError,"family"):self.f.engine().run(self.f.inputs)

    def test_clean_content_cannot_reuse_edited_binding(self):
        b=self.f.inputs["clean_text"]
        self.f.inputs["clean_text"]=replace(b,inputs=replace(b.inputs,ordered_sha256=(h("wrong"),)*4))
        with self.assertRaisesRegex(ArtifactError,"clean content"):self.f.engine().run(self.f.inputs)

    def test_wrong_attack_original_binding_rejected(self):
        b=self.f.inputs["adv_text"];self.f.inputs["adv_text"]=replace(b,original_sha256=(h("wrong"),)*4)
        with self.assertRaisesRegex(ArtifactError,"original"):self.f.engine().run(self.f.inputs)

    def test_attack_origin_modality_and_unknown_attack_rejected(self):
        b=self.f.inputs["adv_text"]
        with self.assertRaises(ArtifactError):replace(b.origin,attack_id="future")
        with self.assertRaises(ArtifactError):replace(b.origin,attack_id="gv2")

    def test_payload_mutation_rejected(self):
        self.f.inputs["clean_text"].batch.payload[0,0]+=1
        with self.assertRaisesRegex(ArtifactError,"tampered"):self.f.engine().run(self.f.inputs)

    def test_cache_file_tamper_rejected(self):
        self.f.file.write_text("changed")
        with self.assertRaises(ArtifactError):self.f.engine().run(self.f.inputs)

    def test_modelspace_tokenizer_or_preprocess_conflict(self):
        for key in ("tokenizer_sha256","preprocess_sha256","checkpoint_sha256"):
            with self.subTest(key=key):
                old=self.f.adapter.space
                self.f.adapter.space=replace(old,**{key:h("wrong")})
                with self.assertRaisesRegex(ArtifactError,"cache conflict"):self.f.engine().run(self.f.inputs)
                self.f.adapter.space=old

    def test_dataset_manifest_conflict(self):
        b=self.f.inputs["clean_text"];self.f.inputs["clean_text"]=replace(b,inputs=replace(b.inputs,dataset_manifest_sha256=h("wrong")))
        with self.assertRaises(ArtifactError):self.f.engine().run(self.f.inputs)

    def test_score_wrong_shape_rejected(self):
        original=self.f.adapter.score
        self.f.adapter.score=lambda q,g:replace(original(q,g),payload=torch.zeros(1,1))
        with self.assertRaisesRegex(ArtifactError,"score shape"):self.f.engine().run(self.f.inputs)

    def test_score_foreign_model_or_protocol_rejected(self):
        original=self.f.adapter.score
        for field in ("space_id","scoring_protocol_sha256"):
            self.f.adapter.score=lambda q,g,field=field:replace(original(q,g),**{field:h("wrong")})
            with self.assertRaisesRegex(ArtifactError,"foreign"):self.f.engine().run(self.f.inputs)

    def test_scoring_mutation_of_feature_input_rejected(self):
        old=self.f.adapter.score
        def score(q,g):
            result=old(q,g)
            g.payload[0,0]+=1
            return result
        self.f.adapter.score=score
        with self.assertRaisesRegex(ArtifactError,"tampered"):self.f.engine().run(self.f.inputs)

    def test_subset_modes_do_not_drop_query_members(self):
        inputs={k:self.f.inputs[k] for k in ("clean_text","clean_gallery")}
        p,r,s=self.f.engine().run(inputs,modes=("Clean",))
        self.assertEqual(r["query_count"],4);self.assertEqual(set(r["groups"]),{"Clean"})

    def test_score_nonfinite_and_float64_rejected(self):
        original=self.f.adapter.score
        for value in (torch.full((4,12),float("nan")),torch.zeros(4,12,dtype=torch.float64)):
            self.f.adapter.score=lambda q,g:replace(original(q,g),payload=value)
            with self.assertRaises(ArtifactError):self.f.engine().run(self.f.inputs)

    def test_missing_positive_and_pid_namespace_rejected(self):
        with self.assertRaises(ValueError):
            EvaluationIdentity(("q",),("g",),(PersonID("a","1"),),(PersonID("b","1"),))

    def test_gallery_too_small_for_r10_rejected(self):
        identity=EvaluationIdentity(("q",),("g",),(PersonID("a","1"),),(PersonID("a","1"),))
        with self.assertRaisesRegex(ArtifactError,"R@10"):ranking(torch.zeros(1,1),identity)

    def test_chunk_requires_native_capability(self):
        self.f.adapter.query_independent_scores=False
        with self.assertRaisesRegex(ArtifactError,"independence"):self.f.engine().run(self.f.inputs,query_chunk=1)

    def test_duplicate_modes_and_invalid_chunks_rejected(self):
        with self.assertRaises(ArtifactError):self.f.engine().run(self.f.inputs,modes=("Clean","Clean"))
        for n in (0,-1,True):
            with self.assertRaises(ArtifactError):self.f.engine().run(self.f.inputs,query_chunk=n)

    def test_target_scorer_gets_no_identity_or_attack_policy(self):
        self.f.engine().run(self.f.inputs)
        for q,g in self.f.adapter.calls:
            self.assertEqual(set(q.__slots__),{"payload","space_id","modality"})
            self.assertNotIn("pid",repr(q.payload));self.assertNotIn("paired",repr(q.payload))
        import inspect
        self.assertNotIn("attack",inspect.signature(EvaluationEngine.__init__).parameters)

    def test_unimplemented_rasa_aptm_or_dataset_rejected(self):
        for model in ("rasa","aptm"):
            self.f.adapter.spec=replace(self.f.adapter.spec,id=model)
            self.f.adapter.space=replace(self.f.adapter.space,model_id=model)
            with self.assertRaises(ArtifactError):self.f.engine()
        with self.assertRaises(ArtifactError):EvaluationEngine(self.f.manifest,NativeMock(),direction="higher_better",execution="readonly_cached")

    def test_surrogate_view_rejected(self):
        self.f.adapter.input_gradients=True
        with self.assertRaisesRegex(ArtifactError,"Surrogate"):self.f.engine()

    def test_safe_stop_before_scoring(self):
        def stop():raise InterruptedError("safe boundary")
        with self.assertRaises(InterruptedError):self.f.engine().run(self.f.inputs,check_stop=stop)
        self.assertEqual(self.f.adapter.calls,[])

    def test_publish_immutable_bound_source_and_compact_storage(self):
        p,r,s=self.f.engine().run(self.f.inputs)
        store=ArtifactStore(self.root/"results",new_base=self.root)
        result=publish_result(store,"one",p,r,s)
        self.assertIn(str(self.f.file),[x["path"] for x in result["sources"]])
        x=load_result(store,"one",expected_protocol_sha256=r["protocol_sha256"])[2]["Clean"]
        self.assertTrue(torch.equal(x,s["Clean"]))
        self.assertEqual(x.untyped_storage().nbytes(),x.numel()*x.element_size())
        with self.assertRaises(ArtifactError):publish_result(store,"one",p,r,s)

    def test_atomic_interruption_pending_and_no_completion(self):
        p,r,s=self.f.engine().run(self.f.inputs)
        store=ArtifactStore(self.root/"results",new_base=self.root)
        def fault(phase):
            if phase=="before_publish":raise InterruptedError("interrupted")
        with self.assertRaises(InterruptedError):publish_result(store,"one",p,r,s,fault=fault)
        self.assertTrue(store.pending());self.assertFalse((store.root/"one").exists())
        with self.assertRaises(ArtifactError):publish_result(store,"one",p,r,s)

    def test_published_source_tamper_refused(self):
        p,r,s=self.f.engine().run(self.f.inputs);store=ArtifactStore(self.root/"results",new_base=self.root)
        publish_result(store,"one",p,r,s);self.f.file.write_text("changed")
        with self.assertRaises(ArtifactError):store.verify("one")

    def test_persisted_score_wrong_order_even_with_valid_file_hash_rejected(self):
        p,r,s=self.f.engine().run(self.f.inputs);store=ArtifactStore(self.root/"results",new_base=self.root)
        md=publish_result(store,"one",p,r,s)
        files={n:(store.root/"one"/n).read_bytes() for n in md["files_sha256"]}
        record=json.loads(files["score_records.json"])
        record["Clean"]["artifact"]["query_order"].reverse()
        files["score_records.json"]=json_bytes(record)
        store.publish("bad-order",files,md["provenance"])
        with self.assertRaisesRegex(ArtifactError,"order"):
            load_result(store,"bad-order",expected_protocol_sha256=r["protocol_sha256"])

    def test_metric_unit_and_dr_mutation_rejected(self):
        for kind in ("unit","drop"):
            p,r,s=self.f.engine().run(self.f.inputs)
            if kind=="unit":r["metric_unit"]="fraction"
            else:r["drop_rate_percent"]["Joint"]["mAP"]=123.0
            with self.assertRaises(ArtifactError):
                publish_result(ArtifactStore(self.root/"results",new_base=self.root),kind,p,r,s)

    def test_score_mutation_before_publication_rejected(self):
        p,r,s=self.f.engine().run(self.f.inputs);s["Clean"][0,0]+=1
        with self.assertRaises(ArtifactError):publish_result(ArtifactStore(self.root/"results",new_base=self.root),"one",p,r,s)

class OfficialRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=Path("/home/lzf/Attack/IRRA-main/IRRA-main/utils/metrics.py")
        if not p.exists():raise unittest.SkipTest("official CPU source absent; synthetic analytical tests still apply")
        cls.source_sha256=sha_file(p)
        expected=json.loads((Path(__file__).resolve().parents[2]/"configs/framework/stage4a_legacy.server.json").read_text())["legacy_space"]["ranking_source"]["sha256"]
        if cls.source_sha256!=expected:raise AssertionError("official rank source SHA mismatch")
        spec=importlib.util.spec_from_file_location("stage4a_independent_official_rank",p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);cls.native=staticmethod(m.rank)

    def compare(self,score,pids):
        n=len(score);g=len(pids)
        ids=EvaluationIdentity(tuple("q"+str(i) for i in range(n)),tuple("g"+str(i) for i in range(g)),
            tuple(PersonID("d",str(i%2)) for i in range(n)),tuple(PersonID("d",str(p)) for p in pids))
        actual,order=ranking(score,ids)
        cmc,ap,inp,official_order=self.native(score,torch.tensor([i%2 for i in range(n)]),torch.tensor(pids))
        self.assertTrue(torch.equal(order,official_order))
        self.assertEqual(actual,{"mAP":float(ap),"R@1":float(cmc[0]),"R@5":float(cmc[4]),"R@10":float(cmc[9]),"mINP":float(inp)})

    def test_official_independent_random_scores(self):
        gen=torch.Generator().manual_seed(42)
        self.compare(torch.randn(7,19,generator=gen),[i%2 for i in range(19)])

    def test_official_ties_exact_without_stable_override(self):
        self.compare(torch.zeros(5,19),[i%2 for i in range(19)])

    def test_official_nonuniform_multiple_positives(self):
        self.compare(torch.arange(48,dtype=torch.float32).reshape(4,12),[0,0,0,1,1,0,2,3,4,5,6,7])

    def test_import_has_no_torch_or_model_side_effects(self):
        import subprocess,os
        code="import sys; import tbps.engines.evaluation,tbps.engines.legacy; assert 'torch' not in sys.modules; assert not any(k.startswith('prototypes.') for k in sys.modules)"
        subprocess.run([sys.executable,"-B","-c",code],check=True,env=os.environ.copy())

if __name__=="__main__":unittest.main()
