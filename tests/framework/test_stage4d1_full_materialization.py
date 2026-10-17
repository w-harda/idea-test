"""新全量调度/会话保护CPU Fixture；没有真实模型恢复或GPU。"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import tempfile
import os
import unittest
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,sha_file,FileRef,read_json
from tbps.materialization.throughput import fixture,text_result,validate
from tbps.materialization.bulk_session import VerifiedSources,StreamingMaterializedStore,BulkSession
from tbps.materialization.full_runner import ValidatedActions,publish_texts,stream_images
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.attacks.gv2 import GV2Plugin
from stage3a_fixtures import support,GBackend

class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-full-");self.root=Path(self.temp.name)
        base=fixture(self.root,queries=8,gallery=2,sources=2,checkpoint_bytes=1024)
        self.store=StreamingMaterializedStore(base.store,base.plan)
    def tearDown(self):self.temp.cleanup()
    def test_cached_source_not_rehashed_but_force_boundary_is_full(self):
        with BulkSession(self.store) as session:
            p=self.store.plan.h.checkpoint
            with patch("tbps.materialization.bulk_session.checked",side_effect=AssertionError("unexpected hash")):
                session.cache.check(p)
            with self.assertRaises(AssertionError),patch("tbps.materialization.bulk_session.checked",side_effect=AssertionError("boundary hash")):
                self.store.verify_plan(force=True)
    def test_stat_checks_reject_change_even_if_mtime_restored(self):
        with self.assertRaises(ArtifactError),BulkSession(self.store) as session:
            p=Path(self.store.plan.h.checkpoint.path);s=p.stat()
            p.write_bytes(b"x"*s.st_size);os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns))
            session.cache.check(self.store.plan.h.checkpoint)
    def test_known_source_different_expected_SHA_rejected(self):
        with BulkSession(self.store) as session:
            ref=self.store.plan.h.checkpoint
            with self.assertRaises(ArtifactError):session.cache.check(FileRef(ref.path,"0"*64))
    def test_unknown_output_file_still_SHA_verified(self):
        with BulkSession(self.store) as session:
            p=self.root/"new-output";p.write_bytes(b"new")
            with self.assertRaises(ArtifactError):session.cache.check(FileRef(str(p),"0"*64))
    def test_namespace_patch_restored_after_exception(self):
        import tbps.artifacts as a
        import tbps.materialization.store as m
        before=a.checked,m.checked
        with self.assertRaises(RuntimeError),BulkSession(self.store):raise RuntimeError("fixture abort")
        self.assertEqual((a.checked,m.checked),before);self.assertIsNone(self.store.session)
    def test_no_nested_or_concurrent_cache(self):
        with BulkSession(self.store):
            with self.assertRaises(ArtifactError):BulkSession(self.store).__enter__()
    def test_index_matches_original_plan_binding(self):
        for q in self.store.plan.manifest.queries:
            self.assertEqual(self.store.text_request(q.query_id).binding,self.store.plan.binding(q.query_id,"text"))
    def test_text_member_lookup_does_not_rescan_plan(self):
        with patch.object(type(self.store.plan),"binding",side_effect=AssertionError("full Manifest scan")):
            req=self.store.text_request("q7");self.store.unit_id(req.binding)
    def test_action_cache_exact_full_record_and_bounded(self):
        calls=[]
        def native(c,a):calls.append(a);return validate(c,a)
        memo=ValidatedActions(native,maxsize=2);_,result=text_result(self.store,0);a=result.trace()["action"]
        self.assertEqual(memo("red shirt",a),memo("red shirt",dict(a)));self.assertEqual(len(calls),1)
        with self.assertRaises(ValueError):memo("different",a)
        self.assertEqual(memo.misses,1)
    def source(self,*,reverse=False):
        store=self.store
        class Source:
            complete=False
            def iter_outputs(self):
                indices=range(7,-1,-1) if reverse else range(8)
                for i in indices:yield text_result(store,i)
                self.complete=True
            def report(self):return {"status":"TEXT_SOURCE_VERIFIED" if self.complete else "PENDING"}
        return Source()
    def test_full_text_collection_has_original_order_and_not_joint(self):
        report,refs=publish_texts(self.store,self.source(),validate,expected={"edited":8,"NOOP":0,"edits":8})
        self.assertEqual(report["units"],8)
        c=read_json(report["collection"]["path"])
        self.assertEqual(c["status"],"TEXT_FULL_MATERIALIZED")
        self.assertEqual(c["ordered_query_ids"],["q"+str(i) for i in range(8)])
        self.assertEqual(c["joint_collection_status"],"PENDING");self.assertEqual(c["H_forwards"],0)
    def test_pilot_cannot_mark_full(self):
        report,refs=publish_texts(self.store,self.source(),validate,limit=2)
        self.assertEqual(len(refs),2);self.assertIsNone(report["collection"])
        self.assertFalse((self.store.store.root/"text-full").exists())
    def test_wrong_query_order_rejected(self):
        with self.assertRaises(ArtifactError):publish_texts(self.store,self.source(reverse=True),validate)
        self.assertFalse((self.store.store.root/"text-full").exists())
    def test_wrong_expected_counts_cannot_publish_completion(self):
        with self.assertRaises(ArtifactError):publish_texts(self.store,self.source(),validate,expected={"edited":7,"NOOP":1,"edits":7})
        self.assertFalse((self.store.store.root/"text-full").exists())
    def test_source_change_prevents_full_semantic_receipt(self):
        original=self.store.verify_plan;count=[0]
        def force(*,force=False):
            if force:
                count[0]+=1
                if count[0]>=2:raise ArtifactError("source changed at seal")
            return original(force=force)
        with patch.object(self.store,"verify_plan",side_effect=force),self.assertRaises(ArtifactError):
            publish_texts(self.store,self.source(),validate)
        self.assertFalse((self.store.store.root/"text-full").exists())
    def test_native_validation_error_preserved_no_completion(self):
        def bad(c,a):raise ArtifactError("illegal Unicode")
        with self.assertRaises(ArtifactError):publish_texts(self.store,self.source(),bad)
        self.assertFalse((self.store.store.root/"text-full").exists())
    def test_new_CPU_entry_has_no_GPU_flag(self):
        from tbps.materialization.full_prepare import main
        with self.assertRaises(SystemExit):main(["--config","fixture","--allow-gpu-smoke"])
    def test_G_stream_single_factory_and_no_second_forward_on_complete(self):
        with legacy_imports(support(),allow_import=True) as old:
            backend=GBackend(old("inference_once"),pixels=torch.full((1,3,384,128),.5))
            plugin=GV2Plugin(backend=backend,provenance=self.store.plan.g)
            calls=[]
            @contextmanager
            def factory():
                calls.append("load");yield plugin
            report=stream_images(self.store,factory)
            self.assertEqual(report["generated"],2);self.assertEqual(len(calls),1)
            self.assertEqual(report["H_loads"],report["IRRA_loads"],0)
            with patch.object(plugin,"generate",side_effect=AssertionError("repeat G")):
                again=stream_images(self.store,factory)
                self.assertEqual(again["generated"],0);self.assertEqual(again["skipped"],2)
            self.assertEqual(len(calls),1)
    def test_G_uncommitted_claim_rejected_before_factory(self):
        req=self.store.image_request("g0");unit=self.store.unit_id(req.binding)
        self.store.store.publish("generation-claim-"+unit,{"generation.json":b"{}"},{"fixture":True})
        with patch("builtins.print"),self.assertRaises(ArtifactError):
            stream_images(self.store,lambda:(_ for _ in ()).throw(AssertionError("model loaded")))
    def test_G_pending_rejected_before_factory(self):
        (self.store.store.root/".pending-fixture").mkdir()
        with self.assertRaises(ArtifactError):stream_images(self.store,lambda:(_ for _ in ()).throw(AssertionError("model loaded")))
