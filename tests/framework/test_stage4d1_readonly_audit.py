"""独立全文审计控制流：小型合成CPU Fixture，不读取真实历史资产。"""
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from tbps.artifacts import ArtifactError,json_bytes
from tbps.materialization.throughput import fixture,text_result,validate
from tbps.materialization.bulk_session import StreamingMaterializedStore
from tbps.materialization.full_runner import publish_texts
from tbps.materialization.full_audit import verify_text_units
from tbps.materialization.store import utf8_sha

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-audit-")
        base=fixture(Path(self.temp.name),queries=2,gallery=1,sources=1,checkpoint_bytes=64)
        self.store=StreamingMaterializedStore(base.store,base.plan)
        store=self.store
        class Source:
            complete=False
            def iter_outputs(self):
                for i in range(2):
                    req,result=text_result(store,i);trace=result.trace()
                    trace.update(original_utf8_sha256=utf8_sha(req.public.caption),
                                 adversarial_utf8_sha256=utf8_sha(result.payload))
                    yield req,replace(result,trace_json=json_bytes(trace))
                self.complete=True
            def report(self):
                return {"status":"TEXT_SOURCE_VERIFIED" if self.complete else "PENDING"}
        self.source_type=Source
        self.expected={"edited":2,"NOOP":0,"edits":2}
    def tearDown(self):self.temp.cleanup()
    def publish(self):
        publish_texts(self.store,self.source_type(),validate,expected=self.expected)
    def test_readonly_full_source_exact_and_files_unchanged(self):
        self.publish()
        before={str(p):(p.read_bytes(),p.stat().st_mtime_ns)
                for p in self.store.store.root.rglob("*") if p.is_file()}
        result=verify_text_units(self.store,self.source_type(),validate,self.expected)
        self.assertEqual(result["units"],2)
        self.assertEqual(before,{str(p):(p.read_bytes(),p.stat().st_mtime_ns)
                                for p in self.store.store.root.rglob("*") if p.is_file()})
    def test_missing_unit_rejected(self):
        with self.assertRaises(ArtifactError):
            verify_text_units(self.store,self.source_type(),validate,self.expected)
    def test_corrupt_saved_UTF8_rejected(self):
        self.publish()
        ref=self.store.reference(self.store.text_request("q0").binding)
        Path(ref.data.path).write_text("tampered",encoding="utf-8")
        with self.assertRaises(ArtifactError):
            verify_text_units(self.store,self.source_type(),validate,self.expected)
    def test_wrong_expected_count_rejected(self):
        self.publish()
        with self.assertRaises(ArtifactError):
            verify_text_units(self.store,self.source_type(),validate,{"edited":1,"NOOP":1,"edits":1})
