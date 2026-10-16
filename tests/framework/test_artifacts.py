import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from tbps.artifacts import *
from tbps.contracts import FileRef

class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-tests-");self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.store=ArtifactStore(self.base/"new",new_base=self.base)
    def publish(self,**kwargs):return self.store.publish("unit",{"data.bin":b"abc"},{"schema":"fixture-1"},**kwargs)
    def test_constructor_is_read_only(self):
        self.assertFalse(self.store.root.exists());self.assertEqual(self.store.pending(),())
    def test_round_trip_and_manifest_receipt(self):
        m=self.publish();self.assertEqual(m,self.store.verify("unit"));self.assertTrue((self.store.root/"unit/completed.json").exists())
    def test_immutable_no_overwrite(self):
        self.publish()
        with self.assertRaises(ArtifactError):self.publish()
    def test_provenance_conflict(self):
        self.publish()
        with self.assertRaises(ArtifactError):self.store.verify("unit",expected_provenance={"schema":"wrong"})
    def test_tampered_payload(self):
        self.publish();(self.store.root/"unit/data.bin").write_bytes(b"changed")
        with self.assertRaises(ArtifactError):self.store.verify("unit")
    def test_tampered_manifest(self):
        self.publish();(self.store.root/"unit/manifest.json").write_bytes(b"{}")
        with self.assertRaises(ArtifactError):self.store.verify("unit")
    def test_missing_receipt_not_completed(self):
        self.publish();(self.store.root/"unit/completed.json").unlink()
        with self.assertRaises(FileNotFoundError):self.store.verify("unit")
    def test_extra_file_rejected(self):
        self.publish();(self.store.root/"unit/extra").write_bytes(b"x")
        with self.assertRaises(ArtifactError):self.store.verify("unit")
    def test_invalid_schema(self):
        self.publish()
        path=self.store.root/"unit/completed.json";path.write_bytes(json_bytes({"schema":"unknown","manifest_sha256":"1"*64}))
        with self.assertRaises(ArtifactError):self.store.verify("unit")
    def test_sources_read_only_and_sha_guard(self):
        src=self.base/"source";src.write_bytes(b"source")
        before=(sha_file(src),src.stat().st_mtime_ns)
        self.publish(sources=(FileRef(str(src),sha_file(src)),))
        self.assertEqual(before,(sha_file(src),src.stat().st_mtime_ns))
        src.write_bytes(b"changed")
        with self.assertRaises(ArtifactError):self.store.verify("unit")
    def test_wrong_source_sha_before_write(self):
        src=self.base/"source";src.write_bytes(b"source")
        with self.assertRaises(ArtifactError):self.publish(sources=(FileRef(str(src),"1"*64),))
        self.assertFalse(self.store.root.exists())
    def test_source_changed_during_publish(self):
        src=self.base/"source";src.write_bytes(b"source")
        def change(stage):
            if stage=="before_receipt":src.write_bytes(b"changed")
        with self.assertRaises(ArtifactError):self.publish(sources=(FileRef(str(src),sha_file(src)),),fault=change)
        self.assertFalse((self.store.root/"unit").exists());self.assertEqual(len(self.store.pending()),1)
    def test_faults_preserve_pending_without_publication(self):
        for step in ("after_file","before_receipt","before_publish"):
            with self.subTest(step=step):
                s=ArtifactStore(self.base/step,new_base=self.base)
                def fail(stage):
                    if stage==step:raise RuntimeError("interruption")
                with self.assertRaises(RuntimeError):s.publish("unit",{"x":b"x"},{"schema":"fixture"},fault=fail)
                self.assertFalse((s.root/"unit").exists());self.assertEqual(len(s.pending()),1)
                with self.assertRaises(PendingPublication):s.publish("unit",{"x":b"x"},{})
    def test_real_process_exit_before_receipt(self):
        script="""import os,sys
from tbps.artifacts import ArtifactStore
s=ArtifactStore(sys.argv[1]+'/crash',new_base=sys.argv[1])
def fail(stage):
 if stage=='before_receipt':os._exit(9)
s.publish('unit',{'x':b'x'},{'schema':'fixture'},fault=fail)
"""
        r=subprocess.run([sys.executable,"-B","-c",script,str(self.base)])
        self.assertEqual(r.returncode,9)
        s=ArtifactStore(self.base/"crash",new_base=self.base)
        self.assertEqual(len(s.pending()),1)
        with self.assertRaises(ArtifactError):s.verify("unit")
    def test_concurrent_writer_refused(self):
        with self.store._lock():
            with self.assertRaises(ArtifactError):self.publish()
    def test_historical_roots_refused(self):
        for root in ("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/tbps-framework-new","/tmp/prototypes/tbps-framework-new"):
            with self.subTest(root=root),self.assertRaises(ArtifactError):ArtifactStore(root,new_base=root)
    def test_explicit_protected_fixture_root(self):
        with self.assertRaises(ArtifactError):ArtifactStore(self.base/"new",new_base=self.base,protected_roots=(self.base,))
    def test_unsafe_ids_and_filenames(self):
        for id in ("../escape","/absolute",""):
            with self.subTest(id=id),self.assertRaises(ArtifactError):self.store.publish(id,{"x":b"x"},{})
        for name in ("../escape","/absolute","a//b","manifest.json","completed.json","x\\y"):
            with self.subTest(name=name),self.assertRaises(ArtifactError):self.store.publish("unit",{name:b"x"},{})
    def test_output_symlink_escape(self):
        self.store.root.mkdir();(self.store.root/"unit").symlink_to(self.base,target_is_directory=True)
        with self.assertRaises(ArtifactError):self.store.verify("unit")
    def test_strict_json_duplicates_and_nonfinite(self):
        p=self.base/"bad.json"
        for data in (b'{"a":1,"a":2}',b'{"a":NaN}',b'[]'):
            with self.subTest(data=data):
                p.write_bytes(data)
                with self.assertRaises(ArtifactError):read_json(p)
    def test_tensor_save_compacts_storage(self):
        import torch
        big=torch.arange(512*256,dtype=torch.float32).reshape(256,512);view=big[17:18]
        self.store.publish_tensor("unit",view,{"schema":"tensor-fixture"})
        restored=self.store.load_tensor("unit")
        self.assertTrue(torch.equal(view,restored))
        self.assertEqual(restored.untyped_storage().nbytes(),restored.numel()*restored.element_size())
        self.assertLess((self.store.root/"unit/tensor.pt").stat().st_size,10000)
        self.assertFalse(torch.cuda.is_initialized())
    def test_old_view_load_is_compact_and_value_exact(self):
        import torch
        big=torch.arange(512*256,dtype=torch.float32).reshape(256,512);view=big[17:18]
        stream=io.BytesIO();torch.save(view,stream)
        self.assertGreater(len(stream.getvalue()),500000)
        self.store.publish("unit",{"tensor.pt":stream.getvalue()},{"schema":"legacy-view-fixture"})
        oldsha=sha_file(self.store.root/"unit/tensor.pt")
        restored=self.store.load_tensor("unit")
        self.assertTrue(torch.equal(view,restored));self.assertEqual(restored.untyped_storage().nbytes(),2048)
        self.assertEqual(oldsha,sha_file(self.store.root/"unit/tensor.pt"))
    def test_new_old_tensor_artifacts_coexist(self):
        import torch
        x=torch.arange(4096,dtype=torch.float32).reshape(8,512)[1]
        s=io.BytesIO();torch.save(x,s)
        self.store.publish("old",{"tensor.pt":s.getvalue()},{"format":"historical-view-fixture"})
        self.store.publish_tensor("new",x,{"format":"compact"})
        self.assertTrue(torch.equal(self.store.load_tensor("old"),self.store.load_tensor("new")))

if __name__=="__main__":unittest.main()
