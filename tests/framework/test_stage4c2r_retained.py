"""CPU Fixture证明原Plan只读引用、Claim/动作/SHA拒绝；不使用真实模型。"""
from pathlib import Path
import io
import pickle
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactStore,FileRef,sha_file,json_bytes,read_json,ArtifactError
from tbps.materialization.checkpoint_layout import checkpoint_layout
from tbps.materialization.retained_inputs import RetainedInputs
from tbps.materialization.store import MaterializedStore
from tbps.materialization.generation import generate_image_once
import test_stage4c_materialization as fixtures

class LayoutTests(unittest.TestCase):
    def test_layout_no_model_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"checkpoint.pt"
            torch.save({"model":{"w":torch.zeros(3,4)},"optimizer":{"state":{"m":torch.ones(3,4),"v":torch.ones(3,4)}}},p)
            before=p.read_bytes()
            with patch("torch.load",side_effect=AssertionError("model deserialization")), \
                 patch("pickle.loads",side_effect=AssertionError("pickle execution")), \
                 patch("pickle.load",side_effect=AssertionError("pickle execution")):
                d=checkpoint_layout(p)
            self.assertEqual(d["model_storage_bytes"],48);self.assertEqual(d["storage_bytes"],144)
            self.assertEqual(p.read_bytes(),before);self.assertFalse(d["storage_contents_read"])
    def test_unknown_storage_or_corrupt_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.pt"
            with zipfile.ZipFile(p,"w") as z:z.writestr("archive/data.pkl",b"invalid")
            with self.assertRaises((ArtifactError,ValueError)):checkpoint_layout(p)
    def test_global_in_pickle_is_never_executed(self):
        class Evil:
            def __reduce__(self):return (eval,("1/0",))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"bad.pt"
            with zipfile.ZipFile(p,"w") as z:z.writestr("archive/data.pkl",pickle.dumps(Evil(),protocol=2))
            with self.assertRaises(ArtifactError):checkpoint_layout(p)

class RetainedTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.MaterializationTests();self.f.setUp()
        self.root=self.f.root/"attempt"
        self.records=ArtifactStore(self.root,new_base=self.f.root)
        self.store=MaterializedStore(ArtifactStore(self.root/"artifacts",new_base=self.f.root),self.f.plan)
        self.store.publish_plan()
        self.images=[]
        for i in (0,1):
            ref=generate_image_once(self.store,"g"+str(i),self.f.gp,lambda x:self.f.gb.pixels)
            self.images.append({"unit":__import__("dataclasses").asdict(ref)})
        self.texts=[]
        for i in (0,1):
            req=self.store.text_request("q"+str(i));result=self.f.hp.generate(req)
            ref=self.store.publish_text(req,result,self.f.validate)
            self.texts.append({"unit":__import__("dataclasses").asdict(ref)})
        self.g={"status":"PASS","G_loads":1,"model_before":"fixed","model_after":"fixed","images":self.images}
        self.h={"status":"PASS","H_loads":0,"texts":self.texts}
        self.ir={"status":"FAIL","IRRA_loads":0}
        for name,data in (("attempt",{"fixture":True}),("stage-g",self.g),("stage-text",self.h),("stage-irra",self.ir),
                          ("result",{"status":"FAIL","stages":{"g":self.g,"text":self.h,"irra":self.ir}})):
            self.records.publish(name,{"report.json":json_bytes(data)},{"fixture":True})
    def tearDown(self):self.f.tearDown()
    def test_readonly_exact_original_plan_and_units(self):
        before={p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        retained=RetainedInputs(self.root,fixture=True)
        self.assertEqual(retained.plan.id,self.store.plan.id)
        self.assertFalse(hasattr(retained,"publish_image"));self.assertFalse(hasattr(retained,"publish_text"))
        for u in retained.units:
            x,md=retained.load(u,validate=self.f.validate)
            self.assertEqual(md["plan_id"],self.store.plan.id)
        self.assertEqual(retained.descriptor()["original_attempt_status"],"FAIL")
        self.assertFalse(retained.descriptor()["republished"])
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob("*") if p.is_file()})
    def test_real_mode_rejects_another_root(self):
        with self.assertRaisesRegex(ArtifactError,"exact retained"):RetainedInputs(self.root)
    def test_tampered_tensor_refused(self):
        p=Path(self.images[0]["unit"]["data"]["path"]);p.write_bytes(b"bad storage")
        with self.assertRaises(ArtifactError):RetainedInputs(self.root,fixture=True)
    def test_missing_generation_proof_refused(self):
        proof=self.root/"artifacts"/("generation-proof-"+self.images[0]["unit"]["unit_id"])/"completed.json"
        proof.unlink()
        with self.assertRaises((ArtifactError,FileNotFoundError)):RetainedInputs(self.root,fixture=True)
    def test_pending_refused(self):
        (self.root/"artifacts/.pending-fixture").mkdir()
        with self.assertRaisesRegex(ArtifactError,"pending"):RetainedInputs(self.root,fixture=True)
    def test_original_failure_cannot_be_promoted(self):
        # 改写真实证据会先被旧SHA拒绝；即使fixture重新发布完备回执，也必须拒绝整体PASS。
        p=self.root/"result/report.json";d=read_json(p);d["status"]="PASS";p.write_bytes(json_bytes(d))
        with self.assertRaises(ArtifactError):RetainedInputs(self.root,fixture=True)
