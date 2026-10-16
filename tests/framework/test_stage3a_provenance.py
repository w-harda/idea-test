"""条件元数据Fixture及来源拒绝；不加载条件/权重Tensor。"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import json
from tbps.artifacts import ArtifactError, json_bytes, sha_file, digest
from tbps.contracts import FileRef
from tbps.attacks.provenance import LegacyAttackSources

H="1"*64

class ConditionProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def fixture(self,k=2,change_md=None,change_entry=None):
        selected=[{"instance_id":"attr-"+str(i),"S_raw":float(k-i)} for i in range(k)]
        e={"row_id":"cuhk:0:0","image_rel_path":"train/a.jpg","caption_index":0,"split":"train",
           "role":"train","pid":1,"caption":"red shirt","K":k,"tensor_path":str(self.root/"not-loaded.pt"),
           "tensor_sha256":H}
        cp={"schema":"gv2-full-prepare-protocol-2"}
        md={**e,"source_image_sha256":H,"protocol_hash":digest(cp),
            "source_selection":{"selected_attributes":selected,"ranked_attributes":selected,
                                "natural_k":k,"gamma":3,"beta":1},
            "selected_instance_ids":[a["instance_id"] for a in selected],
            "source_score_record":{"row_id":e["row_id"],"pid":1,"caption":e["caption"],
                                   "paired_image":e["image_rel_path"],"attributes":selected,
                                   "retrieval_backward_count":1}}
        if change_md:change_md(md)
        path=self.root/"metadata.json";path.write_bytes(json_bytes(md))
        e.update(metadata_path=str(path),metadata_sha256=sha_file(path))
        if change_entry:change_entry(e)
        manifest={"schema":"gv2-training-cache-2","entries":[e]}
        refs={}
        for name,data in (("cache_protocol",cp),("cache_manifest",manifest)):
            path=self.root/(name+".json");path.write_bytes(json_bytes(data));refs[name]=FileRef(str(path),sha_file(path))
        return SimpleNamespace(ref=lambda component,name:refs[name]),e
    def read(self,source,row="cuhk:0:0"):
        return LegacyAttackSources.condition_record(source,row)
    def test_original_order_scores_image_and_tensor_reference(self):
        source,e=self.fixture()
        r=self.read(source)
        self.assertEqual(r["selected_instance_ids"],["attr-0","attr-1"])
        self.assertEqual(r["S_raw"],[2.,1.]);self.assertEqual(r["K"],2)
        self.assertEqual(r["source_image_sha256"],H)
        self.assertEqual(r["tensor"]["sha256"],H)
        self.assertFalse(Path(e["tensor_path"]).exists())
        self.assertEqual(r["this_run_JVPs"],0)
        self.assertEqual(r["this_run_scoring_backwards"],0)
    def test_K0_preserved(self):
        source,_=self.fixture(k=0);r=self.read(source)
        self.assertEqual(r["K"],0);self.assertEqual(r["S_raw"],[])
        self.assertEqual(r["selected_instance_ids"],[])
    def test_metadata_SHA_tamper_rejected(self):
        source,e=self.fixture();Path(e["metadata_path"]).write_text("{}")
        with self.assertRaisesRegex(ArtifactError,"SHA"):self.read(source)
    def test_missing_row_rejected(self):
        source,_=self.fixture()
        with self.assertRaises(ArtifactError):self.read(source,"cuhk:999:0")
    def test_wrong_protocol_rejected(self):
        source,_=self.fixture(change_md=lambda m:m.update(protocol_hash=H))
        with self.assertRaises(ArtifactError):self.read(source)
    def test_identity_pairing_rejected(self):
        source,_=self.fixture(change_md=lambda m:m["source_score_record"].update(paired_image="wrong.jpg"))
        with self.assertRaisesRegex(ArtifactError,"pairing"):self.read(source)
    def test_selected_ID_order_rejected(self):
        source,_=self.fixture(change_md=lambda m:m.update(selected_instance_ids=list(reversed(m["selected_instance_ids"]))))
        with self.assertRaises(ArtifactError):self.read(source)
    def test_scores_must_be_original_descending(self):
        def change(m):
            m["source_selection"]["selected_attributes"].reverse()
            m["selected_instance_ids"].reverse()
        source,_=self.fixture(change_md=change)
        with self.assertRaises(ArtifactError):self.read(source)
    def test_wrong_dynamic_K_parameters_rejected(self):
        source,_=self.fixture(change_md=lambda m:m["source_selection"].update(gamma=2))
        with self.assertRaises(ArtifactError):self.read(source)
    def test_official_Test_cannot_claim_training_condition(self):
        source,_=self.fixture(change_md=lambda m:m.update(split="test"),change_entry=lambda e:e.update(split="test"))
        with self.assertRaises(ArtifactError):self.read(source)
    def test_raw_score_mismatch_rejected(self):
        def change(m):
            m["source_score_record"]["attributes"]=[{"instance_id":"attr-0","S_raw":7},
                                                     {"instance_id":"attr-1","S_raw":1}]
        source,_=self.fixture(change_md=change)
        with self.assertRaisesRegex(ArtifactError,"raw score"):self.read(source)
    def test_escape_metadata_path_rejected(self):
        source,_=self.fixture(change_entry=lambda e:e.update(metadata_path="/outside/metadata.json"))
        with self.assertRaisesRegex(ArtifactError,"escapes"):self.read(source)
    def test_source_config_changed_after_creation_refused_before_catalog(self):
        p=self.root/"config.json";p.write_bytes(b"{}")
        obj=LegacyAttackSources.__new__(LegacyAttackSources)
        obj.config_ref=FileRef(str(p),sha_file(p));obj.framework_sources={}
        p.write_bytes(b'{"changed":true}')
        with self.assertRaisesRegex(ArtifactError,"SHA"):obj.verify()
    def test_missing_catalog_or_bad_schema_cannot_load_models(self):
        p=self.root/"config.json";p.write_text('{"schema":"wrong"}')
        with self.assertRaises(ArtifactError):LegacyAttackSources(p)

if __name__=="__main__":unittest.main()
