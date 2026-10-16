"""旧Full桥合成元数据；禁止真实特征加载和排名。"""
from dataclasses import asdict,replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tbps.artifacts import ArtifactError,sha_file,digest,json_bytes
from tbps.contracts import FileRef,PersonID
from tbps.engines.legacy import LegacySpace,describe_joint_full,require_replay
from test_stage4a_engine import Fixture,h

class LegacyBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage4a-legacy-");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);f=Fixture(self.root)
        gs=tuple(replace(x,pid=PersonID("cuhk_pedes",x.pid.value)) for x in f.manifest.gallery)
        qs=tuple(replace(x,pid=PersonID("cuhk_pedes",x.pid.value)) for x in f.manifest.queries)
        self.manifest=replace(f.manifest,dataset_id="cuhk_pedes",gallery=gs,queries=qs)
        self.space=LegacySpace("irra",h("IRRA"),h({"pixels":"fixture"}),f.ref,f.ref,f.ref)
        self.protocol={"schema":"hv1-joint-full-1","model_sha256":{"IRRA":h("IRRA")},
            "preprocess":{"pixels":"fixture"},"evidence_sha256":{f.ref.path:f.ref.sha256},
            "gallery_count":len(gs),"query_count":len(qs),"all_queries_retained":True,"Test_feedback_for_selection":False,
            "gallery":[{"index":i,"row_id":g.image_id,"image":g.image_rel_path,"image_sha256":g.image_sha256,"pid":g.pid.value} for i,g in enumerate(gs)],
            "queries":[{"index":i,"row_id":q.query_id,"caption":q.caption,"caption_index":q.caption_index,"pid":q.pid.value,
                        "paired_index":next(j for j,g in enumerate(gs) if g.image_id==q.paired_image_id)} for i,q in enumerate(qs)]}

    def refs(self):
        p=self.root/"protocol.json";p.write_bytes(json_bytes(self.protocol));pr=FileRef(str(p),sha_file(p))
        s={"schema":"hv1-joint-full-result-1","model_sha256":self.protocol["model_sha256"],
            "protocol_sha256":digest(self.protocol),"gallery_count":12,"query_count":4,"all_queries_retained":True,
            "feature_files_sha256":{"features/clean_gallery/000000/features.pt":h("declared only")}}
        p=self.root/"summary.json";p.write_bytes(json_bytes(s));sr=FileRef(str(p),sha_file(p))
        d={"method_audit_passed":True,"protocol_sha256":digest(self.protocol),"summary_sha256":sr.sha256}
        p=self.root/"completed.json";p.write_bytes(json_bytes(d));dr=FileRef(str(p),sha_file(p))
        return pr,sr,dr

    def test_metadata_only_no_torch_load_or_ranking(self):
        import torch
        refs=self.refs();before={r.path:(sha_file(r.path),Path(r.path).stat().st_mtime_ns) for r in refs}
        with patch.object(torch,"load",side_effect=AssertionError("no feature load")):
            index=describe_joint_full(*refs,self.manifest,self.space)
        self.assertEqual(index["status"],"METADATA_VERIFIED_FEATURES_PENDING")
        self.assertFalse(index["replay_executable"]);self.assertEqual(index["features_loaded"],0)
        self.assertEqual(before,{r.path:(sha_file(r.path),Path(r.path).stat().st_mtime_ns) for r in refs})
        with self.assertRaises(ArtifactError):require_replay(index)

    def test_other_victim_rejected(self):
        with self.assertRaises(ArtifactError):replace(self.space,model_id="rasa")

    def test_wrong_model_preprocess_or_tokenizer_rejected(self):
        refs=self.refs()
        for key in ("checkpoint_sha256","preprocess_sha256"):
            with self.subTest(key=key),self.assertRaises(ArtifactError):
                describe_joint_full(*refs,self.manifest,replace(self.space,**{key:h("wrong")}))
        p=self.root/"other";p.write_text("different");ref=FileRef(str(p),sha_file(p))
        with self.assertRaises(ArtifactError):describe_joint_full(*refs,self.manifest,replace(self.space,tokenizer_source=ref))

    def test_protocol_sha_tamper_rejected(self):
        refs=self.refs();Path(refs[0].path).write_text("{}")
        with self.assertRaises(ArtifactError):describe_joint_full(*refs,self.manifest,self.space)

    def test_wrong_order_or_caption_rejected(self):
        self.protocol["gallery"][0],self.protocol["gallery"][1]=self.protocol["gallery"][1],self.protocol["gallery"][0]
        with self.assertRaisesRegex(ArtifactError,"order"):describe_joint_full(*self.refs(),self.manifest,self.space)

    def test_wrong_pair_rejected(self):
        self.protocol["queries"][0]["paired_index"]=2
        with self.assertRaisesRegex(ArtifactError,"paired"):describe_joint_full(*self.refs(),self.manifest,self.space)

    def test_missing_extra_and_truncated_queries_rejected(self):
        self.protocol["queries"].pop()
        with self.assertRaises(ArtifactError):describe_joint_full(*self.refs(),self.manifest,self.space)

    def test_selection_feedback_rejected(self):
        self.protocol["Test_feedback_for_selection"]=True
        with self.assertRaisesRegex(ArtifactError,"isolation"):describe_joint_full(*self.refs(),self.manifest,self.space)

if __name__=="__main__":unittest.main()
