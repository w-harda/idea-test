import copy
from dataclasses import replace, fields
from pathlib import Path
import tempfile
import unittest
from tbps.artifacts import ArtifactError, digest, read_json, sha_file, json_bytes
from tbps.contracts import *
from tbps.datasets.cuhk_pedes import CuhkPedes
from tbps.datasets.historical_membership import load_historical_partitions, AttackPartitions
from stage2_fixtures import dataset, historical, mutate, write

class DatasetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage2-data-");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.ds,self.raw=dataset(self.root)
    def partitions(self,refs=None):
        if refs is None:refs=historical(self.root/"legacy",self.ds)
        return load_historical_partitions(self.ds,refs,expected_counts=(2,1))
    def test_official_split_counts(self):
        self.assertEqual(self.ds.counts(),{"train":{"gallery":3,"queries":4,"pids":3},"val":{"gallery":1,"queries":1,"pids":1},"test":{"gallery":3,"queries":4,"pids":2}})
    def test_test_order_multiple_caption_and_positive(self):
        m=self.ds.manifest("test",expected_counts=(3,4))
        self.assertEqual(tuple(g.image_id for g in m.gallery),("cuhk:4:0","cuhk:5:0","cuhk:6:0"))
        self.assertEqual(tuple(q.query_id for q in m.queries),("cuhk:4:0","cuhk:4:1","cuhk:5:0","cuhk:6:0"))
        self.assertEqual(m.evaluation_identity().positives(1),(0,1))
    def test_native_pid_not_training_relabel(self):
        m=self.ds.manifest("train");self.assertEqual(m.gallery[0].pid,PersonID("cuhk_pedes","1"))
    def test_counts_mismatch(self):
        with self.assertRaises(ArtifactError):self.ds.manifest("test",expected_counts=(3074,6156))
    def test_caption_whitespace_span_exact_and_public_fields(self):
        m=self.ds.manifest("test")
        public=self.ds.public_text(m,"cuhk:5:0",(AttributeSpan("color","red","red",2,5),))
        self.assertEqual(public.caption,"  red shirt  ")
        self.assertEqual({f.name for f in fields(type(public))},{"caption","attributes"})
        image=self.ds.public_image(m,"cuhk:4:0")
        self.assertEqual({f.name for f in fields(type(image))},{"content","pixel_space"})
        with self.assertRaises(ValueError):self.ds.public_text(m,"cuhk:5:0",(AttributeSpan("color","red","red",0,3),))
    def test_correct_pairing_all_queries(self):
        m=self.ds.manifest("test");gs={g.image_id:g for g in m.gallery}
        for q in m.queries:self.assertEqual(q.pid,gs[q.paired_image_id].pid)
    def test_raw_hash_change_rejected(self):
        Path(self.ds.annotation.path).write_bytes(b"[]")
        with self.assertRaises(ArtifactError):CuhkPedes(self.ds.annotation,self.ds.image_root)
    def test_unknown_official_split_rejected(self):
        self.raw[0]["split"]="validation"
        with self.assertRaises(ArtifactError):CuhkPedes(write(self.root,"bad.json",self.raw),self.ds.image_root)
    def test_official_pid_leak_rejected(self):
        self.raw[4]["id"]=1
        with self.assertRaises(ArtifactError):CuhkPedes(write(self.root,"bad.json",self.raw),self.ds.image_root)
    def test_unsafe_image_path_rejected(self):
        self.raw[0]["file_path"]="../escape"
        with self.assertRaises(ArtifactError):CuhkPedes(write(self.root,"bad.json",self.raw),self.ds.image_root)
    def test_image_symlink_escape_rejected(self):
        p=self.ds.image_root/"test/a.jpg";p.unlink();outside=write(self.root,"outside",b"x",True);p.symlink_to(outside.path)
        with self.assertRaises(ArtifactError):self.ds.manifest("test")
    def test_forged_query_caption_rejected(self):
        m=self.ds.manifest("test");m=replace(m,queries=(replace(m.queries[0],caption="fake"),*m.queries[1:]))
        with self.assertRaises(ArtifactError):self.ds.public_text(m,"cuhk:4:0")
    def test_protocol_order_equivalence_and_missing_rejection(self):
        m=self.ds.manifest("test");pos={g.image_id:i for i,g in enumerate(m.gallery)}
        p={"gallery_count":3,"query_count":4,"gallery":[{"index":i,"row_id":g.image_id,"image":g.image_rel_path,"image_sha256":g.image_sha256,"pid":int(g.pid.value)} for i,g in enumerate(m.gallery)],
           "queries":[{"index":i,"row_id":q.query_id,"caption":q.caption,"caption_index":q.caption_index,"pid":int(q.pid.value),"paired_index":pos[q.paired_image_id]} for i,q in enumerate(m.queries)]}
        self.assertEqual(self.ds.match_test_protocol(m,p)["queries"],4)
        p["queries"].pop()
        with self.assertRaises(ArtifactError):self.ds.match_test_protocol(m,p)
    def test_verified_actual_internal_members(self):
        p=self.partitions()
        self.assertEqual((len(p.train.queries),len(p.validation.queries)),(2,1))
        self.assertEqual(p.validation.official_split,"train")
        self.assertEqual(self.ds.manifest("val").official_split,"val")
        self.assertFalse({q.pid for q in p.train.queries}&{q.pid for q in p.validation.queries})
    def test_reward_binding_only_train_actual_members(self):
        p=self.partitions();b=p.reward_binding(p.train.queries[0],p.train.gallery)
        self.assertEqual(b.context.paired_index,0);self.assertEqual(b.membership_sha256,p.fingerprint)
        with self.assertRaises(ArtifactError):p.reward_binding(p.validation.queries[0],p.train.gallery)
        with self.assertRaises(ArtifactError):p.reward_binding(p.train.queries[0],(p.train.gallery[0],p.validation.gallery[0]))
    def test_same_role_forged_member_not_accepted(self):
        p=self.partitions()
        for q in (replace(p.train.queries[0],caption="fake"),replace(p.train.queries[0],pid=PersonID("cuhk_pedes","2"))):
            with self.subTest(q=q),self.assertRaises(ArtifactError):p.require_member(q,"attack_train")
    def test_pid_same_as_train_but_unselected_caption_refused(self):
        p=self.partitions();q=replace(p.train.queries[0],query_id="cuhk:0:1",caption_index=1,caption=self.raw[0]["captions"][1])
        with self.assertRaises(ArtifactError):p.require_member(q,"attack_train")
    def test_false_internal_membership_even_role_matches(self):
        refs=historical(self.root/"legacy",self.ds)
        mutate(refs,"internal_split",lambda d:d.update(train_indices=[1,2]))
        with self.assertRaises(ArtifactError):self.partitions(refs)
    def test_h_training_contains_validation_refused(self):
        refs=historical(self.root/"legacy",self.ds)
        mutate(refs,"h_training",lambda d:d["train_identities"][0].update(role="validation"))
        with self.assertRaises(ArtifactError):self.partitions(refs)
    def test_h_gallery_role_or_member_refused(self):
        refs=historical(self.root/"legacy",self.ds)
        mutate(refs,"h_gallery_validation",lambda d:d["identities"][0].update(pid=1))
        with self.assertRaises(ArtifactError):self.partitions(refs)
    def test_h_span_altered_refused(self):
        refs=historical(self.root/"legacy",self.ds)
        mutate(refs,"h_train_qualification",lambda d:d["entries"][0]["attributes"][0].update(raw_text="not red"))
        with self.assertRaises(ArtifactError):self.partitions(refs)
    def test_metadata_sha_mismatch_refused(self):
        refs=historical(self.root/"legacy",self.ds)
        e=read_json(refs["g_manifest"].path)["entries"][0];Path(e["metadata_path"]).write_bytes(b"{}")
        with self.assertRaises(ArtifactError):self.partitions(refs)
    def test_source_sha_and_mtime_unchanged(self):
        refs=historical(self.root/"legacy",self.ds)
        before={r.path:(sha_file(r.path),Path(r.path).stat().st_mtime_ns) for r in refs.values()}
        self.partitions(refs)
        self.assertEqual(before,{r.path:(sha_file(r.path),Path(r.path).stat().st_mtime_ns) for r in refs.values()})

    def test_raw_partition_constructor_cannot_bypass_verification(self):
        p=self.partitions()
        with self.assertRaises(ArtifactError):AttackPartitions(p.train,p.validation,p.sources,p.metadata_refs)

    def test_forged_gallery_sha_even_with_correct_role_refused(self):
        m=self.ds.manifest("test")
        forged=replace(m,gallery=(replace(m.gallery[0],image_sha256="1"*64),*m.gallery[1:]))
        with self.assertRaises(ArtifactError):self.ds.public_text(forged,"cuhk:4:0")
    def test_unknown_query_not_added_to_cached_manifest(self):
        m=self.ds.manifest("test")
        with self.assertRaises(ArtifactError):self.ds.public_text(m,"cuhk:0:0")

    def test_validation_id_in_h_scheduler_refused(self):
        refs=historical(self.root/"legacy",self.ds)
        mutate(refs,"h_training",lambda d:d["query_order"].__setitem__(0,"cuhk:2:0"))
        with self.assertRaisesRegex(ArtifactError,"nontraining"):self.partitions(refs)
    def test_missing_h_scheduler_schema_refused(self):
        refs=historical(self.root/"legacy",self.ds)
        mutate(refs,"h_training",lambda d:d.pop("query_order"))
        with self.assertRaisesRegex(ArtifactError,"schema"):self.partitions(refs)

if __name__=="__main__":unittest.main()
