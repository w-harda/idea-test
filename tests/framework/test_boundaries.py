from pathlib import Path
import tempfile
import unittest
from tbps.contracts import *
from tbps.artifacts import ArtifactStore, ArtifactError, digest
from tbps.compat.legacy_assets import describe_asset, bridge_manifest
from tbps.config import ATTACKS, UnsupportedComponent, parse_config
from framework_fixtures import H, full, mutate, config, write, ref

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-boundaries-");self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
    def test_empty_identity_rejected(self):
        p=PersonID("f","1")
        with self.assertRaises(ValueError):EvaluationIdentity(("",),("g",),(p,),(p,))
    def test_official_test_cannot_claim_training_role(self):
        p=PersonID("f","1");g=GalleryItem("g","g.jpg",H,p,"test","attack_train")
        q=QueryItem("q","red",0,"g",p,"test")
        with self.assertRaises(ValueError):DatasetManifest("f","test",FileRef("ann",H),(g,),(q,))
    def test_attack_boundary_cannot_be_mutable(self):
        with self.assertRaises(ValueError):
            AttackSpec("f","v1",frozenset({"image"}),"uap","offline",False,frozenset(),frozenset(),[])
    def test_image_input_has_no_privileged_identity(self):
        with self.assertRaises(TypeError):PublicImageInput(FileRef("x",H),"rgb_0_1",pid="hidden")
    def test_reserved_intent_refused_before_write(self):
        s=ArtifactStore(self.base/"new",new_base=self.base)
        with self.assertRaises(ArtifactError):s.publish("unit",{"intent.json":b"x"},{})
        self.assertFalse(s.root.exists())
    def test_payload_symlink_even_with_valid_hash_refused(self):
        s=ArtifactStore(self.base/"new",new_base=self.base)
        s.publish("unit",{"x":b"x"},{})
        target=s.root/"unit/x";target.unlink();external=self.base/"external";external.write_bytes(b"x");target.symlink_to(external)
        with self.assertRaises(ArtifactError):s.verify("unit")
    def test_child_protocol_binds_parent_digest(self):
        a=full(self.base/"legacy","ablation_full")
        import json
        parent=json.loads(Path(a["files"]["protocol"]["path"]).read_text())
        child=write(self.base,"child.json",{"schema":"fixture-child","parent_sha256":digest(parent)})
        a["files"]["child"]={**ref(child),"format":"json"}
        a["checks"]=[{"file":"child","pointer":"/parent_sha256","equals_digest":"protocol"}]
        self.assertEqual(describe_asset(a).as_dict()["status"],"METADATA_VERIFIED")
        mutate(a,"child",lambda d:d.update(parent_sha256=H))
        with self.assertRaises(ArtifactError):describe_asset(a)
    def test_malformed_catalog_refused_explicitly(self):
        with self.assertRaises(ArtifactError):
            bridge_manifest({"schema":"tbps-legacy-catalog-1","baseline_commit":"a"*40,"assets":[{}]})

    def test_pointer_relation_binds_stored_objects_without_copying_results(self):
        a=full(self.base/"legacy","joint_full")
        a["checks"]=[{"file":"summary","pointer":"/model_sha256","equals_pointer":{"file":"protocol","pointer":"/model_sha256"}}]
        self.assertEqual(describe_asset(a).as_dict()["status"],"METADATA_VERIFIED")
        a["checks"][0]["equals_pointer"]["pointer"]="/missing"
        with self.assertRaises(ArtifactError):describe_asset(a)

if __name__=="__main__":unittest.main()
