import copy
import json
from pathlib import Path
import tempfile
import unittest
from tbps.artifacts import ArtifactError, sha_file, digest
from tbps.compat.legacy_assets import *
from framework_fixtures import full, checkpoint, asset, finish, mutate, H, I

class LegacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="tbps-framework-legacy-");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def describe(self,a):return describe_asset(a).as_dict()
    def test_full_metadata_readonly_sha_mtime(self):
        a=full(self.root)
        before={p:(sha_file(p),Path(p).stat().st_mtime_ns) for p in [r["path"] for r in a["files"].values()]}
        result=self.describe(a)
        self.assertEqual(result["status"],"METADATA_VERIFIED");self.assertFalse(result["full_feature_audit_performed"])
        self.assertEqual(before,{p:(sha_file(p),Path(p).stat().st_mtime_ns) for p in before})
    def test_missing_path_reports_not_ready(self):
        a=full(self.root);Path(a["files"]["summary"]["path"]).unlink()
        result=self.describe(a);self.assertEqual(result["status"],"NOT_READY");self.assertFalse(result["executable"])
    def test_sha_mismatch_refused(self):
        a=full(self.root);Path(a["files"]["metrics"]["path"]).write_bytes(b"changed")
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_protocol_receipt_conflict_refused(self):
        a=full(self.root);mutate(a,"completed",lambda d:d.update(protocol_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_summary_receipt_conflict_refused(self):
        a=full(self.root);mutate(a,"completed",lambda d:d.update(summary_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_model_provenance_conflict_refused(self):
        a=full(self.root);a["models"]["G"]=H
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_clean_verification_refused(self):
        a=full(self.root);mutate(a,"clean_verified",lambda d:d.update(passed=False))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_storage_compatibility_conflict_refused(self):
        a=full(self.root);mutate(a,"compatibility",lambda d:d.update(legacy_protocol_file_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_completion_false_refused(self):
        a=full(self.root);mutate(a,"completed",lambda d:d.update(method_audit_passed=False))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_order_pairing_and_duplicate_checks(self):
        p=gallery_protocol("fixture")
        self.assertIsInstance(_identities_for_test(p),str)
        for modify in (lambda d:d["queries"][0].update(paired_index=1),
                       lambda d:d["gallery"][0].update(index=1),
                       lambda d:d["queries"][1].update(row_id="q0"),
                       lambda d:d.update(query_count=3)):
            with self.subTest(modify=modify):
                d=copy.deepcopy(p);modify(d)
                with self.assertRaises(ArtifactError):_identities_for_test(d)
    def test_g_checkpoint_opaque_and_protocol_binding(self):
        a=checkpoint(self.root);self.assertEqual(self.describe(a)["status"],"METADATA_VERIFIED")
        mutate(a,"sidecar",lambda d:d.update(contract_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_g_cache_source_conflict_refused(self):
        a=checkpoint(self.root);mutate(a,"protocol",lambda d:d.update(cache_manifest_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_h_selection_argmin_and_train_only(self):
        a=checkpoint(self.root,"hv1_selection");self.assertEqual(self.describe(a)["status"],"METADATA_VERIFIED")
        mutate(a,"selection",lambda d:d.update(uses_official_Test=True))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_h_selection_not_argmin_refused(self):
        a=checkpoint(self.root,"hv1_selection")
        mutate(a,"selection",lambda d:d["candidates"][1]["groups"]["Joint"].update(mAP=1))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_h_selection_tie_earliest(self):
        a=checkpoint(self.root,"hv1_selection")
        mutate(a,"selection",lambda d:d["candidates"][1]["groups"]["Joint"].update(mAP=50))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_h_log_binding_refused(self):
        a=checkpoint(self.root,"hv1_selection");mutate(a,"sidecar",lambda d:d.update(sample_log_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_joint_selection_and_baseline(self):
        a=full(self.root,"joint_full");self.assertEqual(self.describe(a)["status"],"METADATA_VERIFIED")
        mutate(a,"selection",lambda d:d.update(selected_checkpoint_sha256=I))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_joint_all_queries_retained_required(self):
        a=full(self.root,"joint_full");mutate(a,"summary",lambda d:d.update(all_queries_retained=False))
        mutate(a,"completed",lambda d:d.update(summary_sha256=a["files"]["summary"]["sha256"]))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_joint_csv_bound(self):
        a=full(self.root,"joint_full");mutate(a,"completed",lambda d:d.update(metrics_csv_sha256=H))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_ablation_log_success_and_trial_count(self):
        a=full(self.root,"ablation_full");self.assertEqual(self.describe(a)["status"],"METADATA_VERIFIED")
        Path(a["files"]["audit_log"]["path"]).write_text("ABLATION_PIPELINE_SUCCESS\n")
        a["files"]["audit_log"]["sha256"]=sha_file(a["files"]["audit_log"]["path"])
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_ablation_trial_missing(self):
        a=full(self.root,"ablation_full")
        mutate(a,"summary",lambda d:d["trials"].pop("0"))
        mutate(a,"completed",lambda d:d.update(summary_sha256=a["files"]["summary"]["sha256"]))
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_ap_metadata_scope(self):
        a=asset(self.root,"ap_metadata");add=a["_add"]
        m=add("metadata",{"victim":{"irra_checkpoint_sha256":I}})
        add("summary",{"scope":"full","smoke_only":False,"metadata":m})
        a=finish(a);r=self.describe(a)
        self.assertFalse(r["full_feature_audit_performed"]);self.assertEqual(r["reuse_scope"],"read_only_reporting")
    def test_explicit_catalog_assertion(self):
        a=full(self.root);a["checks"]=[{"file":"protocol","pointer":"/gallery_count","equals":3074}]
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_explicit_sha_and_digest_relations(self):
        a=full(self.root)
        a["checks"]=[{"file":"completed","pointer":"/summary_sha256","equals_file_sha":"summary"},
                     {"file":"completed","pointer":"/protocol_sha256","equals_digest":"protocol"}]
        self.assertEqual(self.describe(a)["status"],"METADATA_VERIFIED")
    def test_missing_or_unknown_evidence_declaration(self):
        a=full(self.root);a["files"].pop("completed")
        with self.assertRaises(ArtifactError):self.describe(a)
        a=full(self.root/"new");a["kind"]="rasa_full"
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_duplicate_ids_and_manifest_schema(self):
        a=full(self.root);cat={"schema":"tbps-legacy-catalog-1","baseline_commit":"a"*40,"assets":[a,a]}
        with self.assertRaises(ArtifactError):bridge_manifest(cat)
        cat["assets"]=[a];self.assertEqual(bridge_manifest(cat)["schema"],"tbps-legacy-bridge-1")
        cat["schema"]="unknown"
        with self.assertRaises(ArtifactError):bridge_manifest(cat)
    def test_feature_reuse_never_cross_model_or_unaudited(self):
        a=full(self.root);desc=describe_asset(a)
        for model in (H,I):
            with self.subTest(model=model),self.assertRaises(ArtifactError):desc.require_feature_reuse(model,desc.as_dict()["protocol_digest"])
    def test_descriptor_immutable_copy(self):
        d=describe_asset(full(self.root));copy=d.as_dict();copy["models"]["IRRA"]=H
        self.assertEqual(d.as_dict()["models"]["IRRA"],I)
    def test_catalog_cannot_claim_executable_reuse(self):
        a=full(self.root);a["reuse_scope"]="rasa_feature_reuse"
        with self.assertRaises(ArtifactError):self.describe(a)
    def test_invalid_json_pointer_or_relation(self):
        a=full(self.root);a["checks"]=[{"file":"summary","pointer":"/missing","equals":1}]
        with self.assertRaises(ArtifactError):self.describe(a)
        with self.assertRaises(ArtifactError):pointer({},"bad")

from framework_fixtures import gallery_protocol
from tbps.compat.legacy_assets import _identities as _identities_for_test
if __name__=="__main__":unittest.main()
