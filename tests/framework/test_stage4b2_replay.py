"""Stage4B2 CPU合成测试；不调用真实Full或模型。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactError,ArtifactStore,FileRef,sha_file,read_json
from tbps.engines.evaluation import EvaluationEngine,MODES,publish_result,load_result
from tbps.engines.historical_contracts import ENGINE_FAMILIES
from tbps.engines.historical_loader import JointFullLoader,feature_only_cpu
from tbps.engines.replay_control import ResourceGuard,ResourceLimit,historical_snapshot,unchanged
from tbps.engines.replay_oracle import RankOracle,official_rank,verify_feature_values,compare_metrics,compare_drops
from tbps.engines.replay_acceptance import FIXED,validate_plan
from tbps.engines import replay_cpu
from tbps.models.irra_cached import IRRACachedScorer
from stage4b_fixtures import ReplayFixture

class Stage4B2Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-stage4b2-")
        self.root=Path(self.temp.name);self.f=ReplayFixture(self.root)
        self.s=self.f.sources();self.loader=JointFullLoader(self.s,self.f.validate)
        _,self.arrays=self.loader.verify_features(retain=True)
        self.inputs={k:self.arrays[v] for k,v in ENGINE_FAMILIES.items()}
        self.scorer=IRRACachedScorer(self.s.space,self.s.identity)
        self.engine=EvaluationEngine(self.s.manifest,self.scorer,direction="higher_better",execution="historical_fixture_replay")
    def tearDown(self):self.temp.cleanup()
    def native_rank(self):
        path=Path("/home/lzf/Attack/IRRA-main/IRRA-main/utils/metrics.py")
        if not path.exists():self.skipTest("official independent CPU rank source unavailable")
        return official_rank(FileRef(str(path),sha_file(path)))
    def saved(self):
        p,m,x=self.engine.run(self.inputs,query_chunk=128)
        store=ArtifactStore(self.root/"scores",new_base=self.root)
        publish_result(store,"evaluation",p,m,x)
        return store,p,m,x
    def test_prior_identity_canonical_JSON_exact_but_tuple_list_safe(self):
        from dataclasses import asdict
        from tbps.artifacts import json_bytes
        import json
        stored=json.loads(json_bytes(asdict(self.s.identity)))
        replay_cpu.verify_prior_identity(self.s.identity,stored)
        stored["feature_index_sha256"]="0"*64
        with self.assertRaises(ArtifactError):replay_cpu.verify_prior_identity(self.s.identity,stored)
    def test_historical_score_readback_rejects_wrong_shape(self):
        store,p,m,x=self.saved()
        with patch("torch.load",return_value=torch.zeros(1,12)):
            with self.assertRaisesRegex(ArtifactError,"shape/dtype/finite"):
                load_result(store,"evaluation",expected_protocol_sha256=m["protocol_sha256"])
    def test_historical_score_readback_rejects_wrong_dtype(self):
        store,p,m,x=self.saved()
        with patch("torch.load",return_value=torch.zeros(24,12,dtype=torch.float64)):
            with self.assertRaisesRegex(ArtifactError,"shape/dtype/finite"):
                load_result(store,"evaluation",expected_protocol_sha256=m["protocol_sha256"])
    def test_historical_score_readback_rejects_nonfinite(self):
        store,p,m,x=self.saved()
        for val in (float("nan"),float("inf")):
            with self.subTest(value=val),patch("torch.load",return_value=torch.full((24,12),val)):
                with self.assertRaisesRegex(ArtifactError,"shape/dtype/finite"):
                    load_result(store,"evaluation",expected_protocol_sha256=m["protocol_sha256"])
    def test_historical_score_readback_rejects_non_tensor(self):
        store,p,m,x=self.saved()
        with patch("torch.load",return_value={"x":torch.zeros(24,12)}):
            with self.assertRaisesRegex(ArtifactError,"type"):
                load_result(store,"evaluation",expected_protocol_sha256=m["protocol_sha256"])
    def test_valid_historical_readback_is_compact_and_exact(self):
        store,p,m,x=self.saved()
        _,saved,values=load_result(store,"evaluation",expected_protocol_sha256=m["protocol_sha256"])
        self.assertEqual(saved,m)
        for k,t in values.items():
            self.assertTrue(torch.equal(t,x[k]))
            self.assertEqual(t.untyped_storage().nbytes(),24*12*4)
    def test_independent_feature_file_read_is_exact(self):
        result=verify_feature_values(self.loader,self.arrays)
        self.assertTrue(result["feature_values_exact"]);self.assertEqual(result["independent_feature_batches"],6)
    def test_independent_feature_comparison_refuses_mutation(self):
        self.arrays["clean_text"].batch.payload[0].zero_()
        with self.assertRaisesRegex(ArtifactError,"not exact"):verify_feature_values(self.loader,self.arrays)
    def test_official_score_rank_and_metrics_independently_exact(self):
        oracle=RankOracle(self.inputs,self.s.protocol,self.native_rank(),FIXED)
        with oracle.instrument(tuple(MODES)):p,m,x=self.engine.run(self.inputs,query_chunk=128)
        self.assertEqual(oracle.calls,4);self.assertEqual(oracle.score_calls,self.scorer.score_calls)
        for record in oracle.records.values():
            self.assertTrue(record["ranking_indices_exact"])
            self.assertEqual(record["native_score_max_abs_error"],0)
            self.assertEqual(max(record["official_metric_errors_pp"].values()),0)
    def test_wrong_official_indices_fail_and_keep_diagnostics(self):
        fn=self.native_rank()
        def wrong(*args,**kw):
            cmc,ap,inp,indices=fn(*args,**kw)
            indices[:,[0,1]]=indices[:,[1,0]]
            return cmc,ap,inp,indices
        oracle=RankOracle(self.inputs,self.s.protocol,wrong,FIXED)
        with self.assertRaisesRegex(ArtifactError,"indices mismatch"):
            with oracle.instrument(tuple(MODES)):self.engine.run(self.inputs,query_chunk=128)
        self.assertGreater(oracle.records["Clean"]["different_indices"],0)
        self.assertTrue(oracle.records["Clean"]["first_index_mismatches"])
    def test_wrong_reference_score_fails_without_relaxation(self):
        oracle=RankOracle(self.inputs,self.s.protocol,self.native_rank(),FIXED)
        original=self.scorer.score
        def wrong(q,g):
            from tbps.models.base import ScoreBatch
            v=original(q,g)
            return ScoreBatch(v.payload+.01,v.space_id,v.scoring_protocol_sha256)
        with patch.object(self.scorer,"score",side_effect=wrong),self.assertRaisesRegex(ArtifactError,"Score mismatch"):
            with oracle.instrument(tuple(MODES)):self.engine.run(self.inputs,query_chunk=128)
        self.assertGreater(oracle.records["Clean"]["native_score_max_abs_error"],1e-6)
    def test_metric_and_DR_tolerances_fixed(self):
        vals={k:50. for k in ("mAP","R@1","R@5","R@10","mINP")}
        bad={**vals,"mAP":50.00002}
        with self.assertRaises(ArtifactError):compare_metrics(vals,bad,FIXED["metric_atol_percentage_points"])
        with self.assertRaises(ArtifactError):compare_drops({"x":vals},{"x":bad},FIXED["DR_atol_percentage_points"])
    def test_no_explicit_authorization_refuses_before_preflight(self):
        with patch.object(replay_cpu,"preflight") as pre,self.assertRaisesRegex(ArtifactError,"explicit"):
            replay_cpu.execute_once("not-read","not-read")
        pre.assert_not_called()
    def test_noncanonical_retry_output_refused_before_sources(self):
        with self.assertRaisesRegex(ArtifactError,"single fixed"):
            replay_cpu.execute_once("not-read","not-read",authorized=True,output=self.root/"retry")
    def test_existing_attempt_refused_no_retry_or_overwrite(self):
        attempt=self.root/"attempt";attempt.mkdir()
        with patch.object(replay_cpu,"ATTEMPT",attempt),patch.object(replay_cpu,"preflight") as pre:
            with self.assertRaisesRegex(ArtifactError,"already claimed"):
                replay_cpu.execute_once("not-read","not-read",authorized=True,output=attempt)
        pre.assert_not_called()
    def test_default_CLI_only_preflight(self):
        with patch.object(replay_cpu,"preflight",return_value=({"status":"CPU_REPLAY_PREFLIGHT_PASS"},)),patch.object(replay_cpu,"execute_once") as run:
            self.assertEqual(replay_cpu.main([]),0)
        run.assert_not_called()
    def test_explicit_CLI_dispatches_single_authorized_entry(self):
        with patch.object(replay_cpu,"execute_once",return_value=1) as run:
            self.assertEqual(replay_cpu.main(["--authorize-cpu-full-replay"]),1)
        self.assertTrue(run.call_args.kwargs["authorized"]);self.assertEqual(run.call_count,1)
    def test_RSS_budget_boundary_rejected(self):
        g=ResourceGuard(100,100,sample=lambda:100,clock=lambda:0)
        with self.assertRaisesRegex(ResourceLimit,"RSS"):g.check()
    def test_time_budget_boundary_rejected(self):
        now=[0.]
        g=ResourceGuard(100,30,sample=lambda:1,clock=lambda:now[0]);now[0]=30.
        with self.assertRaisesRegex(ResourceLimit,"time"):g.check()
    def test_resource_watch_thread_stops_and_restores_signals(self):
        import signal
        before=signal.getsignal(signal.SIGTERM)
        with ResourceGuard(2**40,10,sample=lambda:1) as g:g.check()
        self.assertFalse(g.thread.is_alive());self.assertEqual(signal.getsignal(signal.SIGTERM),before)
    def test_CPU_feature_guard_prevents_model_checkpoint_load(self):
        with feature_only_cpu(set()),self.assertRaises(ArtifactError):
            torch.load(dict(self.s.models)["G"].path,map_location="cpu",weights_only=True)
        self.assertFalse(torch.cuda.is_initialized())
    def test_frozen_plan_still_false_and_exact(self):
        p=Path(__file__).resolve().parents[2]/"configs/framework/stage4b2_acceptance.json"
        self.assertEqual(validate_plan(p),FIXED);self.assertIs(validate_plan(p)["execution_authorized"],False)
    def test_protection_rejects_content_or_mtime_change(self):
        from tbps.artifacts import json_bytes
        p=self.root/"protected";p.write_bytes(b"fixed")
        index=self.root/"index.json";index.write_bytes(json_bytes({str(p):{"sha256":sha_file(p),"mtime_ns":p.stat().st_mtime_ns}}))
        snapshot=historical_snapshot(index);self.assertEqual(unchanged(snapshot),1)
        p.write_bytes(b"changed")
        with self.assertRaises(ArtifactError):unchanged(snapshot)

if __name__=="__main__":unittest.main()
