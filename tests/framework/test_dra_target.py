"""8/16原生评分Smoke桥的CPU契约；不启动模型。"""
import ast
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
from tbps.artifacts import ArtifactError,read_json
from tbps.baselines.dra_irra import configuration
from tbps.baselines.dra_official import ordered_dataset,MODES
from tbps.baselines.dra_target import rank_function,smoke_metrics,EightGalleryEngine

class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=configuration(Path(__file__).parents[2]/"configs/framework/dra_irra_targets.server.json")
        _,_,cls.manifest=ordered_dataset(cls.config,"smoke");cls.rank=staticmethod(rank_function(cls.config))
    def test_native_rank_with_all_positives(self):
        m=self.manifest;x=torch.zeros(16,8)
        for i,q in enumerate(m.queries):
            for j,g in enumerate(m.gallery):
                if q.pid==g.pid:x[i,j]=1
        value,order=smoke_metrics(x,m,self.rank)
        self.assertEqual(value,{"R@1":100.,"R@5":100.,"mAP":100.});self.assertEqual(order.shape,(16,8))
    def test_no_fake_r10(self):
        result,_=smoke_metrics(torch.ones(16,8),self.manifest,self.rank)
        self.assertNotIn("R@10",result)
    def test_bad_dtype_shape_or_nan(self):
        for x in [torch.zeros(16,8,dtype=torch.float64),torch.zeros(16,9),torch.full((16,8),float("nan"))]:
            with self.assertRaises(ArtifactError):smoke_metrics(x,self.manifest,self.rank)
    def test_modes_fixed_order(self):
        self.assertEqual(MODES,("Clean","Text-only","Image-only","Joint"))
    def test_engine_full_gate_unchanged(self):
        p=Path(__file__).parents[2]/"src/tbps/engines/metrics.py"
        self.assertIn("R@10 requires at least 10 Gallery members",p.read_text())
    def test_real_native_cached_session_required_flags(self):
        import tempfile
        from dataclasses import asdict
        from tbps.artifacts import ArtifactStore,FileRef,json_bytes
        from tbps.models.base import ModelSpace
        from tbps.engines.materialized_cache import CachedIRRA,ref
        old=Path("/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v3/smoke/irra/protocol/protocol.json")
        session=read_json(old)
        checkpoint=FileRef(**self.config["irra"]["checkpoint"]);space=ModelSpace(**session["model_space"])
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/"tbps-framework-fixture";store=ArtifactStore(base/"native",new_base=base)
            session.update(G_H_generation=False,ranking_feedback_to_attack=False)
            store.publish("protocol",{"protocol.json":json_bytes(session)},{"kind":"fixture"},sources=(checkpoint,))
            scorer=CachedIRRA(space,checkpoint,cache_protocol=ref(store.root/"protocol/manifest.json"))
            self.assertTrue(scorer.source_verified)
if __name__=="__main__":unittest.main()
