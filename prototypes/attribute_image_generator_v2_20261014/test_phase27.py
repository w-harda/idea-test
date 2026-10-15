"""受控恢复与Epoch 1–3分析的CPU测试，不使用正式GPU。"""
import copy,unittest
from unittest.mock import patch
import torch
import cache_io as io
import epoch_checkpoint as ck
import epoch_training as et
import phase27_resume as runner
import phase27_analysis as analysis
from phase26_support import state_hash,FIELDS
from test_phase24 import Base,sample,build_cache,tree_equal
import test_phase24 as fixtures

class ControlledTests(unittest.TestCase):
    def test_only_resume_two_new_epochs_persistent_ten(self):
        config={"epochs":10};seen=[]
        def fake(cache,out,config,**kw):seen.append((config,kw));return "ok"
        self.assertEqual(runner.launch_two(fake,"cache","out",config),"ok")
        self.assertEqual(seen,[({"epochs":10},{"resume":True,"stop_after_epochs":2})])
        self.assertEqual(config,{"epochs":10})
    def test_target_change_rejected_before_runner(self):
        with self.assertRaises(ValueError):runner.launch_two(lambda *a,**k:self.fail(),None,None,{"epochs":3})
    def test_state_boundary_exact_and_mismatch_rejected(self):
        g=torch.nn.Linear(2,2);o=torch.optim.Adam(g.parameters())
        for p in g.parameters():o.state[p]={"step":torch.tensor(8068.),"exp_avg":torch.zeros_like(p),"exp_avg_sq":torch.zeros_like(p)}
        value={"completed_epoch":1,"next_epoch":2,"generator":ck.cpu_tree(g.state_dict()),
               "optimizer":ck.cpu_tree(o.state_dict()),"rng":ck.rng_state()}
        runner.assert_restored(g,o,value,1)
        for key in ("completed_epoch","next_epoch"):
            bad=copy.deepcopy(value);bad[key]=99
            with self.subTest(key=key),self.assertRaises(ValueError):runner.assert_restored(g,o,bad,1)
        bad=copy.deepcopy(value);bad["generator"]["weight"].add_(1)
        with self.assertRaises(ValueError):runner.assert_restored(g,o,bad,1)
        bad=copy.deepcopy(value);bad["optimizer"]["state"][0]["exp_avg"].add_(1)
        with self.assertRaises(ValueError):runner.assert_restored(g,o,bad,1)
        torch.rand(1)
        with self.assertRaises(ValueError):runner.assert_restored(g,o,value,1)
        ck.restore_rng(value["rng"])
    def test_no_CUDA_initialized(self):self.assertFalse(torch.cuda.is_initialized())
    def test_default_cli_only_preflight(self):
        with patch.object(runner,"preflight",return_value=({},{"counts":{}},{"next_epoch":2})), \
             patch.object(runner,"run",side_effect=AssertionError("training forbidden")):
            runner.main([])

class AnalysisTests(unittest.TestCase):
    def fixtures(self):
        train={"row_id":"a","role":"train","K":1,"selected_instance_ids":["attr"]}
        val={"row_id":"b","role":"validation","K":0}
        rows=[]
        for index,phase in enumerate(("attribute","joint"),1):
            r={field:.1 for field in FIELDS}
            r.update(row_id="a",K=1,sample_position=1,update_index=index,phase=phase,
                     instance_id="attr" if phase=="attribute" else None,
                     cumulative_linf=.01,finite_gradients=True)
            rows.append(r)
        p={"epoch":2,"samples":1,"optimizer_steps":2,"sample_order":["a"],"sample_logs":[
            {"row_id":"a","role":"train","K":1,"sample_position":1,"rounds":copy.deepcopy(rows)}],
            "validation":[{"row_id":"b","optimizer_steps":0,"global":.3,"local":.01,"cumulative_linf":.01}]}
        return p,[train,val],rows
    def test_valid_epoch_and_round_groups(self):
        p,e,r=self.fixtures();v=analysis.audit_epoch(p,e,r,2)
        self.assertEqual((v["attribute_updates"],v["joint_updates"],v["optimizer_steps"]),(1,1,2))
        self.assertEqual(v["statistics"]["K1_attribute_round1"]["rounds"],1)
        self.assertEqual(v["statistics"]["attribute_round7"]["rounds"],0)
    def test_identity_order_steps_and_round_tampering_rejected(self):
        p,e,r=self.fixtures()
        for field,value in (("sample_order",["bad"]),("samples",2),("optimizer_steps",3),("epoch",3)):
            bad=copy.deepcopy(p);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):analysis.audit_epoch(bad,e,r,2)
        bad=copy.deepcopy(r);bad[0]["instance_id"]="wrong"
        with self.assertRaises(ValueError):analysis.audit_epoch(p,e,bad,2)
    def test_budget_nonfinite_and_projection_ratio_rejected(self):
        p,e,r=self.fixtures()
        for field,value in (("cumulative_linf",.04),("G_gradient_l2",float("nan")),("budget_saturation_ratio",1.1)):
            badr=copy.deepcopy(r);badp=copy.deepcopy(p)
            badr[0][field]=value;badp["sample_logs"][0]["rounds"][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):analysis.audit_epoch(badp,e,badr,2)
    def test_missing_round_or_sample_rejected(self):
        p,e,r=self.fixtures();p["sample_logs"]=[]
        with self.assertRaises(ValueError):analysis.audit_epoch(p,e,r,2)
        p,e,r=self.fixtures();p["sample_logs"][0]["rounds"].pop()
        with self.assertRaises(ValueError):analysis.audit_epoch(p,e,r,2)
    def test_validation_update_and_identity_rejected(self):
        for field,value in (("optimizer_steps",1),("row_id","wrong"),("global",float("inf"))):
            p,e,r=self.fixtures();p["validation"][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):analysis.audit_epoch(p,e,r,2)
    def test_fixed_validation_paired_comparison(self):
        a=[{"row_id":"a","global":1.,"local":.5},{"row_id":"b","global":.5,"local":.2}]
        b=[{"row_id":"b","global":.6,"local":.1},{"row_id":"a","global":.8,"local":.6}]
        v=analysis.validation_comparison(a,b)
        self.assertAlmostEqual(v["global"]["mean_delta"],-.05)
        self.assertEqual((v["global"]["decreased"],v["global"]["increased"]),(1,1))
        with self.assertRaises(ValueError):analysis.validation_comparison(a,b[:1])

class ResumeIntegrationTests(Base):
    loader=fixtures.ScheduleTests.loader
    def test_resume_one_then_two_stops_at_three_exact_and_preserves_epoch1(self):
        cs=[sample(0),sample(1,1),sample(3,2),sample(4,3,"validation")]
        cache=self.root/"cache";build_cache(cache,cs)
        g0,e0,_=self.components();cfg=et.configuration("synthetic",10,42,"cpu")
        cfg["architecture"]={"width":2,"blocks":1,"epsilon":.004}
        def run(out,resume,limit):
            return et.run_epochs(cache,out,cfg,resume=resume,stop_after_epochs=limit,
                generator=copy.deepcopy(g0),encoder=copy.deepcopy(e0),loader=self.loader(cs,True))
        ga,oa,pa=run(self.root/"continuous",False,3);rng_a=ck.rng_state()
        run(self.root/"resumed",False,1)
        old=self.root/"resumed/checkpoints/epoch_0001/checkpoint.pt";sha=io.sha(old)
        gb,ob,pb=run(self.root/"resumed",True,2)
        self.assertEqual(pb["completed_epoch"],3)
        self.assertEqual([p["epoch"] for p in pb["epochs_this_run"]],[2,3])
        self.assertEqual(cfg["epochs"],10);self.assertEqual(io.sha(old),sha)
        self.assertFalse((self.root/"resumed/checkpoints/epoch_0004").exists())
        self.assertTrue(tree_equal(ga.state_dict(),gb.state_dict()))
        self.assertTrue(tree_equal(oa.state_dict(),ob.state_dict()))
        self.assertTrue(tree_equal(rng_a,ck.rng_state()))
        for a,b in zip(pa["epochs_this_run"][1:],pb["epochs_this_run"]):
            self.assertEqual(a["sample_logs"],b["sample_logs"])

if __name__=="__main__":unittest.main()
