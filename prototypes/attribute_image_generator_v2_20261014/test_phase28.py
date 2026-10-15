"""Phase2.8 CPU：受控剩余Epoch、最高完整恢复、纯标量监控及最终image-only接口。"""
import copy,unittest
from unittest.mock import patch
import numpy as np
import torch
import cache_io as io
import epoch_checkpoint as ck
import epoch_training as et
import phase28_resume as runner
import phase28_analysis as analysis
from phase26_support import state_hash
from test_phase24 import Base,sample,build_cache,tree_equal
import test_phase24 as fixtures

class ControlledTests(unittest.TestCase):
    def test_start_epoch4_limit7_total10(self):
        calls=[]
        def fake(*args,**kwargs):calls.append(kwargs);return 7
        self.assertEqual(runner.launch_remaining(fake,None,None,{"epochs":10},3),7)
        self.assertEqual(calls,[{"resume":True,"stop_after_epochs":7}])
    def test_resume_limits_highest4_to9(self):
        for n in range(4,10):
            calls=[]
            runner.launch_remaining(lambda *a,**kw:calls.append(kw),None,None,{"epochs":10},n)
            self.assertEqual(calls,[{"resume":True,"stop_after_epochs":10-n}])
    def test_wrong_boundary_or_target_rejected(self):
        for n in (0,2,10,11,True,3.):
            with self.subTest(n=n),self.assertRaises(ValueError):
                runner.launch_remaining(lambda *a,**k:self.fail(),None,None,{"epochs":10},n)
        with self.assertRaises(ValueError):runner.launch_remaining(None,None,None,{"epochs":7},3)
    def test_default_is_preflight_only(self):
        with patch.object(runner,"preflight",return_value=({},{"counts":{}},{"next_epoch":4,"completed_epoch":3})), \
             patch.object(runner,"run",side_effect=AssertionError("GPU forbidden")):runner.main([])
    def test_resume_flag_without_gpu_flag_rejected(self):
        with self.assertRaises(SystemExit):runner.main(["--resume-phase28"])
    def test_scalar_monitor_rejects_detached_tensors_arrays_and_graphs(self):
        for x in (torch.tensor(1.),torch.tensor(1.,requires_grad=True)*2,np.zeros(1),lambda:None):
            with self.subTest(type=type(x)),self.assertRaises(ValueError):analysis.scalar_only({"nested":[x]})
    def test_scalar_monitor_accepts_records(self):
        analysis.scalar_only({"row":"a","refs":[None,1,.2,True],"counts":{"n":8068}})
    def test_scalar_monitor_nonfinite_rejected(self):
        for x in (float("inf"),float("nan"),-float("inf")):
            with self.assertRaises(ValueError):analysis.scalar_only({"value":x})
    def test_scalar_monitor_invalid_key_rejected(self):
        with self.assertRaises(ValueError):analysis.scalar_only({1:"bad"})
    def test_CPU_no_CUDA(self):self.assertFalse(torch.cuda.is_initialized())

class DirectoryTests(Base):
    def setup_versions(self,n=3):
        root=self.root/"run";(root/"checkpoints").mkdir(parents=True)
        for k in range(1,n+1):(root/"checkpoints"/f"epoch_{k:04d}").mkdir()
        return root
    def test_contiguous_highest_and_pending_ignored(self):
        r=self.setup_versions(5);(r/"checkpoints/.pending-interrupted").mkdir()
        self.assertEqual(analysis.published_epochs(r),[1,2,3,4,5])
    def test_gap_or_unauthorized_version_rejected(self):
        r=self.setup_versions(3);(r/"checkpoints/epoch_0002").rmdir()
        with self.assertRaises(ValueError):analysis.published_epochs(r)
        r2=self.root/"other"
        (r2/"checkpoints").mkdir(parents=True)
        for n in range(1,12):(r2/"checkpoints"/f"epoch_{n:04d}").mkdir()
        with self.assertRaises(ValueError):analysis.published_epochs(r2)
    def test_no_checkpoints_or_malformed_name_rejected(self):
        with self.assertRaises(ValueError):analysis.published_epochs(self.root)
        r=self.setup_versions();(r/"checkpoints/epoch_bad").mkdir()
        with self.assertRaises(ValueError):analysis.published_epochs(r)
    def test_bound_round_log_path_discovery(self):
        val={"completed_epoch":4,"progress":{"supplementary_log_sha256":{"/x/epoch_0004_round_audit.jsonl":"sha","/x/val.json":"sha"}}}
        self.assertEqual(str(analysis.round_path(val,self.root)),"/x/epoch_0004_round_audit.jsonl")
        val["progress"]["supplementary_log_sha256"]["/y/epoch_0004_round_audit.jsonl"]="sha"
        with self.assertRaises(ValueError):analysis.round_path(val,self.root)

class IntegrationTests(Base):
    loader=fixtures.ScheduleTests.loader
    def setup_training(self):
        cs=[sample(k,i) for i,k in enumerate((0,1,3,7))]
        cs.append(sample(4,4,"validation"))
        cache=self.root/"cache";build_cache(cache,cs);g,e,_=self.components()
        cfg=et.configuration("synthetic",10,42,"cpu");cfg["architecture"]={"width":2,"blocks":1,"epsilon":.004}
        def run(out,*,resume=False,limit=None):
            return et.run_epochs(cache,out,cfg,resume=resume,stop_after_epochs=limit,
                generator=copy.deepcopy(g),encoder=copy.deepcopy(e),loader=self.loader(cs,True))
        return cache,cfg,g,e,cs,run
    def test_continuous10_matches3_plus7_model_adam_rng_logs(self):
        cache,cfg,g,e,cs,run=self.setup_training()
        ga,oa,pa=run(self.root/"continuous",limit=10);rng=ck.rng_state()
        out=self.root/"resumed";run(out,limit=3)
        hashes={str(p):io.sha(p) for p in (out/"checkpoints").glob("epoch_*/checkpoint.pt")}
        gb,ob,pb=runner.launch_remaining(et.run_epochs,cache,out,cfg,3,generator=copy.deepcopy(g),encoder=copy.deepcopy(e),loader=self.loader(cs,True))
        io.verify(hashes)
        self.assertEqual((pb["completed_epoch"],len(pb["epochs_this_run"])),(10,7))
        self.assertEqual([p["epoch"] for p in pb["epochs_this_run"]],list(range(4,11)))
        self.assertTrue(tree_equal(ga.state_dict(),gb.state_dict()))
        self.assertTrue(tree_equal(oa.state_dict(),ob.state_dict()));self.assertTrue(tree_equal(rng,ck.rng_state()))
        for a,b in zip(pa["epochs_this_run"][3:],pb["epochs_this_run"]):self.assertEqual(a["sample_logs"],b["sample_logs"])
        self.assertFalse((out/"checkpoints/epoch_0011").exists())
        self.assertEqual({int(s["step"]) for s in ob.state.values()},{150})
    def test_interruption_after_epoch5_resumes_only6_to10_preserves_history(self):
        cache,cfg,g,e,cs,run=self.setup_training();out=self.root/"interrupted"
        run(out,limit=3);run(out,resume=True,limit=2)
        hashes={str(p):io.sha(p) for p in (out/"checkpoints").glob("epoch_*/checkpoint.pt")}
        _,opt,p=runner.launch_remaining(et.run_epochs,cache,out,cfg,5,generator=copy.deepcopy(g),encoder=copy.deepcopy(e),loader=self.loader(cs,True))
        io.verify(hashes);self.assertEqual([v["epoch"] for v in p["epochs_this_run"]],[6,7,8,9,10])
        self.assertEqual({int(s["step"]) for s in opt.state.values()},{150})
    def test_highest_corrupt_rejected_before_model_optimizer_restore(self):
        _,_,_,_,_,run=self.setup_training();out=self.root/"bad"
        g,o,p=run(out,limit=3)
        before=state_hash(g.state_dict());(out/"checkpoints/epoch_0003/checkpoint.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"SHA"):ck.load_latest(out,p["contract"],g,o)
        self.assertEqual(state_hash(g.state_dict()),before)
    def test_final_inference_one_forward_without_supervision_imports(self):
        g,e,o=self.components()
        p=ck.save_epoch(self.root,g,o,10,{"config":{"architecture":{"width":2,"blocks":1,"epsilon":.004}}},{})
        result=analysis.inference_check(p/"checkpoint.pt")
        self.assertEqual(result["G_forwards"],1);self.assertEqual(result["shape"],[1,3,384,128])
        self.assertLessEqual(result["maximum_linf"],8/255)

if __name__=="__main__":unittest.main()
