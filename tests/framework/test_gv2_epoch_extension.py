"""续训协议/真实冻结调度的CPU合成测试；不运行真实模型或CUDA。"""
import copy,json,random,tempfile,unittest,io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tbps.artifacts import ArtifactStore,json_bytes,digest,sha_file,FileRef
from tbps.ablations import gv2_epoch_state as s,gv2_epochs as runner

LEGACY="/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014"
et,ck,_=s.legacy({"legacy_root":LEGACY})
from test_phase22 import condition,ToyEncoder
torch.set_num_threads(2)

class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="tbps-framework-gv2-test-",dir="/home/lzf/ldx/tmp")
        self.base=Path(self.temp.name)
        self.parent=self.base/"parent.pt";self.parent.write_bytes(b"synthetic parent; not real training")
        self.conf=self.base/"config.json";self.conf.write_text("{}")
        self.parent_patch=patch.object(s,"PARENT_SHA",sha_file(self.parent));self.parent_patch.start()
        original={"config":{"epochs":10,"architecture":{"width":2,"blocks":1},"optimizer":{"lr":.0002,"betas":[.5,.999]},
            "weights":{"lambda_global":1.,"lambda_local":1.,"lambda_retain":1.},"seed":42}}
        self.c={"path":str(self.conf),"parent_checkpoint":{"path":str(self.parent),"sha256":sha_file(self.parent)},
            "formal_output":str(self.base/"train"),"smoke_output":str(self.base/"probe")}
        self.contract=s.extension_contract(original,self.c)
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        self.g=et.ImageOnlyGenerator(width=2,blocks=1,epsilon=.004)
        self.encoder=ToyEncoder().eval().requires_grad_(False)
        self.opt=torch.optim.Adam(self.g.parameters(),lr=.0002,betas=(.5,.999))
        self.items={};self.entries=[];self.vals=[]
        gen=torch.Generator().manual_seed(99)
        for i,(k,role) in enumerate(((3,"train"),(0,"train"),(1,"validation"))):
            c=condition(k);c.update(row_id=f"cuhk:{20+i}:0",pid=i+1,role=role)
            e={key:c[key] for key in ("row_id","image_rel_path","caption_index","split","role","pid","caption","K")}
            self.items[e["row_id"]]=(torch.rand(1,3,16,8,generator=gen)*.6+.2,c)
            (self.entries if role=="train" else self.vals).append(e)
        self.loader=lambda e:self.items[e["row_id"]]
    def tearDown(self):
        self.parent_patch.stop();self.temp.cleanup()
    def epoch(self,e=11):
        (self.base/"train/logs").mkdir(parents=True,exist_ok=True)
        return runner.run_epoch(et,ck,self.g,self.encoder,self.opt,self.entries,self.vals,self.loader,e,
            self.base/f"train/logs/log{e}.jsonl",et.LossWeights(),limit=1)
    def checkpoint(self,e=11):
        p=self.epoch(e);value=s.pack(self.g,self.opt,e,self.contract,p,ck)
        store=ArtifactStore(self.base/"train/checkpoints",new_base=self.base)
        return store,value
    def publish(self,e=11):
        store,value=self.checkpoint(e);s.save_checkpoint(store,value,formal=False);return store,value
    def protocol_claim(self,e=11):
        a=ArtifactStore(self.base/"train",new_base=self.base)
        a.publish("protocol",{"protocol.json":json_bytes(self.contract)},{"kind":"test"})
        a.publish(f"claim_{e:04d}",{"claim.json":b"{}"},{"kind":"test"})
        return a
    def test_contract_retains_full_original(self):
        original=copy.deepcopy(self.contract["original_training_contract"])
        s.validate_contract(self.contract)
        self.assertEqual(self.contract["original_training_contract"],original)
        self.assertEqual(original["config"]["epochs"],10)
        self.assertEqual(self.contract["effective_training_configuration"]["epochs"],60)
    def test_other_configuration_changes_rejected(self):
        for key,value in (("architecture",{"width":99}),("optimizer",{"lr":1.}),("weights",{}),("seed",100)):
            with self.subTest(key=key):
                c=copy.deepcopy(self.contract);c["effective_training_configuration"][key]=value
                with self.assertRaises(ValueError):s.validate_contract(c)
    def test_original_contract_tampering_rejected(self):
        c=copy.deepcopy(self.contract);c["original_training_contract"]["config"]["seed"]=12
        with self.assertRaises(ValueError):s.validate_contract(c)
    def test_parent_sha_rejected(self):
        c=copy.deepcopy(self.contract);c["parent_checkpoint"]["sha256"]="0"*64
        with self.assertRaises(ValueError):s.validate_contract(c)
    def test_parent_file_sha_rejected(self):
        self.parent.write_bytes(b"tampered")
        with self.assertRaises(ValueError):s.verify_parent_reference(self.c["parent_checkpoint"])
    def test_wrong_epoch_and_points_rejected(self):
        for key,value in (("start_epoch",10),("target_epochs",61),("evaluation_points",[10,60])):
            c=copy.deepcopy(self.contract);c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.validate_contract(c)
    def test_identical_output_rejected(self):
        c=copy.deepcopy(self.contract);c["smoke_output"]=c["formal_output"]
        with self.assertRaises(ValueError):s.validate_contract(c)
    def test_absolute_epoch_order_and_global_rng_isolation(self):
        before=s.tree_hash(ck.rng_state())
        a=et.epoch_order(self.entries,42,11);b=et.epoch_order(list(reversed(self.entries)),42,11)
        self.assertEqual(a,b);self.assertEqual(before,s.tree_hash(ck.rng_state()))
        p=self.epoch(11);self.assertEqual(p["epoch"],11)
        self.assertEqual(p["sample_order"],[a[0]["row_id"]])
    def test_real_serial_scheduler_and_observer(self):
        frozen=s.tree_hash(self.encoder.state_dict())
        with runner.Observation(self.g,self.encoder,self.opt) as o:p=self.epoch()
        n=p["optimizer_steps"]
        self.assertEqual(o.counts,{"G":n+1,"IRRA":n+1,"step":n,"backward":n})
        self.assertEqual(o.inputs[0],o.inputs[n-1])
        self.assertEqual(s.tree_hash(self.encoder.state_dict()),frozen)
        self.assertTrue(all(x>0 for x in o.input_gradient))
    def test_validation_does_not_update_adam_or_rng(self):
        self.epoch();before=s.tree_hash((self.g.state_dict(),self.opt.state_dict(),ck.rng_state()))
        et.validate_epoch(self.g,self.encoder,self.vals,self.loader)
        self.assertEqual(before,s.tree_hash((self.g.state_dict(),self.opt.state_dict(),ck.rng_state())))
    def test_atomic_checkpoint_readback_and_restore(self):
        store,v=self.publish();loaded=s.load_checkpoint(store,11,self.contract,formal=False)
        self.assertEqual(s.tree_hash(v),s.tree_hash(loaded))
        g=et.ImageOnlyGenerator(width=2,blocks=1,epsilon=.004);opt=torch.optim.Adam(g.parameters(),lr=.0002,betas=(.5,.999))
        exact=s.restore(loaded,g,opt,ck)
        self.assertTrue(all(exact[k] for k in ("G_exact","Adam_exact","RNG_exact")))
        self.assertEqual(s.tree_hash(self.opt.state_dict()),s.tree_hash(opt.state_dict()))
    def test_continuous_vs_boundary_restore_cpu_exact(self):
        store,v=self.publish();self.epoch(12)
        wanted=s.tree_hash((self.g.state_dict(),self.opt.state_dict(),ck.rng_state()))
        order=et.epoch_order(self.entries,42,12)
        loaded=s.load_checkpoint(store,11,self.contract,formal=False)
        self.g=et.ImageOnlyGenerator(width=2,blocks=1,epsilon=.004)
        self.opt=torch.optim.Adam(self.g.parameters(),lr=.0002,betas=(.5,.999))
        s.restore(loaded,self.g,self.opt,ck)
        (self.base/"train/logs/log12.jsonl").rename(self.base/"train/logs/original12.jsonl")
        self.epoch(12)
        self.assertEqual(wanted,s.tree_hash((self.g.state_dict(),self.opt.state_dict(),ck.rng_state())))
        self.assertEqual((self.base/"train/logs/log12.jsonl").read_bytes(),(self.base/"train/logs/original12.jsonl").read_bytes())
        self.assertEqual(order,et.epoch_order(self.entries,42,12))
    def test_duplicate_save_refuses(self):
        store,v=self.publish()
        with self.assertRaises(ValueError):s.save_checkpoint(store,v,formal=False)
    def test_interrupted_publication_preserves_pending(self):
        store,v=self.checkpoint()
        def fail(point):
            if point=="before_publish":raise RuntimeError("simulated crash")
        with self.assertRaises(RuntimeError):s.save_checkpoint(store,v,formal=False,fault=fail)
        self.assertFalse((store.root/"epoch_0011").exists());self.assertEqual(len(store.pending()),1)
        with self.assertRaises(ValueError):s.save_checkpoint(store,v,formal=False)
    def test_corrupt_checkpoint_refused(self):
        store,v=self.publish();(store.root/"epoch_0011/checkpoint.pt").write_bytes(b"bad")
        with self.assertRaises(ValueError):s.load_checkpoint(store,11,self.contract,formal=False)
    def test_changed_sample_log_refused(self):
        store,v=self.publish();Path(v["progress"]["sample_log_path"]).write_text("[]")
        with self.assertRaises(ValueError):s.load_checkpoint(store,11,self.contract,formal=False)
    def test_changed_configuration_source_refused(self):
        store,v=self.publish();self.conf.write_text('{"changed":true}')
        with self.assertRaises(ValueError):s.load_checkpoint(store,11,self.contract,formal=False)
    def test_probe_cannot_be_formal_checkpoint(self):
        store,v=self.checkpoint();v.update(schema="gv2-epoch11-smoke-probe-1",completed_epoch=None,resume_as_formal_allowed=False)
        with self.assertRaises(ValueError):s.save_checkpoint(store,v,formal=False)
    def test_wrong_formal_counts_refused(self):
        p=self.epoch()
        with self.assertRaises(ValueError):s.validate_progress(p,formal=True)
    def test_validation_in_training_refused(self):
        with self.assertRaises(ValueError):et.train_epoch(self.g,self.encoder,self.opt,self.vals,self.loader,42,11,et.LossWeights())
    def test_invalid_loss_and_budget_refused(self):
        p=self.epoch()
        for key,value in (("total",float("nan")),("cumulative_linf",.1),("finite_gradients",False)):
            bad=copy.deepcopy(p);bad["sample_logs"][0]["rounds"][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):s.validate_progress(bad,formal=False)
    def test_adam_nonfinite_refused(self):
        self.epoch();state=self.opt.state_dict();next(iter(state["state"].values()))["exp_avg"].fill_(float("nan"))
        with self.assertRaises(ValueError):s.finite_state(state)
    def test_new_and_explicit_boundary_resume(self):
        self.assertIsNone(s.restore_boundary(self.base/"train",self.contract))
        self.protocol_claim();store,v=self.publish()
        with self.assertRaises(ValueError):s.restore_boundary(self.base/"train",self.contract)
        got=s.restore_boundary(self.base/"train",self.contract,resume=True,formal=False)
        self.assertEqual(got["next_epoch"],12)
    def test_resume_mismatched_contract_refused(self):
        self.protocol_claim();self.publish()
        c=copy.deepcopy(self.contract);c["formal_output"]+="-other"
        with self.assertRaises(ValueError):s.restore_boundary(self.base/"train",c,resume=True,formal=False)
    def test_uncommitted_next_epoch_claim_refused(self):
        a=self.protocol_claim();self.publish();a.publish("claim_0012",{"claim.json":b"{}"},{"kind":"test"})
        with self.assertRaises(ValueError):s.restore_boundary(self.base/"train",self.contract,resume=True,formal=False)
    def test_no_fallback_from_corrupt_highest_boundary(self):
        a=self.protocol_claim();store,v=self.publish()
        a.publish("claim_0012",{"claim.json":b"{}"},{"kind":"test"})
        p=self.epoch(12);s.save_checkpoint(store,s.pack(self.g,self.opt,12,self.contract,p,ck),formal=False)
        (store.root/"epoch_0012/checkpoint.pt").write_bytes(b"corrupt")
        with self.assertRaises(ValueError):s.restore_boundary(self.base/"train",self.contract,resume=True,formal=False)
    def test_missing_epoch_gap_refused(self):
        self.protocol_claim(12);store,v=self.checkpoint(12);s.save_checkpoint(store,v,formal=False)
        with self.assertRaises(ValueError):s.restore_boundary(self.base/"train",self.contract,resume=True,formal=False)
    def test_pending_boundary_refuses_resume(self):
        self.protocol_claim();store,v=self.checkpoint()
        def fail(point):
            if point=="before_publish":raise RuntimeError("stop")
        with self.assertRaises(RuntimeError):s.save_checkpoint(store,v,formal=False,fault=fail)
        with self.assertRaises(ValueError):s.restore_boundary(self.base/"train",self.contract,resume=True,formal=False)
    def test_actual_runner_control_flow_with_cpu_mock_encoder(self):
        from types import SimpleNamespace
        # 原Runner执行到底；仅设备/真实模型加载/资源预检由CPU Fixture替换。
        # 算法仍是冻结train_epoch -> train_single_sample；不使用真实权重。
        self.epoch()
        for state in self.opt.state.values():state["step"].fill_(80680)
        original=copy.deepcopy(self.contract["original_training_contract"])
        original["training_source_sha256"]={}
        original["config"]["architecture"]["epsilon"]=.004
        self.contract=s.extension_contract(original,self.c)
        parent={"completed_epoch":10,"generator":ck.cpu_tree(self.g.state_dict()),
            "optimizer":ck.cpu_tree(self.opt.state_dict()),"rng":ck.rng_state()}
        (self.base/"train").rename(self.base/"fixture_previous_logs")
        member=et.epoch_order(self.entries,42,11)[0]
        c={**self.c,"legacy_root":LEGACY,"output_base":str(self.base),"cache":"fixture",
            "smoke_member":{k:member[k] for k in ("row_id","K")},
            "protected_roots":[],"legacy_refs_paths":[str(self.parent)]}
        manifest={"entries":self.entries+self.vals}
        factory=et.ImageOnlyGenerator
        class CPUPixels(factory):
            def to(self,*a,**kw):return super().to("cpu")
        with patch.object(runner,"preflight",return_value=(self.contract,parent,manifest,{"status":"CPU fixture"})), \
             patch.object(runner,"resources",return_value={"policy":"CPU fixture"}), \
             patch.object(et,"ImageOnlyGenerator",CPUPixels), \
             patch.object(et,"make_loader",return_value=self.loader), \
             patch.object(et,"checked_cache",return_value=manifest), \
             patch.object(et,"contract_for",return_value=original), \
             patch.dict("sys.modules",{"irra_image_adapter":SimpleNamespace(load_official=lambda device:ToyEncoder().eval().requires_grad_(False))}), \
             patch.object(torch.cuda,"max_memory_allocated",return_value=0), \
             patch.object(torch.cuda,"max_memory_reserved",return_value=0):
            with runner.no_cuda(),redirect_stdout(io.StringIO()):runner.execute(c,smoke=True)
        root=Path(c["smoke_output"])
        self.assertTrue((root/"completed/report.json").is_file())
        self.assertFalse(Path(c["formal_output"]).exists())
        self.assertFalse((root/"checkpoints").exists())
        probe=torch.load(root/"probe_state/state.pt",map_location="cpu",weights_only=False)
        self.assertIsNone(probe["completed_epoch"]);self.assertFalse(probe["resume_as_formal_allowed"])
        self.assertEqual(s.adam_steps(probe["optimizer"]),[80680+member["K"]+1])
        with self.assertRaises(ValueError):s.restore_boundary(root,self.contract,resume=True,formal=False)

    def test_no_cuda_preflight_gate(self):
        with runner.no_cuda():
            with self.assertRaises(ValueError):torch.cuda._lazy_init()
        self.assertFalse(torch.cuda.is_initialized())
    def test_session_duplicate_lock_refused(self):
        with s.session_lock(self.base):
            with self.assertRaises(ValueError):
                with s.session_lock(self.base):pass
    def test_historical_read_hash_mtime_unchanged(self):
        before=(sha_file(self.parent),self.parent.stat().st_mtime_ns)
        s.verify_parent_reference(self.c["parent_checkpoint"])
        self.assertEqual(before,(sha_file(self.parent),self.parent.stat().st_mtime_ns))
    def test_source_snapshot_change_refused(self):
        c={"protected_roots":[str(self.base)],"legacy_refs_paths":[str(self.parent)]}
        snap=s.snapshot(c);self.parent.write_bytes(b"changed")
        with self.assertRaises(ValueError):s.check_snapshot(snap)

if __name__=="__main__":unittest.main()
