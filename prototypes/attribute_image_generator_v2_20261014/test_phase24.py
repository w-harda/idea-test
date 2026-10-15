"""Phase 2.4仅CPU：事务中断、Epoch恢复、缓存协议、调度隔离和单步推理。"""
import copy,json,os,random,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import cache_io as io
import training_data as data
import prepare_full as prep
import epoch_checkpoint as ck
import epoch_training as et
import inference_once as infer
from test_phase22 import ToyEncoder,condition,CountingSGD
from image_only_generator import ImageOnlyGenerator
from serial_training import LossWeights
torch.set_num_threads(2)

def sample(k,index=0,role="train"):
    c=condition(k)
    c.update(row_id=f"cuhk:{10+index}:0",image_rel_path=f"synthetic/{index}.png",
             pid=100+index,role=role)
    return c

def metadata(c):
    return {"source_selection":{"natural_k":c["K"],"selected_attributes":[
        {"instance_id":i,"S_raw":s} for i,s in zip(c["selected_instance_ids"],c["S_raw"].tolist())]},
        "source_image_sha256":"synthetic","new_JVPs":c["K"],"new_JVP_autograd_grad_calls":0}

def rows_for(conditions):
    return [({"row_id":c["row_id"]},{"source":"synthetic"},data.identity(c)) for c in conditions]

def build_cache(out,conditions):
    values={c["row_id"]:c for c in conditions}
    def producer(e,md):
        c=values[e["row_id"]];return c,metadata(c)
    return prep.prepare(out,rows_for(conditions),{},"synthetic",producer)

def tree_equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b

class Base(unittest.TestCase):
    def setUp(self):
        io.OUTPUTS.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix="phase24_cpu_",dir=io.OUTPUTS)
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def components(self):
        torch.manual_seed(71)
        g=ImageOnlyGenerator(width=2,blocks=1,epsilon=.004);enc=ToyEncoder()
        opt=torch.optim.Adam(g.parameters(),lr=.001,betas=(.5,.999))
        return g,enc,opt

class CacheTests(Base):
    def test_schema_and_K0_reconstruction(self):
        m=build_cache(self.root/"cache",[sample(0),sample(3,1),sample(4,2,"validation")])
        self.assertEqual(m["counts"],{"train":2,"validation":1})
        c,_,r=io.read_entry(m["entries"][0])
        self.assertEqual(tuple(c["attribute_jvps"].shape),(0,512));self.assertFalse(c["has_local"])
        self.assertEqual(r["max_abs_error"],0.)
    def test_completed_reentry_no_producer_no_overwrite(self):
        cs=[sample(1),sample(7,1)];out=self.root/"cache";m=build_cache(out,cs)
        hashes={e["tensor_path"]:io.sha(e["tensor_path"]) for e in m["entries"]}
        with patch("prepare_full.torch.autograd.grad",side_effect=AssertionError("no JVP")):
            second=prep.prepare(out,rows_for(cs),{},"synthetic",lambda *a:self.fail("producer called"))
        self.assertEqual(m,second);io.verify(hashes)
    def test_interrupted_preparation_reuses_published_sample(self):
        cs=[sample(1),sample(3,1)];calls=[]
        def producer(e,md):
            calls.append(e["row_id"])
            if len(calls)==2:raise RuntimeError("interrupted")
            return cs[0],metadata(cs[0])
        out=self.root/"cache"
        with self.assertRaisesRegex(RuntimeError,"interrupted"):prep.prepare(out,rows_for(cs),{},"synthetic",producer)
        first=io.entry_at(out/"samples"/cs[0]["row_id"].replace(":","_"));sha=io.sha(first["tensor_path"])
        second_calls=[]
        def restart(e,md):
            second_calls.append(e["row_id"]);return cs[1],metadata(cs[1])
        prep.prepare(out,rows_for(cs),{},"synthetic",restart)
        self.assertEqual(second_calls,[cs[1]["row_id"]]);self.assertEqual(io.sha(first["tensor_path"]),sha)
    def test_entry_corruption_rejected(self):
        m=build_cache(self.root/"cache",[sample(1)])
        Path(m["entries"][0]["tensor_path"]).write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError,"SHA"):data.checked_cache(self.root/"cache")
    def test_formal_rejects_small_before_model(self):
        out=self.root/"cache";build_cache(out,[sample(1)])
        with patch("epoch_training.ImageOnlyGenerator",side_effect=AssertionError("model loaded")):
            with self.assertRaisesRegex(ValueError,"2048"):et.run_epochs(out,self.root/"run",et.configuration())
    def test_actual_producer_files_only(self):
        d=self.root/"sources";d.mkdir();(d/"a.py").write_text("a")
        locks=data.file_contract(["a.py"],d);(d/"unrelated_training.py").write_text("b")
        io.verify(locks)
        (d/"a.py").write_text("changed")
        with self.assertRaisesRegex(ValueError,"SHA"):io.verify(locks)
    def test_training_has_no_live_Gv1_source_dependency(self):
        out=self.root/"cache";m=build_cache(out,[sample(0)])
        p=io.load(out/"protocol.json");p["upstream_sha256"]={"/nonexistent/G-v1/source.pt":"historical"}
        io.save_json(out/"protocol.json",p)
        e=m["entries"][0];c,meta,_=io.read_entry(e);meta["protocol_hash"]=io.digest(p)
        # 仅合成测试重建独立缓存，生产已完成样本永不覆盖。
        other=self.root/"other";other.mkdir();io.save_json(other/"protocol.json",p)
        new=io.write_entry(other,c,meta);data.publish_manifest(other,[new],p)
        self.assertEqual(data.checked_cache(other)["counts"]["train"],1)
    def test_producer_source_changed_rejected(self):
        out=self.root/"cache";build_cache(out,[sample(0)])
        p=io.load(out/"protocol.json");p["producer_sha256"]={str(self.root/"source.py"):"bad"}
        (self.root/"source.py").write_text("modified")
        io.save_json(out/"protocol.json",p)
        # 固定manifest SHA链跟随新协议，仍必须拒绝实际producer SHA。
        m=io.load(out/"manifest.json");m["protocol_sha256"]=io.sha(out/"protocol.json")
        io.save_json(out/"manifest.json",m);io.save_json(out/"completed.json",{"method_audit_passed":True,"manifest_sha256":io.sha(out/"manifest.json")})
        with self.assertRaisesRegex(ValueError,"SHA"):data.checked_cache(out)
    def test_preparation_protocol_change_rejected(self):
        out=self.root/"cache";build_cache(out,[sample(0)])
        with self.assertRaisesRegex(ValueError,"protocol changed"):prep.prepare(out,rows_for([sample(1)]),{},"synthetic")
    def test_train_validation_PID_overlap_rejected(self):
        a=sample(1);b=sample(4,1,"validation");b["pid"]=a["pid"]
        with self.assertRaisesRegex(ValueError,"PID leakage"):build_cache(self.root/"cache",[a,b])
    def test_no_duplicate_identity(self):
        with self.assertRaises(ValueError):build_cache(self.root/"cache",[sample(0),sample(0)])
    def test_pair_checks_annotation_and_caption_index(self):
        c=sample(1);raw=[{}]*11;raw[10]={"split":"train","id":c["pid"],"file_path":c["image_rel_path"],"captions":[c["caption"]]}
        split={"records":[c],"train_indices":[0],"validation_indices":[]}
        c["caption_index"]=1
        with self.assertRaisesRegex(ValueError,"caption_index"):data.check_pair(c,c,metadata(c),raw,split)

    def test_pair_checks_raw_caption_PID_path_and_internal_role(self):
        c=sample(1);e=data.identity(c)
        raw=[{}]*11;raw[10]={"split":"train","id":c["pid"],"file_path":c["image_rel_path"],"captions":[c["caption"]]}
        split={"records":[c],"train_indices":[0],"validation_indices":[]}
        image=self.root/c["image_rel_path"];image.parent.mkdir(parents=True);image.write_bytes(b"pixels")
        md=metadata(c);md["source_image_sha256"]=io.sha(image)
        with patch.object(data,"IMAGES",self.root):
            self.assertEqual(data.check_pair(e,c,md,raw,split),image)
            for field,value in (("id",99),("captions",["wrong"]),("file_path","wrong.png")):
                wrong=copy.deepcopy(raw);wrong[10][field]=value
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,"pairing"):
                    data.check_pair(e,c,md,wrong,split)
            with self.assertRaisesRegex(ValueError,"role"):
                data.check_pair(e,c,md,raw,{**split,"train_indices":[]})
    def test_pair_image_SHA_and_symlink_escape_rejected(self):
        c=sample(1);raw=[{}]*11;raw[10]={"split":"train","id":c["pid"],"file_path":c["image_rel_path"],"captions":[c["caption"]]}
        split={"records":[c],"train_indices":[0],"validation_indices":[]}
        image=self.root/c["image_rel_path"];image.parent.mkdir(parents=True);image.write_bytes(b"pixels")
        md=metadata(c);md["source_image_sha256"]=io.sha(image);image.write_bytes(b"changed")
        with patch.object(data,"IMAGES",self.root):
            with self.assertRaisesRegex(ValueError,"SHA"):data.check_pair(c,c,md,raw,split)
            image.unlink();image.symlink_to("/etc/hostname")
            with self.assertRaisesRegex(ValueError,"escape"):data.check_pair(c,c,md,raw,split)
    def test_manifest_and_protocol_corruption_rejected(self):
        for name in ("manifest.json","protocol.json"):
            with self.subTest(file=name):
                out=self.root/name.replace(".","_");build_cache(out,[sample(0)])
                (out/name).write_text("{}")
                with self.assertRaisesRegex(ValueError,"SHA"):data.checked_cache(out)
    def test_dataset_protocol_SHA_is_live_verified(self):
        out=self.root/"cache";build_cache(out,[sample(0)])
        source=self.root/"annotation.json";source.write_text("new")
        p=io.load(out/"protocol.json");p["dataset_sha256"]={str(source):"wrong"}
        io.save_json(out/"protocol.json",p)
        m=io.load(out/"manifest.json");m["protocol_sha256"]=io.sha(out/"protocol.json")
        io.save_json(out/"manifest.json",m);io.save_json(out/"completed.json",{"method_audit_passed":True,"manifest_sha256":io.sha(out/"manifest.json")})
        with self.assertRaisesRegex(ValueError,"SHA"):data.checked_cache(out)
    def test_real_process_exit_preparation_then_resume_reuses_first(self):
        out=self.root/"cache"
        script=f"""import os
from test_phase24 import sample,rows_for,metadata
import prepare_full as p
cs=[sample(1),sample(3,1)]
def producer(e,md):
 if e['row_id']==cs[1]['row_id']:os._exit(43)
 return cs[0],metadata(cs[0])
p.prepare({str(out)!r},rows_for(cs),{{}},'synthetic',producer)
"""
        result=subprocess.run([sys.executable,"-c",script],cwd=io.ROOT,
            env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},capture_output=True)
        self.assertEqual(result.returncode,43)
        first=io.entry_at(out/"samples/cuhk_10_0");sha=io.sha(first["tensor_path"]);calls=[]
        cs=[sample(1),sample(3,1)]
        def producer(e,md):
            calls.append(e["row_id"]);return cs[1],metadata(cs[1])
        prep.prepare(out,rows_for(cs),{},"synthetic",producer)
        self.assertEqual(calls,[cs[1]["row_id"]]);self.assertEqual(io.sha(first["tensor_path"]),sha)

class ScheduleTests(Base):
    def loader(self,cs,randomized=False):
        values={c["row_id"]:c for c in cs}
        def load(e):
            c=values[e["row_id"]]
            image=torch.full((1,3,16,8),.5)
            if randomized:image=image+.01*(torch.rand_like(image)+random.random()+np.random.rand())
            return image,c
        return load
    def test_epoch_steps_K0_1_3_7_and_frozen_encoder(self):
        cs=[sample(k,i) for i,k in enumerate((0,1,3,7))]
        g,enc,_=self.components();before=copy.deepcopy(enc.state_dict())
        opt=CountingSGD(g.parameters(),lr=.01)
        p=et.train_epoch(g,enc,opt,cs,self.loader(cs),42,1,LossWeights())
        self.assertEqual((opt.steps,p["optimizer_steps"],p["samples"]),(15,15,4))
        self.assertEqual(p["updates_by_K"],{"0":1,"1":2,"3":4,"7":8})
        self.assertTrue(tree_equal(before,enc.state_dict()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in enc.parameters()))
    def test_shuffle_independent_of_global_rng_and_input_order(self):
        cs=[sample(0,i) for i in range(15)]
        a=et.epoch_order(cs,42,1);random.seed(999);torch.rand(8)
        self.assertEqual(a,et.epoch_order(cs[::-1],42,1))
        self.assertNotEqual(a,et.epoch_order(cs,42,2))
    def test_validation_cannot_enter_train_schedule(self):
        g,e,o=self.components();c=sample(4,0,"validation")
        with self.assertRaisesRegex(ValueError,"validation"):et.train_epoch(g,e,o,[c],lambda _:self.fail("loaded"),42,1,LossWeights())
        self.assertEqual(o.state,{})
    def test_wrong_loaded_caption_rejected_before_any_update(self):
        g,e,o=self.components();c=sample(1);bad=copy.deepcopy(c);bad["caption"]="wrong"
        with self.assertRaises(ValueError):et.train_epoch(g,e,o,[c],lambda _:(torch.full((1,3,16,8),.5),bad),42,1,LossWeights())
        self.assertEqual(o.state,{})
    def test_validation_no_grad_no_updates_one_forward(self):
        g,e,o=self.components();c=sample(4,0,"validation");g_before=copy.deepcopy(g.state_dict());calls=[]
        h=g.register_forward_hook(lambda *a:calls.append(torch.is_grad_enabled()))
        results=et.validate_epoch(g,e,[c],self.loader([c]));h.remove()
        self.assertEqual(calls,[False]);self.assertEqual(results[0]["optimizer_steps"],0)
        self.assertTrue(tree_equal(g_before,g.state_dict()));self.assertEqual(o.state,{})
    def test_each_sample_calls_existing_single_sample_once(self):
        cs=[sample(k,i) for i,k in enumerate((0,1,3))]
        g,e,o=self.components()
        with patch.object(et,"train_single_sample",wraps=et.train_single_sample) as call:
            et.train_epoch(g,e,o,cs,self.loader(cs),42,1,LossWeights())
        self.assertEqual(call.call_count,3)
    def test_logs_finite_budget_and_complete_rounds(self):
        cs=[sample(7)];g,e,o=self.components()
        p=et.train_epoch(g,e,o,cs,self.loader(cs),42,1,LossWeights(),self.root/"samples.jsonl")
        saved=json.loads((self.root/"samples.jsonl").read_text())
        self.assertEqual(saved,p["sample_logs"][0]);self.assertEqual(len(saved["rounds"]),8)
        for r in saved["rounds"]:
            self.assertLessEqual(r["cumulative_linf"],8/255)
            for k in ("global","local","retain","G_gradient_l2","budget_saturation_ratio","strict_projection_clipping_ratio"):
                self.assertTrue(np.isfinite(r[k]))
    def test_two_epoch_boundary_resume_exact_model_adam_order_and_RNG(self):
        cs=[sample(0),sample(1,1),sample(3,2),sample(4,3,"validation")]
        cache=self.root/"cache";build_cache(cache,cs)
        g0,e0,_=self.components();cfg=et.configuration("synthetic",2,42,"cpu")
        cfg["architecture"]={"width":2,"blocks":1,"epsilon":.004}
        def run(out,resume=False,limit=None):
            return et.run_epochs(cache,out,cfg,resume=resume,stop_after_epochs=limit,
                generator=copy.deepcopy(g0),encoder=copy.deepcopy(e0),loader=self.loader(cs,True))
        ga,oa,pa=run(self.root/"continuous")
        rng_a=ck.rng_state()
        _,_,p1=run(self.root/"interrupted",limit=1)
        self.assertEqual(p1["completed_epoch"],1)
        random.random();np.random.rand();torch.rand(100)
        gb,ob,pb=run(self.root/"interrupted",True)
        self.assertTrue(tree_equal(ga.state_dict(),gb.state_dict()))
        self.assertTrue(tree_equal(oa.state_dict(),ob.state_dict()))
        self.assertTrue(tree_equal(rng_a,ck.rng_state()))
        self.assertEqual(pa["epochs_this_run"][1]["sample_order"],pb["epochs_this_run"][0]["sample_order"])
        self.assertEqual(pa["epochs_this_run"][1]["sample_logs"],pb["epochs_this_run"][0]["sample_logs"])
        self.assertEqual(pb["completed_epoch"],2)
    def test_existing_run_requires_resume(self):
        cs=[sample(0)];cache=self.root/"cache";build_cache(cache,cs);g,e,_=self.components()
        cfg=et.configuration("synthetic",1,42,"cpu");cfg["architecture"]={"width":2,"blocks":1,"epsilon":.004}
        et.run_epochs(cache,self.root/"run",cfg,generator=g,encoder=e,loader=self.loader(cs))
        with self.assertRaisesRegex(ValueError,"explicit resume"):
            et.run_epochs(cache,self.root/"run",cfg,generator=g,encoder=e,loader=self.loader(cs))

class CheckpointTests(Base):
    def make(self,epoch=1):
        g,e,o=self.components();image=torch.full((1,3,16,8),.5)
        o.zero_grad();g(image).sum().backward();o.step()
        contract={"config":{"architecture":{"width":2,"blocks":1,"epsilon":.004}},"source":"fixed","cache":"fixed"}
        p=ck.save_epoch(self.root,g,o,epoch,contract,{"optimizer_steps":1})
        return g,o,contract,p
    def test_restore_model_optimizer_and_RNG(self):
        g,o,c,p=self.make();gs=copy.deepcopy(g.state_dict());os_=copy.deepcopy(o.state_dict())
        rng=ck.rng_state();random.random();np.random.rand();torch.rand(10)
        for x in g.parameters():x.data.zero_()
        result=ck.load_latest(self.root,c,g,o)
        self.assertEqual(result["next_epoch"],2)
        self.assertTrue(tree_equal(gs,g.state_dict()));self.assertTrue(tree_equal(os_,o.state_dict()))
        self.assertTrue(tree_equal(rng,ck.rng_state()))
    def test_immutable_no_overwrite(self):
        g,o,c,p=self.make()
        with self.assertRaisesRegex(ValueError,"immutable"):ck.save_epoch(self.root,g,o,1,c,{})
    def test_tensor_corruption_rejected_before_load(self):
        g,o,c,p=self.make();(p/"checkpoint.pt").write_bytes(b"broken")
        with patch("epoch_checkpoint.torch.load",side_effect=AssertionError("unsafe load")):
            with self.assertRaisesRegex(ValueError,"SHA"):ck.load_latest(self.root,c,g,o)
    def test_sidecar_contract_corruption_rejected(self):
        g,o,c,p=self.make();side=io.load(p/"checkpoint.json");side["contract_sha256"]="bad";io.save_json(p/"checkpoint.json",side)
        with self.assertRaisesRegex(ValueError,"contract mismatch"):ck.load_latest(self.root,c,g,o)
    def test_source_config_cache_mismatch_each_rejected(self):
        g,o,c,p=self.make()
        for name in c:
            bad=copy.deepcopy(c);bad[name]="changed"
            with self.subTest(field=name),self.assertRaisesRegex(ValueError,"contract mismatch"):ck.load_latest(self.root,bad,g,o)
    def test_pending_ignored_and_pointer_repaired(self):
        g,o,c,p=self.make();(self.root/"checkpoints/.pending-crash").mkdir()
        (self.root/"current.json").write_text("broken")
        self.assertEqual(ck.load_latest(self.root,c,g,o)["completed_epoch"],1)
        self.assertEqual(io.load(self.root/"current.json")["epoch"],1)
    def test_highest_corrupt_rejected_no_silent_fallback(self):
        g,o,c,p=self.make();p2=ck.save_epoch(self.root,g,o,2,c,{})
        (p2/"checkpoint.pt").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError,"SHA"):ck.load_latest(self.root,c,g,o)
    def test_failure_before_publish_no_version(self):
        g,e,o=self.components()
        with patch("epoch_checkpoint.os.rename",side_effect=OSError("publication failure")):
            with self.assertRaises(OSError):ck.save_epoch(self.root,g,o,1,{}, {})
        self.assertFalse((self.root/"checkpoints/epoch_0001").exists())
        with self.assertRaisesRegex(ValueError,"no published"):ck.load_latest(self.root,{},g,o)
    def test_failure_after_publish_recovers_by_scan(self):
        g,e,o=self.components();original=io.save_json
        def fail_pointer(path,value):
            if Path(path).name=="current.json":raise OSError("pointer failure")
            return original(path,value)
        with patch.object(io,"save_json",side_effect=fail_pointer):
            with self.assertRaises(OSError):ck.save_epoch(self.root,g,o,1,{}, {})
        self.assertTrue((self.root/"checkpoints/epoch_0001").exists())
        self.assertEqual(ck.load_latest(self.root,{},g,o)["next_epoch"],2)
    def test_real_process_exit_before_publish_then_recovery(self):
        script=f"""import os,torch
from unittest.mock import patch
import epoch_checkpoint as ck
g=torch.nn.Linear(2,2);o=torch.optim.Adam(g.parameters())
with patch('epoch_checkpoint.os.rename',side_effect=lambda *a:os._exit(39)):
 ck.save_epoch({str(self.root)!r},g,o,1,{{}},{{}})
"""
        p=subprocess.run([sys.executable,"-c",script],cwd=io.ROOT,env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"})
        self.assertEqual(p.returncode,39)
        g=nn.Linear(2,2);o=torch.optim.Adam(g.parameters())
        with self.assertRaisesRegex(ValueError,"no published"):ck.load_latest(self.root,{},g,o)
        ck.save_epoch(self.root,g,o,1,{},{});self.assertEqual(ck.load_latest(self.root,{},g,o)["next_epoch"],2)
    def test_concurrent_writer_lock_rejected(self):
        with io.output_lock(self.root):
            with self.assertRaisesRegex(ValueError,"another"):
                with io.output_lock(self.root):pass
    def test_inference_loads_only_G_and_verifies_SHA(self):
        g,o,c,p=self.make();loaded=infer.load_generator(p/"checkpoint.pt")
        self.assertTrue(tree_equal(g.state_dict(),loaded.state_dict()))
        (p/"checkpoint.pt").write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError,"SHA"):infer.load_generator(p/"checkpoint.pt")

class InferenceTests(Base):
    def test_exactly_one_image_only_forward_budget(self):
        g,e,o=self.components();calls=[];h=g.register_forward_pre_hook(lambda m,a:calls.append(len(a)))
        image=torch.rand(1,3,17,9);r=infer.generate_once(g,image);h.remove()
        self.assertEqual(calls,[1]);self.assertFalse(r["adversarial"].requires_grad)
        self.assertLessEqual(float((r["adversarial"].double()-image.double()).abs().max()),8/255)
        self.assertTrue(((r["adversarial"]>=0)&(r["adversarial"]<=1)).all())
    def test_inference_import_does_not_load_supervision_or_IRRA(self):
        code="import sys,inference_once; assert all(x not in sys.modules for x in ('cache_io','training_data','irra_image_adapter','integration','deps','serial_training'))"
        p=subprocess.run([sys.executable,"-c",code],cwd=io.ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_training_modules_do_not_import_offline_frontend_or_Gv1(self):
        code="import sys,epoch_training; assert all(x not in sys.modules for x in ('prepare_full','prep_text_frontend','integration','deps','generator_model'))"
        p=subprocess.run([sys.executable,"-c",code],cwd=io.ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
    def test_CPU_suite_has_no_CUDA_initialization(self):self.assertFalse(torch.cuda.is_initialized())

if __name__=="__main__":unittest.main()
