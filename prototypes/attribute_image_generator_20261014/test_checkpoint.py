"""仅CPU：保存故障注入、真实子进程退出、Adam恢复和锁时序回归。"""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import torch
from torch import nn
import common as c
import checkpoint_io as ck
import train
from generator_model import ConditionalGenerator
torch.set_num_threads(2)

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=c.ROOT/"outputs/checkpoint_repair")
        self.root=Path(self.tmp.name);self.path=self.root/"last.pt"
        torch.manual_seed(17)
        self.g=ConditionalGenerator(width=4,blocks=1,condition_hidden=8)
        self.opt=torch.optim.Adam(self.g.parameters(),lr=.001,betas=(.5,.999))
        self.config={"seed":42}
    def tearDown(self):self.tmp.cleanup()
    def save(self,epoch):train.checkpoint(self.path,self.g,self.opt,epoch,self.config,"cache","training")
    def read(self):return train.load_checkpoint(self.path,"cpu")
    def inject(self,point,epoch=2):
        def fail(stage):
            if stage==point:raise OSError("injected "+point)
        with patch.object(ck,"_fault",side_effect=fail):
            with self.assertRaises(OSError):self.save(epoch)
    def test_payload_format_and_public_paths(self):
        self.save(1);cp=self.read()
        self.assertEqual(cp["schema"],"conditional-G-image-1")
        self.assertEqual(cp["epoch"],1);self.assertEqual(cp["config"],self.config)
        self.assertTrue(self.path.is_symlink());self.assertTrue(Path(str(self.path)+".json").is_symlink())
        self.assertEqual(c.sha(self.path),c.load(str(self.path)+".json")["sha256"])
        self.assertEqual(torch.load(self.path,weights_only=True)["epoch"],1)
    def test_old_immutable_version_preserved(self):
        self.save(1);_,_,old=ck.resolve_checkpoint(self.path);h=c.sha(old)
        self.save(2);self.assertEqual(c.sha(old),h);self.assertEqual(self.read()["epoch"],2)
    def test_abort_after_tensor_keeps_previous(self):
        self.save(1);self.inject("after_tensor");self.assertEqual(self.read()["epoch"],1)
    def test_abort_after_sidecar_keeps_previous(self):
        self.save(1);self.inject("after_sidecar");self.assertEqual(self.read()["epoch"],1)
    def test_abort_before_directory_commit_keeps_previous(self):
        self.save(1);self.inject("before_commit");self.assertEqual(self.read()["epoch"],1)
    def test_abort_after_commit_recovers_new_epoch(self):
        self.save(1);self.inject("after_commit")
        self.assertEqual(torch.load(self.path,weights_only=True)["epoch"],1)
        self.assertEqual(self.read()["epoch"],2)
        self.assertEqual(torch.load(self.path,weights_only=True)["epoch"],2)
    def test_abort_before_pointer_recovers_new_epoch(self):
        self.save(1);self.inject("before_pointer");self.assertEqual(self.read()["epoch"],2)
    def test_abort_after_pointer_recovers_new_epoch(self):
        self.save(1);self.inject("after_pointer");self.assertEqual(self.read()["epoch"],2)
    def test_tensor_write_failure_preserves_old_bytes(self):
        self.save(1);h=c.sha(self.path)
        def disk_full(value,file):
            file.write(b"partial");raise OSError("no space")
        with patch.object(ck.torch,"save",side_effect=disk_full):
            with self.assertRaises(OSError):self.save(2)
        self.assertEqual(c.sha(self.path),h);self.assertEqual(self.read()["epoch"],1)
    def test_sidecar_write_failure_preserves_old_bytes(self):
        self.save(1);h=c.sha(self.path)
        with patch.object(ck.json,"dump",side_effect=OSError("no space")):
            with self.assertRaises(OSError):self.save(2)
        self.assertEqual(c.sha(self.path),h);self.assertEqual(self.read()["epoch"],1)
    def test_temporary_pair_not_accepted_on_first_epoch(self):
        self.inject("after_sidecar",1)
        with self.assertRaises(FileNotFoundError):self.read()
    def test_first_epoch_committed_without_alias_can_resume(self):
        self.inject("after_commit",1)
        self.assertFalse(self.path.exists());self.assertEqual(self.read()["epoch"],1)
        self.assertTrue(self.path.exists())
    def test_first_alias_partial_install_can_resume(self):
        self.inject("after_alias:last.pt",1)
        self.assertEqual(self.read()["epoch"],1)
        self.assertEqual(c.sha(self.path),c.load(str(self.path)+".json")["sha256"])
    def test_legacy_checkpoint_read_and_migration(self):
        self.save(1);payload=self.read()
        old=self.root/"legacy_entry.pt";c.tensor_save(old,payload)
        c.save(str(old)+".json",{"sha256":c.sha(old),"epoch":1,"scope":"training","cache_manifest_sha256":"cache"})
        h=c.sha(old)
        self.assertEqual(train.load_checkpoint(old,"cpu")["epoch"],1)
        train.checkpoint(old,self.g,self.opt,2,self.config,"cache","training")
        self.assertEqual(c.sha(ck._store(old)/"legacy/checkpoint.pt"),h)
        self.assertEqual(train.load_checkpoint(old,"cpu")["epoch"],2)
    def test_corrupt_latest_rolls_back_and_can_continue(self):
        self.save(1);self.save(2);_,_,new=ck.resolve_checkpoint(self.path)
        new.write_bytes(b"corrupt")
        with warnings.catch_warnings(record=True) as seen:
            self.assertEqual(self.read()["epoch"],1);self.assertTrue(seen)
            self.save(2)
            self.assertEqual(self.read()["epoch"],2)
    def test_source_mismatch_not_silently_rolled_back(self):
        self.save(1);cp=self.read();cp["epoch"]=2
        cp["inference_source_sha256"]={str(c.ROOT/"generator_model.py"):"invalid"}
        ck.save_checkpoint(self.path,cp)
        with self.assertRaisesRegex(ValueError,"SHA mismatch"):self.read()
    def test_resume_adam_model_and_next_update_exact(self):
        torch.manual_seed(23)
        images=torch.rand(2,3,32,16)
        z=torch.nn.functional.normalize(torch.randn(2,512),dim=1)
        a=torch.randn(2,512)
        def update(g,opt):
            opt.zero_grad(set_to_none=True)
            loss=g(images,z,a).square().mean()+g(images,z,a).mean()
            loss.backward();opt.step()
        update(self.g,self.opt);self.save(1)
        update(self.g,self.opt);wanted=copy.deepcopy(self.g.state_dict())
        opt_wanted=copy.deepcopy(self.opt.state_dict())
        cp=self.read();g2=ConditionalGenerator(**cp["architecture"])
        opt2=torch.optim.Adam(g2.parameters(),lr=.001,betas=(.5,.999))
        g2.load_state_dict(cp["model"]);opt2.load_state_dict(cp["optimizer"])
        self.assertEqual(cp["epoch"]+1,2)
        update(g2,opt2)
        for k,v in wanted.items():self.assertTrue(torch.equal(v,g2.state_dict()[k]),k)
        for i,state in opt_wanted["state"].items():
            for k,v in state.items():self.assertTrue(torch.equal(v,opt2.state_dict()["state"][i][k]))
    def test_epoch_shuffle_resume_exact(self):
        def order(epoch):return torch.randperm(13,generator=torch.Generator().manual_seed(42+epoch))
        self.save(4);self.assertTrue(torch.equal(order(5),order(self.read()["epoch"]+1)))
    def test_independent_process_termination(self):
        self.save(1)
        for stage,wanted in (("after_tensor",1),("after_sidecar",1),("after_commit",2),("before_pointer",3)):
            code='''import os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import checkpoint_io as ck
path=Path(sys.argv[2])
payload,_,_=ck.resolve_checkpoint(path)
payload["epoch"]=int(sys.argv[4])
def stop(stage):
 if stage==sys.argv[3]:os._exit(73)
ck._fault=stop
ck.save_checkpoint(path,payload)
'''
            epoch=2 if wanted<=2 else 3
            run=subprocess.run([sys.executable,"-B","-c",code,str(c.ROOT),str(self.path),stage,str(epoch)],capture_output=True,text=True)
            self.assertEqual(run.returncode,73,run.stderr)
            self.assertEqual(self.read()["epoch"],wanted)
    def test_lock_blocks_concurrent_process_and_releases(self):
        out=self.root/"run"
        code='''import sys
sys.path.insert(0,sys.argv[1])
import checkpoint_io as ck
try:
 with ck.training_lock(sys.argv[2]):pass
except BlockingIOError:sys.exit(79)
'''
        with ck.training_lock(out):
            run=subprocess.run([sys.executable,"-B","-c",code,str(c.ROOT),str(out)],capture_output=True,text=True)
            self.assertEqual(run.returncode,79,run.stderr)
            self.assertFalse((out/"protocol.json").exists())
        with ck.training_lock(out):pass
    def test_main_first_output_write_is_inside_lock(self):
        tree=ast.parse((c.ROOT/"train.py").read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        lock=next(n for n in main.body if isinstance(n,ast.With))
        self.assertEqual(ast.unparse(lock.items[0].context_expr),"ck.training_lock(out)")
        before=main.body[:main.body.index(lock)]
        self.assertFalse(any(isinstance(n,ast.Call) and ast.unparse(n.func) in ("c.save","c.tensor_save") for stmt in before for n in ast.walk(stmt)))
        self.assertTrue(any(isinstance(n,ast.Call) and ast.unparse(n.func)=="c.save" for n in ast.walk(lock)))
    def test_training_core_AST_unchanged(self):
        before=json.loads((c.ROOT/"outputs/checkpoint_repair/before_sources.json").read_text())["train.py"]
        def funcs(source):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        old,new=funcs(before),funcs((c.ROOT/"train.py").read_text())
        for name in ("ConditionDataset","batch_to","weights","step","validate","parser"):
            self.assertEqual(old[name],new[name],name)
    def test_model_losses_conditions_unchanged(self):
        old=json.loads((c.ROOT/"outputs/checkpoint_repair/before_sources.json").read_text())
        for name in ("generator_model.py","conditioning.py","common.py","generate.py"):
            self.assertEqual(old[name],(c.ROOT/name).read_text(),name)
    def test_training_epoch_loop_AST_unchanged(self):
        before=json.loads((c.ROOT/"outputs/checkpoint_repair/before_sources.json").read_text())["train.py"]
        def loop(source):
            main=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="main")
            return next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=="epoch")
        self.assertEqual(ast.dump(loop(before)),ast.dump(loop((c.ROOT/"train.py").read_text())))
    def test_lock_exception_releases(self):
        out=self.root/"run"
        with self.assertRaises(RuntimeError):
            with ck.training_lock(out):raise RuntimeError("interrupted")
        with ck.training_lock(out):pass
    def test_legacy_alias_install_interruption_preserves_valid_pair(self):
        self.save(1);payload=self.read()
        old=self.root/"legacy_interrupted.pt";c.tensor_save(old,payload)
        c.save(str(old)+".json",{"sha256":c.sha(old),"epoch":1,"scope":"training","cache_manifest_sha256":"cache"})
        def fail(stage):
            if stage=="after_alias:legacy_interrupted.pt":raise OSError("interrupted migration")
        with patch.object(ck,"_fault",side_effect=fail):
            with self.assertRaises(OSError):train.checkpoint(old,self.g,self.opt,2,self.config,"cache","training")
        self.assertEqual(c.sha(old),c.load(str(old)+".json")["sha256"])
        self.assertEqual(train.load_checkpoint(old,"cpu")["epoch"],1)
    def test_old_smoke_gate_does_not_accept_arbitrary_change(self):
        old={"method_audit_passed":True,"source_sha256":{str(c.ROOT/"train.py"):"bad"}}
        with self.assertRaises((ValueError,FileNotFoundError,KeyError)):ck.verify_training_smoke(old)

if __name__=="__main__":unittest.main()
