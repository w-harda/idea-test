"""CPU验证：真实缓存约束、故障拒绝和事务边界；不加载victim。"""
import copy, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import torch
import cache_io as io
from condition_schema import build, validate, error, IDENTITY
torch.set_num_threads(2)

def fixture(k=3):
    identity = {"row_id":"cuhk:10:0","image_rel_path":"train_query/x.jpg","caption_index":0,
                "split":"train","role":"train","pid":1,"caption":"A black shirt."}
    scores = [float(k-i) for i in range(k)]
    weights = [s/sum(scores) for s in scores]
    vectors = torch.eye(512)[:k] * torch.arange(1,k+1).float()[:,None]
    return build(identity,[f"id{i}" for i in range(k)],scores,weights,vectors,torch.eye(512)[10])

def metadata(v):
    return {"protocol_hash":"protocol","source_selection":{"natural_k":v["K"],
        "selected_attributes":[{"instance_id":i,"S_raw":s} for i,s in zip(v["selected_instance_ids"],v["S_raw"].tolist())]}}

class SchemaTests(unittest.TestCase):
    def test_K0_complete_contract(self):
        x=fixture(0); r=validate(x)
        self.assertEqual(x["selected_instance_ids"],[]); self.assertEqual(x["S_raw"].tolist(),[])
        self.assertEqual(x["score_weights"].tolist(),[]); self.assertEqual(tuple(x["attribute_jvps"].shape),(0,512))
        self.assertFalse(x["has_local"]); self.assertFalse(x["v_local"].any()); self.assertTrue(r["exact"])
    def test_K1_scale_preserved(self):
        x=fixture(1); self.assertTrue(torch.equal(x["v_local"],x["attribute_jvps"][0]))
    def test_variable_K7(self): self.assertEqual(fixture(7)["attribute_jvps"].shape,(7,512))
    def test_order_and_aggregate(self):
        x=fixture(); expected=torch.zeros(512); expected[:3]=torch.tensor([.5,2/3,.5])
        self.assertTrue(torch.allclose(x["v_local"],expected)); self.assertEqual(x["S_raw"].tolist(),[3.,2.,1.])
    def test_no_JVP_normalization(self):
        x=fixture(); self.assertEqual(float(x["attribute_jvps"][2].norm()),3.)
    def test_detached_storage(self):
        x=fixture(); self.assertTrue(all(not v.requires_grad and v.device.type=="cpu" for v in x.values() if torch.is_tensor(v)))
    def test_missing_identity(self):
        for key in IDENTITY:
            x=fixture(); del x[key]
            with self.subTest(key=key), self.assertRaises(ValueError): validate(x)
    def test_caption_index_mismatch(self):
        x=fixture(); x["caption_index"]=1
        with self.assertRaises(ValueError): validate(x)
    def test_official_split_rejects_test(self):
        x=fixture(); x["split"]="test"
        with self.assertRaises(ValueError): validate(x)
    def test_internal_role_separate_from_split(self):
        x=fixture(); x["role"]="validation"; validate(x); self.assertEqual(x["split"],"train")
    def test_image_path_rejection(self):
        for p in ("../x","/home/x","x/../y","C:/x","x\\y","./x"):
            x=fixture(); x["image_rel_path"]=p
            with self.subTest(path=p), self.assertRaises(ValueError): validate(x)
    def test_duplicate_IDs(self):
        x=fixture(); x["selected_instance_ids"][1]=x["selected_instance_ids"][0]
        with self.assertRaises(ValueError): validate(x)
    def test_non_descending_scores(self):
        x=fixture(); x["S_raw"]=x["S_raw"].flip(0)
        with self.assertRaises(ValueError): validate(x)
    def test_invalid_scores(self):
        for s in (0.,-1.,float("nan"),float("inf")):
            x=fixture(); x["S_raw"][0]=s
            with self.subTest(s=s), self.assertRaises(ValueError): validate(x)
    def test_power_weights_rejected(self):
        x=fixture(); x["score_weights"]=x["S_raw"].pow(3); x["score_weights"]/=x["score_weights"].sum()
        with self.assertRaises(ValueError): validate(x)
    def test_wrong_weight_length(self):
        x=fixture(); x["score_weights"]=x["score_weights"][:1]
        with self.assertRaises(ValueError): validate(x)
    def test_bad_JVP_shape(self):
        x=fixture(); x["attribute_jvps"]=x["attribute_jvps"][:,None,:]
        with self.assertRaises(ValueError): validate(x)
    def test_zero_JVP_rejected(self):
        x=fixture(); x["attribute_jvps"][1]=0
        with self.assertRaises(ValueError): validate(x)
    def test_nonfinite_JVP_rejected(self):
        x=fixture(); x["attribute_jvps"][0,0]=float("nan")
        with self.assertRaises(ValueError): validate(x)
    def test_wrong_tensor_precision(self):
        x=fixture(); x["S_raw"]=x["S_raw"].float()
        with self.assertRaises(ValueError): validate(x)
    def test_grad_target_rejected(self):
        x=fixture(); x["attribute_jvps"].requires_grad_(True)
        with self.assertRaises(ValueError): validate(x)
    def test_nonunit_caption(self):
        x=fixture(); x["z_t"]*=2
        with self.assertRaises(ValueError): validate(x)
    def test_K_mask_rejected(self):
        x=fixture(); x["has_local"]=False
        with self.assertRaises(ValueError): validate(x)
    def test_K0_nonzero_aggregate_rejected(self):
        x=fixture(0); x["v_local"][0]=1e-8
        with self.assertRaises(ValueError): validate(x)
    def test_K0_wrong_empty_JVP_shape(self):
        x=fixture(0); x["attribute_jvps"]=torch.empty(0)
        with self.assertRaises(ValueError): validate(x)
    def test_aggregate_corruption_rejected(self):
        x=fixture(); x["v_local"][0]+=.01
        with self.assertRaises(ValueError): validate(x)
    def test_error_reports_nonzero_difference(self):
        a=torch.zeros(512); b=a.clone(); b[0]=.001
        r=error(a,b); self.assertFalse(r["exact"]); self.assertGreater(r["max_abs_error"],0)
    def test_identity_crosscheck(self):
        x=fixture(); identity={k:x[k] for k in IDENTITY}; identity["pid"]=2
        with self.assertRaises(ValueError): validate(x,identity)

class CacheTests(unittest.TestCase):
    def setUp(self):
        io.OUTPUTS.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=io.OUTPUTS); self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name); self.v=fixture(); self.md=metadata(self.v)
    def write(self): return io.write_entry(self.out,self.v,self.md)
    def rewrite_meta(self,e,fn):
        p=Path(e["metadata_path"]); m=io.load(p); fn(m); io.save_json(p,m)
        e["metadata_sha256"]=io.sha(p)
    def test_roundtrip(self):
        e=self.write(); v,md,r=io.read_entry(e,"protocol")
        self.assertTrue(all(torch.equal(v[k],self.v[k]) for k in ("S_raw","score_weights","attribute_jvps","z_t","v_local")))
        self.assertTrue(r["exact"])
    def test_K0_roundtrip(self):
        self.v=fixture(0); self.md=metadata(self.v); e=self.write()
        self.assertFalse(io.read_entry(e)[0]["has_local"])
    def test_no_overwrite(self):
        self.write()
        with self.assertRaises(ValueError): self.write()
    def test_tensor_SHA_damage(self):
        e=self.write()
        with Path(e["tensor_path"]).open("ab") as f: f.write(b"damage")
        with self.assertRaises(ValueError): io.read_entry(e)
    def test_metadata_SHA_damage(self):
        e=self.write()
        with Path(e["metadata_path"]).open("a") as f: f.write(" ")
        with self.assertRaises(ValueError): io.read_entry(e)
    def test_protocol_mismatch(self):
        e=self.write()
        with self.assertRaises(ValueError): io.read_entry(e,"other")
    def test_metadata_identity_mismatch(self):
        e=self.write(); self.rewrite_meta(e,lambda m:m.update(pid=999))
        with self.assertRaises(ValueError): io.read_entry(e)
    def test_selection_order_mismatch(self):
        e=self.write(); self.rewrite_meta(e,lambda m:m["source_selection"]["selected_attributes"].reverse())
        with self.assertRaises(ValueError): io.read_entry(e)
    def test_resume_reads_published_sample(self):
        e=self.write(); resumed=io.entry_at(Path(e["tensor_path"]).parent)
        self.assertEqual(e,resumed); io.read_entry(resumed,"protocol")
    def test_failed_publication_no_partial_sample(self):
        with patch("cache_io.os.rename",side_effect=OSError("injected")):
            with self.assertRaises(OSError): self.write()
        self.assertFalse((self.out/"samples"/"cuhk_10_0").exists())
        self.assertEqual(list((self.out/"samples").iterdir()),[])
        self.write()
    def test_process_exit_pending_not_published(self):
        script = """import os
from unittest.mock import patch
from pathlib import Path
from test_conditions import fixture, metadata
import cache_io as io
v=fixture()
with patch("cache_io.os.rename", side_effect=lambda *a:os._exit(7)):
    io.write_entry(Path(__import__("sys").argv[1]),v,metadata(v))
"""
        env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":""}
        r=subprocess.run([sys.executable,"-c",script,str(self.out)],cwd=io.ROOT,env=env,capture_output=True)
        self.assertEqual(r.returncode,7,r.stderr.decode())
        self.assertFalse((self.out/"samples"/"cuhk_10_0").exists())
        self.assertEqual(len(list((self.out/"samples").glob(".pending-*"))),1)
        self.write()
    def test_write_escape_rejected(self):
        with self.assertRaises(ValueError): io.safe(io.ROOT.parent/"attribute_image_generator_20261014/outputs/x")
    def test_symlink_escape_rejected(self):
        link=self.out/"escape"; link.symlink_to(io.ROOT.parent/"attribute_image_generator_20261014",target_is_directory=True)
        with self.assertRaises(ValueError): io.safe(link/"x")
    def test_concurrent_lock_rejected(self):
        with io.output_lock(self.out):
            with self.assertRaises(ValueError):
                with io.output_lock(self.out): pass

class CliTests(unittest.TestCase):
    def test_default_is_CPU_preflight(self):
        import prepare
        with patch.object(prepare,"preflight") as cpu, patch.object(prepare,"run_smoke",side_effect=AssertionError):
            prepare.main([]); cpu.assert_called_once()
    def test_no_full_flag(self):
        import prepare
        with self.assertRaises(SystemExit): prepare.main(["--allow-preparation"])
    def test_no_training_modules(self):
        for p in ("train.py","generator_model.py","generate.py"):
            self.assertFalse((io.ROOT/p).exists())

if __name__ == "__main__": unittest.main(verbosity=2)
