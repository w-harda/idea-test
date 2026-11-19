"""原目标对照、合法候选覆盖、历史精确复用和预取身份测试。"""
import sys,types,tempfile,unittest,copy,random
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116")
sys.path.insert(0,str(SOURCE/"src"))
import tbps
tbps.__path__.append(str(ROOT/"src/tbps"))
import torch
from tbps.performance.fast import candidate_rewards,fast_predict
from tbps.performance.images import ImagePipeline
from tbps.attraction.objectives import ProfileKL,Settings
from tbps.attribute_response.teacher import ResponseTeacher
from tbps.attribute_response.settings import Settings as TS
from tbps.attribute_response.objectives import conditional_advantage
from tbps.augmented_attraction.views import recipe,apply_views,ViewSettings
from tbps.artifacts import sha_file
torch.set_num_threads(2)
class MathTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(42)
  self.p=ProfileKL(torch.randn(37,16),Settings())
  self.f=torch.randn(19,16);self.c=torch.randn(1,16);self.part=torch.randn(3,16)
 def reference(self):
  return torch.tensor([float(self.p.terms(f[None].expand(3,-1),self.c.expand(3,-1),self.part)["reward"].mean()) for f in self.f])
 def test_all_candidates_match_original(self):
  y=torch.tensor(candidate_rewards(self.p,self.f,self.c,self.part,device="cpu",chunk=4))
  torch.testing.assert_close(y,self.reference(),atol=3e-6,rtol=3e-6)
 def test_chunking_preserves_order(self):
  a=candidate_rewards(self.p,self.f,self.c,self.part,device="cpu",chunk=1)
  b=candidate_rewards(self.p,self.f,self.c,self.part,device="cpu",chunk=8)
  torch.testing.assert_close(torch.tensor(a),torch.tensor(b),atol=3e-6,rtol=3e-6)
 def test_conditional_advantage_and_gradient(self):
  ref=self.reference();v=candidate_rewards(self.p,self.f,self.c,self.part,device="cpu")
  probs=torch.rand(len(ref));probs/=probs.sum()
  a,b=conditional_advantage(ref,probs,5);aa,bb=conditional_advantage(v,probs,5)
  torch.testing.assert_close(a,aa,atol=3e-6,rtol=3e-6)
  x=torch.randn(19,requires_grad=True);loss=-a*x.log_softmax(0)[5];g=torch.autograd.grad(loss,x)[0]
  loss=-aa*x.log_softmax(0)[5];gg=torch.autograd.grad(loss,x)[0]
  torch.testing.assert_close(g,gg,atol=3e-6,rtol=3e-6)
 def test_empty_candidates(self):
  self.assertEqual(candidate_rewards(self.p,self.f[:0],self.c,self.part,device="cpu"),[])
 def test_nonfinite_rejected(self):
  self.f[0,0]=float("nan")
  with self.assertRaises(ValueError):candidate_rewards(self.p,self.f,self.c,self.part,device="cpu")
 def test_zero_feature_rejected(self):
  self.f[0]=0
  with self.assertRaises(ValueError):candidate_rewards(self.p,self.f,self.c,self.part,device="cpu")
 def test_bad_shape_rejected(self):
  with self.assertRaises(ValueError):candidate_rewards(self.p,self.f,self.c,self.part[:2],device="cpu")
 def test_dtype_rejected(self):
  with self.assertRaises(ValueError):candidate_rewards(self.p,self.f.double(),self.c,self.part,device="cpu")
 def test_anchor_cache_invalidation(self):
  a=candidate_rewards(self.p,self.f,self.c,self.part,device="cpu")
  self.p.anchors.mul_(-1)
  b=candidate_rewards(self.p,self.f,self.c,self.part,device="cpu")
  self.assertNotEqual(a,b)
 def test_no_input_mutation_or_grad(self):
  original=self.f.clone();self.f.requires_grad_(True)
  candidate_rewards(self.p,self.f,self.c,self.part,device="cpu")
  self.assertTrue(torch.equal(original,self.f));self.assertIsNone(self.f.grad)

class TeacherTests(unittest.TestCase):
 def setUp(self):
  self.t=ResponseTeacher(TS())
  for head in (self.t.model.g_to_h,self.t.model.h_to_g):
   torch.nn.init.normal_(head.weight,std=.1)
  events=[{"state":[.1]*12,"response":[.2,.3],"version":v} for v in range(1,5)]
  from tbps.attribute_response.memory import semantic_key
  self.key=semantic_key({"semantic_axis":"color","canonical_value":"red","region":"upper"})
  self.batch={"keys":[self.key]*12,"states":[[.4]*12]*12,"actions":[[i/20]*6 for i in range(12)],
   "histories":[copy.deepcopy(events) for _ in range(12)],"versions":[8]*12}
 def test_exact_output_and_rng(self):
  before=torch.get_rng_state().clone()
  a=self.t.predict(self.batch);b=fast_predict(self.t,self.batch)
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(before,torch.get_rng_state()))
 def test_history_cell_invocations_deduplicated(self):
  counts=[]
  h=self.t.model.history.register_forward_hook(lambda *a:counts.append(1))
  self.t.predict(self.batch);old=len(counts);counts.clear()
  fast_predict(self.t,self.batch);new=len(counts);h.remove()
  self.assertEqual(old,48);self.assertEqual(new,4)
 def test_mixed_histories_versions_and_keys(self):
  self.batch["keys"][2]=self.key.replace("red","blue");self.batch["versions"][3]=3;self.batch["histories"][4]=[]
  self.assertTrue(torch.equal(self.t.predict(self.batch),fast_predict(self.t,self.batch)))
 def test_no_model_optimizer_memory_mutation(self):
  from tbps.alternating.state import tree_sha
  before=tree_sha(self.t.state_dict());fast_predict(self.t,self.batch)
  self.assertEqual(before,tree_sha(self.t.state_dict()))
  self.assertTrue(all(p.grad is None for p in self.t.model.parameters()))
 def test_empty(self):
  self.assertEqual(tuple(fast_predict(self.t,{"keys":[]}).shape),(0,2))

class ImageTests(unittest.TestCase):
 def setUp(self):
  from PIL import Image
  import numpy as np
  base=Path("/home/lzf/ldx/tmp/gh_performance_tests");base.mkdir(exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
  root=Path(self.temp.name);entries={}
  for i in range(5):
   p=root/f"{i}.png";Image.fromarray(np.arange(16*8*3,dtype=np.uint8).reshape(16,8,3)+i).save(p)
   entries[str(i)]={"identity":{"image_rel_path":p.name,"image_sha256":sha_file(p)}}
  def pixels(im):return torch.from_numpy(np.array(im.convert("RGB"),copy=True)).permute(2,0,1).float()/255.
  self.b=types.SimpleNamespace(p={"train_ids":list(entries),"image_root":str(root)},entries=entries,
   view_settings=ViewSettings(),native=types.SimpleNamespace(get=lambda _:types.SimpleNamespace(pixels_from_pil=pixels)))
  self.pipe=ImagePipeline(self.b,workers=2,window=3,pin=False);self.addCleanup(self.pipe.close)
 def test_views_exact_across_round_and_order(self):
  before=torch.get_rng_state().clone();py=random.getstate()
  for version in (31,32):
   for row in ("0","3","1","4"):
    x=self.pipe.cpu(row)
    expected=apply_views(x,recipe(self.b.view_settings,row,version,*x.shape[-2:]))
    self.assertTrue(torch.equal(expected,self.pipe.views(row,version)))
  self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(py,random.getstate())
 def test_cached_original_cannot_be_modified_by_consumer(self):
  a=self.pipe.cpu("0");expected=a.clone();a.fill_(0)
  self.assertTrue(torch.equal(expected,self.pipe.cpu("0")))
 def test_source_change_rejected_even_after_prefetch(self):
  self.pipe.views("0",31)
  p=Path(self.b.p["image_root"])/"1.png";p.write_bytes(b"changed source")
  with self.assertRaises(ValueError):self.pipe.views("1",31)


class DeploymentTests(unittest.TestCase):
 def setUp(self):
  import importlib.util,json
  spec=importlib.util.spec_from_file_location("performance_supervisor_test",ROOT/"operations/gh_performance_supervisor.py")
  self.s=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.s)
  base=Path("/home/lzf/ldx/tmp/gh_performance_tests");base.mkdir(exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
  root=Path(self.temp.name);self.ctl=root/"control";self.old=root/"old";self.train=root/"training"
  self.base=root/"tbps-framework-fixture"
  for p in (self.ctl,self.old,self.train,self.base/"train/boundaries"):p.mkdir(parents=True)
  self.write(self.ctl/"benchmark.json",{"status":"GH_PERFORMANCE_BENCHMARK_PASS",
   "manifest_sha256":sha_file(self.s.MANIFEST),"candidate_KL":{"speedup":2},"teacher_history":{"speedup":2}})
  self.write(self.ctl/"cpu_tests.json",{"status":"PASS","manifest_sha256":sha_file(self.s.MANIFEST)})
  self.write(self.old/"started.json",{"pid":100})
  self.write(self.old/"child.json",{"pid":200,"label":"train_to_40"})
  self.write(self.train/"status.json",{"pid":200,"round":32,"cursor":{"round":31}})
  self.parent={"pid":100,"state":"S","start_ticks":123,"cmd":"python gh_attribute_response_recovery.py run"}
  self.child={"pid":200,"state":"R","start_ticks":456,"cmd":"python gh_attribute_response.py --action train"}
  from tbps.artifacts import ArtifactStore,json_bytes
  store=ArtifactStore(self.base/"train/boundaries",new_base=self.base)
  store.publish("round_0032",{"state.pt":b"complete fixture state"},{"state_sha":"a"*64})
  manifest=store.root/"round_0032/manifest.json"
  self.write(store.root/"latest.json",{"round":32,"state_sha":"a"*64,
      "manifest":{"path":str(manifest),"sha256":sha_file(manifest)}})
  ArtifactStore(self.base/"train",new_base=self.base).publish("stopped_32",{"report.json":b"{}"},{"fixture":True})
  self.patches=patch.multiple(self.s,CTL=self.ctl,OLD=self.old,OLD_TRAIN=self.train,BASE=self.base)
  self.patches.start();self.addCleanup(self.patches.stop)
 def write(self,p,v):
  import json
  p.write_text(json.dumps(v))
 def test_exact_owned_process_handoff_at_committed_boundary(self):
  with patch.object(self.s,"proc",side_effect=lambda pid:self.parent if pid==100 else self.child),patch.object(self.s,"wait_exit") as wait,patch.object(self.s.os,"kill") as kill:
   self.assertEqual(self.s.handoff(),32)
  self.assertEqual([x.args[0] for x in kill.call_args_list],[100,200])
  self.assertTrue((self.ctl/"handoff_completed.json").exists());self.assertEqual(wait.call_count,2)
 def test_benchmark_failure_never_stops_existing_training(self):
  self.write(self.ctl/"benchmark.json",{"status":"FAIL"})
  with patch.object(self.s.os,"kill") as kill:
   with self.assertRaises(RuntimeError):self.s.handoff()
  kill.assert_not_called()
 def test_wrong_process_never_signalled(self):
  self.parent["cmd"]="external GPU program"
  with patch.object(self.s,"proc",side_effect=lambda pid:self.parent if pid==100 else self.child),patch.object(self.s.os,"kill") as kill:
   with self.assertRaisesRegex(RuntimeError,"exact owned"):self.s.handoff()
  kill.assert_not_called()
 def test_partial_later_claim_blocks_resume(self):
  (self.base/"train/boundaries/claim_0033").mkdir()
  with patch.object(self.s,"proc",side_effect=lambda pid:self.parent if pid==100 else self.child),patch.object(self.s,"wait_exit"),patch.object(self.s.os,"kill"):
   with self.assertRaisesRegex(RuntimeError,"later claim"):self.s.handoff()
  self.assertFalse((self.ctl/"handoff_completed.json").exists())
 def test_wrong_boundary_sha_rejected(self):
  from tbps.artifacts import read_json
  p=self.base/"train/boundaries/latest.json";v=read_json(p);v["state_sha"]="b"*64;self.write(p,v)
  with patch.object(self.s,"proc",side_effect=lambda pid:self.parent if pid==100 else self.child),patch.object(self.s,"wait_exit"),patch.object(self.s.os,"kill"):
   with self.assertRaisesRegex(RuntimeError,"binding"):self.s.handoff()
 def test_tests_from_other_version_rejected(self):
  self.write(self.ctl/"cpu_tests.json",{"status":"PASS","manifest_sha256":"0"*64})
  with patch.object(self.s.os,"kill") as kill:
   with self.assertRaisesRegex(RuntimeError,"matching CPU"):self.s.handoff()
  kill.assert_not_called()
 def test_missing_tool_source_rejected(self):
  from tbps.performance import runtime
  from tbps.artifacts import read_json
  manifest=read_json(runtime.MANIFEST);manifest["tools"].pop(next(iter(manifest["tools"])))
  p=self.ctl/"bad_manifest.json";self.write(p,manifest)
  with patch.object(runtime,"MANIFEST",p):
   with self.assertRaisesRegex(ValueError,"all runtime"):runtime.verify_version()
 def test_wrong_original_source_hash_rejected(self):
  from tbps.performance import runtime
  from tbps.artifacts import read_json
  manifest=read_json(runtime.MANIFEST);manifest["original_sources"][next(iter(manifest["original_sources"]))]="0"*64
  p=self.ctl/"bad_manifest.json";self.write(p,manifest)
  with patch.object(runtime,"MANIFEST",p):
   with self.assertRaisesRegex(ValueError,"SHA mismatch"):runtime.verify_version()


class RealImageTests(unittest.TestCase):
 def test_real_train_384_128_views_exact(self):
  import json
  from PIL import Image
  from tbps.attribute_response.sources import preflight
  from tbps.alternating.native_sources import native_session
  cfg=SOURCE/"configs/framework/gh_attribute_response.server.json"
  f,c,bundle,gate,proof=preflight(cfg,"train")
  from tbps.artifacts import read_json
  entries=read_json(Path(c["prototype"]["history_root"])/"qualification/train.json")["entries"]
  backend=types.SimpleNamespace(p=c["prototype"],entries={e["identity"]["row_id"]:e for e in entries},
      native=bundle.bindings,view_settings=ViewSettings(**f["augmentation"]))
  with native_session(c,reuse=bundle.modules):
   pipeline=ImagePipeline(backend,workers=4,window=4,pin=False)
   try:
    pixels=bundle.bindings.get("irra_image_adapter").pixels_from_pil
    for row in c["prototype"]["train_ids"][:3]:
     e=backend.entries[row]["identity"]
     with Image.open(Path(c["prototype"]["image_root"])/e["image_rel_path"]) as im:
      image=pixels(im).unsqueeze(0)
     self.assertEqual(tuple(image.shape),(1,3,384,128))
     for version in (31,32):
      expected=apply_views(image,recipe(backend.view_settings,row,version,384,128))
      self.assertTrue(torch.equal(expected,pipeline.views(row,version)))
   finally:pipeline.close()
 def test_complete_runtime_manifest_valid(self):
  from tbps.performance.runtime import verify_version
  self.assertEqual(verify_version()["schema"],"gh-performance-runtime-1")

if __name__=="__main__":unittest.main()
