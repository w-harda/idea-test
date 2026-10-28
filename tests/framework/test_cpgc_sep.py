"""本阶段CPU契约/原算法组合测试；Mock不代表真实GPU Smoke。"""
import ast,copy,json,math,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,digest
from tbps.baselines.cpgc_sep.official import definitions,cpgc_text_class,cpgc_pre_caption,sep_ensemble_class,sep_functions
from tbps.baselines.cpgc_sep.sep_bridge import make_ensemble,pid_logits,legal_semantic_flags,prepare_caption
from tbps.baselines.cpgc_sep.runtime import unit,load_tensor,tensor_bytes
from tbps.baselines.cpgc_sep.dataset import read_test,scoped_manifest,SPECS
from tbps.baselines.cpgc_sep.victims import VictimSource
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/framework/cpgc_sep_tbps.server.json"
@pytest.fixture
def c():return json.loads(CONFIG.read_text())
@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
 monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:(_ for _ in ()).throw(AssertionError("CPU test initialized CUDA")))
 torch.set_num_threads(1)
class Batch:
 def __init__(self,texts):self.texts=texts
 def to(self,device):return self
class Tokenizer:
 def __call__(self,texts,**kwargs):return Batch(texts)
 def tokenize(self,text):return [text] if text else []
class SourceFixture:
 def __init__(self):self.calls=0
 def inference_text(self,b):
  self.calls+=len(b.texts)
  out=[]
  for t in b.texts:
   words=t.split();m=next((i+1 for i,w in enumerate(words) if w=="[UNK]"),0)
   out.append([1.0,0.2+m,0.3*m])
  return {"text_feat":torch.tensor(out)}
@pytest.mark.parametrize("text",["a man in a blue shirt","a person wearing black pants","the woman walks"])
def test_cpgc_official_get_adv_text_composed(c,text):
 klass,_=cpgc_text_class(c);source=SourceFixture()
 a=klass(SimpleNamespace(device="cpu"),Tokenizer(),cls=False,max_length=30,number_perturbation=1,adv_words=["pauline"],topk=10,threshold_pred_score=.3)
 result=a.get_adv_text(source,[text])[0]
 assert len(result.split())==len(text.split())
 assert sum(x!=y for x,y in zip(result.split(),text.split()))<=1
 assert source.calls==1+len(text.split())
 assert a.generator is None
def test_cpgc_noop_all_filter_words(c):
 klass,_=cpgc_text_class(c);source=SourceFixture()
 a=klass(SimpleNamespace(device="cpu"),Tokenizer(),adv_words=["pauline"])
 assert a.get_adv_text(source,["a the is"])==["a the is"]
def test_cpgc_pre_caption_official(c):
 fn=cpgc_pre_caption(c)
 assert fn("A PERSON, wearing-blue/shirt.",30)=="a person wearing blue shirt"
 assert len(fn(" ".join(["person"]*35),30).split())==30
def test_official_source_sha_rejection(c,tmp_path):
 p=tmp_path/"x.py";p.write_text("def value():return 1\n")
 with pytest.raises(ArtifactError):definitions(p,["value"],"0"*64,{})
def test_definitions_are_original_functions(c):
 ns=sep_functions(c)
 assert ns["attack"].__code__.co_filename.endswith("/SEP_Attack.py")
 assert ns["optimize"].__code__.co_filename.endswith("/SEP_Attack.py")
def test_dpp_two_models_original_rule(c):
 klass=sep_ensemble_class(c)
 obj=klass.__new__(klass);obj.models_name={"irra":1,"rasa":2};obj.device="cpu"
 torch.manual_seed(0);obj.init_weights()
 assert obj.weights_zoo.shape==(2,2)
 assert torch.isfinite(obj.weights_zoo).all() and (obj.weights_zoo>=0).all()
 assert torch.allclose(obj.weights_zoo.sum(1),torch.ones(2))
 first=obj.next_weights();second=obj.next_weights()
 assert not torch.equal(first,second)
def test_pid_multi_positive_native_score_mapping():
 s=torch.tensor([[.2,.9,.4,-100.],[.8,.1,.6,.7]])
 z,p=pid_logits(s,["A","A","B","C"])
 assert p==("A","B","C")
 assert torch.equal(z,torch.tensor([[.9,.4,-100.],[.8,.6,.7]]))
class LogitsFixture(torch.nn.Module):
 def __init__(self,values):super().__init__();self.values=values
 def forward(self,texts):return SimpleNamespace(logits=torch.tensor([self.values]*len(texts)))
def test_original_ensemble_margin_with_tbps_io(c):
 models={"irra":LogitsFixture([2.,1.,0.]),"rasa":LogitsFixture([1.,2.,0.])}
 e=make_ensemble(c,models,"cpu");e.weights=torch.tensor([.3,.7])
 result=e(["a man"],0)
 ps=[torch.softmax(torch.tensor(v),dim=-1) for v in [[2.,1.,0.],[1.,2.,0.]]]
 expected=.3*(ps[0][0]-ps[0][1])+.7*(ps[1][0]-ps[1][1])
 assert torch.allclose(result,expected.unsqueeze(0))
def test_ensemble_fixed_members(c):
 with pytest.raises(ArtifactError):make_ensemble(c,{"irra":LogitsFixture([1,0])},"cpu")
def test_sep_default_caption_bridge(c):
 assert prepare_caption(c,"The Man's blue-shirt.")=="the man 's blue shirt"
@pytest.mark.parametrize("a,b,kind",[("a blue shirt","a red shirt","COLOR_ATTRIBUTE_CHANGED"),("a man walks","a woman walks","GENDER_ATTRIBUTE_CHANGED")])
def test_semantics_not_assumed(a,b,kind):
 r=legal_semantic_flags(a,b)
 assert r["identity_semantic_flags"][0]["type"]==kind
 assert r["full_semantic_equivalence"]=="NOT_VERIFIED"
def test_grey_gray_not_false_color_change():
 assert not legal_semantic_flags("grey shirt","gray shirt")["identity_semantic_flags"]
def test_atomic_unit_and_readback(tmp_path):
 base=tmp_path/"tbps-framework-fixture";s=ArtifactStore(base/"run",new_base=base)
 unit(s,"row",lambda:{"features.pt":tensor_bytes(torch.tensor([1.,2.]))},{"id":"x"})
 assert torch.equal(load_tensor(s,"row"),torch.tensor([1.,2.]))
def test_unfinished_claim_refuses_replay(tmp_path):
 base=tmp_path/"tbps-framework-fixture";s=ArtifactStore(base/"run",new_base=base)
 def broken():raise RuntimeError("fixture IO failure")
 with pytest.raises(RuntimeError):unit(s,"row",broken,{"id":"x"})
 assert (s.root/"row.Claim.json").exists()
 with pytest.raises(ArtifactError):unit(s,"row",lambda:{"ok":b"1"},{"id":"x"},resume=True)
def test_completed_unit_requires_resume_and_same_identity(tmp_path):
 base=tmp_path/"tbps-framework-fixture";s=ArtifactStore(base/"run",new_base=base)
 unit(s,"row",lambda:{"ok":b"1"},{"id":"x"})
 with pytest.raises(ArtifactError):unit(s,"row",lambda:{"ok":b"2"},{"id":"x"})
 with pytest.raises(ArtifactError):unit(s,"row",lambda:{"ok":b"2"},{"id":"y"},resume=True)
 unit(s,"row",lambda:(_ for _ in ()).throw(AssertionError("must not rerun")),{"id":"x"},resume=True)
def test_victim_dataset_mismatch_rejected(c):
 bad=copy.deepcopy(c);bad["victims"]["rasa"]["datasets"]["icfg"]["checkpoint"]=bad["victims"]["rasa"]["datasets"]["cuhk"]["checkpoint"]
 with pytest.raises(ArtifactError):VictimSource(bad,"rasa","icfg").verify()
def test_no_cuda_cpu_suite():
 assert not torch.cuda.is_initialized()

def test_native_rasa_full_top128_not_truncated_gallery():
 from tbps.baselines.cpgc_sep.cache import NativeScorer
 from tbps.models.base import ModelSpace,ScoreBatch,RepresentationBatch
 space=ModelSpace("rasa_fixture",*["0"*64]*5,"fixture")
 class Bank:
  def __init__(self,global_,tokens):
   self.globals=global_;self.tokens=tokens;self.ids=tuple(map(str,range(len(global_))))
  def fetch(self,indices):return {"global":self.globals[indices],"tokens":self.tokens[indices]}
 class Adapter:
  device="cpu";source_verified=True;spec=SimpleNamespace(id="rasa_fixture")
  def __init__(self):self.space=space
  def score(self,q,g):
   return ScoreBatch(g.payload["tokens"].reshape(1,-1)*2.5,space.id,space.score_id)
 q=Bank(torch.tensor([[1.,0.]]),torch.tensor([[0.]]))
 g=Bank(torch.stack([torch.arange(130,dtype=torch.float32),torch.ones(130)],1),torch.arange(130,dtype=torch.float32).unsqueeze(1))
 scorer=NativeScorer(Adapter(),"rasa",{"q":q,"g":g})
 qb=scorer.slice_query(RepresentationBatch(q.globals,space.id,"text"),0,1)
 result=scorer.score(qb,RepresentationBatch(g.globals,space.id,"image")).payload
 assert torch.equal(result[:,:2],torch.full((1,2),-100.))
 assert torch.equal(result[:,2:],(torch.arange(2,130,dtype=torch.float32)*2.5).unsqueeze(0))
 assert scorer.native_pairs==128
def test_official_defined_core_has_no_reimplementation(c):
 klass=sep_ensemble_class(c)
 for name in ["_fast_map_dpp","_generating_weights","__call__"]:
  assert getattr(klass,name).__code__.co_filename.endswith("/SEP_utils.py")

def test_sep_full_official_candidate_flow_and_tau_observer(c):
 import nltk,sys
 nltk.data.path.insert(0,c["nltk_root"])
 ns=sep_functions(c);attack=ns["attack"]
 wordvals={"a":0.0,"man":.9,"person":.1,"wearing":.8,"dressed":.2,"blue":.7,"azure":.1,"shirt":.6,"top":.1}
 class Model(torch.nn.Module):
  def __init__(self,scale):super().__init__();self.scale=scale
  def forward(self,texts):
   scores=[sum(wordvals.get(w,0) for w in t.split())*.1*self.scale for t in texts]
   return SimpleNamespace(logits=torch.tensor([[s,0.,-.2] for s in scores]))
 models={"irra":Model(1.),"rasa":Model(.8)}
 ensemble=make_ensemble(c,models,"cpu")
 words=list(wordvals);idx={w:i for i,w in enumerate(words)}
 pairs={"man":"person","wearing":"dressed","blue":"azure","shirt":"top"}
 neighbors=[[(1.,i),(.9,idx[pairs.get(w,w)])] for i,w in enumerate(words)]
 node=ast.parse(Path(c["official"]["sep"]["root"],"SEP_Attack.py").read_text())
 marker=next(n.lineno for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.iter,ast.Subscript) and isinstance(n.iter.value,ast.Name) and n.iter.value.id=="log_stack_sub")
 captured=[]
 def trace(frame,event,arg):
  if frame.f_code is attack.__code__:
   if event=="line" and frame.f_lineno==marker and frame.f_locals.get("ms_idx")==1:
    captured[:]=[(" ".join(t),float(s)) for t,s in frame.f_locals["log_stack_sub"]]
   return trace
  return None
 torch.manual_seed(0);np.random.seed(0);random.seed(0);old=sys.gettrace()
 sys.settrace(trace)
 try:result=attack(1000000,["a","man","wearing","blue","shirt"],0,lambda *args:False,idx,dict(enumerate(words)),neighbors,ensemble,"cpu",.9,0,"fixture","fixture")
 finally:sys.settrace(old)
 assert result[0]==0 and captured
 assert captured==sorted(captured,key=lambda x:x[1])
 assert ensemble.len_weights==2
 for text,score in captured:
  assert sum(a!=b for a,b in zip(text.split(),["a","man","wearing","blue","shirt"]))<=1
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize("free,util,allowed",[(18433,90,True),(10240,100,True),(10239,0,False)])
def test_shared_gpu_gate_keeps_memory_not_utilization(c,monkeypatch,free,util,allowed):
 from tbps.baselines.cpgc_sep.runtime import resources
 import tbps.baselines.cpgc_sep.runtime as rt
 def fake(args,**kwargs):
  field=next((x for x in args if x.startswith("--query-gpu=")),"")
  value=str(free) if "memory.free" in field else "24564" if "memory.total" in field else str(util) if "utilization.gpu" in field else "1234, other-user-python, 6000"
  return SimpleNamespace(returncode=0,stdout=value)
 monkeypatch.setattr(rt.subprocess,"run",fake)
 monkeypatch.setattr(rt.shutil,"disk_usage",lambda p:SimpleNamespace(free=10**13))
 monkeypatch.setattr(rt.os,"statvfs",lambda p:SimpleNamespace(f_favail=10**9))
 if allowed:assert resources(c)["memory.free"]==free
 else:
  with pytest.raises(ArtifactError,match="RESOURCE_BUSY"):resources(c)
def test_resource_query_error_safely_rejects(c,monkeypatch):
 import tbps.baselines.cpgc_sep.runtime as rt
 monkeypatch.setattr(rt.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1,stdout=""))
 with pytest.raises(ArtifactError):rt.resources(c)
def test_native_irra_runtime_is_part_of_cache_identity(c):
 first=VictimSource(c,"irra","cuhk")
 other=copy.deepcopy(c);other["runtime_versions"]["victim"]["torch"]="different"
 assert first.space.id!=VictimSource(other,"irra","cuhk").space.id
