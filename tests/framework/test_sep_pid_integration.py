"""生产PIDPlan集成回归，旧入口来自固定Commit原AST。"""
from pathlib import Path
from dataclasses import FrozenInstanceError
import ast,json,random
import numpy as np
import pytest,torch
from tbps.baselines.cpgc_sep import sep_bridge as bridge,sep_source as source
from tbps.contracts import PersonID
from sep_cpu_review_support import synthetic_case,exact,first_difference,rng_signature,bit_bytes,downstream
from sep_pid_integration_support import baseline_api,baseline_text,World,invoke,official_fixture,ROOT,BASE
@pytest.fixture(autouse=True)
def no_runtime(monkeypatch):
 def forbid(*a,**k):raise AssertionError('CPU integration forbids CUDA, checkpoints and backward')
 for obj,name in ((torch.cuda,'_lazy_init'),(torch.cuda,'init'),(torch.Tensor,'cuda'),(torch.nn.Module,'cuda'),(torch,'load'),(torch.Tensor,'backward'),(torch.autograd,'grad')):monkeypatch.setattr(obj,name,forbid)
 torch.set_num_threads(1);assert not torch.cuda.is_initialized();yield;assert not torch.cuda.is_initialized()
def same(a,b):assert exact(a,b),first_difference(a,b)

@pytest.mark.parametrize('seed',range(6))
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_new_production_plan_matches_old_production(seed,dtype):
 pids,s=synthetic_case(73,11,8,seed,dtype);old,p=baseline_api().pid_logits(s,pids);plan=bridge.PIDPlan.build(pids);new,q=plan.aggregate(s)
 assert p==q and plan.members==tuple(tuple(i for i,x in enumerate(pids) if x==pid) for pid in p);same(old,new)

@pytest.mark.parametrize('case',['zero','negative_zero','nan','inf','extreme','noncontiguous'])
def test_production_plan_preserves_boundary_bits(case):
 pids=('B','A','B','C','A','C');s=torch.ones(4,6)
 if case=='zero':s.zero_()
 if case=='negative_zero':s[:,::2]=-0.
 if case=='nan':s[0,0]=float('nan')
 if case=='inf':s[0,0]=float('inf');s[1,2]=float('-inf')
 if case=='extreme':s*=torch.finfo(torch.float32).max;s[1,2]=-torch.finfo(torch.float32).max
 if case=='noncontiguous':s=s.t().contiguous().t()
 same(baseline_api().pid_logits(s,pids)[0],bridge.PIDPlan.build(pids).aggregate(s)[0])

@pytest.mark.parametrize('kind',['empty','nan_pid','dimension','width'])
def test_old_and_new_production_exception_exact(kind):
 pids=('A','B');s=torch.ones(2,2)
 if kind=='empty':pids=();s=torch.empty(2,0)
 if kind=='nan_pid':pids=(float('nan'),);s=torch.ones(1,1)
 if kind=='dimension':s=torch.ones(2)
 if kind=='width':s=torch.ones(1,3)
 def fail(fn):
  try:fn()
  except Exception as ex:return(type(ex).__name__,str(ex))
  raise AssertionError('expected error')
 assert fail(lambda:baseline_api().pid_logits(s,pids))==fail(lambda:bridge.PIDPlan.build(pids).aggregate(s))

@pytest.mark.parametrize('count',[16,8200])
def test_actual_classifier_entry_cache_rng_and_scores_exact(count):
 pids=('B','A','B','C');data=torch.arange(count*4,dtype=torch.float32).reshape(count,4)*.001;stream=[str(i) for i in range(count)]+['0','1',str(count-1),'0'];records=[]
 for factory in (baseline_api().classifier,source.classifier):
  world=World(pids,lambda name,text:data[int(text)]);obj=world.factory(factory);before=rng_signature();x=invoke(obj,stream);assert before==rng_signature();records.append((x,tuple(obj.cache),obj.calls,obj.hits,list(world.encoded),list(world.scored)))
 same(records[0][0],records[1][0]);assert records[0][1:]==records[1][1:]

@pytest.mark.parametrize('seed',[0,1,7])
def test_actual_production_entry_official_selected_text_and_rng_exact(seed):
 import nltk
 c=json.loads((ROOT/'configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json').read_text());nltk.data.path.insert(0,c['nltk_root'])
 a=official_fixture(baseline_api().classifier,c,seed);b=official_fixture(source.classifier,c,seed)
 assert a==b and a['captured'] and a['captured'][0][0]==b['captured'][0][0]

def test_one_plan_per_gallery_and_classifier_no_global_reuse():
 pids=('A','B','A');world=World(pids,lambda m,t:torch.tensor([1.,2.,3.]));a=world.factory(source.classifier,'irra');b=world.factory(source.classifier,'rasa');other=World(('A','A','B'),lambda m,t:torch.tensor([1.,2.,3.]));c=other.factory(source.classifier)
 assert a.pid_plan is not b.pid_plan and a.pid_plan is not c.pid_plan
 assert a.pid_plan.members==((0,2),(1,)) and c.pid_plan.members==((0,1),(2,))
 same(invoke(a,['x']),baseline_api().pid_logits(torch.tensor([[1.,2.,3.]]),pids)[0]);same(invoke(c,['x']),baseline_api().pid_logits(torch.tensor([[1.,2.,3.]]),('A','A','B'))[0])
 with pytest.raises(FrozenInstanceError):a.pid_plan.members=()

def test_plan_built_once_not_per_candidate(monkeypatch):
 original=bridge.PIDPlan.build.__func__;calls=[]
 def build(cls,pids):calls.append(tuple(pids));return original(cls,pids)
 monkeypatch.setattr(bridge.PIDPlan,'build',classmethod(build));world=World(('B','A','B'),lambda m,t:torch.tensor([1.,2.,3.]));obj=world.factory(source.classifier);invoke(obj,['a','b','a','c']);invoke(obj,['d']);assert len(calls)==1

def test_input_scores_and_all_rng_unchanged():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);pids,s=synthetic_case(3074,1000,8,42);copy=s.clone();before=rng_signature();plan=bridge.PIDPlan.build(pids);plan.aggregate(s);same(s,copy);assert before==rng_signature()

def test_exact_authorized_production_delta_only():
 old=baseline_text('src/tbps/baselines/cpgc_sep/sep_source.py');expected=old.replace('prepare_caption,pid_logits,legal_semantic_flags','prepare_caption,pid_logits,legal_semantic_flags,PIDPlan',1).replace('super().__init__();self.backend=backend;self.cache=OrderedDict();self.calls=0;self.hits=0','super().__init__();self.backend=backend;self.cache=OrderedDict();self.calls=0;self.hits=0\n   self.pid_plan=PIDPlan.build(tuple(x.pid for x in manifest.gallery))',1).replace('value,order=pid_logits(matrix,tuple(x.pid for x in manifest.gallery));','value,order=self.pid_plan.aggregate(matrix);',1)
 assert (ROOT/'src/tbps/baselines/cpgc_sep/sep_source.py').read_text()==expected
 current=(ROOT/'src/tbps/baselines/cpgc_sep/sep_bridge.py').read_text();node=next(n for n in ast.parse(current).body if isinstance(n,ast.ClassDef) and n.name=='PIDPlan');lines=current.splitlines(keepends=True);without=''.join(lines[:node.lineno-2]+lines[node.end_lineno:]).replace('from dataclasses import dataclass\n','',1).replace('from tbps.contracts import PersonID\n','',1)
 assert without==baseline_text('src/tbps/baselines/cpgc_sep/sep_bridge.py')


def test_actual_classifier_production_two_cpu_threads_exact():
 old_threads=torch.get_num_threads()
 try:
  torch.set_num_threads(2);pids,matrix=synthetic_case(3074,1000,1,20261103);records=[]
  for factory in (baseline_api().classifier,source.classifier):
   world=World(pids,lambda m,t:matrix[0]);obj=world.factory(factory);before=rng_signature();records.append(invoke(obj,['CPU-thread-fixture']));assert before==rng_signature()
  same(records[0],records[1])
 finally:torch.set_num_threads(old_threads)
