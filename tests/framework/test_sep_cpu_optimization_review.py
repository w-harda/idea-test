"""固定合成输入CPU审查；不加载权重、编码真实文本或运行SEP attack。"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import ast,json,random,sys
import numpy as np
import pytest,torch
from tbps.contracts import PersonID
from tbps.baselines.cpgc_sep.sep_bridge import pid_logits
from sep_cpu_review_support import PIDPlan,exact,first_difference,rng_signature,synthetic_case,downstream,observe_selector,pause_trace
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json'
@pytest.fixture(scope='module')
def config():return json.loads(CONFIG.read_text())
@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
 def forbid(*a,**k):raise AssertionError('CPU review forbids CUDA, real checkpoint load, backward or optimizer')
 for obj,name in ((torch.cuda,'_lazy_init'),(torch.cuda,'init'),(torch.Tensor,'cuda'),(torch.nn.Module,'cuda'),(torch,'load'),(torch.Tensor,'backward'),(torch.autograd,'grad')):monkeypatch.setattr(obj,name,forbid)
 assert not torch.cuda.is_initialized()
 torch.set_num_threads(1)
 yield
 assert not torch.cuda.is_initialized()

def same(a,b):assert exact(a,b),first_difference(a,b)

@pytest.mark.parametrize('seed',range(12))
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_seeded_multi_pid_bitwise(seed,dtype):
 pids,s=synthetic_case(53,9,8,seed,dtype);plan=PIDPlan.build(pids);old,order=pid_logits(s,pids);new,new_order=plan.aggregate(s)
 assert order==new_order;assert plan.fast_domain;same(old,new)
 assert plan.members==tuple(tuple(i for i,p in enumerate(pids) if p==pid) for pid in order)

@pytest.mark.parametrize('pids',[('B','A','B','C','A'),('A',)*17,tuple('ABCDEFGH'),('B','A','A','A','D','A','A','A','D','C'),('a','b')*16])
@pytest.mark.parametrize('batch',[0,1,8])
def test_uneven_members_and_repeated_positives(pids,batch):
 s=torch.arange(batch*len(pids),dtype=torch.float32).reshape(batch,len(pids))*.01;plan=PIDPlan.build(pids);x,p=pid_logits(s,pids);y,q=plan.aggregate(s)
 assert p==q==tuple(dict.fromkeys(pids));same(x,y);assert sorted(i for group in plan.members for i in group)==list(range(len(pids)))

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('kind',['ties','negative','extreme','negative_zero','nan','infinity'])
def test_ties_extremes_and_nonfinite_original_semantics(dtype,kind):
 pids=('B','A','B','C','A','C');limit=torch.finfo(dtype).max
 vals={'ties':[1,1,1,1,1,1],'negative':[-5,-1,-5,-9,-3,-9],'extreme':[-limit,limit,limit,-limit,limit,0],'negative_zero':[-0.,0.,0.,-0.,-0.,0.],'nan':[float('nan'),1.,2.,-3.,float('nan'),4.],'infinity':[float('-inf'),float('inf'),0.,float('inf'),1.,float('-inf')]}[kind]
 s=torch.tensor([vals,vals[::-1]],dtype=dtype);plan=PIDPlan.build(pids);old,_=pid_logits(s,pids);new,_=plan.aggregate(s);same(old,new)
 for pid,indices in zip(plan.order,plan.members):
  ref=[i for i,p in enumerate(pids) if p==pid];a=s[:,ref].max(1);b=s[:,list(indices)].max(1);same(a.values,b.values);assert torch.equal(a.indices,b.indices)
 same(torch.softmax(old,-1),torch.softmax(new,-1))

@pytest.mark.parametrize('layout',['transpose','slice','expanded','plain'])
def test_noncontiguous_inputs(layout):
 pids,s=synthetic_case(30,7,4,9)
 if layout=='transpose':s=s.t().contiguous().t()
 if layout=='slice':s=torch.stack((s,s),-1)[...,0]
 if layout=='expanded':s=s[:1].expand(4,-1)
 x,p=pid_logits(s,pids);y,q=PIDPlan.build(pids).aggregate(s);assert p==q;same(x,y)

@pytest.mark.parametrize('failure',['empty_gallery','wrong_width','wrong_dimension','nan_pid_empty_group','unhashable_pid'])
def test_original_error_type_and_message_preserved(failure):
 pids=('A','B');s=torch.ones(1,2)
 if failure=='empty_gallery':pids=();s=torch.empty(2,0)
 if failure=='wrong_width':s=torch.ones(1,3)
 if failure=='wrong_dimension':s=torch.ones(2)
 if failure=='nan_pid_empty_group':pids=(float('nan'),);s=torch.ones(1,1)
 if failure=='unhashable_pid':pids=(['A'],);s=torch.ones(1,1)
 def outcome(fn):
  try:fn()
  except Exception as e:return(type(e).__name__,str(e))
  raise AssertionError('expected original exception')
 assert outcome(lambda:pid_logits(s,pids))==outcome(lambda:PIDPlan.build(pids).aggregate(s))

@pytest.mark.parametrize('seed',range(8))
@pytest.mark.parametrize('tie',[False,True])
def test_original_ensemble_softmax_margin_and_candidate_selection_exact(config,seed,tie):
 pids,a=synthetic_case(45,7,12,seed);_,b=synthetic_case(45,7,12,seed+99);a[8]=a[2];b[8]=b[2]
 if tie:a.zero_();b.zero_()
 plan=PIDPlan.build(pids);old=downstream(config,pid_logits(a,pids)[0],pid_logits(b,pids)[0]);new=downstream(config,plan.aggregate(a)[0],plan.aggregate(b)[0])
 for k in ('softmax_a','softmax_b','margin_weighted','margin_equal'):same(old[k],new[k])
 assert old['rank']==new['rank'] and old['selected']==new['selected'] and old['calls']==new['calls']
 if tie:assert new['rank']==tuple(range(12)) and new['selected']=='0'

@pytest.mark.parametrize('seed',range(4))
def test_grouping_and_scoring_preserve_all_rng_inputs_and_identities(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);pids,s=synthetic_case(30,4,4,seed);before=rng_signature();original=s.clone();ids=tuple('q'+str(i) for i in range(4));ident=id(pids)
 plan=PIDPlan.build(pids);plan.aggregate(s);assert before==rng_signature();same(s,original);assert id(pids)==ident and ids==tuple('q'+str(i) for i in range(4))

@pytest.mark.parametrize('size',[16,8200])
def test_production_classifier_fixture_cache_keys_order_and_eviction(monkeypatch,size):
 # 只替换独立测试进程中的合成编码/评分；生产文件字节保持，未加载任何真实模型。
 from tbps.baselines.cpgc_sep import sep_source as source
 pids=('B','A','B','C');plan=PIDPlan.build(pids);all_scores=torch.arange(size*4,dtype=torch.float32).reshape(size,4)*.001
 stream=[str(i) for i in range(size)]+['0','1',str(size-1),'0'];manifest=SimpleNamespace(gallery=tuple(SimpleNamespace(pid=p) for p in pids));bank=SimpleNamespace(globals=torch.zeros(4,1));events=[]
 class Backend(torch.nn.Module):
  def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
 class Scorer:
  def __init__(self,*a,**kw):pass
  def slice_query(self,q,start,end):return q
  def score(self,q,g):events.append(int(q.payload[0,0]));return SimpleNamespace(payload=all_scores[int(q.payload[0,0]):int(q.payload[0,0])+1])
 monkeypatch.setattr(source,'encode_texts',lambda a,texts,m:torch.tensor([[float(texts[0])]]));monkeypatch.setattr(source,'NativeScorer',Scorer)
 outputs=[]
 for reducer in (pid_logits,lambda scores,p:plan.aggregate(scores)):
  monkeypatch.setattr(source,'pid_logits',reducer);events.clear();obj=source.classifier(SimpleNamespace(space=SimpleNamespace(id='0'*64)),Backend(),'irra',bank,manifest)
  before=rng_signature();out=obj(stream).logits;assert rng_signature()==before
  outputs.append((out,tuple(obj.cache),obj.calls,obj.hits,tuple(events)))
 same(outputs[0][0],outputs[1][0]);assert outputs[0][1:]==outputs[1][1:]
 assert outputs[0][2]==len(stream) and len(outputs[0][1])<=8192

@pytest.mark.parametrize('pause',[False,True])
def test_trace_observer_capture_same_without_touching_official_attack(pause):
 pids,s=synthetic_case(17,4,7,4);before=sys.gettrace();ref=observe_selector(s,pids,pause=False);test=observe_selector(s,pids,pause=pause)
 assert sys.gettrace() is before and ref[:2]==test[:2]
 if pause:assert test[2]['calls']<ref[2]['calls']

def test_trace_restored_on_exception():
 old=sys.gettrace()
 def marker(*a):return marker
 sys.settrace(marker)
 try:
  with pytest.raises(ValueError):
   with pause_trace():assert sys.gettrace() is None;raise ValueError('fixture')
  assert sys.gettrace() is marker
 finally:sys.settrace(old)

@pytest.mark.parametrize('kind',['zero','constant_nonzero'])
def test_original_importance_loop_boundary_bounded_cpu_observation(config,kind):
 from tbps.baselines.cpgc_sep.official import sep_functions
 ns=sep_functions(config);opt=ns['optimize'];opt.__globals__['get_important_order']=lambda *a:([0.,0.] if kind=='zero' else [1.,1.],[0,1])
 class LoopBudget(Exception):pass
 seen=[]
 tree=ast.parse(Path(config['official']['sep']['root'],'SEP_Attack.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='optimize');line=next(n.body[0].lineno for n in ast.walk(node) if isinstance(n,ast.While))
 def tracer(frame,event,arg):
  if frame.f_code is opt.__code__:
   if event=='line' and frame.f_lineno==line:
    seen.append(frame.f_locals['wi_score'].clone())
    if len(seen)==16:raise LoopBudget('synthetic boundary only; no official function modification')
   return tracer
  return None
 old=sys.gettrace();sys.settrace(tracer)
 try:
  with pytest.raises(LoopBudget):opt(['a','b'],['a','b'],None,0,[(0,'a'),(1,'b')],1,1,{})
 finally:sys.settrace(old)
 assert len(seen)==16
 if kind=='zero':assert all(torch.count_nonzero(x).item()==0 for x in seen)
 else:assert all((x.max()-x.min()).item()==0 for x in seen)

def test_namespaced_pid_not_collapsed():
 p=(PersonID('cuhk_pedes','1'),PersonID('rstp_reid','1'),PersonID('cuhk_pedes','1'));plan=PIDPlan.build(p)
 assert plan.order==p[:2] and plan.members==((0,2),(1,))

def test_source_files_still_frozen():
 import hashlib
 old=Path('/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-no-gpu-floor-20261101')
 import subprocess
 for name in ('sep_bridge.py','sep_source.py','cache.py','official.py','victims.py'):
  relative='src/tbps/baselines/cpgc_sep/'+name
  baseline=subprocess.check_output(['git','show','769648d0b8330e7a9bf93a096abac1397fb13940:'+relative],cwd=ROOT)
  assert hashlib.sha256((old/relative).read_bytes()).digest()==hashlib.sha256(baseline).digest()
  if name not in ('sep_bridge.py','sep_source.py'):
   assert hashlib.sha256((ROOT/relative).read_bytes()).digest()==hashlib.sha256(baseline).digest()
 # 两个授权文件的精确AST变更白名单由test_sep_pid_integration额外核验。


def test_nan_payload_bits_are_preserved():
 s=torch.tensor([[0x7fc00001,0x7fc00002,0,0x3f800000]],dtype=torch.int32).view(torch.float32);pids=('B','B','A','A')
 same(pid_logits(s,pids)[0],PIDPlan.build(pids).aggregate(s)[0])

def test_adjacent_float32_values_and_order(config):
 one=torch.tensor(1.,dtype=torch.float32);hi=torch.nextafter(one,torch.tensor(float('inf')));lo=torch.nextafter(one,torch.tensor(float('-inf')))
 s=torch.stack((torch.stack((one,hi,lo)),torch.stack((hi,one,lo)),torch.stack((lo,one,hi))))
 pids=('B','A','C');plan=PIDPlan.build(pids);a=downstream(config,pid_logits(s,pids)[0],pid_logits(s.flip(1),pids)[0]);b=downstream(config,plan.aggregate(s)[0],plan.aggregate(s.flip(1))[0])
 for k in ('softmax_a','softmax_b','margin_weighted','margin_equal'):same(a[k],b[k])
 assert a['rank']==b['rank'] and a['selected']==b['selected']

def test_hot_aggregation_removes_personid_equality_scan(monkeypatch):
 pids,s=synthetic_case(64,8,1,12);plan=PIDPlan.build(pids);original=PersonID.__eq__;count=[0]
 def compare(a,b):count[0]+=1;return original(a,b)
 monkeypatch.setattr(PersonID,'__eq__',compare);old,_=pid_logits(s,pids);old_calls=count[0];count[0]=0;new,_=plan.aggregate(s)
 assert old_calls>=64*8 and count[0]==0;same(old,new)

def test_plan_is_only_valid_for_its_fixed_gallery():
 # 记录错误复用的反例，不能因长度相同就在其他Manifest复用旧Plan。
 original=('A','B','A');changed=('A','A','B');s=torch.tensor([[1.,2.,3.]])
 old_plan=PIDPlan.build(original)
 assert not exact(old_plan.aggregate(s)[0],pid_logits(s,changed)[0])


@pytest.mark.parametrize('seed',[0,1,7])
def test_official_attack_synthetic_final_text_capture_and_rng_exact(config,seed):
 # 仅固定人工词表/分数夹具；非真实Query、非Source/Full，不加载任何网络。
 import nltk
 from tbps.baselines.cpgc_sep.official import sep_functions
 from tbps.baselines.cpgc_sep.sep_bridge import make_ensemble
 nltk.data.path.insert(0,config['nltk_root']);attack=sep_functions(config)['attack']
 source_ast=ast.parse(Path(config['official']['sep']['root'],'SEP_Attack.py').read_text());marker=next(n.lineno for n in ast.walk(source_ast) if isinstance(n,ast.For) and isinstance(n.iter,ast.Subscript) and isinstance(n.iter.value,ast.Name) and n.iter.value.id=='log_stack_sub')
 wordvals={'a':0.,'man':.9,'person':.1,'wearing':.8,'dressed':.2,'blue':.7,'azure':.1,'shirt':.6,'top':.1};words=list(wordvals);idx={w:i for i,w in enumerate(words)};pairs={'man':'person','wearing':'dressed','blue':'azure','shirt':'top'};neighbors=[[(1.,i),(.9,idx[pairs.get(w,w)])] for i,w in enumerate(words)]
 pids=tuple(PersonID('cuhk_pedes',str(i//4)) for i in range(12));plan=PIDPlan.build(pids);records=[]
 for reducer in (lambda s:pid_logits(s,pids),plan.aggregate):
  calls=[];oracles=[];captured=[]
  class Model:
   def __init__(self,scale):self.scale=scale
   def eval(self):return self
   def __call__(self,texts):
    vals=[sum(wordvals.get(w,0) for w in t.split())*.1*self.scale for t in texts]
    full=torch.tensor([[s-.03,s,s-.01,s-.02,0.,-.01,-.02,-.03,-.2,-.21,-.22,-.23] for s in vals]);logits=reducer(full)[0]
    from sep_cpu_review_support import bit_bytes
    calls.append((tuple(texts),bit_bytes(logits)));return SimpleNamespace(logits=logits)
  models={'irra':Model(1.),'rasa':Model(.8)};ensemble=make_ensemble(config,models,'cpu')
  def oracle(texts,label,hash_qrs):oracles.append((tuple(texts),label));return False
  def trace(frame,event,arg):
   if frame.f_code is attack.__code__:
    if event=='line' and frame.f_lineno==marker and frame.f_locals.get('ms_idx')==1:captured[:]=[(' '.join(t),float(s)) for t,s in frame.f_locals['log_stack_sub']]
    return trace
   return None
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);old=sys.gettrace();sys.settrace(trace)
  try:result=attack(1000000,['a','man','wearing','blue','shirt'],0,oracle,idx,dict(enumerate(words)),neighbors,ensemble,'cpu',.9,0,'fixture','CPU-fixture')
  finally:sys.settrace(old)
  from sep_cpu_review_support import bit_bytes
  records.append((result,list(captured),list(calls),list(oracles),rng_signature(),bit_bytes(ensemble.weights_zoo)))
 assert records[0]==records[1] and records[0][1]
 assert records[0][1][0][0]==records[1][1][0][0]


def test_production_two_cpu_threads_fp32_exact():
 old_threads=torch.get_num_threads()
 try:
  torch.set_num_threads(2);pids,s=synthetic_case(3074,1000,8,20261103);a,p=pid_logits(s,pids);b,q=PIDPlan.build(pids).aggregate(s)
  assert p==q;same(a,b)
 finally:torch.set_num_threads(old_threads)
