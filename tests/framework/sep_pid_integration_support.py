"""新旧生产入口CPU对照夹具；仅合成分数，不加载真实编码器。"""
from pathlib import Path
from types import SimpleNamespace
import ast,subprocess,random,sys
import numpy as np
import torch
from tbps.contracts import PersonID
from tbps.baselines.cpgc_sep import sep_source as source,sep_bridge as bridge
from sep_cpu_review_support import exact,first_difference,bit_bytes,rng_signature
ROOT=Path(__file__).resolve().parents[2]
BASE='769648d0b8330e7a9bf93a096abac1397fb13940'
def baseline_text(path):return subprocess.check_output(['git','show',BASE+':'+path],cwd=ROOT,text=True)
def baseline_api():
 ns=dict(vars(bridge));node=next(n for n in ast.parse(baseline_text('src/tbps/baselines/cpgc_sep/sep_bridge.py')).body if isinstance(n,ast.FunctionDef) and n.name=='pid_logits');exec(compile(ast.Module(body=[node],type_ignores=[]),'git:'+BASE+':sep_bridge.py','exec'),ns);pid=ns['pid_logits']
 ns=dict(vars(source));ns['pid_logits']=pid;node=next(n for n in ast.parse(baseline_text('src/tbps/baselines/cpgc_sep/sep_source.py')).body if isinstance(n,ast.FunctionDef) and n.name=='classifier');exec(compile(ast.Module(body=[node],type_ignores=[]),'git:'+BASE+':sep_source.py','exec'),ns)
 return SimpleNamespace(pid_logits=pid,classifier=ns['classifier'])
class Backend(torch.nn.Module):
 def __init__(self):super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
class World:
 def __init__(self,pids,score_fn):
  self.pids=tuple(pids);self.score_fn=score_fn;self.encoded=[];self.scored=[];self.tokens={};self.texts=[]
  self.manifest=SimpleNamespace(gallery=tuple(SimpleNamespace(pid=p,image_id='g'+str(i)) for i,p in enumerate(pids)))
  self.bank=SimpleNamespace(globals=torch.zeros(len(pids),1),ids=tuple('g'+str(i) for i in range(len(pids))))
 def factory(self,factory,model='irra'):
  world=self
  def encode(adapter,texts,name):
   assert len(texts)==1;world.encoded.append((name,tuple(texts)));text=texts[0]
   if text not in world.tokens:world.tokens[text]=len(world.texts);world.texts.append(text)
   return torch.tensor([[world.tokens[text]]],dtype=torch.float32)
  class Scorer:
   def __init__(self,adapter,name,banks):self.name=name
   def slice_query(self,q,start,end):return q
   def score(self,q,g):
    text=world.texts[int(q.payload[0,0])];world.scored.append((self.name,text));return SimpleNamespace(payload=world.score_fn(self.name,text).reshape(1,-1).clone())
  # 仅给此函数的globals注入CPU合成I/O，context退出还原；不改文件字节。
  ns=factory.__globals__;old=(ns['encode_texts'],ns['NativeScorer']);ns['encode_texts'],ns['NativeScorer']=encode,Scorer
  try:
   # forward运行时仍需合成I/O，调用方使用activate上下文。
   obj=factory(SimpleNamespace(space=SimpleNamespace(id=('0' if model=='irra' else '1')*64)),Backend(),model,self.bank,self.manifest)
  finally:ns['encode_texts'],ns['NativeScorer']=old
  obj._fixture_io=(ns,encode,Scorer)
  return obj
 def clear_events(self):self.encoded.clear();self.scored.clear()

def invoke(obj,texts):
 ns,encode,scorer=obj._fixture_io;old=(ns['encode_texts'],ns['NativeScorer']);ns['encode_texts'],ns['NativeScorer']=encode,scorer
 try:return obj(texts).logits
 finally:ns['encode_texts'],ns['NativeScorer']=old

class Activated:
 def __init__(self,obj):self.obj=obj
 def eval(self):return self
 def __call__(self,texts):return SimpleNamespace(logits=invoke(self.obj,texts))

def official_fixture(factory,config,seed):
 from tbps.baselines.cpgc_sep.official import sep_functions
 attack=sep_functions(config)['attack'];tree=ast.parse(Path(config['official']['sep']['root'],'SEP_Attack.py').read_text());marker=next(n.lineno for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Subscript) and isinstance(n.iter.value,ast.Name) and n.iter.value.id=='log_stack_sub')
 words={'a':0.,'man':.9,'person':.1,'wearing':.8,'dressed':.2,'blue':.7,'azure':.1,'shirt':.6,'top':.1};vocab=list(words);idx={w:i for i,w in enumerate(vocab)};pairs={'man':'person','wearing':'dressed','blue':'azure','shirt':'top'};neighbors=[[(1.,i),(.9,idx[pairs.get(w,w)])] for i,w in enumerate(vocab)]
 def scores(model,text):
  s=sum(words.get(w,0) for w in text.split())*.1*(1. if model=='irra' else .8)
  return torch.tensor([s-.03,s,s-.01,s-.02,0.,-.01,-.02,-.03,-.2,-.21,-.22,-.23])
 pids=tuple(PersonID('cuhk_pedes',str(i//4)) for i in range(12));world=World(pids,scores);objects={name:world.factory(factory,name) for name in ('irra','rasa')};ensemble=bridge.make_ensemble(config,{k:Activated(v) for k,v in objects.items()},'cpu');captured=[];oracles=[]
 def oracle(texts,label,hash_qrs):oracles.append((tuple(texts),label));return False
 def trace(frame,event,arg):
  if frame.f_code is attack.__code__:
   if event=='line' and frame.f_lineno==marker and frame.f_locals.get('ms_idx')==1:captured[:]=[(' '.join(t),float(s)) for t,s in frame.f_locals['log_stack_sub']]
   return trace
  return None
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);old=sys.gettrace();sys.settrace(trace)
 try:result=attack(1000000,['a','man','wearing','blue','shirt'],0,oracle,idx,dict(enumerate(vocab)),neighbors,ensemble,'cpu',.9,0,'fixture','CPU-only-fixture')
 finally:sys.settrace(old)
 return {'result':result,'captured':captured,'oracles':oracles,'encoded':world.encoded,'scored':world.scored,'caches':{k:(tuple(v.cache),v.calls,v.hits,tuple((text,bit_bytes(value)) for text,value in v.cache.items())) for k,v in objects.items()},'rng':rng_signature(),'DPP_weights':bit_bytes(ensemble.weights_zoo)}
