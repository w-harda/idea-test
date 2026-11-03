"""独立CPU审查原型；生产模块不导入本文件，不执行攻击或加载模型。"""
from dataclasses import dataclass
from collections import OrderedDict
from contextlib import contextmanager
import hashlib,random,sys
import numpy as np
import torch
from tbps.contracts import PersonID
from tbps.baselines.cpgc_sep.dataset import require
from tbps.baselines.cpgc_sep.sep_bridge import pid_logits

@dataclass(frozen=True)
class PIDPlan:
 gallery_count:int
 order:tuple
 members:tuple
 fast_domain:bool
 @classmethod
 def build(cls,gallery_pids):
  order=tuple(dict.fromkeys(gallery_pids))
  fast=all(type(pid) in (PersonID,str) for pid in gallery_pids)
  if fast:
   by_pid={pid:i for i,pid in enumerate(order)};groups=[[] for _ in order]
   for i,pid in enumerate(gallery_pids):groups[by_pid[pid]].append(i)
  else:
   # 非正式Manifest类型沿用原相等比较，保留NaN PID等产生空组的原行为。
   groups=[[i for i,p in enumerate(gallery_pids) if p==pid] for pid in order]
  return cls(len(gallery_pids),order,tuple(tuple(x) for x in groups),fast)
 def aggregate(self,scores):
  require(scores.ndim==2 and scores.shape[1]==self.gallery_count,'complete native Gallery scores')
  # 精确保留原Python list高级索引、成员顺序、max(dim=1)、stack(dim=1)。
  values=torch.stack([scores[:,list(indices)].max(dim=1).values for indices in self.members],dim=1)
  return values,self.order

def bit_bytes(x):return x.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and bit_bytes(a)==bit_bytes(b)
def first_difference(a,b):
 if a.shape!=b.shape or a.dtype!=b.dtype:return {'shape_dtype':(str(a.dtype),str(b.dtype),list(a.shape),list(b.shape))}
 x=a.detach().contiguous().reshape(-1).view(torch.uint8);y=b.detach().contiguous().reshape(-1).view(torch.uint8);where=(x!=y).nonzero()
 return None if not len(where) else {'byte_offset':int(where[0,0]),'reference_byte':int(x[where[0,0]]),'prototype_byte':int(y[where[0,0]])}
def rng_signature():
 s=np.random.get_state()
 return (repr(random.getstate()),s[0],s[1].tobytes(),s[2:],bit_bytes(torch.get_rng_state()))

def synthetic_case(gallery=3074,pids=1000,batch=1,seed=20261103,dtype=torch.float32):
 g=torch.Generator(device='cpu').manual_seed(seed)
 labels=[PersonID('cuhk_pedes',str(i%pids)) for i in range(gallery)]
 permutation=torch.randperm(gallery,generator=g).tolist();labels=tuple(labels[i] for i in permutation)
 scores=torch.randn(batch,gallery,generator=g,dtype=dtype)
 return labels,scores

class ScoreLookup:
 """原Ensemble的CPU已给定logits输入夹具；没有神经模型。"""
 def __init__(self,values):self.values=values;self.calls=[]
 def eval(self):return self
 def __call__(self,texts):
  from types import SimpleNamespace
  self.calls.extend(texts)
  return SimpleNamespace(logits=self.values[[int(t) for t in texts]])

def downstream(config,logits_a,logits_b,*,weights=(.3,.7),label=0):
 from tbps.baselines.cpgc_sep.sep_bridge import make_ensemble
 models={'irra':ScoreLookup(logits_a),'rasa':ScoreLookup(logits_b)}
 e=make_ensemble(config,models,'cpu');e.weights=torch.tensor(weights,dtype=logits_a.dtype)
 texts=[str(i) for i in range(len(logits_a))]
 score,unique=e(texts,label,unique=True)
 # SEP原候选选择使用Python stable sorted，而非替换成新的torch排序算法。
 rank=tuple(sorted(range(len(texts)),key=lambda i:float(unique[i])))
 return {'softmax_a':torch.softmax(logits_a,-1),'softmax_b':torch.softmax(logits_b,-1),'margin_weighted':score,'margin_equal':unique,'rank':rank,'selected':None if not rank else texts[rank[0]],'calls':{k:v.calls for k,v in models.items()}}

@contextmanager
def pause_trace():
 old=sys.gettrace();sys.settrace(None)
 try:yield
 finally:sys.settrace(old)

def synthetic_selector(reducer,scores,pids,heavy_calls=2):
 """固定合成候选的观测夹具；不调用官方attack，不对应历史44/45条。"""
 values=None
 for _ in range(heavy_calls):values,_=reducer(scores,pids)
 log_stack_sub=[([str(i)],float(v)) for i,v in enumerate(values[:,0])]
 log_stack_sub=sorted(log_stack_sub,key=lambda x:x[-1])
 for ms_idx in range(2):
  for asn_text,score in log_stack_sub[:10]:
   pass
 return log_stack_sub

def observe_selector(scores,pids,*,pause=False):
 selected=[];count={'calls':0,'lines':0};target=synthetic_selector.__code__
 def tracer(frame,event,arg):
  if event=='call':count['calls']+=1
  if frame.f_code is target:
   if event=='line':
    count['lines']+=1
    if frame.f_locals.get('ms_idx')==1 and 'log_stack_sub' in frame.f_locals:selected[:]=frame.f_locals['log_stack_sub']
   return tracer
  return None
 def reduce(s,p):
  if pause:
   with pause_trace():return pid_logits(s,p)
  return pid_logits(s,p)
 old=sys.gettrace();sys.settrace(tracer)
 try:result=synthetic_selector(reduce,scores,pids)
 finally:sys.settrace(old)
 return result,selected,count
