"""局部Token按不可变块落盘/按Top128读取，避免ICFG把两份图库Token全放RAM。"""
from pathlib import Path
from dataclasses import asdict
from collections import OrderedDict
import io,time
from tbps.artifacts import ArtifactStore,digest,json_bytes
from tbps.models.base import RepresentationBatch,ScoreBatch,InputBinding
from tbps.engines.provenance import BoundRepresentation,payload_digest
from tbps.engines.evaluation import EvaluationEngine
from tbps.engines.native_cache import compact
from .runtime import unit,load_tensor,ref
from .dataset import require
class BlockCache:
 """进程内有界CPU缓存；只缓存通过Manifest/SHA验证的不可变Token块。"""
 def __init__(self,maximum_bytes):
  self.maximum_bytes=maximum_bytes;self.used=0;self.entries=OrderedDict()
  self.hits=0;self.misses=0;self.read_seconds=0.;self.bytes_read=0
 @staticmethod
 def nbytes(x):
  return sum(v.numel()*v.element_size() for v in x.values()) if isinstance(x,dict) else x.numel()*x.element_size()
 def get(self,key,loader):
  if key in self.entries:
   self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
  self.misses+=1;t=time.monotonic();x=loader();self.read_seconds+=time.monotonic()-t
  n=self.nbytes(x);self.bytes_read+=n
  while self.entries and self.used+n>self.maximum_bytes:
   _,old=self.entries.popitem(last=False);self.used-=self.nbytes(old)
  if n<=self.maximum_bytes:self.entries[key]=x;self.used+=n
  return x
 def statistics(self):
  return dict(hits=self.hits,misses=self.misses,resident_bytes=self.used,limit_bytes=self.maximum_bytes,read_seconds=self.read_seconds,bytes_read=self.bytes_read)
class FeatureBank:
 def __init__(self,c,store,family,adapter,manifest,ids,input_hashes,*,modality,chunk=8):
  self.c,self.store,self.family,self.adapter,self.manifest=c,store,family,adapter,manifest
  self.ids=tuple(ids);self.input_hashes=tuple(input_hashes);self.modality=modality;self.chunk=chunk;self.lru=OrderedDict()
  self.binding=InputBinding(adapter.space.id,digest(asdict(manifest)),self.input_hashes,modality,self.ids)
  self.globals=None
  shared=getattr(adapter,"token_block_cache",None)
  self.block_cache=shared
 def key(self,start):return self.family+"-"+str(start//self.chunk).zfill(6)
 def provenance(self,start,end):
  return {"kind":"native_feature_block","binding":asdict(self.binding),"start":start,"end":end,"config_sha":self.c["sha256"]}
 def put(self,start,end,payload,*,resume=False):
  provenance=self.provenance(start,end)
  def producer():
   x=compact(payload() if callable(payload) else payload);b=io.BytesIO()
   import torch
   torch.save(x,b)
   return {"features.pt":b.getvalue(),"details.json":json_bytes({"payload_sha256":payload_digest(x)})}
  unit(self.store,self.key(start),producer,provenance,sources=(ref(self.store.root/"protocol/manifest.json"),),resume=resume)
 def block(self,start):
  key=self.key(start);end=min(start+self.chunk,len(self.ids))
  self.store.verify(key,expected_provenance=self.provenance(start,end))
  import torch
  x=torch.load(self.store.root/key/"features.pt",map_location="cpu",weights_only=True)
  return x
 def read_globals(self):
  import torch
  gs=[]
  for start in range(0,len(self.ids),self.chunk):
   x=self.block(start);gs.append((x["global"] if isinstance(x,dict) else x).detach().clone())
  self.globals=torch.cat(gs)
  require(len(self.globals)==len(self.ids),"encoded member count")
  return self.globals
 def fetch(self,indices):
  import torch
  parts=[]
  for i in indices:
   require(0<=i<len(self.ids),"feature index")
   start=(i//self.chunk)*self.chunk
   if self.block_cache is not None:
    x=self.block_cache.get((str(self.store.root),self.family,start),lambda:self.block(start))
   else:
    if start not in self.lru:
     self.lru[start]=self.block(start)
     if len(self.lru)>16:self.lru.popitem(last=False)
    x=self.lru[start];self.lru.move_to_end(start)
   j=i-start
   parts.append({k:v[j:j+1] for k,v in x.items()} if isinstance(x,dict) else x[j:j+1])
  return {k:torch.cat([p[k] for p in parts]) for k in parts[0]} if isinstance(parts[0],dict) else torch.cat(parts)
 def bound(self,origin=None,originals=()):
  g=self.globals if self.globals is not None else self.read_globals()
  refs=tuple(ref(self.store.root/self.key(i)/"manifest.json") for i in range(0,len(self.ids),self.chunk))
  return BoundRepresentation(RepresentationBatch(g,self.adapter.space.id,self.modality),self.binding,payload_digest(g),refs,origin,originals)
class NativeScorer:
 execution_kind="materialized_native";query_independent_scores=True;input_gradients=False;score_direction="higher_better"
 def __init__(self,adapter,model,banks,*,store=None,config=None,resume=False):
  self.score_store,self.config,self.resume=store,config,resume
  self.adapter,self.model,self.banks=adapter,model,banks
  self.space,self.spec,self.source_verified=adapter.space,adapter.spec,True
  self.refs={id(b.globals):b for b in banks.values()};self.cursor=None;self.native_pairs=0
 def slice_query(self,batch,start,end):
  self.cursor=(self.refs[id(batch.payload)],start,end)
  return RepresentationBatch(batch.payload[start:end].clone(),self.space.id,"text")
 def score(self,query,gallery):
  if self.score_store is None:return self._compute(query,gallery)
  qb,start,end=self.cursor;gb=self.refs[id(gallery.payload)]
  key="scorechunk-"+qb.family+"--"+gb.family+"-"+str(start).zfill(6)
  provenance={"kind":"exact_native_score_chunk","space":self.space.id,"query_binding":digest(asdict(qb.binding)),"gallery_binding":digest(asdict(gb.binding)),"start":start,"end":end}
  def producer():
   result=self._compute(query,gallery).payload
   from .runtime import tensor_bytes
   return {"scores.pt":tensor_bytes(result)}
  unit(self.score_store,key,producer,provenance,sources=(ref(self.score_store.root/"protocol/manifest.json"),),resume=self.resume)
  result=load_tensor(self.score_store,key,"scores.pt")
  print("NATIVE_EXACT_SCORE_PROGRESS",self.model,qb.family,gb.family,end,len(qb.ids),flush=True)
  return ScoreBatch(result,self.space.id,self.space.score_id)
 def _compute(self,query,gallery):
  import torch
  if self.model=="irra":
   device=next(self.adapter._backend.parameters()).device
   with torch.inference_mode():scores=(query.payload.to(device)@gallery.payload.to(device).T).float().cpu()
   return ScoreBatch(scores,self.space.id,self.space.score_id)
  qb,start,end=self.cursor;gb=self.refs[id(gallery.payload)]
  result=torch.full((end-start,len(gb.ids)),-100.,dtype=torch.float32)
  # 候选在完整Gallery上按官方GPU FP32全局相似度确定，绝非缩小图库再TopK。
  with torch.inference_mode():
   sims=query.payload.to(self.adapter.device)@gallery.payload.to(self.adapter.device).T
   for j,row in enumerate(sims):
    indices=row.topk(k=min(128,len(gb.ids))).indices.cpu().tolist()
    q=RepresentationBatch(qb.fetch([start+j]),self.space.id,"text")
    g=RepresentationBatch(gb.fetch(indices),self.space.id,"image")
    # 复用原RaSa融合评分；子集是完整Top128，全部128个候选均执行官方ITM。
    score=self.adapter.score(q,g).payload[0]
    result[j,indices]=score
    self.native_pairs+=len(indices)
  return ScoreBatch(result,self.space.id,self.space.score_id)
class TBPSEvaluationEngine(EvaluationEngine):
 """仅扩展显式Dataset身份门禁；run/_validate/完整排名仍为既有Engine原实现。"""
 def __init__(self,manifest,adapter,checkpoint):
  from tbps.models.base import ModelAdapter
  from tbps.contracts import DatasetManifest
  from tbps.artifacts import checked
  require(type(manifest)is DatasetManifest and manifest.dataset_id in ("cuhk_pedes","icfg_pedes","rstp_reid"),"official TBPS dataset identity")
  require(manifest.official_split=="test" and isinstance(adapter,ModelAdapter),"native official Test evaluator")
  require(adapter.source_verified and not adapter.input_gradients,"verified inference adapter")
  require(checkpoint.sha256==adapter.space.checkpoint_sha256,"Victim provenance")
  checked(checkpoint)
  self.manifest,self.adapter,self.direction,self.execution=manifest,adapter,"higher_better","materialized_native"
  self.model_sources=(checkpoint,);self.identity=manifest.evaluation_identity()
  self.manifest_sha256=digest(asdict(manifest));self.space,self.spec=adapter.space,adapter.spec
