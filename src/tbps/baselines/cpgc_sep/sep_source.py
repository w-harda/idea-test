"""SEP检索适配：作者原始候选流程，IRRA+RaSa原生检索PID接口，不训练模型。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import ExitStack
from collections import OrderedDict
from types import SimpleNamespace
import sys,json,time,random,pickle,resource,math,copy,ast
from tbps.artifacts import json_bytes,digest
from tbps.models.base import RepresentationBatch
from .dataset import read_test,scoped_manifest,require
from .runtime import attempt,unit,ref,tensor_bytes,resources
from .victims import load_victim,native_pixels,encode_images,encode_texts,snapshot,frozen,global_features
from .cache import FeatureBank,NativeScorer
from .official import sep_functions
from .sep_bridge import make_ensemble,prepare_caption,pid_logits,legal_semantic_flags
def capture_state():
 import torch,numpy as np
 n=np.random.get_state()
 return {"python":random.getstate(),"numpy":{"name":n[0],"keys":n[1].tolist(),"position":n[2],"gauss":n[3],"cached":n[4]},
  "torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all()}
def restore_state(s):
 import torch,numpy as np
 def tuples(x):return tuple(tuples(y) for y in x) if isinstance(x,(list,tuple)) else x
 random.setstate(tuples(s["python"]));n=s["numpy"];np.random.set_state((n["name"],np.array(n["keys"],dtype=np.uint32),n["position"],n["gauss"],n["cached"]))
 torch.set_rng_state(s["torch_cpu"]);torch.cuda.set_rng_state_all(s["torch_cuda"])
class EphemeralQuery:
 family="candidate_query"
 def __init__(self,payload):
  self.payload=payload;self.globals=global_features(payload);self.ids=tuple(str(i) for i in range(len(self.globals)))
 def fetch(self,indices):
  import torch
  return {k:v[indices].clone() for k,v in self.payload.items()} if isinstance(self.payload,dict) else self.payload[indices].clone()
def classifier(adapter,backend,model,bank,manifest):
 import torch
 class PIDClassifier(torch.nn.Module):
  def __init__(self):
   super().__init__();self.backend=backend;self.cache=OrderedDict();self.calls=0;self.hits=0
  def forward(self,texts):
   values=[]
   for text in texts:
    self.calls+=1
    if text in self.cache:
     self.hits+=1;value=self.cache[text]
    else:
     payload=encode_texts(adapter,[text],model);qb=EphemeralQuery(payload)
     scorer=NativeScorer(adapter,model,{"q":qb,"g":bank})
     q=RepresentationBatch(qb.globals,adapter.space.id,"text");g=RepresentationBatch(bank.globals,adapter.space.id,"image")
     q=scorer.slice_query(q,0,1)
     matrix=scorer.score(q,g).payload
     value,order=pid_logits(matrix,tuple(x.pid for x in manifest.gallery));value=value[0].detach().cpu()
     self.cache[text]=value
     if len(self.cache)>8192:self.cache.popitem(last=False)
    values.append(value.to(next(backend.parameters()).device))
   return SimpleNamespace(logits=torch.stack(values))
 return PIDClassifier().eval()
def run(c,dataset,victim,scope,*,resume=False):
 import torch,numpy as np,nltk
 require(not torch.cuda.is_initialized(),"CPU prerequisites before CUDA")
 snapshot_resources=resources(c);torch.set_num_threads(2);nltk.data.path.insert(0,c["nltk_root"])
 manifest=scoped_manifest(read_test(c["datasets"][dataset]),scope)
 path=Path(c["output_base"])/scope/"sep_source"/dataset/victim
 start=time.monotonic();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
 words=[line.split(" ",1)[0] for line in Path(c["resources"]["sep_vectors"]["path"]).read_text().splitlines()]
 word2idx={w:i for i,w in enumerate(words)};idx2word=dict(enumerate(words))
 with Path(c["resources"]["sep_neighbors"]["path"]).open("rb") as f:neighbors=pickle.load(f)
 require(len(neighbors)==len(words)==65713 and all(len(r)==50 for r in neighbors),"official top50/vocabulary pairing")
 functions=sep_functions(c);attack=functions["attack"]
 source_ast=ast.parse(Path(c["official"]["sep"]["root"],"SEP_Attack.py").read_text())
 # 在作者tau排序后、任何Victim成功查询前观测指定gamma分支，绝不修改候选/随机序列。
 marker=next(n.lineno for n in ast.walk(source_ast) if isinstance(n,ast.For) and isinstance(n.iter,ast.Subscript) and isinstance(n.iter.value,ast.Name) and n.iter.value.id=="log_stack_sub")
 with attempt(c,path,scope,resume=resume) as store,ExitStack() as stack:
  protocol={"method":"SEP_TBPS_IRRA_RASA_ENSEMBLE","scope":scope,"dataset":dataset,"victim":victim,"manifest":asdict(manifest),
    "config_sha256":c["sha256"],"official":c["official"]["sep"],"surrogates":["irra","rasa"],"victim_in_surrogates":True,
    "confidence":"sum w_m max(softmax(max_same_PID_native_scores)[y]-max_other_PID_probability,-k)",
    "candidate_selection":"top1 by original tau for gamma=.02 before Victim success results; no oracle-based candidate selection",
    "DPP":{"random_space":100,"keep_rule":"len(surrogates)","actual_keep":2},"training":False}
  if not (path/"protocol").exists():store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"config_sha":c["sha256"]},sources=(ref(c["path"]),))
  else:store.verify("protocol",expected_provenance={"config_sha":c["sha256"]})
  source_config=copy.deepcopy(c);source_config["evaluation"]["fusion_batch"]=c["evaluation"].get("sep_fusion_batch",16)
  adapters={};models={};before={};banks={}
  for name in ("irra","rasa"):
   adapter,model,source=stack.enter_context(load_victim(source_config,name,dataset,"cuda:0",smoke=scope=="smoke"))
   adapters[name]=adapter;models[name]=model;before[name]=snapshot(model)
   bank=FeatureBank(c,store,"gallery-"+name,adapter,manifest,tuple(g.image_id for g in manifest.gallery),tuple(g.image_sha256 for g in manifest.gallery),modality="image")
   for i in range(0,len(manifest.gallery),8):
    end=min(i+8,len(manifest.gallery))
    def produce():
     from .target import cat_payload
     return cat_payload([encode_images(adapter,native_pixels(Path(c["datasets"][dataset]["root"])/"imgs"/g.image_rel_path,name),name) for g in manifest.gallery[i:end]])
    bank.put(i,end,produce,resume=resume)
    if end%128==0 or end==len(manifest.gallery):print("SEP_GALLERY_PROGRESS",dataset,victim,scope,name,end,len(manifest.gallery),flush=True)
   bank.read_globals();banks[name]=bank
  classifiers={name:classifier(adapters[name],models[name],name,banks[name],manifest) for name in ("irra","rasa")}
  ensemble=make_ensemble(c,classifiers,"cuda:0")
  pid_order=tuple(dict.fromkeys(g.pid for g in manifest.gallery))
  target_model=classifiers[victim];oracle_calls=[]
  def oracle(texts,label,hash_qrs):
   oracle_calls.append({"input_count":len(texts),"model":victim,"phase":"official success measurement / initial Clean gate"})
   logits=target_model(texts=texts).logits
   return (torch.argmax(logits,dim=-1)!=label).squeeze()
  completed=[];resumed_cursor=-1
  for i,q in enumerate(manifest.queries):
   key=f"text-{i:06d}"
   if (path/key/"manifest.json").exists():
    require(resume,"source existing Query requires resume");store.verify(key);completed.append(i);resumed_cursor=i
   else:break
  if resumed_cursor>=0:
   state=torch.load(path/f"text-{resumed_cursor:06d}/rng.pt",map_location="cpu",weights_only=True);restore_state(state)
  total_edits=0;semantic_flags=0;noop=0;dpp_counts=[]
  for i,q in enumerate(manifest.queries):
   key=f"text-{i:06d}";clean=prepare_caption(c,q.caption);label=pid_order.index(q.pid)
   provenance={"kind":"SEP_official_text","query_id":q.query_id,"raw_sha":digest(q.caption),"config_sha":c["sha256"]}
   def produce():
    nonlocal total_edits,semantic_flags,noop
    selected=[];old_trace=sys.gettrace()
    def tracer(frame,event,arg):
     if frame.f_code is attack.__code__:
      if event=="line" and frame.f_lineno==marker and frame.f_locals.get("ms_idx")==1:
       current=frame.f_locals.get("log_stack_sub",[])
       selected[:]=[(" ".join(tokens),float(score)) for tokens,score in current]
      return tracer
     return None
    sys.settrace(tracer)
    try:
     result=attack(1000000,clean.split(),label,oracle,word2idx,idx2word,neighbors,ensemble,"cuda:0",.9,i,victim,str(path),max_change_rate=.2)
    finally:sys.settrace(old_trace)
    early=(result[0] is None)
    adv=selected[0][0] if selected else clean
    analysis=legal_semantic_flags(clean,adv)
    total_edits+=analysis["word_change_count"];semantic_flags+=len(analysis["identity_semantic_flags"]);noop+=not analysis["edits"]
    if not early:dpp_counts.append(ensemble.len_weights)
    row={"query_id":q.query_id,"original_caption":q.caption,"matched_clean":clean,"adversarial":adv,**analysis,
      "NOOP":not analysis["edits"],"NOOP_reason":"original Victim Clean wrong" if early else "no distinct legal selected candidate" if not selected else None,
      "candidate_selection":"gamma .02 / original code floor / top1 surrogate tau; no Victim feedback selection",
      "selected_surrogate_score":selected[0][1] if selected else None,"candidate_count":len(selected),
      "target_in_ensemble":True,"original_success_statistics":[None if x is None else x for x in result]}
    for classifier_ in classifiers.values():classifier_.cache.clear()
    return {"text.json":json_bytes(row),"rng.pt":tensor_bytes(capture_state())}
   unit(store,key,produce,provenance,sources=(ref(path/"protocol/manifest.json"),),resume=resume)
   if i%4==0 or i+1==len(manifest.queries):print("SEP_QUERY_PROGRESS",dataset,victim,scope,i+1,len(manifest.queries),flush=True)
  for name,model in models.items():frozen(model);require(snapshot(model)==before[name],"SEP surrogate weights/buffers changed")
  report={"status":"PASS","scope":scope,"dataset":dataset,"victim":victim,"queries":len(manifest.queries),"gallery":len(manifest.gallery),
    "surrogates":["irra","rasa"],"target_in_surrogates":True,"DPP_selected_weight_counts":dpp_counts,"edits_new_segment":total_edits,
    "semantic_flags_new_segment":semantic_flags,"NOOP_new_segment":noop,"oracle_calls":oracle_calls,
    "source_frozen":True,"model_before":before,"backward":0,"optimizer":0,"resources_before":snapshot_resources,
    "elapsed_seconds":time.monotonic()-start,"peak_GPU_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_GPU_reserved_bytes":torch.cuda.max_memory_reserved(),
    "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
  refs=tuple(ref(path/f"text-{i:06d}/manifest.json") for i in range(len(manifest.queries)))
  store.publish("completed",{"report.json":json_bytes(report)},{"kind":"SEP_source_completed"},sources=refs)
 print("SEP_TBPS_ENSEMBLE_SOURCE_PASS",scope,dataset,victim,flush=True);return 0
