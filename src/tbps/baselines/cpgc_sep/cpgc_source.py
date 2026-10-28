"""作者ALBEF/Flickr30K epoch39预训练联合UAP，只读加载；从不创建或训练生成器。"""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from dataclasses import asdict
import sys,json,time
from tbps.artifacts import ArtifactStore,json_bytes,digest,checked,FileRef
from .dataset import require,read_test,scoped_manifest
from .official import cpgc_text_class,cpgc_pre_caption
from .runtime import attempt,unit,ref,resources,load_tensor
from .victims import snapshot,frozen
def resource_path(c,k):return c["resources"][k]["path"]
def decode_words(c,tokenizer,device):
 """eval.py原始最近词循环，保持torch.dist、topk与filter_words顺序。"""
 import torch,numpy as np
 _,filter_words=cpgc_text_class(c)
 embeddings=torch.load(resource_path(c,"cpgc_embedding"),map_location=device,weights_only=True)
 word_idxs=np.load(resource_path(c,"cpgc_word_idx"))
 word_embeddings=torch.load(resource_path(c,"cpgc_word_embedding"),map_location=device,weights_only=True)
 require(embeddings.shape[0]==1 and embeddings.shape[-1]==word_embeddings.shape[-1],"official text UAP dimensions")
 adv_words=[]
 for i in range(embeddings.size(0)):
  uap_embedding=embeddings[i]
  available_word_embeddings=word_embeddings[word_idxs]
  similarity=torch.stack([torch.dist(uap_embedding,w,p=2) for w in available_word_embeddings])
  for j in range(similarity.size(0)):
   values,indices=torch.topk(similarity,k=j+1,largest=False,sorted=True)
   min_id=word_idxs[indices[j].item()]
   word=tokenizer._convert_id_to_token(int(min_id))
   if word not in filter_words:
    adv_words.append(word);break
 require(len(adv_words)==1,"one official universal adversarial word")
 return adv_words
@contextmanager
def load_source(c,device):
 """只保留官方ALBEF加载分支；不加载无关的5种Target，也不调用train.py。"""
 import importlib,torch
 root=c["official"]["cpgc"]["root"];old=list(sys.path)
 require(not any(n=="models" or n.startswith("models.") for n in sys.modules),"source namespace occupied")
 before=set(sys.modules);sys.path.insert(0,root)
 try:
  native=importlib.import_module("models.model_retrieval")
  token=importlib.import_module("models.tokenization_bert")
  tokenizer=token.BertTokenizer.from_pretrained(c["bert_source"],local_files_only=True)
  import yaml
  v=yaml.safe_load((Path(root)/"configs/Retrieval_flickr_test.yaml").read_text())
  v["bert_config"]=str(Path(root)/"configs/config_bert.json")
  model=native.ALBEF(config=v,text_encoder=c["bert_source"],tokenizer=tokenizer)
  raw=torch.load(resource_path(c,"albef_flickr"),map_location="cpu",weights_only=True)
  state=raw.get("model",raw)
  for key in list(state):
   if "bert" in key:
    state[key.replace("bert.","")]=state[key];del state[key]
  mismatch=model.load_state_dict(state,strict=False)
  allowed_missing=lambda k: k in ("image_queue","text_queue","idx_queue","queue_ptr") or "_m." in k
  require(all(allowed_missing(k) for k in mismatch.missing_keys),"missing active Source weights")
  # 官方本身strict=False；完整记录不能把不匹配伪称strict PASS。
  model.float().eval().requires_grad_(False).to(device)
  del raw,state
  yield model,tokenizer,{"missing_keys":list(mismatch.missing_keys),"unexpected_keys":list(mismatch.unexpected_keys)}
 finally:
  sys.path[:]=old
  for n in set(sys.modules)-before:
   if n=="models" or n.startswith("models.") or n=="utils":sys.modules.pop(n,None)
def run(c,dataset,scope,*,resume=False):
 import torch,resource
 require(not torch.cuda.is_initialized(),"Source CPU gates before CUDA")
 snapshot_resources=resources(c)
 pre=cpgc_pre_caption(c)
 raw=read_test(c["datasets"][dataset])
 manifest=scoped_manifest(raw,scope)
 path=Path(c["output_base"])/scope/"cpgc_source"/dataset
 start=time.monotonic()
 with attempt(c,path,scope,resume=resume) as store:
  protocol={"method":"CPGC_OFFICIAL_UAP_TBPS_JOINT","dataset":dataset,"scope":scope,"config_sha256":c["sha256"],
    "official":c["official"]["cpgc"],"UAP_selection":"ALBEF/flickr30k/epoch39; README/eval default, no Test selection",
    "resources":{k:v for k,v in c["resources"].items() if k.startswith("cpgc") or k=="albef_flickr"},
    "manifest":asdict(manifest),"caption_bridge":"official pre_caption(...,30) applied identically to Matched Clean and attack",
    "training":False,"optimizer_steps":0,"victim_feedback":False}
  if (path/"protocol").exists():store.verify("protocol",expected_provenance={"config_sha":c["sha256"],"scope":scope})
  else:store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"config_sha":c["sha256"],"scope":scope},sources=(ref(c["path"]),))
  torch.manual_seed(42);torch.cuda.manual_seed_all(42)
  with load_source(c,"cuda:0") as (model,tokenizer,load_report):
   before=snapshot(model);words=decode_words(c,tokenizer,"cuda:0");klass,_=cpgc_text_class(c)
   # 官方TextAttacker在评测时仅用ref_net.device；复用实际Source的BERT设备，不加载无用MLM。
   attacker=klass(model.text_encoder,tokenizer,cls=False,max_length=30,number_perturbation=1,
     adv_words=words,topk=10,threshold_pred_score=0.3)
   counts={"source_text_samples":0,"source_text_batches":0,"source_image_samples":0,"backward":0,"optimizer":0,"generator_forward":0}
   def hook(m,args,kwargs):
    ids=args[0] if args else kwargs.get("input_ids")
    counts["source_text_samples"]+=len(ids);counts["source_text_batches"]+=1
   handle=model.text_encoder.register_forward_pre_hook(hook,with_kwargs=True)
   try:
    for i,q in enumerate(manifest.queries):
     clean=pre(q.caption,30);key=f"text-{i:06d}"
     provenance={"kind":"CPGC_official_text","query_id":q.query_id,"raw_sha256":digest(q.caption),"config_sha256":c["sha256"],"adv_words":words}
     def produce():
      with torch.inference_mode():adv=attacker.get_adv_text(model,[clean])[0]
      a,b=clean.split(" "),adv.split(" ");edits=[{"position":j,"original":x,"replacement":y} for j,(x,y) in enumerate(zip(a,b)) if x!=y]
      require(len(a)==len(b) and len(edits)<=1,"official CPGC word budget")
      return {"text.json":json_bytes({"query_id":q.query_id,"original_caption":q.caption,"matched_clean":clean,"adversarial":adv,"edits":edits,"NOOP":not edits})}
     unit(store,key,produce,provenance,sources=(ref(path/"protocol/manifest.json"),),resume=resume)
     if i%32==0 or i+1==len(manifest.queries):print("CPGC_TEXT_PROGRESS",dataset,scope,i+1,len(manifest.queries),flush=True)
   finally:handle.remove()
   frozen(model);after=snapshot(model);require(before==after,"CPGC Source weights/buffers changed")
  files=[ref(path/f"text-{i:06d}/manifest.json") for i in range(len(manifest.queries))]
  report={"status":"PASS","scope":scope,"dataset":dataset,"queries":len(manifest.queries),"gallery":len(manifest.gallery),
    "adv_words":words,"Source":"ALBEF Flickr30K official","training":False,"source_load_report":load_report,
    "counts":counts,"source_before":before,"source_after":after,"resources_before":snapshot_resources,
    "peak_GPU_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_GPU_reserved_bytes":torch.cuda.max_memory_reserved(),
    "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"elapsed_seconds":time.monotonic()-start}
  store.publish("completed",{"report.json":json_bytes(report)},{"kind":"CPGC_official_source_completed"},sources=tuple(files))
 print("CPGC_OFFICIAL_UAP_SOURCE_PASS",scope,dataset,flush=True)
 return 0
