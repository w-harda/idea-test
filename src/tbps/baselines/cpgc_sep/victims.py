"""三数据集官方Victim配置：复用原加载隔离、Tokenizer与原生评分；新模型空间，不复用旧Clean。"""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from contextlib import contextmanager
import json,io
from tbps.artifacts import ArtifactError,checked,digest
from tbps.contracts import FileRef
from tbps.models.base import ModelSpace,RepresentationBatch,ScoreBatch
from tbps.models.irra import IRRAAdapter,IRRAView
from tbps.models.irra_source import official_model_modules
from tbps.models.irra_io import official_tokenizer,PREPROCESS as IRRA_PREPROCESS,METRICS
from tbps.models.rasa import load_rasa,RasaAdapter,validate_payload,PREPROCESS as RASA_PREPROCESS,SCORING
from tbps.config import IRRA
from .dataset import require
class VictimSource:
 def __init__(self,config,model,dataset):
  self.config=config;self.model=model;self.dataset=dataset
  base=config["victims"][model];self.data={**base,**base["datasets"][dataset]}
  self.root=self.data["root"];self.checkpoint=FileRef(**self.data["checkpoint"])
  if model=="irra":
   self.clip_bootstrap=FileRef(**self.data["clip_bootstrap"]);self.bpe=FileRef(**self.data["bpe"])
   self.code=tuple((k,FileRef(v["path"],v["sha256"])) for k,v in self.data["code"].items())
  self.fingerprint=digest({"source":self.data,"runtime":config["runtime_versions"]["victim"]})
 def verify(self):
  checked(self.checkpoint)
  if self.model=="rasa":
   import subprocess
   require(subprocess.check_output(["git","-C",self.root,"rev-parse","HEAD"],text=True).strip()==self.config["victim_git"]["rasa"],"RaSa official Git identity")
   require(not subprocess.check_output(["git","-C",self.root,"status","--porcelain"],text=True).strip(),"RaSa official tree modified")
  if self.model=="irra":
   require(self.data["num_classes"]=={"cuhk":11003,"icfg":3102,"rstp":3701}[self.dataset],"IRRA training PID count")
  elif self.dataset!="cuhk":
   r=self.config["resources"]["rasa_"+self.dataset]
   require(self.checkpoint.sha256==r["sha256"] and self.checkpoint.path==r["path"],"RaSa publisher dataset resource binding")
  for p,h in self.data["files"].items():checked(FileRef(p,h))
  if self.model=="irra":
   for r in [self.clip_bootstrap,self.bpe]:checked(r)
   for k,r in self.code:checked(r)
  require(self.data["weight_dataset"]==self.dataset,"Victim weight dataset mismatch")
  require(self.data["model"]==self.model,"Victim source model mismatch")
 def ref(self,n):return dict(self.code)[n]
 def settings(self):
  import yaml
  checked(FileRef(**self.data["base_config"]))
  v=yaml.load(Path(self.data["base_config"]["path"]).read_text(),Loader=yaml.FullLoader)
  require(v["pretrain_choice"]=="ViT-B/16" and tuple(v["img_size"])==(384,128) and v["text_length"]==77,"native IRRA architecture")
  # 唯一数据域接线：分类头数来自对应数据集Checkpoint CPU审计；其他官方架构设置不变。
  v["dataset_name"]={"cuhk":"CUHK-PEDES","icfg":"ICFG-PEDES","rstp":"RSTPReid"}[self.dataset]
  return v
 @property
 def space(self):
  if self.model=="irra":
   return ModelSpace("irra",self.checkpoint.sha256,self.fingerprint,digest({"native_PIL":"RGB Resize384x128 bilinear ToTensor",**IRRA_PREPROCESS}),digest({"bpe":asdict(self.bpe),"context":77}),digest(METRICS),"official_convert_weights_mixed_parameters;float32_features")
  return ModelSpace("rasa",self.checkpoint.sha256,self.fingerprint,digest({"native_PIL":"RGB direct bicubic384x384",**RASA_PREPROCESS}),digest({"bert":self.data["bert"],"pre_caption_words":50,"tokens":50}),digest({**SCORING,"fusion_batch":self.config["evaluation"].get("fusion_batch",128)}),"native_fp32")
class NativeIRRAView(IRRAView):
 def _run(self,callback):
  import torch
  self._adapter._frozen()
  with torch.no_grad(),torch.autocast("cuda",enabled=False):
   value=callback()
  return value.detach().clone()
class NativeIRRA(IRRAAdapter):
 def inference(self):return NativeIRRAView(self,False)
 def _initialize_native(self,backend,tokenizer,space):
  self._backend=backend.eval().requires_grad_(False);self._tokenizer=tokenizer
  self.spec=IRRA;self.space=space;self.source_verified=True
class NativeRasa(RasaAdapter):
 def score(self,query,gallery):
  """复用官方Retrieval.py Top128全局候选→全部128-pair分块融合→ITM正类logit。"""
  import torch
  q,g=query.payload,gallery.payload
  validate_payload(q,len(q["global"]),"text");validate_payload(g,len(g["global"]),"image")
  ng=len(g["global"]);require(ng>=128 or self.smoke,"Full native Top128")
  result=torch.full((len(q["global"]),ng),-100.,dtype=torch.float32)
  with torch.inference_mode():
   sims=q["global"].to(self.device)@g["global"].to(self.device).T
   for i,row in enumerate(sims):
    ids=row.topk(k=min(128,ng)).indices
    batch=getattr(self,"fusion_batch",128)
    require(type(batch)is int and 1<=batch<=128,"native fusion batch range")
    for start in range(0,len(ids),batch):
     selected=ids[start:start+batch]
     image=g["tokens"][selected.cpu()].to(self.device)
     image_atts=torch.ones(image.size()[:-1],dtype=torch.long,device=self.device)
     output=self.backend.text_encoder.bert(encoder_embeds=q["tokens"][i:i+1].to(self.device).repeat(len(selected),1,1),
       attention_mask=q["attention"][i:i+1].to(self.device).repeat(len(selected),1),
       encoder_hidden_states=image,encoder_attention_mask=image_atts,return_dict=True,mode="fusion")
     scores=self.backend.itm_head(output.last_hidden_state[:,0,:])[:,1]
     result[i,selected.cpu()]=scores.float().cpu()
    self.counts["fusion_pairs"]+=len(ids)
  self.counts["score_calls"]+=1
  return ScoreBatch(result,self.space.id,self.space.score_id)
@contextmanager
def load_victim(config,model,dataset,device="cuda:0",*,smoke=False):
 import torch
 from unittest.mock import patch
 require(device=="cuda:0","explicit cuda:0 required")
 source=VictimSource(config,model,dataset);source.verify()
 if model=="rasa":
  with load_rasa(source,device=device,smoke=smoke) as base:
   adapter=NativeRasa(base.backend,base.tokenizer,base.pre_caption,base.space,device,smoke=smoke)
   adapter.neural_counts=base.neural_counts
   adapter.fusion_batch=config["evaluation"].get("fusion_batch",128)
   yield adapter,adapter.backend,source
  return
 def local_clip(url,root):
  require(url.split("/")[-2]==source.clip_bootstrap.sha256,"CLIP bootstrap source")
  return source.clip_bootstrap.path
 with official_model_modules(source) as (build,clip):
  with patch.object(clip,"_download",local_clip):
   backend=build.build_model(SimpleNamespace(**source.settings()),num_classes=source.data["num_classes"])
  ck=torch.load(source.checkpoint.path,map_location="cpu",weights_only=False)
  require(tuple(ck["model"]["classifier.weight"].shape)==(source.data["num_classes"],512),"IRRA classifier dataset identity")
  backend.load_state_dict(ck["model"],strict=True);del ck
 backend=backend.eval().requires_grad_(False).to(device)
 adapter=NativeIRRA.__new__(NativeIRRA)
 adapter._initialize_native(backend,official_tokenizer(source),source.space)
 try:yield adapter,backend,source
 finally:del adapter,backend
def native_pixels(path,model):
 from PIL import Image
 from torchvision import transforms
 size=(384,128) if model=="irra" else (384,384)
 interpolation=transforms.InterpolationMode.BILINEAR if model=="irra" else transforms.InterpolationMode.BICUBIC
 with Image.open(path) as im:
  return transforms.Compose([transforms.Resize(size,interpolation=interpolation),transforms.ToTensor()])(im.convert("RGB")).unsqueeze(0)
def encode_images(adapter,x,model):
 return adapter.inference().encode_images(x.to(next(adapter._backend.parameters()).device)).payload.cpu() if model=="irra" else adapter.encode_images(x)
def encode_texts(adapter,texts,model):
 return adapter.inference().encode_texts(texts)[0].payload.cpu() if model=="irra" else adapter.encode_texts(texts)
def global_features(x):return x["global"] if isinstance(x,dict) else x
def snapshot(model):
 from tbps.engines.provenance import payload_digest
 return {k:payload_digest(v.detach().cpu()) for k,v in model.state_dict().items()}
def frozen(model):
 require(not any(m.training for m in model.modules()),"Victim eval invariant")
 require(all(not p.requires_grad and p.grad is None for p in model.parameters()),"Victim frozen/no gradient")
