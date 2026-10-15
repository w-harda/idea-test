"""官方IRRA图像/文本编码；G-v2仅接收RGB像素，全部inference/eval。"""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from PIL import Image
from torch.nn import functional as F
from eval_common import *
from eval_metrics import normalized
import irra_image_adapter as adapter
from inference_once import load_generator,generate_once
from phase26_support import state_hash

class ForwardBudget:
 def __init__(self,n_gallery,n_query):
  self.limits={"text":n_query,**{k:n_gallery for k in GROUPS},**{f"G_{k}":n_gallery for k in GROUPS[1:]}}
  self.samples={k:0 for k in self.limits};self.calls={k:0 for k in self.limits}
 def consume(self,key,n):
  if type(n) is not int or n<1 or self.samples[key]+n>self.limits[key]:raise ValueError("real forward sample limit exceeded: "+key)
  if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():raise ValueError("inference mode required")
  self.samples[key]+=n;self.calls[key]+=1

def pixels(rows):
 batch=[]
 for row in rows:
  path=(IMAGES/row["image"]).resolve()
  if not path.is_relative_to(IMAGES.resolve()):raise ValueError("unsafe source image")
  io.verify({str(path):row["image_sha256"]})
  # 与官方read_image相同的PIL RGB读取；单次失败即停止，避免官方无限IO重试。
  with Image.open(path) as im:batch.append(adapter.pixels_from_pil(im))
 return torch.stack(batch)

class RealBackend:
 def __init__(self,protocol,budget):
  sizes={"smoke":(10,10),"diagnostic100":(100,100),"full":(3074,6156)}
  if protocol.get("mode") not in sizes or (protocol["gallery_count"],protocol["query_count"])!=sizes[protocol["mode"]]:
   raise ValueError("real backend fixed sample limit mismatch")
  if protocol["mode"]!="smoke" and not protocol.get("user_manual_run_confirmed"):
   raise ValueError("real large backend requires user manual confirmation")
  if protocol["mode"]=="smoke" and protocol["reuse_clean_reference"]:
   raise ValueError("real Smoke requires fresh selected10+10 encoding")
  self.protocol=protocol;self.budget=budget;self.encoder=adapter.load_official("cuda")
  self.encoder.eval().requires_grad_(False);self.encoder_hash=state_hash(self.encoder.state_dict())
  _,TextDataset,_=official_imports();self.TextDataset=TextDataset
  self.g=None;self.g_group=None;self.g_hash=None;self.handles=[];self.family=None
  def hook(module,args):
   if module.training or any(p.requires_grad for p in module.parameters()):raise ValueError("IRRA not frozen")
   self.budget.consume(self.family,len(args[0]))
  self.handles.append(self.encoder.register_forward_pre_hook(hook))
 def text(self,rows):
  ds=self.TextDataset([r["pid"] for r in rows],[r["caption"] for r in rows],text_length=77,truncate=True)
  tokens=torch.stack([ds[i][1] for i in range(len(ds))]).cuda()
  self.budget.consume("text",len(rows))
  feature=F.normalize(self.encoder.model.encode_text(tokens).float(),dim=-1).cpu()
  normalized(feature,len(rows));return feature,{"text_samples":len(rows),"G_samples":0,"IRRA_image_samples":0}
 def ensure_g(self,family):
  if self.g_group==family:return
  self.verify_g()
  self.g=None
  epoch=int(family.removeprefix("epoch"))
  self.g=load_generator(self.protocol["checkpoint_paths"][str(epoch)],"cuda").eval().requires_grad_(False)
  self.g_group=family;self.g_hash=state_hash(self.g.state_dict())
 def verify_g(self):
  if self.g is not None and (self.g.training or any(p.requires_grad or p.grad is not None for p in self.g.parameters())
      or state_hash(self.g.state_dict())!=self.g_hash):raise ValueError("G state changed")
 def image(self,family,rows):
  self.family=family;image=pixels(rows).cuda();linfs=[];g_calls=0
  if family!="clean":
   self.ensure_g(family)
   def hook(module,args):
    if len(args)!=1 or args[0].requires_grad or module.training:raise ValueError("G must be image-only inference")
    self.budget.consume("G_"+family,len(args[0]))
   h=self.g.register_forward_pre_hook(hook)
   try:adv=generate_once(self.g,image)["adversarial"]
   finally:h.remove()
   for a,b in zip(adv,image):
    linf=float((a.double()-b.double()).abs().max())
    if linf>8/255 or not torch.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError("strict adversarial budget/range violation")
    linfs.append(linf)
   g_calls=len(rows)
  else:adv=image;linfs=[0.]*len(rows)
  feature=self.encoder(adv).cpu();normalized(feature,len(rows))
  return feature,{"text_samples":0,"G_samples":g_calls,"IRRA_image_samples":len(rows),"linfs":linfs}
 def finish(self):
  self.verify_g()
  if state_hash(self.encoder.state_dict())!=self.encoder_hash or any(p.grad is not None for p in self.encoder.parameters()):
   raise ValueError("IRRA parameters/buffers changed")
  for h in self.handles:h.remove()
  return {"IRRA_parameters_buffers_exact":True,"G_parameters_exact":True,
   "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,"peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20}
