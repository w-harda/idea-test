"""冻结 AttributeInstance，经原 IRRA text JVP 映射后调用原样官方 TTA。"""
from __future__ import annotations
import copy, hashlib, json, math, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from attributes.generator_framework import IMAGE_EPSILON, project_image
from attributes.tta_adapter import _tokenize, load_official_attack

REPO=Path('/home/lzf/ldx/projects/idea-TBPS-test1')
P51=REPO/'prototypes/attribute_importance_step51_20261003'
P61=REPO/'prototypes/power_dynamic_topk_step61_20261003'
BASIS=REPO/'prototypes/attribute_unit_v55_interface_20261003/outputs/full/semantic_bases.json'
OFFICIAL=Path('/home/lzf/ldx/external/Transform_to_Transfer_Attack')


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class SelectedAttribute:
 metadata: dict
 selection_rank: int
 w: torch.Tensor
 Q: torch.Tensor
 @property
 def instance_id(self):return self.metadata['instance_id']


def prepared_plan(payload,record,bases,device='cpu'):
 # 已保存的gamma3 clean selection，禁止重新调用scorer/TopK。
 if payload['gamma']!=3 or payload['caption_id']!=record['row_id']:raise ValueError('frozen gamma3 caption plan required')
 order=payload['ranked_attributes'];chosen=payload['selected_attributes']
 if chosen!=order[:payload['k_star']] or len(chosen)>4:raise ValueError('stored selection must be fixed raw-ranked prefix')
 if any(a['S_raw']<b['S_raw'] for a,b in zip(order,order[1:])):raise ValueError('raw ranking changed')
 lookup={x['instance_id']:x for x in record['attributes']};out=[]
 for rank,a in enumerate(chosen,1):
  x=lookup[a['instance_id']];e=a.get('semantic_representation');span=a['source_span']
  if a['unresolved'] or x['status']!='scored' or not e or e['basis_rank']<=0:raise ValueError('unsupported selected attribute')
  if a!={**x['instance'],'S_raw':x['S_raw']}:raise ValueError('frozen selected metadata or score mismatch')
  if record['caption'][span['start']:span['end']]!=span['raw'] or span['raw']!=a['raw_text']:raise ValueError('source span invalid')
  b=bases[a['semantic_axis']]
  if b['basis_id']!=e['basis_id'] or b['rank']!=e['basis_rank']:raise ValueError('basis provenance mismatch')
  Q=torch.tensor(b['Q'],dtype=torch.float64,device=device).reshape(512,b['rank']);w=torch.tensor(x['w'],dtype=torch.float32,device=device)
  if not torch.allclose(Q.T@Q,torch.eye(b['rank'],dtype=Q.dtype,device=device),atol=1e-8,rtol=1e-8):raise ValueError('basis not orthonormal')
  if abs(float(w.double().norm())-1)>2e-6:raise ValueError('localization must be frozen unit L2')
  out.append(SelectedAttribute(copy.deepcopy(a),rank,w.detach(),Q.detach()))
 if len({x.instance_id for x in out})!=len(out):raise ValueError('duplicate instance IDs')
 return tuple(out)


def attribute_delta(a):
 # e为冻结projected_context；只取方向，不重新评分、训练或构造basis。
 e=torch.tensor(a.metadata['semantic_representation']['projected_context'],dtype=torch.float64,device=a.w.device)
 if e.shape!=(512,) or not torch.isfinite(e).all() or not e.norm()>0:raise ValueError('zero/unavailable attribute representation; no fallback')
 residual=(e-a.Q@(a.Q.T@e)).norm()/e.norm()
 if float(residual)>1e-6:raise ValueError('frozen representation outside its basis')
 d=(e/e.norm()).float();B=a.w[:,None,None]*d[None,None,:]
 if abs(float(B.double().norm())-1)>3e-6:raise ValueError('unit contextual direction failed')
 return B.detach()


class TextTransport:
 """Context只定义原caption forward的线性化点，不作为额外loss项。"""
 def __init__(self,victim,caption):
  self.victim=victim;self.caption=caption;self.block=victim.model.base_model.transformer.resblocks[7];self.H=None
  def capture(m,args,out):self.H=out.detach().clone()
  handle=self.block.register_forward_hook(capture)
  try:
   with torch.no_grad():self.clean=victim.encode_texts([caption]).detach()
  finally:handle.remove()
  if self.H.shape!=(77,1,512) or self.H.dtype!=torch.float32:raise ValueError('block8 fp32 required')
  zero=self.feature(torch.zeros((),device=self.H.device),torch.zeros_like(self.H))
  if not torch.equal(zero,self.clean):raise ValueError('zero intervention not exact')
 def feature(self,t,B):
  # 复用原encoder剩余blocks/ln_final/text_projection/EOT/normalize；不旁路编码。
  h=self.block.register_forward_hook(lambda m,args,out:out.detach()+t*B)
  try:return self.victim.encode_texts([self.caption])
  finally:h.remove()
 def read(self,a):
  B=attribute_delta(a)
  if B.shape!=self.H.shape:raise ValueError('frozen localization length mismatch')
  t=torch.zeros((),device=self.H.device,requires_grad=True)
  # 原MHA使用SDPA；math backend支持JVP所需的double backward。
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
   value,v=torch.autograd.functional.jvp(lambda eta:self.feature(eta,B),t,torch.ones_like(t),create_graph=False,strict=True)
   with torch.no_grad():fd=(self.feature(torch.tensor(.01,device=t.device),B)-self.feature(torch.tensor(-.01,device=t.device),B))/.02
  if not torch.equal(value,self.clean):raise ValueError('JVP base differs from original encoder')
  if not torch.isfinite(v).all() or not v.norm()>0:raise ValueError('zero/nonfinite transported direction; no fallback')
  error=float((fd-v).double().norm()/v.double().norm())
  # v在最终归一化text feature切空间中，与最终image feature合法点积。
  return v.detach(),{'instance_id':a.instance_id,'direction_norm':float(v.double().norm()),'central_fd_relative_error':error,'tangent_radial_dot':float((self.clean*v).sum()),'unit_hidden_delta_norm':float(B.double().norm()),'zero_exact':True,'mapping':'JVP original block8->blocks9-12->ln_final->text_projection->EOT->L2'}


class FixedGuidanceBridge(nn.Module):
 def __init__(self,source,features,labels):
  super().__init__();self.source=source;self.features=features.detach().clone();self.labels=tuple(labels);self.text_requests=[]
 def inference_image(self,pixels):return self.source.inference_image(pixels)
 def inference_text(self,batch):
  if tuple(batch.texts)!=self.labels:raise ValueError('unexpected text condition: full caption/raw text forbidden in attribute mode')
  self.text_requests.append(tuple(batch.texts));return {'text_feat':self.features}


class AttributeTTA:
 """只替换官方Attack的text supervision；img_attack与投影原样复用。"""
 def __init__(self,source,original,*,steps=10,transforms=6,scales=(.5,.75,1.25,1.5),official=OFFICIAL):
  self.source=source.eval();self.original=original.detach().clone();self.steps=steps;self.scales=scales
  self.attack=load_official_attack(official)(None,_tokenize,imgs_eps=IMAGE_EPSILON,step_size=2/255);self.attack.N_trans=transforms
  self.momentum=torch.zeros_like(original);self.events=[]
 def step(self,current,attribute,vector):
  if float((current-self.original).abs().max())>IMAGE_EPSILON+1e-6:raise ValueError('incoming cumulative budget invalid')
  bridge=FixedGuidanceBridge(self.source,vector,['attribute_unit:'+attribute.instance_id])
  with torch.enable_grad():proposed,m=self.attack.img_attack(bridge,list(bridge.labels),current.detach(),self.original,[0],self.steps,self.momentum,current.device,scales=self.scales)
  if not torch.isfinite(proposed).all() or not torch.isfinite(m).all():raise ValueError('nonfinite TTA update')
  self.momentum=m.detach();image=project_image(self.original,proposed.detach()).detach()
  self.events.append({'instance_id':attribute.instance_id,'selection_rank':attribute.selection_rank,'linf':float((image-self.original).abs().max()),'labels_passed':list(bridge.labels),'text_encoder_calls_during_attack':0,'caption_loss_terms':0})
  return image
 def run(self,plan,vectors):
  current=self.original.clone()
  for a in plan:current=self.step(current,a,vectors[a.instance_id])
  if [x['instance_id'] for x in self.events]!=[a.instance_id for a in plan]:raise ValueError('fixed order changed')
  return current


def image_gradient(source,image,vector):
 x=image.detach().clone().requires_grad_(True);loss=-(source.encode_images(x)*vector.detach()).sum()
 g,=torch.autograd.grad(loss,x)
 if not torch.isfinite(g).all() or not g.norm()>0:raise ValueError('zero/nonfinite attribute image gradient')
 return g.detach()
