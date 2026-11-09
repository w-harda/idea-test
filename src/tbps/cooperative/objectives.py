"""可微四模式交互、残余证据权重、多描述目标及逐编辑贡献。"""
import math
import torch
from tbps.artifacts import ArtifactError

def need(ok,message):
 if not ok:raise ArtifactError(message)

def interaction(clean_image,adv_image,clean_text,adv_text):
 need(clean_image.shape==adv_image.shape==clean_text.shape==adv_text.shape,'four-mode matched shape')
 s00=(clean_image*clean_text).sum(-1).mean();s10=(adv_image*clean_text).sum(-1).mean()
 s01=(clean_image*adv_text).sum(-1).mean();s11=(adv_image*adv_text).sum(-1).mean()
 return s10+s01-s11-s00,{'s00':s00,'s10':s10,'s01':s01,'s11':s11}

def residual_weights(image,jvps,prior,*,mix=.5,temperature=1.):
 need(jvps.ndim==2 and image.shape==(1,jvps.shape[1]) and prior.shape==(len(jvps),),'attribute evidence shape')
 need(0<=mix<=1 and temperature>0 and torch.isfinite(prior).all() and (prior>=0).all(),'attribute weight parameters')
 if not len(jvps):return prior.clone(),prior.clone()
 with torch.no_grad():
  unit=torch.nn.functional.normalize(jvps.detach(),dim=-1)
  evidence=(unit*image.detach()).sum(-1)
  centered=(evidence-evidence.mean())/evidence.std(unbiased=False).clamp_min(1e-6)
  logits=centered/temperature+prior.detach().clamp_min(1e-12).log()
  weighted=len(prior)*logits.softmax(0)
  result=(1-mix)*torch.ones_like(weighted)+mix*weighted
  need(torch.isfinite(result).all() and (result>0).all(),'finite positive detached weights')
 return result,evidence

def caption_set_loss(image,texts,temperature=.1):
 need(image.ndim==2 and len(image)==1 and texts.ndim==2 and image.shape[1]==texts.shape[1] and len(texts)>0,'caption set shape')
 need(temperature>0,'positive caption temperature')
 scores=(image*texts).sum(-1)
 return temperature*(torch.logsumexp(scores/temperature,0)-math.log(len(texts)))

def augmented_reward(reward,weight):
 need(weight>=0 and math.isfinite(weight),'reward coefficient')
 return reward.total+weight*reward.synergy_increment

def edit_credit_loss(log_prob,group_log_probs,credits,weight):
 need(len(group_log_probs)==len(credits) and weight>=0,'one contribution per actual character edit')
 if not credits:return log_prob*0
 values=torch.tensor(credits,dtype=log_prob.dtype,device=log_prob.device).detach()
 need(torch.isfinite(values).all(),'finite counterfactual credits')
 return -weight*(values*torch.stack(group_log_probs)).mean()

def group_edit_log_probs(decisions,log_probs,edits):
 need(len(decisions)==len(log_probs),'complete action probability trace')
 groups=[];current=[]
 for d,lp in zip(decisions,log_probs):
  if d['head']=='attribute':current=[lp]
  elif d['head']=='position':current.append(lp)
  elif d['head']=='replacement':
   current.append(lp);groups.append(torch.stack(current).sum());current=[]
  else:raise ArtifactError('unknown original action head')
 need(len(groups)==len(edits),'attribute/position/replacement group per emitted edit')
 return groups
