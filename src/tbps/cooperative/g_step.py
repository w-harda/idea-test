"""暂态扩展原G K+1训练；原函数/模块文件不改，退出时还原所有绑定。"""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from .objectives import interaction,residual_weights,caption_set_loss,need

@contextmanager
def cooperative_g(serial,extras,config,output):
 original_attribute=serial.attribute_losses;original_joint=serial.joint_losses;original_weighted=serial._weighted
 state={'index':0,'weights':None,'pending':None,'details':[]}
 def attribute(feature,z,local,**kwargs):
  if state['weights'] is None:
   state['weights'],evidence=residual_weights(feature,extras['jvps'].to(feature),extras['prior'].to(feature),
     mix=config['residual_mix'],temperature=config['attribute_temperature'])
   output.update(attribute_weights=state['weights'].detach().cpu().tolist(),attribute_evidence=evidence.cpu().tolist(),
                 attribute_prior=extras['prior'].tolist(),weights_mean_one=True)
  idx=state['index'];state['index']+=1;terms=original_attribute(feature,z,local,**kwargs)
  state['pending']={'attribute_index':idx,'local_weight':float(state['weights'][idx])}
  return terms
 def joint(feature,z,local,has_local):
  terms=original_joint(feature,z,local,has_local)
  synergy,four=interaction(extras['clean_image'].to(feature),feature,extras['clean_text'].to(feature),z.detach())
  texts=torch.cat((extras['caption_set'].to(feature),z.detach()),0)
  setloss=caption_set_loss(feature,texts,config['caption_temperature'])
  terms['cooperative_extra']=-config['g_synergy_weight']*synergy+config['caption_set_weight']*setloss
  state['pending']={'synergy':float(synergy.detach()),'four_modes':{k:float(v.detach()) for k,v in four.items()},
    'caption_set_loss':float(setloss.detach()),'caption_scores':(feature.detach()*texts).sum(-1).cpu().tolist(),'caption_set_size':len(texts),'extra':float(terms['cooperative_extra'].detach())}
  return terms
 def weighted(terms,weights,joint):
  base=original_weighted(terms,weights,joint)
  if joint:total=base+terms['cooperative_extra']
  else:total=base+weights.lambda_local*(state['weights'][state['index']-1]-1)*terms['local']
  state['details'].append(dict(state['pending'],base_total=float(base.detach()),total=float(total.detach())))
  return total
 try:
  with patch.object(serial,'attribute_losses',attribute),patch.object(serial,'joint_losses',joint),patch.object(serial,'_weighted',weighted):
   yield state['details']
 finally:
  need(serial.attribute_losses is original_attribute and serial.joint_losses is original_joint and serial._weighted is original_weighted,
       'restore frozen G function bindings')
