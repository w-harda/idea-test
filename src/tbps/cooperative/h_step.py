"""原SCST单次Optimizer更新内加入交互奖励与逐编辑反事实贡献。"""
from dataclasses import asdict,replace
from contextlib import contextmanager
from unittest.mock import patch
import hashlib
import torch
from .objectives import need,augmented_reward,edit_credit_loss,group_edit_log_probs

class CreditReward:
 def __init__(self,base,example,constraints,config,capture):
  self.base,self.runtime,self.example,self.constraints,self.config,self.capture=base,base.runtime,example,constraints,config,capture
 def evaluate_many(self,original,variants):
  sampled=self.capture['sample'][0];greedy=self.capture['greedy'][0]
  need(tuple(variants)==(sampled.adversarial,greedy.adversarial),'same original sampled/greedy')
  counter=[self.constraints.replay(self.example,[e for j,e in enumerate(sampled.edits) if j!=i],sampled.budget_k) for i in range(len(sampled.edits))]
  # 使用原TrainGallery/原R_sim+R_rank，扩展一次文本编码批次。
  evaluator=type(self.base.evaluator)(self.base.evaluator.gallery,self.encode,1.,1.)
  rewards=evaluator.evaluate_many(original,tuple(variants)+tuple(counter))
  self.old_rewards=rewards;totals=[augmented_reward(r,self.config['h_synergy_weight']) for r in rewards]
  self.credits=[totals[0]-r for r in totals[2:]]
  self.capture['reward']=self
  self.runtime.assert_frozen()
  return tuple(replace(r,total=totals[i]) for i,r in enumerate(rewards[:2]))
 def encode(self,texts):
  from feedback_backend import validate_variant
  need(not hasattr(self,'features'),'one bounded counterfactual feature batch')
  for text in texts[1:]:validate_variant(self.example,text,self.base.budget,self.runtime.probe)
  self.texts=tuple(texts);self.features=self.runtime.encode_text(texts)
  return self.features
 def diagnostics(self):
  need(len(self.features)==3+len(self.credits),'sample/greedy plus every leave-one-edit-out')
  return {'IRRA_text_calls':1,'IRRA_text_samples':len(self.features),'G_calls':0,'IRRA_image_calls':0,
   'text_feature_numeric_sha256':hashlib.sha256(self.features.numpy().tobytes()).hexdigest(),
   'cooperative':{'raw_rewards':[asdict(r) for r in self.old_rewards],
     'augmented_totals':[augmented_reward(r,self.config['h_synergy_weight']) for r in self.old_rewards],
     'counterfactual_texts':list(self.texts[3:]),'edit_credits':self.credits,
     'base_SCST_loss':self.capture['base_loss'],'credit_loss':self.capture['credit_loss'],
     'group_log_probs':self.capture['group_log_probs'],'h_synergy_weight':self.config['h_synergy_weight'],
     'edit_credit_weight':self.config['edit_credit_weight']}}

def train_step(phase,policy_module,constraints,policy,optimizer,example,sampler,make_reward,step,row_id,config):
 capture={};original_rollout=policy.rollout;original_choose=policy_module.choose;original_loss=phase.scst_loss
 had='rollout' in policy.__dict__;old_bound=policy.__dict__.get('rollout')
 def rollout(example,mode='sample',generator=None):
  choices=[]
  def choose(*a,**kw):
   idx,lp=original_choose(*a,**kw);choices.append(lp);return idx,lp
  with patch.object(policy_module,'choose',choose):out=original_rollout(example,mode,generator)
  capture[mode]=(out,choices);return out
 def reward(k):return CreditReward(make_reward(k),example,constraints,config,capture)
 def loss(log_prob,sampled,greedy):
  base,advantage=original_loss(log_prob,sampled,greedy)
  out,lps=capture['sample'];groups=group_edit_log_probs(out.decisions,lps,out.edits)
  credit=edit_credit_loss(log_prob,groups,capture['reward'].credits,config['edit_credit_weight'])
  capture.update(base_loss=float(base.detach()),credit_loss=float(credit.detach()),group_log_probs=[float(p.detach()) for p in groups])
  return base+credit,advantage
 try:
  policy.rollout=rollout
  with patch.object(phase,'scst_loss',loss):
   return phase.train_step(policy,optimizer,example,sampler,reward,step,row_id)
 finally:
  if had:policy.rollout=old_bound
  else:del policy.rollout
  need(policy_module.choose is original_choose and phase.scst_loss is original_loss,'restore frozen H function bindings')
  capture.clear()
