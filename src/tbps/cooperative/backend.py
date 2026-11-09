"""双向协同仅进入训练监督；推理仍为G(I)、H(T)。"""
from contextlib import contextmanager
from pathlib import Path
import torch,json
from tbps.artifacts import read_json,json_bytes,FileRef,sha_file
from tbps.alternating.fresh_training import FreshBackend
from tbps.alternating.native_sources import native_session
from tbps.alternating.state import clone_tree,tree_sha
from tbps.alternating.formal_state import tensor_bytes
from .objectives import need
from .g_step import cooperative_g
from .h_step import train_step

def read_annotation(path):
 from tbps.artifacts import _object
 def reject(value):raise ValueError('nonfinite annotation '+value)
 value=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=_object,parse_constant=reject)
 need(isinstance(value,list),'official annotation array')
 return value

class CooperativeBackend(FreshBackend):
 def __init__(self,*args):
  super().__init__(*args);self.annotation=read_annotation(self.p['annotation']);self.annotation_sha=sha_file(self.p['annotation'])
 def condition(self,arm,row,trace):
  result=super().condition(arm,row,trace)
  action=self.rollout_log[self.p['samples'].index(row)]
  need(action['original']==trace.original and action['adversarial']==trace.adversarial,'G primary rollout identity')
  self.conditions_log[-1]['G_rollout']=clone_tree(action)
  return result
 def g_update(self,row,condition):
  cfg=self.c['cooperative'];idx=self.p['train_ids'].index(row)
  captions=self.annotation[int(row.split(':')[1])]['captions'];primary=self.entries[row]['caption']
  extra={'clean_image':self.clean[idx:idx+1].detach().clone(),
   'clean_text':self.original[row].payload['z_t'].detach().cpu().reshape(1,-1).clone(),
   'jvps':condition.payload['attribute_jvps'].detach().cpu().clone(),
   'prior':condition.payload['score_weights'].detach().cpu().clone()}
  metadata={'row':row,'annotation_sha256':self.annotation_sha,'primary_caption':primary,
   'condition_identity':condition.identity.key,'H_state':self.h_fingerprint(),'secondary':None,
   'secondary_selection':'first distinct native nontruncated training caption; no scoring selection'}
  vectors=[extra['clean_text']]
  for j,caption in enumerate(captions):
   if caption==primary or self.bundle.probe.inspect(caption).truncated:continue
   with torch.no_grad():
    example=self.bundle.frontend(caption);action=self.h.attack(example)
    self.native.get('constraints').replay(example,action.edits,action.budget_k)
    feats=self.victim.encode_texts((caption,action.adversarial)).detach().cpu().clone()
   vectors.append(feats);metadata['secondary']={'caption_index':j,'caption':caption,'action':action.record()};break
  extra['caption_set']=torch.cat(vectors,0)
  details={};serial=self.native.get('serial_training')
  with cooperative_g(serial,extra,cfg,details) as steps:result=super().g_update(row,condition)
  need(len(steps)==condition.payload['K']+1,'one cooperative record per original G update')
  for rec,step in zip(result['records'],steps):rec['cooperative']=step
  key='cooperative_'+str(idx).zfill(4)
  self.store.publish(key,{'context.pt':tensor_bytes(extra),'metadata.json':json_bytes(metadata)},
    {'configuration':self.c['fresh_config_sha256'],'content_sha256':tree_sha(extra)},
    sources=(FileRef(self.gentries[row]['tensor_path'],self.gentries[row]['tensor_sha256']),))
  result['cooperative']={**details,'metadata':metadata,'context_sha256':tree_sha(extra)}
  return result
 def h_update(self,row,bank):
  self.grant.validate(self.c,'smoke');need(bank==self.last_bank,'current reward bank required')
  self.cursor['h_visits']+=1;e=self.entries[row]
  if not e['eligible']:return {'row_id':row,'fixed_NOOP':True,'optimizer_steps_this_step':0,'IRRA_text_samples':0}
  self.h.requires_grad_(True);example=self.native.get('formal_training.data').example_from(e);query=self.catalog.get_train(row)
  def bind(k):return self.reward.bind(query,self.runtime,example,k)
  log,_=train_step(self.native.get('phase4_training'),self.native.get('policy'),self.native.get('constraints'),
   self.h,self.ha,example,self.sampler,bind,self.cursor['h_visits'],row,self.c['cooperative'])
  self.cursor['h_updates']+=1;log['bank_sha256']=bank.adversarial_sha256
  return log

@contextmanager
def loaded(c,bundle,grant):
 from tbps.models.irra import IRRAAdapter
 from tbps.models.irra_source import IRRASource
 torch.cuda.manual_seed_all(42);cuda=clone_tree(torch.cuda.get_rng_state_all())
 for side in ('G','H'):bundle.state[side+'_RNG']['cuda']=clone_tree(cuda)
 source=IRRASource.from_dict(read_json(c['sources']['dataset_config']['path'])['irra'])
 adapter=IRRAAdapter.from_source(source,device='cuda:0',allow_model_load=True)
 with native_session(c,reuse=bundle.modules) as session:
  backend=CooperativeBackend(c,bundle,grant,adapter);backend.native_session=session
  try:yield backend
  finally:backend.close()
