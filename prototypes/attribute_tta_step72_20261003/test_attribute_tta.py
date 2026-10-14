import copy,json,sys
from pathlib import Path
import pytest,torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,'/home/lzf/ldx/projects/idea-TBPS-test1/src')
import attribute_tta as mod
from attribute_tta import *

@pytest.fixture
def real_inputs():
 rows=json.loads((P51/'outputs/diagnostic/results.json').read_text());payloads=json.loads((P61/'outputs/diagnostic/selection_payloads.json').read_text());bases=json.loads(BASIS.read_text())
 return {r['row_id']:r for r in rows},{r['caption_id']:r for r in payloads if r['gamma']==3},bases

def test_all_saved_metadata_scores_and_sequences(real_inputs):
 rows,ps,bs=real_inputs
 for cid,p in ps.items():
  before=copy.deepcopy(p);plan=prepared_plan(p,rows[cid],bs)
  assert [a.instance_id for a in plan]==[x['instance_id'] for x in p['selected_attributes']]
  assert [a.selection_rank for a in plan]==list(range(1,len(plan)+1))
  assert p==before
  for a in plan:assert abs(float(attribute_delta(a).norm())-1)<3e-6 and not a.Q.requires_grad and not a.w.requires_grad

@pytest.mark.parametrize('failure',['gamma','source','unresolved','rank0','score','basis'])
def test_invalid_inputs_rejected(real_inputs,failure):
 rows,ps,bs=copy.deepcopy(real_inputs);cid='cuhk:35:1';p=ps[cid];record=rows[cid];item=p['selected_attributes'][0]
 if failure=='gamma':p['gamma']=2
 elif failure=='source':record['caption']='changed source'
 elif failure=='unresolved':item['unresolved']=True
 elif failure=='rank0':item['semantic_representation']['basis_rank']=0
 elif failure=='score':item['S_raw']+=1
 elif failure=='basis':bs[item['semantic_axis']]['basis_id']='wrong basis'
 with pytest.raises(ValueError):prepared_plan(p,record,bs)

def test_same_axis_different_region(real_inputs):
 rows,ps,bs=real_inputs;plan=prepared_plan(ps['cuhk:0:1'],rows['cuhk:0:1'],bs);colors=[a for a in plan if a.metadata['semantic_axis']=='color']
 assert {a.metadata['region'] for a in colors}=={'upper','lower'} and len({a.instance_id for a in colors})==2
 assert not torch.equal(attribute_delta(colors[0]),attribute_delta(colors[1]))

def test_no_caption_raw_encoding_in_bridge():
 class Source(nn.Module):
  def inference_text(self,*args):raise AssertionError('text encoder forbidden')
 bridge=FixedGuidanceBridge(Source(),torch.ones(1,512),['attribute_unit:123'])
 assert torch.equal(bridge.inference_text(mod._tokenize(['attribute_unit:123']))['text_feat'],torch.ones(1,512))
 for bad in ('a full caption','black'):
  with pytest.raises(ValueError):bridge.inference_text(mod._tokenize([bad]))

class Block(nn.Module):
 def forward(self,x):return x
class ToyVictim(nn.Module):
 def __init__(self):
  super().__init__();self.fixed=nn.Parameter(torch.randn(77,1,512),requires_grad=False)
  self.model=nn.Module();self.model.base_model=nn.Module();self.model.base_model.transformer=nn.Module();self.model.base_model.transformer.resblocks=nn.ModuleList([Block() for _ in range(8)])
 def encode_texts(self,texts):
  H=self.model.base_model.transformer.resblocks[7](self.fixed)
  return F.normalize(H.tanh().sum(0),dim=-1)

def test_transport_original_graph_jvp(real_inputs):
 rows,ps,bs=real_inputs;a=prepared_plan(ps['cuhk:35:1'],rows['cuhk:35:1'],bs)[0];victim=ToyVictim();t=TextTransport(victim,'context');v,report=t.read(a)
 assert report['zero_exact'] and report['central_fd_relative_error']<.005
 assert abs(report['tangent_radial_dot'])<1e-7
 assert all(not p.requires_grad and p.grad is None for p in victim.parameters())
 # JVP映射前的512D不可直接当最终retrieval方向。
 assert not torch.allclose(v.flatten(),torch.tensor(a.metadata['semantic_representation']['projected_context'],dtype=v.dtype))

@pytest.mark.parametrize('k',[0,1,2,3,4])
def test_fixed_image_schedule_and_cumulative_projection(real_inputs,monkeypatch,k):
 rows,ps,bs=real_inputs;plan=prepared_plan(ps['cuhk:3828:0'],rows['cuhk:3828:0'],bs)[:k]
 class Attack:
  def __init__(self,*args,**kw):pass
  def img_attack(self,bridge,texts,current,original,ids,steps,momentum,device,scales):
   bridge.inference_text(mod._tokenize(texts));return current+.02,momentum+1
 monkeypatch.setattr(mod,'load_official_attack',lambda root:Attack)
 image=torch.ones(1,3,2,2)*.5;engine=AttributeTTA(nn.Identity(),image,steps=1);vectors={a.instance_id:torch.ones(1,512) for a in plan};y=engine.run(plan,vectors)
 assert [x['instance_id'] for x in engine.events]==[a.instance_id for a in plan]
 assert float((y-image).abs().max())<=IMAGE_EPSILON+1e-6
 assert len(engine.events)==k


def test_different_attributes_real_math_gradient():
 class Source:
  def encode_images(self,x):return F.normalize(x.flatten().reshape(1,512),dim=-1)
 x=torch.linspace(.1,.9,512).reshape(1,1,16,32);e1=torch.zeros(1,512);e2=e1.clone();e1[0,1]=1;e2[0,2]=1
 g1=image_gradient(Source(),x,e1);g2=image_gradient(Source(),x,e2)
 assert not torch.equal(g1,g2)
 assert torch.equal(g1,image_gradient(Source(),x,e1))
