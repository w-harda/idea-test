"""CPU审计白名单与真实v1 Artifact的隔离副本回归，不运行GPU或重试旧Attempt。"""
from pathlib import Path
from dataclasses import asdict
import json,io,hashlib,types
import pytest
import torch
from PIL import Image
from torchvision import transforms
from tbps.transfer import gv2_core as core,gv2_execute as execute
from tbps.artifacts import ArtifactStore,read_json,json_bytes,FileRef,sha_file
from tbps.engines.evaluation import load_result,publish_result
OLD=Path('/home/lzf/ldx/outputs/tbps-framework-gv2-cross-dataset-20261102')
OLD_CONTROL=Path('/home/lzf/ldx/tmp/gv2_cross_dataset_20261102')
OLD_CONFIG=Path('/home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-20261102/configs/framework/gv2_cross_dataset.server.json')

def state_snapshot():
 result={}
 for root in (OLD,OLD_CONTROL):
  for p in sorted(root.rglob('*')):
   if p.is_file():result[str(p)]=(core.signature(p),sha_file(p))
 return result

@pytest.mark.parametrize('tensor_input',[False,True])
def test_real_resize_and_to_tensor_within_cpu_guard(tensor_input):
 im=Image.new('RGB',(63,141),(31,90,153));resize=transforms.Resize((384,128))
 x=transforms.ToTensor()(im) if tensor_input else im
 expected=resize(x)
 if not tensor_input:expected=transforms.ToTensor()(expected)
 with core.no_backward(cpu=True):
  actual=resize(x)
  if not tensor_input:actual=transforms.ToTensor()(actual)
 assert actual.dtype==torch.float32 and torch.equal(actual,expected)
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('kind',['Linear','Conv2d','Sequential','G'])
def test_real_neural_modules_still_blocked(kind):
 if kind=='G':
  c=json.loads((core.ROOT/'configs/framework/gv2_cross_dataset.server.json').read_text())
  with core.no_backward(cpu=True),core.frozen_generator(c,'cpu') as (g,_,_):
   with pytest.raises(RuntimeError,match='forbids'):g(torch.zeros(1,3,384,128))
  return
 module={'Linear':lambda:torch.nn.Linear(3,3),'Conv2d':lambda:torch.nn.Conv2d(3,3,1),'Sequential':lambda:torch.nn.Sequential(torch.nn.Identity())}[kind]()
 with core.no_backward(cpu=True),pytest.raises(RuntimeError,match='forbids'):module(torch.zeros(1,3,8,8))

@pytest.mark.parametrize('kind',['subclass','parameter','buffer','child','hook','modified_forward','grad_tensor'])
def test_resize_allowlist_is_narrow(kind):
 class ModifiedResize(transforms.Resize):pass
 m=ModifiedResize((16,16)) if kind=='subclass' else transforms.Resize((16,16))
 im=Image.new('RGB',(23,31))
 if kind=='parameter':m.register_parameter('unexpected',torch.nn.Parameter(torch.ones(1)))
 if kind=='buffer':m.register_buffer('unexpected',torch.ones(1))
 if kind=='child':m.add_module('unexpected',torch.nn.Identity())
 if kind=='hook':m.register_forward_hook(lambda *args:None)
 if kind=='modified_forward':m.forward=types.MethodType(lambda self,x:x,m)
 if kind=='grad_tensor':im=torch.ones(3,23,31,requires_grad=True)
 with core.no_backward(cpu=True),pytest.raises(RuntimeError,match='forbids'):m(im)

@pytest.mark.parametrize('operation',['cuda_init','cuda_tensor','tensor_backward','autograd_backward','autograd_grad','functional_jvp','func_jvp','dual'])
def test_forbidden_entries(operation):
 x=torch.tensor([1.],requires_grad=True);y=x.square();t=torch.ones_like(x)
 callbacks={'cuda_init':torch.cuda.init,'cuda_tensor':lambda:torch.empty(1,device='cuda:0'),'tensor_backward':lambda:y.backward(),
 'autograd_backward':lambda:torch.autograd.backward(y),'autograd_grad':lambda:torch.autograd.grad(y,x),
 'functional_jvp':lambda:torch.autograd.functional.jvp(lambda z:z*z,x,t),'func_jvp':lambda:torch.func.jvp(lambda z:z*z,(x,),(t,)),
 'dual':lambda:torch.autograd.forward_ad.make_dual(x,t)}
 with core.no_backward(cpu=True),pytest.raises(RuntimeError,match='forbids'):callbacks[operation]()
 assert x.grad is None and not torch.cuda.is_initialized()

@pytest.mark.parametrize('optimizer',['Adam','SGD','AdamW','LBFGS'])
def test_actual_optimizer_override_step_blocked_and_restored(optimizer):
 p=torch.nn.Parameter(torch.ones(1));opt=getattr(torch.optim,optimizer)([p],lr=.01);p.grad=torch.ones_like(p)
 before=p.detach().clone();old=type(opt).step
 with core.no_backward(cpu=True),pytest.raises(RuntimeError,match='forbids'):opt.step(lambda:p.square().sum())
 assert torch.equal(p.detach(),before) and type(opt).step is old
 if optimizer!='LBFGS':opt.step();assert not torch.equal(p.detach(),before)

@pytest.mark.parametrize('fault',['user_exception','guard_exception'])
def test_guards_restore_on_exception(fault):
 pairs=[(torch.nn.Module,'_call_impl'),(torch.cuda,'_lazy_init'),(torch.Tensor,'backward'),(torch.autograd,'grad'),
 (torch.autograd,'backward'),(torch.optim.Adam,'step'),(torch.optim.SGD,'step'),(torch.func,'jvp'),(torch.autograd.functional,'jvp'),(torch.autograd.forward_ad,'make_dual')]
 originals=[getattr(obj,name) for obj,name in pairs]
 with pytest.raises(RuntimeError):
  with core.no_backward(cpu=True):
   if fault=='user_exception':raise RuntimeError('deliberate CPU test exception')
   torch.nn.Linear(1,1)(torch.ones(1,1))
 assert all(getattr(obj,name)is old for (obj,name),old in zip(pairs,originals))
 layer=torch.nn.Linear(1,1);layer(torch.ones(1,1)).sum().backward();assert layer.weight.grad is not None
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('corrupt',[False,True])
def test_real_rstp_artifact_copy_complete_audit_path(tmp_path,corrupt):
 # 旧failed输出仅作为测试输入；不调用旧audit、不写旧回执，不把本测试当新真实Smoke。
 before=state_snapshot();c=json.loads(OLD_CONFIG.read_text());c['path']=str(OLD_CONFIG);c['sha256']=sha_file(OLD_CONFIG)
 base=tmp_path/'tbps-framework-audit-fixture';c['output_base']=str(base);c['control']=str(tmp_path/'test-control')
 c['protected_roots']=[str(OLD),str(OLD_CONTROL)]
 s=core.store_for(c,'rstp','smoke');old=ArtifactStore(OLD/'smoke/rstp',new_base=OLD)
 for directory in sorted(old.root.iterdir()):
  if not directory.is_dir() or directory.name in ('failure','result','cpu_audit') or directory.name.startswith('ranking-'):continue
  if not (directory/'manifest.json').exists():continue
  m=old.verify(directory.name);files={name:(directory/name).read_bytes() for name in m['files_sha256']}
  if corrupt and directory.name=='image-000000':
   arrays=torch.load(io.BytesIO(files['pixels.pt']),map_location='cpu',weights_only=True)
   arrays={k:torch.roll(v,1,-1) for k,v in arrays.items()}
   md=json.loads(files['metadata.json']);md['pixels']=core.image_checks(arrays['original'],arrays['adversarial'],arrays['effective_delta'],(384,128))
   files['pixels.pt']=core.tensor_bytes(arrays);files['metadata.json']=json_bytes(md)
  s.publish(directory.name,files,m['provenance'],sources=tuple(FileRef(**v) for v in m['sources']))
 protocol,report,scores=load_result(old,'result',expected_protocol_sha256=core.digest(read_json(old.root/'result/protocol.json')))
 # JSON读回将tuple变为list；恢复框架原发布接口要求的不可变绑定类型，数值/协议digest不变。
 for bound in protocol['inputs'].values():
  for field in ('ordered_sha256','ordered_ids'):bound['inputs'][field]=tuple(bound['inputs'][field])
 publish_result(s,'result',protocol,report,scores,sources=(core.ref(old.root/'result/manifest.json'),))
 pre=read_json(OLD/'preflight/report.json');rootstore=ArtifactStore(base,new_base=base)
 rootstore.publish('preflight',{'report.json':json_bytes(pre),'history.json':json_bytes(core.history_snapshot(c))},{'kind':'isolated_test_fixture_only'},sources=(core.ref(OLD/'preflight/manifest.json'),))
 if corrupt:
  with pytest.raises(ValueError,match='original FP32 RGB mismatch'):execute.audit(c,'rstp','smoke')
 else:
  execute.audit(c,'rstp','smoke');a=read_json(s.root/'cpu_audit/report.json')
  assert a['status']=='PASS' and a['Gallery']==a['Query']==10 and a['metrics_recomputed']
  assert a['groups']==read_json(old.root/'completed/summary.json')['groups']
  assert a['neural_forwards']==a['backward']==a['optimizer']==a['JVP']==0
  assert (s.root/'ranking-Clean/order.pt').is_file() and (s.root/'ranking-Image-only/order.pt').is_file()
 assert state_snapshot()==before
 assert not torch.cuda.is_initialized()
