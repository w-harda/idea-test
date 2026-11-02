"""CUHK G-v2 E60跨数据集推理的来源、身份与只读保护；不实现训练算法。"""
from pathlib import Path
from dataclasses import asdict,replace
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch
import io,json,os,subprocess,time,hashlib
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,sha_file,digest,json_bytes,read_json
from tbps.baselines.cpgc_sep.dataset import read_test,require,SPECS
from tbps.baselines.cpgc_sep.victims import VictimSource,native_pixels
from tbps.baselines.cpgc_sep.runtime import resources,ref,tensor_bytes
from tbps.attacks.legacy_runtime import legacy_imports,G_REL
from tbps.ablations.gv2_sweep import generator_state,ARCH
from tbps.ablations.gv2_epoch_state import tree_hash
from tbps.materialization.store import image_checks
ROOT=Path(__file__).resolve().parents[3]
DATASETS=('rstp','icfg')
KEYS=('R@1','R@5','R@10','mAP')
REFERENCE='6021f3bfdec181785d1baccd7ce4f80a1428b069'

def load_config(path):
 c=read_json(path,'gv2-cross-dataset-1');c['path']=str(Path(path).resolve());c['sha256']=sha_file(path)
 require(c['datasets_order']==list(DATASETS) and set(c['datasets'])==set(DATASETS),'two fixed targets')
 require(c['epoch']==60 and c['architecture']==ARCH and c['image_size']==[384,128] and c['epsilon']==8/255,'original G numerical contract')
 require(c['image_batch']==1 and c['text_batch']==8 and c['query_chunk']==128 and c['seed']==42,'fixed batching/seed; never adapt on failure')
 require(c['gpu_policy']=='shared_observe_only' and set(c['resource_gate'])=={'RAM_bytes','disk_bytes','inodes'},'no GPU VRAM floor permitted')
 require(set(c['modes'])=={'Clean','Image-only'} and len(c['modes'])==2,'image-only protocol')
 base=Path(c['output_base']).resolve()
 require(base.parent==Path('/home/lzf/ldx/outputs') and base.name=='tbps-framework-gv2-cross-dataset-v2-20261102','isolated output namespace')
 for p,h in c['producers'].items():checked(FileRef(p,h))
 for r in c['sources'].values():checked(FileRef(**r))
 for p,h in c['legacy_sources'].items():checked(FileRef(p,h))
 require(c['checkpoint']==c['sources']['checkpoint'],'single exact generator')
 targets=c['victims']['irra']['datasets']
 require(all(targets[ds]['weight_dataset']==ds and targets[ds]['checkpoint']['sha256']!=c['CUHK_surrogate_sha256'] for ds in DATASETS),'target must use its own official IRRA')
 require(targets['rstp']['checkpoint']['sha256']!=targets['icfg']['checkpoint']['sha256'],'distinct target IRRA weights')
 return c

def legacy(c):return SimpleNamespace(root=ROOT,config={'runtime_sources':c['legacy_sources']})

def get_manifest(c,ds,scope):
 require(ds in DATASETS and scope in ('smoke','full'),'explicit dataset/scope')
 m=read_test(c['datasets'][ds])
 require(digest(asdict(m))==c['manifest_sha256'][ds],'full official manifest identity')
 if scope=='full':return m
 gs=m.gallery[:10];ids={x.image_id for x in gs}
 # 固定官方顺序前10图库，取其中前10原始Caption；不按模型效果选样。
 qs=tuple(q for q in m.queries if q.paired_image_id in ids)[:10]
 require(len(gs)==len(qs)==10 and all(q.pid in {g.pid for g in gs} for q in qs),'Smoke positive coverage')
 return replace(m,gallery=gs,queries=qs)

def store_for(c,ds,scope):return ArtifactStore(Path(c['output_base'])/scope/ds,new_base=c['output_base'])

def signature(p):
 s=Path(p).stat();return [s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def history_snapshot(c):
 rows={}
 for raw in c['protected_roots']:
  root=Path(raw).resolve();require(root.exists() and not root.is_relative_to(Path(c['output_base'])),'protected root')
  for p in sorted(root.rglob('*')):
   if p.is_file():rows[str(p)]=signature(p)
 return rows

def verify_history(c):
 p=Path(c['output_base'])/'preflight/history.json';old=read_json(p)
 require(history_snapshot(c)==old,'historical CUHK Full/checkpoints changed')
 return len(old)

@contextmanager
def no_backward(cpu=False):
 # 首次导入会初始化Dynamo规则表；先在原Torch函数身份下导入。
 from torchvision.transforms import Resize
 from PIL.Image import Image as PILImage
 from contextlib import ExitStack
 def forbid(*a,**k):
  raise RuntimeError('inference/audit forbids neural forward, CUDA, backward, optimizer and JVP')
 original_call=torch.nn.Module._call_impl
 resize_forward=Resize.forward
 def cpu_image(value):
  return isinstance(value,PILImage) or (isinstance(value,torch.Tensor) and value.device.type=='cpu' and not value.requires_grad)
 def cpu_module_call(module,*args,**kwargs):
  # 仅放行实际审计所需的精确Resize类型；ToTensor不是nn.Module，不触发此门禁。
  allowed=(type(module)is Resize and len(args)==1 and not kwargs and cpu_image(args[0])
    and not module._parameters and not module._buffers and not module._modules
    and not module._forward_pre_hooks and not module._forward_hooks
    and getattr(module.forward,'__func__',None)is resize_forward)
  if not allowed:forbid()
  value=original_call(module,*args,**kwargs)
  if not cpu_image(value):forbid()
  return value
 def optimizer_classes():
  found=set();pending=[torch.optim.Optimizer]
  while pending:
   cls=pending.pop()
   if cls not in found:found.add(cls);pending.extend(cls.__subclasses__())
  return found
 with ExitStack() as stack:
  for obj,key in ((torch.Tensor,'backward'),(torch.autograd,'backward'),(torch.autograd,'grad'),
                  (torch.autograd.functional,'jvp'),(torch.func,'jvp'),(torch.autograd.forward_ad,'make_dual')):
   stack.enter_context(patch.object(obj,key,forbid))
  # Adam/SGD等覆写step；只补丁基类无法拦住这些真实Optimizer。
  for cls in optimizer_classes():
   if 'step' in cls.__dict__:stack.enter_context(patch.object(cls,'step',forbid))
  if cpu:
   stack.enter_context(patch.object(torch.cuda,'_lazy_init',forbid))
   stack.enter_context(patch.object(torch.nn.Module,'_call_impl',cpu_module_call))
  yield

def read_weights(c):
 checked(FileRef(**c['checkpoint']))
 data=torch.load(c['checkpoint']['path'],map_location='cpu',weights_only=False)
 weights=generator_state(data,60,read_json(c['sources']['original_contract']['path']),c['parent_epoch10'])
 require(all(torch.isfinite(v).all() for v in weights.values()),'nonfinite G weight')
 return weights

@contextmanager
def frozen_generator(c,device):
 with legacy_imports(legacy(c),allow_import=True) as old:
  weights=read_weights(c)
  model=old('image_only_generator').ImageOnlyGenerator(**ARCH)
  model.load_state_dict(weights,strict=True);model=model.to(device).eval().requires_grad_(False)
  require(tree_hash(model.state_dict())==tree_hash(weights),'strict restored G values');del weights
  yield model,old('inference_once').generate_once,old('irra_image_adapter').pixels_from_pil

def token_hashes(source,manifest):
 from tbps.models.irra_io import official_tokenizer
 tok=official_tokenizer(source);h=hashlib.sha256()
 for start in range(0,len(manifest.queries),8):
  tokens,_=tok.batch([q.caption for q in manifest.queries[start:start+8]])
  h.update(tokens.numpy().tobytes())
 return {'all_raw_token_ids_sha256':h.hexdigest(),'tokenizer_fingerprint':tok.fingerprint,'context_length':77}

def preflight(c):
 require(not torch.cuda.is_initialized(),'CPU Preflight only');torch.set_num_threads(2)
 base=Path(c['output_base']);s=ArtifactStore(base,new_base=base)
 require(not base.exists(),'preflight already consumed; preserve Attempt')
 with no_backward(cpu=True):
  evidence=read_json(c['sources']['training_evidence']['path'])
  require(evidence['points']['60']==c['checkpoint'] and evidence['completed']['status']=='PASS' and evidence['audit']['status']=='PASS','completed CUHK training provenance')
  report={'status':'PASS','CUDA':False,'epoch':60,'architecture':ARCH,'G_checkpoint':c['checkpoint'],'targets':{},'resources':resources(c),'reference_commit':REFERENCE}
  with frozen_generator(c,'cpu') as (g,fn,pixels):
   report['G_state_sha256']=tree_hash(g.state_dict());report['G_parameters']=sum(p.numel() for p in g.parameters())
   require(report['G_parameters']==1377411,'frozen G architecture parameter count')
  for ds in DATASETS:
   source=VictimSource(c,'irra',ds);source.verify()
   require(source.checkpoint.sha256!=c['CUHK_surrogate_sha256'],'target must not use CUHK IRRA')
   state=torch.load(source.checkpoint.path,map_location='cpu',weights_only=False)
   require(tuple(state['model']['classifier.weight'].shape)==(source.data['num_classes'],512),'dataset specific IRRA classifier');del state
   full=get_manifest(c,ds,'full');smoke=get_manifest(c,ds,'smoke')
   report['targets'][ds]={'gallery':len(full.gallery),'query':len(full.queries),'PIDs':len({g.pid for g in full.gallery}),'manifest_sha256':digest(asdict(full)),
    'checkpoint':asdict(source.checkpoint),'model_space':asdict(source.space),'tokens':token_hashes(source,full),'smoke_gallery':[g.image_id for g in smoke.gallery],'smoke_queries':[q.query_id for q in smoke.queries]}
  require(len({x['checkpoint']['sha256'] for x in report['targets'].values()})==2,'distinct target weights')
  history=history_snapshot(c);report['protected_files']=len(history)
  s.publish('preflight',{'report.json':json_bytes(report),'history.json':json_bytes(history)}, {'kind':'CPU_PREFLIGHT'},sources=(ref(c['path']),*tuple(FileRef(**r) for r in c['sources'].values())))
 require(not torch.cuda.is_initialized(),'CPU initialized CUDA')
 print('CROSS_DATASET_CPU_PREFLIGHT_PASS',json.dumps({k:v for k,v in report.items() if k!='targets'}),flush=True)
 return report

def ensure_preflight(c):
 s=ArtifactStore(c['output_base'],new_base=c['output_base']);s.verify('preflight')
 r=read_json(s.root/'preflight/report.json');require(r['status']=='PASS' and r['G_checkpoint']==c['checkpoint'],'Preflight identity')
 return r

def progress(c,ds,scope,stage,done,total):
 p=Path(c['control'])/f'progress_{scope}_{ds}.json';p.parent.mkdir(parents=True,exist_ok=True)
 row={'time':time.time(),'dataset':ds,'scope':scope,'stage':stage,'done':done,'total':total}
 t=p.with_suffix('.tmp');t.write_bytes(json_bytes(row));t.replace(p)
 if done%32==0 or done==total:print('CROSS_DATASET_PROGRESS',json.dumps(row),flush=True)

def load_pixels_unit(store,index):
 key=f'image-{index:06d}';store.verify(key)
 arrays=torch.load(store.root/key/'pixels.pt',map_location='cpu',weights_only=True)
 md=read_json(store.root/key/'metadata.json')
 require(image_checks(**{'clean':arrays['original'],'adv':arrays['adversarial'],'delta':arrays['effective_delta'],'size':(384,128)})==md['pixels'],'persisted pixel proof')
 return arrays,md
