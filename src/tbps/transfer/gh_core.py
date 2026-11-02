"""冻结CUHK H7文本迁移与已审计G60图像依赖；不实现攻击算法。"""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from contextlib import contextmanager
import io,json,os,time,hashlib
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,digest,json_bytes,read_json,sha_file
from tbps.transfer import gv2_core as image
from tbps.baselines.cpgc_sep.victims import VictimSource,snapshot
from tbps.baselines.cpgc_sep.runtime import ref,resources,tensor_bytes
from tbps.baselines.cpgc_sep.dataset import require,SPECS
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.engines.provenance import BoundRepresentation,AttackOrigin,payload_digest
from tbps.models.base import RepresentationBatch,InputBinding
from tbps.attacks.gv2 import BUDGET as G_BUDGET
from tbps.attacks.hv1 import BUDGET as H_BUDGET
ROOT=Path(__file__).resolve().parents[3]
DATASETS=('rstp','icfg');MODES=('Text-only','Joint');KEYS=('R@1','R@5','R@10','mAP')
H_SHA='487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df'
G_SHA='9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049'

def config(path):
 c=read_json(path,'gh-cross-dataset-1');c['path']=str(Path(path).resolve());c['sha256']=sha_file(path)
 require(c['H_checkpoint']['sha256']==H_SHA and c['G_checkpoint']['sha256']==G_SHA,'fixed CUHK G60/H7')
 require(c['datasets']==list(DATASETS) and c['modes']==list(MODES),'fixed two targets/two added modes')
 require(c['H_device']=='cpu' and c['image_batch']==1 and c['text_batch']==8 and c['query_chunk']==128,'original H CPU/native target batching')
 require(c['gpu_policy']=='shared_observe_only' and set(c['resource_gate'])=={'RAM_bytes','disk_bytes','inodes'},'no VRAM floor')
 require(Path(c['output_base']).resolve()==Path('/home/lzf/ldx/outputs/tbps-framework-gh-cross-dataset-20261102'),'isolated GH output namespace')
 for p,h in c['producers'].items():checked(FileRef(p,h))
 for p,h in c['legacy_sources'].items():checked(FileRef(p,h))
 for v in c['sources'].values():checked(FileRef(**v))
 return c

def image_config(c):
 checked(FileRef(**c['image_config']));v=image.load_config(c['image_config']['path'])
 require(v['checkpoint']==c['G_checkpoint'],'same G60 source')
 return v

def manifest(c,ds,scope):return image.get_manifest(image_config(c),ds,scope)
def scope_store(c,ds,scope):return ArtifactStore(Path(c['output_base'])/scope/ds,new_base=c['output_base'])
def result_store(c,ds,scope,mode):
 require(mode in MODES,'one added mode')
 return ArtifactStore(Path(c['output_base'])/scope/ds/'evaluation'/mode,new_base=c['output_base'])
def legacy(c):return SimpleNamespace(root=Path(c['H_source_root']),config={'runtime_sources':c['legacy_sources']})

def H_payload(c):
 checked(FileRef(**c['H_checkpoint']));p=torch.load(c['H_checkpoint']['path'],map_location='cpu',weights_only=False)
 protocol=read_json(c['sources']['H_training_protocol']['path'])
 require(p['schema']=='hv1-formal-periodic-checkpoint-1' and p['cursor']==14336 and digest(p['protocol'])==digest(protocol)
  and p['protocol_sha256']==digest(protocol) and protocol['epoch_size']==2048,'original H7 training protocol/cursor')
 require(all(v.startswith('cuhk:') for v in protocol['query_order']),'H trained on CUHK only')
 return p,protocol['config']['architecture']

@contextmanager
def frozen_H(c,*,validator_only=False):
 # 仅复用原TextAttacker/Canonical/Unicode/BPE，不构造Backend或OfficialTextEncoder。
 with legacy_imports(legacy(c),allow_import=True) as old:
  backend=old('phase5_joint.backend');probe=old('tokenizer_probe').official_probe();frontend=old('text_frontend').CaptionFrontend()
  if validator_only:
   yield None,lambda caption,action:backend.validate_action(caption,action,probe,frontend)
   return
  p,arch=H_payload(c)
  policy=old('policy').CharacterPolicy(probe,**arch)
  policy.load_state_dict(p['policy'],strict=True);policy.eval().requires_grad_(False)
  attacker=backend.TextAttacker(policy,frontend,probe)
  require(attacker.before==old('formal_training.validation').state_sha_from(p['policy']),'H policy-only strict restoration')
  del p
  yield attacker,lambda caption,action:backend.validate_action(caption,action,probe,frontend)
  attacker.finish()

def dependency(c,ds,scope,*,required=True):
 """只有对应v2独立CPU审计完整发布后才允许读取图库/特征。"""
 ic=image_config(c);s=image.store_for(ic,ds,scope)
 if not (s.root/'cpu_audit/completed.json').is_file():
  require(not (s.root/'failure').exists(),'image source failed; no GH continuation')
  global_status=Path(c['image_control'])/'status.json'
  if global_status.exists():require(read_json(global_status).get('phase')!='FAIL','image Supervisor failed')
  if required:raise ValueError('WAITING_FOR_AUDITED_IMAGE_SOURCE')
  return None
 s.verify('cpu_audit');a=read_json(s.root/'cpu_audit/report.json')
 m=manifest(c,ds,scope);expected=(len(m.gallery),len(m.queries))
 require(a['status']=='PASS' and a['dataset']==ds and a['scope']==scope and (a['Gallery'],a['Query'])==expected
  and a['G_checkpoint']==c['G_checkpoint'] and a['IRRA_checkpoint']==asdict(VictimSource(ic,'irra',ds).checkpoint),'audited image source identity')
 s.verify('completed');s.verify('native_protocol');s.verify('result')
 native=read_json(s.root/'native_protocol/protocol.json')
 require(native['manifest_sha256']==digest(asdict(m)) and native['model_space']==asdict(VictimSource(ic,'irra',ds).space),'same native feature space')
 return s,m,a,native

@contextmanager
def H_inference_guard():
 from unittest.mock import patch
 def forbidden(*a,**k):raise RuntimeError("H CPU generation forbids CUDA initialization")
 with image.no_backward(),patch.object(torch.cuda,"_lazy_init",forbidden):yield

def static_snapshot(c):
 out={}
 for p in c['protected_files']:
  out[p]={'sha256':sha_file(p),'stat':image.signature(p)}
 return out

def check_protection(c):
 before=read_json(Path(c['output_base'])/'preflight/protected.json')
 require(static_snapshot(c)==before,'frozen failed/history/checkpoint sources changed')
 return len(before)

def progress(c,ds,scope,stage,done,total,mode=None):
 p=Path(c['control'])/f'progress_{scope}_{ds}.json';p.parent.mkdir(parents=True,exist_ok=True)
 d={'time':time.time(),'dataset':ds,'scope':scope,'stage':stage,'done':done,'total':total,'mode':mode}
 t=p.with_suffix('.tmp');t.write_bytes(json_bytes(d));t.replace(p)
 if done%32==0 or done==total:print('GH_TRANSFER_PROGRESS',json.dumps(d),flush=True)

def actions(c,ds,scope,*,validate=False):
 s=scope_store(c,ds,scope);s.verify('text_completed');m=manifest(c,ds,scope);rows=[]
 for p in sorted(s.root.glob('text-*/rows.json')):
  s.verify(p.parent.name);rows.extend(json.loads(p.read_text()))
 require(len(rows)==len(m.queries),'all target captions retained including NOOP')
 for q,row in zip(m.queries,rows):require(row['query_id']==q.query_id and row['action']['original']==q.caption,'original target Caption/order')
 if validate:
  with image.no_backward(cpu=True),frozen_H(c,validator_only=True) as (_,validator):
   for q,row in zip(m.queries,rows):validator(q.caption,row['action'])
 return s,m,rows

def original_features(c,ds,scope,space):
 s,m,a,native=dependency(c,ds,scope)
 require(asdict(space)==native['model_space'],'target checkpoint/tokenizer/precision must match reused features')
 from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
 verified=VerifiedArtifactStore(s.root,new_base=c['image_output']);parts={'clean_gallery':[],'adv_gallery':[],'clean_text':[]};refs={k:[] for k in parts};adv_hash=[]
 for p in sorted(s.root.glob('gallery-feature-*/features.pt')):
  verified.verify(p.parent.name);v=torch.load(p,map_location='cpu',weights_only=True)
  for key in ('clean_gallery','adv_gallery'):parts[key].append(v[key].detach().clone());refs[key].append(ref(p.parent/'manifest.json'))
 for p in sorted(s.root.glob('text-feature-*/features.pt')):
  verified.verify(p.parent.name);parts['clean_text'].append(torch.load(p,map_location='cpu',weights_only=True).detach().clone());refs['clean_text'].append(ref(p.parent/'manifest.json'))
 p=read_json(s.root/'result/protocol.json');adv_hash=list(p['inputs']['adv_gallery']['inputs']['ordered_sha256'])
 require(len(adv_hash)==len(m.gallery),'audited image content binding')
 arrays={k:torch.cat(v) for k,v in parts.items()}
 require(arrays['clean_gallery'].shape==arrays['adv_gallery'].shape==(len(m.gallery),512) and arrays['clean_text'].shape==(len(m.queries),512),'complete audited feature coverage')
 return s,m,arrays,refs,tuple(adv_hash)
