"""冻结H目标Caption迁移、已发布缓存门禁和后台串行链的CPU回归。"""
from pathlib import Path
from dataclasses import asdict
import copy,importlib.util,json
from types import SimpleNamespace
import pytest,torch
from tbps.transfer import gh_core as core,gh_execute as execute,gh_results as results
from tbps.artifacts import ArtifactStore,json_bytes,digest
from tbps.models.irra import fp32_runtime
CONFIG=core.ROOT/'configs/framework/gh_cross_dataset.server.json'

@pytest.fixture(scope='module')
def c():return core.config(CONFIG)

def test_original_h7_architecture_and_training_only_cuhk(c):
 with core.image.no_backward(cpu=True):
  p,a=core.H_payload(c)
  assert p['cursor']==14336 and p['updates']==14259 and a==dict(width=64,layers=2,heads=4,max_bytes=1024)
  assert all(x.startswith('cuhk:') for x in p['protocol']['query_order'])
  with core.frozen_H(c) as (h,_):
   assert sum(p.numel() for p in h.policy.parameters())==154594
   assert all(p.device.type=='cpu' and not p.requires_grad and p.grad is None for p in h.policy.parameters())
 assert not torch.cuda.is_initialized()

def test_real_h_wrapper_matches_original_inference_and_json_audit(c):
 caption='A man wearing a black shirt and blue pants.'
 with core.H_inference_guard(),fp32_runtime(),core.frozen_H(c) as (h,validate):
  before=h.before;calls=[]
  hook=h.policy.register_forward_pre_hook(lambda m,args:calls.append(set(vars(args[0]))))
  action=h.attack(caption);validate(caption,json.loads(json_bytes(action)))
  assert calls==[{'caption','attributes'}] and action['original']==caption and action['H_forward']==1
  assert h.finish()['H_state_sha256_after']==before
  hook.remove()
 with core.legacy_imports(core.legacy(c),allow_import=True) as old:
  p,a=core.H_payload(c);probe=old('tokenizer_probe').official_probe();policy=old('policy').CharacterPolicy(probe,**a);policy.load_state_dict(p['policy']);native=old('phase5_joint.backend').TextAttacker(policy,old('text_frontend').CaptionFrontend(),probe)
  with core.H_inference_guard(),fp32_runtime():expected=native.attack(caption)
  assert json_bytes(action)==json_bytes(expected)
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('caption',['the person is standing',' '.join(['black shirt']*100)])
def test_original_legal_noop_and_truncation_retained(c,caption):
 with core.H_inference_guard(),fp32_runtime(),core.frozen_H(c) as (h,validate):
  action=h.attack(caption);validate(caption,action)
  assert action['original']==caption
  if not action['eligible']:assert action['adversarial']==caption and action['H_forward']==0 and action['edits']==[]
  assert h.finish()['H_unchanged']

@pytest.mark.parametrize('value',[{'caption':'a man','pid':'1'},('a man',1),42,None])
def test_h_accepts_caption_string_only(c,value):
 with core.H_inference_guard(),core.frozen_H(c) as(h,_),pytest.raises(TypeError,match='Caption string only'):h.attack(value)

@pytest.mark.parametrize('operation',['cuda','backward','grad','optimizer','jvp','dual'])
def test_h_guard_retains_forbidden_operations_and_restores(operation):
 x=torch.ones(1,requires_grad=True);opt=torch.optim.Adam([x]);before=torch.cuda._lazy_init
 calls={'cuda':lambda:torch.cuda._lazy_init(),'backward':lambda:x.sum().backward(),'grad':lambda:torch.autograd.grad(x.sum(),x),'optimizer':lambda:opt.step(),'jvp':lambda:torch.func.jvp(lambda a:a,(x,),(torch.ones_like(x),)),'dual':lambda:torch.autograd.forward_ad.make_dual(x,x)}
 with pytest.raises(RuntimeError,match='forbid'):
  with core.H_inference_guard():calls[operation]()
 assert torch.cuda._lazy_init is before
 x.sum().backward();assert x.grad.item()==1

@pytest.mark.parametrize('missing',[None,('rstp','Text-only'),('rstp','Joint'),('icfg','Text-only'),('icfg','Joint')])
def test_all_four_independent_smoke_audits_gate_full(tmp_path,missing):
 c={'output_base':str(tmp_path/'tbps-framework-new-attempt'),'sha256':'a'*64}
 for ds in core.DATASETS:
  for mode in core.MODES:
   if (ds,mode)==missing:continue
   s=core.result_store(c,ds,'smoke',mode);s.publish('cpu_audit',{'report.json':json_bytes({'status':'PASS','scope':'smoke','mode':mode,'config_sha256':c['sha256']})},{'kind':'CPU-fixture'})
 if missing is None:execute.gate_full(c)
 else:
  with pytest.raises(Exception):execute.gate_full(c)

@pytest.mark.parametrize('failed',[False,True])
def test_unpublished_image_source_never_reads_partial_cache(tmp_path,monkeypatch,failed):
 base=tmp_path/'tbps-framework-source';s=ArtifactStore(base/'source',new_base=base);s.root.mkdir(parents=True);(s.root/'gallery-feature-000000').mkdir();(s.root/'gallery-feature-000000/features.pt').write_bytes(b'partial not readable')
 c={'image_control':str(tmp_path/'control')};monkeypatch.setattr(core,'image_config',lambda c:{});monkeypatch.setattr(core.image,'store_for',lambda *a:s)
 monkeypatch.setattr(core,'manifest',lambda *a:pytest.fail('must not inspect uncommitted cache'))
 if failed:(s.root/'failure').mkdir()
 if failed:
  with pytest.raises(ValueError,match='source failed'):core.dependency(c,'icfg','full',required=False)
 else:
  assert core.dependency(c,'icfg','full',required=False) is None
  with pytest.raises(ValueError,match='WAITING_FOR_AUDITED_IMAGE_SOURCE'):core.dependency(c,'icfg','full')

@pytest.mark.parametrize('key,value',[('H_device','cuda'),('text_batch',16),('query_chunk',64),('modes',['Clean','Joint']),('G_checkpoint',{'sha256':'0'*64}),('H_checkpoint',{'sha256':'0'*64}),('resource_gate',{'RAM_bytes':0,'disk_bytes':0,'inodes':0,'GPU_MiB':0})])
def test_fixed_protocol_rejects_drift(c,tmp_path,key,value):
 d=copy.deepcopy(c);d[key]=value;p=tmp_path/'config.json';p.write_bytes(json_bytes(d))
 with pytest.raises(ValueError):core.config(p)

def test_supervisor_serial_smokes_then_full_without_g_regeneration():
 p=core.ROOT/'operations/run_gh_cross_dataset.py';spec=importlib.util.spec_from_file_location('gh_transfer_supervisor',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
 steps=s.pipeline();assert len(steps)==21
 first=next(i for i,x in enumerate(steps) if x[3]=='full');assert len([x for x in steps[:first] if x[1]=='audit'])==4
 assert [(x[2],x[4]) for x in steps if x[1]=='audit' and x[3]=='full']==[('rstp','Text-only'),('rstp','Joint'),('icfg','Text-only'),('icfg','Joint')]
 assert all(x[1] in ('preflight','generate','evaluate','audit') for x in steps)

@pytest.mark.parametrize('mode',core.MODES)
def test_csv_full_precision_no_smoke_substitution(mode):
 groups={mode:{k:12.34567890123 for k in core.KEYS}}
 raw=execute.csv_data('rstp',mode,groups).decode();assert raw.count('12.34567890123')==4 and mode in raw
 assert execute.csv_data('icfg',mode,groups,True).decode().count('12.35')==4

def test_final_eight_rows_only_defined_input_combinations():
 values={ds:{mode:{k:12.34567890123 for k in core.KEYS} for mode in results.MODES4} for ds in core.DATASETS}
 data=results.table_bytes(values).decode().splitlines();assert len(data)==9
 assert sum('Text-only' in row for row in data)==2 and sum('Joint' in row for row in data)==2

def test_protected_history_detects_change(tmp_path):
 p=tmp_path/'failed-evidence';p.write_text('keep');base=tmp_path/'new';(base/'preflight').mkdir(parents=True);c={'protected_files':[str(p)],'output_base':str(base)}
 (base/'preflight/protected.json').write_bytes(json_bytes(core.static_snapshot(c)));assert core.check_protection(c)==1
 p.write_text('changed')
 with pytest.raises(ValueError,match='sources changed'):core.check_protection(c)

def test_real_two_image_smoke_receipts_and_feature_space(c):
 for ds in core.DATASETS:
  old,m,a,native=core.dependency(c,ds,'smoke')
  assert a['status']=='PASS' and len(m.queries)==len(m.gallery)==10
  _,_,arrays,refs,hashes=core.original_features(c,ds,'smoke',core.VictimSource(core.image_config(c),'irra',ds).space)
  assert arrays['adv_gallery'].shape==(10,512) and arrays['clean_text'].shape==(10,512) and len(hashes)==10
 assert not torch.cuda.is_initialized()


def test_cpu_artifact_full_path_shared_text_no_image_forward(c,tmp_path,monkeypatch):
 """CPU夹具：真实冻结H，合成IRRA特征/评分，绝不作为真实Smoke证据。"""
 from contextlib import contextmanager
 from tbps.contracts import DatasetManifest,GalleryItem,QueryItem,PersonID,ModelSpec
 from tbps.models.base import ModelSpace
 from tbps.artifacts import FileRef
 base=tmp_path/'tbps-framework-cpu-gh-fixture';cfg=copy.deepcopy(c);cfg.update(output_base=str(base),control=str(tmp_path/'control'),protected_files=[])
 origin=tmp_path/'fixture-source.json';origin.write_bytes(b'{"fixture":true}');file=core.ref(origin);cfg['G_checkpoint']=asdict(file)
 ck=ModelSpace('irra_fixture',file.sha256,digest('code'),digest('prep'),digest('tok'),digest('score'),'fp32')
 gs=tuple(GalleryItem('g'+str(i),str(i)+'.jpg',digest(['image',i]),PersonID('rstp_reid',str(i)),'test') for i in range(10))
 captions=('A man wearing a black shirt and blue pants.','A woman in a white dress and black shoes.')
 qs=tuple(QueryItem('q'+str(i),caption,0,gs[i].image_id,gs[i].pid,'test') for i,caption in enumerate(captions))
 m=DatasetManifest('rstp_reid','test',file,gs,qs);root=ArtifactStore(base,new_base=base);root.publish('preflight',{'protected.json':json_bytes({})},{'kind':'CPU-fixture'})
 oldbase=tmp_path/'tbps-framework-audited-image-fixture';old=ArtifactStore(oldbase,new_base=oldbase)
 old.publish('protocol',{'protocol.json':json_bytes({'fixture':True})},{'kind':'CPU-fixture'});old.publish('cpu_audit',{'report.json':json_bytes({'status':'PASS'})},{'kind':'CPU-fixture'})
 monkeypatch.setattr(core,'manifest',lambda *a:m);monkeypatch.setattr(execute,'manifest',lambda *a:m)
 monkeypatch.setattr(execute,'resources',lambda c:{'fixture':True});monkeypatch.setattr(execute,'dependency',lambda *a:(old,m,{'status':'PASS'},{}))
 monkeypatch.setattr(execute,'image_config',lambda c:{'legacy_sources':{'fixture':'0'*64}})
 arrays={'clean_text':torch.eye(2,512),'clean_gallery':torch.eye(10,512),'adv_gallery':torch.eye(10,512).flip(0)}
 refs={key:[core.ref(old.root/'protocol/manifest.json')] for key in arrays};monkeypatch.setattr(execute,'original_features',lambda *a:(old,m,arrays,refs,tuple(digest(['adv',i]) for i in range(10))))
 class Model(torch.nn.Module):
  def __init__(self):
   super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1),requires_grad=False);self.base_model=torch.nn.Module();self.base_model.visual=torch.nn.Identity();self.base_model.transformer=torch.nn.Identity();self.eval()
 model=Model();adapter=SimpleNamespace(_backend=model,space=ck,spec=ModelSpec('irra_fixture','fixture',(384,128),None,512,'dot','native',frozenset({'score'})))
 @contextmanager
 def victim(*a,**kw):yield adapter,model,SimpleNamespace(checkpoint=file)
 monkeypatch.setattr(execute,'load_victim',victim);encoded=[]
 def text_encoder(adapter,texts,kind):
  encoded.extend(texts);model.base_model.transformer(torch.zeros(77,len(texts),1));return torch.nn.functional.normalize(torch.ones(len(texts),512),dim=1)
 monkeypatch.setattr(execute,'encode_texts',text_encoder)
 execute.generate(cfg,'rstp','smoke')
 execute.evaluate(cfg,'rstp','smoke','Text-only');count=len(encoded);execute.audit(cfg,'rstp','smoke','Text-only')
 execute.evaluate(cfg,'rstp','smoke','Joint');execute.audit(cfg,'rstp','smoke','Joint')
 assert len(encoded)==count
 for mode in core.MODES:
  receipt=json.loads((core.result_store(cfg,'rstp','smoke',mode).root/'cpu_audit/report.json').read_text())
  assert receipt['status']=='PASS' and receipt['Query']==2 and receipt['neural_forwards']==0
 assert not torch.cuda.is_initialized()
