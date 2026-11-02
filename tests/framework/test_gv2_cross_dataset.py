"""跨数据集身份、只读保护、阶段门禁与推理边界CPU回归。"""
from pathlib import Path
from dataclasses import replace,asdict
import json,importlib.util
import pytest
import torch
from tbps.transfer import gv2_core as core
from tbps.transfer import gv2_execute as execute
from tbps.artifacts import ArtifactStore,digest,json_bytes
from tbps.contracts import FileRef,PersonID,GalleryItem,QueryItem,DatasetManifest

def manifest(tmp_path):
 p=tmp_path/'ann.json';p.write_text('[]')
 gs=tuple(GalleryItem(f'rstp:{i}:0',f'{i}.jpg','1'*64,PersonID('rstp_reid',str(i//2)),'test') for i in range(12))
 qs=tuple(QueryItem(f'rstp:{i}:{j}',f'Original Caption {i} {j}',j,g.image_id,g.pid,'test') for i,g in enumerate(gs) for j in range(2))
 return DatasetManifest('rstp_reid','test',core.ref(p),gs,qs)

def test_fixed_smoke_original_order_caption_and_positives(tmp_path,monkeypatch):
 m=manifest(tmp_path);monkeypatch.setattr(core,'read_test',lambda _:m)
 c={'datasets':{'rstp':{}},'manifest_sha256':{'rstp':digest(asdict(m))}}
 out=core.get_manifest(c,'rstp','smoke')
 assert out.gallery==m.gallery[:10] and out.queries==m.queries[:10]
 assert all(q.pid in {g.pid for g in out.gallery} for q in out.queries)
 assert all(q.caption==m.queries[i].caption for i,q in enumerate(out.queries))
 assert core.get_manifest(c,'rstp','full') is m

@pytest.mark.parametrize('field', ['caption','pid','order'])
def test_manifest_identity_drift_rejected(tmp_path,monkeypatch,field):
 m=manifest(tmp_path)
 if field=='pid':
  gs=tuple(replace(g,pid=m.gallery[-1].pid) if i<2 else g for i,g in enumerate(m.gallery))
  qs=tuple(replace(q,pid=gs[int(q.query_id.split(':')[1])].pid) for q in m.queries)
  changed=replace(m,gallery=gs,queries=qs)
 else:changed=replace(m,queries=m.queries[::-1]) if field=='order' else replace(m,queries=(replace(m.queries[0],caption='tampered'),*m.queries[1:]))
 monkeypatch.setattr(core,'read_test',lambda _:changed)
 with pytest.raises(ValueError,match='manifest identity'):core.get_manifest({'datasets':{'rstp':{}},'manifest_sha256':{'rstp':digest(asdict(m))}},'rstp','full')

@pytest.mark.parametrize('function',[lambda:torch.nn.Linear(2,2)(torch.ones(1,2)),lambda:torch.ones(1,requires_grad=True).sum().backward(),lambda:torch.autograd.grad(torch.ones(1,requires_grad=True),torch.ones(1,requires_grad=True)),lambda:torch.cuda._lazy_init()])
def test_cpu_gate_forbids_models_gradients_and_cuda(function):
 with core.no_backward(cpu=True),pytest.raises(RuntimeError,match='forbids'):function()
 assert not torch.cuda.is_initialized()

def test_shared_gpu_zero_free_does_not_veto(monkeypatch,tmp_path):
 from tbps.baselines.cpgc_sep import runtime
 from types import SimpleNamespace
 def run(args,**kwargs):
  s=''
  if '--query-gpu=memory.free' in args:s='0'
  elif '--query-gpu=memory.total' in args:s='24564'
  elif '--query-gpu=utilization.gpu' in args:s='100'
  return SimpleNamespace(returncode=0,stdout=s)
 monkeypatch.setattr(runtime.subprocess,'run',run)
 c={'output_parent':str(tmp_path),'resource_gate':{'RAM_bytes':0,'disk_bytes':0,'inodes':0,'GPU_MiB':999999}}
 result=runtime.resources(c)
 assert result['status']=='READY' and result['memory.free']==0 and not result['blocked']

@pytest.mark.parametrize('tamper',['none','missing','wrongscope','wrongsha'])
def test_both_independent_smoke_audits_gate_full(tmp_path,tamper):
 base=tmp_path/'tbps-framework-transfer-fixture';c={'output_base':str(base),'sha256':'1'*64}
 for ds in core.DATASETS:
  if tamper=='missing' and ds=='icfg':continue
  s=core.store_for(c,ds,'smoke');r={'status':'PASS','dataset':ds,'scope':'full' if tamper=='wrongscope' else 'smoke','config_sha256':'2'*64 if tamper=='wrongsha' else c['sha256']}
  s.publish('cpu_audit',{'report.json':json_bytes(r)},{'kind':'fixture'})
 if tamper=='none':execute.gate_full(c)
 else:
  with pytest.raises(ValueError):execute.gate_full(c)

def test_history_stat_mutation_stops(tmp_path):
 old=tmp_path/'frozen';old.mkdir();p=old/'artifact';p.write_text('unchanged')
 base=tmp_path/'tbps-framework-namespace';(base/'preflight').mkdir(parents=True)
 c={'protected_roots':[str(old)],'output_base':str(base)}
 (base/'preflight/history.json').write_bytes(json_bytes(core.history_snapshot(c)))
 assert core.verify_history(c)==1
 p.write_text('changed')
 with pytest.raises(ValueError,match='historical'):core.verify_history(c)

def test_no_h_or_extra_modes_and_fp32_csv():
 groups={mode:{k:12.3456789012 for k in core.KEYS} for mode in ('Clean','Image-only')}
 assert execute.csv_bytes('rstp',groups).decode().count('12.3456789012')==8
 assert execute.csv_bytes('icfg',groups,True).decode().count('12.35')==8
 assert b'Joint' not in execute.csv_bytes('rstp',groups)

def test_supervisor_order_and_failure_boundary():
 p=core.ROOT/'operations/run_gv2_cross_dataset.py';spec=importlib.util.spec_from_file_location('transfer_supervisor',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 steps=m.pipeline();assert len(steps)==13
 assert steps[0]==('CPU_PREFLIGHT','preflight',None,'smoke')
 first=next(i for i,x in enumerate(steps) if x[3]=='full')
 assert {(x[2],x[1]) for x in steps[:first] if x[0]=='SMOKE_AUDIT'}=={('rstp','audit'),('icfg','audit')}
 assert [x[2] for x in steps[first:] if x[1]=='generate']==['rstp','icfg']
 assert steps[-1]==('ICFG_AUDIT','audit','icfg','full')

def test_real_e60_cpu_identity_without_forward():
 # 真实Checkpoint的CPU严格加载；没有合成攻击结果冒充真实Smoke。
 c=json.loads((core.ROOT/'configs/framework/gv2_cross_dataset.server.json').read_text())
 with core.no_backward(cpu=True),core.frozen_generator(c,'cpu') as (g,fn,pixels):
  assert sum(p.numel() for p in g.parameters())==1377411
  assert all(not p.requires_grad and p.grad is None for p in g.parameters())
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize('mutation',['valid','epoch','batch','epsilon','modes','VRAM','victim'])
def test_protocol_mutations_fail_closed(tmp_path,mutation):
 c=json.loads((core.ROOT/'configs/framework/gv2_cross_dataset.server.json').read_text())
 if mutation=='epoch':c['epoch']=10
 if mutation=='batch':c['image_batch']=2
 if mutation=='epsilon':c['epsilon']=12/255
 if mutation=='modes':c['modes'].append('Joint')
 if mutation=='VRAM':c['resource_gate']['minimum_free_mib']=6144
 if mutation=='victim':c['victims']['irra']['datasets']['rstp']['checkpoint']=c['victims']['irra']['datasets']['cuhk']['checkpoint']
 p=tmp_path/'config.json';p.write_text(json.dumps(c))
 if mutation=='valid':assert core.load_config(p)['epoch']==60
 else:
  with pytest.raises(ValueError):core.load_config(p)
