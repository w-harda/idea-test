from pathlib import Path
from types import SimpleNamespace
import json,copy,pytest,torch
from tbps.artifacts import ArtifactError
from tbps.baselines.cpgc_sep import runtime
from tbps.baselines.cpgc_sep.audit import require_smoke_headroom
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
 monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:(_ for _ in ()).throw(AssertionError('CPU must not initialize CUDA')))
@pytest.fixture
def c():return json.loads((ROOT/'configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json').read_text())
def setup(monkeypatch,free):
 def smi(cmd,**kw):
  q=next((s for s in cmd if s.startswith('--query-gpu=')),'')
  value=str(free)if'memory.free'in q else'24564'if'memory.total'in q else'99'if'utilization.gpu'in q else'2749989, ollama, 19700'
  return SimpleNamespace(returncode=0,stdout=value)
 monkeypatch.setattr(runtime.subprocess,'run',smi)
 monkeypatch.setattr(runtime.shutil,'disk_usage',lambda p:SimpleNamespace(free=10**14))
 monkeypatch.setattr(runtime.os,'statvfs',lambda p:SimpleNamespace(f_favail=10**9))
@pytest.mark.parametrize('free',[0,1,4343,8191,8192,24051])
def test_all_gpu_occupancy_is_diagnostic(c,monkeypatch,free):
 setup(monkeypatch,free);v=runtime.resources(c)
 assert v['memory.free']==free and not v['blocked']and v['status']=='READY'
 assert 'ollama'in v['compute_processes'][0]
def test_legacy_gpu_floor_cannot_veto(c,monkeypatch):
 setup(monkeypatch,0);c['resource_gate']['minimum_free_mib']=999999
 assert runtime.resources(c)['status']=='READY'
@pytest.mark.parametrize('kind',['RAM_bytes','disk_bytes','inodes'])
def test_real_host_shortage_still_blocks(c,monkeypatch,kind):
 setup(monkeypatch,4343);c['resource_gate'][kind]=10**20
 v=runtime.resource_snapshot(c);assert kind in v['blocked']
 assert all('GPU'not in k and 'mib'not in k for k in v['blocked'])
 with pytest.raises(ArtifactError,match='RESOURCE_BUSY'):runtime.resources(c)
def test_peak_is_measurement_not_admission(c):
 for n in [1,1024**3,128*1024**3]:require_smoke_headroom(c,{'peak_GPU_reserved_bytes':n})
 with pytest.raises(ArtifactError):require_smoke_headroom(c,{'peak_GPU_reserved_bytes':0})
def test_no_fixed_floor_in_active_config(c):
 assert 'minimum_free_mib'not in c['resource_gate']
 assert c['resource_gate']['RAM_bytes']==96*1024**3
 assert c['evaluation']['fusion_batch']==16 and c['evaluation']['query_chunk']==32
 assert c['evaluation']['image_batch']==1 and c['evaluation']['ITM_TopK']==128
def test_policy_compatibility_retains_numerical_producers(c):
 from tbps.baselines.cpgc_sep.step_reuse import compatible
 ref=next(iter(c['successful_step_reuse'].values()))['config'];parent=json.loads(Path(ref['path']).read_text())
 compatible(c,parent)
 bad=copy.deepcopy(c);bad['evaluation']['fusion_batch']=32
 with pytest.raises(ArtifactError,match='protocol'):compatible(bad,parent)
 bad=copy.deepcopy(c);key=next(p for p in bad['producers']if p.endswith('/cache.py')and'/cpgc_sep/'in p);bad['producers'][key]='0'*64
 with pytest.raises(ArtifactError,match='producer'):compatible(bad,parent)
def test_existing_fail_and_partial_never_replayed(tmp_path):
 from tbps.baselines.cpgc_sep.step_reuse import verify_completed
 root=tmp_path/'tbps-framework-fixture';root.mkdir()
 with pytest.raises(ArtifactError,match='partial'):verify_completed(root,root,'smoke','sep','icfg','irra','target','0'*64)
def test_scheduler_never_uses_old_gpu_gate():
 s=(ROOT/'operations/cpgc_sep_throughput_run.py').read_text()
 assert 'm.wait_ready('not in s and 'm.resources('not in s
 assert 'snap=resource_snapshot(c)'in s and 'WAITING_HOST_RESOURCES'in s
 assert '6+sum'in s
def test_active_runtime_and_audit_have_no_gpu_free_comparison():
 import inspect
 from tbps.baselines.cpgc_sep.runtime import resources,resource_snapshot
 from tbps.baselines.cpgc_sep.audit import require_smoke_headroom
 source=inspect.getsource(resources)+inspect.getsource(resource_snapshot)+inspect.getsource(require_smoke_headroom)
 assert 'minimum_free_mib'not in source and 'safety_margin_mib'not in source
 assert not torch.cuda.is_initialized()
