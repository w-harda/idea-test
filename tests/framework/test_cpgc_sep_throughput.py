import torch,pytest,copy
from pathlib import Path
from types import SimpleNamespace
from tbps.baselines.cpgc_sep.cache import BlockCache,FeatureBank
from tbps.artifacts import ArtifactError,ArtifactStore
@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
 monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:(_ for _ in ()).throw(AssertionError('CUDA in CPU test')))
def test_cache_hit_is_readonly_and_bounded():
 c=BlockCache(32);x=torch.arange(8,dtype=torch.float32);calls=[]
 assert torch.equal(c.get('a',lambda:(calls.append('a') or x)),x)
 assert c.get('a',lambda:(_ for _ in ()).throw(AssertionError())) is x
 c.get('b',lambda:x+1)
 assert c.used==32 and list(c.entries)==['b'] and calls==['a']
def test_cache_rejects_unverified_loader():
 c=BlockCache(128)
 with pytest.raises(ArtifactError):c.get('x',lambda:(_ for _ in ()).throw(ArtifactError('SHA mismatch')))
 assert not c.entries
def test_oversized_not_retained():
 c=BlockCache(4);c.get('x',lambda:torch.zeros(2));assert c.used==0
def test_cache_full_identity():
 c=BlockCache(100)
 c.get(('old','clean',0),lambda:torch.tensor([1.]))
 c.get(('new','clean',0),lambda:torch.tensor([2.]))
 assert len(c.entries)==2
def test_fetch_preserves_duplicates_order_and_storage():
 b=FeatureBank.__new__(FeatureBank);b.ids=tuple(map(str,range(16)));b.chunk=8;b.family='g'
 b.store=SimpleNamespace(root=Path('/fixture'));b.block_cache=BlockCache(1024);b.lru={}
 values=torch.arange(16,dtype=torch.float32).reshape(16,1)
 b.block=lambda i:{'global':values[i:i+8],'tokens':values[i:i+8]*2}
 x=b.fetch([15,0,8,15]);assert torch.equal(x['global'],values[[15,0,8,15]])
 x['global'].zero_();assert torch.equal(values,torch.arange(16.).reshape(16,1))
 assert b.block_cache.misses==2 and b.block_cache.hits==2
def test_fetch_out_of_range_rejected():
 b=FeatureBank.__new__(FeatureBank);b.ids=('a',);b.chunk=8
 with pytest.raises(ArtifactError):b.fetch([1])
def test_target_uses_real_batches():
 import inspect
 from tbps.baselines.cpgc_sep.target import evaluate
 s=inspect.getsource(evaluate)
 assert 'torch.cat(pixels[b:b+image_batch])' in s
 assert 'modality="text",chunk=text_batch' in s
def test_native_core_remains_readonly():
 import json,hashlib
 root=Path(__file__).resolve().parents[2];c=json.loads((root/'configs/framework/cpgc_sep_tbps.throughput_v3.server.json').read_text())
 for v in c['official'].values():
  for p,h in v['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 assert c['evaluation']['ITM_TopK']==128

def test_digest_invalidation_even_if_mtime_preserved(tmp_path):
 from tbps.baselines.cpgc_sep.verified_store import DigestSession
 from tbps.artifacts import FileRef,sha_file
 import os
 p=tmp_path/'source';p.write_bytes(b'abc');old=p.stat();ref=FileRef(str(p),sha_file(p))
 c=DigestSession();c.checked(ref);c.checked(ref);assert c.misses==1 and c.hits==1
 p.write_bytes(b'xyz');os.utime(p,ns=(old.st_atime_ns,old.st_mtime_ns))
 with pytest.raises(ArtifactError):c.checked(ref)
@pytest.mark.parametrize('mutation',['payload','receipt','source','unexpected','missing'])
def test_verified_store_enforces_all_original_integrity_checks(tmp_path,mutation):
 from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
 from tbps.artifacts import FileRef,sha_file
 base=tmp_path/'tbps-framework-fixture';store=VerifiedArtifactStore(base/'run',new_base=base)
 source=tmp_path/'source';source.write_bytes(b'abc')
 store.publish('row',{'x':b'payload'},{'kind':'fixture'},sources=(FileRef(str(source),sha_file(source)),))
 store.verify('row')
 if mutation=='payload':(store.root/'row/x').write_bytes(b'changed')
 elif mutation=='receipt':(store.root/'row/completed.json').write_text('{}')
 elif mutation=='source':source.write_bytes(b'changed')
 elif mutation=='unexpected':(store.root/'row/extra').write_bytes(b'x')
 else:(store.root/'row/x').unlink()
 with pytest.raises((ArtifactError,KeyError,FileNotFoundError)):store.verify('row')
def test_source_reuse_rejects_semantic_change():
 import json,copy
 from tbps.baselines.cpgc_sep.source_reuse import resolve
 root=Path(__file__).resolve().parents[2]
 c=json.loads((root/'configs/framework/cpgc_sep_tbps.throughput_v3.server.json').read_text())
 c['cpgc_protocol']['extra_unauthorized_change']=True
 with pytest.raises(ArtifactError,match='semantic mismatch'):resolve(c,'cpgc','icfg','smoke')
def test_sep_source_preserves_original_math_batch():
 import inspect
 from tbps.baselines.cpgc_sep.sep_source import run
 s=inspect.getsource(run)
 assert 'source_config["evaluation"]["fusion_batch"]=c["evaluation"].get("sep_fusion_batch",16)' in s
 assert 'load_victim(source_config,name,dataset' in s

def test_v3_schedule_keeps_five_old_full_out_of_gpu_plan():
 import importlib.util,sys,json
 root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('v3_schedule_fixture',root/'operations/cpgc_sep_throughput_run.py')
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 c=json.loads((root/'configs/framework/cpgc_sep_tbps.throughput_v3.server.json').read_text())
 steps=list(m.schedule(c));full=[s for s in steps if s.scope=='full' and s.kind=='target']
 assert len(full)==7 and full[0].label=='cpgc_full_icfg_rasa_target'
 assert [(s.method,s.dataset,s.victim)for s in full[1:]]==[('sep',d,v)for d in('cuhk','rstp','icfg')for v in('irra','rasa')]
 assert sum(s.scope=='smoke'and s.method=='cpgc'and s.kind=='audit'for s in steps)==6
def test_v3_failure_guard():
 root=Path(__file__).resolve().parents[2];s=(root/'operations/cpgc_sep_throughput_run.py').read_text()
 assert 'm.safe_prelaunch_busy' in s and 'global_lock' in s and 'started' in s
 assert 'preserve and stop' in s

def test_independent_smoke_parity_real_readonly_reference():
 import importlib.util,sys
 root=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('parity_cpu_fixture',root/'operations/cpgc_sep_v3_parity.py')
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 result=m.compare({'output_base':str(m.OLD)},'cuhk','rasa')
 assert result['status']=='PASS' and result['GPU']is False
 assert all(x['exact']and x['ranking_exact']and x['max_abs']==0 for x in result['scores'].values())
 assert not torch.cuda.is_initialized()
