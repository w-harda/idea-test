"""共享资源接线CPU契约，不模拟GPU成功。"""
from types import SimpleNamespace
import copy,json
from pathlib import Path
import pytest,torch
from tbps.artifacts import ArtifactError
from tbps.models.base import RepresentationBatch,ModelSpace
from tbps.baselines.cpgc_sep.victims import NativeRasa
from tbps.baselines.cpgc_sep.audit import require_smoke_headroom
@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
 monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:(_ for _ in ()).throw(AssertionError("CPU must not initialize CUDA")))
 torch.set_num_threads(1)
def test_all_top128_pairs_are_scored_in_batches_of16(monkeypatch):
 import tbps.baselines.cpgc_sep.victims as v
 monkeypatch.setattr(v,"validate_payload",lambda *args:None)
 calls=[]
 def bert(**kw):
  calls.append(len(kw["encoder_hidden_states"]))
  return SimpleNamespace(last_hidden_state=kw["encoder_hidden_states"])
 def itm(x):return torch.cat((torch.zeros_like(x),x*2.5),1)
 a=NativeRasa.__new__(NativeRasa);a.device="cpu";a.smoke=False
 a.space=ModelSpace("fixture",*["0"*64]*5,"fp32");a.counts={"fusion_pairs":0,"score_calls":0}
 a.backend=SimpleNamespace(text_encoder=SimpleNamespace(bert=bert),itm_head=itm)
 q={"global":torch.tensor([[1.,0.]]),"tokens":torch.zeros(1,1,1),"attention":torch.ones(1,1,dtype=torch.long)}
 g={"global":torch.stack((torch.arange(128,dtype=torch.float32),torch.ones(128)),1),"tokens":torch.arange(128,dtype=torch.float32).reshape(128,1,1)}
 qb=RepresentationBatch(q,a.space.id,"text");gb=RepresentationBatch(g,a.space.id,"image")
 a.fusion_batch=128;reference=a.score(qb,gb).payload
 calls.clear();a.fusion_batch=16;actual=a.score(qb,gb).payload
 assert calls==[16]*8 and torch.equal(reference,actual)
 assert a.counts["fusion_pairs"]==256
@pytest.mark.parametrize("peak,allowed",[(6144,True),(6145,False),(0,False)])
def test_full_requires_measured_real_smoke_headroom(peak,allowed):
 c={"shared_resource_validation":{"safety_margin_mib":2048},"resource_gate":{"minimum_free_mib":8192}}
 r={"peak_GPU_reserved_bytes":peak*1024**2}
 if allowed:require_smoke_headroom(c,r)
 else:
  with pytest.raises(ArtifactError):require_smoke_headroom(c,r)
def test_derived_config_keeps_attack_and_top128():
 W=Path(__file__).resolve().parents[2]
 old=json.loads((W/"configs/framework/cpgc_sep_tbps.server.json").read_text())
 new=json.loads((W/"configs/framework/cpgc_sep_tbps.shared.server.json").read_text())
 assert old["resource_gate"]["minimum_free_mib"]==10240
 assert new["resource_gate"]["minimum_free_mib"]==8192
 for key in ["cpgc_protocol","sep_protocol","resources","datasets"]:assert old[key]==new[key]
 assert new["evaluation"]["ITM_TopK"]==old["evaluation"]["ITM_TopK"]==128
 assert new["evaluation"]["fusion_batch"]==16 and new["evaluation"]["query_chunk"]==32
 assert old["evaluation"]["precision"]==new["evaluation"]["precision"]
 assert not torch.cuda.is_initialized()
