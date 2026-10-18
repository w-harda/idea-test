"""容量入口只改变选样范围/输出位置，不替换冻结攻击函数。"""
from pathlib import Path
from unittest.mock import patch
import copy
import pytest
import torch
from tbps.artifacts import ArtifactError
from tbps.baselines import sga_batch8_capacity as c

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/framework/sga_batch8_capacity.server.json"

def test_actual_order_all_captions():
    cfg=c.check_config(CONFIG);_,raw,matched=c.select_first_eight(cfg,c.SCOPE)
    assert len(raw.gallery)==8 and len(raw.queries)==16
    assert [x.image_id for x in raw.gallery]==["cuhk:"+str(i)+":0" for i in range(37132,37140)]
    assert [x.query_id for x in raw.queries]==cfg["capacity"]["query_ids"]
    assert all(a.paired_image_id==b.paired_image_id and a.pid==b.pid for a,b in zip(raw.queries,matched.queries))

def test_preflight_forbids_model_cuda_and_forward():
    with patch.object(torch,"load",side_effect=AssertionError("model forbidden")),patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")),patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("forward forbidden")):
        assert c.main(["--config",str(CONFIG)])==0
    assert not torch.cuda.is_initialized()

def test_explicit_gpu_permission():
    with patch.object(c.frozen,"source_run",side_effect=AssertionError("must not run")):
        with pytest.raises(ArtifactError,match="authorization"):
            c.main(["--config",str(CONFIG),"--action","run"])

def test_single_delegate_no_retry():
    calls=[]
    def fail(path,scope):
        calls.append(scope)
        cfg=c.check_config(path)
        _,raw,_=c.frozen.dataset(cfg,scope)
        assert len(raw.gallery)==8 and len(raw.queries)==16
        return 7
    with patch.object(c.frozen,"source_run",fail):
        assert c.main(["--config",str(CONFIG),"--action","run","--allow-gpu-capacity"])==7
    assert calls==[c.SCOPE]
    assert c.frozen.dataset is not c.select_first_eight

@pytest.mark.parametrize("field",["gallery_ids","query_ids"])
def test_identity_change_rejected(field):
    cfg=c.check_config(CONFIG);cfg=copy.deepcopy(cfg);cfg["capacity"][field].reverse()
    with pytest.raises(ArtifactError,match="first-eight"):c.select_first_eight(cfg,c.SCOPE)

def test_cannot_select_full_scope():
    cfg=c.check_config(CONFIG)
    with pytest.raises(ArtifactError,match="capacity-only"):c.select_first_eight(cfg,"full")

def test_original_algorithm_parameters():
    cfg=c.check_config(CONFIG)
    assert (cfg["batch_size"],cfg["steps"],cfg["seed"])==(8,10,42)
    assert (cfg["epsilon"],cfg["step_size"])==(2/255,.5/255)
    assert cfg["scales"]==[.5,.75,1.25,1.5] and cfg["capacity"]["source_only"]
