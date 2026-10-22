"""完整2048流式刷新调度的CPU Mock：不调用真实G/IRRA或保存大图。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import pytest
from tbps.alternating.native_backend import NativeBackend
from tbps.alternating.state import clone_tree,tree_sha

@pytest.mark.parametrize("failure",[None,3])
def test_real_refresh_control_2048_claim_forward_commit(tmp_path,failure):
    import tbps.alternating.native_backend as native
    published={};serialized={};events=[];steps={"G":0,"encoder":0}
    class Store:
        def __init__(self,root,**kwargs):self.root=Path(root)
        def publish(self,key,files,provenance):
            assert key not in published
            published[key]={"files":files,"provenance":provenance}
            events.append(("publish",key))
    def pack(value):
        key=str(len(serialized)).encode()
        serialized[key]=clone_tree(value);return key
    def load(path,**kwargs):
        assert kwargs=={"map_location":"cpu","weights_only":True}
        return clone_tree(serialized[published[Path(path).parent.name]["files"]["data.pt"]])
    backend=NativeBackend.__new__(NativeBackend)
    ids=[f"train:{i}" for i in range(2048)]
    backend.c={"output_base":str(tmp_path/"tbps-framework-fixture")}
    backend.p={"train_ids":ids,"clean_gallery_sha256":"a"*64,"source_space_sha256":"b"*64}
    backend.root=Path(backend.c["output_base"])/"smoke_v1"
    backend.grant=SimpleNamespace(validate=lambda c,a:None)
    backend.g=torch.nn.Linear(1,1);backend.g_fingerprint=lambda:tree_sha(backend.g.state_dict())
    def image(row):
        i=ids.index(row)
        assert f"claim_{i:04d}" in published
        assert f"image_{i:04d}" not in published
        if i==failure:raise RuntimeError("fixture generation stopped; Claim retained")
        return torch.full((1,3,1,1),.5)
    backend.image=image
    def generate(g,x):
        steps["G"]+=1;events.append(("G",steps["G"]-1))
        return {"adversarial":x+.01,"effective_delta":x+.01-x}
    backend.native=SimpleNamespace(get=lambda name:SimpleNamespace(generate_once=generate))
    def encode(x):
        steps["encoder"]+=1
        return torch.nn.functional.normalize(torch.ones(1,512),dim=1)
    backend.encoder=encode
    backend.current_features=torch.nn.functional.normalize(torch.ones(2048,512),dim=1)
    with patch.object(native,"ArtifactStore",Store),patch.object(native,"tensor_bytes",pack),\
         patch.object(native,"assert_pixels",lambda a,b,d:None),patch.object(torch,"load",load):
        if failure is not None:
            with pytest.raises(RuntimeError):backend.refresh_bank(ids,"B",1)
            assert steps=={"G":failure,"encoder":failure}
            assert f"claim_{failure:04d}" in published and "completed" not in published
            assert f"image_{failure:04d}" not in published
        else:
            identity=backend.refresh_bank(ids,"B",1)
            assert identity.ordered_train_ids==tuple(ids)
            assert identity.adversarial_sha256==tree_sha(backend.current_features)
            assert steps=={"G":2048,"encoder":2048}
            assert len([x for x in published if x.startswith("claim_")])==2048
            assert len([x for x in published if x.startswith("image_")])==2048
            assert "completed" in published
            for i in range(2048):
                assert events.index(("publish",f"claim_{i:04d}"))<events.index(("G",i))<events.index(("publish",f"image_{i:04d}"))
    assert not torch.cuda.is_initialized()

def test_partial_or_reordered_train_bank_refused(tmp_path):
    b=NativeBackend.__new__(NativeBackend)
    b.c={};b.p={"train_ids":[str(i) for i in range(2048)]}
    b.grant=SimpleNamespace(validate=lambda c,a:None)
    for ids in (b.p["train_ids"][:-1],list(reversed(b.p["train_ids"]))):
        with pytest.raises(ValueError):b.refresh_bank(ids,"A1",1)
