import hashlib
import importlib.util
from pathlib import Path
import pytest
import torch
spec=importlib.util.spec_from_file_location("review_v2",Path(__file__).parents[2]/"operations/gh_smoke_readonly_review_v2.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SOURCE="""def state_sha(module):
    h=hashlib.sha256()
    for name,x in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(x.shape),x.dtype)).encode())
        h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
"""
def test_correct_native_H_hash_space(tmp_path):
    from tbps.alternating.state import tree_sha
    p=tmp_path/"native_hash.py";p.write_text(SOURCE)
    h=m.legacy_h_hasher(p,hashlib.sha256(p.read_bytes()).hexdigest())
    state={"weight":torch.tensor([1.,2.])}
    direct=hashlib.sha256()
    direct.update(b"weight");direct.update(str(((2,),torch.float32)).encode());direct.update(state["weight"].numpy().tobytes())
    assert h(state)==direct.hexdigest()
    assert h(state)!=tree_sha(state)
    assert h({"weight":state["weight"].clone()})==h(state)
def test_hash_source_tamper_rejected(tmp_path):
    p=tmp_path/"native_hash.py";p.write_text(SOURCE)
    with pytest.raises(AssertionError,match="source SHA"):m.legacy_h_hasher(p,"0"*64)
