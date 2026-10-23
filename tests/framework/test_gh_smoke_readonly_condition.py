import importlib.util
import json
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location("review_v3",Path(__file__).parents[2]/"operations/gh_smoke_readonly_review_v3.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_all_real_condition_JSON_roundtrip_readonly():
    o=Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2")
    rows=json.loads((o/"completed/report.json").read_text())["conditions"]
    keys=[m.condition_key_from_json(e["identity"]) for e in rows]
    expected=[]
    for arm in ["A1","B"]:
        for i in [1,2]:
            expected.extend(json.loads((o/"rounds"/f"{arm}-r{i}"/"ledger.json").read_text())["conditions"])
    assert keys==expected and len(keys)==32
    assert all(type(e["identity"]["selected_g_ids"]) is list and type(e["identity"]["scores"]) is list for e in rows)
    assert not torch.cuda.is_initialized()
def test_changed_context_is_not_original_identity():
    o=Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2")
    x=json.loads((o/"completed/report.json").read_text())["conditions"][1]["identity"]
    original=m.condition_key_from_json(x)
    assert m.condition_key_from_json({**x,"context_caption_sha256":"0"*64})!=original
