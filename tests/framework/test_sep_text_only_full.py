"""只验证本轮接线与实际错误保护，不重跑176项历史回归。"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import pytest
import torch

from operations.sep_text_only.runtime import query_key,source_guard,CONFIG
from operations.sep_source_only import attack
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan
from tbps.contracts import PersonID

ROOT=Path(__file__).resolve().parents[2]


def test_single_source_official_dpp_and_pidplan():
    c=json.loads((ROOT/"configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json").read_text())
    class Source:
        def eval(self):return self
    e=attack.s1_ensemble(c,Source(),"cpu")
    e.init_weights()
    assert e.len_weights==1 and e.next_weights().tolist()==[1.]
    assert PIDPlan.build(("B","A","B")).members==((0,2),(1,))


def test_source_only_file_gate_in_subprocess(tmp_path):
    # Create only fixture data; the audit hook must reject real annotation names.
    code="""
from operations.sep_text_only.runtime import source_guard
from pathlib import Path
import sys
root=Path(sys.argv[1])
bundle={"train_root":str(root),"train_gallery":[],
"config":{"victims":{"irra":{"clip_bootstrap":{"path":str(root/"clip.pt")},
"datasets":{"cuhk":{"checkpoint":{"path":str(root/"source.pth")}}}}}}}
source_guard({"output":str(root/"out"),"control":str(root/"control")},bundle)
for p in ("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json",
"/home/lzf/Attack/IRRA/logs/RSTPReid/best.pth",str(root/"target.pth")):
    try:open(p,"rb")
    except PermissionError:pass
    else:raise AssertionError("target information accessible")
"""
    subprocess.run([sys.executable,"-B","-c",code,str(tmp_path)],cwd=ROOT,check=True)


def test_query_seed_and_identity_stable():
    key=query_key("cuhk","row1")
    assert key==query_key("cuhk","row1") and key!=query_key("rstp","row1") and len(key)==64


@pytest.mark.parametrize("noop",[True,False])
def test_reused_search_edit_and_noop(noop):
    import nltk
    c=json.loads((ROOT/"configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json").read_text())
    nltk.data.path.insert(0,c["nltk_root"])
    vals={"man":.4,"wearing":.33,"blue":.25,"shirt":.12,"person":.1,"dressed":.09,"azure":.08,"top":.04}
    words=list(vals);mapping={w:i for i,w in enumerate(words)}
    pairs={"man":"person","wearing":"dressed","blue":"azure","shirt":"top"}
    neighbors=[[(1.,i),(.9,i if noop else mapping[pairs.get(w,w)])] for i,w in enumerate(words)]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=self.hits=0;self.cache={}
            self.pid_plan=PIDPlan.build((PersonID("cuhk_train","0"),PersonID("cuhk_train","1")))
        def forward(self,texts):
            rows=[]
            for t in texts:
                self.calls+=1
                if t in self.cache:self.hits+=1
                else:self.cache[t]=torch.tensor([sum(vals.get(w,0) for w in t.split()),-.5])
                rows.append(self.cache[t])
            return SimpleNamespace(logits=torch.stack(rows))
    import random,numpy as np
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    row,_=attack.attack_caption(c,"A man wearing a blue shirt.",Model(),"cpu",
                               (mapping,dict(enumerate(words)),neighbors),lambda *a,**k:None)
    assert row["noop"] is noop and row["DPP"]["N"]==row["DPP"]["E"]==1


def test_text_only_target_modes():
    from operations.sep_text_only import execute
    import inspect
    text=inspect.getsource(execute.evaluate)
    assert 'mode":"Text-only"' in text and 'Image-only' not in text and 'Matched Clean' not in text
    c=json.loads(CONFIG.read_text())
    assert c["smoke_queries"]==8 and c["fusion_batch"]==16
    assert c["query_chunk"]==32 and c["image_batch"]==1


def test_official_zero_importance_is_explicit_failure():
    import nltk
    c=json.loads((ROOT/"configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json").read_text())
    nltk.data.path.insert(0,c["nltk_root"])
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__();self.calls=self.hits=0
            self.pid_plan=PIDPlan.build((PersonID("cuhk_train","0"),PersonID("cuhk_train","1")))
        def forward(self,texts):
            self.calls+=len(texts)
            return SimpleNamespace(logits=torch.tensor([[1.,0.]]*len(texts)))
    terms=["man","wearing","blue","shirt"]
    mapping={word:i for i,word in enumerate(terms)}
    with pytest.raises(RuntimeError,match="cannot terminate"):
        attack.attack_caption(c,"a man wearing blue shirt",Constant(),"cpu",
                              (mapping,dict(enumerate(terms)),[[(1.,i),(.9,i)] for i in range(4)]),
                              lambda *a,**k:None)
