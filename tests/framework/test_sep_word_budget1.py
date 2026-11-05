"""仅固定原始单词预算所需回归。"""
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from operations.sep_text_only import word_budget as w
from tbps.artifacts import ArtifactError
from tbps.baselines.cpgc_sep.sep_bridge import PIDPlan
from tbps.contracts import PersonID
ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def config():
    import nltk
    c=json.loads((ROOT/"configs/framework/cpgc_sep_tbps.no_gpu_floor.server.json").read_text())
    nltk.data.path.insert(0,c["nltk_root"])
    torch.set_num_threads(2)
    return c


@pytest.mark.parametrize("candidate",["A man wears blue shirt.","A person wears blue shirt."])
def test_zero_or_one_original_word(candidate):
    assert len(w.budget_check("A man wears blue shirt.",candidate))<=1


@pytest.mark.parametrize("candidate",["A person wears red shirt.","A man wears shirt.","A tall man wears blue shirt.","man A wears blue shirt."])
def test_two_positions_insert_delete_reorder_rejected(candidate):
    with pytest.raises(ArtifactError):
        w.budget_check("A man wears blue shirt.",candidate)


def test_original_spacing_and_other_words_preserved():
    raw="  A  man\twith BLUE shirt.  "
    assert w.restore_original(raw,"A person with BLUE shirt.")=="  A  person\twith BLUE shirt.  "
    assert w.restore_original(raw,"A man with BLUE shirt.")==raw


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__();self.calls=self.hits=0;self.cache={}
        self.pid_plan=PIDPlan.build((PersonID("cuhk_train","0"),PersonID("cuhk_train","1")))
    def forward(self,texts):
        values={"man":.4,"wearing":.33,"blue":.25,"shirt":.12,"person":.1,"dressed":.09,"azure":.08,"top":.04}
        rows=[]
        for text in texts:
            self.calls+=1
            if text in self.cache:self.hits+=1
            else:
                words=text.split()
                score=sum((i+1)*values.get(x.lower().strip("."),0) for i,x in enumerate(words))/max(len(words)**2,1)
                self.cache[text]=torch.tensor([score,-.5])
            rows.append(self.cache[text])
        return SimpleNamespace(logits=torch.stack(rows))


def vocabulary():
    words=["man","wearing","blue","shirt","person","dressed","azure","top"]
    lookup={v:i for i,v in enumerate(words)}
    pairs={"man":"person","wearing":"dressed","blue":"azure","shirt":"top"}
    return lookup,dict(enumerate(words)),[[(1.,i),(.9,lookup[pairs.get(word,word)])] for i,word in enumerate(words)]


@pytest.mark.parametrize("text",["man","A man wearing BLUE shirt.","A  man\twearing blue shirt.","man " * 260])
def test_official_search_all_probes_mutations_and_final_budget(config,text):
    random.seed(0);np.random.seed(0);torch.manual_seed(0)
    row,_=w.attack_caption(config,text,Source(),"cpu",vocabulary(),lambda *a,**k:None)
    assert row["protocol"]==w.PROTOCOL and row["word_budget"]==1
    assert row["word_positions"]==w.budget_check(text,row["adversarial"])
    assert row["budget_violations"]==0 and row["mutation_checks"]>0
    assert row["DPP"]=={"N":1,"E":1,"draws":100,"weights":[[1.]]}
    for request in row["requests"]:
        for probe in request["texts"]:w.budget_check(text,probe)
    for candidate in row["candidates"]:w.budget_check(text,candidate["text"])
    assert all(t["max_change_num"]==t["att_succ_num"]==1 for t in row["timings"] if t["stage"]=="OFFICIAL_OPTIMIZE")
    assert len(text.split())==len(row["adversarial"].split())


@pytest.mark.parametrize("text",["a","a an the","zzzoovvv"])
def test_no_legal_candidate_is_explicit_raw_noop(config,text):
    row,_=w.attack_caption(config,text,Source(),"cpu",vocabulary(),lambda *a,**k:None)
    assert row["adversarial"]==text and row["noop"] and row["noop_reason"]=="NO_LEGAL_SYNONYM"
    assert row["word_positions"]==[] and row["DPP"]["draws"]==0


def test_budget_ast_and_probe_are_single_replacement(config):
    funcs=w.budget_functions(config,["man","wearing","blue","shirt"])
    captured=[]
    def ensemble(texts,label):
        captured.extend(texts)
        return torch.arange(len(texts),dtype=torch.float32)
    funcs["get_important_order"](["person","wearing","blue","shirt"],ensemble,0,[(0,"man"),(1,"wearing")])
    assert captured==["[UNK] wearing blue shirt","man [UNK] blue shirt"]
    for text in captured:w.budget_check("man wearing blue shirt",text)
