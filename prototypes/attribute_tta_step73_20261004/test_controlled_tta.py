import copy
import importlib.util
import json
from pathlib import Path
import sys
import ast
import pytest
import torch
from torch import nn
from torch.nn import functional as F
REPO=Path("/home/lzf/ldx/projects/idea-TBPS-test1")
sys.path.insert(0,str(REPO/"src"))
sys.path.insert(0,str(REPO/"prototypes/attribute_tta_step72_20261003"))
import controlled_tta as mod
from controlled_tta import *
from attribute_tta import prepared_plan, seed_all, digest, BASIS, P51, P61
from attributes.tta_adapter import _tokenize

class ToySource(nn.Module):
    def __init__(self):
        super().__init__(); self.text_calls=0
    def encode_images(self,x):
        return F.normalize(x.mean(dim=(-2,-1)),dim=-1)
    def inference_image(self,x):
        return {"image_feat":self.encode_images(x)}
    def inference_text(self,*args):
        self.text_calls+=1
        raise AssertionError("attack may not encode text")

@pytest.fixture
def plans():
    rows={r["row_id"]:r for r in json.loads((P51/"outputs/diagnostic/results.json").read_text(encoding="utf-8"))}
    payloads={r["caption_id"]:r for r in json.loads((P61/"outputs/diagnostic/selection_payloads.json").read_text(encoding="utf-8")) if r["gamma"]==3}
    bases=json.loads(BASIS.read_text(encoding="utf-8"))
    return {cid:prepared_plan(p,rows[cid],bases) for cid,p in payloads.items()}

def test_frozen_sources_and_completed_fixed25(plans):
    lock=json.loads((Path(__file__).parent/"provenance.lock.json").read_text(encoding="utf-8"))
    for p,h in lock["frozen_sha256"].items():assert digest(p)==h
    assert len(plans)==25 and sum(map(len,plans.values()))==75

def test_same_plan_all_modes_and_raw_provenance(plans):
    for cid,plan in plans.items():
        before=copy.deepcopy([a.metadata for a in plan])
        for a in plan:
            new={a.instance_id:torch.tensor([[1.,2.,3.]])}
            raw={a.instance_id:torch.tensor([[3.,2.,1.]])}
            clean=torch.tensor([[4.,5.,6.]])
            for mode in MODES+("LEGACY",):
                v,labels=guidance(mode,a,new,raw,clean)
                if mode=="NEW":assert torch.equal(v,new[a.instance_id]) and labels==["attribute_unit:"+a.instance_id]
                if mode=="RAW":assert torch.equal(v,raw[a.instance_id]) and labels==[a.metadata["raw_text"]]
                if mode=="NONE":assert torch.equal(v,clean) and v.shape[0]==1
                if mode=="LEGACY":assert torch.equal(v,torch.cat([clean,raw[a.instance_id]]))
        assert [a.metadata for a in plan]==before

def test_categorical_replay_couples_actual_choices():
    seed_all(42);t=CategoricalTape()
    draws=[t.multinomial(torch.tensor([.95,.05]),8,True) for _ in range(4)]
    expected_rng=rng_fingerprint();t.finish()
    seed_all(42);other=CategoricalTape(t.records)
    replay=[other.multinomial(torch.tensor([.01,.99]),8,True) for _ in range(4)]
    assert all(torch.equal(a,b) for a,b in zip(draws,replay))
    assert rng_fingerprint()==expected_rng and other.finish()==t.finish()
    with pytest.raises(ValueError):other.multinomial(torch.ones(2),8,True)

def test_projection_logging_on_known_clipping():
    before=torch.tensor([.5,.5]);proposal=torch.tensor([.6,.4]);after=torch.tensor([.55,.45])
    s=projection_stats(before,proposal,after)
    assert s["projection_clipping_ratio"]==1
    assert s["pre_projection_update"]["linf"]==pytest.approx(.1,abs=1e-7)
    assert s["post_projection_actual_change"]["linf"]==pytest.approx(.05,abs=1e-7)
    assert s["absolute_step_magnitude_removed_fraction"]==pytest.approx(.5,abs=1e-6)

def test_projection_logging_outside_initialization_does_not_hide_ratio():
    s=projection_stats(torch.tensor([.9]),torch.tensor([.91]),torch.tensor([.6]))
    assert s["projection_removed_l2_over_requested_l2"]>1
    assert s["absolute_step_magnitude_removed_fraction"]==0

def test_zero_and_undefined_cosine():
    s=projection_stats(torch.ones(2),torch.ones(2),torch.ones(2))
    assert s["actual_change_exact_zero"] and s["projection_removed_l2_over_requested_l2"] is None
    assert cosine(torch.zeros(2),torch.ones(2)) is None

@pytest.mark.parametrize("mode",MODES+("LEGACY",))
def test_observer_official_output_exact_and_no_text_encoder(plans,mode):
    source=ToySource();a=plans["cuhk:35:1"][0]
    image=torch.linspace(.2,.8,3*16*16).reshape(1,3,16,16)
    features=torch.tensor([[.2,.5,-.3]])
    if mode=="LEGACY":features=torch.cat([features,torch.tensor([[.1,-.2,.4]])])
    labels=[str(i) for i in range(len(features))]
    seed_all(42)
    plain=load_official_attack(OFFICIAL)(None,_tokenize,imgs_eps=IMAGE_EPSILON,step_size=2/255)
    plain.N_trans=1
    bridge=FixedGuidanceBridge(source,features,labels)
    expected,m=plain.img_attack(bridge,labels,image,image,[0]*len(labels),2,torch.zeros_like(image),"cpu",scales=None)
    end=rng_fingerprint()
    seed_all(42)
    tape=CategoricalTape()
    engine=ControlledTTA(source,image,mode,steps=2,transforms=1,scales=None,tape=tape)
    result=engine.step(image,a,features,labels)
    assert torch.equal(result,project_image(image,expected.detach()))
    assert torch.equal(engine.momentum,m.detach()) and rng_fingerprint()==end
    assert source.text_calls==0 and len(engine.events[0]["steps"])==2
    assert engine.events[0]["caption_loss_terms"]==int(mode in ("NONE","LEGACY"))
    assert engine.events[0]["cumulative_perturbation_linf"]<=IMAGE_EPSILON+1e-6

def test_retrieval_evaluation_uses_caption_only_as_detached_metrics():
    source=ToySource();text=F.normalize(torch.tensor([[1.,2.,3.]]),dim=-1)
    gallery=F.normalize(torch.tensor([[2.,1.,3.],[1.,3.,2.],[2.,3.,1.]]),dim=-1)
    metric=RetrievalMetrics(source,text,gallery,1,tau=.02)
    image=torch.tensor([.2,.4,.8]).reshape(1,3,1,1)
    out=metric(image)
    paired=(text*source.encode_images(image)).sum()
    scores=(text@gallery.T).flatten()
    scores[1]=paired
    # 复用已冻结真实函数的AST作为数值oracle，不加载旧模型/训练组件。
    p=Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/src/attributes/irra_soft_rank_gradient.py")
    node=next(n for n in ast.parse(p.read_text(encoding="utf-8")).body if isinstance(n,ast.FunctionDef) and n.name=="paired_soft_rank")
    scope={"torch":torch,"math":__import__("math")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(p),"exec"),scope)
    assert out["paired_soft_rank_F"]==float(scope["paired_soft_rank"](scores,1,.02))
    assert out["positive_image_rank"]==1+int((torch.cat([scores[:1],scores[2:]])>paired).sum())
    assert metric_delta(out,out)=={"paired_cosine":0.,"paired_soft_rank_F":0.,"positive_image_rank":0}
    assert source.text_calls==0 and not metric.text.requires_grad

def test_no_full_or_unprotected_diagnostic_cli():
    spec=importlib.util.spec_from_file_location("step73_run",Path(__file__).parent/"run.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    assert runner.parser().parse_args([]).mode=="smoke"
    with pytest.raises(SystemExit):runner.parser().parse_args(["--mode","full"])
    with pytest.raises(SystemExit):runner.main(["--mode","diagnostic"])
    assert len(runner.SMOKE_IDS)==2

def test_official_source_observation_independent_of_process_locale():
    import locale
    prior=locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE,"C")
        assert observed_attack_type(CategoricalTape()) is not None
    finally:
        locale.setlocale(locale.LC_CTYPE,prior)
