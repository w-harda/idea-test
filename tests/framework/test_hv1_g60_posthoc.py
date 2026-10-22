"""事后统计的CPU正确性：多正例AP、PID Bootstrap、动作和输入保护。"""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("posthoc",ROOT/"src/tbps/ablations/hv1_g60_posthoc.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_per_query_multiple_positive_AP():
    p=torch.tensor([1,1,2,3])
    s=torch.tensor([[.9,.7,.8,.6],[.7,.5,.8,.6],[.5,.4,.1,.3],[.1,.2,.3,.9]])
    q,z=m.query_metrics(s,p)
    assert np.allclose(q["AP"],[5/6,.5,.25,1])
    assert q["rank1"].tolist()==[True,False,False,True]
    assert q["positives"].tolist()==[2,2,1,1]
    assert q["best_positive_rank"].tolist()==[1,2,4,1]
    assert z["R@1"]==50.

def test_native_ranking_exact_and_ties():
    p=torch.arange(12)//2
    s=torch.ones(12,12)
    q,z=m.query_metrics(s,p,m.native_rank())
    assert np.isfinite(q["AP"]).all()

def test_bootstrap_identical_is_zero():
    a=np.array([.1,.5,.2]);p=np.array([1,1,2])
    z=m.paired_bootstrap(a,a,p,n=200)
    assert z["difference_b_minus_a_pp"]==z["CI95_lower_pp"]==z["CI95_upper_pp"]==0
    assert z["clusters"]==2 and z["queries"]==3

def test_cluster_bootstrap_constant_delta_and_seed():
    a=np.array([.1,.5,.2]);b=a+.01;p=np.array([1,1,2])
    z=m.paired_bootstrap(a,b,p,n=200)
    assert np.isclose(z["CI95_lower_pp"],1) and np.isclose(z["CI95_upper_pp"],1)
    assert z==m.paired_bootstrap(a,b,p,n=200)

def test_action_STOP_not_NOOP_probability():
    a=dict(attributes=[dict(semantic_axis="gender",region="person",canonical_value="male",char_start=0,char_end=3,raw_text="man")],
           edits=[],adversarial="man",budget_k=2,decisions=[dict(head="attribute",index=1,legal_indices=[0,1])])
    x=m.Actions("fixture",1,"greedy");v=x.add(a,"man",dict(legal_attribute_count=1))
    row=x.finish()
    assert v["STOP"]==v["NOOP"]==1 and row["STOP_rollout_rate"]==1
    assert row["mean_edits"]==0 and row["mean_budget"]==2

def test_Unicode_action_replay_and_category():
    attrs=[dict(semantic_axis="gender",region="person",canonical_value="male",char_start=0,char_end=3,raw_text="man")]
    e=dict(position=1,source_span=[0,3],source_raw="man",original_char="a",replacement_char="а",
           original_codepoint="U+0061",replacement_codepoint="U+0430",semantic_axis="gender",
           region="person",canonical_value="male",position_in_attribute=1)
    a=dict(attributes=attrs,edits=[e],adversarial="mаn",budget_k=1,decisions=[])
    x=m.Actions("fixture",1,"sample");x.add(a,"man",dict(legal_attribute_count=1))
    assert x.h["axis_region"]["gender/person"]==1
    a["edits"][0]["original_codepoint"]="U+0062"
    with pytest.raises(ValueError):x.add(a,"man",dict(legal_attribute_count=1))

def test_input_change_rejected(tmp_path):
    p=tmp_path/"source.json";p.write_text("{}")
    t=m.Inputs();t.add(p);p.write_text("changed")
    with pytest.raises(ValueError):t.finish()

def test_missing_positive_rejected():
    with pytest.raises(ValueError):
        m.query_metrics(torch.ones(1,1),torch.tensor([1]))

def test_distribution_not_missing_zero():
    x=m.dist([0.,0.,1.,-1.])
    assert x["zero_count"]==2 and x["positive_count"]==x["negative_count"]==1
    with pytest.raises(ValueError):m.dist([])

def test_no_models_and_output_isolation():
    assert m.OUT!=m.REPAIR/"docs/framework/results/hv1_g60_adaptation_epoch60"
    assert not torch.cuda.is_initialized()
