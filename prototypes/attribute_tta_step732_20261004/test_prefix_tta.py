import copy, importlib.util, json, sys
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent))
import prefix_tta as p
from attribute_tta import SelectedAttribute, seed_all, digest, IMAGE_EPSILON, FixedGuidanceBridge
from controlled_tta import ControlledTTA, CategoricalTape, RetrievalMetrics, rng_fingerprint
from attributes.tta_adapter import load_official_attack
from attribute_tta import OFFICIAL
class Toy(nn.Module):
    def __init__(self):super().__init__();self.text_calls=0
    def encode_images(self,x):return F.normalize(x.mean((-2,-1)),dim=-1)
    def inference_image(self,x):return {"image_feat":self.encode_images(x)}
    def inference_text(self,*a):self.text_calls+=1;raise AssertionError("caption/raw forbidden")
def fixture(k):
    plan=tuple(SelectedAttribute({"instance_id":str(i),"semantic_axis":"color","region":"upper",
         "canonical_value":str(i),"S_raw":100-i,"source_span":{"start":i,"end":i+1,"raw":"x"}},
         i,torch.ones(1),torch.ones(1,1)) for i in range(1,k+1))
    values=[[.2,.5,-.3],[-.4,.3,.1],[.7,-.1,.2],[.1,.2,.6]]
    vectors={a.instance_id:torch.tensor([values[i]]) for i,a in enumerate(plan)}
    source=Toy();image=torch.linspace(.2,.8,3*16*16).reshape(1,3,16,16)
    evaluator=RetrievalMetrics(source,F.normalize(torch.tensor([[1.,2.,3.]]),dim=-1),
                              F.normalize(torch.tensor([[1.,2.,3.],[3.,1.,2.]]),dim=-1),0)
    return plan,vectors,source,image,evaluator
@pytest.mark.parametrize("k",[1,2,3,4])
def test_prefix_members_and_original_vectors_not_mutated(k):
    plan,v,_,_,_=fixture(k)
    before=copy.deepcopy([a.metadata for a in plan]);vs={key:x.clone() for key,x in v.items()}
    cached=p.cached_round_guidance("PREFIX_NEW",plan,v)
    for i,(members,aggregate) in enumerate(cached,1):
        assert [a.instance_id for a in members]==[a.instance_id for a in plan[:i]]
        assert torch.equal(aggregate,torch.stack([v[a.instance_id] for a in members]).mean(0))
    assert [a.metadata for a in plan]==before
    assert all(torch.equal(v[k],x) for k,x in vs.items())
def test_no_unit_normalization_or_score_weighting():
    plan,v,_,_,_=fixture(2)
    actual=p.cached_round_guidance("PREFIX_NEW",plan,v)[-1][1]
    for a in plan:a.metadata["S_raw"]=1e9
    assert torch.equal(actual,p.cached_round_guidance("PREFIX_NEW",plan,v)[-1][1])
    assert not torch.allclose(actual.norm(),torch.tensor(1.))
@pytest.mark.parametrize("k",[1,2,3,4])
def test_loss_is_exact_unweighted_mean_and_old_terms_have_gradients(k):
    plan,v,source,image,_=fixture(k)
    members,aggregate=p.cached_round_guidance("PREFIX_NEW",plan,v)[-1]
    x=image.clone().requires_grad_(True);f=source.encode_images(x)
    terms=torch.stack([-(f*v[a.instance_id]).sum() for a in members])
    loss=-(f*aggregate).sum()
    assert torch.allclose(loss,terms.mean(),atol=1e-7,rtol=1e-6)
    g,=torch.autograd.grad(loss,x)
    out=p.gradient_decomposition(source,image,members,v)
    assert out["prefix_aggregate_image_gradient_norm"]==pytest.approx(float(g.double().norm()),rel=1e-5)
    assert all(t["unweighted_image_gradient_norm"]>0 and t["mean_weighted_image_gradient_contribution_norm"]>0 for t in out["attribute_terms"])
    assert out["mean_gradient_reconstruction_relative_error"]<1e-5
    assert out["prefix_total_loss"]/k==pytest.approx(out["prefix_mean_loss"],rel=1e-6)
def test_official_loss_multiscale_mean_equivalence():
    plan,v,_,_,_=fixture(4)
    f=F.normalize(torch.arange(1.,16.).reshape(5,3),dim=-1)
    attack=load_official_attack(OFFICIAL)
    for i in [1,2,3,4]:
        vec=[v[a.instance_id] for a in plan[:i]]
        mean=torch.stack(vec).mean(0)
        for fn in ("loss_func","loss_func1"):
            if fn=="loss_func":
                expected=torch.stack([attack.loss_func(None,z,f,[0]) for z in vec]).mean(0)
                actual=attack.loss_func(None,mean,f,[0])
            else:
                expected=torch.stack([attack.loss_func1(None,f,z,[0]) for z in vec]).mean(0)
                actual=attack.loss_func1(None,f,mean,[0])
            assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)
@pytest.mark.parametrize("k",[1,2,3,4])
def test_official_sequential_exact_prefix_r1_budget_rng(k):
    plan,v,source,image,evaluator=fixture(k)
    seed_all(42);tape=CategoricalTape()
    seq,inputs,momentum=p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,v),
        evaluator,steps=2,transforms=1,scales=None,tape=tape)
    seed_all(42);old=ControlledTTA(source,image,"NEW",steps=2,transforms=1,scales=None,tape=CategoricalTape())
    cur=image.clone()
    for a in plan:cur=old.step(cur,a,v[a.instance_id],["attribute_unit:"+a.instance_id])
    assert [e["post_projection_image_sha256"] for e in old.events]==[e["post_projection_image_sha256"] for e in seq["rounds"]]
    assert torch.equal(old.momentum,momentum) and rng_fingerprint()==seq["rng_end_sha256"]
    seed_all(42);replay=CategoricalTape(tape.records)
    prefix,pi,pm=p.execute(source,image,plan,p.cached_round_guidance("PREFIX_NEW",plan,v),
        evaluator,steps=2,transforms=1,scales=None,tape=replay)
    for key in ("random_trace","noise_sha256","rng_end_sha256"):
        assert prefix[key]==seq[key]
    assert prefix["states"][1]==seq["states"][1]
    assert prefix["rounds"][0]["post_projection_image_sha256"]==seq["rounds"][0]["post_projection_image_sha256"]
    if k==1:assert torch.equal(pm,momentum)
    p.budget_and_prefix_checks("SEQUENTIAL_NEW",seq,plan,IMAGE_EPSILON)
    p.budget_and_prefix_checks("PREFIX_NEW",prefix,plan,IMAGE_EPSILON)
    p.attach_decompositions(source,prefix,pi,p.cached_round_guidance("PREFIX_NEW",plan,v),v)
    assert source.text_calls==0
def test_frozen_source_cache_SHA_and_fixed25():
    spec=importlib.util.spec_from_file_location("prefix_runner_test",Path(__file__).parent/"run.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    hashes,rows,payloads,bases,protocol,history=m.inputs()
    assert len(history)==25 and sum(c["k_star"] for c in history.values())==75
    assert hashes[str(p.P73/"controlled_tta.py")]==digest(p.P73/"controlled_tta.py")
def test_unprotected_full_rejected_and_short_smoke_fixed():
    spec=importlib.util.spec_from_file_location("prefix_runner_boundary",Path(__file__).parent/"run.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.parser().parse_args([]).mode=="smoke" and len(m.SMOKE_IDS)==2
    with pytest.raises(SystemExit):m.main(["--mode","diagnostic"])
    with pytest.raises(SystemExit):m.parser().parse_args(["--mode","full"])
def test_saved_regression_rejects_changed_image_metrics_rng():
    plan,v,source,image,ev=fixture(1)
    seed_all(42)
    seq,_,_=p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,v),ev,
        steps=1,transforms=1,scales=None,tape=CategoricalTape())
    assert p.saved_regression(seq,seq)["saved_metrics_exact"]
    for field in ("states","rounds","noise_sha256"):
        prior=copy.deepcopy(seq)
        if field=="states":prior[field][-1]["paired_soft_rank_F"]+=1
        elif field=="rounds":prior[field][0]["post_projection_image_sha256"]="changed"
        else:prior[field]=["changed"]
        with pytest.raises(ValueError):p.saved_regression(seq,prior)
def test_invalid_mode_round_order_fail():
    plan,v,_,_,_=fixture(2)
    with pytest.raises(ValueError):p.round_members("RAW",plan,1)
    with pytest.raises(ValueError):p.round_members("PREFIX_NEW",plan,0)
    with pytest.raises(ValueError):p.round_members("PREFIX_NEW",plan[::-1],1)

@pytest.mark.parametrize("key",["paired_cosine","paired_soft_rank_F"])
def test_saved_metrics_one_fp32_ulp_allowed_with_images_rng_exact(key):
    plan,v,source,image,ev=fixture(1);seed_all(42)
    seq,_,_=p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,v),ev,
        steps=1,transforms=1,scales=None,tape=CategoricalTape())
    other=copy.deepcopy(seq)
    base=torch.tensor(seq["states"][1][key],dtype=torch.float32)
    other["states"][1][key]=float(torch.nextafter(base,torch.tensor(float("inf"))))
    report=p.saved_regression(other,seq)
    assert report["saved_round_images_bitwise_exact"] and report["saved_noise_trace_rng_exact"]
    assert report["saved_metrics_allclose_existing_fp32_tolerance"] and not report["saved_metrics_exact"]
    assert report["metric_differences"][0]["field"]=="states[1]."+key
@pytest.mark.parametrize("key",["positive_image_rank","positive_image_rank_tie_count","state_index"])
def test_discrete_state_fields_stay_exact(key):
    seq={"fixed_sequence":[],"rounds":[],"states":[{"state_index":0,"paired_cosine":.5,
        "paired_soft_rank_F":1.5,"positive_image_rank":1,"positive_image_rank_tie_count":0}],
        "random_trace":{},"noise_sha256":[],"rng_end_sha256":"same"}
    other=copy.deepcopy(seq);other["states"][0][key]+=1
    with pytest.raises(p.RegressionMismatch) as exc:p.saved_regression(other,seq)
    assert any(x["field"]=="states[0]."+key for x in exc.value.report["failures"])
@pytest.mark.parametrize("key",["fixed_sequence","noise_sha256","rng_end_sha256","random_trace"])
def test_scheduler_rng_checks_do_not_get_float_tolerance(key):
    seq={"fixed_sequence":[],"rounds":[],"states":[],"random_trace":{},"noise_sha256":[],"rng_end_sha256":"same"}
    other=copy.deepcopy(seq);other[key]="changed"
    with pytest.raises(p.RegressionMismatch):p.saved_regression(other,seq)
def test_regression_report_precise_numeric_field_and_error():
    seq={"fixed_sequence":[],"rounds":[],"states":[{"state_index":0,"paired_cosine":.5,
        "paired_soft_rank_F":1.5,"positive_image_rank":1,"positive_image_rank_tie_count":0}],
        "random_trace":{},"noise_sha256":[],"rng_end_sha256":"same"}
    other=copy.deepcopy(seq);other["states"][0]["paired_cosine"]+=.01
    with pytest.raises(p.RegressionMismatch) as exc:p.saved_regression(other,seq)
    f=exc.value.report["failures"][0]
    assert f["field"]=="states[0].paired_cosine"
    assert f["absolute_error"]==pytest.approx(.01) and f["relative_error"]==pytest.approx(.02)

def test_audited_old_reference_variant_accepts_only_observed_snapshot():
    spec=importlib.util.spec_from_file_location("own_prefix_policy_runner",Path(__file__).parent/"run.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    hashes,rows,payloads,bases,protocol,history=runner.inputs()
    base=history["cuhk:21:0"]["modes"]["NEW"]
    vs=runner.load(Path(__file__).parent/"regression_reference_variants.json")["variants"]["cuhk:21:0"]
    observed=vs[0]["reference"]
    with pytest.raises(p.RegressionMismatch):p.saved_regression(observed,base)
    report=p.saved_regression(observed,base,vs)
    assert report["accepted_verified_reference_variant"]
    assert not report["historical_reference_comparison"]["saved_round_images_bitwise_exact"]
    assert report["accepted_reference_comparison"]["saved_round_images_bitwise_exact"]
    assert p.saved_regression(base,base,vs)["saved_metrics_exact"]
def test_reference_variant_does_not_accept_unknown_image_or_changed_rng():
    spec=importlib.util.spec_from_file_location("own_prefix_policy_runner",Path(__file__).parent/"run.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    _,_,_,_,_,history=runner.inputs()
    base=history["cuhk:21:0"]["modes"]["NEW"]
    vs=runner.load(Path(__file__).parent/"regression_reference_variants.json")["variants"]["cuhk:21:0"]
    for field in ["image","rng"]:
        bad=copy.deepcopy(vs[0]["reference"])
        if field=="image":bad["rounds"][-1]["post_projection_image_sha256"]="unseen-image"
        else:bad["rng_end_sha256"]="changed-rng"
        with pytest.raises(p.RegressionMismatch):p.saved_regression(bad,base,vs)
def test_prefix_method_AST_unchanged_by_regression_fix():
    import ast,hashlib
    root=Path(__file__).parent
    proof=json.loads((root/"outputs/regression_scheduler_unchanged.json").read_text())
    tree=ast.parse((root/"prefix_tta.py").read_text())
    text=(root/"prefix_tta.py").read_text()
    now={n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()
         for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in proof["source_before"]}
    assert {k:v for k,v in now.items() if k!="execute"}=={k:v for k,v in proof["source_before"].items() if k!="execute"}
    # execute仅授权新增公共round1状态分支；prefix pooling与其余数学函数仍锁原SHA。

@pytest.mark.parametrize("k",[1,2,3,4])
def test_common_first_round_checkpoint_exact_complete_state_and_rng(k):
    plan,v,source,image,evaluator=fixture(k)
    capture=[];seed_all(42);tape=CategoricalTape()
    seq,si,sm=p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,v),
        evaluator,steps=2,transforms=1,scales=None,tape=tape,first_round_capture=capture)
    assert len(capture)==1
    before=tape.records[:capture[0]["tape_position"]]
    assert before==capture[0]["tape_records"]
    cp=capture[0];momentum=cp["momentum"].clone();out=cp["image"].clone()
    # 分支恢复的是公共round1实际RNG；当前ambient RNG不得决定PREFIX续跑。
    torch.rand(17)
    replay=CategoricalTape(tape.records)
    pre,pi,pm=p.execute(source,image,plan,p.cached_round_guidance("PREFIX_NEW",plan,v),
        evaluator,steps=2,transforms=1,scales=None,tape=replay,first_round_checkpoint=cp)
    assert pre["rounds"][0]==seq["rounds"][0]
    assert pre["states"][:2]==seq["states"][:2]
    assert torch.equal(cp["image"],out) and torch.equal(cp["momentum"],momentum)
    assert all(pre[key]==seq[key] for key in ("random_trace","noise_sha256","rng_end_sha256","fixed_sequence"))
    if k==1:
        assert pre==seq and torch.equal(pm,sm)
    p.budget_and_prefix_checks("PREFIX_NEW",pre,plan,IMAGE_EPSILON)

def test_common_first_round_runs_original_attack_only_once(monkeypatch):
    plan,v,source,image,ev=fixture(3);count=[]
    base=p.ControlledTTA
    class Count(base):
        def step(self,current,a,features,labels):
            count.append(a.selection_rank)
            return super().step(current,a,features,labels)
    monkeypatch.setattr(p,"ControlledTTA",Count)
    cp=[];seed_all(42);tape=CategoricalTape()
    p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,v),ev,
        steps=1,transforms=1,scales=None,tape=tape,first_round_capture=cp)
    p.execute(source,image,plan,p.cached_round_guidance("PREFIX_NEW",plan,v),ev,
        steps=1,transforms=1,scales=None,tape=CategoricalTape(tape.records),first_round_checkpoint=cp[0])
    assert count==[1,2,3,2,3]

@pytest.mark.parametrize("field",["guidance","tape","settings"])
def test_common_first_round_rejects_changed_conditions(field):
    plan,v,source,image,ev=fixture(2);cp=[];seed_all(42);tape=CategoricalTape()
    p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,v),ev,
        steps=1,transforms=1,scales=None,tape=tape,first_round_capture=cp)
    bad=dict(cp[0]);replay=CategoricalTape(tape.records)
    if field=="guidance":bad["guidance"]=cp[0]["guidance"]+.1
    elif field=="settings":bad["settings"]=(2,1,None)
    else:replay.records[0]["ids"]=[999]
    with pytest.raises(ValueError):
        p.execute(source,image,plan,p.cached_round_guidance("PREFIX_NEW",plan,v),ev,
            steps=1,transforms=1,scales=None,tape=replay,first_round_checkpoint=bad)
