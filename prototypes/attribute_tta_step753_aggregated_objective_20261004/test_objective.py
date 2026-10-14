import ast,inspect,copy
from types import SimpleNamespace
from pathlib import Path
import pytest,torch
import deps
from aggregated_ig import AggregatedCollector,fast_methods
from priority import MODES
from modes import execute_three,cached_guidance,BASE_STEP,ControlledTTA
from test_dynamic_k import fixture
from attribute_tta import seed_all,OFFICIAL,IMAGE_EPSILON
from controlled_tta import CategoricalTape
import prefix_tta as old
@pytest.mark.parametrize("k",[1,3,5,6,7])
@pytest.mark.parametrize("mode",MODES)
def test_objective_and_constant_node_backward_count(k,mode):
    gen=torch.Generator().manual_seed(42)
    members=[SimpleNamespace(instance_id=str(i),metadata={"S_raw":float(k-i),"semantic_axis":"color","region":"upper","canonical_value":"black"}) for i in range(k)]
    vs={str(i):torch.randn(1,7,generator=gen,dtype=torch.float64) for i in range(k)}
    x=torch.randn(1,3,2,2,generator=gen,dtype=torch.float64,requires_grad=True)
    A=torch.randn(x.numel(),7,generator=gen,dtype=x.dtype)
    c=AggregatedCollector(members,vs,mode);c.start_step();actual=0.;history=0.;new=0.
    scores=[m.metadata["S_raw"] for m in members]
    for kind,scale in [("scaled_ig",1.),("scaled_ig",.5),("transformed_ig",.75),("transformed_ig",1.25)]:
        f=torch.sin((x*scale).flatten()@A).unsqueeze(0);path=x.detach()*scale
        terms=torch.stack([-(f*vs[m.instance_id]).sum() for m in members])
        w=x.new_tensor([1/k]*k if mode==MODES[0] else [s/sum(scores) for s in scores])
        loss=(terms*w).sum();g=torch.autograd.grad(loss,x,retain_graph=True)[0]
        node=c.node(f,x,path,kind,loss);actual=actual+node*path
        if mode==MODES[2] and k>1:
            h=torch.autograd.grad((terms[:-1]*x.new_tensor([s/sum(scores[:-1]) for s in scores[:-1]])).sum(),x,retain_graph=True)[0]
            n=torch.autograd.grad(terms[-1],x,retain_graph=True)[0];history=history+h*path;new=new+n*path
        else:assert torch.equal(node,g)
    out=c.finish(actual);c.verify_update(out)
    assert c.backwards==4*(2 if mode==MODES[2] and k>1 else 1)
    if mode==MODES[2] and k>1:
        dot=(history*new).sum()
        repaired=new-dot/(history.square().sum()+torch.finfo(x.dtype).eps)*history if dot<0 else new
        expected=(sum(scores[:-1])/sum(scores))*history+(scores[-1]/sum(scores))*repaired
        assert torch.allclose(out,expected,atol=4e-13,rtol=4e-13)
    else:assert out is actual
@pytest.mark.parametrize("conflict",[False,True])
def test_asymmetric_projection_and_actual_update_identity(conflict):
    m=[SimpleNamespace(instance_id=str(i),metadata={"S_raw":s,"semantic_axis":"color","region":"upper","canonical_value":"black"}) for i,s in enumerate([5.,3.,1.])]
    c=AggregatedCollector(m,{str(i):torch.ones(1,2) for i in range(3)},MODES[2]);c.start_step()
    c.history=torch.tensor([1.,.2]);c.new=torch.tensor([-2.,1.] if conflict else [2.,1.])
    original_history=c.history.clone();original_new=c.new.clone();c.nodes={"scaled_ig":1,"transformed_ig":1}
    result=c.finish((8*c.history+c.new)/9);log=c.records[-1]
    assert log["conflict"]==conflict
    assert torch.equal(c.history,original_history) and torch.equal(c.new,original_new)
    if conflict:assert abs(log["cosine_after"])<64*torch.finfo(torch.float32).eps
    else:
        assert log["projected_new_unchanged_exact"] and log["projection_component_max_abs"]==0.
        assert torch.equal(c.snapshot["projected_g_N"],c.new)
    c.verify_update(result)
    with pytest.raises(ValueError):c.verify_update(result.clone())
def test_AST_only_replaces_gradient_nodes_and_preserves_TTA_rules():
    methods=fast_methods((Path(OFFICIAL)/"attacker_TTA.py").read_text(encoding="utf-8"))
    code="\n".join(ast.unparse(x) for x in methods.values())
    assert "_priority.capture" not in code and code.count("_priority.node(")==2
    for s in ["_priority.finish(ig_grad)","_priority.verify_update(ig_grad)","aug_param.add_(aug_grad)","grad.sign()","momentum * 0.9","origin_imgs - self.imgs_eps"]:
        assert s in code
def test_CPU_instrumentation_changes_observation_only():
    a=ast.parse(inspect.getsource(ControlledTTA.step).lstrip()).body[0]
    # BASE_STEP compiled from same function: source transformation is exposed and bounded.
    from modes import observed_step
    t=inspect.getsource(observed_step)
    assert 'self.make_observer()' in t and 'self.expected_observed_steps' in t
    assert 'torch.autograd.grad' not in t and 'step_size' not in t
@pytest.mark.parametrize("k",[1,2,3,5,7])
def test_original_PREFIX_baseline_and_shared_round1(k):
    plan,v,source,image,ev=fixture(k)
    data,checks=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    seed_all(42)
    reference,_,_=old.execute(source,image,plan,old.cached_round_guidance("PREFIX_NEW",plan,v),ev,
        steps=1,transforms=1,scales=None,tape=CategoricalTape())
    assert data[MODES[0]]["states"]==reference["states"]
    for a,b in zip(data[MODES[0]]["rounds"],reference["rounds"]):
        for key in ["post_projection_image_sha256","initialization","steps","metric_delta"]:assert a[key]==b[key]
    for mode in MODES:
        assert data[mode]["states"][:2]==data[MODES[0]]["states"][:2]
        assert data[mode]["random_trace"]==data[MODES[0]]["random_trace"]
        assert data[mode]["noise_sha256"]==data[MODES[0]]["noise_sha256"]
        assert data[mode]["rng_end_sha256"]==data[MODES[0]]["rng_end_sha256"]
        assert data[mode]["final_linf"]<=IMAGE_EPSILON+1e-6
        for e in data[mode]["rounds"]:
            assert e["prefix_attribute_instance_ids"]==[a.instance_id for a in plan[:e["round_index"]]]
            assert all(x["actual_momentum_input_verified"] for x in e["priority_steps"])
    assert checks["baseline_unmodified"] and not checks["slow_equivalence_gate"]
    if k==1:assert checks["k1_final_exact"]
def test_observer_short_smoke_keeps_ten_step_argument_and_full_plan():
    plan,v,source,image,ev=fixture(3)
    data,checks=execute_three(source,image,plan,v,ev,steps=10,transforms=1,scales=None,observed_limit=1)
    assert checks["configured_steps_per_round"]==10
    for mode in MODES:
        assert len(data[mode]["rounds"])==3
        for e in data[mode]["rounds"]:
            assert e["configured_steps_per_round"]==10 and e["observed_steps"]==1
            assert e["short_smoke_observer_limit"]==1
def test_same_state_no_conflict_does_not_modify_new_gradient():
    plan,v,source,image,ev=fixture(2);v[plan[1].instance_id]=v[plan[0].instance_id].clone()
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    e=data[MODES[2]]["rounds"][1]["priority_steps"][0]
    assert not e["conflict"] and e["projected_new_unchanged_exact"] and e["projection_component_max_abs"]==0.
def test_multiscale_actual_IG_budget_and_report_output():
    from report import summarize
    plan,v,source,image,ev=fixture(2)
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=2,scales=(.5,1.25))
    for mode in MODES:
        rec=data[mode]["rounds"][1]["priority_steps"][0]
        assert rec["IG_nodes"]=={"scaled_ig":3,"transformed_ig":6}
        assert rec["image_gradient_backward_calls"]==9*(2 if mode==MODES[2] else 1)
    summary,rounds,conflicts,projection=summarize([{"caption_id":"test","natural_k":2,"modes":data}])
    assert set(summary["modes"])==set(MODES)
    assert projection[MODES[2]]["all"]["no_conflict_exact_failures"]==0
    assert conflicts[MODES[1]]["all"]["measured_conflict_step_count"]==0
def test_frozen_sources_and_manifest():
    hashes=deps.frozen_hashes();assert len(hashes)>=148;deps.assert_formal_clean()
    reader=deps.module("step753_test_manifest",deps.P74/"run.py")
    manifest,_,_=reader.checked_manifest(deps.P74/"outputs/clean_scores/selection_manifest.json","diagnostic")
    assert len(manifest["rows"])==100 and sum(p["natural_k"] for p in manifest["rows"])==358
def test_no_automatic_full_before_model_loading():
    import run
    with pytest.raises(SystemExit):run.main(["--mode","diagnostic"])
    with pytest.raises(SystemExit):run.main(["--mode","diagnostic","--allow-diagnostic","--case-id","x"])
def test_formal_gates_do_not_require_slow_equivalence():
    text=(deps.ROOT/"run.py").read_text(encoding="utf-8")
    assert "saved_regression(" not in text and "SLOW_REFERENCE" not in text
    assert "score_instances(" not in text and "baseline_gradient(" not in text

def test_conflict_reaches_real_official_CPU_update_path():
    plan,v,source,image,ev=fixture(2)
    v[plan[1].instance_id]=-v[plan[0].instance_id]+torch.tensor([[.05,0.,0.]])
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    rec=data[MODES[2]]["rounds"][1]["priority_steps"][0]
    assert rec["conflict"] and rec["projection_intervention"] and rec["actual_momentum_input_verified"]
    assert abs(rec["cosine_after"])<64*torch.finfo(torch.float32).eps
    assert rec["projection_component_max_abs"]>0.
def test_unsupported_mode_or_reordered_scores_rejected():
    gen_members=[SimpleNamespace(instance_id=str(i),metadata={"S_raw":s,"semantic_axis":"color","region":"upper","canonical_value":"black"}) for i,s in enumerate([1.,2.])]
    c=AggregatedCollector(gen_members,{str(i):torch.ones(1,2) for i in range(2)},MODES[1]);c.start_step()
    x=torch.ones(1,2,requires_grad=True)
    with pytest.raises(ValueError):c.node(x,x,x.detach(),"scaled_ig",x.sum())
def test_constant_backward_is_not_a_loss_aggregation_slow_gate():
    text=(deps.ROOT/"aggregated_ig.py").read_text(encoding="utf-8")
    assert "IGCollector(" not in text and "is_grads_batched" not in text
    assert "grad((terms[:-1]*hw).sum())" in text and "grad(terms[-1])" in text
