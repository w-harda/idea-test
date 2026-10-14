import ast,copy,json,sys
from pathlib import Path
import pytest
import torch
import deps
from priority import MODES,normalized_weights,combine
from real_ig import instrument_methods
from modes import execute_three,cached_guidance
import prefix_tta as old
from controlled_tta import CategoricalTape
from attribute_tta import seed_all,OFFICIAL,IMAGE_EPSILON
from test_dynamic_k import fixture
@pytest.mark.parametrize("mode",MODES)
@pytest.mark.parametrize("scores",[[1.],[3.,2.,1.],[10.,.1,.001]])
def test_weights_frozen_raw_not_power(scores,mode):
    w=normalized_weights(scores,mode)
    assert sum(w)==pytest.approx(1)
    assert w==pytest.approx([1/len(scores)]*len(scores) if mode==MODES[0] else [s/sum(scores) for s in scores])
@pytest.mark.parametrize("scores",[[0.],[-1.],[float("nan")],[1.,2.],[]])
def test_invalid_scores_no_fallback(scores):
    with pytest.raises(ValueError):normalized_weights(scores,MODES[1])
def test_score_rank_and_identity_no_mutation():
    scores=[3.,1.];gs=[torch.tensor([1.,0.]),torch.tensor([0.,1.])]
    out,log=combine(gs,scores,MODES[1])
    assert torch.equal(out,torch.tensor([.75,.25]))
    assert log["S_raw"]==scores and log["normalized_weights"]==[.75,.25]
def test_baseline_keeps_original_ig_object_even_roundoff_difference():
    gs=[torch.tensor([1.,0.]),torch.tensor([0.,1.])];orig=torch.tensor([.5000001,.5])
    out,log=combine(gs,[2.,1.],MODES[0],orig)
    assert out is orig and log["baseline_original_tensor_identity"]
@pytest.mark.parametrize("new",[[0.,1.],[1.,1.],[0.,0.]])
def test_no_conflict_projection_exact_and_zero_intervention(new):
    gs=[torch.tensor([1.,0.]),torch.tensor(new)]
    weighted,_=combine(gs,[2.,1.],MODES[1])
    projected,log=combine(gs,[2.,1.],MODES[2])
    assert torch.equal(weighted,projected)
    assert not log["conflict"] and not log["projection_intervention"]
    assert log["removed_conflict_component_norm"]==0 and log["no_conflict_projection_exact"]
def test_conflict_only_new_history_unchanged_and_dot_near_zero():
    gs=[torch.tensor([1.,0.]),torch.tensor([-1.,2.])]
    copies=[g.clone() for g in gs]
    result,log=combine(gs,[3.,1.],MODES[2])
    assert log["conflict"] and log["projection_intervention"]
    assert abs(log["dot_after"])<1e-6
    assert torch.equal(result,torch.tensor([.75,.5]))
    assert all(torch.equal(a,b) for a,b in zip(gs,copies))
    assert log["projection_epsilon"]==torch.finfo(torch.float64).eps
def test_zero_history_finite_no_projection():
    gs=[torch.zeros(2),torch.ones(2)]
    out,log=combine(gs,[2.,1.],MODES[2])
    assert torch.isfinite(out).all() and not log["conflict"] and log["cosine_before"] is None
def test_ascent_loss_and_negative_dot_has_correct_sign():
    x=torch.tensor([1.,2.],requires_grad=True)
    first=-(x*torch.tensor([1.,0.])).sum()
    new=-(x*torch.tensor([-1.,1.])).sum()
    g1=torch.autograd.grad(first,x)[0];g2=torch.autograd.grad(new,x)[0]
    assert torch.dot(g1,g2)<0
    out,log=combine([g1,g2],[2.,1.],MODES[2])
    assert log["conflict"] and float((out*g1).sum())>0
def test_official_AST_all_original_statements_preserved():
    text=(Path(OFFICIAL)/"attacker_TTA.py").read_text(encoding="utf-8")
    methods,anchors=instrument_methods(text)
    tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="Attack")
    originals={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods}
    class Strip(ast.NodeTransformer):
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func).startswith(("self._priority.","self._audit.")):
                return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=="self._priority.finish":
                return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if ast.unparse(node.func)=="self._audit.capture_noise":return self.visit(node.args[0])
            return self.generic_visit(node)
    for name,method in methods.items():
        assert ast.dump(Strip().visit(copy.deepcopy(method)))==ast.dump(originals[name])
    assert anchors=={"scaled":1,"transformed":1,"start":1,"finish":1,"verify":1}
@pytest.mark.parametrize("k",[1,2,3,5,7])
def test_real_official_CPU_baseline_regression_and_shared_round1(k):
    plan,v,source,image,ev=fixture(k)
    data,checks=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    baseline=data[MODES[0]]
    seed_all(42)
    reference,_,_=old.execute(source,image,plan,old.cached_round_guidance("PREFIX_NEW",plan,v),ev,
                             steps=1,transforms=1,scales=None,tape=CategoricalTape())
    assert baseline["states"]==reference["states"]
    for a,b in zip(baseline["rounds"],reference["rounds"]):
        for key in ["post_projection_image_sha256","initialization","steps","metric_delta"]:
            assert a[key]==b[key]
    for mode in MODES:
        assert data[mode]["states"][:2]==baseline["states"][:2]
        assert data[mode]["random_trace"]==baseline["random_trace"]
        assert data[mode]["noise_sha256"]==baseline["noise_sha256"]
        assert data[mode]["rng_end_sha256"]==baseline["rng_end_sha256"]
        assert data[mode]["final_linf"]<=IMAGE_EPSILON+1e-6
        for event in data[mode]["rounds"]:
            assert event["prefix_attribute_instance_ids"]==[a.instance_id for a in plan[:event["round_index"]]]
            for s,obs in zip(event["priority_steps"],event["steps"]):
                assert s["actual_momentum_input_verified"]
                assert s["aggregate_gradient_norm"]==pytest.approx(obs["ig_gradient_l2"],rel=1e-10)
                if event["round_index"]>1:
                    assert s["IG_nodes"]=={"scaled_ig":1,"transformed_ig":1}
                    assert s["IG_reconstruction_relative_error"]<1e-5
    assert checks["round1_image_momentum_metrics_noise_transform_RNG_exact"]
    if k==1:assert all(data[m]["states"]==baseline["states"] for m in MODES)
def test_actual_no_conflict_step_weighted_projection_same_update():
    plan,v,source,image,ev=fixture(2);v[plan[1].instance_id]=v[plan[0].instance_id].clone()
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    w,p=data[MODES[1]],data[MODES[2]]
    assert w["states"]==p["states"]
    assert w["rounds"][1]["post_projection_image_sha256"]==p["rounds"][1]["post_projection_image_sha256"]
    assert p["rounds"][1]["priority_steps"][0]["no_conflict_projection_exact"]
def test_actual_conflict_step_projected_before_momentum():
    plan,v,source,image,ev=fixture(2);v[plan[1].instance_id]=-v[plan[0].instance_id]+torch.tensor([[.05,0.,0.]])
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    p=data[MODES[2]]["rounds"][1]["priority_steps"][0]
    assert p["conflict"] and p["projection_intervention"] and p["actual_momentum_input_verified"]
    assert abs(p["cosine_after"] or 0)<1e-5
    assert p["history_components_unchanged"] and p["only_new_attribute_modified"]
def test_fixed_full_manifest_gate_without_reselection():
    reader=deps.module("step75_test74_run",deps.P74/"run.py")
    manifest,_,records=reader.checked_manifest(deps.P74/"outputs/clean_scores/selection_manifest.json","diagnostic")
    assert len(manifest["rows"])==100 and sum(p["natural_k"] for p in manifest["rows"])==358
    assert sum(p["natural_k"]>4 for p in manifest["rows"])==35
    for cid in ["cuhk:25488:1","cuhk:23198:1","cuhk:14818:1","cuhk:24308:1"]:
        assert cid in records
def test_user_only_full_before_loading_model():
    import run
    with pytest.raises(SystemExit):run.main(["--mode","diagnostic"])
    with pytest.raises(SystemExit):run.main(["--mode","diagnostic","--allow-diagnostic","--case-id","x"])
def test_runner_never_scores_or_imports_caption_guidance():
    text=(Path(__file__).parent/"run.py").read_text(encoding="utf-8")
    assert all(x not in text for x in ["baseline_gradient(","score_instances(","scoring_runner(","FrozenLocalization("])
def test_frozen_sources_SHA_and_formal_diff():
    hashes=deps.frozen_hashes();assert len(hashes)>=119
    deps.assert_formal_clean()
def test_report_output_empty_conflicts_and_three_modes():
    from report import summarize
    plan,v,source,image,ev=fixture(2)
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    cs=[{"caption_id":"test","natural_k":2,"modes":data}]
    summary,rounds,conflicts,projection=summarize(cs)
    assert set(summary["modes"])==set(MODES)
    assert set(rounds)==set(MODES) and conflicts[MODES[2]]["all"]["step_count"]==2
    assert projection[MODES[2]]["all"]["no_conflict_exact_failures"]==0

def test_multiscale_transformed_batch_real_components():
    plan,v,source,image,ev=fixture(2)
    data,_=execute_three(source,image,plan,v,ev,steps=1,transforms=2,scales=(.5,1.25))
    for mode in MODES:
        entry=data[mode]["rounds"][1]["priority_steps"][0]
        assert entry["IG_nodes"]=={"scaled_ig":3,"transformed_ig":6}
        assert entry["IG_reconstruction_relative_error"]<1e-5
        assert entry["actual_momentum_input_verified"]
