"""仅CPU回放已完成5次35:1，验证保守gate；不再运行GPU。"""
import copy,json,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import prefix_tta as p
from repeatability_gate import load_reference,compare_numerical_repeatability
@pytest.fixture(scope="module")
def records():
    ref=load_reference();out=ROOT/"outputs/repeatability_35"
    prior=json.loads((ROOT/"outputs/diagnostic_round1_fixed/regression_failure.json").read_text())["saved_NEW"]
    rows=[json.loads((out/f"run{i}.json").read_text()) for i in range(1,6)]
    images=[]
    for i in range(1,6):
        t=torch.load(out/f"run{i}_tensors.pt",weights_only=True,map_location="cpu")
        images.append([t["initial_image"]]+t["round_outputs"])
    return ref,prior,rows,images

@pytest.mark.parametrize("index",[2,3,4])
def test_real_repeats_matching_history_round1_pass(records,index):
    ref,prior,rows,images=records
    checks=p.saved_regression(rows[index],prior,numerical_reference=ref,current_images=images[index],execution_contract=ref["manifest"]["execution_contract"])
    if index==2:assert checks["saved_round_images_bitwise_exact"]
    else:
        assert checks["accepted_empirical_repeatability"]
        assert checks["repeatability_comparison"]["image_comparison"]["coordinate_envelope_checked"]

@pytest.mark.parametrize("index",[0,1])
def test_larger_early_numerical_branches_remain_rejected(records,index):
    ref,prior,rows,images=records
    with pytest.raises(p.RegressionMismatch) as exc:
        p.saved_regression(rows[index],prior,numerical_reference=ref,current_images=images[index],execution_contract=ref["manifest"]["execution_contract"])
    assert "round1 outside historical exact condition" in exc.value.report["numerical_reference_rejected"]["failures"]

def test_original_failed_case_inside_observed_conditional_envelope(records):
    ref,prior,rows,images=records
    failed=json.loads((ROOT/"outputs/diagnostic_round1_fixed/regression_failure.json").read_text())["current_sequential"]
    assert failed["states"]==rows[4]["states"]
    assert [e["post_projection_image_sha256"] for e in failed["rounds"]]==[e["post_projection_image_sha256"] for e in rows[4]["rounds"]]
    assert p.saved_regression(failed,prior,numerical_reference=ref,current_images=images[4],execution_contract=ref["manifest"]["execution_contract"])["accepted_empirical_repeatability"]

@pytest.mark.parametrize("field",["fixed_sequence","noise_sha256","random_trace","rng_end_sha256","rank","steps","metric"])
def test_exact_controls_and_outside_range_rejected(records,field):
    ref,prior,rows,images=records;bad=copy.deepcopy(rows[4])
    if field in ("fixed_sequence","noise_sha256"):bad[field][0]="changed"
    elif field=="random_trace":bad[field]["categorical_draw_count"]+=1
    elif field=="rng_end_sha256":bad[field]="changed"
    elif field=="rank":bad["states"][2]["positive_image_rank"]+=1
    elif field=="steps":bad["rounds"][1]["steps"].pop()
    else:bad["states"][2]["paired_soft_rank_F"]=ref["manifest"]["metric_bounds"]["paired_soft_rank_F"]["max"]+1
    with pytest.raises(p.RegressionMismatch):
        p.saved_regression(bad,prior,numerical_reference=ref,current_images=images[4],execution_contract=ref["manifest"]["execution_contract"])

@pytest.mark.parametrize("kind",["unobserved_coordinate","observed_coordinate_outside_range","missing_tensor","epsilon"])
def test_no_blanket_image_tolerance_and_budget_changes(records,kind):
    ref,prior,rows,images=records
    actual=[x.clone() for x in images[4]];contract=dict(ref["manifest"]["execution_contract"])
    if kind=="unobserved_coordinate":actual[2][0,0,0,0]+=.001
    elif kind=="observed_coordinate_outside_range":
        idx=(ref["tensors"]["lower"]!=ref["tensors"]["upper"]).nonzero()[0].tolist()
        actual[2][tuple(idx)]=ref["tensors"]["upper"][tuple(idx)]+.001
    elif kind=="missing_tensor":actual=None
    else:contract["epsilon"]*=2
    with pytest.raises(p.RegressionMismatch):
        p.saved_regression(rows[4],prior,numerical_reference=ref,current_images=actual,execution_contract=contract)

def test_algorithm_AST_and_frozen_provenance(records):
    import ast
    def fn(s):return {x.name:x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef)}
    old=fn((ROOT/"outputs/repeatability_prefix_before.py.txt").read_text())
    new=fn((ROOT/"prefix_tta.py").read_text())
    f=new["execute"];i=next(i for i,x in enumerate(f.args.kwonlyargs) if x.arg=="final_image_capture")
    f.args.kwonlyargs.pop(i);f.args.kw_defaults.pop(i)
    f.body=[x for x in f.body if not (isinstance(x,ast.If) and ast.unparse(x.test)=="final_image_capture is not None")]
    assert ast.dump(f)==ast.dump(old["execute"])
    for name in ("round_members","cached_round_guidance","gradient_decomposition","budget_and_prefix_checks"):
        assert ast.dump(new[name])==ast.dump(old[name])
    import importlib.util
    spec=importlib.util.spec_from_file_location("repeatability_runner",ROOT/"run.py")
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    hashes,*_=run.inputs()
    assert str(ROOT/"repeatability_reference_35.json") in hashes
    assert str(ROOT/"repeatability_reference_35.pt") in hashes

@pytest.mark.parametrize("field",["epsilon","steps","metadata"])
def test_exact_saved_SHA_does_not_bypass_controls(records,field):
    ref,prior,rows,images=records;bad=copy.deepcopy(rows[2]);contract=dict(ref["manifest"]["execution_contract"])
    if field=="epsilon":contract["epsilon"]*=2
    elif field=="steps":bad["rounds"][1]["steps"].pop()
    else:bad["rounds"][1]["S_raw"]+=1
    with pytest.raises(p.RegressionMismatch):
        p.saved_regression(bad,prior,numerical_reference=ref,current_images=images[2],execution_contract=contract)
