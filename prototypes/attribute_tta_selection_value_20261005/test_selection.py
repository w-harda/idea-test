"""选择集合与原PREFIX控制变量的针对性CPU测试。"""
import copy,random,sys
from dataclasses import replace
from pathlib import Path
import pytest,torch
import deps
from selection import choose,mode_payload,verify_row,build,MODES
from selection_audit import selection,digest,verify,load
_saved_path=sys.path[:]
_fixture=deps.module("selection_value_test_fixture",deps.P74/"test_dynamic_k.py")
sys.path[:]=_saved_path
fixture=_fixture.fixture
record=_fixture.record
def payload(m=6,k=3):
    p=selection(record(m))
    p["natural_k"]=p["k_star"]=k;p["capped_k"]=min(k,4)
    p["selected_attributes"]=p["ranked_attributes"][:k]
    p.update(choose(p))
    return p
@pytest.mark.parametrize("m,k",[(3,1),(4,2),(6,3),(8,5),(9,7)])
def test_same_K_candidate_source_and_descending_subset(m,k):
    p=payload(m,k);original=copy.deepcopy(p)
    for mode in MODES:
        ids=p["selection_ids"][mode]
        assert len(ids)==k and len(set(ids))==k
        assert set(ids)<=set(a["instance_id"] for a in p["ranked_attributes"])
        selected=mode_payload(p,mode)["selected_attributes"]
        assert selected==[a for a in p["ranked_attributes"] if a["instance_id"] in ids]
    assert p==original
def test_top_bottom_exact_and_tie_order():
    p=payload(6,3)
    assert p["selection_ids"][MODES[0]]==["0","1","2"]
    assert p["selection_ids"][MODES[2]]==["3","4","5"]
def test_random_reproducible_and_does_not_consume_attack_rng():
    p=payload();state=random.getstate()
    assert choose(p)==choose(copy.deepcopy(p))
    assert random.getstate()==state
def test_random_seed_depends_on_caption_id():
    p=payload();q=copy.deepcopy(p);q["caption_id"]="other"
    assert choose(p)["random_seed_hex"]!=choose(q)["random_seed_hex"]
def test_overlap_not_forced_disjoint():
    p=payload(5,4)
    assert p["set_overlap"][MODES[0]+"_vs_"+MODES[2]]["intersection_count"]==3
@pytest.mark.parametrize("m,k",[(3,0),(3,3),(3,4)])
def test_noncomparative_rows_rejected(m,k):
    with pytest.raises(ValueError):payload(m,k)
def test_duplicate_candidate_rejected():
    p=payload();p["ranked_attributes"][1]["instance_id"]="0"
    with pytest.raises(ValueError):choose(p)
def test_frozen_metadata_and_score_gate():
    r=record(6);p=payload(6,3);verify_row(p,r)
    for field,value in [("S_raw",9.),("region","lower"),("canonical_value","changed")]:
        q=copy.deepcopy(p);q["ranked_attributes"][0][field]=value
        with pytest.raises(ValueError):verify_row(q,r)
def test_manifest_selection_mutation_rejected():
    r=record(6);p=payload(6,3)
    p["selection_ids"][MODES[1]]=p["selection_ids"][MODES[1]][::-1]
    with pytest.raises(ValueError):verify_row(p,r)
def test_unresolved_and_unsupported_excluded():
    r=record(4)
    r["attributes"][0]["status"]="unresolved";r["attributes"][0]["instance"]["unresolved"]=True
    r["attributes"][1]["status"]="score_unsupported";r["attributes"][1]["rank"]=0
    assert [a["instance_id"] for a in selection(r)["ranked_attributes"]]==["2","3"]
def test_repeated_values_different_spans_are_distinct_instances():
    p=payload(6,3)
    assert len({a["canonical_value"] for a in p["ranked_attributes"]})==1
    assert len({a["source_span"]["start"] for a in p["ranked_attributes"]})==6
def test_sha_modification_rejected(tmp_path):
    p=tmp_path/"x";p.write_text("x");h=digest(p);p.write_text("y")
    with pytest.raises(ValueError):verify({str(p):h})
def test_no_full_before_model_loading():
    import run
    with pytest.raises(SystemExit):run.main(["--mode","diagnostic"])
def test_cache_only_contract_in_source():
    text=(deps.ROOT/"run.py").read_text()
    assert "baseline_gradient(" not in text and "score_instances(" not in text
    assert "prepare_scores" not in text
@pytest.mark.parametrize("k",[1,2,3,5,7])
def test_original_PREFIX_regression_and_rng_control(k):
    from engines import execute_three
    from controlled_tta import CategoricalTape
    from attribute_tta import seed_all,IMAGE_EPSILON
    import prefix_tta as old
    plan,v,source,image,ev=fixture(k+2)
    subsets=[plan[:k],plan[1:k+1],plan[-k:]]
    plans={m:tuple(replace(a,selection_rank=i) for i,a in enumerate(sub,1)) for m,sub in zip(MODES,subsets)}
    data,checks=execute_three(source,image,plans,v,ev,steps=1,transforms=1,scales=None)
    seed_all(42)
    ref,_,_=old.execute(source,image,plans[MODES[0]],old.cached_round_guidance("PREFIX_NEW",plans[MODES[0]],v),ev,
                       steps=1,transforms=1,scales=None,tape=CategoricalTape())
    assert data[MODES[0]]["states"]==ref["states"]
    assert [e["post_projection_image_sha256"] for e in data[MODES[0]]["rounds"]]==[e["post_projection_image_sha256"] for e in ref["rounds"]]
    assert checks["noise_transform_RNG_exact"]
    for m in MODES:
        assert data[m]["fixed_sequence"]==[a.instance_id for a in plans[m]]
        assert data[m]["final_linf"]<=IMAGE_EPSILON+1e-6
        assert data[m]["noise_sha256"]==data[MODES[0]]["noise_sha256"]
        assert data[m]["random_trace"]==data[MODES[0]]["random_trace"]
        assert data[m]["rng_end_sha256"]==data[MODES[0]]["rng_end_sha256"]
        for e in data[m]["rounds"]:
            assert not e["score_weighting"] and not e["priority_projection"] and e["caption_loss_terms"]==0
def test_identical_sequence_reuses_actual_trajectory():
    from engines import execute_three
    plan,v,source,image,ev=fixture(3)
    data,_=execute_three(source,image,{m:plan for m in MODES},v,ev,steps=1,transforms=1,scales=None)
    for m in MODES:
        assert data[m]["states"]==data[MODES[0]]["states"]
    assert data[MODES[1]]["execution_reuse_of"]==MODES[0]
def test_short_smoke_preserves_configuration_and_all_rounds():
    from engines import execute_three
    plan,v,source,image,ev=fixture(3)
    data,_=execute_three(source,image,{m:plan for m in MODES},v,ev,steps=10,transforms=2,scales=(.5,1.25),observed_limit=1)
    for m in MODES:
        assert len(data[m]["rounds"])==3
        for e in data[m]["rounds"]:
            assert e["configured_steps_per_round"]==10 and len(e["steps"])==1
def test_report_final_early_rounds_and_K_groups():
    from engines import execute_three
    from report import summarize
    plan,v,source,image,ev=fixture(2)
    data,_=execute_three(source,image,{m:plan for m in MODES},v,ev,steps=1,transforms=1,scales=None)
    p=payload(3,2)
    summary,rounds=summarize([{"natural_k":2,"modes":data,"set_overlap":p["set_overlap"]}])
    assert summary["comparisons"][MODES[0]+"_vs_"+MODES[1]]["paired_soft_rank_F"]["ties"]==1
    assert set(rounds)=={"1","2"} and summary["by_K"]["2"]["count"]==1
def test_frozen_manifest_sources_and_natural_K():
    hashes=deps.frozen_hashes();assert len(hashes)>=157;deps.assert_formal_clean()
    m=load(deps.ROOT/"outputs/manifest/selection_manifest.json")
    assert m["case_count"]==100 and m["score_cache_caption_count"]==13700
    assert all(p["candidate_count"]>p["natural_k"] for p in m["rows"])
    for p in m["rows"]:assert choose(p)["selection_ids"]==p["selection_ids"]

def test_manifest_deterministic_across_cache_iteration_order():
    rs=[record(m,cid="cuhk:"+str(i)+":0",pid=i) for i,m in enumerate([4,5,6,7],1)]
    for r in rs:
        r["retrieval_backward_count"]=1;r["zero_intervention_exact"]=True
        for i,a in enumerate(r["attributes"]):a["S_raw"]=10. if i==0 else 1.
    validation={r["row_id"]:r for r in rs}
    a=build(iter(rs),validation,{"rows":[]},target=3)
    b=build(iter(rs[::-1]),validation,{"rows":[]},target=3)
    assert a["rows"]==b["rows"] and a["K_distribution"]==b["K_distribution"]
def test_manifest_rejects_PID_source_mismatch():
    r=record(4);r["retrieval_backward_count"]=1;r["zero_intervention_exact"]=True
    v=copy.deepcopy(r);v["pid"]=-1
    with pytest.raises(ValueError):build(iter([r]),{r["row_id"]:v},{"rows":[]},target=1)
