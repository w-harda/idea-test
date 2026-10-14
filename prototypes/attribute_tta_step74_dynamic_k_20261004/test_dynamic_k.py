import copy,json,sys
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent))
import selection_audit as a
import deps
import paired_prefix as pair
import prefix_tta as old
from attribute_tta import SelectedAttribute,seed_all,IMAGE_EPSILON
from controlled_tta import CategoricalTape,RetrievalMetrics
def record(k,cid="cuhk:100:0",pid=100):
    attrs=[]
    for i in range(k):
        x={"instance_id":str(i),"row_id":cid,"semantic_axis":"color","region":"upper",
           "canonical_value":"black","unresolved":False,"raw_text":"x","char_start":i,"char_end":i+1,
           "source_span":{"raw":"x","start":i,"end":i+1},
           "semantic_representation":{"basis_rank":1,"basis_id":"test","projected_context":[1.]+[0.]*511},
           "linked_object":None,"provenance":{"original":"preserved"}}
        attrs.append({"instance":x,"instance_id":str(i),"status":"scored","rank":1,"S_raw":1.,
                      "w":[1.]+[0.]*76})
    return {"row_id":cid,"pid":pid,"caption":"x"*k,"attributes":attrs,
            "retrieval_backward_count":1,"zero_intervention_exact":True}
def basis():
    return {"color":{"rank":1,"basis_id":"test","Q":[[1.]]+[[0.]]*511}}
@pytest.mark.parametrize("k",[0,1,2,3,4,5,6,7,9])
def test_k_uncapped_cap_and_fixed_ranking(k):
    r=record(k);p=a.selection(r)
    assert p["natural_k"]==k and p["capped_k"]==min(k,4)
    assert p["truncated"]==(k>4)
    assert p["selected_attributes"]==p["ranked_attributes"][:k]
    if k:assert sum(p["probabilities"])==pytest.approx(1.)
    assert [x["instance_id"] for x in p["ranked_attributes"]]==[str(i) for i in range(k)]
@pytest.mark.parametrize("k",[1,4,5,7,9])
def test_uncapped_reuses_original_span_and_basis_gate(k):
    r=record(k);p=a.selection(r);plan=pair.prepare_uncapped(p,r,basis(),"cpu")
    assert len(plan)==k
    assert [x.selection_rank for x in plan]==list(range(1,k+1))
    assert [x.metadata for x in plan]==p["selected_attributes"]
@pytest.mark.parametrize("field",["span","unresolved","rank","score"])
def test_bad_candidates_never_bypass_gate(field):
    r=record(1);x=r["attributes"][0]
    if field=="span":x["instance"]["source_span"]["start"]=1
    if field=="unresolved":x["instance"]["unresolved"]=True
    if field=="rank":x["instance"]["semantic_representation"]["basis_rank"]=0
    if field=="score":x["S_raw"]=float("nan")
    with pytest.raises(ValueError):a.selection(r)
def test_unsupported_unresolved_and_zero_excluded():
    r=record(4);r["attributes"][0]["status"]="unresolved_excluded"
    r["attributes"][0]["instance"]["unresolved"]=True
    r["attributes"][1]["status"]="score_unsupported";r["attributes"][1]["rank"]=0
    r["attributes"][2]["S_raw"]=0
    assert a.selection(r)["candidate_count"]==1
def test_same_value_different_source_span_metadata_preserved():
    r=record(2);p=a.selection(r);before=copy.deepcopy(r)
    assert p["selected_attributes"][0]["source_span"]!=p["selected_attributes"][1]["source_span"]
    assert r==before
def test_stratification_deterministic_rare_bins_and_exact100():
    records=[record(k,f"cuhk:{1000+k*100+i}:0") for k in range(1,10) for i in range(20 if k!=1 else 2)]
    ids={r["row_id"] for r in records}
    one=a.stratify(iter(records),ids);two=a.stratify(iter(reversed(records)),ids)
    assert one==two and one["case_count"]==100
    assert one["bin_selected"]["1"]==one["bin_available"]["1"]==2
    assert all(one["bin_selected"][b]>0 for b in a.BINS)
    assert all(x["caption_id"] in ids for x in one["rows"])
def test_population_not_validation_not_silently_admitted():
    records=[record(2,"cuhk:1:0"),record(2,"cuhk:2:0")]
    m=a.stratify(records,{"cuhk:2:0"},100)
    assert m["case_count"]==1 and m["excluded"]["not_validation"]==1
class Toy(nn.Module):
    def encode_images(self,x):return F.normalize(x.mean((-2,-1)),dim=-1)
    def inference_image(self,x):return {"image_feat":self.encode_images(x)}
    def inference_text(self,*args):raise AssertionError("caption guidance forbidden")
def fixture(k):
    plan=tuple(SelectedAttribute({"instance_id":str(i),"semantic_axis":"color","region":"upper",
        "canonical_value":str(i),"S_raw":k-i+1,"source_span":{"raw":"x","start":i,"end":i+1}},
        i,torch.ones(1),torch.ones(1,1)) for i in range(1,k+1))
    values=[[.2,.5,-.3],[-.4,.3,.1],[.7,-.1,.2],[.1,.2,.6]]
    v={x.instance_id:torch.tensor([values[i%4]]) for i,x in enumerate(plan)}
    source=Toy();image=torch.linspace(.2,.8,3*16*16).reshape(1,3,16,16)
    ev=RetrievalMetrics(source,F.normalize(torch.tensor([[1.,2.,3.]]),dim=-1),
        F.normalize(torch.tensor([[1.,2.,3.],[3.,1.,2.]]),dim=-1),0)
    return plan,v,source,image,ev
@pytest.mark.parametrize("k",[1,2,3,4,5,6,7,9])
def test_actual_original_prefix_regression_and_only_suffix_changes(k):
    plan,v,source,image,ev=fixture(k)
    data,checks=pair.execute_pair(source,image,plan,v,ev,steps=1,transforms=1,scales=None)
    capped,natural=[data[m] for m in a.MODES]
    assert capped["rounds"]==natural["rounds"][:min(k,4)]
    assert capped["states"]==natural["states"][:min(k,4)+1]
    if k<=4:assert capped==natural
    seed_all(42);tape=CategoricalTape()
    original,_,_=old.execute(source,image,plan[:4],old.cached_round_guidance("PREFIX_NEW",plan[:4],v),
                           ev,steps=1,transforms=1,scales=None,tape=tape)
    assert capped==original # CPU原官方TTA/旧PREFIX独立执行逐值回归。
    assert checks["score_backwards_during_attack"]==0 and checks["caption_loss_terms"]==0
    assert natural["final_linf"]<=IMAGE_EPSILON+1e-6
def test_sha_mutation_rejected(tmp_path):
    f=tmp_path/"input.json";f.write_text("frozen");h=a.digest(f)
    a.verify({str(f):h});f.write_text("changed")
    with pytest.raises(ValueError):a.verify({str(f):h})
def test_boundaries_require_manual_opt_in_before_loading_models():
    import run,prepare_scores
    with pytest.raises(SystemExit):run.main(["--mode","diagnostic","--manifest","missing"])
    with pytest.raises(SystemExit):prepare_scores.main(["--mode","prepare"])
    assert run.parser().parse_args(["--manifest","fake"]).mode=="smoke"
def test_new_caption_scoring_reuses_single_frozen_backward():
    import importance
    from unittest.mock import patch
    class Probe:
        def encode(self,caption,layer):
            H=torch.ones(77,1,512,requires_grad=True)
            return H.sum(0),H,torch.zeros_like(H)
        def objective(self,feature):return feature.square().sum(),feature.sum()
    with patch("torch.autograd.grad",wraps=torch.autograd.grad) as counter:
        _,_,G,_,_,_=importance.baseline_gradient(Probe(),"caption")
    assert counter.call_count==1 and G.shape==(77,1,512)
def test_material_rule_and_metric_win_signs_registered():
    def result(fs):
        states=[{"paired_soft_rank_F":f,"paired_cosine":1-f/10000,
                 "positive_image_rank":int(f),"state_index":i} for i,f in enumerate(fs)]
        events=[{"metric_delta":{key:states[i+1][key]-states[i][key] for key in a.METRICS},
                 "post_projection_actual_change":{"l2":1},"projection_clipping_ratio_mean_over_steps":.5,
                 "actual_post_projection_update_cosine_previous_round":None,"image_gradient_l2_at_round_start":.1}
                for i in range(len(fs)-1)]
        return {"states":states,"rounds":events,"clean_to_final":{key:states[-1][key]-states[0][key] for key in a.METRICS}}
    cases=[]
    for i,end in enumerate([500,300,400]):
        natural=result([0,100,200,300,400,end]);cap=result([0,100,200,300,400])
        cases.append({"caption_id":str(i),"natural_k":5,"modes":dict(zip(a.MODES,[cap,natural]))})
    summary,rounds=a.summarize(cases)
    assert summary["after_round4"]["counts"]=={"material_improvement":1,"material_worse":1,"nearly_unchanged":1}
    assert summary["comparisons_all"]["paired_soft_rank_F"]["natural_wins"]==1
    assert summary["comparisons_all"]["paired_cosine"]["natural_wins"]==1
    assert rounds["5"]["recovery_count"]==1 and rounds["7+"]["round_count"]==0
def test_attack_runner_has_no_scoring_import_or_function():
    text=(Path(__file__).parent/"run.py").read_text()
    assert "baseline_gradient(" not in text and "score_instances(" not in text and "FrozenLocalization(" not in text
    assert "scoring_runner(" not in text

def test_attack_manifest_gate_does_not_recompute_topk(monkeypatch):
    import run
    manifest=deps.ROOT/"outputs/score_smoke/selection_manifest.json"
    monkeypatch.setattr(run,"selection",lambda *args:(_ for _ in ()).throw(AssertionError("TopK recomputed")))
    m,_,records=run.checked_manifest(manifest,"smoke")
    assert m["case_count"]==3 and set(records)=={"cuhk:34:0","cuhk:26:0","cuhk:9781:0"}
def test_frozen_payload_gate_rejects_metadata_or_order_tampering():
    import run
    r=record(5);p=a.selection(r);run.verify_fixed_payload(p,r)
    p["ranked_attributes"][0]["S_raw"]=2
    with pytest.raises(ValueError):run.verify_fixed_payload(p,r)
