"""新方法CPU机制测试；真实BPE/H接口，图像编码使用显式合成fixture。"""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from tbps.artifacts import ArtifactError, ArtifactStore, FileRef, sha_file, json_bytes
from tbps.alternating.state import tree_sha, capture_rng
from tbps.attribute_response.settings import Settings
from tbps.attribute_response.memory import AttributeMemory, semantic_key
from tbps.attribute_response.teacher import ResponseTeacher
from tbps.attribute_response.isolation import sandbox, clone_trainable
from tbps.attribute_response.objectives import conditional_advantage, coordinated_backward
from tbps.attraction.objectives import ProfileKL, Settings as KLSettings

torch.set_num_threads(2)
def key(value="blue"):
    return semantic_key({"semantic_axis": "color", "canonical_value": value, "region": "upper"})

@pytest.mark.parametrize("field,value", [
    ("probe_interval",0), ("history",False), ("width",-2), ("weight_mix",1.1),
    ("teacher_lr",float("nan")), ("response_weight",-1), ("coordinate_gradients",1),
    ("weight_temperature",0), ("replay_capacity",1), ("seed",False)])
def test_settings_reject_invalid(field, value):
    with pytest.raises(ArtifactError):
        Settings(**{field:value})

def test_semantic_memory_shares_attributes_without_sample_or_span_identity():
    a={"semantic_axis":"color","canonical_value":"blue","region":"upper","char_start":2,"pid":123}
    b={**a,"char_start":80,"pid":777}
    assert semantic_key(a)==semantic_key(b)==key()
    memory=AttributeMemory(2,2)
    for i in range(4):
        memory.append(key(),[float(i)]*12,[i,-i],version=i)
    assert len(memory.read(key()))==2 and memory.read(key())[-1]["state"][0]==3
    saved=memory.state_dict();other=AttributeMemory(2,2);other.load_state_dict(saved)
    assert other.state_dict()==saved
    with pytest.raises(ArtifactError):
        memory.append(key(),[0.]*12,[0.,0.],version=0)
    with pytest.raises(ArtifactError):
        memory.append(key(),[0.]*12,[0.,0.],version=5,role="test")
    memory.append(key("red"),[0.]*12,[0.,0.],version=5)
    memory.append(key("green"),[0.]*12,[0.,0.],version=5)
    assert not memory.read(key()) and len(memory.records)==2

def test_teacher_learns_both_directions_history_and_exact_resume():
    settings=Settings(width=12,teacher_batch=4)
    teacher=ResponseTeacher(settings)
    original_rng=torch.get_rng_state().clone()
    keys=[key(),key("red")];states=[[.2]*12,[.7]*12]
    cold,before=teacher.weights(keys,states,[.8,.2],version=1)
    assert math.isclose(float(cold.mean()),1.,abs_tol=1e-6)
    batch=teacher.features(keys,states,version=1)
    for _ in range(4):
        teacher.observe(batch,[1.,-1.],0)
        teacher.observe(batch,[-.5,.8],1)
    teacher.remember(key(),states[0],[.4,-.2],version=1)
    batch=teacher.features(keys,states,version=2)
    for _ in range(3):
        teacher.observe(batch,[.7,-.4],0)
    assert teacher.model.g_to_h.weight.grad is not None
    assert teacher.model.history.weight_ih.grad is not None
    assert torch.equal(original_rng,torch.get_rng_state())
    state=teacher.state_dict();restored=ResponseTeacher(settings);restored.load_state_dict(state)
    assert tree_sha(restored.state_dict())==tree_sha(state)
    assert torch.equal(teacher.predict(batch),restored.predict(batch))
    assert teacher.observe(batch,[.3,-.6],1)==restored.observe(batch,[.3,-.6],1)
    assert tree_sha(restored.state_dict())==tree_sha(teacher.state_dict())

def test_teacher_rejects_test_feedback_and_wrong_checkpoint_settings():
    t=ResponseTeacher(Settings(width=8));b=t.features([key()],[[0.]*12],version=1)
    with pytest.raises(ArtifactError):t.observe(b,[1.],0,role="validation")
    value=t.state_dict();value["settings"]["width"]=10
    with pytest.raises(ArtifactError):t.load_state_dict(value)

@pytest.mark.parametrize("raises",[False,True])
def test_sandbox_preserves_live_adam_grads_modes_and_all_rng(raises):
    g=torch.nn.Linear(3,2);opt=torch.optim.Adam(g.parameters(),lr=.01)
    g(torch.ones(1,3)).sum().backward();opt.step()
    sampler=torch.Generator().manual_seed(31)
    before=(tree_sha(g.state_dict()),tree_sha(opt.state_dict()),tree_sha(capture_rng(sampler)))
    error=RuntimeError("virtual failure")
    try:
        with sandbox((g,),(opt,),sampler) as proof:
            clone,co=clone_trainable(g,opt)
            co.zero_grad();clone(torch.randn(2,3)).square().sum().backward();co.step()
            torch.rand(4,generator=sampler)
            if raises:raise error
    except RuntimeError as e:
        assert raises and e is error
    assert before==(tree_sha(g.state_dict()),tree_sha(opt.state_dict()),tree_sha(capture_rng(sampler)))
    assert proof["isolated"] and proof["rng_before"]==proof["rng_after"]

def test_conditional_advantage_matches_expectation_not_noop_baseline():
    adv,base=conditional_advantage([1.,4.,-2.],[.2,.5,.3],1)
    assert math.isclose(float(base),1.6,abs_tol=1e-6) and math.isclose(float(adv),2.4,abs_tol=1e-6)
    with pytest.raises(ArtifactError):conditional_advantage([1.,2.],[.5,.9],0)

def test_gradient_coordination_only_projects_conflicting_attribute_component():
    x=torch.nn.Parameter(torch.tensor([1.,2.]))
    info=coordinated_backward(x[0],-2*x[0]+3*x[1],[x])
    assert info["conflict"] and torch.allclose(x.grad,torch.tensor([1.,3.]))

@pytest.fixture(scope="module")
def native():
    from tbps.attribute_response.sources import load
    from tbps.alternating.fresh_sources import fresh_bundle
    from tbps.alternating.native_sources import native_session,strict_import_preflight
    f,c=load(ROOT/"configs/framework/gh_attribute_response.server.json","smoke")
    strict_import_preflight(c);bundle=fresh_bundle(f,c)
    with native_session(c,reuse=bundle.modules) as session:
        yield f,c,bundle,session

class TinyG(torch.nn.Module):
    def __init__(self):
        super().__init__();self.v=torch.nn.Parameter(torch.tensor([.001,-.001,.0005])[None,:,None,None])
    def forward(self,x):return self.v.tanh().expand_as(x)
class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w",torch.randn(3,512,generator=torch.Generator().manual_seed(99)))
    def forward(self,x):return F.normalize(x.mean((2,3))@self.w,dim=-1)

@pytest.mark.parametrize("k",[0,1,3,7])
@pytest.mark.parametrize("coordinate",[False,True])
def test_weighted_G_keeps_K_plus_one_budget_frozen_encoder_and_all_views(native,k,coordinate):
    _,c,bundle,_=native
    from tbps.attribute_response.g_step import train_single_sample
    gm=json.loads(Path(c["prototype"]["g_manifest"]).read_text())["entries"]
    e=next(x for x in gm if x["role"]=="train" and x["K"]==k)
    payload=torch.load(e["tensor_path"],map_location="cpu",weights_only=True)
    g=TinyG();opt=torch.optim.Adam(g.parameters(),lr=.0002);enc=TinyEncoder().eval().requires_grad_(False)
    x=.2+.6*torch.rand(3,3,16,16);clean=enc(x).detach()
    profile=ProfileKL(F.normalize(torch.randn(8,512),dim=-1),KLSettings())
    before=tree_sha(enc.state_dict());weights=torch.linspace(.5,1.5,k)
    log=train_single_sample(g,enc,opt,x,payload,profile=profile,clean_features=clean,
                            attribute_weights=weights,coordinate_gradients=coordinate)
    assert log["optimizer_steps"]==k+1 and tree_sha(enc.state_dict())==before
    assert all(p.grad is None and not p.requires_grad for p in enc.parameters())
    for i,r in enumerate(log["records"]):
        assert r["cumulative_linf"]<=8/255
        weight=1. if r["phase"]=="joint" else weights[i].item()
        assert math.isclose(r["local"],r["attribute_response"]["unweighted_local"]*weight,abs_tol=1e-8)
        assert math.isclose(r["total"],r["global"]+r["local"]+r["retain"],abs_tol=1e-5)
        assert len(r["attraction"]["view_linf"])==3

def example_and_seed(native,multiple_replacements=False):
    _,c,bundle,_=native;entries=json.loads((Path(c["prototype"]["history_root"])/"qualification/train.json").read_text())["entries"]
    h=bundle.h
    for e in entries[:80]:
        if not e["eligible"]:continue
        ex=bundle.bindings.get("formal_training.data").example_from(e)
        for seed in range(1000,1064):
            with torch.no_grad():out=h.rollout(ex,"sample",torch.Generator().manual_seed(seed))
            if out.edits and (not multiple_replacements or any(
                len(bundle.bindings.get("constraints").CONFUSABLES.get(edit.original_char,()))>1
                for edit in out.edits)):
                return e,ex,seed
    raise AssertionError("no legal sampled edits")

class HashRuntime:
    def __init__(self,probe):
        self.probe=probe;self.samples={"G":0,"clean":0,"adv":0,"text":0}
    def assert_frozen(self):pass
    def encode_text(self,texts):
        self.samples["text"]+=len(texts)
        self.last_features=F.normalize(torch.stack([torch.randn(512,generator=torch.Generator().manual_seed(
          int.from_bytes(hashlib.sha256(t.encode()).digest()[:4],"little"))) for t in texts]),dim=-1)
        return self.last_features

def test_real_H_three_heads_complete_counterfactual_support_and_independent_arithmetic(native):
    _,c,bundle,_=native;e,example,seed=example_and_seed(native,True);m=bundle.bindings
    from tbps.attribute_response.h_step import train_step
    from tbps.attribute_response.audit_math import audit_conditional
    h=copy.deepcopy(bundle.h).requires_grad_(True);opt=torch.optim.Adam(h.parameters(),lr=.0001)
    runtime=HashRuntime(bundle.probe);clean=F.normalize(torch.randn(4,512),dim=-1)
    partners=F.normalize(torch.randn(3,512),dim=-1);profile=ProfileKL(clean,KLSettings())
    rt=m.get("formal_training.runtime");query=SimpleNamespace(caption=example.caption)
    gallery=rt.TrainGallery(clean,clean.clone(),torch.tensor([True,False,False,False]),0,"train",("train",)*4,"train",("train",)*4)
    make=lambda k:rt.FreshReward(gallery,query,example,k,runtime)
    teacher=ResponseTeacher(Settings(width=12));states=[[.2]*12 for a in example.attributes]
    log,_=train_step(m.get("phase4_training"),m.get("policy"),m.get("constraints"),h,opt,example,
       torch.Generator().manual_seed(seed),make,1,"fixture",profile,partners,teacher,states,version=1)
    md=log["attribute_response"]
    assert log["optimizer_steps_this_step"]==1 and log["gradient_norm"]>0
    assert len(md["credits"])==len(log["sampled"]["edits"])>0
    assert log["group_gradient_norm"]["attribute_head"]>0
    assert log["group_gradient_norm"]["position_head"]>0 and log["group_gradient_norm"]["replacement_head"]>0
    texts=md["counterfactual_features"]["texts"]
    features=torch.tensor(md["counterfactual_features"]["features"],dtype=torch.float64)
    for i,credit in enumerate(md["credits"]):
        audit_conditional(credit,log["sampled"],i,texts,features,torch.tensor(md["original_features"][:1],dtype=torch.float64),
          partners.double(),clean,{"temperature":.07,"pull_weight":1.,"push_weight":1.},teacher.settings,
          m.get("contracts"),m.get("constraints"),bundle.probe)
        assert any(x["position"] is None for x in credit["support"])
        corrupted=copy.deepcopy(credit);corrupted["baseline"]+=.1
        with pytest.raises(ArtifactError):
            audit_conditional(corrupted,log["sampled"],i,texts,features,torch.tensor(md["original_features"][:1],dtype=torch.float64),
              partners.double(),clean,{"temperature":.07,"pull_weight":1.,"push_weight":1.},teacher.settings,
              m.get("contracts"),m.get("constraints"),bundle.probe)
    assert log["IRRA_text_samples"]==runtime.samples["text"]==3+len(texts)
    assert math.isclose(md["optimized_loss"],log["SCST_loss"]+md["credit_loss"]+md["attribute_distill_loss"],abs_tol=1e-5)

def test_probe_trajectory_probability_exact_and_sensitive_to_H_update(native):
    _,_,bundle,_=native;e,example,seed=example_and_seed(native)
    from tbps.attribute_response.probe_policy import trajectory_log_prob
    h=copy.deepcopy(bundle.h).requires_grad_(True)
    out=h.rollout(example,"sample",torch.Generator().manual_seed(seed))
    exact=trajectory_log_prob(h,example,out.record())
    assert torch.allclose(exact,out.log_prob,atol=2e-6)
    opt=torch.optim.Adam(h.parameters(),lr=.001);opt.zero_grad();(-exact).backward();opt.step()
    assert trajectory_log_prob(h,example,out.record())>exact.detach()

def test_default_config_and_supervisor_do_not_authorize_or_start_GPU():
    cfg=json.loads((ROOT/"configs/framework/gh_attribute_response.server.json").read_text())
    assert not any(cfg["authorization"][k] for k in ("GPU_experiment","automatic_training","new_Test"))
    source=(ROOT/"operations/gh_attribute_response_supervisor.py").read_text()
    assert source.index("configuration currently authorizes")<source.index("CTL.mkdir")
    assert "for n in (10,20,30,40,50,60)" in source and "FAIL_STOPPED_NO_RETRY" in source

def test_bidirectional_real_Adam_lookahead_isolated_and_auditable(native):
    _,c,bundle,_=native
    from tbps.attribute_response.probes import ResponseProber
    from tbps.attribute_response.audit_math import audit_probe
    from tbps.attribute_response.memory import semantic_key
    qualification=json.loads((Path(c["prototype"]["history_root"])/"qualification/train.json").read_text())["entries"]
    gm={x["row_id"]:x for x in json.loads(Path(c["prototype"]["g_manifest"]).read_text())["entries"] if x["role"]=="train"}
    support=next(e for e in qualification if e["eligible"] and gm[e["identity"]["row_id"]]["K"]==1)
    rest=[e for e in qualification if e["eligible"] and e["identity"]["row_id"]!=support["identity"]["row_id"]][:3]
    entries=[support]+rest;ids=[e["identity"]["row_id"] for e in entries]
    payload=torch.load(gm[ids[0]]["tensor_path"],map_location="cpu",weights_only=True)
    attrs=json.loads(Path(gm[ids[0]]["metadata_path"]).read_text())["source_selection"]["selected_attributes"]
    b=SimpleNamespace(native=bundle.bindings,p={"samples":ids,"train_ids":ids},entries=dict(zip(ids,entries)),
       g=TinyG().train(),h=copy.deepcopy(bundle.h).eval().requires_grad_(False),
       sampler=torch.Generator().manual_seed(123),counts={"G":0,"H":0,"IRRA_image":0,"IRRA_text":0},
       virtual_counts={"G":0,"H":0},round_index=1,probe_cursor=0,cursor={"g_updates":0},
       response_settings=Settings(width=8,probe_attributes=1,probe_interval=1))
    b.ga=torch.optim.Adam(b.g.parameters(),lr=.0002);b.ha=torch.optim.Adam(b.h.parameters(),lr=.0001)
    b.encoder=TinyEncoder().eval().requires_grad_(False)
    b.encoder.register_forward_pre_hook(lambda m,args:b.counts.__setitem__("IRRA_image",b.counts["IRRA_image"]+len(args[0])))
    images={row:.2+.6*torch.rand(3,3,16,16,generator=torch.Generator().manual_seed(i+70)) for i,row in enumerate(ids)}
    b.make_views=lambda row:images[row].clone()
    with torch.no_grad():b.clean=torch.stack([b.encoder(images[row])[0] for row in ids])
    b.current_view_features=b.clean[:,None].expand(-1,3,-1).clone();b.current_features=b.clean.clone()
    b.profile_cpu=ProfileKL(b.clean,KLSettings());b.profile_g=b.profile_cpu
    b.example=lambda row:b.native.get("formal_training.data").example_from(b.entries[row])
    b.catalog=SimpleNamespace(train=[SimpleNamespace(pid=e["identity"]["pid"]) for e in entries],
          get_train=lambda row:SimpleNamespace(pid=b.entries[row]["identity"]["pid"],caption=b.entries[row]["caption"]))
    b.runtime=HashRuntime(bundle.probe)
    original_encode=b.runtime.encode_text
    def encode(texts):
        b.counts["IRRA_text"]+=len(texts);return original_encode(texts)
    b.runtime.encode_text=encode
    b.teacher=ResponseTeacher(b.response_settings)
    prober=ResponseProber(b);b.prober=prober
    before=(tree_sha(b.g.state_dict()),tree_sha(b.h.state_dict()),tree_sha(b.ga.state_dict()),tree_sha(b.ha.state_dict()),tree_sha(capture_rng(b.sampler)))
    result=prober.g_to_h(ids[0],SimpleNamespace(payload=payload),[semantic_key(a) for a in attrs],[[.2]*12],b.encoder(images[ids[0]]).detach(),images[ids[0]])
    assert result["performed"] and len(result["candidates"])==2
    assert all(x["virtual_G_steps"]==2 and x["virtual_H_steps"]==1 for x in result["candidates"])
    assert before==(tree_sha(b.g.state_dict()),tree_sha(b.h.state_dict()),tree_sha(b.ga.state_dict()),tree_sha(b.ha.state_dict()),tree_sha(capture_rng(b.sampler)))
    positive={row:b.entries[row]["identity"]["pid"] for row in ids}
    for candidate in result["candidates"]:
        item=dict(candidate,support=ids[0],probe=result["probe"],role="train")
        audit_probe(item,b.current_view_features,b.clean,{"temperature":.07,"pull_weight":1.,"push_weight":1.},b.response_settings,ids,positive)
    target,proof=prober.h_to_g(ids[0],b.runtime.encode_text([b.entries[ids[0]]["caption"]])[0],0)
    audit_probe(proof,b.current_view_features,b.clean,{"temperature":.07,"pull_weight":1.,"push_weight":1.},b.response_settings,ids,positive)
    assert proof["Global_only"] and proof["new_JVPs"]==0 and math.isfinite(target)
    assert before==(tree_sha(b.g.state_dict()),tree_sha(b.h.state_dict()),tree_sha(b.ga.state_dict()),tree_sha(b.ha.state_dict()),tree_sha(capture_rng(b.sampler)))

def test_production_backend_G_H_paths_and_extended_numeric_audit(native,tmp_path):
    _,c,bundle,_=native
    from tbps.attribute_response.backend import AttributeResponseBackend
    from tbps.attribute_response.audit_math import audit_attraction
    from tbps.augmented_attraction.views import ViewSettings,recipe
    from tbps.artifacts import digest
    qualification=json.loads((Path(c["prototype"]["history_root"])/"qualification/train.json").read_text())["entries"]
    gm={x["row_id"]:x for x in json.loads(Path(c["prototype"]["g_manifest"]).read_text())["entries"] if x["role"]=="train"}
    e,example,seed=example_and_seed(native);row=e["identity"]["row_id"]
    rest=[x for x in qualification if x["eligible"] and x["identity"]["row_id"]!=row][:3]
    entries=[e]+rest;ids=[x["identity"]["row_id"] for x in entries]
    b=AttributeResponseBackend.__new__(AttributeResponseBackend)
    base=tmp_path/"tbps-framework-backend-fixture"
    b.c=copy.deepcopy(c);b.c["output_base"]=str(base);b.p={"samples":ids,"train_ids":ids}
    b.native=bundle.bindings;b.bundle=bundle;b.entries=dict(zip(ids,entries))
    b.g=TinyG();b.ga=torch.optim.Adam(b.g.parameters(),lr=.0002)
    b.h=copy.deepcopy(bundle.h).eval().requires_grad_(False);b.ha=torch.optim.Adam(b.h.parameters(),lr=.0001)
    b.encoder=TinyEncoder().eval().requires_grad_(False)
    b.cursor={"g_updates":0,"h_visits":0,"h_updates":0};b.round_index=1
    b.grant=SimpleNamespace(validate=lambda *x:None);b.sampler=torch.Generator().manual_seed(seed)
    b.counts={"G":0,"H":0,"IRRA_image":0,"IRRA_text":0};b.virtual_counts={"G":0,"H":0}
    b.response_settings=Settings(width=8,probe_interval=99999);b.teacher=ResponseTeacher(b.response_settings)
    # 本测试聚焦正式接线；虚拟分支另有真实Adam测试。
    b.response_visits={"G":1,"H":1};b.probe_cursor=1;b.h_probe_cursor=1;b.view_settings=ViewSettings()
    b.anchors={};b.original={}
    for rid in ids:
        entry=gm[rid];payload=torch.load(entry["tensor_path"],map_location="cpu",weights_only=True)
        b.original[rid]=SimpleNamespace(payload=payload)
        b.anchors[rid]=json.loads(Path(entry["metadata_path"]).read_text())["source_selection"]["selected_attributes"]
    images={rid:.2+.6*torch.rand(3,3,384,128,generator=torch.Generator().manual_seed(i+66)) for i,rid in enumerate(ids)}
    b.make_views=lambda rid:images[rid].clone()
    with torch.no_grad():b.clean=torch.stack([b.encoder(images[rid])[0] for rid in ids])
    b.current_features=b.clean.clone();b.current_view_features=b.clean[:,None].expand(-1,3,-1).clone()
    b.anchor_sha=tree_sha(b.clean);b.profile_cpu=ProfileKL(b.clean,KLSettings());b.profile_g=b.profile_cpu
    b.store=ArtifactStore(base/"io",new_base=base)
    b.runtime=HashRuntime(bundle.probe)
    b.catalog=SimpleNamespace(train=[SimpleNamespace(pid=x["identity"]["pid"]) for x in entries],
        get_train=lambda rid:SimpleNamespace(pid=b.entries[rid]["identity"]["pid"],caption=b.entries[rid]["caption"]))
    identity={"original_caption_sha256":"a"*64,"context_caption_sha256":"b"*64,
       "selected_g_ids":list(b.original[row].payload["selected_instance_ids"]),
       "scores":b.original[row].payload["S_raw"].tolist(),"source_space_sha256":"c"*64,"anchor_mapping_sha256":"d"*64}
    condition=SimpleNamespace(payload=b.original[row].payload,identity=SimpleNamespace(key=digest(identity)))
    initial=tree_sha(b.teacher.state_dict());glog=b.g_update(row,condition);glog["row_id"]=row
    b.g.eval().requires_grad_(False);b.g.zero_grad(set_to_none=True)
    with torch.no_grad():
        from budget_projection import add_projected_delta
        b.current_view_features=torch.stack([b.encoder(add_projected_delta(images[rid],images[rid],b.g(images[rid]))) for rid in ids])
    b.current_features=b.current_view_features[:,0].clone();b.view_bank_sha=tree_sha(b.current_view_features)
    bank=SimpleNamespace(g_state_sha256=b.g_fingerprint(),adversarial_sha256=tree_sha(b.current_features));b.last_bank=bank
    rt=b.native.get("formal_training.runtime")
    query=b.catalog.get_train(row);positive=torch.tensor([x.pid==query.pid for x in b.catalog.train])
    gallery=rt.TrainGallery(b.clean,b.current_features,positive,0,"train",("train",)*len(ids),"train",("train",)*len(ids))
    b.reward=SimpleNamespace(bind=lambda q,r,ex,k:rt.FreshReward(gallery,q,ex,k,r))
    hlog=b.h_update(row,bank)
    md=hlog["attribute_response"];ref=md["counterfactual_features"];folder=Path(ref["artifact"])
    md["counterfactual_features"]={"texts":json.loads((folder/"texts.json").read_text())["texts"],
         "features":torch.load(folder/"features.pt",map_location="cpu",weights_only=True).tolist()}
    bindings=[]
    for i,rid in enumerate(ids):
        bindings.append({"row":rid,"input_sha256":tree_sha(images[rid]),
          "recipe_sha256":digest(recipe(ViewSettings(),rid,1,384,128)),"view_linf":[0.,0.,0.],
          "feature_sha256":tree_sha(b.current_view_features[i])})
    ledger={"round":1,"G_before":glog["G_before"],"G_after":glog["G_after"],
       "condition_H_state":glog["attraction"]["teacher_H_state"],"G":[glog],"H":[hlog],
       "conditions":[{"row":row,"identity":identity}],"g_orders":[[row]],"h_orders":[[row]],
       "bank":{"ordered_train_ids":ids},"bank_view_inputs":bindings,"attribute_teacher_after":tree_sha(b.teacher.state_dict())}
    report=audit_attraction(ledger,{"temperature":.07,"pull_weight":1.,"push_weight":1.},b.native.get("contracts"),
       b.native.get("constraints"),b.clean,b.current_features,b.current_view_features,
       {"seed":42,"views":3,"local_rows":4,"local_cols":2,"global_chain":3,"rotation_degrees":5.,"translation_fraction":.04},
       response_cfg=b.response_settings.record(),native_config=c,expected_teacher_before=initial)
    assert report["G_steps_independently_recomputed"]==glog["K"]+1
    assert report["H_steps_independently_recomputed"]==1 and report["no_G_H_victim_forward_in_audit"]

def test_complete_attribute_backend_state_restores_teacher_memory_rng_and_counters():
    from tbps.attribute_response.backend import AttributeResponseBackend
    from tbps.augmented_attraction.backend import AttractionBackend
    from unittest.mock import patch
    b=AttributeResponseBackend.__new__(AttributeResponseBackend)
    b.teacher=ResponseTeacher(Settings(width=8));b.virtual_counts={"G":30,"H":14}
    b.probe_cursor=2;b.h_probe_cursor=1;b.response_visits={"G":32,"H":32}
    batch=b.teacher.features([key()],[[.2]*12],version=10)
    b.teacher.observe(batch,[.7],0);b.teacher.remember(key(),[.2]*12,[.7,0.],version=10)
    with patch.object(AttractionBackend,"complete_state",return_value={"base":"preserved"}):
        saved=b.complete_state(10)
    b.teacher.observe(batch,[-.5],1);b.virtual_counts={"G":999,"H":999};b.h_probe_cursor=99
    with patch.object(AttractionBackend,"restore_complete") as parent:
        b.restore_complete(saved)
    parent.assert_called_once_with({"base":"preserved"})
    assert tree_sha(b.teacher.state_dict())==tree_sha(saved["attribute_teacher"])
    assert b.virtual_counts=={"G":30,"H":14} and b.h_probe_cursor==1

def test_new_native_Joint_engine_uses_original_metrics_fixture(tmp_path):
    import importlib.util
    from tbps.alternating import fresh_evaluation
    from tbps.attribute_response.evaluation import joint_score
    from unittest.mock import patch
    spec=importlib.util.spec_from_file_location("attribute_response_joint_fixture",ROOT/"tests/framework/test_gh_alternating_v2.py")
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    with patch.object(fresh_evaluation,"joint_score",joint_score):
        fixture.test_Joint_only_native_engine_fixture(tmp_path)

def test_training_rejects_missing_GPU_capability_before_preflight():
    from tbps.attribute_response.run import execute
    with pytest.raises(ValueError,match="explicit GPU capability"):
        execute("not-a-config","train",allow=False)
    assert not torch.cuda.is_initialized()

def test_six_node_report_and_curves_require_all_audited_results(tmp_path,monkeypatch):
    from tbps.attribute_response import report
    root=tmp_path/"repo";base=tmp_path/"tbps-framework-report-fixture";cfg=tmp_path/"config.json";cfg.write_text("{}")
    (root/"docs").mkdir(parents=True)
    (root/"docs/PROJECT_STATE.md").write_text("<!-- GH_AR_CURRENT_BEGIN -->pending<!-- GH_AR_CURRENT_END -->")
    frozen=json.loads((ROOT/"configs/framework/gh_attribute_response.server.json").read_text())
    f={"test_rounds":[10,20,30,40,50,60],"output_base":str(base),"test_output":str(base/"test"),
       "audits":{"train":str(base/"audit")},"attraction":frozen["attraction"],"augmentation":frozen["augmentation"],
       "attribute_response":frozen["attribute_response"],"report_runtime":frozen["report_runtime"]}
    train=base/"audit/round_60/completed";train.mkdir(parents=True)
    (train/"report.json").write_bytes(json_bytes({"status":"GH_AR_TRAIN_CPU_AUDIT_PASS","fresh_initialization":{"CPU_fixture":True}}))
    for n in f["test_rounds"]:
        d=base/"test"/f"round_{n:02d}";(d/"cpu_audit/completed").mkdir(parents=True);(d/"completed").mkdir()
        metrics={"R@1":float(80-n),"R@5":float(90-n),"R@10":float(95-n),"mAP":float(70-n)}
        (d/"completed/summary.json").write_bytes(json_bytes({"metrics":metrics}))
        (d/"cpu_audit/completed/report.json").write_bytes(json_bytes({"status":"GH_AR_JOINT_TEST_CPU_AUDIT_PASS",
            "scope":"official_test","metrics":metrics,"round":n,"gallery":3074,"query":6156,"CPU_fixture":True}))
    monkeypatch.setattr(report,"load",lambda *args:(f,{}))
    monkeypatch.setattr(report,"__file__",str(root/"src/tbps/attribute_response/report.py"))
    monkeypatch.setenv("MPLCONFIGDIR",str(tmp_path/"mpl"))
    outcome=report.publish(cfg)
    out=root/"docs/framework/results/gh_attribute_response"
    assert len(outcome["rows"])==6 and len(list(out.glob("*.svg")))==2 and len(list(out.glob("*.png")))==2
    assert "GH_ATTRIBUTE_RESPONSE.md" in (out/"REPORT.md").read_text()
    assert json.loads((out/"SOURCE_EVIDENCE.json").read_text())["attribute_response"]==f["attribute_response"]
    (base/"test/round_60/cpu_audit/completed/report.json").unlink()
    with pytest.raises(FileNotFoundError):report.publish(cfg)
