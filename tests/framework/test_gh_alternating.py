"""仅合成CPU状态/特征；不恢复或前向真实G/H/IRRA。"""
import copy
import random
import sys
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from tbps.artifacts import digest, ArtifactError
from tbps.alternating.contracts import (
    Anchor, Edit, TextTrace, ConditionIdentity, BankIdentity, cache_key, text_sha)
from tbps.alternating.conditions import PreparedCondition, prepare_condition, LegacyContextKernel, LegacySteps
from tbps.alternating.state import (
    clone_tree, tree_sha, capture_rng, restore_rng, snapshot, restore, RoundStore)
from tbps.alternating.runner import Prototype

@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("CPU tests must never initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)

@pytest.fixture
def case():
    text = "blue shirt"
    a = Anchor("color", "blue", "upper", "blue", 0, 4)
    trace = TextTrace(text, "bluе shirt", (a,), (Edit(a.h_id, 3, "e", "е"),), 1, "a"*64)
    identity = ConditionIdentity(text_sha(text), text_sha(text), ("G-ID",), (2.0,),
                                 "b"*64, digest([{"original": "source"}]))
    z = torch.zeros(512); z[0] = 1
    payload = dict(row_id="cuhk:1:0", caption=text, image_rel_path="a.png",
                   caption_index=0, split="train", role="train", pid=1,
                   K=1, selected_instance_ids=["G-ID"], S_raw=torch.tensor([2.],dtype=torch.float64),
                   score_weights=torch.tensor([1.],dtype=torch.float64), attribute_jvps=z[None],
                   z_t=z, v_local=z, has_local=True)
    return a, trace, PreparedCondition(identity, payload, {"verified": True})

class Kernel:
    def __init__(self):
        self.calls = []
        self.bad = None

    def recompute(self, text, mapping, original):
        self.calls.append((text, mapping))
        out = clone_tree(original.payload)
        out["caption"] = text
        out["z_t"] = torch.roll(out["z_t"], 1)
        out["attribute_jvps"] = torch.roll(out["attribute_jvps"], 1, -1)
        out["v_local"] = out["attribute_jvps"][0].clone()
        evidence = dict(context_caption_sha256=text_sha(text),
                        source_space_sha256=original.identity.source_space_sha256,
                        new_JVPs=1, new_text_encodings=5, fresh_localization=True,
                        zero_exact=True, JVP_checks=[dict(zero_exact=True, central_fd_relative_error=.001)])
        if self.bad:
            evidence.update(self.bad)
        return out, evidence

def prepared(case, arm="B", kernel=None, trace=None, anchors=None):
    a, t, original = case
    return prepare_condition(arm, trace or t, original, anchors or (a,),
                             legal_pairs=frozenset({("e","е")}), kernel=kernel or Kernel())

def test_exact_unicode_mapping(case):
    a, t, _ = case
    t.validate({("e","е")})
    m = t.mapping(a)
    assert m["edited_span"] == [0, 4]
    assert m["edited_utf8_span"] == [0, 5]
    assert m["original"]["canonical_value"] == "blue"
    assert m["edited_raw_text"] == "bluе"

@pytest.mark.parametrize("field,value", [
    ("adversarial", "BLUE shirt"), ("budget_k", 0), ("action_sha256", "bad"),
    ("edits", (Edit("unknown",3,"e","е"),)),
    ("edits", (Edit(Anchor("color","blue","upper","blue",0,4).h_id,3,"e","x"),)),
])
def test_bad_trace_rejected(case, field, value):
    _, t, _ = case
    with pytest.raises(ValueError):
        replace(t, **{field:value}).validate({("e","е")})

def test_ambiguous_anchor_rejected(case):
    a,t,_=case
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        replace(t, attributes=(a,a)).validate({("e","е")})

def test_g_ids_do_not_determine_alignment(case):
    a,t,original=case
    assert a.h_id != original.identity.selected_g_ids[0]
    assert prepared(case).identity.context_caption_sha256 == text_sha(t.adversarial)

def test_nonnoop_recomputes_context_and_jvp(case):
    k=Kernel(); result=prepared(case,kernel=k)
    assert len(k.calls)==1
    assert not torch.equal(result.payload["z_t"],case[2].payload["z_t"])
    assert not torch.equal(result.payload["attribute_jvps"],case[2].payload["attribute_jvps"])
    assert result.identity.selected_g_ids==case[2].identity.selected_g_ids

def test_a1_does_not_consume_h_conditions(case):
    k=Kernel(); result=prepared(case,arm="A1",kernel=k)
    assert not k.calls and result.payload is case[2].payload

def test_noop_reuses_original_only(case):
    a,t,original=case;k=Kernel()
    result=prepared(case,trace=replace(t, adversarial=t.original,edits=()),kernel=k)
    assert not k.calls and result.identity == original.identity
    assert result.evidence["mode"]=="NOOP_identity_reuse"

@pytest.mark.parametrize("bad", [
    {"new_JVPs":0}, {"new_text_encodings":0}, {"fresh_localization":False},
    {"context_caption_sha256":"0"*64}, {"source_space_sha256":"0"*64},
    {"zero_exact":False}, {"JVP_checks":[dict(zero_exact=True,central_fd_relative_error=.03)]},
    {"JVP_checks":[]},
])
def test_old_jvp_relabel_and_bad_receipts_refused(case,bad):
    k=Kernel();k.bad=bad
    with pytest.raises(ValueError):prepared(case,kernel=k)

def test_unmatched_canonical_fails(case):
    with pytest.raises(ValueError):
        prepared(case,anchors=(replace(case[0],canonical_value="red"),))

def test_invalid_g_selection():
    with pytest.raises(ValueError):
        ConditionIdentity("a"*64,"a"*64,("x","y"),(1.,2.),"b"*64,"c"*64)

def test_cache_dependencies():
    a=cache_key("adversarial",g="G0",space="IRRA",ordered_ids=("a","b"))
    assert a!=cache_key("adversarial",g="G1",space="IRRA",ordered_ids=("a","b"))
    assert a!=cache_key("adversarial",g="G0",space="RaSa",ordered_ids=("a","b"))
    assert a!=cache_key("adversarial",g="G0",space="IRRA",ordered_ids=("b","a"))
    assert cache_key("text",text="T",space="IRRA",g="G0")==cache_key("text",text="T",space="IRRA",g="G1")
    c=ConditionIdentity("a"*64,"a"*64,("x",),(1.,),"b"*64,"c"*64)
    assert c.key!=replace(c,context_caption_sha256="d"*64).key
    assert c.key!=replace(c,scores=(2.,)).key
    assert c.key!=replace(c,anchor_mapping_sha256="e"*64).key
    with pytest.raises(ValueError):cache_key("condition",text="T",space="IRRA")
    with pytest.raises(ValueError):cache_key("reward",text="T",space="IRRA",g="G")
    assert cache_key("reward",text="T",space="IRRA",g="G",reward_context="PID_A")!=cache_key("reward",text="T",space="IRRA",g="G",reward_context="PID_B")

def test_full_bank_and_version():
    ids=tuple(f"q{i}" for i in range(2048))
    b=BankIdentity("G",ids,"clean","adv","IRRA",1,"A1")
    b.validate(ids,"G",1,"A1")
    for bad in (replace(b,ordered_train_ids=ids[:8]),replace(b,g_state_sha256="old"),
                replace(b,round=2),replace(b,arm="B")):
        with pytest.raises(ValueError):bad.validate(ids,"G",1,"A1")

def model_state():
    torch.manual_seed(42)
    g,h=torch.nn.Linear(2,2),torch.nn.Linear(2,2)
    ga=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999))
    ha=torch.optim.Adam(h.parameters(),lr=1e-4)
    for m,opt in [(g,ga),(h,ha)]:
        opt.zero_grad();m(torch.ones(1,2)).square().sum().backward();opt.step()
    gen=torch.Generator().manual_seed(42)
    rng=capture_rng(gen)
    parents={"G":"G60","H":"H10","protocol":"new"}
    cursor={"round":0,"g_updates":484080,"h_visits":20480,"h_updates":20370}
    state=snapshot(g,h,ga,ha,g_rng=rng,h_rng=rng,cursor=cursor,parents=parents)
    return g,h,ga,ha,gen,state

def test_complete_dual_adam_rng_restore():
    g,h,ga,ha,gen,state=model_state()
    expected=[random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),generator=gen))]
    with torch.no_grad():
        for p in (*g.parameters(),*h.parameters()):p.add_(5)
    gr,hr,cursor=restore(state,g,h,ga,ha,expected_parents=state["parents"])
    restore_rng(gr,gen)
    actual=[random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),generator=gen))]
    assert actual==expected and tree_sha(gr)==tree_sha(hr)
    assert cursor==state["cursor"]
    assert tree_sha(ga.state_dict())==tree_sha(state["G_Adam"])

def test_wrong_parent_restore_refused():
    g,h,ga,ha,_,state=model_state()
    with pytest.raises(ValueError):
        restore(state,g,h,ga,ha,expected_parents={**state["parents"],"H":"different"})

def test_gpu_rng_cannot_initialize_cuda():
    _,_,_,_,gen,state=model_state()
    rng=clone_tree(state["G_RNG"]);rng["cuda"]=[torch.tensor([1],dtype=torch.uint8)]
    before=capture_rng(gen)
    with pytest.raises(ValueError):restore_rng(rng,gen)
    assert tree_sha(before)==tree_sha(capture_rng(gen))

def test_storage_clone():
    large=torch.arange(10000)
    result=clone_tree({"row":large[:4]})["row"]
    assert result.untyped_storage().nbytes()==4*large.element_size()

def test_round_atomic_roundtrip_and_immutable(tmp_path):
    *_,state=model_state()
    state["cursor"]["round"]=1
    s=RoundStore(tmp_path/"tbps-framework-fixture"/"rounds","p")
    s.begin("A1",1,"parent")
    ledger={"arm":"A1","round":1,"all_stages_verified":True}
    s.commit("A1",1,state,ledger)
    loaded=s.load("A1",1,expected_parents=state["parents"])
    assert tree_sha(loaded)==tree_sha(state)
    with pytest.raises(ArtifactError):s.begin("A1",1,"parent")
    s.begin("A1",2,tree_sha(state))
    with pytest.raises(ValueError):s.load("A1",1,expected_parents=state["parents"])

def test_interrupted_commit_retained(tmp_path):
    *_,state=model_state();state["cursor"]["round"]=1
    s=RoundStore(tmp_path/"tbps-framework-fixture"/"rounds","p");s.begin("A1",1,"parent")
    def fault(stage):
        if stage=="before_publish":raise RuntimeError("simulated crash")
    with pytest.raises(RuntimeError):
        s.commit("A1",1,state,{"arm":"A1","round":1,"all_stages_verified":True},fault=fault)
    assert s.store.pending()
    with pytest.raises(ValueError):s.begin("A1",1,"parent")
    with pytest.raises(ValueError):s.load("A1",1,expected_parents=state["parents"])

def test_checkpoint_tamper_refused(tmp_path):
    *_,state=model_state();state["cursor"]["round"]=1
    s=RoundStore(tmp_path/"tbps-framework-fixture"/"rounds","p");s.begin("A1",1,"parent")
    s.commit("A1",1,state,{"arm":"A1","round":1,"all_stages_verified":True})
    (s.root/"A1-r1/state.pt").write_bytes(b"tampered fixture")
    with pytest.raises(ArtifactError):s.load("A1",1,expected_parents=state["parents"])

def test_legacy_step_argument_order():
    calls=[]
    def g(*a,**kw):calls.append(a);return {}
    steps=LegacySteps(g,g)
    p=PreparedCondition(None,{"x":1},{})
    steps.update_g("G","Adam","IRRA","image",p)
    assert calls[0]==("G","IRRA","Adam","image",{"x":1})


class FakeBackend:
    execution_kind="synthetic_cpu_fixture"
    def __init__(self,case):
        self.case=case;self.state=None;self.events=[];self.bank=None
    def restore_complete(self,state):
        self.state=clone_tree(state)
    def enter_phase_rng(self,phase):self.events.append(("enter",phase))
    def leave_phase_rng(self,phase):self.events.append(("leave",phase))
    def public_text(self,row):
        a,t,_=self.case
        return {"caption":t.original,"attributes":(a,)}
    def h_rollout(self,public):
        assert set(public)=={"caption","attributes"}
        return self.case[1]
    def condition(self,arm,row,trace):return prepared(self.case,arm=arm)
    def h_fingerprint(self):return str(self.state["h"])
    def g_fingerprint(self):return str(self.state["g"])
    def g_update(self,row,condition):
        self.state["g"]+=1 if condition.payload is self.case[2].payload else 2
        self.events.append(("G",row))
        return {"optimizer_steps":condition.payload["K"]+1}
    def refresh_bank(self,ids,arm,index):
        self.events.append(("bank",tuple(ids)))
        return BankIdentity(self.g_fingerprint(),tuple(ids),"c"*64,"adv","b"*64,index,arm)
    def install_new_reward_bank(self,bank):self.bank=bank
    def h_update(self,row,bank):
        assert bank is self.bank
        self.state["h"]+=self.state["g"]*.1
        self.events.append(("H",row))
        return {"reward":self.state["g"],"optimizer_steps":1}
    def assert_frozen_sources(self):pass
    def complete_state(self,index):
        self.state["cursor"]={"round":index}
        return clone_tree(self.state)

class Journal:
    def __init__(self):self.claims=[];self.commits=[]
    def begin(self,*args):self.claims.append(args)
    def commit(self,*args):self.commits.append(args)

def run_protocol():
    ids=[f"q{i}" for i in range(2048)]
    return {"arms":["A1","B"],"rounds":2,"train_ids":ids,"samples":ids[:8],
            "source_space_sha256":"b"*64,"clean_gallery_sha256":"c"*64}

def fixture_parent():
    parent=model_state()[-1]
    parent["parents"]["protocol"]=digest(run_protocol())
    parent.update(g=0.,h=0.)
    return parent

def test_real_two_round_control_flow(case):
    b=FakeBackend(case);j=Journal();parent=fixture_parent();parent_sha=tree_sha(parent)
    result=Prototype(run_protocol(),b,j).run(parent)
    assert len(j.claims)==len(j.commits)==4
    assert j.claims[0][2]==j.claims[2][2]  # 两臂相同完整父状态摘要
    assert len([e for e in b.events if e[0]=="G"])==32
    assert len([e for e in b.events if e[0]=="H"])==32
    banks=[e for e in b.events if e[0]=="bank"]
    assert len(banks)==4 and all(len(e[1])==2048 for e in banks)
    assert result[("A1",1)]["H"][0]["reward"]!=result[("B",1)]["H"][0]["reward"]
    assert tree_sha(parent)==parent_sha
    # 每轮完整顺序G*8 -> bank*2048 -> H*8，无partial bank。
    substantive=[e[0] for e in b.events if e[0] in ("G","bank","H")]
    assert substantive==(["G"]*8+["bank"]+["H"]*8)*4

@pytest.mark.parametrize("failure",["public","G_steps","stale_bank","partial_bank","wrong_space","wrong_clean","H_changes_G"])
def test_composed_failure_stops_without_round_commit(case,failure):
    b=FakeBackend(case);j=Journal()
    if failure=="public":
        b.public_text=lambda row:{"caption":"text","attributes":(),"pid":1}
    elif failure=="G_steps":
        b.g_update=lambda *a:{"optimizer_steps":0}
    elif failure in ("stale_bank","partial_bank","wrong_space","wrong_clean"):
        original=b.refresh_bank
        def bad(ids,arm,index):
            bank=original(ids,arm,index)
            if failure=="stale_bank":return replace(bank,g_state_sha256="stale")
            if failure=="wrong_space":return replace(bank,source_space_sha256="APTM")
            if failure=="wrong_clean":return replace(bank,clean_sha256="other")
            return replace(bank,ordered_train_ids=tuple(ids[:8]))
        b.refresh_bank=bad
    else:
        original=b.h_update
        def bad(*a):
            b.state["g"]+=1
            return original(*a)
        b.h_update=bad
    with pytest.raises(ValueError):
        Prototype(run_protocol(),b,j).run(fixture_parent())
    assert not j.commits and len(j.claims)==1

def test_gpu_backend_not_open(case):
    b=FakeBackend(case);b.execution_kind="frozen_legacy_gpu"
    with pytest.raises(ValueError,match="not authorized"):
        Prototype(run_protocol(),b,Journal())

def test_original_losses_gradient_difference_and_noop(case):
    # 只读加载原Loss纯函数；合成feature参数，不加载G/IRRA。
    import importlib.util
    path=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/losses.py")
    spec=importlib.util.spec_from_file_location("gh_loss_cpu_fixture",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    def gradient(condition):
        p=torch.tensor([.4,.6]+[0.]*510,requires_grad=True)
        feature=torch.nn.functional.normalize(p,dim=0)[None]
        terms=mod.joint_losses(feature,condition.payload["z_t"][None],
                               condition.payload["v_local"][None],torch.tensor([True]))
        (terms["global"]+terms["local"]).backward()
        return p.grad.clone()
    original=case[2];changed=prepared(case)
    noop=prepared(case,trace=replace(case[1],adversarial=case[1].original,edits=()))
    assert torch.equal(gradient(original),gradient(noop))
    assert not torch.equal(gradient(original),gradient(changed))

def test_legacy_kernel_calls_real_stages_with_synthetic_transport(case):
    a,t,original=case;counter=[0];calls=[]
    class Transport:
        def __init__(self,caption):
            calls.append(("text",caption));counter[0]+=2
            self.clean=original.payload["z_t"][None];self.H=torch.ones(77,1,512)
        def read(self,attr):
            calls.append(("JVP",attr.instance_id));counter[0]+=3
            return self.clean.clone(),dict(instance_id=attr.instance_id,zero_exact=True,central_fd_relative_error=.001)
    def locate(text,hidden,mappings,old):
        calls.append(("locate",text));assert hidden.shape==(77,1,512)
        return (SimpleNamespace(instance_id="G-ID"),),dict(context_caption_sha256=text_sha(text),
            fresh_localization=True,truncated=False,unresolved=False)
    def build(identity,ids,scores,weights,jvps,z):
        return {**original.payload,**identity,"selected_instance_ids":ids,"attribute_jvps":jvps,"z_t":z}
    kernel=LegacyContextKernel(transport_factory=Transport,locate_and_plan=locate,build=build,
                              source_space_sha256=original.identity.source_space_sha256,text_counter=lambda:counter[0])
    result=prepared(case,kernel=kernel)
    assert [e[0] for e in calls]==["text","locate","JVP"]
    assert result.evidence["new_text_encodings"]==5

@pytest.mark.parametrize("g_free,util,ram,disk,expected",[
    (18433,74,64,40,True),(18433,90,64,40,True),(6143,0,64,40,False),
    (18433,0,1,40,False),(18433,0,64,1,False)])
def test_resource_snapshot_shared_policy(monkeypatch,g_free,util,ram,disk,expected):
    import tbps.alternating.preflight as pf
    commands=[]
    def command(args,**kw):
        commands.append(args)
        return f"RTX 4090, {g_free}, 24564, {util}" if "--id=0" in args else "GPU-id, 123, other-user, 6000"
    monkeypatch.setattr(pf.subprocess,"check_output",command)
    orig=Path.read_text
    monkeypatch.setattr(Path,"read_text",lambda self,*a,**k:f"MemAvailable: {ram*2**20} kB" if str(self)=="/proc/meminfo" else orig(self,*a,**k))
    monkeypatch.setattr(pf.shutil,"disk_usage",lambda p:SimpleNamespace(free=disk*2**30))
    monkeypatch.setattr(pf.os,"statvfs",lambda p:SimpleNamespace(f_favail=20000))
    cfg={"resource_probe_directory":"/home/lzf/ldx",
         "proposed_gpu_limits":{"minimum_free_mib":6144,"minimum_ram_bytes":32*2**30,
                               "minimum_disk_bytes":25*2**30,"minimum_free_inodes":10000}}
    result=pf.resource_snapshot(cfg)
    assert result["resource_eligible"] is expected
    assert result["GPU_authorized"] is False
    assert len(commands)==2 and all(c[0]=="nvidia-smi" for c in commands)

def test_resource_read_error_refuses(monkeypatch):
    import tbps.alternating.preflight as pf
    def fail(*a,**k):raise RuntimeError("nvidia-smi failed")
    monkeypatch.setattr(pf.subprocess,"check_output",fail)
    with pytest.raises(RuntimeError):pf.resource_snapshot({})

def test_source_sha_tamper_fixture(tmp_path):
    from tbps.alternating.preflight import verify_refs
    from tbps.artifacts import sha_file
    p=tmp_path/"source.txt";p.write_text("source")
    ref={"path":str(p),"sha256":sha_file(p)}
    assert len(verify_refs([ref]))==1
    p.write_text("changed fixture")
    with pytest.raises(ArtifactError):verify_refs([ref])


def test_original_reward_changes_with_new_bank_not_policy_input(monkeypatch):
    import importlib.util,sys
    path=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/rewards.py")
    name="gh_original_reward_cpu_fixture"
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,name,mod);spec.loader.exec_module(mod)
    e=torch.eye(512)[:2]
    new=torch.nn.functional.normalize(e+torch.tensor([.3,.2])[:,None]*e.flip(0),dim=1)
    texts=torch.stack((e[0],e[1],torch.nn.functional.normalize(e[0]+e[1],dim=0)))
    def rewards(adv):
        bank=mod.TrainGallery(e,adv,torch.tensor([True,False]),0,"train",("train","train"),"train",("train","train"))
        return mod.RewardEvaluator(bank,lambda captions:texts,1.,1.).evaluate_many("original",("sampled","greedy"))
    old_reward=rewards(e);fresh_reward=rewards(new)
    assert old_reward[0].total!=fresh_reward[0].total
    assert old_reward[0].total-old_reward[1].total!=fresh_reward[0].total-fresh_reward[1].total
    assert torch.equal(texts[0],e[0])  # 新G只改变bank，不改变本次文本或公开policy输入

@pytest.mark.parametrize("bad",[False,True])
def test_canonical_locator_reuses_rules_not_old_localization(case,bad):
    from dataclasses import dataclass,asdict
    from tbps.alternating.conditions import CanonicalLocator
    a,t,condition=case
    @dataclass(frozen=True)
    class Instance:
        semantic_axis:str="color"
        canonical_value:str="blue"
        region:str="upper"
        raw_text:str="blue"
        char_start:int=0
        char_end:int=4
        linked_object:object=None
        semantic_representation:object="old"
        representation_status:str="ready"
        @property
        def instance_id(self):return "G-ID"
        def to_dict(self):
            return {**asdict(self),"source_span":{"start":0,"end":4,"raw":self.raw_text}}
    calls=[]
    class Localizer:
        def attach(self,instances,caption,H,cache_index):
            calls.append((caption,cache_index))
            assert instances[0].raw_text=="bluе"
            assert instances[0].semantic_representation is None
            fresh=replace(instances[0],semantic_representation={"projected_context":[1.]},
                          representation_status="unavailable" if bad else "ready")
            return (fresh,),{"G-ID":torch.ones(77)/77**.5},{"status":"valid"},[]
    oldplan=SimpleNamespace(instance_id="G-ID",metadata={"S_raw":2.},selection_rank=1,Q=torch.eye(512))
    def selected(md,rank,w,q):return SimpleNamespace(metadata=md,selection_rank=rank,w=w,Q=q)
    locator=CanonicalLocator(t.original,(Instance(),),(oldplan,),Localizer(),selected)
    if bad:
        with pytest.raises(ValueError):
            locator(t.adversarial,torch.zeros(77,1,512),(t.mapping(a),),condition)
    else:
        plan,audit=locator(t.adversarial,torch.zeros(77,1,512),(t.mapping(a),),condition)
        assert calls==[(t.adversarial,None)]
        assert plan[0].metadata["instance_id"]=="G-ID"
        assert plan[0].metadata["raw_text"]=="bluе"
        assert audit["fresh_localization"] and plan[0].Q is oldplan.Q

def test_next_update_after_restore_matches_continuous():
    g,h,ga,ha,gen,state=model_state()
    def step(model,adam):
        adam.zero_grad()
        loss=model(torch.rand(1,2)).square().sum()
        loss.backward();adam.step()
    restore_rng(state["G_RNG"],gen);step(g,ga)
    restore_rng(state["H_RNG"],gen);step(h,ha)
    expected=[tree_sha(x.state_dict()) for x in (g,h,ga,ha)]
    restore(state,g,h,ga,ha,expected_parents=state["parents"])
    restore_rng(state["G_RNG"],gen);step(g,ga)
    restore_rng(state["H_RNG"],gen);step(h,ha)
    assert [tree_sha(x.state_dict()) for x in (g,h,ga,ha)]==expected

def test_interrupted_stages_do_not_commit_or_retry(case):
    b=FakeBackend(case);j=Journal()
    def fail(*args):
        raise RuntimeError("fixture source failure")
    b.refresh_bank=fail
    with pytest.raises(RuntimeError):
        Prototype(run_protocol(),b,j).run(fixture_parent())
    assert len(j.claims)==1 and not j.commits
    assert len([e for e in b.events if e[0]=="G"])==8
    assert not [e for e in b.events if e[0]=="H"]


def test_imports_are_inert_in_fresh_process():
    import os,subprocess
    script="import sys; import tbps.alternating.contracts, tbps.alternating.conditions, tbps.alternating.state, tbps.alternating.runner, tbps.alternating.preflight; assert 'torch' not in sys.modules"
    result=subprocess.run([sys.executable,"-B","-c",script],env=os.environ,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_claim_protocol_cannot_be_rebound(tmp_path):
    *_,state=model_state();state["cursor"]["round"]=1
    root=tmp_path/"tbps-framework-fixture"/"rounds"
    RoundStore(root,"first").begin("A1",1,"parent")
    with pytest.raises(ArtifactError):
        RoundStore(root,"second").commit("A1",1,state,{"arm":"A1","round":1,"all_stages_verified":True})


def test_native_full_action_validation_precedes_bridge(case):
    from dataclasses import dataclass
    from tbps.alternating.conditions import trace_from_legacy
    a,t,_=case
    @dataclass
    class NativeEdit:
        attribute_id:str=a.h_id
        position:int=3
        original_char:str="e"
        replacement_char:str="е"
        original_codepoint:str="U+0065"
        replacement_codepoint:str="U+0435"
        original_utf8_offset:int=3
    example=SimpleNamespace(caption=t.original,attributes=(a,))
    edits=(NativeEdit(),);seen=[]
    def native_replay(example,records,budget):
        seen.append(records);assert records[0].replacement_codepoint=="U+0435"
        return t.adversarial
    trace=trace_from_legacy(example,t.adversarial,edits,1,{"decisions":["replace"]},replay=native_replay)
    trace.validate({("e","е")})
    assert seen==[edits] and trace.adversarial==t.adversarial

def test_native_replay_failure_never_relabelled(case):
    from tbps.alternating.conditions import trace_from_legacy
    def native_replay(*args):raise ValueError("original codepoint gate")
    with pytest.raises(ValueError,match="original codepoint"):
        trace_from_legacy(None,"",(),0,{},replay=native_replay)


def test_stage_rng_domains_remain_independent():
    *_,sampler,state=model_state()
    g_rng=clone_tree(state["G_RNG"])
    random.seed(99);np.random.seed(99);torch.manual_seed(99);sampler.manual_seed(99)
    h_rng=capture_rng(sampler)
    restore_rng(g_rng,sampler)
    g_draw=(random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),generator=sampler)))
    restore_rng(h_rng,sampler)
    h_draw=(random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),generator=sampler)))
    assert g_draw!=h_draw
    restore_rng(g_rng,sampler)
    assert g_draw==(random.random(),float(np.random.rand()),float(torch.rand(())),float(torch.rand((),generator=sampler)))

def test_incomplete_parent_stops_before_claim(case):
    j=Journal()
    with pytest.raises(ValueError,match="complete G/H"):
        Prototype(run_protocol(),FakeBackend(case),j).run({"g":0.,"h":0.})
    assert not j.claims


def test_nested_round_storage_and_native_log_payload(tmp_path):
    base=tmp_path/"tbps-framework-fixture"
    store=RoundStore(base/"smoke_v1"/"A1"/"rounds","p",new_base=base)
    store.begin("A1",1,"parent")
    assert (store.root/"A1-r1-claim"/"completed.json").is_file()
    def g(*args,**kwargs):
        return {"optimizer_steps":1,"serial_adversarial":torch.ones(1),"joint_adversarial":torch.ones(1)}
    def h(*args,**kwargs):
        return {"loss":1.},SimpleNamespace(not_json_serializable=torch.ones(1))
    steps=LegacySteps(g,h)
    assert steps.update_g(None,None,None,None,PreparedCondition(None,{},{}))=={"optimizer_steps":1}
    assert steps.update_h(None,None,None,None,None,1,"q")=={"loss":1.}
