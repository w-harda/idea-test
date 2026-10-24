"""B正式调度CPU Fixture：真实小Adam状态+不可变Artifact，不接真实模型/CUDA。"""
import copy
import random
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tbps.artifacts import ArtifactStore,FileRef,json_bytes,sha_file
from tbps.alternating.state import snapshot,capture_rng,restore,tree_sha,clone_tree
from tbps.alternating.contracts import BankIdentity
from tbps.alternating.formal_state import FormalJournal,run_rounds
from tbps.alternating.formal_sources import FormalGrant
from tbps.alternating.formal_evaluation import asr_at_r1

@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def deny(*a,**k):raise AssertionError("CPU test attempted CUDA")
    monkeypatch.setattr(torch.cuda,"_lazy_init",deny)
    monkeypatch.setattr(torch.cuda,"init",deny)
    torch.set_num_threads(1)

class Fake:
    def __init__(self,root):
        self.root=Path(root);random.seed(42);np.random.seed(42);torch.manual_seed(42);self.p={"train_ids":["x","y"],"source_space_sha256":"a"*64,"clean_gallery_sha256":"b"*64,"rounds":2,"samples":["x","y"]}
        self.g=torch.nn.Linear(1,1,bias=False);self.h=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():self.g.weight.fill_(.1);self.h.weight.fill_(.2)
        self.ga=torch.optim.Adam(self.g.parameters(),lr=.01);self.ha=torch.optim.Adam(self.h.parameters(),lr=.01)
        self.sampler=torch.Generator().manual_seed(42);rng=capture_rng(self.sampler)
        self.rng={"G":clone_tree(rng),"H":clone_tree(rng)}
        self.cursor={"round":0,"g_updates":0,"h_visits":0,"h_updates":0}
        self.parents={"G":"g","H":"h","protocol":"p"};self.restore_calls=0;self.counts={"G":0};self.seen=[]
    def complete_state(self,index):
        self.cursor["round"]=index
        return snapshot(self.g,self.h,self.ga,self.ha,g_rng=self.rng["G"],h_rng=self.rng["H"],cursor=self.cursor,parents=self.parents)
    def restore_complete(self,state):
        self.restore_calls+=1;self.rng["G"],self.rng["H"],self.cursor=restore(state,self.g,self.h,self.ga,self.ha,expected_parents=self.parents)
    def begin_round(self,i):self.i=i;self.conditions_log=[]
    def enter_phase_rng(self,n):pass
    def leave_phase_rng(self,n):pass
    def public_text(self,row):return row
    def h_rollout(self,row):return float(self.h.weight.detach()[0,0])
    def condition(self,arm,row,trace):
        assert arm=="B";self.seen.append((self.i,row,trace))
        self.conditions_log.append({"H":trace,"row":row,"payload_content_sha256":tree_sha({"K":0,"fixture":torch.ones(1)}),
            "identity":{"original_caption_sha256":"d"*64,"context_caption_sha256":"d"*64,
            "selected_g_ids":[],"scores":[],"source_space_sha256":"a"*64,"anchor_mapping_sha256":"e"*64},
            "evidence":{"new_JVPs":0,"new_text_encodings":0}})
        return SimpleNamespace(payload={"K":0,"signal":trace})
    def g_fingerprint(self):return tree_sha(self.g.state_dict())
    def h_fingerprint(self):return tree_sha(self.h.state_dict())
    def g_update(self,row,c):
        self.ga.zero_grad();(self.g.weight.sum()-c.payload["signal"]).square().backward();self.ga.step()
        self.cursor["g_updates"]+=1;self.counts["G"]+=1
        return {"optimizer_steps":1}
    def refresh_bank(self,ids,arm,index):
        from dataclasses import asdict
        from tbps.alternating.formal_state import tensor_bytes
        assert list(ids)==self.p["train_ids"]
        features=torch.eye(len(ids),512)
        self.bank=BankIdentity(self.g_fingerprint(),tuple(ids),"b"*64,tree_sha(features),"a"*64,index,arm)
        p=self.root/"B"/f"bank_{index}";base=next(x for x in [self.root,*self.root.parents] if x.name.startswith("tbps-framework"))
        s=ArtifactStore(p,new_base=base)
        s.publish("G_state",{"state.pt":tensor_bytes(self.g.state_dict())},{"G":self.g_fingerprint()})
        for j,row in enumerate(ids):
            z=torch.zeros(1,3,384,128)
            payload={"original":z,"adversarial":z.clone(),"effective_delta":z.clone(),"feature":features[j:j+1]}
            s.publish(f"claim_{j:04d}",{"claim.json":json_bytes({"id":row})},{"G":self.g_fingerprint()})
            s.publish(f"image_{j:04d}",{"data.pt":tensor_bytes(payload)},
                {"row":row,"index":j,"G":self.g_fingerprint(),"content_sha256":tree_sha(payload)})
        s.publish("completed",{"identity.json":json_bytes(asdict(self.bank)),"features.pt":tensor_bytes(features)},{"fixture":True})
        self.counts["G"]+=len(ids)
        return self.bank
    def bank_ref(self,i):
        p=self.root/"B"/f"bank_{i}"/"completed/manifest.json";return FileRef(str(p),sha_file(p))
    def install_new_reward_bank(self,b):assert b.g_state_sha256==self.g_fingerprint()
    def h_update(self,row,bank):
        assert bank==self.bank
        self.ha.zero_grad();(self.h.weight.sum()-float(self.g.weight.detach().sum())).square().backward();self.ha.step()
        self.cursor["h_visits"]+=1;self.cursor["h_updates"]+=1
        return {"optimizer_steps_this_step":1}
    def assert_frozen_sources(self):pass

def launch(tmp_path,stop=False):
    base=tmp_path/"tbps-framework-fixture";b=Fake(base/"banks");state=b.complete_state(0);j=FormalJournal(base/"train","a"*64,base)
    j.initialize({"members":["x","y"]},state)
    result,done=run_rounds(b,j,state,2,lambda i:(["x","y"],["y","x"]),stop_requested=lambda:stop)
    return b,j,result,done

def test_round2_carries_real_updated_H_and_G(tmp_path):
    b,j,state,done=launch(tmp_path)
    assert done and b.restore_calls==1
    assert b.seen[0][2]!=b.seen[2][2]
    assert state["cursor"]=={"round":2,"g_updates":4,"h_visits":4,"h_updates":4}
    l1=json.loads((j.root/"round_0001/ledger.json").read_text());l2=json.loads((j.root/"round_0002/ledger.json").read_text())
    assert l1["G_after"]==l2["G_before"] and l1["H_after"]==l2["H_before"]
    assert tree_sha(state["G_Adam"])!=tree_sha(state["H_Adam"])

def test_explicit_boundary_resume_matches_continuous_CPU(tmp_path):
    a,j,state,done=launch(tmp_path/"split",True);assert not done
    value,ledger=j.load_boundary(1,a.parents)
    assert tree_sha(value)==tree_sha(state)
    b=Fake(tmp_path/"split/tbps-framework-fixture/banks");j.record_resume(1)
    resumed,done=run_rounds(b,j,value,2,lambda i:(["x","y"],["y","x"]),start=1)
    c,_,expected,_=launch(tmp_path/"continuous")
    assert done and tree_sha(resumed)==tree_sha(expected)
    assert resumed["G_RNG"].keys()=={"python","numpy","torch_cpu","cuda","sampler"}

def test_uncommitted_later_claim_blocks_resume(tmp_path):
    b,j,state,_=launch(tmp_path,True);j.begin(2,state)
    with pytest.raises(ValueError,match="Claim"):j.load_boundary(1,b.parents)

def test_duplicate_round_and_root_refused(tmp_path):
    b,j,state,_=launch(tmp_path,True)
    with pytest.raises(Exception):j.begin(1,state)
    with pytest.raises(ValueError):j.initialize({},state)

def test_corrupt_checkpoint_rejected(tmp_path):
    b,j,state,_=launch(tmp_path,True)
    p=j.root/"round_0001/state.pt";p.write_bytes(p.read_bytes()+b"corrupt")
    with pytest.raises(Exception):j.load_boundary(1,b.parents)

def test_wrong_parent_rejected(tmp_path):
    b,j,state,_=launch(tmp_path,True)
    with pytest.raises(ValueError,match="source"):j.load_boundary(1,{"G":"wrong"})

def test_no_rewind_completed_training(tmp_path):
    b,j,_,_=launch(tmp_path)
    with pytest.raises(ValueError):j.load_boundary(1,b.parents)

def test_source_capability_rejects_cross_scope_and_mutation():
    c={"formal_binding":{"scope":"pilot"}}
    g=FormalGrant(c,"pilot",allow_pilot=True);g.validate(c,"smoke")
    d=copy.deepcopy(c);d["changed"]=1
    with pytest.raises(ValueError):g.validate(d,"smoke")
    with pytest.raises(ValueError):FormalGrant(c,"train",allow_training=True,ready=False)
    with pytest.raises(ValueError):FormalGrant(c,"pilot")

def test_failed_round_not_committed(tmp_path):
    base=tmp_path/"tbps-framework-fixture";b=Fake(base/"banks");state=b.complete_state(0);j=FormalJournal(base/"train","a"*64,base);j.initialize({},state)
    def fail(*a):raise RuntimeError("fixture interrupted")
    b.h_update=fail
    with pytest.raises(RuntimeError):run_rounds(b,j,state,2,lambda i:(["x","y"],["x","y"]))
    assert (j.root/"round_0001_claim").exists() and not (j.root/"round_0001").exists()
    with pytest.raises(Exception):j.load_boundary(1,b.parents)

def test_asr_conditional_denominator_and_same_pid():
    from tbps.contracts import EvaluationIdentity,PersonID
    p=tuple(PersonID("cuhk_pedes_fixture",str(i)) for i in range(3))
    ids=EvaluationIdentity(("q0","q1","q2"),("g0","g1","g2"),p,p)
    clean=torch.tensor([[3.,1,0],[0,3,1],[3,1,0]])
    attack=torch.tensor([[1.,3,0],[0,3,1],[0,1,3]])
    result=asr_at_r1(clean,attack,ids)
    assert result["denominator_clean_correct"]==2 and result["successes"]==1 and result["ASR@R1"]==50
    assert result["all_query_count"]==3

def test_asr_undefined_if_no_clean_correct():
    from tbps.contracts import EvaluationIdentity,PersonID
    p=(PersonID("cuhk_pedes_fixture","1"),PersonID("cuhk_pedes_fixture","2"))
    ids=EvaluationIdentity(("a","b"),("a","b"),p,p)
    x=torch.tensor([[0.,1],[1,0]])
    assert asr_at_r1(x,x,ids)["ASR@R1"] is None

def test_evaluation_authorization_before_any_test_work(monkeypatch):
    import tbps.alternating.formal_evaluation as ev
    monkeypatch.setattr("tbps.alternating.formal_sources.prepare_cpu",lambda *a:({"evaluation":{"authorized":False}},None,None,None,{}))
    with pytest.raises(ValueError,match="not authorized"):ev.run("unused","baseline",allow=True)

def test_common_IRRA_four_mode_engine_and_output_contract(tmp_path):
    from dataclasses import asdict
    from tbps.contracts import DatasetManifest,GalleryItem,QueryItem,PersonID,AttackBudget,ImageBudget,TextBudget
    from tbps.models.base import ModelSpace,InputBinding
    from tbps.engines.materialized_cache import FeatureCache
    from tbps.engines.provenance import AttackOrigin
    from tbps.alternating.formal_evaluation import score_cached,utf8
    from tbps.artifacts import digest
    base=tmp_path/"tbps-framework-eval-fixture";base.mkdir()
    annotation=base/"annotation.json";annotation.write_text("CPU fixture")
    ck=base/"checkpoint.bin";ck.write_bytes(b"fixture, no model")
    rf=lambda p:FileRef(str(p),sha_file(p))
    gs=tuple(GalleryItem("g"+str(i),str(i)+".png",digest(["original",i]),PersonID("cuhk_pedes_fixture",str(i//2)),"test") for i in range(10))
    qs=tuple(QueryItem("q"+str(i),"caption "+str(i),0,"g"+str(i),gs[i].pid,"test") for i in range(10))
    manifest=DatasetManifest("cuhk_pedes_fixture","test",rf(annotation),gs,qs)
    space=ModelSpace("irra_fixture",sha_file(ck),*(digest(i) for i in range(4)),"fp32/math/no_tf32")
    store=ArtifactStore(base/"features",new_base=base);protocol={"fixture":True}
    store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"fixture":True})
    cache=FeatureCache(store,protocol,space)
    originals={"image":tuple(g.image_sha256 for g in gs),"text":tuple(utf8(q.caption) for q in qs)}
    mh=digest(asdict(manifest));bindings={};origins={}
    z=torch.eye(10,512)
    payloads={"clean_gallery":z,"clean_text":z,"adv_gallery":torch.roll(z,2,0),"adv_text":torch.roll(z,4,0)}
    for family,x in payloads.items():
        modality="image" if family.endswith("gallery") else "text"
        ids=tuple(g.image_id for g in gs) if modality=="image" else tuple(q.query_id for q in qs)
        content=originals[modality] if family.startswith("clean") else tuple(digest([family,i]) for i in range(10))
        b=InputBinding(space.id,mh,content,modality,ids);bindings[family]=b;cache.publish(family,b,0,10,x)
        if family.startswith("adv"):
            budget=AttackBudget(image=ImageBudget("linf",8/255,"RGB [0,1]")) if modality=="image" else AttackBudget(text=TextBudget("unicode_character",None,"original Natural-K","fixture",1,1))
            origins[family]=AttackOrigin("fixture-"+family,rf(ck),rf(annotation),budget,digest("surrogate"))
    out=ArtifactStore(base/"results",new_base=base)
    result=score_cached(manifest,store,protocol,space,rf(ck),bindings,origins,originals,out,fixture=True)
    assert list(result["groups"])==["Clean","Image-only","Text-only","Joint"]
    assert result["groups"]["Clean"]["R@1"]==100
    assert result["ASR"]["Image-only"]["ASR@R1"]==100
    assert all(set(v)=={"R@1","R@5","R@10","mAP"} for v in result["groups"].values())
    assert (out.root/"completed/metrics_2dp.csv").is_file()
    assert "mINP" not in (out.root/"completed/metrics.csv").read_text()

def test_historical_index_extra_metadata_does_not_bypass_SHA(tmp_path):
    from tbps.alternating.formal_cli import locked_ref
    from tbps.artifacts import checked
    p=tmp_path/"evidence";p.write_bytes(b"original")
    d={"path":str(p),"sha256":sha_file(p),"metadata":[8,123,456]}
    assert checked(locked_ref(d))
    p.write_bytes(b"changed")
    with pytest.raises(Exception):checked(locked_ref(d))

@pytest.mark.parametrize("key",["new_G_updates","new_H_updates","gallery_images","final_H_visits","final_G_updates"])
def test_budget_mislabel_refused(key):
    from tbps.alternating.formal_sources import validate_budget
    members={"Train":2048,"G_updates_per_round":8068,"eligible_H":2037,"G_K_total":6020}
    f={"train_rounds":50,"budget":{"G_parent_absolute_epoch":60,"H_parent_absolute_epoch":10,"seed":42,
       "G_samples_per_round":2048,"G_updates_per_round":8068,"H_visits_per_round":2048,
       "H_valid_updates_per_round":2037,"gallery_images_per_round":2048,"new_G_updates":403400,
       "new_H_visits":102400,"new_H_updates":101850,"final_G_updates":887480,
       "final_H_visits":122880,"final_H_updates":122220,"full_gallery_refreshes":50,
       "gallery_images":102400,"new_JVP_upper_bound":301000}}
    assert validate_budget(f,members)["new_G_updates"]==403400
    f["budget"][key]+=1
    with pytest.raises(ValueError,match="budget"):validate_budget(f,members)

def test_no_READY_cannot_enable_formal_training(tmp_path):
    from tbps.alternating.formal_cli import ready_for_training
    assert ready_for_training({"ready_receipt":str(tmp_path/"absent")}) is False

@pytest.mark.parametrize("damage",[False,True])
def test_independent_audit_complete_artifact_flow(tmp_path,monkeypatch,damage):
    from tbps.artifacts import digest
    from tbps.alternating import formal_audit
    base=tmp_path/"tbps-framework-audit-fixture";scope=base/"pilot";cfg=tmp_path/"config.json";cfg.write_text("CPU fixture")
    backend=Fake(scope);backend.parents["protocol"]=digest(backend.p)
    backend.rng["G"]["cuda"]=[torch.zeros(8,dtype=torch.uint8)]
    backend.rng["H"]["cuda"]=[torch.ones(8,dtype=torch.uint8)]
    parent=backend.complete_state(0)
    journal=FormalJournal(scope/"rounds",sha_file(cfg),base);journal.initialize({"scope":"pilot"},parent)
    state,done=run_rounds(backend,journal,parent,2,lambda i:(["x","y"],["y","x"]))
    store=ArtifactStore(scope,new_base=base)
    c={"prototype":backend.p,"sources":{"G60":{"sha256":"g"},"H10":{"sha256":"h"}}}
    from tbps.alternating.native_sources import source_key
    from tbps.alternating.formal_state import tensor_bytes
    for i in (1,2):
        ledger=json.loads((scope/f"rounds/round_{i:04d}/ledger.json").read_text())
        io=ArtifactStore(scope/"round_io"/f"round_{i:04d}",new_base=base)
        for j,condition in enumerate(ledger["conditions"]):
            io.publish(f"rollout_{j:04d}",{"action.json":json_bytes({"CPU_fixture":True})},
                {"source":source_key(c),"action":"f"*64})
            io.publish(f"condition_{j:04d}",{"condition.pt":tensor_bytes({"K":0,"fixture":torch.ones(1)}),
                "evidence.json":json_bytes(condition)},{"source":source_key(c),"condition":digest(condition["identity"])})
    store.publish("generated",{"report.json":json_bytes({"complete":True,"frozen_sources":True,
        "scope":"pilot","status":"GH_B_FORMAL_PILOT_GENERATED","native_source":source_key(c),
        "counts":{"G":backend.counts["G"],"IRRA_image":backend.counts["G"]}})},
        {"formal_config":sha_file(cfg),"scope":"pilot"})
    f={"output_base":str(base),"outputs":{"pilot":str(scope)},"audits":{"pilot":str(base/"audit")},"producers":{}}
    c={"prototype":backend.p,"sources":{"G60":{"sha256":"g"},"H10":{"sha256":"h"}}}
    monkeypatch.setattr(formal_audit,"source_preflight",lambda *a:(f,c,{"formal_config_sha256":sha_file(cfg)}))
    monkeypatch.setattr(formal_audit,"history",lambda *a:{"CPU_fixture":True})
    if damage:
        (scope/"B/bank_2/image_0001/data.pt").write_bytes(b"damaged in Fixture only")
        with pytest.raises(Exception):formal_audit.audit(str(cfg),"pilot")
        assert not (base/"audit/completed").exists()
    else:
        result=formal_audit.audit(str(cfg),"pilot")
        assert result["status"]=="GH_B_FORMAL_PILOT_CPU_AUDIT_PASS"
        assert len(result["rounds"])==2 and (base/"readiness/ready.json").is_file()

def test_progress_exposes_each_round_stage_without_extra_updates(tmp_path):
    base=tmp_path/"tbps-framework-progress";b=Fake(base/"scope");seen=[]
    b.progress=lambda label,index,visited,total:seen.append((label,index))
    parent=b.complete_state(0);j=FormalJournal(base/"scope/rounds","a"*64,base);j.initialize({},parent)
    state,done=run_rounds(b,j,parent,2,lambda i:(["x","y"],["y","x"]))
    expected=["H_ROLLOUT","NEW_CONTEXT_JVP","G_TRAIN","FULL_TRAIN_BANK","H_SCST_TRAIN"]
    assert [x[0] for x in seen]==expected*2 and done
    assert state["cursor"]["g_updates"]==4 and state["cursor"]["h_updates"]==4

def test_common_materialization_claim_and_image_only_caption_only_interfaces(tmp_path,monkeypatch):
    from PIL import Image
    from tbps.contracts import DatasetManifest,GalleryItem,QueryItem,PersonID
    from tbps.alternating.formal_evaluation import materialize
    base=tmp_path/"tbps-framework-materialize-fixture";base.mkdir()
    annotation=base/"annotations.json";annotation.write_text("CPU fixture")
    images=base/"images";images.mkdir()
    gs=[];qs=[];calls={"G":0,"H":0}
    for j in range(2):
        path=images/f"{j}.png";Image.new("RGB",(128,384),(20+j,30,40)).save(path)
        pid=PersonID("cuhk_pedes_fixture",str(j))
        gs.append(GalleryItem(f"g{j}",path.name,sha_file(path),pid,"test"))
        qs.append(QueryItem(f"q{j}","blue shirt "+str(j),0,f"g{j}",pid,"test"))
    m=DatasetManifest("cuhk_pedes_fixture","test",FileRef(str(annotation),sha_file(annotation)),tuple(gs),tuple(qs))
    original_to=torch.Tensor.to
    def cpu_device(t,*args,**kwargs):
        if args and args[0]=="cuda:0":args=("cpu",*args[1:])
        return original_to(t,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,"to",cpu_device)
    class G(torch.nn.Module):
        def forward(self,x):
            calls["G"]+=1;assert x.shape==(1,3,384,128) and x.dtype==torch.float32
            return torch.zeros_like(x)
    class H(torch.nn.Module):
        def attack(self,example):
            calls["H"]+=1;assert set(vars(example))=={"caption","attributes"}
            return SimpleNamespace(adversarial=example.caption,edits=(),budget_k=0,record=lambda:{"CPU_fixture":True})
    def generate(g,x):
        delta=g(x);return {"adversarial":x+delta,"effective_delta":delta}
    native={"irra_image_adapter":SimpleNamespace(pixels_from_pil=lambda im:torch.ones(3,384,128)*.25),
        "inference_once":SimpleNamespace(generate_once=generate),
        "constraints":SimpleNamespace(replay=lambda example,edits,k:example.caption)}
    b=SimpleNamespace(g=G(),h=H(),bindings=SimpleNamespace(get=native.__getitem__),
        frontend=SimpleNamespace(convert=lambda text:(SimpleNamespace(caption=text,attributes=()),{"input":"caption-only"})))
    c={"output_base":str(base),"prototype":{"image_root":str(images)}}
    store=materialize(m,b,c,base/"attacks")
    store.verify("claim");store.verify("completed")
    assert (store.root/"claim/dataset_manifest.json").is_file()
    assert calls=={"G":2,"H":2}
    value=torch.load(store.root/"image_0000/data.pt",map_location="cpu",weights_only=True)
    assert value["original"].dtype==value["adversarial"].dtype==torch.float32
    assert value["effective_delta"].abs().max()==0
    with pytest.raises(ValueError,match="immutable"):materialize(m,b,c,base/"attacks")

def test_comparison_uses_precommitted_final_B_checkpoint(tmp_path):
    from tbps.alternating.formal_evaluation import comparison_checkpoint
    p=tmp_path/"round_0050.pt";p.write_bytes(b"CPU fixture")
    ref=FileRef(str(p),sha_file(p))
    f={"references":{"comparison_B_checkpoint":{"path":ref.path,"sha256":ref.sha256}}}
    assert comparison_checkpoint(f,"B",None)==ref
    assert comparison_checkpoint(f,"B",ref)==ref
    assert comparison_checkpoint(f,"baseline",None) is None
    with pytest.raises(ValueError):comparison_checkpoint(f,"B",FileRef(str(p),"0"*64))
    with pytest.raises(ValueError):comparison_checkpoint(f,"baseline",ref)

def test_eval_protocol_cannot_be_prepared_before_training_completed(tmp_path,monkeypatch):
    import importlib.util
    p=Path(__file__).resolve().parents[2]/"operations/gh_formal_prepare_eval.py"
    spec=importlib.util.spec_from_file_location("formal_eval_preparation_fixture",p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,"load_formal",lambda _:{"outputs":{"train":str(tmp_path/"absent")}})
    with pytest.raises(ValueError,match="training not complete"):module.prepare("unused",tmp_path/"new.json")
    assert not (tmp_path/"new.json").exists()

def test_formal_factory_passes_explicit_cuda_zero_without_real_CUDA(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from tbps.alternating import formal_backend
    from tbps.models.irra import IRRAAdapter
    from tbps.models.irra_source import IRRASource
    path=tmp_path/"dataset.json";path.write_text('{"irra":{}}')
    called={};source=object()
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:True) # 只模拟已获授权的入口状态，不初始化设备。
    monkeypatch.setattr(IRRASource,"from_dict",lambda _:source)
    def load(s,*,device,allow_model_load):
        called.update(source=s,device=device,allowed=allow_model_load);return object()
    monkeypatch.setattr(IRRAAdapter,"from_source",load)
    @contextmanager
    def session(*a,**k):yield SimpleNamespace()
    monkeypatch.setattr(formal_backend,"native_session",session)
    def backend(*args):
        return SimpleNamespace(close=lambda:called.update(closed=True))
    monkeypatch.setattr(formal_backend,"FormalBackend",backend)
    c={"sources":{"dataset_config":{"path":str(path)}}}
    with formal_backend.loaded_formal_backend(c,SimpleNamespace(modules={}),SimpleNamespace(validate=lambda *a:None)):
        assert called["device"]=="cuda:0" and called["source"] is source and called["allowed"]
    assert called["closed"]
