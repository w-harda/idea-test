"""G60适配Extension的CPU恢复/调度/事务与原SCST回归，无GPU或真实模型前向。"""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import copy,json
import pytest
import torch
from tbps.artifacts import ArtifactStore,ArtifactError,digest,read_json
from tbps.ablations.hv1_g60_state import extension_contract,H10,tree_hash,config,BASE
from tbps.ablations.hv1_g60_adaptation import CachedBank
from tbps.attacks.provenance import LegacyAttackSources
from tbps.attacks.legacy_runtime import legacy_imports

torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(scope="module")
def old():
    source=LegacyAttackSources(ROOT/"configs/framework/stage3br_plugins.server.json")
    with patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("GPU forbidden")),legacy_imports(source,allow_import=True) as load:
        yield load
@pytest.fixture
def parent():
    original={"config":{"epochs":10,"seed":42,"optimizer":{"lr":1e-4,"betas":[.9,.999]}},
      "epoch_size":2048,"query_order":[f"row{i%2048}" for i in range(20480)],
      "eligible_by_row":{f"row{i}":i>=11 for i in range(2048)},"train_identities":[],
      "catalog_sha256":"a"*64,"qualification_sha256":"b"*64}
    return {"schema":"hv1-formal-periodic-checkpoint-1","cursor":20480,"updates":20370,
            "reason":"epoch_complete","protocol":original,"protocol_sha256":digest(original)}
def extended(p,bank=None):
    return extension_contract(p,bank or {"G60":"fixed"},[f"row{i%2048}" for i in range(122880)],{"source":"sha"})

def test_valid_extension_preserves_original(parent):
    original=copy.deepcopy(parent);e=extended(parent)
    assert parent==original and e["original_training_protocol"]==original["protocol"]
    assert e["config"]["epochs"]==60 and e["original_training_protocol"]["config"]["epochs"]==10
    assert e["parent_checkpoint_sha256"]==H10
    assert e["start_epoch"]==11 and e["target_epochs"]==60 and not e["validation_feedback"]

@pytest.mark.parametrize("key,value",[("cursor",20479),("updates",0),("reason","periodic"),("schema","fake")])
def test_bad_parent_rejected(parent,key,value):
    parent[key]=value
    with pytest.raises(ValueError):extended(parent)

def test_changed_original_contract_digest_rejected(parent):
    parent["protocol"]["config"]["epochs"]=60
    with pytest.raises(ValueError):extended(parent)

def test_changed_order_rejected(parent):
    order=[f"row{i%2048}" for i in range(122880)];order[2047]="wrong"
    with pytest.raises(ValueError):extension_contract(parent,{},order,{})

def test_bank_different_protocol_identity(parent):
    assert digest(extended(parent,{"G60":"one"}))!=digest(extended(parent,{"G60":"two"}))

def test_absolute_epoch11_original_order(old):
    members=[SimpleNamespace(row_id=f"row{i}") for i in range(2048)]
    order=old("formal_training.training").order(members,60,42)
    direct=torch.randperm(2048,generator=torch.Generator().manual_seed(52)).tolist()
    assert order[20480:22528]==[members[i].row_id for i in direct]
    assert order[:20480]==old("formal_training.training").order(members,10,42)

def test_qualified_counts_unchanged(parent):
    e=extended(parent)
    updates=sum(e["eligible_by_row"][q] for q in e["query_order"][20480:])
    assert updates==101850 and updates+20370==122220
    assert len(e["query_order"][20480:])==102400

def test_H_Adam_rng_exact_restore(old):
    torch.manual_seed(17)
    model=torch.nn.Linear(2,2);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    sampler=torch.Generator().manual_seed(42)
    ck=old("formal_training.checkpoint")
    payload={"policy":copy.deepcopy(model.state_dict()),"optimizer":copy.deepcopy(optimizer.state_dict()),"rng":ck.capture_rng(sampler)}
    other=torch.nn.Linear(2,2);opt2=torch.optim.Adam(other.parameters(),lr=9.)
    gen2=torch.Generator();ck.restore(payload,other,opt2,gen2,allow_cuda=False)
    assert ck.same(model.state_dict(),other.state_dict())
    assert ck.same(optimizer.state_dict(),opt2.state_dict())
    assert ck.same(payload["rng"],ck.capture_rng(gen2))
    assert opt2.param_groups[0]["lr"]==1e-4

def test_checkpoint_chain_atomic_restore_no_duplicate_step(old,tmp_path):
    ck=old("formal_training.checkpoint")
    protocol={"query_order":["one","two"],"eligible_by_row":{"one":True,"two":True}}
    model=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(model.parameters())
    sampler=torch.Generator().manual_seed(42)
    work=ck.begin(tmp_path,protocol,"parent")
    model(torch.ones(1,1)).sum().backward();optimizer.step()
    log={"visit":1,"row_id":"one","optimizer_steps_this_step":1,"committed_update_candidate":1}
    ck.append(work,log)
    path,parent=ck.publish(tmp_path,work,model,optimizer,sampler,protocol,1,1,"parent",0,0,"safe_stop")
    assert not (tmp_path/"pending").exists()
    payload,md=ck.verify(path,protocol,"parent",0,0)
    other=torch.nn.Linear(1,1);opt2=torch.optim.Adam(other.parameters());gen2=torch.Generator()
    ck.restore(payload,other,opt2,gen2,allow_cuda=False)
    assert md["cursor"]==1 and md["updates"]==1
    assert ck.same(other.state_dict(),model.state_dict())
    # 续训从visit2而非重复visit1；旧提交版本不可覆盖。
    work2=ck.begin(tmp_path,protocol,parent)
    ck.append(work2,{"visit":2,"row_id":"two","optimizer_steps_this_step":1,"committed_update_candidate":2})
    with pytest.raises(ValueError):ck.publish(tmp_path,work2,other,opt2,gen2,protocol,1,2,parent,1,1,"bad")

def test_pending_refuses_resume_before_load(tmp_path):
    from tbps.ablations.hv1_g60_state import audit_versions
    (tmp_path/"pending").mkdir()
    with pytest.raises(ValueError,match="uncommitted"):audit_versions(tmp_path,{},lambda _:None)

def test_wrong_parent_chain_rejected(old,tmp_path):
    ck=old("formal_training.checkpoint");protocol={"query_order":[],"eligible_by_row":{}}
    model=torch.nn.Linear(1,1);opt=torch.optim.Adam(model.parameters());sampler=torch.Generator()
    work=ck.begin(tmp_path,protocol,"correct")
    path,_=ck.publish(tmp_path,work,model,opt,sampler,protocol,0,0,"correct",0,0,"fixture")
    with pytest.raises(ValueError):ck.verify(path,protocol,"wrong",0,0)

def test_source_configuration_rejects_missing_producer(tmp_path):
    c=json.loads((ROOT/"configs/framework/hv1_g60_adaptation.server.json").read_text())
    c["producers"].pop(next(iter(c["producers"])))
    p=tmp_path/"config.json";p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="complete frozen"):config(p)

def test_fixed_H10_source_rejection(tmp_path):
    c=json.loads((ROOT/"configs/framework/hv1_g60_adaptation.server.json").read_text())
    c["parent"]["sha256"]="0"*64
    p=tmp_path/"config.json";p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match="fixed H10"):config(p)

def test_original_SCST_only_H_updates(old):
    torch.manual_seed(42)
    probe=old("tokenizer_probe").official_probe();frontend=old("text_frontend").CaptionFrontend()
    example=frontend("A man wearing a blue shirt.")
    policy=old("policy").CharacterPolicy(probe,width=8,layers=1,heads=2,max_bytes=128)
    cfg={"optimizer":{"lr":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":0.}}
    optimizer=old("phase4_training").optimizer_for(policy,cfg)
    frozen=torch.nn.Linear(2,2).eval().requires_grad_(False);before=tree_hash(frozen.state_dict())
    class Reward:
        runtime=SimpleNamespace(assert_frozen=lambda:None)
        def evaluate_many(self,original,variants):
            self.features=torch.nn.functional.normalize(torch.ones(3,512),dim=1)
            R=old("rewards").Reward
            return R(1.,.6,.4,.1,.6,.5,.2),R(0.,0.,0.,0.,0.,0.,0.)
        def diagnostics(self):return {"IRRA_text_samples":3}
    log,_=old("phase4_training").train_step(policy,optimizer,example,torch.Generator().manual_seed(42),
                    lambda k:Reward(),1,"fixture")
    assert log["optimizer_steps_this_step"]==1 and log["gradient_norm"]>0
    assert log["parameter_delta_l2"]>0 and tree_hash(frozen.state_dict())==before
    assert all(p.grad is None for p in frozen.parameters())

def test_reward_bank_validation_rejected(old):
    with pytest.raises(ValueError):old("formal_training.runtime").RewardBank(None,SimpleNamespace(role="validation"),None)

def test_new_output_not_historical(parent):
    assert str(BASE).startswith("/home/lzf/ldx/outputs/tbps-framework-")
    assert "/outputs/formal_v1" not in str(BASE)
    e=extended(parent);e["config"]["epochs"]=77
    assert parent["protocol"]["config"]["epochs"]==10

def test_backend_has_no_G_and_forbids_images():
    import inspect
    from tbps.ablations.hv1_g60_adaptation import TextRuntime
    s=inspect.getsource(TextRuntime)
    assert "generator" not in s and "training/Validation cannot encode images" in s
    assert "IRRAAdapter.from_source" in s and "encode_tokens" in s

def test_no_cuda():
    assert not torch.cuda.is_initialized()


def test_interrupted_resume_matches_uninterrupted_Adam_and_rng(old,tmp_path):
    ck=old("formal_training.checkpoint");torch.manual_seed(9)
    model=torch.nn.Linear(2,1);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    sampler=torch.Generator().manual_seed(42);initial_rng=ck.capture_rng(sampler)
    initial_weights=copy.deepcopy(model.state_dict())
    def update(m,o):
        o.zero_grad(set_to_none=True);loss=m(torch.randn(1,2)).square().sum();loss.backward();o.step()
    for _ in range(4):update(model,opt)
    expected_H=copy.deepcopy(model.state_dict());expected_Adam=copy.deepcopy(opt.state_dict())
    expected_rng=ck.capture_rng(sampler)
    second=torch.nn.Linear(2,1);second.load_state_dict(initial_weights);opt2=torch.optim.Adam(second.parameters(),lr=1e-4)
    gen2=torch.Generator();ck.restore_rng(initial_rng,gen2,allow_cuda=False)
    for _ in range(2):update(second,opt2)
    protocol={"query_order":["a","b","c","d"],"eligible_by_row":dict.fromkeys("abcd",True)}
    work=ck.begin(tmp_path,protocol,"parent")
    for i,key in enumerate("ab",1):ck.append(work,{"visit":i,"row_id":key,"optimizer_steps_this_step":1,"committed_update_candidate":i})
    path,_=ck.publish(tmp_path,work,second,opt2,gen2,protocol,2,2,"parent",0,0,"safe_stop")
    payload,_=ck.verify(path,protocol,"parent",0,0)
    resumed=torch.nn.Linear(2,1);opt3=torch.optim.Adam(resumed.parameters(),lr=1e-4);gen3=torch.Generator()
    ck.restore(payload,resumed,opt3,gen3,allow_cuda=False)
    for _ in range(payload["cursor"],4):update(resumed,opt3)
    assert ck.same(expected_H,resumed.state_dict()) and ck.same(expected_Adam,opt3.state_dict())
    assert ck.same(expected_rng,ck.capture_rng(gen3))

def test_launcher_stops_at_safe_boundary_and_no_retry():
    script=(ROOT/"src/tbps/ablations/run_hv1_g60_adaptation.sh").read_text()
    assert "nohup" not in script and "trap stop_training TERM INT" in script
    assert "wait" in script and "PARTIAL_USER_STOP" in script
    assert "retry" not in script and "RaSa" not in script and "APTM" not in script

def test_partial_receipts_cannot_overwrite_final():
    import inspect
    from tbps.ablations.hv1_g60_adaptation import execute
    text=inspect.getsource(execute)
    assert "partial_attempts/visit_" in text and "completed.json" in text
    assert 'cursor-20480' in text and 'updates-20370' in text
