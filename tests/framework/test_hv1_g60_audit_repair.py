"""执行修复后完整CPU审计控制流；真实Checkpoint事务和stat保护，模型仅合成Fixture。"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch
import importlib.util,json,copy
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("independent_h60_repair",ROOT/"src/tbps/ablations/hv1_g60_audit_repair.py")
fix=importlib.util.module_from_spec(spec);spec.loader.exec_module(fix)
from tbps.artifacts import ArtifactStore,json_bytes,sha_file,read_json
from tbps.ablations.hv1_g60_state import H10,stat,ref
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.attacks.provenance import LegacyAttackSources

@pytest.fixture(autouse=True)
def no_gpu():
    with patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("GPU forbidden")), \
         patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("neural forward forbidden")):
        yield
    assert not torch.cuda.is_initialized()

@pytest.fixture
def case(tmp_path,monkeypatch):
    m=fix.load_fixed(ROOT/"src/tbps/ablations/hv1_g60_audit.py")
    base=tmp_path/"tbps-framework-audit-fixture";base.mkdir()
    root=base/"smoke_v1";root.mkdir()
    protocol={"config":{"architecture":{},"optimizer":{"lr":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":0.}},
              "query_order":["unused"]*20480+["row0","row1","row2","row3"],
              "eligible_by_row":dict.fromkeys(["row0","row1","row2","row3"],True)}
    legacy=LegacyAttackSources(fix.OLD/"configs/framework/stage3br_plugins.server.json")
    with legacy_imports(legacy,allow_import=True) as old:
        ck=old("formal_training.checkpoint")
        model=torch.nn.Linear(2,2);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        # 合成已提交Adam状态；不执行模型forward、训练或Optimizer.step。
        for p in model.parameters():
            opt.state[p]={"step":torch.tensor(20374.),"exp_avg":torch.zeros_like(p),"exp_avg_sq":torch.zeros_like(p)}
        sampler=torch.Generator().manual_seed(42)
        work=ck.begin(root,protocol,H10)
        reward=dict(R_sim=0.,R_rank=0.,total=0.,synergy_increment=0.,adv_sim_drop=0.,clean_sim_drop=0.)
        action=dict(original="blue shirt",adversarial="blue shirt",attributes=[],edits=[],budget_k=0)
        for i in range(4):
            ck.append(work,dict(visit=20481+i,row_id="row"+str(i),epoch=11,epoch_position=i+1,
              optimizer_steps_this_step=1,committed_update_candidate=20371+i,IRRA_text_samples=3,
              sampled_reward=reward,greedy_reward=reward,sampled=action,greedy=action,
              SCST_loss=0.,gradient_norm=0.,parameter_delta_l2=0.))
        path,h=ck.publish(root,work,model,opt,sampler,protocol,20484,20374,H10,20480,20370,"smoke_probe")
        result=dict(status="HV1_G60_SMOKE_PASS",parent_H10={"sha256":H10},model_updates_only_H=True,
          H_changed=True,H_Adam_RNG_reload_exact=True,runtime=dict(IRRA_state_before="same",IRRA_state_after="same",
          parameters_buffers_unchanged=True,samples=dict(G=0,clean=0,adv=0)),
          cursor=20484,updates=20374,last_checkpoint_sha256=h,new_visits=4,new_updates=4,final_epoch=10)
        (root/"completed.json").write_bytes(json_bytes(result));(root/"protocol.json").write_bytes(json_bytes(protocol))
        protected=[]
        for i in range(3):
            p=tmp_path/("history"+str(i));p.write_text("immutable");protected.append(p)
        snapshot=tmp_path/"snapshot.json";snapshot.write_bytes(json_bytes({str(p):stat(p) for p in protected}))
        source=tmp_path/"producer.py";source.write_text("# frozen fixture")
        ctx=dict(c={"parent":{"sha256":H10},"G60":{"sha256":"g60"},"smoke_visits":4,
                    "sources":{"protection_snapshot":{"path":str(snapshot)}}},
                 report={"sources":{str(source):__import__("tbps.artifacts",fromlist=["checked"]).checked(ref(source))}},
                 legacy=legacy)
        def proxy(name):
            if name=="tokenizer_probe":return SimpleNamespace(official_probe=lambda:None)
            if name=="policy":return SimpleNamespace(CharacterPolicy=lambda probe,**kw:torch.nn.Linear(2,2))
            return old(name)
        @contextmanager
        def imports(*a,**k):yield proxy
        monkeypatch.setattr(m,"legacy_imports",imports);monkeypatch.setattr(m,"BASE",base)
        calls=[]
        def actual_stat(p):
            calls.append(str(p));return stat(p)
        monkeypatch.setattr(m,"stat",actual_stat)
        yield SimpleNamespace(m=m,ctx=ctx,base=base,root=root,path=path,protected=protected,calls=calls,source=source)

def test_full_smoke_control_flow_and_real_protection_loop(case):
    case.m.audit_training(case.ctx,smoke=True)
    report=read_json(case.base/"smoke_audit/result/report.json")
    assert report["status"]=="HV1_G60_SMOKE_CPU_AUDIT_PASS"
    assert report["epochs"]=={"11":{"visits":4,"updates":4,"skipped":0}}
    assert case.calls==list(map(str,case.protected))
    assert not (case.base/"training").exists()
    ArtifactStore(case.base/"smoke_audit",new_base=case.base).verify("result")

@pytest.mark.parametrize("index",[0,1,2])
def test_real_historical_metadata_tamper_rejected(case,index):
    case.protected[index].write_text("fixture altered")
    with pytest.raises(ValueError,match="protected historical"):case.m.audit_training(case.ctx,smoke=True)
    assert case.calls==list(map(str,case.protected[:index+1]))
    assert not (case.base/"smoke_audit/result").exists()

@pytest.mark.parametrize("file",["checkpoint.pt","steps.jsonl","intent.json"])
def test_checkpoint_and_log_sha_tamper_rejected(case,file):
    p=case.path/file;p.write_bytes(p.read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA mismatch"):case.m.audit_training(case.ctx,smoke=True)
    assert not case.calls

def test_protocol_mismatch_rejected(case):
    p=case.root/"protocol.json";v=read_json(p);v["unexpected"]=True;p.write_bytes(json_bytes(v))
    with pytest.raises(ValueError,match="SHA mismatch"):case.m.audit_training(case.ctx,smoke=True)

def test_producer_changed_rejected(case):
    case.source.write_text("changed")
    with pytest.raises(Exception):case.m.audit_training(case.ctx,smoke=True)
    assert not (case.base/"smoke_audit/result").exists()

def test_pending_refused(case):
    (case.root/"pending").mkdir()
    with pytest.raises(ValueError,match="uncommitted"):case.m.audit_training(case.ctx,smoke=True)

def test_formal_cannot_consume_smoke(case):
    with pytest.raises((ValueError,FileNotFoundError)):case.m.audit_training(case.ctx,smoke=False)
    assert not (case.base/"training_audit/result").exists()

def test_bad_frozen_state_rejected(case):
    p=case.root/"completed.json";v=read_json(p);v["runtime"]["IRRA_state_after"]="changed"
    p.write_bytes(json_bytes(v))
    with pytest.raises(ValueError,match="frozen IRRA"):case.m.audit_training(case.ctx,smoke=True)

def test_only_local_stat_rename():
    original=(fix.OLD/"src/tbps/ablations/hv1_g60_audit.py").read_text()
    fixed=(ROOT/"src/tbps/ablations/hv1_g60_audit.py").read_text()
    assert fixed==fix.rename_only(original)
    assert "stat(p)==expected" in fixed and "epoch_stat=per_epoch" in fixed

def test_binding_and_original_namespace():
    b,c=fix.verify_binding(ROOT/"configs/framework/hv1_g60_audit_repair.server.json")
    assert len(c["producers"])==113
    assert b["execution_root"]!=b["repair_root"]
    assert b["execution_commit"]=="9765a536a11a9a0e9111f2f834230c53d41bf2d8"

def test_binding_tamper_rejected(tmp_path):
    p=ROOT/"configs/framework/hv1_g60_audit_repair.server.json";b=read_json(p)
    b["old_auditor"]["sha256"]="0"*64;q=tmp_path/"bad.json";q.write_bytes(json_bytes(b))
    with pytest.raises(Exception):fix.verify_binding(q)

def test_controller_uses_original_training_and_fixed_audit():
    s=(ROOT/"src/tbps/ablations/run_hv1_g60_adaptation_repaired.sh").read_text()
    assert '--action train --allow-gpu --allow-training' in s
    assert "--action smoke" not in s and "--action gallery" not in s
    assert "audit-training" in s and "launch-gate" in s and "smoke_gate" in s
    assert "PYTHONPATH=\"$OLD/src\"" in s and "PARTIAL_USER_STOP" in s
    assert "retry" not in s

def test_missing_repair_pin_rejected(tmp_path):
    b=read_json(ROOT/"configs/framework/hv1_g60_audit_repair.server.json")
    b["repair_sources"].pop()
    q=tmp_path/"missing.json";q.write_bytes(json_bytes(b))
    with pytest.raises(ValueError,match="complete independent"):fix.verify_binding(q)
