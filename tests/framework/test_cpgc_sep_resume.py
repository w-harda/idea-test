"""恢复调度器CPU测试；仅Fixture，不运行GPU或原CPU审计。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json
import pytest
R=Path(__file__).resolve().parents[2]/"operations/cpgc_sep_resume.py"
spec=importlib.util.spec_from_file_location("cpgc_sep_resume_tested",R);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def test_native_commands_remain_frozen():
 c={"source_python":"sourcepy","victim_python":"targetpy"}
 s=m.Step("cpgc","full","cuhk","rasa","target",Path("/unused"))
 a=m.native_command(c,s)
 assert a[0]=="targetpy" and "--allow-full" in a and "--resume" not in a
 assert a[a.index("--config")+1]==str(m.CONFIG)
def test_plan_keeps_serial_order_and_independent_audits():
 p=list(m.steps({"output_base":"/tbps-framework-fixture"}))
 i=next(i for i,s in enumerate(p) if s.label=="cpgc_full_cuhk_rasa_target")
 assert p[i-3].label=="cpgc_full_cuhk_source"
 assert p[i+1].label=="cpgc_full_cuhk_rasa_audit"
 assert sum(s.kind=="target" and s.scope=="full" for s in p)==12
 assert sum(s.kind=="audit" and s.scope=="full" for s in p)==12
@pytest.mark.parametrize("exists,kind,log,allowed",[
 (False,"target",'File "/x/tbps/baselines/cpgc_sep/runtime.py", line 28, in resources\ntbps.artifacts.ArtifactError: RESOURCE_BUSY',True),
 (True,"target",'File "/x/tbps/baselines/cpgc_sep/runtime.py", line 28, in resources\ntbps.artifacts.ArtifactError: RESOURCE_BUSY',False),
 (False,"audit",'File "/x/tbps/baselines/cpgc_sep/runtime.py", line 28, in resources\ntbps.artifacts.ArtifactError: RESOURCE_BUSY',False),
 (False,"target","torch.OutOfMemoryError: CUDA out of memory",False),
 (False,"target","tbps.artifacts.ArtifactError: SHA mismatch",False),
 (False,"target","tbps.artifacts.ArtifactError: RESOURCE_BUSY",False)])
def test_only_true_prelaunch_resource_busy_can_reenter(tmp_path,exists,kind,log,allowed):
 p=tmp_path/"attempt"
 if exists:p.mkdir()
 s=m.Step("cpgc","full","cuhk","rasa",kind,p)
 assert m.safe_prelaunch_busy(s,log) is allowed
def test_wait_budget_is_bounded_and_can_recover():
 budget=m.Budget(maximum=6,interval=5);clock=[0.]
 def sleep(n):clock[0]+=n
 budget.sleep(sleep,lambda:clock[0]);budget.sleep(sleep,lambda:clock[0])
 assert budget.used==6
 with pytest.raises(m.WaitExpired):budget.sleep(sleep,lambda:clock[0])
 sequence=iter([{"status":"RESOURCE_BUSY"},{"status":"READY"}]);events=[];b=m.Budget(12,5)
 result=m.wait_ready({},b,events.append,lambda _:next(sequence),sleep,lambda:clock[0])
 assert result["status"]=="READY" and b.used==5 and len(events)==2
def fake_resources(monkeypatch,free=24051,ram_threshold=0,disk=200*1024**3,inodes=1000000):
 def run(args,**kw):
  return SimpleNamespace(returncode=0,stdout=str(free)+", 24564, 100" if "--id=0" in args else "",stderr="")
 monkeypatch.setattr(m.subprocess,"run",run)
 monkeypatch.setattr(m.shutil,"disk_usage",lambda _:SimpleNamespace(free=disk))
 monkeypatch.setattr(m.os,"statvfs",lambda _:SimpleNamespace(f_favail=inodes))
 return {"output_parent":"/unused","resource_gate":{"minimum_free_mib":8192,"RAM_bytes":ram_threshold,"disk_bytes":100*1024**3,"inodes":80000}}
@pytest.mark.parametrize("free,expected",[(8192,"READY"),(8191,"RESOURCE_BUSY")])
def test_memory_threshold_preserved_high_util_only_recorded(monkeypatch,free,expected):
 c=fake_resources(monkeypatch,free=free);r=m.resources(c)
 assert r["status"]==expected and r["GPU_utilization"]==100
 if expected=="RESOURCE_BUSY":assert r["blocked"]["minimum_free_mib"]["deficit"]==1
@pytest.mark.parametrize("kind",["RAM_bytes","disk_bytes","inodes"])
def test_identifies_each_blocker(monkeypatch,kind):
 c=fake_resources(monkeypatch,ram_threshold=10**30 if kind=="RAM_bytes" else 0,disk=1 if kind=="disk_bytes" else 200*1024**3,inodes=1 if kind=="inodes" else 1000000)
 r=m.resources(c);assert r["status"]=="RESOURCE_BUSY" and kind in r["blocked"]
def test_gpu_query_failure_is_not_resource_wait(monkeypatch):
 c=fake_resources(monkeypatch)
 monkeypatch.setattr(m.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=1,stdout="",stderr="query failed"))
 with pytest.raises(m.Stop,match="unavailable"):m.resources(c)
@pytest.fixture
def committed(tmp_path):
 from tbps.artifacts import ArtifactStore,json_bytes
 base=tmp_path/"tbps-framework-fixture";root=base/"full/cpgc/cuhk/irra"
 s=m.Step("cpgc","full","cuhk","irra","target",root)
 c={"output_base":str(base),"sha256":"a"*64};store=ArtifactStore(root,new_base=base);root.mkdir(parents=True)
 (root/"started.json").write_text(json.dumps({"config_sha256":c["sha256"],"scope":"full"}))
 store.publish("protocol",{"protocol.json":json_bytes({"config_sha256":c["sha256"]})},{})
 groups={mode:{k:float(i+1) for i,k in enumerate(("R@1","R@5","R@10","mAP"))} for mode in ("Clean","Joint","Matched Clean")}
 report={"status":"PASS","scope":"full","dataset":"cuhk","victim":"irra","method":"cpgc","Gallery":3074,"Query":6156,"model_before":{},"model_after":{},"counts":{"backward":0,"optimizer":0},"groups":groups}
 csv="Mode,R@1,R@5,R@10,mAP\n"+"\n".join(mode+",1.0,2.0,3.0,4.0" for mode in groups)+"\n"
 disp="Mode,R@1,R@5,R@10,mAP\n"+"\n".join(mode+",1.00,2.00,3.00,4.00" for mode in groups)+"\n"
 store.publish("completed",{"summary.json":json_bytes(report),"metrics.csv":csv.encode(),"metrics_2dp.csv":disp.encode()},{})
 return c,s
def test_complete_result_reused_readonly(committed):
 c,s=committed;before={str(p):(p.stat().st_size,p.stat().st_mtime_ns,m.sha(p)) for p in s.root.rglob("*") if p.is_file()}
 assert m.verify_step(c,s)["deep_payload_SHA"]
 assert before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns,m.sha(p)) for p in s.root.rglob("*") if p.is_file()}
@pytest.mark.parametrize("mode",["FAIL","pending","orphan","SHA","config"])
def test_never_skips_or_repairs_corrupt_completed(committed,mode):
 c,s=committed
 if mode=="FAIL":(s.root/"FAIL.json").write_text("{}")
 elif mode=="pending":(s.root/".pending-fixture").mkdir()
 elif mode=="orphan":(s.root/"unfinished.Claim.json").write_text("{}")
 elif mode=="SHA":(s.root/"completed/metrics.csv").write_text("tampered")
 else:c["sha256"]="b"*64
 with pytest.raises((m.Stop,ValueError)):m.verify_step(c,s)
def test_partial_attempt_requires_new_identity(tmp_path):
 base=tmp_path/"tbps-framework-fixture";s=m.Step("cpgc","full","cuhk","rasa","target",base/"partial");s.root.mkdir(parents=True)
 with pytest.raises(m.Stop,match="partial Attempt"):m.verify_step({"output_base":str(base),"sha256":"a"*64},s)
def test_operations_never_changes_native_or_calls_gpu_directly():
 t=R.read_text()
 assert "import torch" not in t.replace('"""不import torch；失败项逐个记录，利用率只观测。"""',"")
 assert "Optimizer" not in t and "kill(" not in t and "terminate(" not in t
