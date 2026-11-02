"""RSTP/ICFG扩展的CPU协议回归；真实来源由各Case的CPU Preflight另行验证。"""
from pathlib import Path
from dataclasses import asdict
import ast,copy,importlib.util,json,subprocess
import pytest,torch
from tbps.baselines.cpgc_sep import modal_ablation as a
from tbps.artifacts import ArtifactError,FileRef,checked,sha_file

ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
 monkeypatch.setattr(torch.cuda,"_lazy_init",lambda:(_ for _ in ()).throw(AssertionError("CPU test CUDA")))
def ops(name):
 spec=importlib.util.spec_from_file_location("cpu_"+name,ROOT/"operations"/(name+".py"))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.mark.parametrize("ds,size",[("rstp",(1000,2000)),("icfg",(19848,19848))])
def test_exact_authorized_full_dimensions(ds,size):
 assert a.dimensions({"dataset":ds,"expected_size":list(size)})==size
@pytest.mark.parametrize("ds,size",[("rstp",[10,10]),("icfg",[1000,2000]),("cuhk",[3074,6156]),("unknown",[1,2])])
def test_wrong_or_unrequested_scope_rejected(ds,size):
 with pytest.raises(ArtifactError):a.dimensions({"dataset":ds,"expected_size":size})
@pytest.mark.parametrize("name",["banks_for","matched_inputs","CandidateRecordingScorer","csv_bytes","check_store","unchanged"])
def test_cuhk_validated_numeric_bridge_exact_ast(name):
 old=ast.parse(subprocess.check_output(["git","show","7973128ec1fc73ab8e2428dfbdea600a565421a9:src/tbps/baselines/cpgc_sep/modal_ablation.py"],cwd=ROOT,text=True))
 current=ast.parse((ROOT/"src/tbps/baselines/cpgc_sep/modal_ablation.py").read_text())
 get=lambda t:next(x for x in t.body if getattr(x,"name",None)==name)
 assert ast.dump(get(old),include_attributes=False)==ast.dump(get(current),include_attributes=False)
@pytest.mark.parametrize("path",["src/tbps/baselines/cpgc_sep/cache.py","src/tbps/baselines/cpgc_sep/victims.py","src/tbps/engines/evaluation.py","src/tbps/engines/metrics.py","src/tbps/baselines/cpgc_sep/verified_store.py"])
def test_native_numerics_unchanged(path):
 assert (ROOT/path).read_bytes()==subprocess.check_output(["git","show","7973128:"+path],cwd=ROOT)
@pytest.mark.parametrize("dataset,victim,version",[("rstp","irra","v2"),("rstp","rasa","v2"),("icfg","irra","v2"),("icfg","rasa","v3")])
def test_case_historical_identity_and_bounded_cache(dataset,victim,version):
 p=ROOT/f"configs/framework/cpgc_modal_{dataset}_{victim}.server.json";c=json.loads(p.read_text())
 assert c["historical_version"]==version and c["dataset"]==dataset and list(c["targets"])==[victim]
 assert f"official-{version}/full/cpgc/{dataset}/{victim}"in c["targets"][victim]["path"]
 assert c["token_cache_bytes"]==8*1024**3
 checked(FileRef(**c["parent_config"]));checked(FileRef(**c["targets"][victim]["manifest"]))
 assert a.result_root(c,victim)==Path(c["output_base"])/dataset/victim
 assert a.result_root(c,victim,audit=True)==Path(c["output_base"])/"audit"/dataset/victim
def test_result_root_wrong_victim_rejected():
 with pytest.raises(ArtifactError):a.result_root({"targets":{"irra":{}}},"rasa")
def test_sha_mutation_rejected_fixture(tmp_path):
 p=tmp_path/"artifact";p.write_bytes(b"old");r=FileRef(str(p),sha_file(p));p.write_bytes(b"changed")
 with pytest.raises(ArtifactError):checked(r)
def test_full_audit_uses_authoritative_dynamic_shape():
 import inspect
 s=inspect.getsource(a.audit)
 assert '(dimensions(c)[1],dimensions(c)[0])'in s
 assert 'range(0,dimensions(c)[1],old["evaluation"]["query_chunk"])'in s
 assert '"all original global Top128 members fused"'in s and '"matched text feature content binding"'in s
def test_configured_dataset_used_by_actual_loader():
 import inspect
 s=inspect.getsource(a.run)
 assert 'load_victim(old,victim,c["dataset"],"cuda:0",smoke=False)'in s
 for bad in ('encode_images(', 'encode_texts(', 'native_pixels(', '.backward(', '.step('):assert bad not in s
def test_serial_order_and_gpu_authorization_only_for_evaluate():
 m=ops("run_cpgc_modal_three_datasets")
 assert m.ORDER==(("rstp","irra"),("rstp","rasa"),("icfg","irra"),("icfg","rasa"))
 for ds,v in m.ORDER:
  case={"dataset":ds,"victim":v,"config":{"path":f"/fixture/{ds}_{v}.json"}}
  for action in ("preflight","evaluate","audit"):
   cmd=m.native_command(case,action,"/fixture/pre.json")
   assert cmd[cmd.index("--victim")+1]==v and cmd[cmd.index("--config")+1]==case["config"]["path"]
   assert ("--allow-gpu-evaluation"in cmd)==(action=="evaluate")
def test_scheduler_never_stops_sep_or_waits_on_fixed_gpu_floor():
 s=(ROOT/"operations/run_cpgc_modal_three_datasets.py").read_text()
 for bad in ("os.kill(", "killall", "minimum_free_mib", "cpgc_sep_v2_resume.lock", "time.sleep("):assert bad not in s
 assert 'flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)'in s and '"started").open("x")'in s
 assert 'for action,phase in (("preflight","CPU_PREFLIGHT"),("evaluate","RUNNING_FULL"),("audit","RUNNING_CPU_AUDIT"))'in s
def test_report_requires_24_rows_and_no_smoke_or_cuhk_run():
 s=(ROOT/"operations/cpgc_modal_three_dataset_report.py").read_text()
 assert 'len(rows)==24'in s and 'a["CUDA"]is False'in s
 assert 'MODE' in s and 'CUHK complete immutable verification' in s
 m=ops("cpgc_modal_three_dataset_report");assert m.ORDER==(("rstp","irra"),("rstp","rasa"),("icfg","irra"),("icfg","rasa"))
def test_plan_rejects_wrong_case_order_before_dependencies(tmp_path):
 m=ops("cpgc_modal_three_dataset_report");p=tmp_path/"plan.json"
 p.write_text(json.dumps({"schema":"tbps-cpgc-three-dataset-modality-1","cases":[{"dataset":"icfg","victim":"rasa"}]}))
 with pytest.raises(ArtifactError,match="order"):m.load_plan(p)
