import importlib.util,sys
from pathlib import Path
import pytest
from tbps.artifacts import ArtifactError
r=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("shared_modal_cpu",r/"operations/run_cpgc_modal_shared.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.mark.parametrize("free,util",[(0,99),(4343,90),(21035,0)])
def test_shared_gpu_not_blocked_by_other_processes(free,util):
 assert m.admission({"blocked":{},"memory.free":free,"utilization.gpu":util,"compute_processes":["3041291 SEP"]})
def test_host_resource_still_blocks():
 with pytest.raises(RuntimeError,match="Host RESOURCE_BUSY"):m.admission({"blocked":{"RAM_bytes":1}})
def test_child_uses_frozen_source_and_preflight():
 for action in ("evaluate","audit"):
  c=m.child_command(action,"rasa")
  assert str(m.CONFIG)in c and str(m.PREFLIGHT)in c
  assert ("--allow-gpu-evaluation"in c)==(action=="evaluate")
def test_no_sep_lock_or_process_intervention():
 s=(r/"operations/run_cpgc_modal_shared.py").read_text()
 assert "cpgc_sep_v2_resume.lock" not in s and "os.kill("not in s and "SIGTERM"not in s
 assert 'PREVIOUS/"process.lock"'in s and '"started").open("x")' in s
def test_wrapper_changes_no_numeric_source():
 import subprocess
 assert not subprocess.check_output(["git","diff","7973128ec1fc73ab8e2428dfbdea600a565421a9","--","src/tbps","configs/framework"],cwd=r,text=True).strip()
