"""Full监督器仅编排/日志；CPU Fixture模拟完整链、失败及重入。"""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import os, subprocess
import pytest
import torch
from tbps.artifacts import ArtifactError
from tbps.baselines import sga_full_execution as e

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"src/tbps/baselines/run_sga_irra_full.sh"

def test_launch_preflight_no_model_or_cuda():
    with patch.object(torch,"load",side_effect=AssertionError("no model load")),patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("no CUDA")),patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("no neural forward")):
        e.launch_preflight()
    assert not torch.cuda.is_initialized()

def test_progress_exact_delegate_and_restore():
    class Backend:
        def encode_image(self,x):return x+1
        def encode_text(self,x):return x+2
    b=Backend();image=b.encode_image;text=b.encode_text
    @contextmanager
    def load(config,model,scope):
        assert (config,model,scope)==("fixture","irra","full")
        yield object(),b,object(),{}
    x=torch.tensor([1.,2.])
    with e.encoding_progress(load,"irra","fixture",scope="full"):
        assert torch.equal(b.encode_image(x),x+1) and torch.equal(b.encode_text(x),x+2)
    assert b.encode_image==image and b.encode_text==text

def test_target_full_requires_permission():
    with pytest.raises(ArtifactError,match="explicit Full"):
        e.main(["--action","evaluate","--model","irra"])

@pytest.mark.parametrize("failed",["","rasa"])
def test_actual_bash_chain_fixture(tmp_path,failed):
    root=tmp_path/"dev";frozen=tmp_path/"frozen";out=tmp_path/"output/full";control=tmp_path/"control"
    (root/"src/tbps/baselines").mkdir(parents=True);(frozen/"configs/framework").mkdir(parents=True);out.parent.mkdir()
    (root/"src/tbps/baselines/sga_full_execution.py").write_text("fixture")
    (frozen/"configs/framework/sga_irra_source.full.server.json").write_text("{}")
    fake=tmp_path/"python"
    fake.write_text("#!/usr/bin/env python3\nimport os,sys\na=' '.join(sys.argv[1:])\n"
      "if '--action source' in a: print('SGA_OFFICIAL_IRRA_SOURCE_PASS full')\n"
      "elif '--action evaluate' in a:\n"
      " m=sys.argv[sys.argv.index('--model')+1]\n"
      " if m==os.environ.get('FAIL_MODEL'): print('fixture failure');sys.exit(9)\n"
      " print('SGA_OFFICIAL_TARGET_PASS full '+m)\n"
      "elif '--action audit' in a: print('SGA_OFFICIAL_IRRA_CPU_AUDIT_PASS full')\n"
      "else: print('SGA_OFFICIAL_FULL_LAUNCH_PREFLIGHT_PASS')\n");fake.chmod(0o755)
    bin=tmp_path/"bin";bin.mkdir()
    for name,body in {
      "nvidia-smi":"echo 'fixture GPU snapshot'",
      "df":"if [[ \"$*\" == *--output=avail* ]]; then printf 'Avail\\n999999999999\\n'; else echo 'fixture disk'; fi",
      "awk":"echo 999999999"}.items():
        f=bin/name;f.write_text("#!/usr/bin/env bash\n"+body+"\n");f.chmod(0o755)
    text=SCRIPT.read_text().replace("ROOT=/home/lzf/ldx/projects/idea-test-tbps-sga-full-launch-fix-20261019","ROOT="+str(root)).replace("FROZEN=/home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019","FROZEN="+str(frozen))
    text=text.replace("SOURCE_PY=/home/lzf/ldx/envs/sga-irra-source/bin/python","SOURCE_PY="+str(fake)).replace("TARGET_PY=/home/lzf/ldx/envs/tbps-image/bin/python","TARGET_PY="+str(fake))
    text=text.replace("OUT=/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full","OUT="+str(out)).replace("CONTROL=/home/lzf/ldx/tmp/sga_official_irra_full_v2","CONTROL="+str(control))
    script=tmp_path/"run.sh";script.write_text(text)
    env={**os.environ,"PATH":str(bin)+":"+os.environ["PATH"],"FAIL_MODEL":failed}
    first=subprocess.run(["bash",str(script),"--allow-full"],env=env,capture_output=True,text=True)
    assert first.returncode==(9 if failed else 0),first.stdout+first.stderr
    assert (control/"phase").read_text().strip()==("FAIL" if failed else "SUCCESS")
    assert (control/"supervisor.pid").read_text().strip().isdigit() and (control/"supervisor.pgid").read_text().strip().isdigit()
    if failed:assert not (out/"aptm.log").exists() and not (out/"audit.log").exists()
    else:assert "SGA_OFFICIAL_IRRA_CPU_AUDIT_PASS full" in (out/"audit.log").read_text()
    original=(control/"status.log").read_bytes()
    second=subprocess.run(["bash",str(script),"--allow-full"],env=env,capture_output=True,text=True)
    assert second.returncode==4 and (control/"status.log").read_bytes()==original

def test_bash_default_no_run():
    p=subprocess.run(["bash",str(SCRIPT)],capture_output=True,text=True)
    assert p.returncode==2


def test_real_df_before_full_output_exists(tmp_path):
    out=tmp_path/"full"
    control=tmp_path/"control";control.mkdir()
    assert not out.exists()
    commands=[line for line in SCRIPT.read_text().splitlines()
              if line.startswith("df ") or line.startswith("available=$(df ")]
    assert len(commands)==3
    assert all('"$(dirname "$OUT")"' in line for line in commands)
    code='set -e\nOUT="$TEST_OUT"\nCONTROL="$TEST_CONTROL"\n'+'\n'.join(commands)+'\n[[ "$available" -gt 0 ]]'
    p=subprocess.run(["bash","-c",code],env={**os.environ,"TEST_OUT":str(out),"TEST_CONTROL":str(control)},capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert (control/"disk_start.log").stat().st_size>0 and (control/"inodes_start.log").stat().st_size>0
    assert not out.exists()

def test_v2_control_root_isolation():
    text=SCRIPT.read_text()
    assert "CONTROL=/home/lzf/ldx/tmp/sga_official_irra_full_v2" in text
    assert "sga_official_irra_full_v1" not in text
    assert "ROOT="+str(ROOT) in text
