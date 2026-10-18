"""共享GPU仅CPU测试：资源Mock及真实Bash控制流，不调用nvidia-smi/模型/CUDA。"""
from pathlib import Path
import json,os,signal,subprocess,sys,time
from unittest.mock import patch
import pytest
import torch
from tbps.baselines import comparison as c
from tbps.artifacts import digest

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"configs/framework/baseline_b3_shared.server.json"

def config_ref(tmp_path):
    p=tmp_path/"config.json";p.write_text("{}")
    return str(p),{"minimum_free_mib":3072}

def gpu(free=3865,util=99):
    return {"name":"RTX4090","free_mib":free,"utilization_percent":util,
        "compute_processes":"726931, /usr/local/lib/ollama/llama-server, 20178","policy":"shared_observe_only"}

def test_high_utilization_and_ollama_allowed_single_snapshot(tmp_path):
    path,conf=config_ref(tmp_path)
    with patch.object(c,"gpu_available",return_value=gpu()) as observe,patch.object(c,"other_b3_processes",return_value=[]):
        value=c.resource_snapshot(path,conf)
    assert value["ready"] and value["GPU"]["utilization_percent"]==99
    observe.assert_called_once_with({"physical_gpu":0})

@pytest.mark.parametrize("free,ready",[(3071,False),(3072,True),(3865,True)])
def test_minimum_free_memory_boundary(tmp_path,free,ready):
    path,conf=config_ref(tmp_path)
    with patch.object(c,"gpu_available",return_value=gpu(free)),patch.object(c,"other_b3_processes",return_value=[]):
        snapshot=c.resource_snapshot(path,conf)
    assert snapshot["ready"] is ready
    if not ready:assert "3072" in snapshot["reason"]

def test_other_b3_rejected_not_external_inference(tmp_path):
    path,conf=config_ref(tmp_path)
    with patch.object(c,"gpu_available",return_value=gpu()),patch.object(c,"other_b3_processes",return_value=[{"pid":123,"command":"B3"}]):
        snapshot=c.resource_snapshot(path,conf)
    assert not snapshot["ready"] and snapshot["reason"]=="existing B3 execution"

def test_process_scan_excludes_controller_and_ollama():
    rows=f"{os.getpid()} python -m tbps.baselines.launch --action resources\n{os.getppid()} bash run_b3_full.sh\n123 ollama llama-server\n124 python -m tbps.baselines.launch --action run --scope full\n125 Xorg\n"
    with patch("subprocess.check_output",return_value=rows):
        result=c.other_b3_processes()
    assert [v["pid"] for v in result]==[124]

def test_pipeline_snapshot_reused_without_smi(tmp_path,monkeypatch):
    path,conf=config_ref(tmp_path)
    with patch.object(c,"gpu_available",return_value=gpu()),patch.object(c,"other_b3_processes",return_value=[]):
        snapshot=c.resource_snapshot(path,conf)
    saved=tmp_path/"startup.json";saved.write_text(json.dumps(snapshot))
    monkeypatch.setenv("TBPS_B3_GPU_SNAPSHOT",str(saved))
    with patch.object(c,"gpu_available",side_effect=AssertionError("second query forbidden")):
        assert c.startup_gpu(path,conf)["free_mib"]==3865
        Path(path).write_text('{"changed":true}')
        with pytest.raises(ValueError,match="snapshot/config"):c.startup_gpu(path,conf)

def test_real_shared_configuration_is_cpu_only():
    with patch.object(torch,"load",side_effect=AssertionError("weights forbidden")),patch.object(torch.cuda,"_lazy_init",side_effect=AssertionError("CUDA forbidden")):
        conf=c.configuration(CONFIG)
    assert conf["gpu_policy"]=="shared_observe_only" and conf["minimum_free_mib"]==3072
    assert not torch.cuda.is_initialized()

STUB=r'''#!/usr/bin/env python3
import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
def value(key,default=""):
    return args[args.index(key)+1] if key in args else default
action=value("--action");scope=value("--scope");model=value("--model")
key=":".join((action,scope,model))
control=Path(os.environ["CPU_FIXTURE_CONTROL"])
with (control/"calls").open("a") as f:f.write(key+"\n")
if key==os.environ.get("CPU_FIXTURE_SLEEP"):time.sleep(30)
if key==os.environ.get("CPU_FIXTURE_FAIL"):
    print("CUDA out of memory (CPU fixture; no CUDA used)",flush=True);sys.exit(42)
if action=="resources":
    print(json.dumps({"ready":True,"GPU":{"free_mib":3865,"utilization_percent":99}}));sys.exit(0)
if action=="preflight":marker="B3_CPU_PREFLIGHT_PASS "+scope
elif action=="pixels":marker="B3_PIXELS_PASS "+scope
elif action=="run":
    out=Path(os.environ["CPU_FIXTURE_BASE"])/scope/model/"completed"
    out.mkdir(parents=True);(out/"summary.json").write_text('{"fixture":true}')
    marker="B3_EVALUATION_PASS "+scope+" "+model
elif action=="audit":
    assert (Path(os.environ["CPU_FIXTURE_BASE"])/scope/model/"completed/summary.json").exists()
    marker="B3_CPU_AUDIT_PASS "+scope+" "+model
else:raise AssertionError(args)
if key!=os.environ.get("CPU_FIXTURE_MISSING"):print(marker,flush=True)
'''

def fixture_script(tmp_path):
    work=tmp_path/"work";work.mkdir()
    control=tmp_path/"control";base=tmp_path/"tbps-framework-b3";legacy=tmp_path/"legacy"
    stub=tmp_path/"cpu_mock_python";stub.write_text(STUB);stub.chmod(0o700)
    # 只替换生产脚本的路径常量；实际控制流、锁、trap、顺序、标志检查均原样执行。
    original=(ROOT/"src/tbps/baselines/run_b3_full.sh").read_text()
    values={"WORK":work,"PY":stub,"BASE":base,"CONTROL":control,"LEGACY":legacy}
    lines=original.splitlines()
    for k,v in values.items():
        found=[i for i,line in enumerate(lines) if line.startswith(k+"=")]
        assert len(found)==1
        lines[found[0]]=k+"="+str(v)
    script=tmp_path/"run_b3_full.sh";script.write_text("\n".join(lines)+"\n")
    env=dict(os.environ,CPU_FIXTURE_CONTROL=str(control),CPU_FIXTURE_BASE=str(base))
    return script,env,control,base

def invoke(script,env):
    return subprocess.run(["bash",str(script),"--allow-full"],env=env,start_new_session=True,capture_output=True,text=True,timeout=10)

def calls(control):
    return (control/"calls").read_text().splitlines() if (control/"calls").exists() else []

def test_actual_bash_chain_order_single_snapshot_and_once_only(tmp_path):
    script,env,control,base=fixture_script(tmp_path)
    first=invoke(script,env);assert first.returncode==0,first.stdout+first.stderr
    expected=["resources::","preflight:full:"]
    for m in ("irra","rasa","aptm"):expected.extend(["run:smoke:"+m,"audit:smoke:"+m])
    expected.append("pixels:full:")
    for m in ("irra","rasa","aptm"):expected.extend(["run:full:"+m,"audit:full:"+m])
    assert calls(control)==expected
    assert (control/"phase").read_text().strip()=="SUCCESS"
    assert (control/"supervisor.pid").read_text()==(control/"supervisor.pgid").read_text()
    second=invoke(script,env)
    assert second.returncode==4 and calls(control)==expected
    assert (control/"phase").read_text().strip()=="SUCCESS"

@pytest.mark.parametrize("fail",["run:smoke:rasa","audit:smoke:rasa","run:full:irra","audit:full:irra"])
def test_any_failure_stops_preserves_and_never_retries(tmp_path,fail):
    script,env,control,base=fixture_script(tmp_path);env["CPU_FIXTURE_FAIL"]=fail
    p=invoke(script,env);assert p.returncode==42
    observed=calls(control);assert observed[-1]==fail and observed.count(fail)==1
    assert (control/"phase").read_text().strip()=="FAIL"
    assert (control/"started").exists()
    assert any("CUDA out of memory (CPU fixture" in x.read_text() for x in (control/"logs").glob("*.log"))

def test_exit_zero_without_exact_audit_marker_refuses_full(tmp_path):
    script,env,control,base=fixture_script(tmp_path);env["CPU_FIXTURE_MISSING"]="audit:smoke:rasa"
    p=invoke(script,env);assert p.returncode!=0
    assert calls(control)[-1]=="audit:smoke:rasa"
    assert not any(x.startswith("run:full:") for x in calls(control))

def test_existing_output_refused_before_any_resource_or_model_call(tmp_path):
    script,env,control,base=fixture_script(tmp_path)
    out=base/"full";out.mkdir(parents=True);evidence=out/"evidence";evidence.write_text("unchanged")
    before=evidence.stat().st_mtime_ns
    p=invoke(script,env);assert p.returncode==4 and calls(control)==[]
    assert evidence.read_text()=="unchanged" and evidence.stat().st_mtime_ns==before
    assert not (control/"started").exists()

def wait_for_fixture_phase(control,expected,process):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        if (control/"phase").exists() and (control/"phase").read_text().strip()==expected:return
        assert process.poll() is None
        time.sleep(.02)
    raise AssertionError("fixture phase timeout")

@pytest.mark.parametrize("concurrent",[False,True])
def test_private_session_cancel_and_concurrent_lock(tmp_path,concurrent):
    script,env,control,base=fixture_script(tmp_path);env["CPU_FIXTURE_SLEEP"]="run:smoke:irra"
    p=subprocess.Popen(["bash",str(script),"--allow-full"],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        wait_for_fixture_phase(control,"RUNNING_SMOKE_irra",p)
        assert int((control/"supervisor.pgid").read_text())==p.pid
        if concurrent:
            second=invoke(script,env);assert second.returncode==3
            assert (control/"phase").read_text().strip()=="RUNNING_SMOKE_irra"
        os.killpg(p.pid,signal.SIGTERM)
        assert p.wait(timeout=5)==143
        assert (control/"phase").read_text().strip()=="CANCELLED"
        assert not any(x.startswith("run:full:") for x in calls(control))
    finally:
        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
