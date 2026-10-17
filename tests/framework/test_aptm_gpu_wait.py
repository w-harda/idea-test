"""真实Bash控制流程CPU测试；模型、nvidia-smi及等待均为合成Fixture。"""
from pathlib import Path
import os,json,hashlib,subprocess,time,signal,fcntl
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/"src/tbps/engines/wait_aptm_full_chain.sh"
def executable(p,s):p.write_text(s);p.chmod(0o755)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def artifact(folder,files,provenance):
    folder.mkdir(parents=True)
    for name,value in files.items():(folder/name).write_text(json.dumps(value))
    manifest={"schema":"tbps-artifact-1","artifact_id":folder.name,"provenance":provenance,
              "files_sha256":{name:sha(folder/name) for name in files},"sources":[]}
    (folder/"manifest.json").write_text(json.dumps(manifest))
    (folder/"completed.json").write_text(json.dumps({"schema":"tbps-artifact-receipt-1","manifest_sha256":sha(folder/"manifest.json")}))

@pytest.fixture
def world(tmp_path):
    bin_=tmp_path/"bin";bin_.mkdir();work=tmp_path/"work";work.mkdir()
    run=tmp_path/"wait";rasa=tmp_path/"rasa";rasa.mkdir();out=tmp_path/"rasa_full"
    (rasa/"phase").write_text("SUCCESS\n")
    (rasa/"launch.log").write_text("2026-10-18 | SUCCESS | RASA_FULL_AND_CPU_AUDIT_SUCCESS\n")
    (rasa/"full_gpu.log").write_text("RASA_MATERIALIZED_EVALUATION_PASS full\n")
    (rasa/"cpu_audit.log").write_text("RASA_MATERIALIZED_CPU_AUDIT_PASS full\n")
    (rasa/"status.log").write_text("SUCCESS\n")
    artifact(out/"completed",{"summary.json":{"gallery_count":3074,"query_count":6156,
        "groups":dict.fromkeys(["Clean","Image-only","Text-only","Joint"],{})},"metrics.csv":{}},{"scope":"full"})
    artifact(out/"execution-000001",{"report.json":{"status":"PASS","model_frozen":True}},{})
    (tmp_path/"gpu").write_text("24000, 0\n"*12)
    (tmp_path/"processes").write_text("\n"*12)
    executable(bin_/"nvidia-smi",'''#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu="* ]]; then
 n=$(cat "$TEST_ROOT/index" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$TEST_ROOT/index"
 value=$(sed -n "${n}p" "$TEST_ROOT/gpu"); [[ -n "$value" ]] || value=$(tail -n 1 "$TEST_ROOT/gpu")
 [[ "$value" != ERROR ]] || exit 1
 echo "$value"
else
 n=$(cat "$TEST_ROOT/index"); sed -n "${n}p" "$TEST_ROOT/processes"
fi
''')
    executable(bin_/"pgrep",'''#!/usr/bin/env bash
if [[ "$*" == *"[a]ptm"* && -e "$TEST_ROOT/aptm_running" ]]; then echo 123; exit 0; fi
if [[ "$*" == *"[r]asa"* && -e "$TEST_ROOT/rasa_running" ]]; then echo 124; exit 0; fi
exit 1
''')
    executable(bin_/"ps",'''#!/usr/bin/env bash
if [[ "$*" == *"124"* ]]; then echo "python rasa_eval.py --action run --scope full"
else echo "python aptm_eval.py --action run --scope smoke"; fi
''')
    executable(bin_/"sleep",'#!/usr/bin/env bash\necho "$*" >> "$TEST_ROOT/ticks"\n')
    executable(bin_/"fake_python",'''#!/usr/bin/env bash
action=""; scope=""
while (( $# )); do
 case "$1" in --action) action="$2"; shift;; --scope) scope="$2"; shift;; esac
 shift
done
stage="${scope}_${action}"
echo "$stage" >> "$TEST_ROOT/calls"
echo "$stage $(cat "$TEST_ROOT/index")" >> "$TEST_ROOT/call_indices"
echo "CUDA=$CUDA_VISIBLE_DEVICES" >> "$TEST_ROOT/cuda_env"
if [[ "${FAIL_STAGE:-}" == "$stage" ]]; then echo "synthetic failure"; exit 9; fi
if [[ "$action" == run ]]; then
 mkdir -p "$APTM_OUTPUT_BASE/$scope/completed"
 echo '{}' > "$APTM_OUTPUT_BASE/$scope/completed/manifest.json"
 echo '{}' > "$APTM_OUTPUT_BASE/$scope/completed/completed.json"
 marker="APTM_MATERIALIZED_EVALUATION_PASS $scope"
else marker="APTM_MATERIALIZED_CPU_AUDIT_PASS $scope"; fi
if [[ "${MISSING_STAGE:-}" != "$stage" ]]; then echo "$marker"; fi
''')
    env=os.environ.copy();env.update(PATH=str(bin_)+":"+env["PATH"],TEST_ROOT=str(tmp_path),
        APTM_WORKTREE=str(work),APTM_PYTHON=str(bin_/"fake_python"),APTM_WAIT_DIR=str(run),
        APTM_OUTPUT_BASE=str(tmp_path/"aptm"),APTM_RASA_WAIT_DIR=str(rasa),APTM_RASA_FULL_DIR=str(out))
    def execute():return subprocess.run(["bash",str(SCRIPT)],env=env,capture_output=True,text=True,timeout=8)
    return tmp_path,bin_,run,rasa,out,env,execute
def calls(root):
    p=root/"calls";return p.read_text().splitlines() if p.exists() else []
def direct(w,code):
    root,_,run,_,_,env,_=w;run.mkdir(exist_ok=True)
    return subprocess.run(["bash","-c",'source "$1"; '+code,"fixture",str(SCRIPT)],env=env,capture_output=True,text=True,timeout=5)

def test_complete_once_legacy_rasa_evidence_and_both_three_sample_gates(world):
    root,_,run,rasa,out,_,execute=world
    before={str(p):(sha(p),p.stat().st_mtime_ns) for base in (rasa,out) for p in base.rglob("*") if p.is_file()}
    x=execute();assert x.returncode==0,x.stderr
    assert calls(root)==["smoke_run","smoke_audit","full_run","full_audit"]
    assert (root/"call_indices").read_text().splitlines()==["smoke_run 3","smoke_audit 3","full_run 6","full_audit 6"]
    assert (root/"cuda_env").read_text().splitlines()==["CUDA=0","CUDA=","CUDA=0","CUDA="]
    assert (run/"phase").read_text().strip()=="SUCCESS"
    assert "SUCCESS | APTM_FULL_AND_CPU_AUDIT" in (run/"launch.log").read_text()
    assert (run/"triggered").exists() and (run/"full_triggered").exists()
    proof=json.loads((run/"rasa_verified.json").read_text());assert proof["not_a_new_rasa_audit"]
    for p,(h,mtime) in before.items():assert (sha(Path(p)),Path(p).stat().st_mtime_ns)==(h,mtime)
    assert execute().returncode==73 and len(calls(root))==4
    assert (run/"phase").read_text().strip()=="SUCCESS"

def test_resource_thresholds_ollama_busy_reset_and_graphics(world):
    root,_,run,_,_,_,execute=world
    (root/"gpu").write_text("24000, 0\n19999, 0\n24000, 11\n20000, 10\n20000, 10\n24000, 93\n20000, 10\n20000, 10\n20000, 10\n20000, 10\n20000, 10\n20000, 10\n")
    (root/"processes").write_text("2436556, /usr/local/lib/ollama/llama-server, 20178\n\n\n"+"123, /usr/lib/xorg/Xorg, 16\n"*9)
    assert execute().returncode==0
    assert (root/"call_indices").read_text().splitlines()[0]=="smoke_run 9"
    assert (root/"call_indices").read_text().splitlines()[2]=="full_run 12"
    assert "20178" in (run/"gpu.log").read_text()

def test_Full_gpu_conditions_restart_after_successful_smoke(world):
    root,_,_,_,_,_,execute=world
    (root/"gpu").write_text("24000, 0\n"*3+"24000, 93\n"+"24000, 0\n"*3)
    assert execute().returncode==0
    assert (root/"call_indices").read_text().splitlines()[2]=="full_run 7"

@pytest.mark.parametrize("stage",["smoke_run","smoke_audit","full_run","full_audit"])
def test_each_failure_stops_immediately_without_retry(world,stage):
    root,_,run,_,_,env,execute=world;env["FAIL_STAGE"]=stage
    assert execute().returncode==9
    expected=["smoke_run","smoke_audit","full_run","full_audit"]
    assert calls(root)==expected[:expected.index(stage)+1]
    assert (run/"phase").read_text().strip()=="FAIL" and (run/"triggered").exists()
    assert execute().returncode==73 and calls(root)==expected[:expected.index(stage)+1]

def test_missing_success_marker_never_advances(world):
    root,_,run,_,_,env,execute=world;env["MISSING_STAGE"]="smoke_audit"
    assert execute().returncode==1
    assert calls(root)==["smoke_run","smoke_audit"] and (run/"phase").read_text().strip()=="FAIL"

def test_explicit_rasa_failure_stops_without_GPU_query(world):
    root,_,run,rasa,_,_,execute=world;(rasa/"phase").write_text("FAILED\n")
    (rasa/"status.log").write_text("FAILED OOM\n")
    assert execute().returncode==1 and calls(root)==[] and not (root/"index").exists()
    assert "OOM" in (run/"poll.log").read_text()

@pytest.mark.parametrize("case",["missing_audit","bad_completion","bad_report","wrong_size","running","canonical_marker"])
def test_rasa_readiness_requires_real_existing_evidence(world,case):
    root,_,run,rasa,out,_,_=world
    if case=="missing_audit":(rasa/"cpu_audit.log").write_text("not complete\n")
    if case=="bad_completion":(out/"completed/completed.json").write_text("{}")
    if case=="bad_report":(out/"execution-000001/report.json").write_text("{}")
    if case=="wrong_size":
        artifact2=out/"completed/summary.json";d=json.loads(artifact2.read_text());d["query_count"]=10;artifact2.write_text(json.dumps(d))
    if case=="running":(root/"rasa_running").touch()
    if case=="canonical_marker":(rasa/"launch.log").write_text("timestamp | SUCCESS | RASA_FULL_AND_CPU_AUDIT\n")
    x=direct(world,"rasa_ready")
    assert x.returncode==({"missing_audit":1,"bad_completion":2,"bad_report":2,"wrong_size":2,"running":1,"canonical_marker":0}[case])

@pytest.mark.parametrize("case",["smoke_output","full_output","process","lock"])
def test_duplicate_output_process_or_lock_refused(world,case):
    root,_,run,_,_,env,execute=world
    if case.endswith("output"):(Path(env["APTM_OUTPUT_BASE"])/case.split("_")[0]).mkdir(parents=True)
    if case=="process":(root/"aptm_running").touch()
    if case=="lock":
        run.mkdir()
        with (run/"wait.lock").open("w") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);assert execute().returncode==73
    else:assert execute().returncode==73
    assert calls(root)==[] and not (root/"index").exists()

def test_query_error_resets_streak(world):
    root,_,run,_,_,_,execute=world
    (root/"gpu").write_text("24000, 0\nERROR\n"+"24000, 0\n"*6)
    assert execute().returncode==0
    assert (root/"call_indices").read_text().splitlines()[0]=="smoke_run 5"
    assert "QUERY_FAILED" in (run/"gpu.log").read_text()

def test_waiting_rasa_can_cancel_only_own_sleep(world):
    root,bin_,run,rasa,_,env,_=world
    (rasa/"phase").write_text("RUNNING_FULL\n")
    executable(bin_/"sleep",'#!/usr/bin/env bash\nexec /usr/bin/sleep "$@"\n')
    p=subprocess.Popen(["bash",str(SCRIPT)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+3
        while not (run/"poll.log").exists() and time.monotonic()<deadline:time.sleep(.01)
        assert (run/"phase").read_text().strip()=="WAITING_RASA"
        p.send_signal(signal.SIGTERM);_,err=p.communicate(timeout=3)
        assert p.returncode==0,err
        assert calls(root)==[] and not (run/"triggered").exists()
        assert (run/"phase").read_text().strip()=="FAIL"
    finally:
        if p.poll() is None:p.kill();p.communicate()
