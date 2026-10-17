"""真实Bash控制流程的CPU测试；nvidia-smi、模型命令和sleep均使用Fixture。"""
from pathlib import Path
import os,subprocess,signal,time,fcntl
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/"src/tbps/engines/wait_rasa_full.sh"
def executable(p,text):
    p.write_text(text);p.chmod(0o755)

@pytest.fixture
def world(tmp_path):
    commands=tmp_path/"bin";commands.mkdir()
    run=tmp_path/"wait";work=tmp_path/"work";work.mkdir()
    gpu=tmp_path/"gpu";process=tmp_path/"processes"
    gpu.write_text("9000, 0\n"*3);process.write_text("\n"*3)
    executable(commands/"nvidia-smi",'''#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu="* ]]; then
 n=$(cat "$TEST_ROOT/index" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$TEST_ROOT/index"
 value=$(sed -n "${n}p" "$TEST_ROOT/gpu"); [[ -n "$value" ]] || value=$(tail -n 1 "$TEST_ROOT/gpu")
 [[ "$value" != FAIL ]] || exit 1
 echo "$value"
else
 n=$(cat "$TEST_ROOT/index"); sed -n "${n}p" "$TEST_ROOT/processes"
fi
''')
    executable(commands/"pgrep","#!/usr/bin/env bash\nexit 1\n")
    executable(commands/"sleep",'#!/usr/bin/env bash\necho "$*" >> "$TEST_ROOT/ticks"\n')
    executable(commands/"fake_python",'''#!/usr/bin/env bash
if [[ "$*" == *"--action run"* ]]; then
 echo FULL >> "$TEST_ROOT/calls"
 if [[ "${FAIL_FULL:-0}" == 1 ]]; then echo "synthetic Full failure"; exit 9; fi
 echo "RASA_MATERIALIZED_EVALUATION_PASS full"
else
 echo AUDIT >> "$TEST_ROOT/calls"
 if [[ "${FAIL_AUDIT:-0}" == 1 ]]; then echo "synthetic audit failure"; exit 6; fi
 echo "RASA_MATERIALIZED_CPU_AUDIT_PASS full"
fi
''')
    env=os.environ.copy();env.update(PATH=str(commands)+":"+env["PATH"],TEST_ROOT=str(tmp_path),
        RASA_WORKTREE=str(work),RASA_WAIT_DIR=str(run),RASA_FULL_OUTPUT=str(tmp_path/"full"),
        RASA_PYTHON=str(commands/"fake_python"))
    def execute():return subprocess.run(["bash",str(SCRIPT)],env=env,capture_output=True,text=True,timeout=5)
    return tmp_path,commands,run,env,execute

def calls(root):
    p=root/"calls";return p.read_text().splitlines() if p.exists() else []

def test_thresholds_and_other_compute_require_three_good_samples(world):
    root,_,run,env,execute=world
    (root/"gpu").write_text("10000, 0\n8191, 0\n10000, 11\n8192, 10\n8192, 10\n8192, 10\n")
    (root/"processes").write_text("2436556, /usr/local/lib/ollama/llama-server, 20178\n\n\n\n\n\n")
    result=execute()
    assert result.returncode==0,result.stderr
    assert calls(root)==["FULL","AUDIT"]
    assert (run/"phase").read_text().strip()=="SUCCESS"
    assert (root/"index").read_text().strip()=="6"
    assert (root/"ticks").read_text().splitlines()==["60"]*5
    assert "3/3" in (run/"poll.log").read_text()
    assert "20178" in (run/"poll.log").read_text()
    assert (run/"triggered").exists()

def test_streak_resets_when_utilization_becomes_busy(world):
    root,_,run,_,execute=world
    (root/"gpu").write_text("9000, 0\n9000, 0\n9000, 93\n9000, 0\n9000, 0\n9000, 0\n")
    (root/"processes").write_text("\n"*6)
    assert execute().returncode==0
    streaks=[x.rsplit(" | ",1)[1] for x in (run/"poll.log").read_text().splitlines() if "| STREAK |" in x]
    assert streaks==["1/3","2/3","0/3","1/3","2/3","3/3"]

def test_query_failure_resets_and_does_not_launch_early(world):
    root,_,run,_,execute=world
    (root/"gpu").write_text("9000, 0\nFAIL\n9000, 0\n9000, 0\n9000, 0\n")
    (root/"processes").write_text("\n"*5)
    assert execute().returncode==0
    assert (root/"index").read_text().strip()=="5"
    assert "QUERY_FAILED" in (run/"poll.log").read_text()

def test_Xorg_graphics_is_exempt(world):
    root,_,_,_,execute=world
    (root/"processes").write_text("123, /usr/lib/xorg/Xorg, 16\n"*3)
    assert execute().returncode==0
    assert calls(root)==["FULL","AUDIT"]

def test_existing_output_prevents_wait_and_launch(world):
    root,_,run,env,execute=world
    Path(env["RASA_FULL_OUTPUT"]).mkdir()
    assert execute().returncode==73
    assert calls(root)==[] and not (root/"index").exists() and not (run/"triggered").exists()

def test_running_Full_prevents_duplicate(world):
    root,commands,_,_,execute=world
    executable(commands/"pgrep","#!/usr/bin/env bash\necho 123\n")
    executable(commands/"ps","#!/usr/bin/env bash\necho '/env/python rasa_eval.py --scope full --action run'\n")
    assert execute().returncode==73
    assert calls(root)==[] and not (root/"index").exists()

def test_flock_prevents_second_controller(world):
    root,_,run,_,execute=world
    run.mkdir()
    with (run/"wait.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert execute().returncode==73
    assert calls(root)==[] and not (root/"index").exists()

def test_Full_failure_never_retries_or_starts_audit(world):
    root,_,run,env,execute=world;env["FAIL_FULL"]="1"
    assert execute().returncode==9
    assert calls(root)==["FULL"] and (run/"phase").read_text().strip()=="FAILED"
    previous=(run/"triggered").read_bytes()
    assert execute().returncode==73
    assert calls(root)==["FULL"] and (run/"triggered").read_bytes()==previous

def test_CPU_audit_failure_stops_without_retry(world):
    root,_,run,env,execute=world;env["FAIL_AUDIT"]="1"
    assert execute().returncode==6
    assert calls(root)==["FULL","AUDIT"]
    assert "CPU audit exit=6" in (run/"status.log").read_text()

def test_cancel_wait_interrupts_own_sleep_only(world):
    root,commands,run,env,_=world
    (root/"gpu").write_text("4000, 93\n")
    executable(commands/"sleep","#!/usr/bin/env bash\nexec /usr/bin/sleep \"$@\"\n")
    process=subprocess.Popen(["bash",str(SCRIPT)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    try:
        deadline=time.monotonic()+3
        while not (run/"poll.log").exists() and time.monotonic()<deadline:time.sleep(.01)
        assert (run/"poll.log").exists()
        process.send_signal(signal.SIGTERM)
        _,stderr=process.communicate(timeout=3)
        assert process.returncode==0,stderr
        assert (run/"phase").read_text().strip()=="CANCELLED"
        assert calls(root)==[] and not (run/"triggered").exists()
    finally:
        if process.poll()is None:process.kill();process.communicate()
