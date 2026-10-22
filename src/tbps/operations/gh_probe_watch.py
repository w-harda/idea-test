"""只轮询资源并单次调用冻结native_cli；标准库，不导入Torch。"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

FROZEN = Path("/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022")
CONTROL = Path("/home/lzf/ldx/tmp/gh_native_probe_wait_v1")
PYTHON = "/home/lzf/ldx/envs/tbps-image/bin/python"
COMMIT = "07a31edb4d1a4fc8e28ce8968a037b03b5681075"
INTERVAL = 180
MIN_GPU = 10240

def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def write(path, value, exclusive=False):
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    target = path if exclusive else path.with_name(path.name + ".tmp")
    with target.open("xb" if exclusive else "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    if not exclusive:
        os.replace(target, path)
    sync_dir(path.parent)

class Lease:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.fd)
            raise RuntimeError("another watcher owns exclusive lock")
    def close(self):
        os.close(self.fd)

@contextmanager
def transition(control):
    with (control / "transition.lock").open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield

def command(args, **kw):
    return subprocess.check_output(args, text=True, timeout=30, **kw).strip()

def gpu_snapshot():
    raw = command(["nvidia-smi", "--id=0",
        "--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"])
    parts = [x.strip() for x in raw.split(",")]
    require(len(parts) == 5 and len(raw.splitlines()) == 1, "malformed GPU0 state")
    uuid, name, free, total, util = parts
    processes = command(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"])
    selected = [line for line in processes.splitlines() if line.split(",")[0].strip() == uuid]
    return dict(time=now(), gpu_uuid=uuid, name=name, free_mib=int(free),
                total_mib=int(total), utilization=int(util), processes=selected)

def other_probe_processes():
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            args = (p / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except (OSError, PermissionError):
            continue
        if any("tbps.alternating.native_cli" == a or a.endswith("/alternating/native_cli.py")
               for a in args) and "--action" in args:
            i = args.index("--action")
            if i + 1 < len(args) and args[i + 1] == "probe":
                found.append(int(p.name))
    return found

class Environment:
    def __init__(self, control, deployment, lock_fd):
        self.control, self.deployment, self.lock_fd = control, deployment, lock_fd
        self.config_path = FROZEN / "configs/framework/gh_bidirectional_backend.server.json"
        self.c = read(self.config_path)
        self.probe = Path(self.c["probe_output"])
        self.log = logging.getLogger("gh_probe_wait")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        handler = RotatingFileHandler(control / "poll.log", maxBytes=1024 * 1024,
                                      backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.log.addHandler(handler)

    def stopped(self):
        return (self.control / "cancel.request.json").exists()

    def guard(self):
        require(not os.path.lexists(self.probe), "probe_v1 already exists; preserve and stop")
        require(not (self.control / "probe.started.json").exists(),
                "single probe launch already consumed; never retry")
        require(not other_probe_processes(), "another native probe process exists")

    def frozen(self):
        require(command(["git", "rev-parse", "HEAD"], cwd=FROZEN) == COMMIT,
                "frozen worktree HEAD changed")
        require(not command(["git", "status", "--porcelain"], cwd=FROZEN), "frozen worktree dirty")
        require(sha(self.config_path) == self.deployment["config_sha256"], "config SHA mismatch")
        require(sha(__file__) == self.deployment["script_sha256"], "watcher script SHA mismatch")
        for p, h in self.c["producers"].items():
            require(sha(p) == h, "producer SHA mismatch: " + p)

    def history(self, label):
        evidence = read(FROZEN / "docs/framework/results/gh_bidirectional_backend/SOURCE_EVIDENCE.json")
        ref = evidence["history_protection"]["snapshot"]
        require(sha(ref["path"]) == ref["sha256"], "history snapshot SHA mismatch")
        snapshot = read(ref["path"])
        bad = []
        for p, expected in snapshot.items():
            try:
                s = Path(p).stat()
                equal = [s.st_size, s.st_mtime_ns, s.st_ino] == expected
            except OSError:
                equal = False
            if not equal:
                bad.append(p)
        result = dict(time=now(), files=len(snapshot), mismatches=bad,
                      semantics="metadata sweep; CPU preflight checks critical source SHA")
        write(self.control / (label + ".json"), result, exclusive=True)
        require(not bad, "historical file metadata changed")

    def resource(self):
        import shutil
        gpu = gpu_snapshot()
        limits = self.c["prototype"]["proposed_gpu_limits"]
        mem = dict((x.split(":")[0], int(x.split()[1]) * 1024)
                   for x in Path("/proc/meminfo").read_text().splitlines())
        disk = shutil.disk_usage(self.c["prototype"]["resource_probe_directory"])
        ino = os.statvfs(self.c["prototype"]["resource_probe_directory"])
        gpu.update(available_ram_bytes=mem["MemAvailable"], free_disk_bytes=disk.free,
                   free_inodes=ino.f_favail, minimum_free_mib=MIN_GPU)
        gpu["checks"] = dict(gpu=gpu["free_mib"] >= MIN_GPU,
            ram=mem["MemAvailable"] >= limits["minimum_ram_bytes"],
            disk=disk.free >= limits["minimum_disk_bytes"],
            inodes=ino.f_favail >= limits["minimum_free_inodes"])
        gpu["eligible"] = all(gpu["checks"].values())
        write(self.control / "resources.latest.json", gpu)
        return gpu

    def status(self, state, **extra):
        previous = read(self.control / "status.json") if (self.control / "status.json").exists() else {}
        value = dict(previous, state=state, time=now(), pid=os.getpid(), ppid=os.getppid(),
                     pgid=os.getpgrp(), sid=os.getsid(0), interval_seconds=INTERVAL, **extra)
        write(self.control / "status.json", value)
        self.log.info(json.dumps(value, ensure_ascii=False))

    def argv(self, *extra):
        return [PYTHON, "-B", "-u", "-m", "tbps.alternating.native_cli",
                "--config", str(self.config_path), *extra]

    def env(self, gpu=False):
        return dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                    PYTHONPATH=str(FROZEN / "src"), TMPDIR=str(self.control),
                    OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                    CUDA_VISIBLE_DEVICES="0" if gpu else "")

    def preflight(self):
        # 资源回落可能导致以后再次CPU预检；仅滚动已通过的CPU日志，失败即退出。
        path = self.control / "cpu_preflight.log"
        if path.exists():
            for i in (2, 1):
                source = path.with_name(path.name + ("" if i == 1 else ".1"))
                if source.exists():
                    os.replace(source, path.with_name(path.name + "." + str(i)))
        with path.open("x") as f:
            p = subprocess.run(self.argv("--action", "preflight", "--check-parent-states"),
                cwd=FROZEN, env=self.env(), stdout=f, stderr=subprocess.STDOUT)
        write(self.control / "cpu_preflight.latest.json",
              dict(time=now(), exit_code=p.returncode, sha256=sha(path)))
        require(p.returncode == 0 and "GH_NATIVE_CPU_PREFLIGHT_PASS" in path.read_text(),
                "CPU Preflight failed; no GPU attempt")
        self.frozen()
        # 完整历史metadata循环只在真正具备启动条件时运行。
        ref = read(FROZEN / "docs/framework/results/gh_bidirectional_backend/SOURCE_EVIDENCE.json")
        src = ref["history_protection"]["snapshot"]
        snapshot = read(src["path"])
        require(sha(src["path"]) == src["sha256"], "history snapshot changed")
        for p, expected in snapshot.items():
            s = Path(p).stat()
            require([s.st_size, s.st_mtime_ns, s.st_ino] == expected, "history changed: " + p)
        self.status("WATCHING", detail="CPU_PREFLIGHT_PASS", history_files=len(snapshot))

    def consume(self):
        with transition(self.control):
            self.guard()
            require(not self.stopped(), "waiting cancelled")
            # fsync + O_EXCL：在Popen之前永久消费授权，崩溃后也不自动重新调用。
            write(self.control / "probe.started.json", dict(time=now(), pid=os.getpid(),
                script_sha256=sha(__file__), frozen_commit=COMMIT, action="probe",
                command=self.argv("--action", "probe", "--allow-gpu-probe"),
                note="launch consumed before Popen; any exit means no retry"), exclusive=True)
            self.status("PROBE_RUNNING", detail="single launch consumed")

    def launch(self):
        args = self.argv("--action", "probe", "--allow-gpu-probe")
        with (self.control / "probe.log").open("x") as f:
            p = subprocess.Popen(args, cwd=FROZEN, env=self.env(True),
                stdout=f, stderr=subprocess.STDOUT, pass_fds=(self.lock_fd,))
            write(self.control / "probe.process.json", dict(pid=p.pid, time=now()), exclusive=True)
            code = p.wait()
        write(self.control / "probe.exit.json", dict(exit_code=code, time=now()), exclusive=True)
        require(code == 0, "native probe failed; retain FAIL/logs and never retry")
        # 只调用原有不可变来源/回执验证器，不另写数值验收算法。
        verify = ("import sys; from tbps.alternating.native_sources import verify_config; "
                  "from tbps.alternating.native_gate import verify_probe; "
                  "print(verify_probe(verify_config(sys.argv[1])))")
        with (self.control / "receipt_check.log").open("x") as f:
            v = subprocess.run([PYTHON, "-B", "-c", verify, str(self.config_path)],
                cwd=FROZEN, env=self.env(), stdout=f, stderr=subprocess.STDOUT)
        require(v.returncode == 0, "native probe receipt verification failed")
        self.frozen()
        self.history("history_after")
        report = read(self.probe / "completed/report.json")
        write(self.control / "probe.pass.json", dict(status=report["status"],
            manifest_sha256=sha(self.probe / "completed/manifest.json"),
            report_sha256=sha(self.probe / "completed/report.json"),
            source_sha256=report["source_sha256"],
            CUDA_RNG_restore_evidence=report.get("CUDA_RNG_restored", "NOT_REPORTED"),
            note="only native numerical probe; no A1/B Smoke or training"), exclusive=True)

class Poller:
    """状态机可用纯CPU Fixture驱动；自身不包含模型或攻击算法。"""
    def __init__(self, env):
        self.env, self.streak, self.checks = env, 0, 0

    def tick(self):
        e = self.env
        if e.stopped():
            e.status("WATCHING", detail="CANCELLED", terminal=True)
            return True
        e.guard()
        self.checks += 1
        resource = e.resource()
        self.streak = self.streak + 1 if resource["eligible"] else 0
        e.status("WATCHING", check_count=self.checks, consecutive=self.streak,
                 free_mib=resource["free_mib"], utilization=resource["utilization"],
                 checks=resource["checks"], processes=resource["processes"])
        if self.streak < 2:
            return False
        e.frozen()
        e.preflight()
        if e.stopped():
            e.status("WATCHING", detail="CANCELLED", terminal=True)
            return True
        e.guard()
        final = e.resource()
        if not final["eligible"]:
            self.streak = 0
            e.status("WATCHING", check_count=self.checks, consecutive=0,
                     detail="resources fell after CPU preflight", free_mib=final["free_mib"])
            return False
        e.consume()
        e.launch()
        e.status("PROBE_PASS", check_count=self.checks, terminal=True)
        return True

def cancel(control):
    with transition(control):
        state = read(control / "status.json")
        require(state["state"] == "WATCHING" and not (control / "probe.started.json").exists(),
                "cancel only allowed while waiting; do not signal a running probe")
        write(control / "cancel.request.json", dict(time=now(), by="manual waiting cancellation"),
              exclusive=True)
    print("cancel request saved; watcher will exit without starting the probe")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default=str(CONTROL))
    ap.add_argument("--cancel", action="store_true")
    a = ap.parse_args()
    control = Path(a.control)
    require(control.resolve() == CONTROL, "fixed dedicated control directory only")
    if a.cancel:
        cancel(control)
        return
    lease = Lease(control / "watcher.lock")
    e = None
    try:
        require(not (control / "watcher.started.json").exists(),
                "watcher already started before; no automatic restart")
        d = read(control / "deployment.json")
        require(sha(__file__) == d["script_sha256"], "deployed watcher SHA mismatch")
        e = Environment(control, d, lease.fd)
        e.guard()
        e.frozen()
        e.history("history_before")
        write(control / "watcher.started.json", dict(time=now(), pid=os.getpid(),
            ppid=os.getppid(), pgid=os.getpgrp(), sid=os.getsid(0), script_sha256=sha(__file__)),
            exclusive=True)
        for key, value in (("pid", os.getpid()), ("pgid", os.getpgrp()), ("sid", os.getsid(0))):
            with (control / ("supervisor." + key)).open("x") as f:
                f.write(str(value) + "\n")
        # 信号仅停止等待；运行中的探针不由此处理器终止。
        def stop(sig, frame):
            if not (control / "probe.started.json").exists() and not e.stopped():
                write(control / "cancel.request.json", dict(time=now(), signal=sig), exclusive=True)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        poller = Poller(e)
        while not poller.tick():
            # 180秒资源周期；取消请求最多1秒生效，不执行额外GPU查询。
            deadline = time.monotonic() + INTERVAL
            while time.monotonic() < deadline and not e.stopped():
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    except BaseException as error:
        if e is not None:
            e.status("PROBE_FAIL", detail=str(error), terminal=True,
                     probe_launch_consumed=(control / "probe.started.json").exists())
            write(control / "failure.json", dict(time=now(), error=repr(error)), exclusive=True)
        raise
    finally:
        lease.close()

if __name__ == "__main__":
    main()
