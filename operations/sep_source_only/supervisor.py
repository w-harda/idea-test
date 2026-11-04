"""独立串行 Smoke；任何真实失败均终止，不重试。"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tbps.artifacts import sha_file
from tbps.baselines.cpgc_sep.dataset import require
from .common import CONFIG, PROTOCOL, ROOT, TARGETS, atomic, code_commit, progress, read, settings
from .preflight import protected

REPORT = Path("docs/framework/results/sep_source_only_smoke/REPORT.md")
BASELINE = Path("docs/framework/results/sep_source_only_smoke/EXECUTION_PROTOCOL.md")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def format_report(control, out, launch, result, failed=None):
    preflight = read(control/"preflight.json")
    lines = ["# SEP-TBPS Source-only Smoke", "", f"执行 Commit `{launch['commit']}`。",
             f"协议 `{PROTOCOL}`。详见[固定执行协议](EXECUTION_PROTOCOL.md)。",
             "", f"CPU Preflight PASS，{preflight['tests']['total']} 项 CPU 测试。",
             f"CUHK 官方 Train 34,054 图、11,003 PID。参考清单 SHA `{preflight['source_train_manifest_sha256']}`。",
             f"Landlock ABI {preflight['isolation']['landlock_abi']}，{preflight['isolation']['denied']} 个真实拒绝探针。",
             "", f"服务器控制目录 `{control}`。", f"输出目录 `{out}`。", "",
             "## 实际运行阶段", "", "| 阶段 | 状态 | 秒 |", "| --- | --- | ---: |"]
    lines += [f"| {r['phase']} | {r['status']} | {r['seconds']:.2f} |" for r in result]
    exact_path = out/"parity/exact/result.json"
    if exact_path.exists():
        exact = read(exact_path)
        lines += ["", "## A 类优化 Exact", "",
                  "同一份真实 FP32 原生矩阵通过两套生产 CPU classifier。PID 顺序和成员、聚合、softmax、原 margin、DPP、候选排序、缓存键与 RNG 一致。",
                  f"候选排序 `{exact['candidate_order']}`。冻结矩阵 SHA 如下。", ""]
        lines += [f"- {name} `{d['scores_sha256']}`，形状 {d['shape']}。" for name,d in exact["models"].items()]
    audit = out/"audit/completed/report.json"
    if audit.exists() and failed is None:
        audited = read(audit)
        lines += ["", "## 独立 CPU 审计 PASS", "",
                  "| Caption | 耗时（秒） | tau 候选数 | Source 文本前向 | Cache 命中率 | 词编辑 | NOOP |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        lines += [f"| {k} | {v['elapsed_seconds']:.3f} | {v['candidates']} | {v['text_forwards']} | {v['cache_hit_rate']:.2%} | {v['edits']} | {v['noop']} |"
                  for k,v in audited["queries"].items()]
        lines += ["", f"Source 一次编码 {audited['gallery']['image_forwards']} 图，耗时 {audited['gallery']['seconds']:.2f} 秒。",
                  "", "两组 CUHK Target 共用同一份冻结对抗文本。每个目标只做 2 Query × 128 Gallery。",
                  "", "| Target | 模式 | R@1 | R@5 | R@10 | mAP | 文本前向总数 | 融合 pair 总数 | 评分阶段秒 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for key, target in audited["targets"].items():
            ev = target["evidence"]
            for mode, metrics in target["metrics"].items():
                lines.append(f"| {key} | {mode} | {metrics['R@1']:.2f} | {metrics['R@5']:.2f} | {metrics['R@10']:.2f} | "
                             f"{metrics['mAP']:.2f} | {ev['text_forwards']} | {ev['fusion_pairs']} | {ev['elapsed_seconds']:.2f} |")
        if not any(q["noop"] for q in audited["queries"].values()):
            lines += ["", "固定真实集合未出现自然 NOOP。完整官方搜索的 NOOP 路径由 CPU 夹具验证。"]
    if failed:
        lines += ["", "## 本次执行停止", "", f"失败阶段 `{failed['phase']}`。错误 `{failed['error']}`。",
                  "未完成阶段保持未运行。详细 traceback、阶段日志和已冻结产物在服务器目录。"]
    lines += ["", "## Full 可行性", "",
              "真实 Smoke 仅用于验证链路与估算已测样本的开销。逐候选全 Train 检索、官方零方差循环和共享 GPU 资源均影响 Full 可行性。Full 尚未验证或授权。", ""]
    path = ROOT/REPORT
    temporary = path.with_suffix(".md.pending")
    with temporary.open("x") as stream:
        stream.write("\n".join(lines))
    os.replace(temporary, path)


def publish_report(control, out, launch, result, failed=None):
    require(code_commit() == launch["commit"] and not git("status", "--porcelain"),
            "report publication requires original clean branch")
    format_report(control, out, launch, result, failed)
    git("diff", "--check")
    git("add", "--", str(REPORT))
    git("commit", "-m", "docs: publish SEP Source-only Smoke execution receipt")
    branch = git("branch", "--show-current")
    git("push", "origin", f"HEAD:refs/heads/{branch}")
    commit = code_commit()
    require(git("ls-remote", "origin", f"refs/heads/{branch}").split()[0] == commit, "report remote SHA")
    atomic(control/"publication.json", {"status":"PUBLISHED","commit":commit,"branch":branch})


def execute():
    c, _, _ = settings()
    out, control = Path(c["output"]), Path(c["control"])
    preflight = read(control/"preflight.json")
    require(preflight["status"] == "PASS" and preflight["commit"] == code_commit()
            and preflight["config_sha256"] == sha_file(CONFIG) and c["scope"] == "smoke"
            and not git("status", "--porcelain"), "matching passed CPU preflight and clean commit")
    started = control/"supervisor.json"
    with (control/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with started.open("x") as f:
            launch = {"pid":os.getpid(),"pgid":os.getpgid(0),"sid":os.getsid(0),
                      "commit":code_commit(),"scope":"smoke","at":time.time()}
            require(launch["pid"] == launch["pgid"] == launch["sid"]
                    and not any(os.isatty(i) for i in (0,1,2)), "detached supervisor session")
            json.dump(launch, f)
        steps = [("parity-gpu",["parity-gpu"],False,1800), ("parity-cpu",["parity-cpu"],True,300),
                 ("source",["source",str(out),str(control),str(c["source_query_timeout_seconds"])],False,18000)]
        steps += [(key,["target",key],False,1800) for key in TARGETS]
        steps.append(("audit",["audit"],True,900))
        result, failed = [], None
        phase = "STARTING"
        try:
            for phase, args, cpu_only, timeout in steps:
                t = time.monotonic()
                atomic(control/"status.json",{**launch,"status":"RUNNING","phase":phase,"completed":result})
                progress(control, "PHASE_START", phase=phase)
                env = os.environ.copy()
                if cpu_only:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                with (control/f"{phase}.log").open("xb") as log:
                    if not cpu_only:
                        # Resource observations are informational. A telemetry failure is not an admission gate.
                        try:
                            subprocess.run(["nvidia-smi"], stdout=log, stderr=log, check=False, timeout=10)
                        except (OSError, subprocess.TimeoutExpired) as error:
                            print("GPU observation unavailable", error, file=sys.stderr)
                    command = [sys.executable,"-B","-m",__package__+".phases",*args]
                    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    atomic(control/"active_child.json",{"phase":phase,"pid":proc.pid})
                    try:
                        code = proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        # The child is in the exact session just created above.
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        raise TimeoutError(f"diagnostic stage timeout {phase}")
                    if code != 0:
                        raise subprocess.CalledProcessError(code, command)
                result.append({"phase":phase,"status":"PASS","seconds":time.monotonic()-t})
                progress(control,"PHASE_PASS",phase=phase)
            phase = "historical-protection"
            require(protected() == read(control/"historical-before.json"), "old stopped SEP changed")
        except BaseException as error:
            failed = {"phase":phase,"error":repr(error),"traceback":traceback.format_exc()}
            result.append({"phase":phase,"status":"FAIL","seconds":time.monotonic()-t})
            atomic(control/"FAIL.json", failed)
            progress(control,"PHASE_FAIL",phase=phase)
        final = "FAIL" if failed else "SUCCESS"
        atomic(control/"status.json",{**launch,"status":final,"phase":phase,"completed":result})
        try:
            publish_report(control, out, launch, result, failed)
        except Exception as error:
            atomic(control/"publication.json",{"status":"FAIL","error":repr(error),"traceback":traceback.format_exc()})
            raise
        return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(execute())
