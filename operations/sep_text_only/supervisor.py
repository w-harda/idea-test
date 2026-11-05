"""Smoke 三域全部通过后自动开始 Text-only Full，真实失败保留且不重试。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tbps.artifacts import json_bytes, sha_file
from tbps.baselines.cpgc_sep.dataset import require
from operations.sep_source_only.common import ROOT,atomic,read
from .runtime import CONFIG,DATASETS,VICTIMS,config,event,store,metadata

REPORT=Path("docs/framework/results/sep_text_only_full/REPORT.md")


def commit():
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def observe():
    result={"time":time.time()}
    for field in ("memory.total","memory.free","utilization.gpu"):
        try:
            result[field]=subprocess.check_output(["nvidia-smi","--query-gpu="+field,"--format=csv,noheader,nounits"],text=True,timeout=10).strip()
        except Exception as error:
            result[field]=repr(error)
    result["RAM"]={line.split(":")[0]:line.split(":")[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith(("MemAvailable:","MemTotal:","SwapFree:"))}
    return result


def final_report(c,launch,status,steps):
    lines=["# SEP Text-only 三数据集 × 两 Victim", "",
           "协议 SEP-TBPS-SRefPseudoPID-S1。Source 为冻结 CUHK IRRA，参考库为完整 CUHK Train。"
           "生成阶段只使用 Caption 和 Source 伪 PID，IRRA/RaSa 共用同一份冻结文本。", "",
           f"执行 Commit `{launch['commit']}`。实际状态 `{status}`。",
           f"控制目录 `{c['control']}`。输出目录 `{c['output']}`。", "",
           "继承176项CPU回归和已有真实分数Exact。本轮只执行受影响接线的定向CPU测试。"
           "原Landlock CUDA初始化失败改为Caption-only输入投影及文件审计钩子。"
           "S1保留原100次随机权重/DPP调用，N/E=1/1，权重退化为[1]。"
           "CUHK IRRA也是Source，因此该行是Source可访问评测。", "",
           "原同义词/POS/搜索预算/T=10/patience=5/gamma=.02保持。"
           "每Caption使用不透明键派生的固定Seed，Smoke产物直接复用于Full。"
           "常量/非有限重要性导致官方缩放循环无法结束时显式FAIL，保留原Query，未添加NOOP回退。", "",
           "## 实际阶段", "", "|阶段|状态|秒|", "|---|---|---:|"]
    lines += [f"|{s['phase']}|{s['status']}|{s['seconds']:.2f}|" for s in steps]
    lines += ["", "## Full结果", "",
              "|Dataset|Victim|Attack|R@1|R@5|R@10|mAP|Query|",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    names={"cuhk":"CUHK-PEDES","rstp":"RSTPReid","icfg":"ICFG-PEDES"}
    rows=[]
    for ds in DATASETS:
        for victim in VICTIMS:
            a=store(c,f"full/audit/{ds}/{victim}")
            if (a.root/"completed").exists():
                a.verify("completed")
                audit=read(a.root/"completed/report.json")
                values=audit["metrics"]
                lines.append(f"|{names[ds]}|{victim.upper() if victim=='irra' else 'RaSa'}|SEP Text-only|"
                             +"|".join(f"{values[k]:.2f}" for k in ("R@1","R@5","R@10","mAP"))+f"|{audit['Query']}|")
                rows.append({"Dataset":names[ds],"Victim":victim,"Attack":"SEP Text-only",**values,
                             "Query":audit["Query"],"checkpoint":audit["checkpoint"],
                             "source_seconds":audit["source_total_seconds"],"target_seconds":audit["target_seconds"],
                             "artifact":str(store(c,f"full/target/{ds}/{victim}").root)})
            else:
                lines.append(f"|{names[ds]}|{victim}|SEP Text-only|—|—|—|—|未完成|")
    (ROOT/REPORT).write_text("\n".join(lines)+"\n")
    if rows:
        import csv,io
        buf=io.StringIO()
        columns=("Dataset","Victim","Attack","R@1","R@5","R@10","mAP","Query")
        writer=csv.DictWriter(buf,fieldnames=columns)
        writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in columns})
        (ROOT/REPORT.parent/"METRICS.csv").write_text(buf.getvalue())
        (ROOT/REPORT.parent/"SOURCES.json").write_bytes(json_bytes(rows))
    subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
    subprocess.run(["git","add","--",str(REPORT.parent)],cwd=ROOT,check=True)
    subprocess.run(["git","commit","-m","docs: publish SEP Text-only execution results"],cwd=ROOT,check=True)
    subprocess.run(["git","push","origin","HEAD:refs/heads/codex/sep-text-only-full"],cwd=ROOT,check=True)
    published=commit()
    remote=subprocess.check_output(["git","ls-remote","origin","refs/heads/codex/sep-text-only-full"],cwd=ROOT,text=True).split()[0]
    require(remote==published,"publication SHA mismatch")
    atomic(Path(c["control"])/"publication.json",{"status":"PUBLISHED","commit":published})


def main():
    c,_=config()
    ctl=Path(c["control"])
    require(read(ctl/"preflight.json")["status"]=="PASS","minimal CPU preflight required")
    with (ctl/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (ctl/"supervisor.json").open("x") as f:
            launch={"pid":os.getpid(),"pgid":os.getpgid(0),"sid":os.getsid(0),"commit":commit(),"time":time.time()}
            require(launch["pid"]==launch["pgid"]==launch["sid"] and not os.isatty(0),"detached supervisor")
            json.dump(launch,f)
        plan=[("reference",["reference"],False)]
        for ds in DATASETS:
            plan.append((f"smoke-source-{ds}",["source",ds,"smoke"],False))
            for v in VICTIMS:
                plan.extend([(f"smoke-target-{ds}-{v}",["target",ds,v,"smoke"],False),
                             (f"smoke-audit-{ds}-{v}",["audit",ds,v,"smoke"],True)])
        for ds in DATASETS:
            plan.append((f"full-source-{ds}",["source",ds,"full"],False))
            for v in VICTIMS:
                plan.extend([(f"full-target-{ds}-{v}",["target",ds,v,"full"],False),
                             (f"full-audit-{ds}-{v}",["audit",ds,v,"full"],True)])
        steps=[]
        status="RUNNING"
        try:
            for phase,args,cpu in plan:
                if phase=="full-source-cuhk":
                    smoke=[]
                    for ds in DATASETS:
                        for v in VICTIMS:
                            s=store(c,f"smoke/audit/{ds}/{v}")
                            s.verify("completed")
                            r=read(s.root/"completed/report.json")
                            require(r["status"]=="PASS" and r["Query"]==8,"all Smoke checks required")
                            smoke.append(r)
                    source_estimate=sum(next(r["mean_source_query_seconds"] for r in smoke if r["dataset"]==ds)
                                        *n for ds,n in (("cuhk",6156),("rstp",2000),("icfg",19848)))
                    atomic(ctl/"full_started.json",{"time":time.time(),"smoke_all_passed":True,
                                                  "estimated_source_hours":source_estimate/3600})
                    event(c,"FULL_STARTED",estimated_source_hours=source_estimate/3600)
                atomic(ctl/"status.json",{**launch,"status":"RUNNING","phase":phase,"completed":steps})
                resources=observe()
                with (ctl/"resources.jsonl").open("ab") as f:f.write(json_bytes(resources)+b"\n")
                event(c,"PHASE_START",phase=phase)
                start=time.monotonic()
                env=os.environ.copy()
                if cpu:env["CUDA_VISIBLE_DEVICES"]=""
                with (ctl/(phase+".log")).open("xb") as log:
                    child=subprocess.Popen([sys.executable,"-B","-m","operations.sep_text_only.execute",*args],
                        cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    atomic(ctl/"active_child.json",{"pid":child.pid,"phase":phase})
                    code=child.wait()
                require(code==0,f"{phase} exit={code}; preserve; no retry")
                steps.append({"phase":phase,"status":"PASS","seconds":time.monotonic()-start})
                event(c,"PHASE_PASS",phase=phase,seconds=time.monotonic()-start)
            status="SUCCESS"
        except BaseException as error:
            status="FAIL"
            steps.append({"phase":phase,"status":"FAIL","seconds":time.monotonic()-start})
            atomic(ctl/"FAIL.json",{"phase":phase,"error":repr(error),"traceback":traceback.format_exc()})
        atomic(ctl/"status.json",{**launch,"status":status,"phase":phase,"completed":steps})
        try:final_report(c,launch,status,steps)
        except BaseException as error:
            atomic(ctl/"publication.json",{"status":"FAIL","error":repr(error),"traceback":traceback.format_exc()})
            raise
        return int(status!="SUCCESS")


if __name__=="__main__":
    raise SystemExit(main())
