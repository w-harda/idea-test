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

REPORT=Path("docs/framework/results/sep_word_budget1/REPORT.md")


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
    from collections import Counter
    import csv,io
    lines=["# SEP-TBPS-TextOnly-WordBudget1", "",
           f"执行 Commit `{launch['commit']}`，当前状态 `{status}`。",
           f"控制目录 `{c['control']}`。输出目录 `{c['output']}`。",
           "", "只执行Text-only。Source为CUHK IRRA与完整Train参考图库，固定Source伪PID。"
           "S1原100随机权重/DPP调用保留，N/E=1/1，最终权重[1]。CUHK IRRA为Source可访问模型。",
           "", "原官方AST仅适配max_change_num=1、att_succ_num=1，禁用整词截断；"
           "重要性探针由删词改为基于原文的单位置[UNK]替换。其他原始单词、空白及标点位置保持，"
           "实际输出仅替换原始一个词跨度。单一可选位置直接保留softmax概率1，"
           "多位置常量/非有限重要性仍显式报错停止。T=10、patience=5、候选top50、tau gamma=.02保持。",
           "", "生成过程和每次Source评分前检查全部候选、搜索状态和最终文本，超预算立即失败。"
           "无合法同义词时记录NO_LEGAL_SYNONYM并原文NOOP。"
           "Source-only伪标签及单代理属于既有检索适配，本版属于固定单词预算SEP-inspired变体。",
           "", "## 实际阶段", "", "|阶段|状态|秒|", "|---|---|---:|"]
    lines += [f"|{s['phase']}|{s['status']}|{s['seconds']:.2f}|" for s in steps]
    lines += ["", "## 正式Full结果", "",
              "|Dataset|Victim|Attack|Budget|R@1|R@5|R@10|mAP|",
              "|---|---|---|---|---:|---:|---:|---:|"]
    names={"cuhk":"CUHK-PEDES","rstp":"RSTPReid","icfg":"ICFG-PEDES"}
    rows=[]
    for ds in DATASETS:
        for victim in VICTIMS:
            a=store(c,f"full/audit/{ds}/{victim}")
            if (a.root/"completed").exists():
                a.verify("completed")
                audit=read(a.root/"completed/report.json")
                require(audit["budget_violations"]==0 and audit["word_budget"]==1,"WordBudget1 Full validity")
                values=audit["metrics"]
                lines.append(f"|{names[ds]}|{victim.upper() if victim=='irra' else 'RaSa'}|SEP Text-only|1 Word|"
                             +"|".join(f"{values[k]:.2f}" for k in ("R@1","R@5","R@10","mAP"))+"|")
                rows.append({"Dataset":names[ds],"Victim":victim,"Attack":"SEP Text-only","Budget":"1 Word",**values,
                             "Query":audit["Query"],"checkpoint":audit["checkpoint"],
                             "SourceCheckpoint":audit["source_checkpoint"],
                             "SourceForwards":audit["source_forward_count"],
                             "WordBudgetViolations":audit["budget_violations"],
                             "OneWordQueries":audit["one_word_queries"],"NOOP":audit["noop_queries"],
                             "NOOPReasons":audit["noop_reasons"],
                             "source_seconds":audit["source_total_seconds"],"target_seconds":audit["target_seconds"],
                             "attack_config_sha256":sha_file(CONFIG),"seed":"uint32(first8hex(SHA256(dataset+NUL+query_id)))",
                             "artifact":str(store(c,f"full/target/{ds}/{victim}").root),
                             "source_artifact":str(store(c,"source/"+ds).root),
                             "result_manifest_sha256":audit["result"]["sha256"]})
            else:
                lines.append(f"|{names[ds]}|{victim}|SEP Text-only|1 Word|—|—|—|—|")
    if rows:
        lines += ["", "## 覆盖与预算", "",
                  "|Dataset|Victim|Query|1词修改|NOOP|违规|Source前向|Source秒|Target秒|",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"|{r['Dataset']}|{r['Victim']}|{r['Query']}|{r['OneWordQueries']}|{r['NOOP']}|"
                         f"{r['WordBudgetViolations']}|{r['SourceForwards']}|{r['source_seconds']:.2f}|{r['target_seconds']:.2f}|")
        lines += ["", "逐组Checkpoint、NOOP原因、配置SHA、Seed及产物SHA见SOURCES.json。"]
        buf=io.StringIO()
        columns=("Dataset","Victim","Attack","Budget","R@1","R@5","R@10","mAP")
        writer=csv.DictWriter(buf,fieldnames=columns)
        writer.writeheader()
        for r in rows:writer.writerow({k:r[k] for k in columns})
        (ROOT/REPORT.parent/"METRICS.csv").write_text(buf.getvalue())
        (ROOT/REPORT.parent/"SOURCES.json").write_bytes(json_bytes(rows))
    (ROOT/REPORT).write_text("\n".join(lines)+"\n")
    subprocess.run(["git","diff","--check"],cwd=ROOT,check=True)
    subprocess.run(["git","add","--",str(REPORT.parent)],cwd=ROOT,check=True)
    subprocess.run(["git","commit","-m","docs: publish SEP WordBudget1 execution results"],cwd=ROOT,check=True)
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip()
    subprocess.run(["git","push","origin","HEAD:refs/heads/"+branch],cwd=ROOT,check=True)
    published=commit()
    remote=subprocess.check_output(["git","ls-remote","origin","refs/heads/"+branch],cwd=ROOT,text=True).split()[0]
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
                            require(r["status"]=="PASS" and r["Query"]==8 and r["word_budget"]==1 and r["budget_violations"]==0,"all WordBudget1 Smoke checks required")
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
