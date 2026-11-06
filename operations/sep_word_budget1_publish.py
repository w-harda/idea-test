"""只读逐组发布SEP Full回执，独立于科学执行进程；无Torch或GPU依赖。"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import csv
import fcntl
import io
import json
import math
import os
import subprocess
import time
import traceback

from tbps.artifacts import ArtifactError,FileRef,checked,sha_file,json_bytes,read_json
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from operations.sep_source_only.common import atomic

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/framework/sep_word_budget1_reports.server.json"
SIZES={"cuhk":(3074,6156),"rstp":(1000,2000),"icfg":(19848,19848)}
GROUPS=tuple((d,v) for d in SIZES for v in ("irra","rasa"))
NAMES={"cuhk":"CUHK-PEDES","rstp":"RSTPReid","icfg":"ICFG-PEDES"}
METRICS=("R@1","R@5","R@10","mAP")


def need(ok,message):
    if not ok:raise ArtifactError(message)


def small_file(root,name):
    need(root.is_dir() and not root.is_symlink(),"committed artifact directory required")
    receipt=read_json(root/"completed.json","tbps-artifact-receipt-1")
    checked(FileRef(str(root/"manifest.json"),receipt["manifest_sha256"]))
    manifest=read_json(root/"manifest.json","tbps-artifact-1")
    need(manifest["artifact_id"]==root.name and name in manifest["files_sha256"],"artifact identity/payload")
    path=root/name
    need(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(),"payload path")
    checked(FileRef(str(path),manifest["files_sha256"][name]))
    return path,manifest


def data_file(root,name):
    path,manifest=small_file(root,name)
    return json.loads(path.read_text()),manifest


def config(path=CONFIG):
    c=json.loads(Path(path).read_text())
    checked(FileRef(**c["experiment_config"]))
    experiment=json.loads(Path(c["experiment_config"]["path"]).read_text())
    need(experiment["protocol"]=="SEP-TBPS-TextOnly-WordBudget1" and experiment["word_budget"]==1
         and experiment["output"]==c["output"],"locked WordBudget1 experiment")
    base_path=Path(c["experiment_worktree"])/experiment["base_config"]
    checked(FileRef(str(base_path),experiment["base_config_sha256"]))
    base=json.loads(base_path.read_text())
    c["source_checkpoint"]=base["victims"]["irra"]["datasets"]["cuhk"]["checkpoint"]
    c["victim_checkpoints"]={f"{ds}_{v}":base["victims"][v]["datasets"][ds]["checkpoint"]
                             for ds,v in GROUPS}
    c["config_sha256"]=c["experiment_config"]["sha256"]
    return c


def read_group(c,ds,victim):
    """只接受正式Full+独立CPU审计；不打开scores.pt或任何模型。"""
    output=Path(c["output"])
    group=f"{ds}_{victim}"
    audit_root=output/"full/audit"/ds/victim
    completed=audit_root/"completed"
    if not (completed/"completed.json").is_file():
        return None
    VerifiedArtifactStore(audit_root,new_base=output).verify("completed")
    audit=json.loads((completed/"report.json").read_text())
    ng,nq=SIZES[ds]
    need(audit["status"]=="PASS" and audit["scope"]=="full" and audit["dataset"]==ds
         and audit["victim"]==victim and audit["CUDA"] is False,"formal independent CPU audit")
    need((audit["Gallery"],audit["Query"])==(ng,nq),"complete official Full counts")
    need(audit["word_budget"]==1 and audit["budget_violations"]==0,"WordBudget1 audit")
    noop,edited=audit["noop_queries"],audit["one_word_queries"]
    need(type(noop) is int and type(edited) is int and 0<=noop<=nq and edited+noop==nq,
         "edit/NOOP coverage")
    need(all(type(n) is int and n>=0 for n in audit["noop_reasons"].values())
         and sum(audit["noop_reasons"].values())==noop,"NOOP reasons count")
    need(audit["checkpoint"]==c["victim_checkpoints"][group]
         and audit["source_checkpoint"]==c["source_checkpoint"],"Source/Victim checkpoint binding")

    target=output/"full/target"/ds/victim/"completed"
    need(Path(audit["result"]["path"])==target/"manifest.json","audit result path")
    checked(FileRef(**audit["result"]))
    summary,target_manifest=data_file(target,"summary.json")
    need(summary["status"]=="PASS" and summary["scope"]=="full" and summary["dataset"]==ds
         and summary["victim"]==victim and summary["mode"]=="Text-only","formal Text-only summary")
    need((summary["Gallery"],summary["Query"])==(ng,nq)
         and summary["checkpoint"]==audit["checkpoint"],"summary identity")
    for values in (summary["metrics"],audit["metrics"]):
        need(set(values)==set(METRICS) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100
                                              for v in values.values()),"finite percent metrics")
    need(summary["metrics"]==audit["metrics"],"summary/audit metrics")
    ids=summary["query_ids"]
    need(len(ids)==len(set(ids))==nq and len(summary["frozen_text_sha256"])==nq,"ordered Query coverage")
    for value in summary["frozen_text_sha256"]:
        need(isinstance(value,str) and len(value)==64 and all(x in "0123456789abcdef" for x in value),"text SHA")
    if victim=="rasa":
        need(summary["native_TopK"]==128 and summary["native_ITM_pairs"]==128*nq,"native RaSa Top128")
    path,_=small_file(target,"metrics.csv")
    rows=list(csv.DictReader(path.open()))
    need(len(rows)==1 and rows[0]["Dataset"]==ds and rows[0]["Victim"]==victim
         and rows[0]["Attack"]=="SEP Text-only","one Text-only CSV row")
    need(all(float(rows[0][k])==audit["metrics"][k] for k in METRICS),"CSV metrics")
    need("scores.pt" in target_manifest["files_sha256"] and (target/"scores.pt").is_file(),"Full score artifact")

    evaluation=output/("inputs/evaluation-"+ds)
    official,_=data_file(evaluation,"manifest-data.json")
    need(ids==[q["query_id"] for q in official["queries"]] and len(official["gallery"])==ng,
         "exact ordered official Query identity")
    source=output/"source"/ds/"completed-full"
    source_report,_=data_file(source,"report.json")
    index,_=data_file(source,"index.json")
    need(source_report["status"]=="PASS" and source_report["scope"]=="full" and source_report["dataset"]==ds
         and source_report["queries"]==nq and source_report["checkpoint"]==c["source_checkpoint"],
         "complete frozen Source texts")
    expected=[__import__("hashlib").sha256((ds+"\0"+qid).encode()).hexdigest() for qid in ids]
    need([r["key"] for r in index]==expected,"complete Source index")
    source_sha=sha_file(source/"manifest.json")
    need(any(r["path"]==str(source/"manifest.json") and r["sha256"]==source_sha
             for r in target_manifest["sources"]),"Target binds completed Full Source")
    for field in ("source_total_seconds","target_seconds","mean_source_query_seconds"):
        need(type(audit[field]) in (int,float) and math.isfinite(audit[field]) and audit[field]>=0,
             "finite real runtime")
    need(audit["target_seconds"]==summary["elapsed_seconds"]
         and math.isclose(audit["mean_source_query_seconds"]*nq,audit["source_total_seconds"],rel_tol=1e-12),
         "runtime consistency")
    return {"group":group,"dataset":ds,"victim":victim,"Query":nq,"Gallery":ng,
            "metrics":audit["metrics"],"noop":noop,"one_word":edited,"noop_reasons":audit["noop_reasons"],
            "budget_violations":0,"source_checkpoint":audit["source_checkpoint"],"checkpoint":audit["checkpoint"],
            "source_seconds":audit["source_total_seconds"],"target_seconds":audit["target_seconds"],
            "mean_query_seconds":audit["mean_source_query_seconds"],"source_forwards":audit["source_forward_count"],
            "audit_path":str(completed),"audit_manifest_sha256":sha_file(completed/"manifest.json"),
            "target_path":str(target.parent),"target_manifest_sha256":sha_file(target/"manifest.json"),
            "scores_sha256":target_manifest["files_sha256"]["scores.pt"],
            "source_path":str(source.parent),"source_manifest_sha256":source_sha,
            "config":c["experiment_config"],"input_sha256":sha_file(evaluation/"manifest.json"),
            "seed":"uint32(first8hex(SHA256(dataset + NUL + query_id)))"}


def render_group(row):
    victim="IRRA" if row["victim"]=="irra" else "RaSa"
    lines=[f"# {NAMES[row['dataset']]} × {victim} · SEP Text-only", "",
           "正式Full及独立CPU审计PASS。协议为SEP-TBPS-TextOnly-WordBudget1。", "",
           "| Dataset | Victim | Attack | Budget | Query | Gallery | R@1 | R@5 | R@10 | mAP |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
           f"| {NAMES[row['dataset']]} | {victim} | SEP Text-only | 1 Word | {row['Query']} | {row['Gallery']} | "
           +" | ".join(f"{row['metrics'][k]:.2f}" for k in METRICS)+" |", "",
           "指标单位为%。完整精度值为 "+json.dumps(row["metrics"],ensure_ascii=False)+"。", "",
           "## 覆盖、编辑和NOOP", "",
           f"- 完整Query覆盖 {row['Query']}/{row['Query']}。",
           f"- 实际替换1词 {row['one_word']}，占 {row['one_word']/row['Query']:.4%}。",
           f"- NOOP {row['noop']}，占 {row['noop']/row['Query']:.4%}。",
           f"- Word Budget违规 {row['budget_violations']}。",
           "- NOOP原因 "+json.dumps(row["noop_reasons"],ensure_ascii=False)+"。",
           "", "## 来源与原生评分", "",
           f"- Source Checkpoint `{row['source_checkpoint']['path']}`，SHA256 `{row['source_checkpoint']['sha256']}`。",
           f"- Victim Checkpoint `{row['checkpoint']['path']}`，SHA256 `{row['checkpoint']['sha256']}`。",
           f"- 攻击配置 `{row['config']['path']}`，SHA256 `{row['config']['sha256']}`。",
           f"- Seed规则 `{row['seed']}`。",
           "- 冻结CUHK IRRA与完整CUHK Train参考图库，固定Source伪PID。生成不读取Target Test PID或评测反馈。"
           "单代理N/E=1/1、DPP权重[1]。CUHK IRRA也作为Source，该行是Source可访问评测。",
           "- Gallery为原始图像。IRRA原生完整Gallery检索；RaSa完整Gallery全局Top128及ITM融合，fusion_batch16。",
           "", "## 实际耗时", "",
           f"- Source文本生成累计 {row['source_seconds']:.3f} 秒，平均每Query {row['mean_query_seconds']:.6f} 秒。",
           f"- Source文本前向 {row['source_forwards']} 次。",
           f"- 本组Target加载、特征编码与评分累计 {row['target_seconds']:.3f} 秒。",
           "- 同一数据集两Victim共用Source文本，Source耗时不应在两行之间重复累计。",
           "", "## 独立审计及产物", "",
           f"- CPU审计 PASS，`{row['audit_path']}`。",
           f"- 审计Manifest SHA256 `{row['audit_manifest_sha256']}`。",
           f"- Target产物及执行Manifest `{row['target_path']}`。",
           f"- Target完成Manifest SHA256 `{row['target_manifest_sha256']}`。",
           f"- 正式Full评分矩阵SHA256 `{row['scores_sha256']}`。",
           f"- 冻结文本及逐词编辑记录 `{row['source_path']}`。",
           f"- Source完成Manifest SHA256 `{row['source_manifest_sha256']}`。",
           "- 发布器读取已提交的CPU回执和小型指标元数据，核验SHA、完整覆盖与字段一致性。"
           "完整评分矩阵及原始候选的数值核对沿用已完成的独立CPU审计。"]
    return "\n".join(lines)+"\n"


def render_summary(records):
    need(set(records)=={f"{ds}_{v}" for ds,v in GROUPS},"all six audited Full reports required")
    lines=["# SEP WordBudget1 六组正式Full汇总", "",
           "六组正式Full与独立CPU审计全部PASS。单位为%。", "",
           "| Dataset | Victim | Attack | Budget | R@1 | R@5 | R@10 | mAP | Query | NOOP | 违规 |",
           "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for ds,v in GROUPS:
        r=records[f"{ds}_{v}"]["data"]
        name="IRRA" if v=="irra" else "RaSa"
        lines.append(f"| {NAMES[ds]} | {name} | SEP Text-only | 1 Word | "
                     +" | ".join(f"{r['metrics'][k]:.2f}" for k in METRICS)
                     +f" | {r['Query']} | {r['noop']} | 0 |")
    lines += ["", "## 独立报告", ""]
    lines += [f"- [{NAMES[ds]} × {'IRRA' if v=='irra' else 'RaSa'}]({ds}_{v}/REPORT.md)。" for ds,v in GROUPS]
    return "\n".join(lines)+"\n"


def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=120).strip()


def log(c,event,**fields):
    record={"time":time.time(),"event":event,**fields}
    with (Path(c["control"])/"events.jsonl").open("ab") as f:f.write(json_bytes(record)+b"\n")
    print(json.dumps(record,ensure_ascii=False),flush=True)


def commit_report(c,relative,content):
    path=ROOT/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        need(path.read_text()==content,"existing report differs; preserve for review")
    else:
        atomic_text=path.with_suffix(".md.pending")
        with atomic_text.open("x") as f:
            f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(atomic_text,path)
    git("add","--",str(relative))
    staged=git("diff","--cached","--name-only").splitlines()
    need(staged in ([],[str(relative)]),"only this report may be committed")
    if staged:
        git("diff","--cached","--check")
        git("commit","-m","docs: publish audited SEP Full "+str(relative.parent.name))
    return git("rev-parse","HEAD")


def push(c,state,force=False):
    pending=[r for r in state["groups"].values() if r["publication"]=="COMMITTED_LOCAL"]
    summary_pending=state.get("summary",{}).get("publication")=="COMMITTED_LOCAL"
    if not pending and not summary_pending:return
    if not force and time.time()-state.get("last_push_attempt",0)<c["push_retry_seconds"]:return
    state["last_push_attempt"]=time.time()
    try:
        git("push","origin","HEAD:refs/heads/"+c["branch"])
        head=git("rev-parse","HEAD")
        need(git("ls-remote","origin","refs/heads/"+c["branch"]).split()[0]==head,"remote publication SHA")
        records=pending+([state["summary"]] if summary_pending else [])
        for record in records:
            record["publication"]="PUBLISHED"
            record["url"]=c["repo_url"]+"/blob/"+record["commit"]+"/"+record["path"]
            log(c,"REPORT_PUBLISHED",url=record["url"],commit=record["commit"])
        state.pop("push_error",None)
    except Exception as error:
        state["push_error"]=repr(error)
        log(c,"PUSH_FAILED",error=repr(error),traceback=traceback.format_exc())


def cycle(c,state):
    for ds,v in GROUPS:
        group=f"{ds}_{v}"
        if group in state["groups"] or group in state["validation_errors"]:continue
        try:
            data=read_group(c,ds,v)
            if data is None:continue
            relative=Path(c["report_path"])/group/"REPORT.md"
            report_commit=commit_report(c,relative,render_group(data))
            state["groups"][group]={"data":data,"path":str(relative),"commit":report_commit,"publication":"COMMITTED_LOCAL"}
            log(c,"REPORT_COMMITTED",group=group,commit=report_commit)
            push(c,state,force=True)
        except Exception as error:
            state["validation_errors"][group]=repr(error)
            log(c,"GROUP_REPORT_FAILED",group=group,error=repr(error),traceback=traceback.format_exc())
    if len(state["groups"])==6 and "summary" not in state:
        relative=Path(c["report_path"])/"SUMMARY.md"
        summary_commit=commit_report(c,relative,render_summary(state["groups"]))
        state["summary"]={"path":str(relative),"commit":summary_commit,"publication":"COMMITTED_LOCAL"}
        push(c,state,force=True)
    push(c,state)
    state["status"]=("COMPLETE" if state.get("summary",{}).get("publication")=="PUBLISHED" else
                     "VALIDATION_ERROR" if state["validation_errors"] else
                     "PUSH_FAILED" if state.get("push_error") else "MONITORING")
    state["updated_at"]=time.time()
    atomic(Path(c["control"])/"status.json",state)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--once",action="store_true")
    args=parser.parse_args()
    c=config()
    control=Path(c["control"]);control.mkdir(parents=True,exist_ok=True)
    need(git("branch","--show-current")==c["branch"],"isolated report branch")
    with (control/"publisher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (control/"started.json").open("x") as f:
            launch={"pid":os.getpid(),"ppid":os.getppid(),"pgid":os.getpgid(0),"sid":os.getsid(0),
                    "commit":git("rev-parse","HEAD"),"config_sha256":sha_file(CONFIG),"time":time.time()}
            json.dump(launch,f)
        state={"launch":launch,"groups":{},"validation_errors":{}}
        while True:
            try:cycle(c,state)
            except Exception as error:
                log(c,"PUBLISHER_CYCLE_FAILED",error=repr(error),traceback=traceback.format_exc())
                state["status"]="PUBLISHER_ERROR";state["updated_at"]=time.time()
                atomic(control/"status.json",state)
            if args.once or state.get("status")=="COMPLETE":break
            time.sleep(c["poll_seconds"])


if __name__=="__main__":
    main()
