"""联合攻击消融：默认CPU预检，正式文本编码及评测仅用户手动执行。"""
import argparse
from pathlib import Path
import csv
import io
import json
import statistics
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch
import torch
from phase2_sources import digest,sha,read_json,checked
from phase5_joint.evaluate import immutable_bytes,csv_bytes
from phase5_joint.backend import action_statistics
from .common import *
from .storage import TrialStore
from .backend import Engine,peak_memory
from .strategies import validate_trial_action,edit_statistics

class SafeStop(Exception):pass
def boundary(stop):
    if stop.requested:raise SafeStop("stop requested at atomic batch/metric boundary")

def trial_result(store,parent,arrays,learned,probe,frontend,check_stop=None):
    store.audit()
    features,stats,files=store.gather("adv_text")
    actions=[a for s in stats for a in s["actions"]]
    if len(actions)!=len(parent["queries"]):raise ValueError("all Query actions must be retained")
    for i,(q,a,ref) in enumerate(zip(parent["queries"],actions,learned)):
        validate_trial_action(q["caption"],a,ref,store.protocol["trial"],probe,frontend,parent["config"]["search_max_nodes"])
        if a["adversarial"]==q["caption"] and not torch.equal(features[i],arrays["clean_text"][i]):
            raise ValueError("NOOP feature changed")
    metrics=legacy("eval_metrics")
    groups={k:parent["fixed_baseline_groups"][k] for k in ("Clean","Image-only")}
    qids=[q["pid"] for q in parent["queries"]];gids=[g["pid"] for g in parent["gallery"]]
    for name,family in (("Text-only","clean_gallery"),("Joint","image_gallery")):
        if check_stop:check_stop()
        groups[name]=metrics.metrics(features,arrays[family],qids,gids)[0]
        print("ABLATION_METRIC_COMPLETE",store.protocol["trial"]["key"],name,flush=True)
        if check_stop:check_stop()
    counts=edit_statistics(actions)
    return {"trial":store.protocol["trial"],"protocol_sha256":store.hash,"groups":groups,
        "DR_mAP_percent":{name:metrics.drops(groups["Clean"],m)["mAP"] for name,m in groups.items()},
        "Joint_minus_ImageOnly_mAP_pp":groups["Joint"]["mAP"]-groups["Image-only"]["mAP"],
        "Joint_minus_LearnedJoint_mAP_pp":groups["Joint"]["mAP"]-parent["fixed_baseline_groups"]["Joint"]["mAP"],
        "edit_statistics":counts,"action_statistics":action_statistics(actions),
        "IRRA_text_committed_samples":sum(s["IRRA_text_samples"] for s in stats),
        "G_new_samples":0,"IRRA_image_new_samples":0,"feature_files_sha256":files,"all_queries_retained":True}

def mean_std(values):
    if not values:raise ValueError("empty seed statistics")
    return {"n":len(values),"mean":statistics.mean(values),
            "std":statistics.stdev(values) if len(values)>1 else None,"ddof":1}

def aggregate(parent,results):
    expected={t["key"] for t in parent["trials"]}
    if {r["trial"]["key"] for r in results}!=expected or len(results)!=len(expected):
        raise ValueError("missing/duplicate configured seed result")
    from phase5_joint.backend import action_statistics as action_stats
    # Learned H metrics already verified in frozen Full; never regenerate or select H.
    families={}
    for kind in ("random_attribute","h_selected","untrained"):
        rows=[r for r in results if r["trial"]["kind"]==kind]
        families[kind]={"seeds":[r["trial"]["seed"] for r in rows],
          "groups":{group:{metric:mean_std([r["groups"][group][metric] for r in rows])
                     for metric in legacy("eval_common").METRICS}
                    for group in ("Clean","Image-only","Text-only","Joint")},
          "NOOP_ratio":mean_std([r["edit_statistics"]["NOOP_ratio"] for r in rows]),
          "exact_count_match_ratio":mean_std([r["edit_statistics"]["exact_count_match_ratio"] for r in rows]),
          "Joint_minus_LearnedJoint_mAP_pp":mean_std([r["Joint_minus_LearnedJoint_mAP_pp"] for r in rows])}
    families["learned_h_v1"]={"seeds":[],"groups":{
        group:{metric:mean_std([parent["fixed_baseline_groups"][group][metric]])
               for metric in legacy("eval_common").METRICS}
        for group in ("Clean","Image-only","Text-only","Joint")},
        "NOOP_ratio":mean_std([parent["learned_edit_statistics"]["NOOP_ratio"]]),
        "exact_count_match_ratio":mean_std([1.]),"Joint_minus_LearnedJoint_mAP_pp":mean_std([0.])}
    return {"schema":"hv1-ablation-result-1","protocol_sha256":digest(parent),
        "Learned_H_v1":{"groups":parent["fixed_baseline_groups"],"source":"frozen audited Phase5 Full",
                        "new_H_forwards":0,"new_IRRA_text_forwards":0,"edit_statistics":parent["learned_edit_statistics"]},
        "trials":results,"aggregate":families,"model_sha256":parent["model_sha256"],
        "untrained_initial_state_sha256":parent["untrained_initial_state_sha256"],
        "queries":len(parent["queries"]),"gallery":len(parent["gallery"]),
        "G_new_samples":0,"IRRA_image_new_samples":0,"Test_model_selection":False,
        "statistical_limit":"5 fixed seeds, sample std ddof1; n1 std null; not a significance claim"}

def detailed_csv(summary):
    out=io.StringIO(newline="");w=csv.writer(out);metrics=legacy("eval_common").METRICS
    w.writerow(["variant","seed","group",*metrics,"DR_mAP_percent","NOOP_ratio","count_match_ratio"])
    baseline=summary["Learned_H_v1"]["groups"]
    drops=legacy("eval_metrics").drops
    for group in ("Clean","Image-only","Text-only","Joint"):
        w.writerow(["Learned_H_v1","",group,*[baseline[group][k] for k in metrics],
                    drops(baseline["Clean"],baseline[group])["mAP"],summary["Learned_H_v1"]["edit_statistics"]["NOOP_ratio"],1.])
    for r in summary["trials"]:
        for group in ("Clean","Image-only","Text-only","Joint"):
            w.writerow([r["trial"]["kind"],r["trial"]["seed"],group,*[r["groups"][group][k] for k in metrics],
                r["DR_mAP_percent"][group],r["edit_statistics"]["NOOP_ratio"],r["edit_statistics"]["exact_count_match_ratio"]])
    return out.getvalue().encode()
def aggregate_csv(summary):
    out=io.StringIO(newline="");w=csv.writer(out);metrics=legacy("eval_common").METRICS
    w.writerow(["variant","group","n",*[v for k in metrics for v in (k+"_mean",k+"_std")],
                "NOOP_mean","NOOP_std","match_ratio_mean","match_ratio_std"])
    for kind in ("learned_h_v1","random_attribute","h_selected","untrained"):
        r=summary["aggregate"][kind]
        for group in ("Clean","Image-only","Text-only","Joint"):
            m=r["groups"][group]
            w.writerow([kind,group,next(iter(m.values()))["n"],*[v for k in metrics for v in (m[k]["mean"],m[k]["std"])],
                        r["NOOP_ratio"]["mean"],r["NOOP_ratio"]["std"],
                        r["exact_count_match_ratio"]["mean"],r["exact_count_match_ratio"]["std"]])
    return out.getvalue().encode()

def read_trial(root,parent,trial,recompute=False,arrays=None,learned=None,engine=None):
    root=Path(root);p=trial_protocol(parent,trial);store=TrialStore(root,p)
    done,result=read_json(root/"completed.json"),read_json(root/"summary.json")
    if (done["protocol_sha256"]!=store.hash or done["summary_sha256"]!=sha(root/"summary.json")
            or result["protocol_sha256"]!=store.hash or result["trial"]!=trial):
        raise ValueError("trial completion/source mismatch")
    checked(root/"execution.json",done["execution_sha256"])
    if read_json(root/"execution.json")["result_digest"]!=digest(result):
        raise ValueError("trial execution/result digest mismatch")
    store.audit()
    for name,h in result["feature_files_sha256"].items():checked(root/name,h)
    if recompute:
        actual=trial_result(store,parent,arrays,learned,engine.probe,engine.frontend)
        if actual!=result:raise ValueError("cached official metrics/action audit mismatch")
    return result

def evidence_files(out):
    out=Path(out);result={}
    for directory in ("attempts","abandoned"):
        for p in (out/directory).rglob("*"):
            if p.is_file():result[str(p.relative_to(out))]=sha(p)
    # 每trial未提交目录恢复归档也永久绑定。
    for p in (out/"variants").glob("*/abandoned/**/*"):
        if p.is_file():result[str(p.relative_to(out))]=sha(p)
    return result

def audit(out,parent,arrays,learned,recompute=True):
    out=Path(out);verify(parent)
    if read_json(out/"protocol.json")!=parent:raise ValueError("parent source/config/seed changed")
    done,summary=read_json(out/"completed.json"),read_json(out/"summary.json")
    if (done["protocol_sha256"]!=digest(parent) or done["summary_sha256"]!=sha(out/"summary.json")
            or done["metrics_csv_sha256"]!=sha(out/"metrics.csv") or done["aggregate_csv_sha256"]!=sha(out/"aggregate.csv")
            or done["evidence_files_sha256"]!=evidence_files(out)):
        raise ValueError("ablation completion/CSV/attempt evidence mismatch")
    engine=Engine(parent)
    results=[]
    for trial in parent["trials"]:
        root=out/"variants"/trial["key"]
        results.append(read_trial(root,parent,trial,recompute,arrays,learned,engine))
    actual=aggregate(parent,results)
    if actual!=summary or (out/"metrics.csv").read_bytes()!=detailed_csv(summary) or (out/"aggregate.csv").read_bytes()!=aggregate_csv(summary):
        raise ValueError("seed statistics/CSV summary mismatch")
    cpu_only()
    print("ABLATION_CPU_AUDIT PASS",flush=True)
    return summary

@torch.inference_mode()
def run(parent,out,arrays,learned,engine_factory=Engine,stop=None,recover_pending=False):
    if parent["execution_backend"]=="official_irra_real":
        safe_output(out)
        if (len(parent["gallery"]),len(parent["queries"]))!=(3074,6156):raise ValueError("official fixed counts")
    elif parent["execution_backend"]!="cpu_fixture":raise ValueError("unknown backend")
    out=Path(out);stop=stop or type("StopFlag",(),{"requested":False})()
    with run_lock(out):
        verify(parent);immutable_json(out/"protocol.json",parent)
        if (out/"completed.json").exists():
            result=audit(out,parent,arrays,learned,False)
            print("ABLATION_ALREADY_COMPLETE_CPU_ONLY",flush=True);return result
        attempt=out/"attempts"/str(time.time_ns());attempt.mkdir(parents=True)
        engine=engine_factory(parent);started=time.monotonic()
        try:
            results=[]
            for trial in parent["trials"]:
                boundary(stop);root=out/"variants"/trial["key"]
                if (root/"completed.json").exists():
                    results.append(read_trial(root,parent,trial));continue
                p=trial_protocol(parent,trial);store=TrialStore(root,p,True)
                store.check_pending(recover_pending);store.audit(False)
                strategy=engine.strategy(trial)
                initial=engine.samples();t0=time.monotonic()
                for index,rows in store.parts("adv_text"):
                    boundary(stop);start=index*parent["text_batch"]
                    store.prepare("adv_text",index,rows,lambda rows=rows,start=start:
                        engine.batch(strategy,tuple(q["caption"] for q in rows),learned[start:start+len(rows)],
                                     arrays["clean_text"][start:start+len(rows)].clone()))
                    print("ABLATION_TEXT_COMPLETE",trial["key"],min(start+len(rows),len(learned)),"/",len(learned),flush=True)
                    boundary(stop)
                state=strategy.finish()
                if trial["kind"]=="untrained" and state["initial_H_state_sha256"]!=parent["untrained_initial_state_sha256"]:
                    raise ValueError("untrained seed42 initial weights changed")
                r=trial_result(store,parent,arrays,learned,engine.probe,engine.frontend,lambda:boundary(stop))
                verify(parent);boundary(stop)
                immutable_json(root/"summary.json",r)
                execution_path=root/"execution.json"
                if execution_path.exists():
                    # 完整特征/summary发布后、completed发布前中断：保留原执行证据，
                    # 仅在其绑定相同结果和冻结策略时补齐完成回执，不重写时间/计数。
                    prior=read_json(execution_path)
                    if (prior["result_digest"]!=digest(r) or prior["strategy"]!=state
                            or prior["G_new_samples"]!=0 or prior["IRRA_image_new_samples"]!=0):
                        raise ValueError("uncommitted trial execution/result mismatch")
                else:
                    immutable_json(execution_path,{"result_digest":digest(r),"strategy":state,
                        "seconds_this_trial_attempt":time.monotonic()-t0,
                        "IRRA_text_new_samples_this_trial_attempt":engine.samples()-initial,
                        "G_new_samples":0,"IRRA_image_new_samples":0,**peak_memory()})
                immutable_json(root/"completed.json",{"protocol_sha256":store.hash,"summary_sha256":sha(root/"summary.json"),
                                                      "execution_sha256":sha(root/"execution.json")})
                results.append(r);print("ABLATION_TRIAL_COMPLETE",trial["key"],flush=True)
            resources=engine.finish()
            summary=aggregate(parent,results)
            verify(parent);boundary(stop)
            immutable_json(out/"summary.json",summary)
            immutable_bytes(out/"metrics.csv",detailed_csv(summary));immutable_bytes(out/"aggregate.csv",aggregate_csv(summary))
            immutable_json(attempt/"execution.json",{"resources":resources,"seconds_this_attempt":time.monotonic()-started})
            immutable_json(out/"completed.json",{"protocol_sha256":digest(parent),"summary_sha256":sha(out/"summary.json"),
              "metrics_csv_sha256":sha(out/"metrics.csv"),"aggregate_csv_sha256":sha(out/"aggregate.csv"),
              "evidence_files_sha256":evidence_files(out),"method_audit_passed":True})
            print("ABLATION_FULL_COMPLETE",flush=True);return summary
        except BaseException as exc:
            try:resources=engine.finish()
            except BaseException as other:resources={"cleanup_error":repr(other)}
            immutable_json(attempt/("interruption.json" if isinstance(exc,SafeStop) else "failure.json"),
                {"error":repr(exc),"traceback":traceback.format_exc(),"resources":resources})
            raise

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action",choices=("preflight","run","audit"),default="preflight")
    p.add_argument("--output",type=Path,default=OUTPUT);p.add_argument("--text-batch",type=int,default=16)
    p.add_argument("--user-run-ablations",action="store_true");p.add_argument("--recover-pending",action="store_true")
    return p
def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=="run" and not args.user_run_ablations:raise ValueError("real ablations USER MANUAL ONLY")
    if args.recover_pending and args.action!="run":raise ValueError("only explicit user run may recover pending")
    out=safe_output(args.output);parent=preflight(args.text_batch)
    print("ABLATION_CPU_PREFLIGHT PASS",parent["trials"],flush=True)
    if args.action=="preflight":return
    arrays,learned=load_inputs(parent)
    if args.action=="audit":audit(out,parent,arrays,learned);return
    with Stop() as stop,ExitStack() as guard:
        guard.enter_context(patch.object(torch.Tensor,"backward",side_effect=AssertionError("no backward")))
        guard.enter_context(patch.object(torch.autograd,"grad",side_effect=AssertionError("no scoring/JVP")))
        guard.enter_context(patch.object(torch.optim.Optimizer,"__init__",side_effect=AssertionError("no optimizer")))
        try:run(parent,out,arrays,learned,stop=stop,recover_pending=args.recover_pending)
        except SafeStop:print("ABLATION_SAFE_STOP; resume exact frozen command",flush=True)

if __name__=="__main__":main()
