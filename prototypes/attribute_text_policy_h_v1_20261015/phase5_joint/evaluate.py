"""官方Joint Full；GPU仅用户--user-run-full执行，默认CPU等待Selection。"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import csv
import io
import json
import time
import traceback
from unittest.mock import patch
import torch
from phase2_sources import digest,sha,read_json,checked
from tokenizer_probe import official_probe
from text_frontend import CaptionFrontend
from .common import *
from .protocol import preflight,resolve_selection,references,adversarial_gallery
from .storage import Store,FAMILIES
from .backend import Backend,validate_action,action_statistics

class SafeStop(Exception):pass
def boundary(stop):
    if stop.requested:raise SafeStop("requested stop after committed batch")
def comparisons(groups):
    image,joint=groups["Image-only"]["mAP"],groups["Joint"]["mAP"]
    return {"Joint_minus_ImageOnly_mAP_pp":joint-image,
            "Joint_relative_extra_mAP_drop_percent":(image-joint)/image*100 if image else None,
            "Joint_better_than_ImageOnly":joint<image}

def baseline_metrics(protocol,clean_text,clean_gallery,image_gallery,check_stop=None):
    metrics=legacy("eval_metrics")
    qids=[q["pid"] for q in protocol["queries"]];gids=[g["pid"] for g in protocol["gallery"]]
    result={}
    for key,x in (("Clean",clean_gallery),("Image-only",image_gallery)):
        if check_stop:check_stop()
        result[key]=metrics.metrics(clean_text,x,qids,gids)[0]
        metrics.check_clean(result[key],protocol["expected_baselines"][key])
        print("JOINT_METRIC_COMPLETE",key,flush=True)
        if check_stop:check_stop()
    return result

def cached_results(store,verify_inputs=True,check_stop=None):
    p=store.protocol;store.audit()
    arrays={};proof={};family_stats={}
    for family in FAMILIES:
        if check_stop:check_stop()
        arrays[family],family_stats[family],files=store.gather(family);proof.update(files)
    actions=[]
    probe,frontend=official_probe(),CaptionFrontend()
    for stats in family_stats["adv_text"]:
        actions.extend(stats["actions"])
    if len(actions)!=len(p["queries"]):raise ValueError("all 6156 Query actions must be retained")
    for i,(q,a) in enumerate(zip(p["queries"],actions)):
        validate_action(q["caption"],a,probe,frontend)
        if a["adversarial"]==q["caption"] and not torch.equal(arrays["adv_text"][i],arrays["clean_text"][i]):
            raise ValueError("NOOP feature must equal original feature exactly")
    if verify_inputs:
        clean,text=references(p)
        if not torch.equal(arrays["clean_gallery"],clean) or not torch.equal(arrays["clean_text"],text):
            raise ValueError("Clean/Text reuse changed feature values")
        adv,evidence=adversarial_gallery(p)
        if not torch.equal(arrays["image_gallery"],adv):raise ValueError("G-v2 image feature values changed")
        checked(store.root/"image_reuse.json",read_json(store.root/"baseline_verified.json")["image_reuse_sha256"])
        if read_json(store.root/"image_reuse.json")!=evidence:raise ValueError("historical image evidence mismatch")
    metrics=legacy("eval_metrics");groups=baseline_metrics(p,arrays["clean_text"],arrays["clean_gallery"],arrays["image_gallery"],check_stop)
    qids=[q["pid"] for q in p["queries"]];gids=[g["pid"] for g in p["gallery"]]
    for key,family in (("Text-only","clean_gallery"),("Joint","image_gallery")):
        if check_stop:check_stop()
        groups[key]=metrics.metrics(arrays["adv_text"],arrays[family],qids,gids)[0]
        print("JOINT_METRIC_COMPLETE",key,flush=True)
        if check_stop:check_stop()
    stats=action_statistics(actions)
    encoded=sum(s["IRRA_text_samples"] for s in family_stats["adv_text"])
    if encoded!=stats["edited_queries"] or sum(s["H_forward_samples"] for s in family_stats["adv_text"])!=stats["H_forward_samples"]:
        raise ValueError("committed text/H sample accounting mismatch")
    rates={key:metrics.drops(groups["Clean"],value) for key,value in groups.items()}
    return {"schema":"hv1-joint-full-result-1","protocol_sha256":store.hash,"groups":groups,
            "drop_rate_percent":rates,"DR_mAP_percent":{k:v["mAP"] for k,v in rates.items()},
            "comparison":comparisons(groups),"gallery_count":len(p["gallery"]),"query_count":len(p["queries"]),
            "model_sha256":p["model_sha256"],"H_selection_sha256":p["H_selection"]["sha256"],
            "query_statistics":stats,"feature_files_sha256":proof,
            "counts":{"G_new_samples":0,"IRRA_image_new_samples":0,"IRRA_text_committed_samples":encoded,
                      "G_historical_samples":len(p["gallery"]),"IRRA_image_historical_samples":len(p["gallery"]),
                      "IRRA_adv_image_historical_samples":len(p["gallery"]),
                      "IRRA_clean_image_reference_samples":len(p["gallery"]),
                      "IRRA_original_text_reference_samples":len(p["queries"]),
                      "H_committed_samples":stats["H_forward_samples"]},
            "metric_unit":"percent","DR_definition":"(mAP_Clean-mAP_group)/mAP_Clean*100",
            "all_queries_retained":True,"selection_uses_Test":False,"optimizer_steps":0,"backwards":0,
            "single_victim":"official IRRA; no transfer claim"}

def csv_bytes(summary):
    f=io.StringIO(newline="")
    w=csv.writer(f);w.writerow(["Group","mAP","R@1","R@5","R@10","mINP","DR_mAP_percent"])
    for group in GROUPS:
        w.writerow([group,*[summary["groups"][group][k] for k in legacy("eval_common").METRICS],
                    summary["DR_mAP_percent"][group]])
    return f.getvalue().encode()

def immutable_bytes(path,value):
    import os,uuid
    from real_feature_cache import fsync_dir
    if path.exists():
        if path.read_bytes()!=value:raise ValueError("immutable CSV mismatch")
        return
    work=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with work.open("xb") as f:f.write(value);f.flush();os.fsync(f.fileno())
    try:os.link(work,path)
    finally:work.unlink()
    fsync_dir(path.parent)

def audit(out,protocol,recompute=True):
    store=Store(out,protocol);verify_protocol(protocol)
    done,summary=read_json(Path(out)/"completed.json"),read_json(Path(out)/"summary.json")
    if (done.get("method_audit_passed") is not True or summary["protocol_sha256"]!=store.hash
            or summary["model_sha256"]!=protocol["model_sha256"]
            or done["protocol_sha256"]!=store.hash or done["summary_sha256"]!=sha(Path(out)/"summary.json")
            or done["metrics_csv_sha256"]!=sha(Path(out)/"metrics.csv")):
        raise ValueError("completion/summary/CSV SHA mismatch")
    checked(Path(out)/done["execution_path"],done["execution_sha256"])
    actual_attempts={str(p.relative_to(out)):sha(p) for p in (Path(out)/"attempts").rglob("*") if p.is_file()}
    if actual_attempts!=done["attempt_files_sha256"]:
        raise ValueError("attempt execution/interruption/failure evidence changed")
    store.audit()
    for name,expected in summary["feature_files_sha256"].items():checked(Path(out)/name,expected)
    baseline=read_json(Path(out)/"baseline_verified.json")
    if baseline["protocol_sha256"]!=store.hash or baseline["passed"] is not True:
        raise ValueError("missing Clean/Image-only verification")
    checked(Path(out)/"baseline_verified.json",done["baseline_verified_sha256"])
    if (Path(out)/"metrics.csv").read_bytes()!=csv_bytes(summary):raise ValueError("CSV/summary values mismatch")
    if recompute and cached_results(store)!=summary:raise ValueError("CPU cached metrics/actions differ from summary")
    cpu_only()
    print("JOINT_FULL_CPU_AUDIT PASS",summary["groups"],flush=True)
    return summary

@torch.inference_mode()
def run(protocol,out,backend_factory=Backend,reference_factory=references,image_factory=adversarial_gallery,
        stop=None,recover_pending=False):
    if protocol["execution_backend"]=="official_irra_real":
        safe_output(out)
        if (protocol["gallery_count"],protocol["query_count"])!=(3074,6156):
            raise ValueError("real Full fixed sample sizes")
    elif protocol["execution_backend"]!="cpu_fixture":raise ValueError("unknown backend")
    out=Path(out);stop=stop or type("StopFlag",(),{"requested":False})()
    started=time.monotonic();backend=None
    with run_lock(out):
        verify_protocol(protocol)
        store=Store(out,protocol,True)
        if (out/"completed.json").exists():
            # 已完成重入只查SHA，不构造H/IRRA或重算完整指标。
            summary=audit(out,protocol,recompute=False)
            print("JOINT_FULL_ALREADY_COMPLETE_CPU_ONLY",flush=True);return summary
        store.check_pending(recover_pending)
        attempt=out/"attempts"/str(time.time_ns());attempt.mkdir(parents=True)
        try:
            clean,text=reference_factory(protocol)
            adv,image_evidence=image_factory(protocol,lambda:boundary(stop))
            # GPU加载前先按官方指标核对固定Clean/Image-only；不匹配立即停。
            groups=baseline_metrics(protocol,text,clean,adv,lambda:boundary(stop))
            immutable_json(out/"image_reuse.json",image_evidence)
            immutable_json(out/"baseline_verified.json",{"protocol_sha256":store.hash,"passed":True,"groups":groups,
                                "image_reuse_sha256":sha(out/"image_reuse.json")})
            for family,array in (("clean_gallery",clean),("image_gallery",adv),("clean_text",text)):
                for index,rows in store.parts(family):
                    boundary(stop);start=index*(64 if family.endswith("gallery") else protocol["text_batch"])
                    store.prepare(family,index,rows,lambda array=array,start=start,rows=rows:(
                        array[start:start+len(rows)].clone(),{"G_new_samples":0,"IRRA_image_new_samples":0,
                                                           "IRRA_text_samples":0,"reference_reused":True}))
                print("JOINT_REFERENCE_COMPLETE",family,len(array),flush=True)
            for index,rows in store.parts("adv_text"):
                boundary(stop)
                saved=store.load("adv_text",index,rows)
                if saved is None:
                    def produce():
                        nonlocal backend
                        if backend is None:backend=backend_factory(protocol)
                        start=index*protocol["text_batch"]
                        # H接口只有Caption字符串和原文feature；feature只用于NOOP复用，H不见feature。
                        return backend.text(tuple(q["caption"] for q in rows),text[start:start+len(rows)].clone())
                    store.prepare("adv_text",index,rows,produce)
                print("JOINT_TEXT_COMPLETE",min((index+1)*protocol["text_batch"],len(protocol["queries"])),
                      "/",len(protocol["queries"]),flush=True)
                boundary(stop)
            resources=backend.finish() if backend is not None else {"no_new_model_forward":True}
            backend=None
            boundary(stop)
            summary=cached_results(store,verify_inputs=False,check_stop=lambda:boundary(stop))
            boundary(stop)
            verify_protocol(protocol)
            boundary(stop)
            immutable_json(out/"summary.json",summary);immutable_bytes(out/"metrics.csv",csv_bytes(summary))
            execution={"protocol_sha256":store.hash,"seconds_this_attempt":time.monotonic()-started,
                       "resources":resources,"G_new_samples":0,"IRRA_image_new_samples":0,
                       "no_optimizer_or_backward":True,"successful_attempt_only":True}
            immutable_json(attempt/"execution.json",execution)
            immutable_json(out/"completed.json",{"protocol_sha256":store.hash,"summary_sha256":sha(out/"summary.json"),
                "metrics_csv_sha256":sha(out/"metrics.csv"),"baseline_verified_sha256":sha(out/"baseline_verified.json"),
                "execution_path":str((attempt/"execution.json").relative_to(out)),
                "execution_sha256":sha(attempt/"execution.json"),"method_audit_passed":True,
                "attempt_files_sha256":{str(p.relative_to(out)):sha(p) for p in (out/"attempts").rglob("*") if p.is_file()}})
            print("JOINT_FULL_EVALUATION_COMPLETE",summary["comparison"],flush=True)
            return summary
        except BaseException as exc:
            try:resources=backend.finish() if backend is not None else None
            except BaseException as cleanup:
                resources={"cleanup_error":repr(cleanup)}
            backend=None
            immutable_json(attempt/("interruption.json" if isinstance(exc,SafeStop) else "failure.json"),
                {"error":repr(exc),"traceback":traceback.format_exc(),"resources":resources,
                 "seconds_this_attempt":time.monotonic()-started,"G_new_samples":0,"IRRA_image_new_samples":0})
            raise
        finally:
            if backend is not None:backend.finish()

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action",choices=("preflight","resolve-selection","run","audit"),default="preflight")
    p.add_argument("--selection",type=Path)
    p.add_argument("--output",type=Path,default=OUTPUT)
    p.add_argument("--text-batch",type=int,default=16)
    p.add_argument("--user-run-full",action="store_true")
    p.add_argument("--recover-pending",action="store_true")
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=="run" and not args.user_run_full:
        raise ValueError("GPU Full USER MANUAL ONLY; --user-run-full required")
    if args.recover_pending and args.action!="run":raise ValueError("recovery only for explicit user run")
    safe_output(args.output)
    if args.action=="resolve-selection":
        source_pins();print(resolve_selection());cpu_only();return
    if args.action=="preflight" and args.selection is None:
        source_pins();cpu_only()
        print("PHASE5_CPU_SOURCE_PREFLIGHT PASS; SELECTION_REQUIRED; GPU_BLOCKED");return
    if args.action=="audit":
        saved=read_json(args.output/"protocol.json")
        selected=args.selection or Path(saved["H_selection"]["path"])
        protocol=preflight(selected,args.text_batch)
        if protocol!=saved:raise ValueError("audit source/selection/options differ from saved protocol")
        audit(args.output,protocol);return
    if args.selection is None:raise ValueError("--selection required after Validation finishes")
    protocol=preflight(args.selection,args.text_batch)
    print("JOINT_FULL_CPU_PREFLIGHT PASS 3074 6156",protocol["model_sha256"],flush=True)
    if args.action=="preflight":return
    with Stop() as stop,ExitStack() as guard:
        guard.enter_context(patch.object(torch.Tensor,"backward",side_effect=AssertionError("no backward in Full")))
        guard.enter_context(patch.object(torch.autograd,"grad",side_effect=AssertionError("no scoring/JVP")))
        guard.enter_context(patch.object(torch.optim.Optimizer,"__init__",side_effect=AssertionError("no optimizer")))
        try:run(protocol,args.output,stop=stop,recover_pending=args.recover_pending)
        except SafeStop:print("JOINT_FULL_SAFE_STOP; resume exact command and output",flush=True)

if __name__=="__main__":main()
