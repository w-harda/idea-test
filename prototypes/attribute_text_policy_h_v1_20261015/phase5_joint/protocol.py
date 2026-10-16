"""执行时才解析Validation选优；Test身份与策略输入严格隔离。"""
from contextlib import redirect_stdout
from pathlib import Path
import io
import torch
from phase2_sources import digest,sha,checked,read_json,RAW,IMAGES,RAW_SHA,EPOCH10_SHA,IRRA_SHA
from training_inputs import load_training_catalog
from formal_training import cli as formal_cli, checkpoint as ck, training as tr, validation as val
from formal_training.data import read_qualification,preflight_models,PREPROCESS
from .common import *

def validate_selection_record(record,results,epoch_shas):
    if record.get("uses_official_Test") is not False:
        raise ValueError("Test-driven H selection forbidden")
    candidates=record.get("candidates",[])
    shas=[x["checkpoint_sha256"] for x in candidates]
    if len(set(shas))!=len(shas) or set(shas)!=set(epoch_shas) or not shas:
        raise ValueError("selection must cover all completed training Epoch validations")
    actual=[results[h] for h in shas]
    if val.select_best(actual)!=record:
        raise ValueError("immutable Validation selection/metrics mismatch")
    return record["selected_checkpoint_sha256"]

def phase4_state():
    catalog=load_training_catalog()
    trainstore=formal_cli.gallery(FORMAL,catalog,"train")
    ledger=read_qualification(FORMAL,catalog,"train")
    protocol=read_json(FORMAL/"training/protocol.json")
    if tr.protocol_for(catalog,trainstore,ledger,protocol["config"])!=protocol:
        raise ValueError("frozen Phase4 training protocol changed")
    payload,audit=ck.audit(FORMAL/"training",protocol)
    if audit["uncommitted_tail_exists"] or payload["cursor"]!=len(protocol["query_order"]):
        raise ValueError("Phase4 training is not completely committed")
    done=read_json(FORMAL/"training/completed.json")
    if done!={"protocol_sha256":digest(protocol),"checkpoint_sha256":audit["published"][-1]["sha256"],
              "visits":payload["cursor"],"updates":payload["updates"]}:
        raise ValueError("Phase4 completion mismatch")
    epochs={x["sha256"]:x for x in audit["published"] if x["reason"]=="epoch_complete"}
    if len(epochs)!=protocol["config"]["epochs"]:raise ValueError("missing Epoch selection candidates")
    store=formal_cli.gallery(FORMAL,catalog,"validation")
    qual=read_qualification(FORMAL,catalog,"validation")
    results={}
    for h in epochs:
        path=FORMAL/"validation"/h
        results[h]=val.audit_result(path,store,qual,protocol)
    return protocol,epochs,results

def read_selection(path,state=None):
    path=Path(path).resolve()
    if path.parent!=(FORMAL/"selections").resolve():
        raise ValueError("only Phase4 immutable Validation selections accepted")
    record=read_json(path)
    if path.name!=digest(record)+".json":raise ValueError("selection filename/content SHA mismatch")
    protocol,epochs,results=phase4_state() if state is None else state
    h=validate_selection_record(record,results,epochs)
    cp=Path(epochs[h]["path"])/"checkpoint.pt";checked(cp,h)
    proof={str(path):sha(path),str(cp):h,str(cp.with_suffix(".json")):sha(cp.with_suffix(".json")),
           str(FORMAL/"training/protocol.json"):sha(FORMAL/"training/protocol.json"),
           str(FORMAL/"training/completed.json"):sha(FORMAL/"training/completed.json")}
    for candidate in epochs:
        for name in ("protocol.json","summary.json","completed.json"):
            p=FORMAL/"validation"/candidate/name;proof[str(p)]=sha(p)
    return {"path":str(path),"sha256":sha(path),"checkpoint":str(cp),"checkpoint_sha256":h,
            "architecture":protocol["config"]["architecture"],"training_protocol_sha256":digest(protocol),
            "selection":record,"proof":proof}

def resolve_selection():
    state=phase4_state()
    valid=[]
    for path in sorted((FORMAL/"selections").glob("*.json")):
        try:read_selection(path,state);valid.append(path)
        except (ValueError,KeyError):continue
    if len(valid)!=1:
        raise ValueError("selection pending/ambiguous; require one fully verified immutable record or explicit --selection")
    return valid[0]

def historical_header():
    ec,ep,compat=legacy("eval_common"),legacy("eval_protocol"),legacy("eval_compat")
    p=read_json(OLD_FULL/"protocol.json")
    compat.verify_sources(p,OLD_FULL,True)  # 保留原Storage修复兼容检查，绝不改旧证据。
    d=read_json(OLD_FULL/"completed.json");s=read_json(OLD_FULL/"summary.json")
    checked(OLD_FULL/"summary.json",d["summary_sha256"])
    if (not d["method_audit_passed"] or d["protocol_sha256"]!=digest(p)
            or not s["method_audit_passed"] or p["mode"]!="full"
            or (p["gallery_count"],p["query_count"])!=(3074,6156)
            or p["checkpoint_sha256"]["10"]!=EPOCH10_SHA):
        raise ValueError("historical official G-v2 Full is not verified")
    cv=read_json(OLD_FULL/"clean_verified.json")
    if cv.get("passed") is not True:raise ValueError("historical Clean failed")
    config=read_json(HERE/"config.json")
    if (s["groups"]["clean"]["metrics"]["mAP"]!=config["expected_clean_mAP"]
            or s["groups"]["epoch10"]["metrics"]["mAP"]!=config["expected_image_only_mAP"]):
        raise ValueError("historical baseline values differ from approved Full")
    proof={str(OLD_FULL/n):sha(OLD_FULL/n) for n in (
        "protocol.json","summary.json","completed.json","clean_verified.json","metrics.csv",
        "storage_fix_compatibility_record.json")}
    proof.update({str(ec.REFERENCE/n):h for n,h in ec.REF_SHA.items()})
    compatible=compat.compatibility_sources(p,OLD_FULL)[0]
    proof.update({path:h for path,h in compatible.items() if path.startswith(str(ec.OFFICIAL)+"/")
                  or path.endswith("irra_cuhk_official.yaml") or path.endswith("/ViT-B-16.pt")})
    return p,s,proof

def test_layout_matches(gallery,queries,prior):
    if gallery!=prior["gallery"] or queries!=prior["queries"]:
        raise ValueError("official Test order/Caption/PID/layout changed")
    if len({g["image"] for g in gallery})!=len(gallery) or len({q["row_id"] for q in queries})!=len(queries):
        raise ValueError("duplicate official identity")
    for q in queries:
        if not 0<=q["paired_index"]<len(gallery) or gallery[q["paired_index"]]["pid"]!=q["pid"]:
            raise ValueError("invalid retrieval positive identity")
    # paired_index/PID仅用于来源审计和最终指标，永不传给H。

def preflight(selection_path,text_batch=16):
    if type(text_batch) is not int or not 1<=text_batch<=64:raise ValueError("text batch 1..64 required")
    pins=source_pins();preflight_models();checked(RAW,RAW_SHA)
    selected=read_selection(selection_path)
    previous,summary,proof=historical_header()
    ec,ep=legacy("eval_common"),legacy("eval_protocol")
    gs,qs=ep.layout(read_json(RAW))
    if (len(gs),len(qs))!=(3074,6156):raise ValueError("CUHK-PEDES Test counts changed")
    official=ec.official_imports()[0](root=str(RAW.parent.parent),verbose=False).test
    if (official["img_paths"]!=[str(IMAGES/g["image"]) for g in gs]
            or official["image_pids"]!=[g["pid"] for g in gs]
            or official["captions"]!=[q["caption"] for q in qs]
            or official["caption_pids"]!=[q["pid"] for q in qs]):
        raise ValueError("official reader layout mismatch")
    ep.reference(gs,qs)
    for g,old in zip(gs,previous["gallery"]):
        g["image_sha256"]=sha(IMAGES/g["image"])
        if g["image_sha256"]!=old["image_sha256"]:raise ValueError("original Test image changed")
        proof[str(IMAGES/g["image"])]=g["image_sha256"]
    test_layout_matches(gs,qs,previous)
    proof.update(selected["proof"]);proof[str(RAW)]=RAW_SHA
    from phase3_inputs import CHECKPOINT,IRRA_WEIGHT
    proof[str(CHECKPOINT)]=EPOCH10_SHA;proof[str(IRRA_WEIGHT)]=IRRA_SHA
    cpu_only()
    return {"schema":"hv1-joint-full-1","execution_backend":"official_irra_real",
        "gallery":gs,"queries":qs,"gallery_count":3074,"query_count":6156,"text_batch":text_batch,
        "model_sha256":{"G":EPOCH10_SHA,"IRRA":IRRA_SHA,"H":selected["checkpoint_sha256"]},
        "H_selection":selected,"source_sha256":pins,"evidence_sha256":proof,
        "historical_protocol_sha256":digest(previous),
        "expected_baselines":{"Clean":summary["groups"]["clean"]["metrics"],
                              "Image-only":summary["groups"]["epoch10"]["metrics"]},
        "preprocess":{**PREPROCESS,"tokens":"official Byte-BPE 77; original truncated Test Query retained as NOOP"},"epsilon":8/255,
        "H_input":"original Caption only; Canonical/type/span extracted from Caption; greedy",
        "all_queries_retained":True,"Test_feedback_for_selection":False,
        "inference_settings":"fp32; TF32/MHA/flash/memory-efficient fastpaths off; math SDP on before H",
        "runtime":{"torch":str(torch.__version__),"python":__import__("sys").version,"cpu_threads":torch.get_num_threads()},
        "reference_reuse":"validated bulk Clean/Text reference and committed G-v2 Epoch10 image units",
        "relevance":"official IRRA rank; all same-PID positives; no camera filtering"}

def references(protocol):
    ec,ep=legacy("eval_common"),legacy("eval_protocol")
    images,text=ep.read_reference({"gallery":protocol["gallery"],"queries":protocol["queries"]})
    return images.clone(),text.clone()

def adversarial_gallery(protocol,check_stop=None):
    """只读已发布E10单元；逐条核对原summary SHA与兼容producer，不运行G。"""
    old=read_json(OLD_FULL/"protocol.json");summary=read_json(OLD_FULL/"summary.json")
    if digest(old)!=protocol["historical_protocol_sha256"]:raise ValueError("old protocol changed")
    storage=legacy("eval_storage")
    values=[];evidence={};max_linf=0.;g_count=image_count=0
    for index,rows in storage.shards(old["gallery"],old["image_batch"]):
        if check_stop:check_stop()
        target=storage.unit_path(OLD_FULL,"epoch10",index)
        for n in ("features.pt","metadata.json"):
            path=str(target/n);expected=summary["feature_file_sha256"].get(path)
            if expected is None:raise ValueError("unbound historical image feature")
            checked(path,expected);evidence[path]=expected
        value=storage.load_unit(OLD_FULL,"epoch10",index,old,rows,check_stop=check_stop)
        if value is None:raise ValueError("missing committed E10 image")
        claim=OLD_FULL/"receipts/epoch10"/f"{index:06d}.json"
        if read_json(claim)!={"family":"epoch10","batch_index":index,"status":"started_before_forward"}:
            raise ValueError("historical image computation receipt mismatch")
        evidence[str(claim)]=sha(claim)
        values.append(value["features"].clone())
        stats=value["stats"];g_count+=stats["G_samples"];image_count+=stats["IRRA_image_samples"]
        max_linf=max(max_linf,max(stats["linfs"]))
    if g_count!=len(protocol["gallery"]) or image_count!=len(protocol["gallery"]):
        raise ValueError("historical once-per-Gallery coverage mismatch")
    return torch.cat(values).clone(),{"G_historical_samples":g_count,"IRRA_image_historical_samples":image_count,
        "maximum_linf":max_linf,"source_files_sha256":evidence,"G_new_samples":0,"IRRA_image_new_samples":0}
