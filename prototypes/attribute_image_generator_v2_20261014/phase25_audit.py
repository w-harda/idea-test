"""Phase 2.5只读审计；不调用模型、评分、Top-K或JVP计算。"""
import argparse,math,json,statistics
from pathlib import Path
from collections import Counter
import torch
import cache_io as io
from condition_schema import validate,error,IDENTITY
from training_data import checked_cache,check_pair,RAW,SPLIT
from prepare_full import source_rows,SOURCE,OLD_SMOKE,SMALL_IDS

def distribution(entries):
    result={}
    for role in ("train","validation","all"):
        rows=[e for e in entries if role=="all" or e["role"]==role]
        ks=[e["K"] for e in rows]
        if any(type(k) is not int or k not in range(8) for k in ks):
            raise ValueError("unexpected K outside frozen 0..7")
        counts=Counter(ks)
        result[role]={"samples":len(rows),"K_counts":{str(k):counts[k] for k in range(8)},
            "mean_K":sum(ks)/len(ks) if ks else None,"JVP_total":sum(ks)}
    return result

def preflight():
    locks=io.load(io.OUTPUTS/"phase25/protected_sources.json");io.verify(locks)
    rows,upstream=source_rows()
    old=io.checked_cache(OLD_SMOKE)
    if set(e["row_id"] for e in old["entries"])!=set(SMALL_IDS):
        raise ValueError("original five smoke identities changed")
    entries=[i for _,_,i in rows]
    pids={r:{e["pid"] for e in entries if e["role"]==r} for r in ("train","validation")}
    if pids["train"]&pids["validation"]:raise ValueError("PID leakage")
    return {"schema":"gv2-phase25-preflight-1","passed":True,"source_count":len(rows),
        "source_SHA_count":len(upstream),"protected_SHA_count":len(locks),
        "distribution":distribution(entries),"train_PIDs":len(pids["train"]),
        "validation_PIDs":len(pids["validation"]),"PID_overlap":0,
        "producer_locked_unchanged":True,"old_smoke_count":5,"new_scoring_backwards":0,
        "CUDA_initialized":torch.cuda.is_initialized()}

def condition_audit(entry,c,meta,source,source_meta,original):
    validate(c,entry)
    selected=source_meta["payload"]["selected_attributes"]
    ids=[a["instance_id"] for a in selected]
    if (c["K"]!=source["K"] or c["K"]!=source_meta["payload"]["natural_k"]
        or selected!=source_meta["payload"]["ranked_attributes"][:c["K"]]
        or c["selected_instance_ids"]!=ids or ids!=source_meta["selected_instance_ids"]):
        raise ValueError("frozen ID/order/K mismatch")
    expected={**{k:source[k] for k in ("row_id","pid","caption")},
        "image_rel_path":source["image"],"caption_index":int(source["row_id"].split(":")[-1]),
        "split":"train","role":source_meta["role"]}
    if any(c[k]!=expected[k] for k in IDENTITY):raise ValueError("source identity mismatch")
    if meta["source_selection"]!=source_meta["payload"] or meta["source_score_record"]!=source_meta["score_record"]:
        raise ValueError("source score/selection changed")
    if meta["source_fingerprint"]!=io.digest([source,source_meta]):raise ValueError("source fingerprint changed")
    if meta["source_image_sha256"]!=source["image_sha256"]:raise ValueError("source image SHA changed")
    if not torch.equal(c["S_raw"],torch.tensor([a["S_raw"] for a in selected],dtype=torch.float64)):
        raise ValueError("S_raw not exact original")
    if not torch.equal(c["score_weights"],torch.tensor(source_meta["score_normalized_weights"],dtype=torch.float64)):
        raise ValueError("weights not exact original")
    if not torch.equal(c["z_t"],original["z_t"]):raise ValueError("z_t not exact original")
    rebuilt=(c["score_weights"].float()[:,None]*c["attribute_jvps"]).sum(0) if c["K"] else torch.zeros(512)
    if not torch.allclose(rebuilt,c["v_local"],atol=1e-9,rtol=1e-5):raise ValueError("aggregate reconstruction")
    if not torch.allclose(c["v_local"],original["v_local"],atol=1e-9,rtol=1e-5):
        raise ValueError("original aggregate mismatch")
    checks=meta["JVP_checks"]
    if len(checks)!=c["K"] or [q["instance_id"] for q in checks]!=ids:raise ValueError("JVP audit coverage/order")
    fd=[]
    for i,q in enumerate(checks):
        x=q["central_fd_relative_error"]
        if q["zero_exact"] is not True or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=.02:
            raise ValueError("JVP FD/zero check failed")
        norm=float(c["attribute_jvps"][i].double().norm())
        if not math.isclose(q["direction_norm"],norm,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("JVP norm audit mismatch")
        fd.append(x)
    unique_vectors={c["attribute_jvps"][i].numpy().tobytes() for i in range(c["K"])}
    return {"row_id":c["row_id"],"K":c["K"],"aggregate_error":error(rebuilt,c["v_local"]),
        "original_aggregate_error":error(c["v_local"],original["v_local"]),"FD_errors":fd,
        "equal_JVP_extra_instances":c["K"]-len(unique_vectors),"z_t_exact":True}

def full_audit(cache,output):
    m=checked_cache(cache,require_full=True)
    io.verify(m["protocol"]["upstream_sha256"])
    locks=io.load(io.OUTPUTS/"phase25/protected_sources.json");io.verify(locks)
    original_manifest=io.load(SOURCE/"manifest.json")
    sources={e["row_id"]:(e,role) for role in ("train","validation") for e in original_manifest[role]}
    if len(sources)!=2176 or set(sources)!={e["row_id"] for e in m["entries"]}:
        raise ValueError("full original coverage mismatch")
    raw=io.load(RAW);split=io.load(SPLIT)
    roles={r:{e["pid"] for e in m["entries"] if e["role"]==r} for r in ("train","validation")}
    if roles["train"]&roles["validation"]:raise ValueError("PID leakage")
    test_pids={x["id"] for x in raw if x["split"]=="test"}
    if (roles["train"]|roles["validation"])&test_pids:raise ValueError("official test PID leakage")
    images={r:{e["image_rel_path"] for e in m["entries"] if e["role"]==r} for r in roles}
    if images["train"]&images["validation"]:raise ValueError("image leakage")
    old=io.checked_cache(OLD_SMOKE);smoke={e["row_id"]:io.read_entry(e)[0] for e in old["entries"]}
    cache_dir=Path(cache)
    published={p.name for p in (cache_dir/"samples").iterdir() if not p.name.startswith(".")}
    if published!={e["row_id"].replace(":","_") for e in m["entries"]}:raise ValueError("orphan/missing sample directories")
    reports=[];reused=0;fd=[]
    out=io.safe(output)
    with io.output_lock(out):
        path=out/"per_sample.jsonl"
        if path.exists():raise ValueError("audit output exists; use new immutable output")
        with path.open("x",encoding="utf-8") as f:
            for e in m["entries"]:
                c,meta,_=io.read_entry(e,io.digest(m["protocol"]))
                src,role=sources[e["row_id"]]
                md=io.load(src["metadata_path"])
                if role!=e["role"] or md["role"]!=role:raise ValueError("role changed")
                check_pair(e,c,meta,raw,split)
                old_tensor=torch.load(src["tensor_path"],map_location="cpu",weights_only=True)
                report=condition_audit(e,c,meta,src,md,old_tensor)
                if e["row_id"] in smoke:
                    prior=smoke[e["row_id"]]
                    for name in ("S_raw","score_weights","attribute_jvps","z_t","v_local"):
                        if not torch.equal(c[name],prior[name]):raise ValueError("original smoke not reused exact")
                    reused+=1
                reports.append(report);fd.extend(report["FD_errors"])
                f.write(json.dumps(report,ensure_ascii=False,allow_nan=False)+"\n")
        if reused!=5:raise ValueError("five original smoke reuse coverage")
        dist=distribution(m["entries"])
        if len(fd)!=dist["all"]["JVP_total"]:raise ValueError("JVP count mismatch")
        fd_tensor=torch.tensor(fd,dtype=torch.float64)
        summary={"schema":"gv2-phase25-audit-1","method_audit_passed":True,
            "cache_path":str(cache_dir),"manifest_sha256":io.sha(cache_dir/"manifest.json"),
            "protocol_sha256":m["protocol_sha256"],"completed_sha256":io.sha(cache_dir/"completed.json"),
            "summary_sha256":io.sha(cache_dir/"summary.json"),"per_sample_sha256":io.sha(path),
            "distribution":dist,"original_coverage_exact":True,"source_SHA_count":len(m["protocol"]["upstream_sha256"]),
            "producer_SHA_count":len(m["protocol"]["producer_sha256"]),"protected_SHA_count":len(locks),
            "train_PIDs":len(roles["train"]),"validation_PIDs":len(roles["validation"]),
            "PID_overlap":0,"official_test_PID_overlap":0,"image_overlap":0,"unique_images":sum(len(v) for v in images.values()),
            "reused_original_smoke_exact":reused,"z_t_all_exact":True,"S_raw_weights_order_all_exact":True,
            "maximum_reconstruction_error":max(r["aggregate_error"]["max_abs_error"] for r in reports),
            "maximum_original_aggregate_error":max(r["original_aggregate_error"]["max_abs_error"] for r in reports),
            "FD_count":len(fd),"FD_min":min(fd),"FD_mean":statistics.mean(fd),"FD_median":statistics.median(fd),
            "FD_p95":float(torch.quantile(fd_tensor,.95)),"FD_max":max(fd),
            "zero_exact_passes":len(fd),"failed_samples":[],"failed_JVPs":0,"zero_JVP_replacements":0,
            "equal_JVP_samples":sum(r["equal_JVP_extra_instances"]>0 for r in reports),
            "equal_JVP_extra_instances":sum(r["equal_JVP_extra_instances"] for r in reports),
            "pending_directories":len(list((cache_dir/"samples").glob(".pending-*"))),
            "CUDA_initialized":torch.cuda.is_initialized(),"new_scoring_backwards":0,"new_JVPs":0,
            "G_forwards":0,"optimizer_steps":0,"formal_training":False}
        io.verify(locks);io.verify(m["protocol"]["upstream_sha256"])
        io.save_json(out/"summary.json",summary)
        io.save_json(out/"completed.json",{"method_audit_passed":True,"summary_sha256":io.sha(out/"summary.json")})
        return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true");p.add_argument("--cache",type=Path,default=io.OUTPUTS/"conditions_full")
    p.add_argument("--output",type=Path,default=io.OUTPUTS/"phase25/audit")
    a=p.parse_args(argv)
    if a.preflight:
        result=preflight();io.save_json(a.output/"preflight.json",result)
    else:result=full_audit(a.cache,a.output)
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=="__main__":main()
