"""Step7.5.3 aggregated objective；默认5句observer短smoke，完整100句仅用户显式执行。"""
import argparse,copy,json,time,sys
from pathlib import Path
import deps
from selection_audit import load,save,digest,verify,new_output
from priority import MODES
from report import summarize
MANIFEST=deps.P74/"outputs/clean_scores/selection_manifest.json"
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("smoke","diagnostic"),default="smoke")
    p.add_argument("--allow-diagnostic",action="store_true")
    p.add_argument("--manifest",type=Path,default=MANIFEST)
    p.add_argument("--output",type=Path)
    p.add_argument("--case-id",action="append")
    p.add_argument("--smoke-observed-steps",type=int,default=1)
    return p
def baseline_history(result,prior):
    # 只读差异审计，不凭image SHA不同接受/拒绝未知CUDA漂移，也不放宽历史gate。
    out={"interpretation":"observational historical comparison; no new empirical tolerance/gate invented",
         "sequence_exact":result["fixed_sequence"]==prior["fixed_sequence"],
         "noise_exact":result["noise_sha256"]==prior["noise_sha256"],
         "transform_trace_exact":result["random_trace"]==prior["random_trace"],
         "rng_exact":result["rng_end_sha256"]==prior["rng_end_sha256"],"states":[]}
    for i,(a,b) in enumerate(zip(result["states"],prior["states"])):
        out["states"].append({"round":i,"cosine_abs_diff":abs(a["paired_cosine"]-b["paired_cosine"]),
            "F_abs_diff":abs(a["paired_soft_rank_F"]-b["paired_soft_rank_F"]),
            "F_rel_diff":abs(a["paired_soft_rank_F"]-b["paired_soft_rank_F"])/abs(b["paired_soft_rank_F"]),
            "rank_difference":a["positive_image_rank"]-b["positive_image_rank"]})
    out["round_image_SHA_exact"]=[a["post_projection_image_sha256"]==b["post_projection_image_sha256"]
                                 for a,b in zip(result["rounds"],prior["rounds"])]
    return out
def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=="diagnostic" and not args.allow_diagnostic:
        raise SystemExit("full100 is user-run only; --allow-diagnostic required")
    if args.mode=="diagnostic" and (args.case_id or args.smoke_observed_steps!=1):
        raise SystemExit("formal100 fixed rows/settings; subset/smoke override forbidden")
    if not 1<=args.smoke_observed_steps<=2:raise SystemExit("short smoke only")
    if digest(args.manifest)!=digest(MANIFEST):raise ValueError("must use unchanged Step74 fixed100 manifest")
    deps.assert_formal_clean()
    hashes=deps.frozen_hashes();own=deps.own_hashes()
    if args.mode=="diagnostic":
        from method_audit import require_smoke
        require_smoke()
    previous=deps.module("step75_readonly_step74_runner",deps.P74/"run.py")
    manifest,score_protocol,records=previous.checked_manifest(args.manifest,"diagnostic")
    payloads=manifest["rows"]
    if args.mode=="smoke":
        ids=args.case_id or ["cuhk:34:0","cuhk:27701:0","cuhk:17359:1","cuhk:25488:1","cuhk:24308:1"]
        if len(ids)!=len(set(ids)) or not 3<=len(ids)<=5:raise SystemExit("smoke requires3-5 existing manifest cases")
        selected=set(ids);payloads=[p for p in payloads if p["caption_id"] in selected]
        if len(payloads)!=len(ids):raise ValueError("smoke case missing from unchanged100 manifest")
    out=new_output(args.output or deps.ROOT/"outputs"/args.mode)
    import torch
    from PIL import Image
    from modes import execute_three
    from paired_prefix import prepare_uncapped
    from controlled_tta import tensor_hash
    from attribute_tta import TextTransport,IMAGE_EPSILON
    from attributes.irra_surrogate import IRRASurrogate
    old=deps.metrics_runner()
    sys.path.insert(0,str(deps.V4));import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim=v4.load_victim();source=IRRASurrogate(victim)
    versions={n:p._version for n,p in victim.named_parameters()}
    features,path_index,gallery_audit=old.load_gallery(hashes)
    bases=load(deps.V55/"outputs/full/semantic_bases.json")
    steps,transforms,scales=10,6,(.5,.75,1.25,1.5)
    locked={**hashes,**own,str(args.manifest.resolve()):digest(args.manifest),
            manifest["score_cache_path"]:manifest["score_cache_sha256"],
            manifest["score_protocol_path"]:manifest["score_protocol_sha256"]}
    protocol={"mode":args.mode,"modes":list(MODES),"case_count":len(payloads),"original_manifest_case_count":100,
        "manifest_sha256":digest(args.manifest),"executed_caption_ids":[p["caption_id"] for p in payloads],
        "natural_K_frozen":True,"k_max":None,"gamma":3,"beta":1,"score":"unchanged frozen S_raw",
        "steps_per_round":steps,"transforms_per_scale":transforms,"scales":scales,"seed":42,
        "torch_version":torch.__version__,
        "epsilon":IMAGE_EPSILON,"step_size":2/255,"momentum":.9,"tau":.02,"gallery_count":34054,"gallery":gallery_audit,
        "weighting":"S_i/sum(prefix S), not S^gamma",
        "projection":"only newest complete actual IG contribution versus S_raw weighted history; before original momentum/sign",
        "projection_numeric_dtype":"float64","projection_epsilon":torch.finfo(torch.float64).eps,
        "adaptive_transform_weights":"unchanged official aug/block probability gradients from each mode aggregate attribute loss; image conflict projection does not modify these parameters",
        "round1":"one actual original single-attribute round; image/momentum/observer/metrics/tape/RNG checkpoint shared",
        "random_control":"baseline anchor categorical trace replay; identical consumption of noncategorical RNG",
        "score_backwards_during_attack":0,"caption_loss_terms":0,"full_training":False,
        "method_definition":"aggregated objective is authoritative; no slow pixel equivalence gate",
        "slow_equivalence_gate":False,"per_attribute_backwards":0,
        "smoke_observed_steps_per_round":args.smoke_observed_steps if args.mode=="smoke" else None,
        "smoke_k1_full_ten_steps":True,"smoke_description":"configured steps=10 always; k1 full10; k>1 observer truncates after complete update",
        "formal_shared_adversarial_gallery":False,"frozen_sha256":locked}
    save(out/"protocol.json",protocol);save(out/"selection_manifest.json",manifest)
    historical={c["caption_id"]:c["modes"]["NATURAL_DYNAMIC_K"] for c in load(deps.P74/"outputs/diagnostic/per_caption_results.json")}
    cases=[];started=time.time()
    for payload in payloads:
        cid=payload["caption_id"];record=records[cid];t0=time.time()
        plan=prepare_uncapped(payload,record,bases,"cuda")
        original_metadata=copy.deepcopy([a.metadata for a in plan]);tv=[(a.w._version,a.Q._version) for a in plan]
        image_path=(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/record["paired_image"]).resolve()
        if not image_path.is_relative_to(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")):raise ValueError("image path")
        if path_index[record["paired_image"]]!=record["paired_gallery_index"]:raise ValueError("paired index")
        with Image.open(image_path) as im:image=victim.pixels(im).unsqueeze(0).cuda()
        transport=TextTransport(victim,record["caption"]);vectors={};mapping=[]
        for a in plan:
            vector,report=transport.read(a)
            if not report["zero_exact"] or report["central_fd_relative_error"]>.02:raise ValueError("unchanged JVP FD gate")
            vectors[a.instance_id]=vector;mapping.append(report)
        vector_sha={key:tensor_hash(v) for key,v in vectors.items()}
        evaluator=old.RetrievalMetrics(source,transport.clean,features,record["paired_gallery_index"],tau=.02)
        data,checks=execute_three(source,image,plan,vectors,evaluator,steps=steps,transforms=transforms,scales=scales,
            observed_limit=args.smoke_observed_steps if args.mode=="smoke" and len(plan)>1 else None)
        if original_metadata!=[a.metadata for a in plan] or any((a.w._version,a.Q._version)!=n for a,n in zip(plan,tv)):
            raise ValueError("frozen attribute/basis/location changed")
        if vector_sha!={key:tensor_hash(v) for key,v in vectors.items()}:raise ValueError("JVP changed")
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):
            raise ValueError("IRRA not frozen")
        checks.update(zero_exact=all(item["zero_exact"] for item in mapping),AttributeInstance_S_TopK_JVP_unchanged=True,IRRA_frozen=True)
        case={"caption_id":cid,"pid":record["pid"],"caption":record["caption"],"paired_image":record["paired_image"],
              "candidate_count":payload["candidate_count"],"natural_k":len(plan),"selected_attributes":original_metadata,
              "JVP_mapping":mapping,"JVP_sha256":vector_sha,"JVP_computations":len(plan),
              "modes":data,"checks":checks,"seconds":time.time()-t0}
        if args.mode=="diagnostic":case["Step74_baseline_history_audit"]=baseline_history(data[MODES[0]],historical[cid])
        cases.append(case);save(out/"per_caption_results.json",cases)
        print("CASE_DONE",cid,"natural_k",len(plan),"seconds",case["seconds"],flush=True)
    verify(locked);deps.assert_formal_clean()
    summary,rounds,conflicts,projection=summarize(cases)
    for name,value in [("summary",summary),("per_round_results",rounds),("conflict_statistics",conflicts),("projection_statistics",projection)]:
        save(out/(name+".json"),value)
    if projection[MODES[2]]["all"]["no_conflict_exact_failures"]:raise ValueError("no-conflict modified gradient")
    audit={"case_count":len(cases),"modes":list(MODES),"steps_per_round":steps,
           "logical_rounds_per_mode":sum(c["natural_k"] for c in cases),
           "physical_rounds":sum(1+3*(c["natural_k"]-1) for c in cases),
           "round1_image_momentum_metrics_noise_transform_RNG_exact":True,"noise_transform_RNG_exact":True,
           "projection_real_momentum_path":True,"no_conflict_projection_exact":True,"only_new_attribute_modified":True,
           "score_backwards_during_attack":0,"score_cache_SHA_unchanged":True,"source_SHA_unchanged":True,
           "IRRA_frozen":True,"AttributeInstance_S_TopK_JVP_unchanged":True,"caption_loss_terms":0,"no_training":True,
           "maximum_linf":max(c["modes"][m]["final_linf"] for c in cases for m in MODES),
           "diagnostic_run":args.mode=="diagnostic","slow_equivalence_gate":False,
           "configured_steps_per_round":steps,
           "actual_optimization_steps":sum(len(e["steps"]) for c in cases for m in MODES for e in c["modes"][m]["rounds"]) - 2*sum(len(c["modes"][MODES[0]]["rounds"][0]["steps"]) for c in cases),
           "seconds":time.time()-started}
    from method_audit import audit as audit_method
    method_report=audit_method(cases)
    save(out/"method_audit.json",method_report)
    if not method_report["passed"]:raise ValueError("Step753 method audit failed")
    audit["method_audit_passed"]=True
    save(out/"audit.json",audit);save(out/"completed.json",audit)
    print("COMPLETE",json.dumps(audit),flush=True)
if __name__=="__main__":main()
