"""冻结S_raw的Top/Random/Bottom选择对照。默认5句短smoke；100句必须用户显式启动。"""
import argparse,copy,json,sys,time
from pathlib import Path
import deps
from selection import MODES,verify_row,cache_sources
from selection_audit import load,save,digest,verify,new_output
MANIFEST=deps.ROOT/"outputs/manifest/selection_manifest.json"
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["smoke","diagnostic"],default="smoke")
    p.add_argument("--allow-diagnostic",action="store_true")
    p.add_argument("--output",type=Path)
    return p
def checked_inputs():
    hashes=deps.frozen_hashes();own=deps.own_hashes()
    manifest=load(MANIFEST);reference,protocol=cache_sources()
    if manifest["case_count"]!=100 or manifest["score_cache_sha256"]!=reference["score_cache_sha256"]:raise ValueError("frozen manifest/cache")
    ids={p["caption_id"] for p in manifest["rows"]}
    records={r["row_id"]:r for r in deps.rows(reference["score_cache_path"]) if r["row_id"] in ids}
    if set(records)!=ids:raise ValueError("missing cached caption")
    split=load(deps.V4/"outputs/split.json")
    validation={split["records"][i]["row_id"]:split["records"][i] for i in split["validation_indices"]}
    for row in manifest["rows"]:
        record=records[row["caption_id"]]
        verify_row(row,record)
        if record["caption"]!=validation[record["row_id"]]["caption"] or record["pid"]!=validation[record["row_id"]]["pid"]:
            raise ValueError("validation/source/PID changed")
    return manifest,records,{**hashes,**own,manifest["score_cache_path"]:manifest["score_cache_sha256"],manifest["score_protocol_path"]:manifest["score_protocol_sha256"]}
def smoke_rows(rows):
    chosen=[]
    for k in (2,3,5,6,7):
        group=[p for p in rows if p["natural_k"]==k]
        if group:chosen.append(group[0])
    if not 3<=len(chosen)<=5:raise ValueError("3-5 fixed smoke cases required")
    return chosen
def require_smoke():
    done=load(deps.ROOT/"outputs/smoke/completed.json")
    p=load(deps.ROOT/"outputs/smoke/protocol.json")
    if not done["method_audit_passed"] or not 3<=done["case_count"]<=5:raise ValueError("successful short smoke required")
    verify(p["frozen_sha256"])
def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=="diagnostic" and not args.allow_diagnostic:raise SystemExit("user-run100 only; --allow-diagnostic required")
    deps.assert_formal_clean()
    manifest,records,locked=checked_inputs()
    if args.mode=="diagnostic":require_smoke()
    payloads=manifest["rows"] if args.mode=="diagnostic" else smoke_rows(manifest["rows"])
    out=new_output(args.output or deps.ROOT/"outputs"/args.mode)
    import torch
    from PIL import Image
    from engines import execute_three,plans_for
    from report import summarize
    from controlled_tta import tensor_hash
    from attribute_tta import TextTransport,IMAGE_EPSILON
    from attributes.irra_surrogate import IRRASurrogate
    sys.path.insert(0,str(deps.V4))
    import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim=v4.load_victim();source=IRRASurrogate(victim)
    versions={n:p._version for n,p in victim.named_parameters()}
    metrics=deps.metrics_runner();features,path_index,gallery=metrics.load_gallery(locked)
    bases=load(deps.V55/"outputs/full/semantic_bases.json")
    protocol={"mode":args.mode,"modes":MODES,"case_count":len(payloads),"steps_per_round":10,
        "scales":[.5,.75,1.25,1.5],"transforms_per_scale":6,"epsilon":IMAGE_EPSILON,"step_size":2/255,
        "momentum":.9,"tau":.02,"gallery_count":34054,"gallery":gallery,"seed":42,
        "scheduler":"unchanged unweighted PREFIX_BASELINE","score_weighting":False,"priority_projection":False,
        "score_backwards_during_attack":0,"caption_loss_terms":0,"no_training":True,
        "manifest_sha256":digest(MANIFEST),"natural_K_frozen":True,
        "random_selection":"uniform without replacement; per-caption SHA256-derived local seed; frozen before attack",
        "random_attack_control":"S_TOPK anchors categorical transform trace; all modes seed42; original adaptive parameter update retained",
        "smoke_observed_steps_per_round":1 if args.mode=="smoke" else None,
        "smoke_note":"all rounds retain configured10 steps; smoke observer stops after one complete original update",
        "formal_shared_adversarial_gallery":False,"frozen_sha256":locked}
    save(out/"protocol.json",protocol)
    save(out/"selection_manifest.json",{**manifest,"executed_caption_ids":[p["caption_id"] for p in payloads]})
    cases=[];started=time.time()
    for row in payloads:
        t=time.time();record=records[row["caption_id"]]
        plans=plans_for(row,record,bases,"cuda")
        all_attributes={a.instance_id:a for plan in plans.values() for a in plan}
        snapshots={m:[(copy.deepcopy(a.metadata),a.w._version,a.Q._version) for a in plan] for m,plan in plans.items()}
        image_root=Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")
        image_path=(image_root/record["paired_image"]).resolve()
        if not image_path.is_relative_to(image_root) or path_index[record["paired_image"]]!=record["paired_gallery_index"]:raise ValueError("paired image/source")
        with Image.open(image_path) as im:image=victim.pixels(im).unsqueeze(0).cuda()
        transport=TextTransport(victim,record["caption"]);vectors={};mapping=[]
        for a in all_attributes.values():
            vector,check=transport.read(a)
            if not check["zero_exact"] or check["central_fd_relative_error"]>.02:raise ValueError("frozen JVP gate failed")
            vectors[a.instance_id]=vector;mapping.append(check)
        vector_hashes={i:tensor_hash(v) for i,v in vectors.items()}
        evaluator=metrics.RetrievalMetrics(source,transport.clean,features,record["paired_gallery_index"],tau=.02)
        results,checks=execute_three(source,image,plans,vectors,evaluator,observed_limit=1 if args.mode=="smoke" else None)
        if snapshots!={m:[(a.metadata,a.w._version,a.Q._version) for a in plan] for m,plan in plans.items()}:
            raise ValueError("AttributeInstance/basis/location changed")
        if vector_hashes!={i:tensor_hash(v) for i,v in vectors.items()}:raise ValueError("JVP changed")
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):raise ValueError("IRRA frozen state")
        checks.update(candidate_source_exact=True,random_selection_reproducible=True,IRRA_frozen=True,
                      zero_exact=all(x["zero_exact"] for x in mapping),AttributeInstance_S_JVP_unchanged=True)
        for m,result in results.items():
            if result["fixed_sequence"]!=row["selection_ids"][m] or len(result["rounds"])!=row["natural_k"]:raise ValueError("K/selection changed")
            for e in result["rounds"]:
                if e["configured_steps_per_round"]!=10 or len(e["steps"])!=(1 if args.mode=="smoke" else 10):raise ValueError("step budget changed")
        case={"caption_id":row["caption_id"],"caption":record["caption"],"pid":record["pid"],
            "candidate_count":row["candidate_count"],"natural_k":row["natural_k"],
            "candidate_attributes":copy.deepcopy(row["ranked_attributes"]),
            "selected_attributes":{m:[copy.deepcopy(a.metadata) for a in plans[m]] for m in MODES},
            "selection_ids":row["selection_ids"],"set_overlap":row["set_overlap"],
            "random_seed_hex":row["random_seed_hex"],"JVP_mapping":mapping,"JVP_sha256":vector_hashes,
            "modes":results,"checks":checks,"seconds":time.time()-t}
        cases.append(case);save(out/"per_caption_results.json",cases)
        print("CASE_DONE",row["caption_id"],"K",row["natural_k"],"m",row["candidate_count"],"seconds",case["seconds"],flush=True)
    verify(locked);deps.assert_formal_clean()
    summary,rounds=summarize(cases)
    save(out/"summary.json",summary);save(out/"round_marginal_effects.json",rounds)
    audit={"method_audit_passed":True,"case_count":len(cases),"modes":MODES,
           "logical_rounds_per_mode":sum(c["natural_k"] for c in cases),
           "physical_rounds":sum(len(c["modes"][m]["rounds"]) for c in cases for m in MODES if not c["modes"][m]["execution_reuse_of"]),
           "actual_optimization_steps":sum(len(e["steps"]) for c in cases for m in MODES if not c["modes"][m]["execution_reuse_of"] for e in c["modes"][m]["rounds"]),
           "steps_per_round":10,"smoke_observed_steps":1 if args.mode=="smoke" else None,
           "same_K":True,"candidate_source_exact":True,"noise_transform_RNG_exact":True,
           "score_cache_source_SHA_unchanged":True,"IRRA_AttributeInstance_S_JVP_frozen":True,
           "score_backwards_during_attack":0,"caption_loss_terms":0,"score_weighting":False,"priority_projection":False,
           "maximum_linf":max(c["modes"][m]["final_linf"] for c in cases for m in MODES),
           "seconds":time.time()-started,"no_training":True,"diagnostic_run":args.mode=="diagnostic"}
    save(out/"audit.json",audit);save(out/"completed.json",audit)
    print("COMPLETE",json.dumps(audit),flush=True)
if __name__=="__main__":main()
