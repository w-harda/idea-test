"""Step7.3.2：默认仅两句2-step smoke；固定25句必须用户显式执行。"""
import os, sys
from pathlib import Path
sys.dont_write_bytecode = True
for key, sub in [("TORCH_HOME","torch"),("HF_HOME","huggingface"),("XDG_CACHE_HOME","xdg"),("CUDA_CACHE_PATH","cuda")]:
    os.environ.setdefault(key, "/home/lzf/ldx/cache/"+sub)
ROOT = Path(__file__).resolve().parent
REPO = Path("/home/lzf/ldx/projects/idea-TBPS-test1")
sys.path.insert(0,str(ROOT))
from prefix_tta import *
import argparse, importlib.util, json, subprocess, time, collections
from PIL import Image
from attribute_tta import prepared_plan, TextTransport, seed_all, digest, IMAGE_EPSILON
from attributes.irra_surrogate import IRRASurrogate
spec = importlib.util.spec_from_file_location("step73_original_runner", P73/"run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
SMOKE_IDS = ("cuhk:34:0", "cuhk:0:1")  # 固定k1与旧saved k3 smoke；k2/k4由CPU tests覆盖。
def load(p): return old.load(p)
def save(p,x): return old.save(p,x)
def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("smoke","diagnostic"),default="smoke")
    p.add_argument("--allow-diagnostic",action="store_true")
    p.add_argument("--output",type=Path)
    return p

def inputs():
    hashes=load(ROOT/"provenance.lock.json")["frozen_sha256"]
    old.verify(hashes)
    _,rows,payloads,bases,prior=old.checked_inputs()
    protocol=load(P73/"outputs/diagnostic/protocol.json")
    completed=load(P73/"outputs/diagnostic/completed.json")
    history={c["caption_id"]:c for c in load(P73/"outputs/diagnostic/results.json")}
    if completed["case_count"]!=25 or protocol["row_ids"]!=prior["row_ids"] or list(history)!=prior["row_ids"]:
        raise ValueError("fixed25 saved population changed")
    if protocol["steps_per_attribute"]!=10 or protocol["epsilon"]!=IMAGE_EPSILON:
        raise ValueError("baseline settings changed")
    return hashes,rows,payloads,bases,protocol,history

def best_prefix(data):
    states=data["states"]
    b=max(range(len(states)),key=lambda i:states[i]["paired_soft_rank_F"])
    return {"best_round":b,"final_equals_best":states[b]["paired_soft_rank_F"]==states[-1]["paired_soft_rank_F"],
            "F_best_minus_final":states[b]["paired_soft_rank_F"]-states[-1]["paired_soft_rank_F"],
            "rank_at_F_best_minus_final":states[b]["positive_image_rank"]-states[-1]["positive_image_rank"],
            "F_final_minus_round1":states[-1]["paired_soft_rank_F"]-states[1]["paired_soft_rank_F"] if len(states)>1 else None}

def summarize(results):
    out={"case_count":len(results),"modes":{},"comparisons":{}}
    for mode in MODES:
        cs=[c["modes"][mode] for c in results]
        mult=[c for c in cs if len(c["rounds"])>=2]
        out["modes"][mode]={
            "clean_to_final":{key:old.stats([c["clean_to_final"][key] for c in cs]) for key in ("paired_cosine","paired_soft_rank_F","positive_image_rank")},
            "multi_round_caption_count":len(mult),
            "final_equals_best_count":sum(best_prefix(c)["final_equals_best"] for c in mult),
            "final_weaker_than_best_count":sum(not best_prefix(c)["final_equals_best"] for c in mult),
            "best_round_distribution":dict(collections.Counter(str(best_prefix(c)["best_round"]) for c in mult)),
            "F_best_minus_final":old.stats([best_prefix(c)["F_best_minus_final"] for c in mult]),
            "rank_at_F_best_minus_final":old.stats([best_prefix(c)["rank_at_F_best_minus_final"] for c in mult]),
            "F_final_minus_round1":old.stats([best_prefix(c)["F_final_minus_round1"] for c in mult]),
            "rounds":{}, "projection_clipping_ratio":old.stats([s["projection_clipping_ratio"] for c in cs for e in c["rounds"] for s in e["steps"]]),
            "budget_linf_max":max([c["final_linf"] for c in cs],default=0),
        }
        for i in range(1,5):
            events=[c["rounds"][i-1] for c in cs if len(c["rounds"])>=i]
            out["modes"][mode]["rounds"][str(i)]={
                "count":len(events),"F_recovery_count":sum(e["metric_delta"]["paired_soft_rank_F"]<0 for e in events),
                "delta_F":old.stats([e["metric_delta"]["paired_soft_rank_F"] for e in events]),
                "actual_update_l2":old.stats([e["post_projection_actual_change"]["l2"] for e in events]),
                "clipping_ratio":old.stats([e["projection_clipping_ratio_mean_over_steps"] for e in events]),
            }
    for key in ("paired_cosine","paired_soft_rank_F","positive_image_rank"):
        xs=[c["modes"]["PREFIX_NEW"]["clean_to_final"][key]-c["modes"]["SEQUENTIAL_NEW"]["clean_to_final"][key] for c in results]
        signs=[-x if key=="paired_cosine" else x for x in xs]
        out["comparisons"][key]={"PREFIX_wins":sum(x>0 for x in signs),"SEQUENTIAL_wins":sum(x<0 for x in signs),
                                 "ties":sum(x==0 for x in signs),"PREFIX_minus_SEQUENTIAL":old.stats(xs)}
    out["per_caption"]=[{"caption_id":c["caption_id"],"k":c["k_star"],"modes":{
        m:{"clean_to_final":c["modes"][m]["clean_to_final"],**best_prefix(c["modes"][m])} for m in MODES}} for c in results]
    out["caveat"]="paired-image fixed clean competition diagnostic; not shared adversarial gallery/PID evaluation. No scheduler freeze from smoke."
    return out

def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=="diagnostic" and not args.allow_diagnostic:
        raise SystemExit("fixed25 is user-run only; supply --allow-diagnostic")
    out=(args.output or ROOT/"outputs"/args.mode).resolve()
    if not out.is_relative_to(Path("/home/lzf/ldx")) or out.exists():
        raise ValueError("new output directory under /home/lzf/ldx required; no overwrite")
    hashes,rows,payloads,bases,prior,history=inputs()
    ids=SMOKE_IDS if args.mode=="smoke" else tuple(prior["row_ids"])
    if args.mode=="smoke" and len(ids)>2: raise ValueError("smoke boundary")
    steps=2 if args.mode=="smoke" else prior["steps_per_attribute"]
    transforms=1 if args.mode=="smoke" else prior["transforms_per_scale"]
    scales=None if args.mode=="smoke" else tuple(prior["scales"])
    out.mkdir(parents=True)
    started=time.time()
    sys.path.insert(0,str(REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
    import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim=v4.load_victim()
    source=IRRASurrogate(victim)
    versions={n:p._version for n,p in victim.named_parameters()}
    features,path_index,gallery_audit=old.load_gallery(hashes)
    own={str(p):digest(p) for p in ROOT.glob("*.py")}
    protocol={
        "mode":args.mode,"row_ids":list(ids),"modes":list(MODES),
        "losses":{"SEQUENTIAL_NEW":"L(ai)","PREFIX_NEW":"mean(L(a1),...,L(ai)); no re-normalization/no weighting"},
        "steps_per_attribute":steps,"transforms_per_scale":transforms,"scales":scales,
        "seed":42,"gamma":3,"beta":1,"k_max":4,"epsilon":IMAGE_EPSILON,"step_size":2/255,"momentum":.9,
        "selection":"original saved clean plan, no scoring/ranking/reselection",
        "random_control":"SEQUENTIAL_NEW predetermined anchor; PREFIX_NEW replays actual categorical IDs with identical noise/RNG consumption",
        "adaptive_weights":"official updates remain mode-specific; transform IDs shared",
        "JVP":"unchanged Step72 TextTransport.read; once per selected instance, precomputed before optimization",
        "gradient_diagnostics":"round input before noise; untransformed; post hoc after both attacks; not IG-node decomposition",
        "gallery_count":34054,"gallery":gallery_audit,"tau":.02,
        "regression_reference_variants":str(ROOT/"regression_reference_variants.json"),
        "conditional_repeatability_reference":str(ROOT/"repeatability_reference_35.json"),
        "round1_execution":"one common original single-attribute update; branch image/momentum/RNG/trace at round2",
        "RAW_history_only":True,"no_training":True,"no_feature_extraction":True,"no_formal_gallery_attack":True,
        "frozen_sha256":hashes,"source_sha256":own,
    }
    save(out/"protocol.json",protocol)
    lookup={r["row_id"]:r for r in rows}
    plans={p["caption_id"]:p for p in payloads}
    smoke_saved={c["caption_id"]:c for c in load(P73/"outputs/smoke_observer_verified/results.json")}
    reference_variants=load(ROOT/"regression_reference_variants.json")["variants"]
    from repeatability_gate import load_reference
    numerical_reference=load_reference()
    numerical_contract={"steps":steps,"transforms":transforms,"scales":list(scales) if scales else None,
                        "epsilon":IMAGE_EPSILON,"step_size":2/255,"momentum":.9,"seed":42}
    results=[]
    for cid in ids:
        t0=time.time();record=lookup[cid]
        plan=prepared_plan(plans[cid],record,bases,"cuda")
        metadata=copy.deepcopy([a.metadata for a in plan])
        tensor_versions=[(a.w._version,a.Q._version) for a in plan]
        paired=path_index[record["paired_image"]]
        if paired!=record["paired_gallery_index"]: raise ValueError("paired gallery mapping")
        image_path=(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/record["paired_image"]).resolve()
        if not image_path.is_relative_to(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")):raise ValueError("image path")
        with Image.open(image_path) as im:image=victim.pixels(im).unsqueeze(0).cuda()
        transport=TextTransport(victim,record["caption"])
        vectors,mapping={},[]
        for a in plan:
            v,report=transport.read(a)
            if report["central_fd_relative_error"]>.02:raise ValueError("JVP FD verification")
            vectors[a.instance_id]=v;mapping.append(report)
        vector_hashes={aid:__import__("controlled_tta").tensor_hash(v) for aid,v in vectors.items()}
        cached={m:cached_round_guidance(m,plan,vectors) for m in MODES}
        evaluator=old.RetrievalMetrics(source,transport.clean,features,paired,tau=.02)
        data,round_inputs,tape_records={},{},None
        regression=None
        common_first=[]
        for mode in MODES:
            final_images=[]
            seed_all(42)
            tape=CategoricalTape() if mode=="SEQUENTIAL_NEW" else CategoricalTape(tape_records)
            result,inputs_at_round,momentum=execute(source,image,plan,cached[mode],evaluator,steps=steps,transforms=transforms,scales=scales,tape=tape,
                first_round_capture=common_first if mode=="SEQUENTIAL_NEW" else None,
                first_round_checkpoint=common_first[0] if mode=="PREFIX_NEW" and common_first else None,
                final_image_capture=final_images if mode=="SEQUENTIAL_NEW" else None)
            if mode=="SEQUENTIAL_NEW":
                tape_records=copy.deepcopy(tape.records)
                if args.mode=="diagnostic":
                    try:
                        regression=saved_regression(result,history[cid]["modes"]["NEW"],reference_variants.get(cid,()),
                            numerical_reference=numerical_reference if cid=="cuhk:35:1" else None,
                            current_images=list(inputs_at_round)+final_images if plan else [image],
                            execution_contract=numerical_contract)
                    except RegressionMismatch as error:
                        failure={"caption_id":cid,"mode":mode,"report":error.report,
                                 "current_sequential":result,"saved_NEW":history[cid]["modes"]["NEW"]}
                        save(out/"regression_failure.json",failure)
                        print("REGRESSION_FAILURE",cid,json.dumps(error.report,ensure_ascii=False),flush=True)
                        raise
                elif cid in smoke_saved:
                    regression=saved_regression(result,smoke_saved[cid]["modes"]["NEW"])
                else:
                    regression={"saved_regression_not_available_for_short_config":True,
                                "reason":"k1 saved diagnostic uses10 steps; smoke only2; old ControlledTTA is directly reused"}
            else:
                ref=data["SEQUENTIAL_NEW"]
                if any(result[k]!=ref[k] for k in ("random_trace","noise_sha256","rng_end_sha256","fixed_sequence")):
                    raise ValueError("controlled replay mismatch")
                if plan and (result["states"][1]!=ref["states"][1] or result["rounds"][0]["post_projection_image_sha256"]!=ref["rounds"][0]["post_projection_image_sha256"]):
                    save(out/"round1_failure.json",{"caption_id":cid,"sequential":ref,"prefix":result})
                    raise ValueError("round1/k1 not bitwise exact")
            budget_and_prefix_checks(mode,result,plan,IMAGE_EPSILON)
            data[mode]=result;round_inputs[mode]=inputs_at_round
            print("MODE_DONE",cid,mode,"k",len(plan),"delta_F",result["clean_to_final"]["paired_soft_rank_F"],flush=True)
        # 在两mode全部更新结束后补算term diagnostics，避免改变baseline的GPU工作区。
        for mode in MODES:
            attach_decompositions(source,data[mode],round_inputs[mode],cached[mode],vectors)
        if metadata!=[a.metadata for a in plan] or any((a.w._version,a.Q._version)!=v for a,v in zip(plan,tensor_versions)):
            raise ValueError("frozen instances changed")
        if vector_hashes!={aid:__import__("controlled_tta").tensor_hash(v) for aid,v in vectors.items()}:
            raise ValueError("JVP vectors changed")
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):
            raise ValueError("IRRA not frozen")
        result={"caption_id":cid,"caption":record["caption"],"paired_image":record["paired_image"],
                "candidate_count":plans[cid]["candidate_count"],"k_star":len(plan),"selected_attributes":metadata,
                "JVP_mapping":mapping,"JVP_sha256":vector_hashes,"JVP_computations":len(plan),"modes":data,
                "checks":{"zero_exact":True,"fixed_plan_exact":True,"metadata_exact":True,"IRRA_frozen":True,
                          "controlled_noise_trace_rng_exact":True,"round1_exact":True,"round1_execution":"common original single-attribute execution; full state checkpoint fork", "caption_loss_terms":0,
                          "sequential_saved_regression":regression},"case_seconds":time.time()-t0}
        results.append(result);save(out/"results.json",results)
        print("CASE_DONE",cid,"k",len(plan),"seconds",result["case_seconds"],flush=True)
    old.verify(hashes);old.verify(own)
    aggregated=summarize(results)
    diff=subprocess.check_output(["git","diff","--stat"],cwd=REPO,text=True)
    staged=subprocess.check_output(["git","diff","--cached","--stat"],cwd=REPO,text=True)
    if diff or staged:raise ValueError("formal tracked source changed")
    audit={"case_count":len(results),"modes":list(MODES),"attribute_rounds_per_mode":sum(c["k_star"] for c in results),
           "steps_per_round":steps,"controlled_noise_trace_rng_exact":True,"round1_k1_exact":True,
           "IRRA_frozen":True,"AttributeInstance_score_selection_JVP_unchanged":True,"caption_loss_terms":0,
           "source_checkpoint_cache_SHA_unchanged":True,"tracked_diff":diff,"staged_diff":staged,
           "no_training":True,"no_feature_extraction":True,"no_formal_gallery_attack":True,
           "diagnostic_run":args.mode=="diagnostic","seconds":time.time()-started}
    save(out/"summary.json",aggregated);save(out/"audit.json",audit)
    lines=["# Step7.3.2 Prefix-preserving诊断","","只有loss scheduler变化；smoke不能用于冻结scheduler。","",
           "| caption | k | mode | Δcosine | ΔF | Δrank | best round | F best-final |",
           "|---|---:|---|---:|---:|---:|---:|---:|"]
    for c in results:
        for m in MODES:
            x=c["modes"][m];d=x["clean_to_final"];b=best_prefix(x)
            lines.append(f"| {c['caption_id']} | {c['k_star']} | {m} | {d['paired_cosine']:.7g} | {d['paired_soft_rank_F']:.7g} | {d['positive_image_rank']} | {b['best_round']} | {b['F_best_minus_final']:.7g} |")
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(out/"completed.json",audit)
    print("COMPLETE",json.dumps(audit),flush=True)
if __name__=="__main__":main()
