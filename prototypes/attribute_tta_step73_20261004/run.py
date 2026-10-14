"""Step7.3：默认只运行固定2句短smoke；完整25句需用户显式--allow-diagnostic。"""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key, sub in [("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda")]:
    os.environ.setdefault(key, "/home/lzf/ldx/cache/"+sub)
REPO = Path("/home/lzf/ldx/projects/idea-TBPS-test1")
ROOT = Path(__file__).resolve().parent
P72 = REPO/"prototypes/attribute_tta_step72_20261003"
sys.path.insert(0, str(REPO/"src"))
sys.path.insert(0, str(P72))
import argparse
import collections
import copy
import importlib.util
import itertools
import json
import subprocess
import time
import numpy as np
import torch
from PIL import Image
from attribute_tta import prepared_plan, TextTransport, AttributeTTA, seed_all, digest, OFFICIAL, IMAGE_EPSILON
from attributes.irra_surrogate import IRRASurrogate
from controlled_tta import ControlledTTA, CategoricalTape, RetrievalMetrics, MODES, DEFINITIONS, guidance, metric_delta, rng_fingerprint, cosine
GALLERY = Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-v2-smoke-20261001/clean-train-gallery.pt")
IRRA_ROOT = Path("/home/lzf/Attack/IRRA-main/IRRA-main")
SMOKE_IDS = ("cuhk:35:1", "cuhk:0:1")  # 承接旧smoke，覆盖sleeve、跨region同色；不按效果挑选。

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
def verify(hashes):
    for p, expected in hashes.items():
        if digest(p) != expected:
            raise ValueError("frozen SHA mismatch: "+p)
def stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    x = np.array(values, dtype=float)
    return {"count": len(x), "mean": float(x.mean()), "median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "diagnostic"), default="smoke")
    p.add_argument("--allow-diagnostic", action="store_true")
    p.add_argument("--include-legacy", action="store_true", help="额外caption+raw基线，不能作为严格attribute-only对照")
    p.add_argument("--output", type=Path)
    return p

def checked_inputs():
    lock = load(ROOT/"provenance.lock.json")
    hashes = lock["frozen_sha256"]
    verify(hashes)
    spec = importlib.util.spec_from_file_location("step72_runner", P72/"run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, rows, payloads, bases = module.inputs_and_records()
    prior = load(P72/"outputs/diagnostic/protocol.json")
    completed = load(P72/"outputs/diagnostic/completed.json")
    if completed["case_count"] != 25 or [r["row_id"] for r in rows] != prior["row_ids"]:
        raise ValueError("not the completed fixed25 diagnostic population")
    if prior["steps"] != 10 or prior["transforms_per_scale"] != 6 or prior["gamma"] != 3:
        raise ValueError("unexpected frozen Step72 configuration")
    return hashes, rows, payloads, bases, prior

def load_gallery(hashes):
    g = torch.load(GALLERY, map_location="cpu", weights_only=False)
    spec = g["spec"]
    if g["features"].shape != (34054, 512) or spec["split"] != "train":
        raise ValueError("fixed clean train gallery required")
    if spec["torch_version"] != str(torch.__version__):
        raise ValueError("gallery torch version mismatch")
    if spec["annotation_sha256"] != hashes["/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json"]:
        raise ValueError("gallery annotation mismatch")
    if spec["checkpoint_sha256"] != hashes["/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth"]:
        raise ValueError("gallery checkpoint mismatch")
    configs = [h for p,h in hashes.items() if p.endswith("irra_cuhk_official.yaml")]
    if not configs or any(h != spec["config_sha256"] for h in configs):
        raise ValueError("gallery config mismatch")
    for rel,h in spec["source_sha256"].items():
        if digest(IRRA_ROOT/rel) != h:
            raise ValueError("gallery official encoder source mismatch: "+rel)
    if not torch.isfinite(g["features"]).all() or not torch.allclose(g["features"].norm(dim=1), torch.ones(34054), atol=2e-4, rtol=0):
        raise ValueError("invalid normalized gallery features")
    paths = spec["gallery_paths"]
    if len(paths) != 34054 or len(set(paths)) != 34054:
        raise ValueError("gallery path index invalid")
    audit = copy.deepcopy(spec)
    audit.pop("gallery_paths")
    audit["gallery_paths_sha256"] = __import__("hashlib").sha256(json.dumps(paths).encode()).hexdigest()
    return g["features"].cuda(), {p:i for i,p in enumerate(paths)}, audit

def summary(results, modes):
    out = {"case_count": len(results), "nonempty_case_count": sum(r["k_star"]>0 for r in results), "modes": {}, "comparisons": {}}
    for mode in modes:
        cases = [r["modes"][mode] for r in results]
        rounds = [x for c in cases for x in c["rounds"]]
        step_events = [x for r in rounds for x in r["steps"]]
        out["modes"][mode] = {
            "clean_to_final_F_increase": stats([c["clean_to_final"]["paired_soft_rank_F"] for c in cases]),
            "clean_to_final_cosine_drop": stats([-c["clean_to_final"]["paired_cosine"] for c in cases]),
            "clean_to_final_positive_rank_increase": stats([c["clean_to_final"]["positive_image_rank"] for c in cases]),
            "per_case": [{"caption_id": r["caption_id"], "k":r["k_star"], **r["modes"][mode]["clean_to_final"]} for r in results],
            "rounds": {}, "axes": {},
            "exact_zero_round_updates": sum(r["actual_change_exact_zero"] for r in rounds),
            "round_update_linf_le_1e_8": sum(r["actual_change_linf_le_1e_8"] for r in rounds),
            "post_step_actual_change_l2": stats([s["post_projection_actual_change"]["l2"] for s in step_events]),
            "projection_clipping_ratio": stats([s["projection_clipping_ratio"] for s in step_events]),
            "steps_more_than_half_coordinates_clipped": sum(s["projection_clipping_ratio"]>.5 for s in step_events),
            "projection_removed_l2_over_requested_l2": stats([s["projection_removed_l2_over_requested_l2"] for s in step_events if s["projection_removed_l2_over_requested_l2"] is not None]),
            "budget_linf_max": max([r["cumulative_perturbation_linf"] for r in rounds], default=0),
        }
        for label, field in [("rounds", "round_index"), ("axes", "semantic_axis")]:
            for key in sorted({r[field] for r in rounds}, key=str):
                xs = [r for r in rounds if r[field]==key]
                out["modes"][mode][label][str(key)] = {
                    "count": len(xs),
                    "delta_F": stats([r["metric_delta"]["paired_soft_rank_F"] for r in xs]),
                    "delta_cosine": stats([r["metric_delta"]["paired_cosine"] for r in xs]),
                    "F_worsened_rounds": sum(r["metric_delta"]["paired_soft_rank_F"]>0 for r in xs),
                    "F_restored_rounds": sum(r["metric_delta"]["paired_soft_rank_F"]<0 for r in xs),
                    "F_unchanged_rounds": sum(r["metric_delta"]["paired_soft_rank_F"]==0 for r in xs),
                }
    for other in [m for m in modes if m != "NEW"]:
        xs = []
        for r in results:
            a,b = r["modes"]["NEW"]["clean_to_final"], r["modes"][other]["clean_to_final"]
            xs.append({"caption_id": r["caption_id"], "k": r["k_star"],
                       "NEW_minus_other_F_degradation": a["paired_soft_rank_F"]-b["paired_soft_rank_F"],
                       "NEW_minus_other_cosine_degradation": b["paired_cosine"]-a["paired_cosine"]})
        out["comparisons"]["NEW_vs_"+other] = {
            "per_case": xs, "tie_rule": "exact saved fp32 metric equality; no performance threshold",
            "all25": {key: {"NEW_wins":sum(x[key]>0 for x in xs), "other_wins":sum(x[key]<0 for x in xs), "ties":sum(x[key]==0 for x in xs), "difference":stats([x[key] for x in xs])}
                      for key in ("NEW_minus_other_F_degradation","NEW_minus_other_cosine_degradation")},
            "nonempty": {key: {"NEW_wins":sum(x[key]>0 for x in xs if x["k"]>0), "other_wins":sum(x[key]<0 for x in xs if x["k"]>0), "ties":sum(x[key]==0 for x in xs if x["k"]>0)}
                         for key in ("NEW_minus_other_F_degradation","NEW_minus_other_cosine_degradation")},
        }
    out["interpretation"] = "固定25句机制诊断，不是共享adversarial gallery或正式PID TBPS性能；round/axis统计混合累计状态，不证明语义纯度。NONE是caption-only，并非zero/no-op。"
    return out

def markdown(out, results, aggregated, audit):
    lines = ["# Step7.3 受控属性引导TTA机制诊断", "", "NONE=原caption-only TTA；RAW=selected原始span；NEW=原text-forward JVP。完整caption只作为NEW的线性化与评价context，未加入NEW监督。", "", "NEW真实categorical变换序列供RAW/NONE回放；各模式保留原loss/概率权重更新、初始化、动量及投影。没有正式共享图库attack性能结论。", "", "## 审计", "", "```json", json.dumps(audit, ensure_ascii=False, indent=2), "```", "", "## 每句 clean→final", "", "| caption | k | mode | Δpaired cosine | ΔF（正=恶化） | Δpaired rank |", "|---|---:|---|---:|---:|---:|"]
    for r in results:
        for mode, data in r["modes"].items():
            d = data["clean_to_final"]
            lines.append(f"| {r['caption_id']} | {r['k_star']} | {mode} | {d['paired_cosine']:.8g} | {d['paired_soft_rank_F']:.8g} | {d['positive_image_rank']} |")
    lines += ["", "均值/中位数、逐句wins/ties、每轮边际变化、按axis结果及projection饱和统计见summary.json；完整更新记录与逐状态metrics见results.json。", "", "LEGACY如启用，比NEW/RAW多full-caption项，不能混作严格controlled优势。没有根据结果修改S/Top-K/reader/basis。"]
    (out/"summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode=="diagnostic" and not args.allow_diagnostic:
        raise SystemExit("fixed25 is user-run only; supply --allow-diagnostic")
    out = (args.output or ROOT/"outputs"/args.mode).resolve()
    if not out.is_relative_to(Path("/home/lzf/ldx")) or out.exists():
        raise ValueError("new output directory under /home/lzf/ldx required; no overwrite")
    hashes, rows, payloads, bases, prior = checked_inputs()
    ids = SMOKE_IDS if args.mode=="smoke" else tuple(prior["row_ids"])
    steps = 2 if args.mode=="smoke" else prior["steps"]
    transforms = 1 if args.mode=="smoke" else prior["transforms_per_scale"]
    scales = None if args.mode=="smoke" else tuple(prior["scales"])
    modes = MODES + (("LEGACY",) if args.include_legacy else ())
    if len(ids)>25 or (args.mode=="smoke" and len(ids)>2):
        raise ValueError("execution boundary exceeded")
    out.mkdir(parents=True)
    started = time.time()
    sys.path.insert(0, str(REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
    import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim = v4.load_victim()
    source = IRRASurrogate(victim)
    versions = {n:p._version for n,p in victim.named_parameters()}
    features, path_index, gallery_audit = load_gallery(hashes)
    own_sources = {str(p):digest(p) for p in ROOT.glob("*.py")}
    protocol = {
        "mode":args.mode, "row_ids":list(ids), "modes":list(modes), "definitions":DEFINITIONS,
        "steps_per_attribute":steps, "transforms_per_scale":transforms, "scales":scales,
        "seed":42, "gamma":3, "beta":1, "k_max":4, "epsilon":IMAGE_EPSILON, "step_size":2/255, "momentum":.9,
        "selection":"saved clean plan exactly once; no scoring, sorting or reselection",
        "random_control":"NEW unmodified realized categorical block/op choices recorded; RAW/NONE/optional LEGACY replay; verify noise SHA and end RNG fingerprints",
        "transform_probability_updates":"official adaptive probabilities/weights stay mode-specific; realized transform IDs are common. NEW anchor is fixed in advance, not chosen by outcome.",
        "gallery":gallery_audit, "gallery_count":34054, "tau":.02,
        "retrieval_metrics":"paired cosine / original paired soft rank against fixed clean competitors / paired-image rank (strict greater, ties separately); mechanism only",
        "projection_logging":"round pre-update=last unprojected proposal minus round input; actual=final minus round input; also every optimization step. Removed L2/requested L2 can exceed1 when original noisy internal point exceeds epsilon.",
        "almost_zero_diagnostic":"actual_change_linf <= 1e-8, numerical logging only; exact-zero separately",
        "no_formal_gallery_evaluation":True, "no_adversarial_dataset":True, "no_training":True,
        "frozen_sha256":hashes, "source_sha256":own_sources,
    }
    save(out/"protocol.json", protocol)
    lookup = {r["row_id"]:r for r in rows}
    plans = {p["caption_id"]:p for p in payloads}
    results = []
    for cid in ids:
        t0 = time.time()
        record = lookup[cid]
        selected = prepared_plan(plans[cid], record, bases, "cuda")
        metadata = copy.deepcopy([a.metadata for a in selected])
        tensor_versions = [(a.w._version,a.Q._version) for a in selected]
        paired = path_index[record["paired_image"]]
        if paired != record["paired_gallery_index"]:
            raise ValueError("paired gallery mapping changed")
        image_path = (Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/record["paired_image"]).resolve()
        if not image_path.is_relative_to(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")):
            raise ValueError("invalid image source path")
        with Image.open(image_path) as im:
            image = victim.pixels(im).unsqueeze(0).cuda()
        transport = TextTransport(victim, record["caption"])
        new_vectors, mapping = {}, []
        for a in selected:
            vector, report = transport.read(a)
            if report["central_fd_relative_error"]>.02:
                raise ValueError("JVP verification failed")
            new_vectors[a.instance_id] = vector
            mapping.append(report)
        with torch.no_grad():
            raw_list = [a.metadata["raw_text"] for a in selected]
            encoded = source.encode_texts(raw_list).detach() if raw_list else None
        raw_features = {a.instance_id:encoded[i:i+1] for i,a in enumerate(selected)}
        evaluator = RetrievalMetrics(source, transport.clean, features, paired, tau=.02)
        clean_metric = evaluator(image)
        mode_results, reference_tape, reference_noise, reference_rng = {}, None, None, None
        new_final = None
        new_momentum = None
        for mode in modes:
            seed_all(42)
            tape = CategoricalTape() if mode=="NEW" else CategoricalTape(reference_tape)
            engine = ControlledTTA(source, image, mode, steps=steps, transforms=transforms, scales=scales, tape=tape)
            current = image.clone()
            states, previous_metric, actual_changes = [dict(state_index=0, **clean_metric)], clean_metric, []
            for a in selected:
                guidance_features, labels = guidance(mode, a, new_vectors, raw_features, transport.clean)
                before = current
                current = engine.step(current, a, guidance_features, labels)
                actual_changes.append((a, (current-before).detach().clone()))
                metric = evaluator(current)
                engine.events[-1]["metrics_before"] = previous_metric
                engine.events[-1]["metrics_after"] = metric
                engine.events[-1]["metric_delta"] = metric_delta(previous_metric,metric)
                states.append(dict(state_index=a.selection_rank, **metric))
                previous_metric = metric
            trace = tape.finish()
            noise = [e["initialization"]["noise_sha256"] for e in engine.events]
            fingerprint = rng_fingerprint()
            if mode=="NEW":
                reference_tape, reference_noise, reference_rng = copy.deepcopy(tape.records), noise, fingerprint
                new_final = current.detach().clone()
                new_momentum = engine.momentum.detach().clone()
            elif noise != reference_noise or fingerprint != reference_rng or tape.records != reference_tape:
                raise ValueError("controlled initialization/transformation RNG mismatch: "+cid+" "+mode)
            update_pairs = []
            for (a, da), (b, db) in itertools.combinations(actual_changes, 2):
                update_pairs.append({
                    "instance_a":a.instance_id,"instance_b":b.instance_id,
                    "axis_a":a.metadata["semantic_axis"],"axis_b":b.metadata["semantic_axis"],
                    "region_a":a.metadata["region"],"region_b":b.metadata["region"],
                    "actual_round_update_cosine":cosine(da,db),
                    "caveat":"不同累计状态下的round净更新，不能当同一图像点的纯语义梯度比较。",
                })
            sequence = [e["instance_id"] for e in engine.events]
            if sequence != [a.instance_id for a in selected] or any(e["caption_loss_terms"]!=0 for e in engine.events if mode in ("NEW","RAW")):
                raise ValueError("fixed selection or attribute-only loss violated")
            if any(s["cumulative_perturbation_linf"]>IMAGE_EPSILON+1e-6 for e in engine.events for s in e["steps"]):
                raise ValueError("step budget exceeded")
            mode_results[mode] = {
                "states":states,"rounds":engine.events,"clean_to_final":metric_delta(clean_metric,previous_metric),
                "actual_round_update_pairs":update_pairs,"random_trace":trace,"noise_sha256":noise,
                "rng_end_sha256":fingerprint,"fixed_sequence":sequence,"original_image_and_initialization_noise_exact":True,
                "final_linf":float((current-image).abs().max()),
            }
            print("MODE_DONE",cid,mode,"k",len(selected),"delta_F",mode_results[mode]["clean_to_final"]["paired_soft_rank_F"], flush=True)
        regression = None
        if args.mode=="smoke":
            # 同一2句短smoke内验证observer/replay的NEW未改变原Step72输出；不增加样本。
            seed_all(42)
            original_engine = AttributeTTA(source, image, steps=steps, transforms=transforms, scales=scales)
            unobserved = image.clone()
            round_images_exact = []
            for a, event in zip(selected,mode_results["NEW"]["rounds"]):
                unobserved=original_engine.step(unobserved,a,new_vectors[a.instance_id])
                from controlled_tta import tensor_hash
                round_images_exact.append(tensor_hash(unobserved)==event["post_projection_image_sha256"])
            regression = {
                "original_step72_NEW_image_bitwise_exact":torch.equal(new_final,unobserved),
                "original_step72_NEW_all_round_images_bitwise_exact":all(round_images_exact),
                "original_step72_NEW_momentum_bitwise_exact":torch.equal(new_momentum,original_engine.momentum),
                "original_step72_NEW_rng_exact":rng_fingerprint()==reference_rng,
                "original_step72_NEW_momentum_allclose_existing_fp32_tolerance":torch.allclose(new_momentum,original_engine.momentum,atol=2e-6,rtol=1e-6),
                "momentum_max_abs_difference":float((new_momentum-original_engine.momentum).abs().max()),
                "momentum_tolerance":{"atol":2e-6,"rtol":1e-6,"source":"Step7.2既有fp32回归容差"},
            }
            print("SMOKE_REGRESSION", cid, regression, "image_maxdiff", float((new_final-unobserved).abs().max()), "momentum_maxdiff", float((new_momentum-original_engine.momentum).abs().max()), flush=True)
            if not regression["original_step72_NEW_momentum_allclose_existing_fp32_tolerance"]:
                seed_all(42)
                repeat_engine=AttributeTTA(source,image,steps=steps,transforms=transforms,scales=scales)
                repeat_image=repeat_engine.run(selected,new_vectors)
                regression["plain_repeat_image_bitwise_exact"]=torch.equal(repeat_image,unobserved)
                regression["plain_repeat_momentum_max_abs_difference"]=float((repeat_engine.momentum-original_engine.momentum).abs().max())
                regression["plain_repeat_momentum_existing_allclose"]=torch.allclose(repeat_engine.momentum,original_engine.momentum,atol=2e-6,rtol=1e-6)
                print("PLAIN_REPEAT_NUMERICS",cid,regression,flush=True)
            if not (regression["original_step72_NEW_image_bitwise_exact"] and regression["original_step72_NEW_all_round_images_bitwise_exact"] and regression["original_step72_NEW_rng_exact"] and regression["original_step72_NEW_momentum_allclose_existing_fp32_tolerance"]):
                raise ValueError("observer changed official NEW attack")
        if metadata != [a.metadata for a in selected] or any((a.w._version,a.Q._version)!=v for a,v in zip(selected,tensor_versions)):
            raise ValueError("frozen attribute metadata/tensor changed")
        if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):
            raise ValueError("IRRA is not frozen")
        result = {
            "caption_id":cid,"caption":record["caption"],"paired_image":record["paired_image"],
            "candidate_count":plans[cid]["candidate_count"],"k_star":len(selected),
            "selected_attributes":[copy.deepcopy(a.metadata) for a in selected],"JVP_mapping":mapping,
            "modes":mode_results,"checks":{"zero_exact":True,"fixed_plan_exact":True,"metadata_exact":True,"IRRA_frozen":True,"all_modes_random_trace_exact":True,"NEW_RAW_no_caption_guidance":True,"smoke_regression":regression},
            "case_seconds":time.time()-t0,
        }
        results.append(result)
        save(out/"results.json", results)
        print("CASE_DONE",cid,"k",len(selected),"seconds",result["case_seconds"],flush=True)
    verify(hashes)
    verify(own_sources)
    aggregated = summary(results,modes)
    audit = {
        "case_count":len(results),"modes":list(modes),"attribute_rounds_per_mode":sum(r["k_star"] for r in results),
        "IRRA_frozen":True,"AttributeInstance_S_TopK_basis_unchanged":True,"source_checkpoint_cache_SHA_unchanged":True,
        "zero_exact":True,"controlled_trace_noise_rng_exact":True,"NEW_RAW_caption_terms":0,
        "no_training":True,"no_feature_extraction":True,"no_formal_gallery_attack":True,"no_adversarial_dataset":True,
        "tracked_diff":subprocess.check_output(["git","diff","--stat"],cwd=REPO,text=True),
        "staged_diff":subprocess.check_output(["git","diff","--cached","--stat"],cwd=REPO,text=True),
        "seconds":time.time()-started,
    }
    if audit["tracked_diff"] or audit["staged_diff"]:
        raise ValueError("unexpected formal source diff")
    save(out/"summary.json",aggregated)
    save(out/"audit.json",audit)
    markdown(out,results,aggregated,audit)
    save(out/"completed.json",audit)
    print("COMPLETE",json.dumps(audit),flush=True)

if __name__=="__main__":
    main()
