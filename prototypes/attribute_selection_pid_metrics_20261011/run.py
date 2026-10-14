"""原固定100句PID-mAP补算；不攻击、不backward、不重新评分。默认仅CPU preflight。"""
import os
import sys
from pathlib import Path
sys.dont_write_bytecode = True
for key, sub in [("TORCH_HOME", "torch"), ("HF_HOME", "huggingface"), ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda")]:
    os.environ.setdefault(key, "/home/lzf/ldx/cache/" + sub)
import argparse
import csv
import importlib.util
import json
import time
import torch
from inputs import ROOT, REPO, SOURCE, checked_inputs, verify, write_json, load
from pid_metrics import MODES, replace_paired, check_saved, pid_row, drop_rate, aggregate_rows, aggregate_victims
from aggregate import markdown

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight", "smoke", "diagnostic"], default="preflight")
    p.add_argument("--allow-diagnostic", action="store_true")
    p.add_argument("--output", type=Path)
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode == "diagnostic" and not args.allow_diagnostic:
        raise SystemExit("用户手动100句补算需 --allow-diagnostic")
    cases, features, pids, spec, locked, contract, source_audit = checked_inputs()
    if args.mode == "preflight":
        print("PREFLIGHT_COMPLETE", json.dumps(source_audit, ensure_ascii=False), flush=True)
        return
    if args.mode == "diagnostic":
        smoke = load(ROOT / "outputs/smoke/completed.json")
        if not smoke["method_audit_passed"] or smoke["case_count"] != 2:
            raise ValueError("successful2-query metric smoke required")
        if load(ROOT / "outputs/smoke/protocol.json")["frozen_sha256"] != locked:
            raise ValueError("sources changed since smoke")
    out = (args.output or ROOT / "outputs" / args.mode).resolve()
    if not out.is_relative_to(Path("/home/lzf/ldx").resolve()):
        raise ValueError("output must remain under /home/lzf/ldx")
    out.mkdir(parents=True, exist_ok=False)
    if args.mode == "smoke":
        cases = [cases[0], next(c for c in cases if c["natural_k"] == 6)]
        contract = dict(contract, query_count=2, query_set_sha256=__import__("inputs").canonical_sha(
            [[c["caption_id"], c["caption"], c["pid"]] for c in cases]))
    protocol = {"scope": args.mode, "source_diagnostic": str(SOURCE),
        "aggregation_contract": contract, "frozen_sha256": locked,
        "precision": "same frozen fp32 singleton text encoder and cached fp32 gallery; TF32 off",
        "attack_forward_calls": 0, "image_encoder_calls": 0, "score_backwards": 0,
        "attack_backwards": 0, "new_training": False, "attack_settings_changed": False,
        "ranking": "existing text_to_image_metrics; torch.argsort descending, PID relevance",
        "round_scope": "clean and final only", "shared_adversarial_gallery": False,
        "mDR_definition": "(aAP_clean-aAP_adversarial)/aAP_clean*100"}
    write_json(out / "protocol.json", protocol)
    write_json(out / "gallery_audit.json", source_audit)
    # 复用原fp32加载器，不导入或调用任何TTA/scorer/JVP执行入口。
    path = REPO / "prototypes/attribute_unit_v4_learned_reader_20261003/learned_reader.py"
    module_spec = importlib.util.spec_from_file_location("pid_metrics_frozen_irra_loader", path)
    loader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loader)
    loader.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim = loader.load_victim()
    versions = {n: p._version for n, p in victim.named_parameters()}
    features, pids = features.cuda(), pids.cuda()
    clean_rows, adv_rows, checks = [], {m: [] for m in MODES}, []
    started = time.time()
    # 此作用域禁止autograd；不重建原攻击样本，只有每句一次clean text forward。
    with torch.no_grad(), (out / "per_query_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            text = victim.encode_texts([case["caption"]])
            cached_scores = (text @ features.T).flatten()
            paired = case["paired_index"]
            clean_scores = replace_paired(cached_scores, paired, case["clean"]["paired_cosine"])
            check = {"caption_id": case["caption_id"], "clean": check_saved(clean_scores, paired, case["clean"]), "modes": {}}
            clean = pid_row(clean_scores, case["pid"], pids)
            clean_rows.append(clean)
            result = {"caption_id": case["caption_id"], "pid": case["pid"], "K": case["natural_k"],
                "paired_path": case["paired_path"], "clean": clean, "adversarial": {}}
            for mode in MODES:
                scores = replace_paired(cached_scores, paired, case["final"][mode]["paired_cosine"])
                check["modes"][mode] = check_saved(scores, paired, case["final"][mode])
                current = pid_row(scores, case["pid"], pids)
                adv_rows[mode].append(current)
                result["adversarial"][mode] = current
            result["reconstruction_checks"] = check
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            checks.append(check)
            print("QUERY_DONE", case["caption_id"], flush=True)
    if any(p.requires_grad or p.grad is not None or p._version != versions[n] for n,p in victim.named_parameters()):
        raise ValueError("IRRA frozen-state failure")
    verify(locked)
    report = {"schema": "tbps-pid-map-victim-report-1",
        "victim_id": "IRRA:" + spec["checkpoint_sha256"],
        "victim_name": "IRRA", "checkpoint_sha256": spec["checkpoint_sha256"],
        "aggregation_contract": contract, "metric_unit": "percent",
        "clean": aggregate_rows(clean_rows), "adversarial": {},
        "single_model_drop_percent": {}, "case_count": len(cases),
        "shared_adversarial_gallery": False, "formal_test_evaluation": False}
    for mode in MODES:
        report["adversarial"][mode] = aggregate_rows(adv_rows[mode])
        report["single_model_drop_percent"][mode] = drop_rate(report["clean"]["mAP"], report["adversarial"][mode]["mAP"])
    aggregated = aggregate_victims([report])
    write_json(out / "victim_summary.json", report)
    write_json(out / "summary.json", aggregated)
    write_json(out / "reconstruction_checks.json", checks)
    (out / "summary.md").write_text(markdown(aggregated), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method","victim_count","IRRA_clean_mAP_percent","IRRA_adv_mAP_percent","single_model_DR_percent","aAP_clean_percent","aAP_adv_percent","mDR_percent","protocol"])
        for mode in MODES:
            a = aggregated["modes"][mode]
            writer.writerow([mode,1,report["clean"]["mAP"],report["adversarial"][mode]["mAP"],
                report["single_model_drop_percent"][mode],aggregated["aAP_clean"],a["aAP_adversarial"],a["mDR"],contract["protocol_kind"]])
    errors = [v["soft_rank_abs_error"] for c in checks for v in [c["clean"],*c["modes"].values()]]
    completed = {"method_audit_passed": True, "case_count": len(cases),
        "scope": args.mode, "IRRA_frozen": True, "source_SHA_unchanged": True,
        "score_backwards": 0, "attack_backwards": 0, "image_encoder_calls": 0,
        "clean_text_forward_calls": len(cases), "paired_rank_and_ties_exact": True,
        "maximum_soft_rank_reconstruction_abs_error": max(errors),
        "shared_adversarial_gallery": False, "formal_test_evaluation": False,
        "seconds": time.time()-started}
    write_json(out / "completed.json", completed)
    print(markdown(aggregated), flush=True)
    print("COMPLETE", json.dumps(completed), flush=True)
if __name__ == "__main__":
    main()
