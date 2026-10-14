"""Phase 1默认CPU只读preflight；显式smoke最多固定5条，不提供full入口。"""
import argparse, copy, json, time, sys
from pathlib import Path
from unittest.mock import patch
import torch
import cache_io as io
from condition_schema import IDENTITY, check_identity, error, validate
CONFIG = io.ROOT/"config.json"
RAW = Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGES = RAW.parent/"imgs"

def own_hashes():
    return {str(p):io.sha(p) for p in [*sorted(io.ROOT.glob("*.py")),CONFIG]}

def audit_source():
    cfg = io.load(CONFIG); root = Path(cfg["source_cache"]); m = io.load(root/"manifest.json")
    done = io.load(root/"completed.json")
    io.verify({str(root/"manifest.json"):done["manifest_sha256"],
               m["protocol_path"]:m["protocol_sha256"]})
    if not done["method_audit_passed"] or m["scope"] != "prepared" or m["test_used"]:
        raise ValueError("completed original train cache required")
    locks = dict(m["frozen_sha256"]); io.verify(locks)
    v1 = root.parent.parent
    locks.update({str(p):io.sha(p) for p in [*sorted(v1.glob("*.py")),v1/"config.json"]})
    for name in ("manifest.json","completed.json","protocol.json"): locks[str(root/name)] = io.sha(root/name)
    raw = io.load(RAW)
    split_path = v1.parent/"attribute_unit_v4_learned_reader_20261003/outputs/split.json"
    split = io.load(split_path)
    memberships = {role:{split["records"][i]["row_id"] for i in split["train_indices" if role=="train" else "validation_indices"]}
                   for role in ("train","validation")}
    roles = {}; all_ids = set(); pids = {}
    for role in ("train","validation"):
        if len(m[role]) != m["counts"][role]: raise ValueError("source count mismatch")
        roles[role] = []; pids[role] = set()
        for e in m[role]:
            if e["row_id"] in all_ids or e["row_id"] not in memberships[role]:
                raise ValueError("source duplicate/split membership")
            all_ids.add(e["row_id"]); pids[role].add(e["pid"])
            io.verify({e["tensor_path"]:e["tensor_sha256"],e["metadata_path"]:e["metadata_sha256"]})
            md = io.load(e["metadata_path"]); r = md["score_record"]; p = md["payload"]
            identity = {"row_id":e["row_id"],"image_rel_path":e["image"],
                "caption_index":int(e["row_id"].split(":")[-1]),"split":"train",
                "role":role,"pid":e["pid"],"caption":e["caption"]}
            check_identity(identity)
            annotation = raw[int(e["row_id"].split(":")[1])]
            if (annotation["split"],annotation["id"],annotation["file_path"],
                annotation["captions"][identity["caption_index"]]) != ("train",e["pid"],e["image"],e["caption"]):
                raise ValueError("raw annotation identity mismatch")
            if (md["row_id"],md["pid"],md["caption"],md["image"],md["role"]) != (
                e["row_id"],e["pid"],e["caption"],e["image"],role):
                raise ValueError("source metadata identity")
            if (r["row_id"],r["pid"],r["caption"],r["paired_image"]) != (
                e["row_id"],e["pid"],e["caption"],e["image"]):
                raise ValueError("source score identity")
            if r["retrieval_backward_count"] != 1 or not r["zero_intervention_exact"]:
                raise ValueError("source scoring contract")
            chosen = p["selected_attributes"]
            if (p["gamma"] != 3 or p["beta"] != 1 or p["caption_id"] != e["row_id"]
                or not e["K"] == md["K"] == p["natural_k"] == p["k_star"] == len(chosen)
                or chosen != p["ranked_attributes"][:e["K"]]
                or md["selected_instance_ids"] != [a["instance_id"] for a in chosen]):
                raise ValueError("source frozen selection mismatch")
            lookup = {a["instance_id"]:a for a in r["attributes"]}
            for a in chosen:
                if a != {**lookup[a["instance_id"]]["instance"],"S_raw":lookup[a["instance_id"]]["S_raw"]}:
                    raise ValueError("source score/selection mismatch")
            value = torch.load(e["tensor_path"],map_location="cpu",weights_only=True)
            for key in ("z_t","c_attr","v_local"):
                x = value[key]
                if x.shape != (512,) or x.dtype != torch.float32 or x.requires_grad or not torch.isfinite(x).all():
                    raise ValueError("source frozen tensor contract")
            if abs(float(value["z_t"].norm())-1)>2e-4 or value["has_local"] != (e["K"]>0):
                raise ValueError("source z/K contract")
            scores = torch.tensor([a["S_raw"] for a in chosen],dtype=torch.float64)
            expected_weights = (scores/scores.sum()).tolist() if len(scores) else []
            if any(abs(a-b)>1e-15 for a,b in zip(expected_weights,md["score_normalized_weights"])) or len(expected_weights)!=len(md["score_normalized_weights"]):
                raise ValueError("source weights mismatch")
            if e["K"]==0 and (md["selected_instance_ids"] or md["score_normalized_weights"]
                or value["v_local"].any() or value["c_attr"].any()): raise ValueError("source K0 mismatch")
            locks[e["tensor_path"]] = e["tensor_sha256"]; locks[e["metadata_path"]] = e["metadata_sha256"]
            roles[role].append((e,md,identity))
    if pids["train"] & pids["validation"]: raise ValueError("source PID leakage")
    picks = []
    for role, ks in (("train",cfg["smoke_train_K"]),("validation",cfg["smoke_validation_K"])):
        for k in ks:
            candidates = [x for x in roles[role] if x[0]["K"] == k]
            if not candidates: raise ValueError("missing predetermined smoke K")
            picks.append(candidates[0]) # manifest顺序固定；不按新实验结果挑选。
    for e,md,identity in picks:
        path = IMAGES/e["image"]
        if io.sha(path) != e["image_sha256"]: raise ValueError("source image SHA")
        locks[str(path)] = e["image_sha256"]
    return cfg, m, locks, picks

def preflight():
    cfg, m, locks, picks = audit_source()
    result = {"mode":"CPU_PREFLIGHT","source_counts":m["counts"],"source_sha_count":len(locks),
        "smoke_samples":[{**i,"K":e["K"]} for e,md,i in picks],
        "new_scoring_backwards_this_run":0,"JVPs_this_run":0,
        "cuda_initialized":torch.cuda.is_initialized(),"z_t_usage":cfg["z_t_usage"]}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

def run_smoke(output):
    start = time.time(); cfg, source, locks, picks = audit_source()
    locks.update(own_hashes())
    protocol = {"schema":"gv2-phase1-smoke-protocol-1","scope":"smoke-five-samples",
        "config":cfg,"source_sha256":locks,"samples":[{**i,"K":e["K"]} for e,md,i in picks],
        "source_counts":source["counts"],"test_used":False,
        "computation_contract":{"new_scoring_backwards":0,"optimizer_steps":0,"G_forwards":0,
            "allowed":"original text-forward JVP and FD checks only"},
        "z_t_usage":cfg["z_t_usage"],"generator_input_contract":cfg["generator_input_contract"],
        "aggregate_reconstruction":"CPU fp32 sum(score_weights.float()[:,None] * attribute_jvps); no vector normalization"}
    with io.output_lock(output) as out:
        protocol_path = out/"protocol.json"
        if protocol_path.exists() and io.load(protocol_path) != protocol: raise ValueError("protocol changed; use new output")
        if (out/"completed.json").exists():
            io.checked_cache(out)
            print(json.dumps({"mode":"ALREADY_COMPLETE","new_scoring_backwards_this_run":0,
                "single_attribute_JVPs_this_run":0,"autograd_grad_calls_this_run":0,
                "cuda_initialized":torch.cuda.is_initialized(),"z_t_usage":cfg["z_t_usage"]}),flush=True)
            return
        io.save_json(protocol_path,protocol)
        entries = []; pending = []
        for e,md,identity in picks:
            directory = out/"samples"/e["row_id"].replace(":","_")
            if directory.exists():
                entry = io.entry_at(directory); value, saved, report = io.read_entry(entry,io.digest(protocol))
                if saved["source_entry"] != e: raise ValueError("resume source changed")
                entries.append(entry)
            else: pending.append((e,md,identity))
        counts = {"new_scoring_backwards_this_run":0,"single_attribute_JVPs_this_run":0,
            "text_transport_samples_this_run":0,"autograd_grad_calls_this_run":0,
            "tensor_backward_calls_this_run":0,"optimizer_steps_this_run":0,"G_forwards_this_run":0}
        if pending:
            from integration import frontend, prepare_sample
            it = frontend(); current, _ = it.input_locks()
            io.verify(current)
            # 这些API若被误调用立即失败；JVP内部autograd.grad单独计数。
            with patch.object(it,"fresh_score",side_effect=AssertionError("scoring forbidden")), \
                 patch.object(it.frozen.score,"baseline_gradient",side_effect=AssertionError("scoring forbidden")), \
                 patch.object(it.frozen.score,"score_instances",side_effect=AssertionError("scoring forbidden")), \
                 patch("torch.Tensor.backward",side_effect=AssertionError("training backward forbidden")), \
                 patch("torch.autograd.grad",wraps=torch.autograd.grad) as grad_calls:
                victim = it.frozen.setup()
                versions = {n:p._version for n,p in victim.named_parameters()}
                reader,bases,_ = it.frozen.fresh_localizer(victim)
                reader_state = {k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
                for e,md,identity in pending:
                    value, meta = prepare_sample(it,victim,bases,e,md,counts)
                    meta["protocol_hash"] = io.digest(protocol)
                    entry = io.write_entry(out,value,meta); entries.append(entry)
                    print(json.dumps({"event":"SUPERVISION_READY","row_id":e["row_id"],"K":value["K"],
                        "aggregate_reconstruction":validate(value),"comparisons":meta["comparisons"],
                        "new_scoring_backwards_this_run":0,"z_t_usage":cfg["z_t_usage"]}),flush=True)
                it.frozen.assert_frozen(victim,versions,reader,reader_state)
                counts["autograd_grad_calls_this_run"] = grad_calls.call_count
                it.frozen.frozen.assert_formal_clean()
            del victim, reader
            torch.cuda.empty_cache()
        rank = {e["row_id"]:n for n,(e,md,i) in enumerate(picks)}; entries.sort(key=lambda e:rank[e["row_id"]])
        reports = []; comparisons = []; checks = []
        for e in entries:
            value,meta,report = io.read_entry(e,io.digest(protocol))
            reports.append({"row_id":e["row_id"],**report})
            comparisons.append({"row_id":e["row_id"],**meta["comparisons"]})
            checks.extend(meta["JVP_checks"])
        io.verify(locks)
        summary = {"scope":"smoke-five-samples","count":len(entries),
            "source_counts":source["counts"],"source_sha_count":len(locks),
            "sample_ids":[e["row_id"] for e in entries],"K":[e["K"] for e in entries],
            **counts,"samples_resumed":len(picks)-len(pending),
            "new_computation_types":["original IRRA text-forward JVP","original central finite-difference/zero checks",
                "CPU aggregate reconstruction and cache SHA validation"],
            "z_t_usage":cfg["z_t_usage"],"generator_input_contract":cfg["generator_input_contract"],
            "aggregate_reconstruction_per_sample":reports,
            "aggregate_reconstruction_max_abs_error":max(x["max_abs_error"] for x in reports),
            "aggregation_comparisons":comparisons,
            "max_JVP_FD_relative_error":max((x["central_fd_relative_error"] for x in checks),default=0),
            "IRRA_reader_frozen":True,"source_SHA_unchanged":True,"test_used":False,
            "seconds":time.time()-start}
        io.save_json(out/"summary.json",summary)
        io.save_json(out/"manifest.json",{"schema":"gv2-condition-manifest-1","count":len(entries),
            "entries":entries,"protocol_path":str(protocol_path),"protocol_sha256":io.sha(protocol_path)})
        io.save_json(out/"completed.json",{"method_audit_passed":True,
            "manifest_sha256":io.sha(out/"manifest.json"),"summary_sha256":io.sha(out/"summary.json")})
        io.checked_cache(out)
        print("PHASE1_SMOKE_COMPLETE "+json.dumps(summary,ensure_ascii=False),flush=True)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--allow-smoke",action="store_true")
    p.add_argument("--output",type=Path,default=io.OUTPUTS/"smoke")
    args = p.parse_args(argv)
    if args.allow_smoke: run_smoke(args.output)
    else: preflight()

if __name__ == "__main__": main()
