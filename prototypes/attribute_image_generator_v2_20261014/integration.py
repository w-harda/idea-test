"""只读复用G-v1冻结plan/JVP；绝不调用fresh_score。"""
import importlib.util, sys
from pathlib import Path
import torch
import cache_io as io
from condition_schema import build, error, validate, ATOL, RTOL
V1 = io.ROOT.parent / "attribute_image_generator_20261014"

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

def frontend():
    # G-v1内部使用绝对import common；显式绑定其原模块，避免与G-v2 IO混用。
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(V1))
    original = sys.modules.get("common")
    sys.modules["common"] = module("gv2_readonly_v1_common", V1/"common.py")
    try: return module("gv2_readonly_v1_integration", V1/"integration.py")
    finally:
        if original is None: sys.modules.pop("common", None)
        else: sys.modules["common"] = original

def prepare_sample(it, victim, bases, entry, metadata, counts):
    from conditioning import aggregate_condition
    source = torch.load(entry["tensor_path"], map_location="cpu", weights_only=True)
    record = metadata["score_record"]; payload = metadata["payload"]
    # 只做原选择实现的逐值复核；正式plan消费已保存selection，不替换它。
    if it.selection(record) != payload: raise ValueError("frozen selection regression")
    plan = it.prepare_uncapped(payload, record, bases, "cuda")
    if [a.instance_id for a in plan] != metadata["selected_instance_ids"]: raise ValueError("plan order changed")
    transport = it.TextTransport(victim, entry["caption"])
    if not torch.equal(transport.clean.cpu().reshape(-1), source["z_t"]):
        raise ValueError("fresh caption feature differs from G-v1; no replacement")
    vectors = {}; checks = []
    for a in plan:
        v, check = transport.read(a); counts["single_attribute_JVPs_this_run"] += 1
        if not check["zero_exact"] or check["central_fd_relative_error"] > .02:
            raise ValueError("original JVP zero/FD gate")
        vectors[a.instance_id] = v; checks.append(check)
    original_condition, weights = aggregate_condition(transport.clean, plan, vectors)
    counts["text_transport_samples_this_run"] += 1
    if weights != metadata["score_normalized_weights"]: raise ValueError("frozen score weights changed")
    vv = torch.cat([vectors[a.instance_id].detach().cpu() for a in plan], dim=0) if plan else torch.empty(0,512)
    identity = {"row_id":entry["row_id"], "image_rel_path":entry["image"],
        "caption_index":int(entry["row_id"].split(":")[-1]), "split":"train",
        "role":metadata["role"], "pid":entry["pid"], "caption":entry["caption"]}
    value = build(identity, metadata["selected_instance_ids"],
        [a.metadata["S_raw"] for a in plan], weights, vv, source["z_t"])
    comparisons = {"original_device_aggregation":error(value["v_local"], original_condition["v_local"]),
                   "G_v1_cached_aggregation":error(value["v_local"], source["v_local"])}
    for reference in (original_condition["v_local"], source["v_local"]):
        if not torch.allclose(value["v_local"], reference, atol=ATOL, rtol=RTOL):
            raise ValueError("original aggregation regression")
    meta = {"source_selection":payload, "source_score_record":record,
        "source_entry":entry, "source_image_sha256":entry["image_sha256"],
        "JVP_checks":checks, "comparisons":comparisons,
        "z_t_usage":"Global Loss supervision only; never G-v2 input",
        "generator_input_contract":"image only: G(I)", "score_reused_exact":True,
        "selection_reused_exact":True, "z_t_reused_exact":True,
        "new_scoring_backwards_this_sample":0, "aggregate_reconstruction":validate(value)}
    return value, meta
