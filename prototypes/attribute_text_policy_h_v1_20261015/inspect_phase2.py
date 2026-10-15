"""最多16条真实train Caption只读CPU检查；不实例化H/G/IRRA网络。"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import torch
from unittest.mock import patch
from training_inputs import load_training_catalog
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from constraints import candidates, make_edit, replay
from policy import natural_k, CharacterPolicy
from phase2_sources import ROOT, new_json

def forbidden(*a, **k):
    raise RuntimeError("model/gradient/CUDA access forbidden during Phase 2 inspection")

def inspect(count=16):
    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("readonly CPU sample limit is 16")
    catalog = load_training_catalog()
    frontend, probe = CaptionFrontend(), official_probe()
    torch.manual_seed(42)
    policy = CharacterPolicy(probe).eval()
    chosen = sorted(catalog.train, key=lambda s: hashlib.sha256(("42|"+s.row_id).encode()).hexdigest())[:count]
    rows = []
    for s in chosen:
        catalog.verify_image(s)
        example, audit = frontend.convert(s.caption)
        supports = [candidates(example, s.caption, i, set(), probe) for i in range(len(example.attributes))]
        # 不能将均匀分数冒称H预测或配对TopK。只报告可编辑候选容量。
        capacity_k = natural_k([1.]*sum(bool(x) for x in supports))
        with torch.inference_mode():
            output = policy.attack(example)
        original = probe.inspect(s.caption)
        traces = []
        for i,support in enumerate(supports):
            for pos, replacements in support.items():
                for replacement in replacements:
                    e = make_edit(example, i, pos, replacement)
                    variant = replay(example, (e,), 1)
                    traces.append({"edit": asdict(e), "tokens": probe.pair_report(s.caption, variant)})
        rows.append({"row_id":s.row_id, "pid":s.pid, "image_rel_path":s.image_rel_path,
                     "caption_index":s.caption_index, "split":s.split, "role":s.role,
                     "caption":s.caption, "source_image_sha256":s.image_sha256,
                     "frontend":audit, "attribute_count":len(example.attributes),
                     "legal_attribute_count":sum(bool(x) for x in supports),
                     "legal_positions":sum(len(x) for x in supports),
                     "candidate_edits":len(traces), "capacity_k_uniform_only":capacity_k,
                     "H_seed42_untrained_K":output.budget_k, "H_greedy_edits":len(output.edits),
                     "original_token_length":original.total_length,
                     "original_truncated":original.truncated,
                     "edit_tokenizer_audits":traces})
    attrs = sum(x["attribute_count"] for x in rows)
    legal = sum(x["legal_attribute_count"] for x in rows)
    summary = {"sample_count":count, "selection":"SHA256(42|row_id), first N train only",
        "catalog":catalog.audit,"catalog_sha256":catalog.fingerprint,
        "identity_sha_verified_units":2176, "image_sha_verified_count":count,
        "caption_attribute_coverage_count":sum(x["attribute_count"]>0 for x in rows),
        "attributes":attrs, "exact_spans":sum(x["frontend"]["exact_span_count"] for x in rows),
        "legal_attributes":legal,"caption_editable_count":sum(x["legal_attribute_count"]>0 for x in rows),
        "legal_candidate_edits":sum(x["candidate_edits"] for x in rows),
        "original_truncated":sum(x["original_truncated"] for x in rows),
        "legal_edit_truncated":sum(e["tokens"]["after"]["truncated"] for x in rows for e in x["edit_tokenizer_audits"]),
        "candidate_count_distribution":dict(Counter(x["attribute_count"] for x in rows)),
        "capacity_k_uniform_only_distribution":dict(Counter(x["capacity_k_uniform_only"] for x in rows)),
        "H_seed42_untrained_K_distribution":dict(Counter(x["H_seed42_untrained_K"] for x in rows)),
        "H_policy_status":"seed42 untrained CPU diagnostic only; not learned importance",
        "real_G_forward":0,"real_IRRA_forward":0,"new_scoring_backwards":0,"new_JVP":0,
        "training_optimizer_steps":0, "rows":rows}
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--output", default=str(ROOT/"outputs/phase2/inspection.json"))
    args=p.parse_args()
    with patch.object(torch, "load", forbidden), \
         patch.object(torch.Tensor, "backward", forbidden), patch.object(torch.autograd, "grad", forbidden), \
         patch.object(torch.cuda, "_lazy_init", forbidden):
        result=inspect(args.samples)
        new_json(args.output, result)
    print("PHASE2_CPU_INSPECTION", {k:v for k,v in result.items() if k!="rows"}, flush=True)
if __name__=="__main__":
    main()
