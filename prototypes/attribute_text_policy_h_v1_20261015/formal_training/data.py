"""2048/128来源核验与caption-only资格账本；不调用模型。"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import torch
from phase2_sources import (ROOT, IMAGES, EPOCH10_SHA, IRRA_SHA, digest, checked, within, read_json)
from training_inputs import load_training_catalog
from text_frontend import CaptionFrontend
from tokenizer_probe import official_probe
from contracts import AttributeSpan, TextExample
from constraints import candidates
from phase3_inputs import CHECKPOINT, IRRA_WEIGHT
from .common import sources, immutable_json, cpu_only

PREPROCESS = {
    "size_hw": [384,128], "pixels": "PIL RGB Resize bilinear + ToTensor [0,1]",
    "mean": [.48145466,.4578275,.40821073], "std": [.26862954,.26130258,.27577711],
    "features": "official IRRA fp32 L2-normalized 512", "epsilon": 8/255,
    "tokens": "official Byte-BPE 77; original truncation retained only as validation NOOP",
}
INELIGIBLE = "train: retain visit, NOOP, no reward/optimizer; validation: retain query, NOOP"
def members(catalog, role):
    if role not in ("train","validation"): raise ValueError("official Test is forbidden")
    samples = tuple(getattr(catalog, role))
    if (not samples or len({s.row_id for s in samples}) != len(samples)
            or len({s.image_rel_path for s in samples}) != len(samples)):
        raise ValueError("unique official image and row required")
    for sample in samples:
        if sample.split != "train" or sample.role != role:
            raise ValueError("train/validation source role mismatch")
        if role == "train": catalog.require_train(sample)
    return samples

def image_path(catalog, sample):
    if sample not in members(catalog, sample.role): raise ValueError("unverified identity")
    return checked_image(sample)

def checked_image(sample):
    path = within(IMAGES, IMAGES/sample.image_rel_path)
    checked(path, sample.image_sha256)
    return path

def qualify_caption(caption, frontend, probe, max_bytes=1024):
    # 不传PID、图像或配对评分；属性序列全部来自caption原文。
    example, audit = frontend.convert(caption)
    token = probe.inspect(caption)
    counts = [sum(len(v) for v in candidates(example, caption, i, set(), probe).values())
              for i in range(len(example.attributes))] if not token.truncated else [0]*len(example.attributes)
    reasons = []
    if token.utf8_length > max_bytes: reasons.append("H_bytes_over_1024")
    if token.truncated: reasons.append("original_BPE_truncated")
    if not example.attributes: reasons.append("no_canonical_attributes")
    if not any(counts): reasons.append("no_legal_token_changing_edit")
    return {"caption": caption, "attributes": [asdict(a) for a in example.attributes],
            "frontend_audit": audit, "attribute_count": len(counts),
            "legal_attribute_count": sum(c>0 for c in counts),
            "legal_candidate_counts": counts, "utf8_bytes": token.utf8_length,
            "BPE_length": token.total_length, "original_truncated": token.truncated,
            "eligible": not reasons, "reasons": reasons, "ineligible_handling": INELIGIBLE,
            # 资格K0只是没有合法属性；H的实际Dynamic-K每step重新预测。
            "structural_K0": not any(counts)}

def example_from(entry):
    return TextExample(entry["caption"], tuple(AttributeSpan(**a) for a in entry["attributes"]))

def summarize(entries):
    n = len(entries)
    return {"samples": n, "eligible": sum(e["eligible"] for e in entries),
            "H_K0_no_attributes": sum(e["attribute_count"]==0 for e in entries),
            "fixed_NOOP_ineligible": sum(not e["eligible"] for e in entries),
            "attribute_covered": sum(e["attribute_count"]>0 for e in entries),
            "legal_edit_covered": sum(e["legal_attribute_count"]>0 for e in entries),
            "structural_K0": sum(e["structural_K0"] for e in entries),
            "original_truncated": sum(e["original_truncated"] for e in entries),
            "H_bytes_over_1024": sum(e["utf8_bytes"]>1024 for e in entries),
            "all_spans_exact": all(e["caption"][a["char_start"]:a["char_end"]]==a["raw_text"]
                                   for e in entries for a in e["attributes"]),
            "attribute_count_distribution": dict(sorted(Counter(str(e["attribute_count"]) for e in entries).items())),
            "legal_attribute_count_distribution": dict(sorted(Counter(str(e["legal_attribute_count"]) for e in entries).items())),
            "ineligible_reason_counts": dict(Counter(r for e in entries for r in e["reasons"]))}

def base_protocol(catalog, role, kind):
    samples = members(catalog, role)
    return {"schema": "hv1-formal-"+kind+"-1", "execution_backend": "official_irra_real",
            "role": role, "catalog_sha256": catalog.fingerprint,
            "training_sources": catalog.source_contract, "identities": [asdict(s) for s in samples],
            "G_checkpoint": str(CHECKPOINT), "G_sha256": EPOCH10_SHA,
            "IRRA_weight": str(IRRA_WEIGHT), "IRRA_sha256": IRRA_SHA,
            "preprocess": PREPROCESS, "source_sha256": sources(kind),
            "ineligible_handling": INELIGIBLE,
            "environment": {**{n:__import__("importlib.metadata",fromlist=["version"]).version(n)
                for n in ("torch","torchvision","Pillow","ftfy","regex","numpy")},
                "torch_cuda_build": torch.version.cuda}}

def preflight_models():
    checked(CHECKPOINT, EPOCH10_SHA); checked(IRRA_WEIGHT, IRRA_SHA)
    if read_json(CHECKPOINT.with_suffix(".json"))["sha256"] != EPOCH10_SHA:
        raise ValueError("G sidecar SHA mismatch")

def qualification(root):
    catalog = load_training_catalog()
    if len(catalog.train)!=2048 or len(catalog.validation)!=128:
        raise ValueError("locked source sizes changed")
    if {s.pid for s in catalog.train}&{s.pid for s in catalog.validation}:
        raise ValueError("PID leakage")
    preflight_models()
    frontend, probe = CaptionFrontend(), official_probe()
    # 跨候选缓存Tokenizer报告；没有任何IRRA前向。
    original_inspect, memo = probe.inspect, {}
    def inspect(text):
        if text not in memo: memo[text] = original_inspect(text)
        return memo[text]
    probe.inspect = inspect
    results = {}
    for role in ("train","validation"):
        protocol = base_protocol(catalog, role, "data")
        entries = []
        for i,s in enumerate(members(catalog,role)):
            checked_image(s)
            entry = qualify_caption(s.caption, frontend, probe)
            entry["identity"] = asdict(s)
            entries.append(entry)
            if (i+1)%128==0: print("CPU_QUALIFICATION",role,i+1,flush=True)
        result = {"protocol": protocol, "protocol_sha256": digest(protocol),
                  "entries": entries, "summary": summarize(entries)}
        immutable_json(Path(root)/"qualification"/(role+".json"), result)
        results[role] = result["summary"]
    immutable_json(Path(root)/"qualification"/"completed.json",
                   {r+"_sha256": __import__("phase2_sources").sha(Path(root)/"qualification"/(r+".json"))
                    for r in ("train","validation")})
    cpu_only()
    return results

def read_qualification(root, catalog, role):
    receipt=read_json(Path(root)/"qualification"/"completed.json")
    path=Path(root)/"qualification"/(role+".json")
    if receipt.get(role+"_sha256")!=__import__("phase2_sources").sha(path):
        raise ValueError("qualification content SHA mismatch")
    result = read_json(path)
    expected = base_protocol(catalog, role, "data")
    if result["protocol"] != expected or result["protocol_sha256"] != digest(expected):
        raise ValueError("qualification source/model/catalog mismatch")
    samples = members(catalog, role)
    if [e["identity"] for e in result["entries"]] != [asdict(s) for s in samples]:
        raise ValueError("qualification identity/order mismatch")
    for e,s in zip(result["entries"],samples):
        if e["caption"]!=s.caption: raise ValueError("qualification Caption mismatch")
        example_from(e)  # exact source spans checked by dataclass
    if summarize(result["entries"])!=result["summary"]: raise ValueError("qualification summary mismatch")
    return result
