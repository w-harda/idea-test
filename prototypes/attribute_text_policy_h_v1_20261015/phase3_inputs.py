"""Phase 3默认CPU来源预检与有界Train Smoke选择。"""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib.metadata
import sys
import torch
from phase2_sources import (ROOT, G2, EPOCH10_SHA, IRRA_SHA, sha, checked,
                            read_json, digest, within)
from training_inputs import load_training_catalog
from tokenizer_probe import official_probe

CHECKPOINT = G2/"outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt"
IRRA_WEIGHT = Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth")
OFFICIAL = Path("/home/lzf/Attack/IRRA-main/IRRA-main")
NEW_SOURCES = ("phase3_inputs.py", "real_feedback.py", "real_feature_cache.py", "smoke_phase3.py")

def ordered(value):
    return hashlib.sha256(("42|"+str(value)).encode()).hexdigest()

def select_smoke(catalog):
    groups = defaultdict(list)
    for sample in catalog.train:
        catalog.require_train(sample)
        groups[sample.pid].append(sample)
    pids = sorted((pid for pid,v in groups.items() if len(v)>=2), key=ordered)[:5]
    if len(pids)!=5:
        raise ValueError("five training PIDs with two distinct images required")
    gallery = tuple(s for pid in pids for s in sorted(groups[pid],key=lambda x:ordered(x.row_id))[:2])
    queries = gallery
    validate_selection(catalog,gallery,queries)
    return gallery,queries

def validate_selection(catalog,gallery,queries):
    if (not 1<=len(gallery)<=10 or not 1<=len(queries)<=10
            or len({s.row_id for s in gallery})!=len(gallery)
            or len({s.image_rel_path for s in gallery})!=len(gallery)
            or len({s.row_id for s in queries})!=len(queries)):
        raise ValueError("unique train Gallery/Query each limited to 10")
    for s in (*gallery,*queries):
        catalog.require_train(s)
    for q in queries:
        positives = sum(q.pid==s.pid for s in gallery)
        if q not in gallery or not 0<positives<len(gallery):
            raise ValueError("paired training positive and training negatives required")

def safe_output(path):
    return within(ROOT/"outputs/phase3",path)

def verify_sources(protocol):
    for name,expected in protocol["source_sha256"].items():
        checked(name,expected)
    if (protocol["execution_backend"]!="official_irra_real"
            or protocol["checkpoint_sha256"]!=EPOCH10_SHA or protocol["IRRA_sha256"]!=IRRA_SHA):
        raise ValueError("not the fixed Phase 3 real model contract")
    checked(protocol["checkpoint"],EPOCH10_SHA)
    checked(protocol["IRRA_weight"],IRRA_SHA)

def preflight():
    config = read_json(ROOT/"config_phase3.json")
    for name,expected in config["source_sha256"].items():
        checked(name,expected)
    checked(CHECKPOINT,EPOCH10_SHA)
    checked(IRRA_WEIGHT,IRRA_SHA)
    side = read_json(CHECKPOINT.with_suffix(".json"))
    if side["sha256"]!=EPOCH10_SHA:
        raise ValueError("G-v2 checkpoint sidecar mismatch")
    catalog = load_training_catalog()
    if catalog.audit["counts"]!={"train":2048,"validation":128}:
        raise ValueError("original train protocol changed")
    gallery,queries = select_smoke(catalog)
    for sample in gallery:
        catalog.verify_image(sample)
    probe = official_probe()
    if any(probe.inspect(q.caption).truncated for q in queries):
        raise ValueError("selected original query truncates; stop rather than resample")
    sources = dict(config["source_sha256"])
    sources.update({str(ROOT/n):sha(ROOT/n) for n in (*NEW_SOURCES,"config_phase3.json")})
    sources[str(CHECKPOINT.with_suffix(".json"))]=sha(CHECKPOINT.with_suffix(".json"))
    sources[str(OFFICIAL/"utils/simple_tokenizer.py")]=__import__("tokenizer_probe").TOKENIZER_SHA
    sources[str(OFFICIAL/"data/bpe_simple_vocab_16e6.txt.gz")]=__import__("tokenizer_probe").BPE_SHA
    protocol = {"schema":"hv1-phase3-real-smoke-1","execution_backend":"official_irra_real",
        "source_sha256":sources,"checkpoint":str(CHECKPOINT),"checkpoint_sha256":EPOCH10_SHA,
        "IRRA_weight":str(IRRA_WEIGHT),"IRRA_sha256":IRRA_SHA,
        "catalog_sha256":catalog.fingerprint,"training_sources":catalog.source_contract,
        "gallery":[asdict(s) for s in gallery],"queries":[asdict(s) for s in queries],
        "selection":config["selection"],"seed":42,"reward_weights":config["reward_weights"],
        "preprocess":{"image_size_hw":[384,128],"PIL":"RGB Resize bilinear + ToTensor",
            "mean":[.48145466,.4578275,.40821073],"std":[.26862954,.26130258,.27577711],
            "image_space":"[0,1] for G and 8/255 projection; IRRA normalization after",
            "retrieval_features":"fp32 L2-normalized 512","text_length":77,
            "text":"official Byte-BPE; reject truncation; raw Unicode is preserved"},
        "limits":{"gallery":10,"queries":10,"G_samples":10,"IRRA_clean_samples":10,
                  "IRRA_adv_samples":10,"IRRA_text_samples":32},
        "environment":{n:importlib.metadata.version(n) for n in ("torch","torchvision","Pillow","ftfy","regex")},
        "H_status":"seed42 untrained CPU; sampled and greedy; no optimizer",
        "train_only":True,"optimizer_steps":0,"new_scoring_backwards":0,"new_JVPs":0}
    if torch.cuda.is_initialized():
        raise ValueError("CPU preflight unexpectedly initialized CUDA")
    return catalog,gallery,queries,probe,protocol

def import_gv2():
    sys.path.insert(0,str(G2))
    import inference_once
    import irra_image_adapter
    for module,name in ((inference_once,"inference_once.py"),(irra_image_adapter,"irra_image_adapter.py")):
        if Path(module.__file__).resolve()!=G2/name:
            raise ValueError("G-v2 module path collision")
    return inference_once,irra_image_adapter
