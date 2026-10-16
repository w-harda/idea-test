"""Phase 4：只读复用锁定Phase 3图库，建立独立32-step训练协议。"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import importlib.metadata
import torch
from phase2_sources import ROOT,digest,sha,checked,read_json,within
from phase3_inputs import validate_selection,verify_sources
from smoke_phase3 import verify_completed
from real_feature_cache import RealFeatureCache
from training_inputs import load_training_catalog
from tokenizer_probe import official_probe

PHASE3=ROOT/"outputs/phase3/smoke10"
SOURCE_FILES=("phase4_inputs.py","phase4_checkpoint.py","phase4_training.py",
              "phase4_real_reward.py","train_phase4.py","config_phase4.json")

def safe_output(path):
    return within(ROOT/"outputs/phase4",path)

def query_order(queries,total_steps=32,seed=42):
    if not 1<=total_steps<=32:
        raise ValueError("Phase 4 total update limit is 32")
    rng=torch.Generator(device="cpu").manual_seed(seed)
    result=[]
    while len(result)<total_steps:
        result.extend(queries[i].row_id for i in torch.randperm(len(queries),generator=rng).tolist())
    return result[:total_steps]

def validate_config(config):
    expected={"schema":"hv1-phase4-scst32-config-1","seed":42,
        "architecture":{"width":64,"layers":2,"heads":4,"max_bytes":1024},
        "optimizer":{"name":"Adam","lr":1e-4,"betas":[.9,.999],"eps":1e-8,"weight_decay":0.0},
        "batch":1,"total_steps":32,"reward_weights":{"lambda_sim":1.0,"lambda_rank":1.0},
        "h_device":"cpu","reward_device":"cuda","checkpoint_every_step":1,
        "midpoint_reload_step":16,"query_order":"repeat torch.randperm(10), separate CPU Generator seed42",
        "sample_generator_seed":42,
        "reward_text_policy":"fresh original/sampled/greedy GPU encoding every step; no Phase3 text reuse",
        "G_new_forward_limit":0,"IRRA_image_new_forward_limit":0,"IRRA_text_sample_limit":96,
        "not_for_model_selection":True}
    if config!=expected:
        raise ValueError("fixed Phase 4 configuration changed")

def read_phase3_gallery(catalog,gallery,p3):
    """仅调用旧load纯读函数，不实例化会mkdir的缓存，不访问其生成路径。"""
    reader=SimpleNamespace(hash=digest(p3))
    root=PHASE3/"real_features/image"
    units=[]
    locks={}
    maximum=0.
    for sample in gallery:
        catalog.require_train(sample)
        catalog.verify_image(sample)
        identity=asdict(sample)
        key=digest({"identity":identity,"texts":None})
        unit=root/key
        extra={"kind":"image","execution_backend":"official_irra_real",
               "execution":{"G_samples":1,"IRRA_clean_samples":1,"IRRA_adv_samples":1,"IRRA_text_samples":0}}
        data,md=RealFeatureCache.load(reader,unit,identity,extra,(1,512),("clean","adversarial"))
        if not 0<=md["linf"]<=8/255:
            raise ValueError("historical Gallery budget mismatch")
        maximum=max(maximum,md["linf"])
        units.append(data)
        for p in (*unit.iterdir(),unit.with_name(unit.name+".receipt.json")):
            if p.is_file():locks[str(p)]=sha(p)
    return {k:torch.cat([x[k] for x in units]).clone() for k in ("clean","adversarial")},locks,maximum

def verify_protocol(protocol):
    if (protocol["schema"]!="hv1-phase4-scst32-protocol-1"
            or protocol["config"]["total_steps"]!=32
            or protocol["gallery_count"]!=10 or protocol["query_count"]!=10):
        raise ValueError("invalid bounded Phase 4 protocol")
    validate_config(protocol["config"])
    verify_sources(protocol["phase3_protocol"])
    for name,expected in protocol["source_sha256"].items():
        checked(name,expected)
    for name,expected in protocol["phase3_evidence_sha256"].items():
        checked(name,expected)
    for name,expected in protocol["gallery_feature_sha256"].items():
        checked(name,expected)

def preflight():
    checked(PHASE3/"summary.json","157fcc0d043960023b88670d04c4c8c64475fab4bbde94d439c813c297ce98af")
    checked(PHASE3/"completed.json","10696268820b832a02d4e1a08b74e80a2b5526c58811eab3cd83e7f507b5cb08")
    summary=verify_completed(PHASE3)
    p3=read_json(PHASE3/"protocol.json")
    catalog=load_training_catalog()
    gallery=tuple(catalog.get_train(x["row_id"]) for x in p3["gallery"])
    queries=tuple(catalog.get_train(x["row_id"]) for x in p3["queries"])
    if ([asdict(x) for x in gallery]!=p3["gallery"] or [asdict(x) for x in queries]!=p3["queries"]
            or catalog.fingerprint!=p3["catalog_sha256"]):
        raise ValueError("Phase 3 train identity/catalog mismatch")
    validate_selection(catalog,gallery,queries)
    features,locks,linf=read_phase3_gallery(catalog,gallery,p3)
    config=read_json(ROOT/"config_phase4.json")
    validate_config(config)
    evidence={str(PHASE3/n):sha(PHASE3/n) for n in ("protocol.json","summary.json","completed.json",
        "attempts/attempt_0001/execution.json","real_features/protocol.json")}
    protocol={"schema":"hv1-phase4-scst32-protocol-1","config":config,
        "phase3_protocol":p3,"phase3_evidence_sha256":evidence,
        "source_sha256":{str(ROOT/n):sha(ROOT/n) for n in SOURCE_FILES},
        "catalog_sha256":catalog.fingerprint,"gallery":[asdict(s) for s in gallery],
        "queries":[asdict(s) for s in queries],"gallery_count":10,"query_count":10,
        "gallery_feature_sha256":locks,"historical_maximum_linf":linf,
        "query_order":query_order(queries),"phase3_text_features_reused":False,
        "new_G_forwards":0,"new_IRRA_image_forwards":0,
        "validation_or_Test_feedback":False,
        "initial_H_state_sha256":summary["H_state_sha256_before"],
        "environment":{**{k:importlib.metadata.version(k) for k in ("torch","torchvision","numpy")},
                       "cpu_threads":torch.get_num_threads()}}
    if torch.cuda.is_initialized():
        raise ValueError("CPU preflight initialized CUDA")
    return catalog,gallery,queries,features,official_probe(),protocol
