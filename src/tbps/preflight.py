"""Stage 2 CPU来源/身份检查；没有模型加载、GPU或排名执行选项。"""
from pathlib import Path
import argparse
import json
from .artifacts import ArtifactError, read_json, checked, digest
from .contracts import FileRef
from .datasets.cuhk_pedes import CuhkPedes
from .datasets.historical_membership import load_historical_partitions, SOURCES
from .models.irra_source import IRRASource

def inspect_sources(config):
    if type(config) is not dict or set(config)!={"schema","annotation","image_root","membership","test_protocols","irra"} or config["schema"]!="tbps-stage2-cpu-sources-1":
        raise ArtifactError("Stage 2 CPU source config fields/schema")
    source=IRRASource.from_dict(config["irra"])
    model=source.verify()  # opaque SHA + YAML；没有torch.load或官方model import。
    dataset=CuhkPedes(FileRef(**config["annotation"]),config["image_root"])
    test=dataset.manifest("test",expected_counts=(3074,6156))
    protocols=config["test_protocols"]
    if set(protocols)!={"gv2_full","joint_full","ablation_full"}:raise ArtifactError("all three historical Test protocols required")
    matches={}
    for key,value in protocols.items():
        ref=FileRef(**value);checked(ref);p=read_json(ref.path)
        matches[key]=dataset.match_test_protocol(test,p);checked(ref)
    partitions=load_historical_partitions(dataset,{k:FileRef(**v) for k,v in config["membership"].items()})
    return {"schema":"tbps-stage2-cpu-preflight-1","status":"CPU_SOURCE_IDENTITY_PASS",
        "official_counts":dataset.counts(),"historical_test_matches":matches,
        "attack_members":{"train":len(partitions.train.queries),"validation":len(partitions.validation.queries),
          "train_pids":len({q.pid for q in partitions.train.queries}),
          "validation_pids":len({q.pid for q in partitions.validation.queries}),"pid_intersection":0,
          "membership_sha256":partitions.fingerprint,"G_metadata_SHA_verified":len(partitions.metadata_refs)},
        "model":model,"model_loaded":False,"model_forwards":0,"ranking_recomputed":False,
        "scope":"annotation/order/pairing/source membership + selected image byte SHA; no condition Tensor/full feature audit"}

def main(argv=None):
    parser=argparse.ArgumentParser(description="CPU metadata and identity only; no run/model-load option")
    parser.add_argument("--sources",required=True)
    args=parser.parse_args(argv)
    try:report=inspect_sources(read_json(args.sources))
    except (ValueError,TypeError,OSError,KeyError,IndexError) as exc:parser.exit(1,"CPU_PREFLIGHT_REJECTED "+str(exc)+"\n")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2))
    print("STAGE2_CPU_PREFLIGHT_PASS")

if __name__=="__main__":main()
