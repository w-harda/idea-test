"""一次性SGA官方Batch8容量检查；复用冻结source_run，不改Full攻击逻辑。"""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import argparse
from tbps.artifacts import FileRef, checked, read_json, sha_file
from tbps.baselines import sga_irra as frozen
from tbps.baselines.sga_official_io import dataset as full_dataset
from tbps.models.sga_irra_source import require

SCOPE="batch8_v1"
def select_first_eight(c,scope):
    require(scope==SCOPE,"capacity-only scope")
    ds,raw,matched=full_dataset(c,"full")
    images=raw.gallery[:8];ids=tuple(g.image_id for g in images)
    queries=tuple(q for q in raw.queries if q.paired_image_id in ids)
    matched_queries=tuple(q for q in matched.queries if q.paired_image_id in ids)
    require(list(ids)==c["capacity"]["gallery_ids"] and
            [q.query_id for q in queries]==c["capacity"]["query_ids"],"fixed official first-eight ordered members")
    require(len(images)==8 and len(queries)==len(matched_queries),"all corresponding Caption members")
    return ds,replace(raw,gallery=images,queries=queries),replace(matched,gallery=images,queries=matched_queries)

def check_config(path):
    c=frozen.configuration(path)
    require(c["capacity"]["scope"]==SCOPE and c["capacity"]["source_only"] is True,"source-only capacity")
    # 同一冻结源码的独立工作树副本，逐文件内容SHA验证实际导入位置。
    for relative,expected in c["capacity"]["executed_modules"].items():
        actual=Path(__file__).resolve().parents[2]/relative
        checked(FileRef(str(actual),expected))
    require(sha_file(frozen.__file__)==c["capacity"]["runner_sha256"],"unchanged full source runner")
    return c

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--action",choices=("preflight","run"),default="preflight")
    p.add_argument("--allow-gpu-capacity",action="store_true")
    a=p.parse_args(argv);c=check_config(a.config)
    with patch.object(frozen,"dataset",select_first_eight):
        if a.action=="preflight":
            frozen.preflight(a.config,SCOPE)
            print("SGA_BATCH8_CPU_PREFLIGHT_PASS",flush=True);return 0
        require(a.allow_gpu_capacity,"explicit one-time capacity GPU authorization")
        result=frozen.source_run(a.config,SCOPE)
    if result:return result
    root=Path(c["output_base"])/SCOPE/"source"
    report=read_json(root/"execution/report.json");collection=read_json(root/"completed/collection.json")
    require(report["status"]=="PASS" and collection["Gallery"]==8 and collection["Query"]==len(c["capacity"]["query_ids"]),"complete actual Batch8")
    require(collection["official_source_calls"]==1 and report["source_model_loads"]==1 and report["MLM_loads"]==1,"one batch / one model restore")
    require(report["model_before"]==report["model_after"] and report["MLM_before"]==report["MLM_after"],"source and MLM frozen")
    print("SGA_BATCH8_CAPACITY_PASS",flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
