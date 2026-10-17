"""Stage4D1默认CPU preflight；仅显式授权执行历史H全量复制，没有GPU入口。"""
from dataclasses import replace,asdict
from pathlib import Path
import argparse
import json
import os
import time
from ..artifacts import ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes
from ..runtime_closure import cpu_only
from ..engines.historical_sources import JointFullSources
from ..engines.historical_loader import native_action_validator
from ..engines.replay_control import rss_bytes
from .contracts import need
from .smoke_support import original_plan
from .legacy_text import HistoricalTextSource
from .bulk_session import StreamingMaterializedStore
from .full_runner import ValidatedActions,publish_texts

BASE="/home/lzf/ldx/outputs/tbps-framework-stage4d1"
FULL=BASE+"/materialized_full_v1"
PILOT=BASE+"/text_pilot_v1"

def fs_resources(path):
    path=Path(path)
    while not path.exists():path=path.parent
    s=os.statvfs(path)
    return {"available_bytes":s.f_bavail*s.f_frsize,"available_inodes":s.f_favail,"block_size":s.f_frsize}

def control(path):
    c=read_json(path,"tbps-full-materialization-preparation-1")
    need(set(c)=={"schema","source_config","c1_receipt","c3_receipt","gallery","query",
        "text_statistics","image_size","image_epsilon","pilot_queries","GPU_generation_authorized"},"full prepare fields")
    need(c["gallery"]==3074 and c["query"]==6156 and c["text_statistics"]=={"edited":6116,"NOOP":40,"edits":13537}
         and c["image_size"]==[384,128] and c["image_epsilon"]==8/255 and c["pilot_queries"]==16
         and c["GPU_generation_authorized"]is False,"frozen full identity/attack boundary")
    for key in ("source_config","c1_receipt","c3_receipt"):
        ref=FileRef(**c[key]);checked(ref)
    for key,status,field in (("c1_receipt","MATERIALIZATION_CPU_SOURCE_PASS","status"),
                             ("c3_receipt","IRRA_MATERIALIZED_SMOKE_VERIFIED","status")):
        folder=Path(c[key]["path"]).parent
        ArtifactStore(folder.parent,new_base=str(folder.parent.parent)).verify(folder.name)
        need(read_json(folder/"report.json")[field]==status,"required historical acceptance")
    return c

def prepare(path):
    c=control(path);parent=read_json(c["source_config"]["path"])
    plan,legacy=original_plan(parent)
    refs={r.path:r for r in plan.sources}
    for ref in (FileRef(str(Path(path).resolve()),sha_file(path)),*(FileRef(**c[k]) for k in ("c1_receipt","c3_receipt")),
                *(FileRef(str(Path(__file__).with_name(n)),sha_file(Path(__file__).with_name(n))) for n in
                ("full_prepare.py","full_runner.py","bulk_session.py","generation.py"))):
        refs[ref.path]=ref
    plan=replace(plan,sources=tuple(refs.values()))
    sources=JointFullSources(parent["replay_config"]["path"])
    need(plan.manifest==sources.manifest and len(plan.manifest.gallery)==c["gallery"]
         and len(plan.manifest.queries)==c["query"],"official full ordered membership")
    fs=fs_resources(BASE)
    memory={line.split(":")[0]:int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith(("MemAvailable:","MemTotal:"))}
    need(memory["MemAvailable"]>=4*1024**3,"insufficient actual RAM for CPU materialization")
    fs["RAM"]=memory
    need(fs["available_bytes"]>=10*1024**3 and fs["available_inodes"]>=100000,"disk/inode headroom for complete materialization")
    return c,plan,legacy,sources,fs

def execute(path,*,full=False,pilot=False):
    need(not (full and pilot),"one CPU action")
    target=FULL if full else PILOT
    if full or pilot:need(not Path(target).exists(),"one controlled output already exists; no automatic retry")
    with cpu_only():c,plan,legacy,sources,fs=prepare(path)
    if not(full or pilot):
        print(json.dumps({"status":"FULL_MATERIALIZATION_CPU_PREFLIGHT_PASS","plan_id":plan.id,
              "gallery":len(plan.manifest.gallery),"query":len(plan.manifest.queries),"filesystems":fs,
              "GPU_generation_authorized":False,"model_loads":0},indent=2));return 0
    # 同一计划、不同Pilot/正式Store；Pilot不会被当成完整集合。
    records=ArtifactStore(target,new_base=BASE)
    records.publish("claim",{"claim.json":json_bytes({"action":"historical_H7_full" if full else "historical_H7_pilot",
        "plan_id":plan.id,"count":6156 if full else 16,"H_loads":0,"H_forwards":0})},
        {"kind":"controlled_CPU_text_materialization"},sources=(FileRef(str(Path(path).resolve()),sha_file(path)),))
    store=StreamingMaterializedStore(ArtifactStore(Path(target)/"artifacts",new_base=BASE),plan)
    store.publish_plan();started=time.perf_counter();result={"status":"RUNNING"}
    try:
        with cpu_only(),native_action_validator(legacy.config_ref) as native:
            validate=ValidatedActions(native)
            source=HistoricalTextSource(sources,plan.h,validate)
            def progress(n,elapsed,stats):
                memory=next(int(x.split()[1])*1024 for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))
                free=fs_resources(target)
                need(memory>=2*1024**3 and free["available_bytes"]>=1024**3,"actual RAM/disk reserve exhausted; stop")
                print("TEXT_MATERIALIZATION_PROGRESS",n,"/ 6156",round(elapsed,3),stats,flush=True)
            result,refs=publish_texts(store,source,validate,expected=c["text_statistics"] if full else None,
                                      limit=None if full else 16,progress=progress)
            result["validator"]={"hits":validate.hits,"misses":validate.misses}
        result["status"]="TEXT_FULL_MATERIALIZED" if full else "TEXT_PILOT_VERIFIED"
        result["plan_id"]=plan.id;result["disk"]=fs_resources(target)
        result["output_bytes"]=sum(p.stat().st_size for p in Path(target).rglob("*") if p.is_file())
        result["output_files"]=sum(1 for p in Path(target).rglob("*") if p.is_file())
    except BaseException as exc:
        import traceback
        result={"status":"FAIL","error":str(exc),"traceback":traceback.format_exc(),"plan_id":plan.id}
    finally:
        result["total_seconds"]=time.perf_counter()-started;result["peak_RSS_bytes"]=rss_bytes()
        result["G_H_IRRA_model_loads"]=0;result["G_H_IRRA_forwards"]=0;result["GPU_attempts"]=0
        refs=(FileRef(str(Path(target)/"claim/manifest.json"),sha_file(Path(target)/"claim/manifest.json")),)
        if result.get("collection"):refs+= (FileRef(**result["collection"]),)
        records.publish("result",{"report.json":json_bytes(result)},{"kind":"CPU_text_materialization_result","status":result["status"]},sources=refs)
    print(json.dumps(result,ensure_ascii=False,indent=2));print(result["status"],flush=True)
    return int(result["status"]=="FAIL")

def main(argv=None):
    p=argparse.ArgumentParser(description="CPU only historical H materialization; no GPU generation flag")
    p.add_argument("--config",required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--run-pilot",action="store_true")
    group.add_argument("--allow-full-h-text-materialization",action="store_true")
    a=p.parse_args(argv);return execute(a.config,full=a.allow_full_h_text_materialization,pilot=a.run_pilot)
if __name__=="__main__":raise SystemExit(main())
