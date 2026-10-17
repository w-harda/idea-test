"""Stage4C2R纯CPU：严格旧产物引用、真实CPU准备、静态Checkpoint布局；无GPU选项。"""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
from ..artifacts import ArtifactStore,FileRef,checked,read_json,sha_file,json_bytes,digest
from ..engines.historical_loader import feature_only_cpu
from ..engines.replay_control import rss_bytes
from .contracts import need
from .retained_inputs import RetainedInputs,ORIGINAL
from .smoke_support import config,preflight,selected_features,RSS,GALLERY,QUERY
from .irra_startup import prepare_inputs,isolation_snapshot
from .checkpoint_layout import checkpoint_layout

BASE="/home/lzf/ldx/outputs/tbps-framework-stage4c2r"

def snapshot(root):
    return {str(p):checked(FileRef(str(p),sha_file(p))) for p in Path(root).rglob("*") if p.is_file()}

def audit(path):
    import torch
    from ..runtime_closure import cpu_only
    from ..attacks.base import tensor_sha
    need(not torch.cuda.is_initialized(),"CPU audit only")
    before=snapshot(ORIGINAL);old=RetainedInputs()
    with cpu_only():
        pre=preflight(path)
        c=config(path)
    old_plan_id=old.plan.id
    need(old_plan_id!=pre["plan_id"],"new workspace source identity must remain separate")
    events=[]
    inputs,gate=prepare_inputs(c,ORIGINAL,store_factory=lambda c,o:(old,old.legacy),
                              feature_reader=selected_features,emit=events.append)
    need(gate.completed and gate.exited and not torch.cuda.is_initialized(),"CPU input contexts must exit without CUDA")
    for u in old.units:
        md=read_json(u.metadata.path)
        need(md["plan_id"]==old_plan_id and md["content_sha256"]==u.content_sha256,"retained metadata")
    g=read_json(Path(ORIGINAL)/"stage-g/report.json")
    h=read_json(Path(ORIGINAL)/"stage-text/report.json")
    ir=read_json(Path(ORIGINAL)/"stage-irra/report.json")
    zero={"G":0,"H":0,"IRRA_image":0,"IRRA_text":0}
    need(g["forwards"]["started"]==g["forwards"]["completed"]=={**zero,"G":2} and
         h["forwards"]["started"]==h["forwards"]["completed"]==zero and
         ir["forwards"]["started"]==ir["forwards"]["completed"]==zero,"original exact accounting")
    for report in (g,h,ir):
        need(report["backward"]==report["optimizer"]==report["JVP"]==0,"original no optimization")
    allowed={str(Path(ORIGINAL)/f"raw-delta-{i}/delta.pt") for i in range(2)}
    with feature_only_cpu(allowed):
        raw_reports=[]
        for i,item in enumerate(g["images"]):
            m=old.records.verify("raw-delta-"+str(i))
            p=Path(ORIGINAL)/f"raw-delta-{i}/delta.pt"
            original=torch.load(p,map_location="cpu",weights_only=True)
            x=original.detach().clone();del original
            need(x.shape==(1,3,384,128) and x.dtype==torch.float32 and torch.isfinite(x).all()
                 and tensor_sha(x)==item["raw_delta_sha256"] and float(x.double().abs().max())==item["raw_delta_linf"]
                 and item["raw_delta_linf"]<=8/255 and sha_file(p)==item["raw_delta_file_sha256"],"original raw delta content/SHA")
            raw_reports.append({"content_sha256":tensor_sha(x),"file_sha256":sha_file(p),"linf":item["raw_delta_linf"]})
    # 仅读取ZIP目录和约192KiB pickle描述，无pickle.load/torch.load模型或Storage读入。
    source=inputs.source;checked(source.checkpoint)
    layout=checkpoint_layout(source.checkpoint.path)
    stage25=Path("/home/lzf/ldx/outputs/tbps-framework-stage25/irra_equivalence_v1/result")
    ArtifactStore(stage25.parent,new_base=str(stage25.parent.parent)).verify("result")
    old_irra=read_json(stage25/"report.json")
    need(old_irra["status"]=="PASS" and old_irra["models_loaded"]==1,"Stage2.5 successful reference")
    stage25_refs=[FileRef(str(p),sha_file(p)) for p in
                  (stage25/"manifest.json",stage25/"report.json",stage25.parent.parent/"gpu_smoke.log")]
    baseline=rss_bytes()
    overlap=layout["storage_bytes"]+layout["model_storage_bytes"]
    budget={"rss_limit_bytes":RSS,"checkpoint_layout":layout,"model_and_complete_checkpoint_overlap_bytes":overlap,
            "runtime_headroom_bytes":RSS-overlap,"CPU_prepare_peak_rss_bytes":baseline,
            "conservative_overlap_plus_CPU_peak_bytes":overlap+baseline,
            "conservative_estimate_exceeds_budget":overlap+baseline>RSS,
            "strict_measured_peak_available":False,"memory_heap_reuse_unknown":True,
            "Stage25":{"status":old_irra["status"],"elapsed_seconds":old_irra["elapsed_seconds"],
                "gpu_peak_allocated_mib":old_irra["gpu_peak_allocated_mib"],
                "gpu_peak_reserved_mib":old_irra["gpu_peak_reserved_mib"],"CPU_RSS_recorded":False},
            "decision":"GPU_APPLICATION_BLOCKED_RESOURCE_BUDGET_UNPROVEN",
            "reason":"full eager checkpoint includes optimizer; CPU model overlaps checkpoint; remaining runtime headroom is small"}
    after=snapshot(ORIGINAL)
    need(before==after,"original failed Attempt bytes/mtime changed")
    need(not torch.cuda.is_initialized(),"CPU audit initialized CUDA")
    result={"schema":"tbps-stage4c2r-cpu-audit-1","status":"CPU_REPAIR_AND_RETAINED_INPUTS_VERIFIED",
            "retained_reference":old.descriptor(),"old_plan_id":old_plan_id,"new_workspace_plan_id":pre["plan_id"],
            "old_attempt_status":"FAIL","old_attempt_files_sha_mtime_unchanged":len(before),
            "CPU_startup_events":events,"CPU_contexts_restored":gate.exited,
            "selected_historical_features":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"finite":bool(torch.isfinite(v).all())}
                                           for k,v in inputs.historical.items()},
            "images":g["images"],"texts":h["texts"],"raw_delta_validation":raw_reports,
            "resource_preflight":budget,"import_closure":{"G_H_runtime_sources":75,"GH_status":pre["GH"]["status"],
                 "GH_executed_files":len(pre["GH"]["executed_sources"]),"IRRA_executed_files":len(pre["IRRA"]["executed_sources"])},
            "new_GPU_attempts":0,"model_deserializations":0,"G_H_IRRA_forwards":0,"CUDA_initialized":False,
            "full_rankings":0,"feature_equivalence":"PENDING","four_2x2_score_equivalence":"PENDING",
            "original_artifact_republished":False}
    refs=tuple({r.path:r for r in (
        FileRef(str(Path(path).resolve()),sha_file(path)),*stage25_refs,*inputs.feature_refs,
        *(FileRef(p,v["sha256"]) for p,v in before.items()),
        *(FileRef(str(Path(__file__).with_name(n)),sha_file(Path(__file__).with_name(n)))
          for n in ("audit_retained.py","retained_inputs.py","irra_startup.py","checkpoint_layout.py")))}.values())
    return result,refs

def main(argv=None):
    p=argparse.ArgumentParser(description="Stage4C2R CPU-only audit; no GPU or model load flag")
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    args=p.parse_args(argv)
    need(not Path(args.output).exists(),"immutable CPU audit output already used")
    report,refs=audit(args.config)
    ArtifactStore(args.output,new_base=BASE).publish("result",{"report.json":json_bytes(report)},
        {"kind":"CPU_only_repair_and_retained_inputs_audit"},sources=refs)
    print(json.dumps({k:v for k,v in report.items() if k not in ("images","texts")},ensure_ascii=False,indent=2))
    print("STAGE4C2R_CPU_AUDIT_PASS")
    print(report["resource_preflight"]["decision"])
    return 0
if __name__=="__main__":raise SystemExit(main())
