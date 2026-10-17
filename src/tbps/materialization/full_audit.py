"""全量文本产物只读CPU审计；不加载模型、Tensor特征或重算排名。"""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import time
from ..artifacts import ArtifactStore,FileRef,read_json,sha_file,json_bytes,digest,checked
from ..runtime_closure import cpu_only
from ..engines.historical_loader import native_action_validator
from ..engines.replay_control import rss_bytes,unchanged,historical_snapshot
from .contracts import need
from .bulk_session import BulkSession,StreamingMaterializedStore
from .full_runner import ValidatedActions
from .full_prepare import prepare,BASE,FULL
from .legacy_text import HistoricalTextSource
from .audit_retained import snapshot

EXPECTED_ACTION_DIGEST="afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868"

def verify_text_units(store,source,validate,expected):
    """逐条独立读回，与旧源完整动作/字符串对照；不生成、不发布单元。"""
    stats={"edited":0,"NOOP":0,"edits":0};ordered=[]
    need(not store.store.pending(),"pending units cannot pass full audit")
    with BulkSession(store):
        for index,(request,result) in enumerate(source.iter_outputs()):
            need(index<len(store.plan.manifest.queries) and
                 request==store.text_request(store.plan.manifest.queries[index].query_id),"audit Query order")
            need(store.available(request.binding),"missing text unit")
            ref=store.reference(request.binding)
            text,metadata=store.load(ref,validate=validate)
            trace=result.trace();action=trace["action"]
            need(text==result.payload and metadata["action"]==action and metadata["trace"]==trace,
                 "historical UTF8/full action/trace exact")
            need(metadata["content_sha256"]==trace["adversarial_utf8_sha256"]==ref.content_sha256,
                 "adversarial UTF8 content SHA mismatch")
            stats["NOOP" if text==request.public.caption else "edited"]+=1
            stats["edits"]+=len(action["edits"])
            ordered.append([request.binding.sample_id,trace["original_utf8_sha256"],
                            trace["adversarial_utf8_sha256"],digest(action)])
        need(len(ordered)==len(store.plan.manifest.queries) and stats==expected,"complete counts/order required")
        need(source.report()["status"]=="TEXT_SOURCE_VERIFIED","source audit incomplete")
    return {"stats":stats,"units":len(ordered),"ordered_text_action_digest":digest(ordered)}

def audit(config):
    started=time.perf_counter()
    with cpu_only():
        c,plan,legacy,sources,resources=prepare(config)
        before=snapshot(FULL)
        records=ArtifactStore(FULL,new_base=BASE);records.verify("result")
        report=read_json(Path(FULL)/"result/report.json")
        need(report["status"]=="TEXT_FULL_MATERIALIZED" and report["plan_id"]==plan.id,"full completion required")
        store=StreamingMaterializedStore(ArtifactStore(Path(FULL)/"artifacts",new_base=BASE),plan)
        store.store.verify("text-full")
        semantic=read_json(Path(FULL)/"artifacts/text-full/text_collection.json")
        need(semantic["status"]=="TEXT_FULL_MATERIALIZED" and semantic["origin"]=="historical_H7_text_reuse"
             and semantic["plan_id"]==plan.id and semantic["image_collection_status"]=="PENDING"
             and semantic["joint_collection_status"]=="PENDING","H-only historical semantic receipt")
        collection=FileRef(**semantic["collection"]);checked(collection)
        store.store.verify(Path(collection.path).parent.name)
        core=read_json(collection.path)
        need(core["mode"]=="text_only" and core["images"]==[] and len(core["texts"])==6156,"text-only full collection")
        need(semantic["ordered_query_ids"]==[q.query_id for q in plan.manifest.queries]
             ==[v["binding"]["sample_id"] for v in core["texts"]],"official complete Query order")
        with native_action_validator(legacy.config_ref) as native:
            validate=ValidatedActions(native)
            result=verify_text_units(store,HistoricalTextSource(sources,plan.h,validate),validate,c["text_statistics"])
        need(result["ordered_text_action_digest"]==EXPECTED_ACTION_DIGEST,"Stage4C1 ordered text/action digest")
        need(all(metadata["modality"]=="text" for metadata in
                 (read_json(v["metadata"]["path"]) for v in core["texts"])),"unexpected image materialization")
        need(all(report[k]==0 for k in ("G_H_IRRA_model_loads","G_H_IRRA_forwards","GPU_attempts")),
             "CPU-only materialization accounting")
        need(snapshot(FULL)==before,"Full output SHA/mtime changed during read-only audit")
        protected=historical_snapshot("/home/lzf/ldx/tmp/tbps-framework-stage2/protected_before.json")
        pre=read_json("/home/lzf/ldx/outputs/tbps-framework-stage4c3/irra_only_v1/attempt/preflight.json")
        feature_count=unchanged(pre["historical_feature_snapshot"])
        retained_count=unchanged(pre["retained_file_snapshot"])
        import subprocess
        baseline=read_json("/home/lzf/ldx/outputs/tbps-framework-stage4c3/cpu_verified_v1/result/report.json")["protected_worktrees"]
        baseline["idea-test-tbps-framework-stage4c3-20261017"]={
            "head":"a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2","clean":True}
        worktrees={}
        for name,state in baseline.items():
            root=Path("/home/lzf/ldx/projects")/name
            head=subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
            clean=not subprocess.run(["git","status","--porcelain"],cwd=root,text=True,capture_output=True,check=True).stdout
            need(head==state["head"] and clean==state["clean"],"protected worktree changed: "+name)
            worktrees[name]={"head":head,"clean":clean}
        result.update(protected_worktrees=worktrees,status="TEXT_FULL_CPU_AUDIT_PASS",plan_id=plan.id,
             output_files_sha_mtime_unchanged=len(before),output_snapshot_digest=digest(before),
             historical_files_unchanged=len(protected),historical_cache_files_unchanged=feature_count,
             retained_FAIL_files_unchanged=retained_count,H_model_loads=0,H_forwards=0,
             G_model_loads=0,G_forwards=0,IRRA_model_loads=0,IRRA_forwards=0,
             image_collection_status="PENDING",joint_collection_status="PENDING",
             elapsed_seconds=time.perf_counter()-started,peak_RSS_bytes=rss_bytes())
    refs=tuple(FileRef(str(p),sha_file(p)) for p in
               (Path(config).resolve(),Path(__file__),Path(FULL)/"result/manifest.json",
                Path(FULL)/"artifacts/text-full/manifest.json",Path(collection.path)))
    return result,refs

def main(argv=None):
    p=argparse.ArgumentParser(description="Read-only CPU H full audit; never model forward or generation")
    p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    a=p.parse_args(argv);need(not Path(a.output).exists(),"immutable audit output already exists")
    result,refs=audit(a.config)
    ArtifactStore(a.output,new_base=BASE).publish("result",{"report.json":json_bytes(result)},
        {"kind":"full_historical_text_readonly_audit"},sources=refs)
    print(json.dumps(result,ensure_ascii=False,indent=2));print("TEXT_FULL_CPU_AUDIT_PASS")
    return 0
if __name__=="__main__":raise SystemExit(main())
