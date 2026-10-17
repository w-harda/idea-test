"""Stage4D2独立CPU只读验收；绝对脚本运行并引用冻结Stage4D1库。"""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import time
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,sha_file,json_bytes,digest
from tbps.materialization.contracts import need
from tbps.materialization.full_prepare import prepare,BASE as OLD_BASE,FULL
from tbps.materialization.full_runner import ValidatedActions
from tbps.materialization.bulk_session import StreamingMaterializedStore,BulkSession
from tbps.materialization.retained_inputs import unit_ref
from tbps.materialization.full_audit import verify_text_units,EXPECTED_ACTION_DIGEST
from tbps.materialization.legacy_text import HistoricalTextSource
from tbps.materialization.audit_retained import snapshot
from tbps.engines.historical_loader import feature_only_cpu,native_action_validator
from tbps.engines.replay_control import historical_snapshot,unchanged,rss_bytes
from tbps.runtime_closure import cpu_only
if __package__:
    from .claim_recovery import FirstClaimRecovery,verify_recovery_completion
    from .full_gpu import control,frozen_identity,ref,verify_subset,BASE,OUTPUT,PLAN,GALLERY,records
else:
    from claim_recovery import FirstClaimRecovery,verify_recovery_completion
    from full_gpu import control,frozen_identity,ref,verify_subset,BASE,OUTPUT,PLAN,GALLERY,records

def collection_units(store,name):
    store.store.verify(name)
    body=read_json(store.store.root/name/("joint_collection.json" if name=="joint-full" else
                                         "image_collection.json" if name=="image-full" else "text_collection.json"))
    need(body["plan_id"]==store.plan.id,"collection original Plan")
    pointer=FileRef(**body["collection"]);checked(pointer)
    if name!="text-full":
        store.store.verify(Path(pointer.path).parent.name)
        index=read_json(pointer.path);pointer=FileRef(**index["collection"]);checked(pointer)
    store.store.verify(Path(pointer.path).parent.name)
    core=read_json(pointer.path)
    return body,core,tuple(unit_ref(x) for x in core["images"]),tuple(unit_ref(x) for x in core["texts"])

def verify_images(store,images,actual_records,old_pixels):
    """只能读取已声明pixels.pt；独立检查字节/内容SHA/预算/Claim/证明/真实计数。"""
    need(tuple(r.binding.sample_id for r in images)==tuple(g.image_id for g in store.plan.manifest.gallery),
         "all ordered Gallery members")
    need([r["image_id"] for r in actual_records]==[g.image_id for g in store.plan.manifest.gallery],
         "actual G forward records order")
    need(not store.store.pending(),"no pending units")
    stats={"count":0,"maximum_linf":0.0,"minimum_linf":1.0};content=[]
    with feature_only_cpu({r.data.path for r in images}),BulkSession(store):
        for unit,actual in zip(images,actual_records):
            arrays,md=store.load(unit)
            need(actual["G_forward_samples"]==md["trace"]["G_forward_samples"]==1,"one actual G forward")
            need(actual["adversarial_pixels_sha256"]==md["pixels"]["adversarial_pixels_sha256"]
                 ==unit.content_sha256 and actual["original_pixels_sha256"]==md["pixels"]["original_pixels_sha256"]
                 and actual["effective_delta_sha256"]==md["pixels"]["effective_delta_sha256"],"actual observed pixel content")
            original=store.image_request(unit.binding.sample_id)
            checked(original.public.content)
            claim="generation-claim-"+unit.unit_id;proof="generation-proof-"+unit.unit_id
            store.store.verify(claim);store.store.verify(proof)
            c=read_json(store.store.root/claim/"generation.json")
            p=read_json(store.store.root/proof/"generation.json")
            need(c=={"schema":"tbps-generation-claim-1","unit_id":unit.unit_id,"plan_id":store.plan.id,
                     "binding":asdict(unit.binding),"may_have_generated":True},"full Claim identity")
            need(p["unit"]==asdict(unit) and p["claim_manifest"]==asdict(ref(store.store.root/claim/"manifest.json")),
                 "complete generation proof")
            if unit.binding.sample_id in old_pixels:
                need(md["pixels"]==old_pixels[unit.binding.sample_id],"historical two-image pixels/shape/budget exact")
            stats["count"]+=1;stats["maximum_linf"]=max(stats["maximum_linf"],md["pixels"]["linf"])
            stats["minimum_linf"]=min(stats["minimum_linf"],md["pixels"]["linf"])
            content.append([unit.binding.sample_id,md["pixels"],asdict(unit.data)])
            del arrays
    return {**stats,"ordered_image_pixel_and_file_digest":digest(content)}

def audit(path):
    started=time.perf_counter()
    with cpu_only():
        c=control(path);frozen_identity()
        rec=records()
        for name in ("attempt","gpu-start","gpu-result","result"):rec.verify(name)
        pre=read_json(Path(OUTPUT)/"attempt/preflight.json")
        report=read_json(Path(OUTPUT)/"gpu-result/report.json")
        need(read_json(Path(OUTPUT)/"result/report.json")["status"]=="JOINT_FULL_MATERIALIZED"
             and report["status"]=="IMAGE_FULL_MATERIALIZED" and report["plan_id"]==PLAN,"completed GPU/Joint required")
        need(report["actual_G_forwards"]["started"]==report["actual_G_forwards"]["completed"]==3074
             and report["counts"]["G_loads"]==report["counts"]["G_load_attempts"]==1
             and report["stream"]["generated"]==3074 and report["stream"]["skipped"]==0,"real GPU accounting")
        need(report["model_before"]==report["model_after"],"frozen G model proof")
        progress=[json.loads(line.split(" ",1)[1]) for line in (Path(OUTPUT)/"gpu.log").read_text().splitlines()
                  if line.startswith("G_FULL_PROGRESS ")]
        actual=report["actual_G_forwards"]["records"]
        need(len(progress)==len(actual)==3074 and all(p["completed"]==i+1==p["G_forward_started"]
             ==p["G_forward_completed"] and p["image_id"]==a["image_id"] and p["readback_SHA_verified"]
             and p["adversarial_pixels_sha256"]==a["adversarial_pixels_sha256"]
             for i,(p,a) in enumerate(zip(progress,actual))),"GPU log / hook / readback complete agreement")
        need(all(report[k]==0 for k in ("H_loads","H_forwards","IRRA_loads","IRRA_forwards",
                                       "backward","optimizer","JVP")),"no other models/optimization")
        old_files=read_json(Path(OUTPUT)/"attempt/old_files.json")
        preserved=verify_subset(old_files)
        _,plan,legacy,sources,_=prepare(c["frozen_config"]["path"])
        need(plan.id==PLAN,"same frozen Plan")
        store=StreamingMaterializedStore(ArtifactStore(Path(FULL)/"artifacts",new_base=OLD_BASE),plan)
        recovery_check=verify_recovery_completion(store,FirstClaimRecovery(FileRef(**c["recovery"])),rec)
        need(report["stream"]["recovered_first"]==1 and report["stream"]["legacy_stream_generated"]==3073
             and report["stream"]["legacy_stream_skipped"]==1,"exact single recovered Claim accounting")
        before=snapshot(FULL)
        _,image_core,images,_=collection_units(store,"image-full")
        _,text_core,_,texts=collection_units(store,"text-full")
        joint,joint_core,jimages,jtexts=collection_units(store,"joint-full")
        need(images==jimages and texts==jtexts and image_core["mode"]=="image_only"
             and text_core["mode"]=="text_only" and joint_core["mode"]=="joint"
             and joint["status"]=="JOINT_FULL_MATERIALIZED" and len(images)==3074 and len(texts)==6156,
             "Joint only references the same complete image/text units")
        with native_action_validator(legacy.config_ref) as native:
            validate=ValidatedActions(native)
            text=verify_text_units(store,HistoricalTextSource(sources,plan.h,validate),
                                   validate,{"edited":6116,"NOOP":40,"edits":13537})
        need(text["ordered_text_action_digest"]==EXPECTED_ACTION_DIGEST,"historical complete H exact")
    # cpu_only禁止所有torch.load；退出后仅开放本集合pixels.pt的weights_only读取。
    image=verify_images(store,images,report["actual_G_forwards"]["records"],pre["old_pixel_reference"])
    for unit,p in zip(images,progress):need(unit.data.sha256==p["pixel_file_sha256"],"logged file SHA")
    with cpu_only():
        need(snapshot(FULL)==before,"read-only Full audit changed files")
        preserved=verify_subset(old_files)
        protected=historical_snapshot("/home/lzf/ldx/tmp/tbps-framework-stage2/protected_before.json")
        c3=read_json("/home/lzf/ldx/outputs/tbps-framework-stage4c3/irra_only_v1/attempt/preflight.json")
        caches=unchanged(c3["historical_feature_snapshot"]);fail=unchanged(c3["retained_file_snapshot"])
        need(not store.store.pending(),"pending after full audit")
        counts={"data_files":sum(p.is_file() for p in Path(FULL).rglob("*"))}
        result={"status":"JOINT_FULL_CPU_AUDIT_PASS","plan_id":PLAN,"images":image,"text":text,
            "first_claim_recovery":recovery_check,
            "G_loads":1,"actual_G_forwards":3074,"generated":3074,"skipped":0,"H_loads":0,"H_forwards":0,
            "IRRA_loads":0,"IRRA_forwards":0,"backward":0,"optimizer":0,"JVP":0,
            "old_Stage4D1_files_SHA_mtime_unchanged":preserved,
            "original_text_store_files_preserved":len(read_json(FirstClaimRecovery(FileRef(**c["recovery"])).spec["baseline_files"]["path"])),
            "preexisting_first_claim_files_preserved":3,"historical_files_unchanged":len(protected),
            "historical_cache_files_unchanged":caches,"old_FAIL_files_unchanged":fail,"pending":[],
            "read_only_full_files_unchanged":len(before),"full_file_snapshot_digest":digest(before),
            "old_two_images_exact":True,"elapsed_seconds":time.perf_counter()-started,"peak_RSS_bytes":rss_bytes(),
            **counts}
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description="Independent CPU read-only Joint materialization audit; no GPU")
    p.add_argument("--config",required=True);p.add_argument("--output",required=True);a=p.parse_args(argv)
    need(not Path(a.output).exists(),"immutable CPU audit output already exists")
    result=audit(a.config)
    ArtifactStore(a.output,new_base=BASE).publish("result",{"report.json":json_bytes(result)},
        {"kind":"independent_CPU_full_materialization_audit"},
        sources=(ref(a.config),ref(__file__),ref(Path(OUTPUT)/"result/manifest.json"),
                 ref(Path(FULL)/"artifacts/joint-full/manifest.json")))
    print(json.dumps(result,ensure_ascii=False,indent=2));print("JOINT_FULL_CPU_AUDIT_PASS")
    return 0
if __name__=="__main__":raise SystemExit(main())
