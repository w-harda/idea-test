"""Stage4D2绝对脚本入口；生产运行只能导入冻结Stage4D1 tbps库。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
import argparse
import io
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,sha_file,json_bytes,digest
from tbps.materialization.contracts import need
from tbps.materialization.full_prepare import prepare,BASE as OLD_BASE,FULL,fs_resources
from tbps.materialization.full_audit import audit as text_audit
from tbps.materialization.audit_retained import snapshot
from tbps.materialization.full_runner import stream_images,ValidatedActions
from tbps.materialization.bulk_session import StreamingMaterializedStore,BulkSession
from tbps.materialization.retained_inputs import RetainedInputs,unit_ref,ORIGINAL
from tbps.materialization.irra_startup import CPUStartupGate
from tbps.engines.historical_loader import feature_only_cpu,native_action_validator
from tbps.runtime_closure import cpu_only,source_execution_audit
from tbps.check_attack_runtime import check as runtime_check
from tbps.attacks.legacy_runtime import load_legacy_plugins
from tbps.smoke_irra import gpu_available,model_snapshot
from tbps.models.irra import fp32_runtime
if __package__:
    from .gpu_safety import FullGuard,GObservation,no_optimization,g_load_only,own_gpu_only,ram
    from .claim_recovery import FirstClaimRecovery,stream_with_recovery
else:
    from gpu_safety import FullGuard,GObservation,no_optimization,g_load_only,own_gpu_only,ram
    from claim_recovery import FirstClaimRecovery,stream_with_recovery

FROZEN="/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017"
BASE="/home/lzf/ldx/outputs/tbps-framework-stage4d2"
OUTPUT=BASE+"/generation_v3"
PLAN="7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5"
GALLERY=("cuhk:37132:0","cuhk:37147:0")
TEXT_RECEIPT="a149a57fc7e8c87a389ede80c7e2eaedafec9d3ffc81dff9a7275915040c615f"
AUDIT_RECEIPT="913b87ecf0620730e537a33f96858e4dff41fbb6de8fa233b085da077fbbc9a2"

def ref(path):return FileRef(str(Path(path).resolve()),sha_file(path))
def records():return ArtifactStore(OUTPUT,new_base=BASE)

def frozen_identity():
    import tbps.artifacts as artifacts
    import tbps.materialization.full_prepare as p
    import tbps.materialization.full_runner as runner
    import tbps.attacks.legacy_runtime as legacy
    for module in (artifacts,p,runner,legacy):
        need(Path(module.__file__).resolve().is_relative_to(Path(FROZEN)/"src/tbps"),
             "production must use frozen Stage4D1 library: "+str(module.__file__))
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=FROZEN,text=True,capture_output=True,check=True).stdout.strip()
    status=subprocess.run(["git","status","--porcelain"],cwd=FROZEN,text=True,capture_output=True,check=True).stdout
    need(head=="82fe88a5a74955838fc6341c1ed9a46921c94d7c" and not status,"frozen Stage4D1 Git identity")
    return {"root":FROZEN,"commit":head,"module_paths":[str(m.__file__) for m in (artifacts,p,runner,legacy)]}

def control(path):
    c=read_json(path,"tbps-g-full-generation-1")
    fixed={"schema":"tbps-g-full-generation-1","frozen_root":FROZEN,"plan_id":PLAN,"output":OUTPUT,
        "gallery":3074,"query":6156,"batch":1,"seed":42,"precision":"fp32/math/no_tf32",
        "max_seconds":7200,"minimum_available_ram_bytes":8*1024**3,
        "system_ram_reserve_bytes":2*1024**3,"minimum_disk_bytes":10*1024**3,
        "runtime_disk_reserve_bytes":2*1024**3,"minimum_inodes":100000,
        "physical_gpu":0,"minimum_free_mib":8192,"maximum_utilization_percent":10}
    need(set(c)==set(fixed)|{"frozen_config","entry_sources","recovery","preprocess_sources"} and all(c[k]==v for k,v in fixed.items()),
         "fixed Stage4D2 scope/resources/precision")
    checked(FileRef(**c["frozen_config"]))
    need(c["frozen_config"]["path"]==FROZEN+"/configs/framework/stage4d1_materialization.server.json",
         "use original producer paths/config, never reconstruct in new tree")
    expected={str(Path(__file__).resolve()),str(Path(__file__).with_name("gpu_safety.py").resolve()),
              str(Path(__file__).with_name("full_gpu_audit.py").resolve()),
              str(Path(__file__).with_name("claim_recovery.py").resolve())}
    need(set(c["entry_sources"])==expected,"complete separately bound new entry/supervisor/audit sources")
    for p,sha in c["entry_sources"].items():checked(FileRef(p,sha))
    checked(FileRef(**c["recovery"]))
    need(c["recovery"]["path"]==str(Path(__file__).resolve().parents[3]/"configs/framework/stage4d2r2_recovery.server.json"),
         "explicit generation_v2 first-Claim recovery contract")
    need(set(c["preprocess_sources"])=={"backend","pixel_reader","torchvision"},"preprocess source fields")
    expected_reader=FROZEN+"/src/tbps/attacks/legacy_runtime.py"
    need(c["preprocess_sources"]["backend"]["path"]==expected_reader and
         c["preprocess_sources"]["pixel_reader"]["path"]==
         "/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/irra_image_adapter.py" and
         c["preprocess_sources"]["torchvision"]["path"]==
         "/home/lzf/ldx/envs/tbps-image/lib/python3.10/site-packages/torchvision/transforms/transforms.py",
         "fixed native preprocessing source paths")
    for value in c["preprocess_sources"].values():checked(FileRef(**value))
    return c

def fresh_images(store,recovery=None):
    if recovery is not None:
        return FirstClaimRecovery(FileRef(**recovery)).verify(store)
    need(not store.store.pending(),"pending store; no GPU loading")
    for g in store.plan.manifest.gallery:
        binding=store.image_request(g.image_id).binding;unit=store.unit_id(binding)
        need(not store.available(binding) and not any((store.store.root/(s+unit)).exists()
             for s in ("generation-claim-","generation-proof-")),"previous image/Claim exists; separate recovery authorization required")
    need(not any((store.store.root/n).exists() for n in ("image-full","joint-full")),"image/Joint already sealed")

def old_pixel_reference():
    with cpu_only():old=RetainedInputs(ORIGINAL)
    allowed={u.data.path for u in old.units if u.modality=="image"}
    expected={}
    with feature_only_cpu(allowed):
        for key in GALLERY:
            unit=old.reference(old.plan.binding(key,"image"))
            arrays,metadata=old.load(unit)
            expected[key]=metadata["pixels"]
            del arrays
    return expected,old.descriptor()

def preflight(path,*,require_fresh=True):
    import torch
    gate=CPUStartupGate();gate.emit("cpu_preflight_started")
    with cpu_only():
        c=control(path);library=frozen_identity()
        checked(FileRef(FULL+"/result/manifest.json",TEXT_RECEIPT))
        checked(FileRef(OLD_BASE+"/full_audit_v1/result/manifest.json",AUDIT_RECEIPT))
        ArtifactStore(OLD_BASE+"/full_audit_v1",new_base=OLD_BASE).verify("result")
        need(read_json(OLD_BASE+"/full_audit_v1/result/report.json")["status"]=="TEXT_FULL_CPU_AUDIT_PASS",
             "Stage4D1 independent text audit required")
        text_report,_=text_audit(c["frozen_config"]["path"])
        parent=read_json(read_json(c["frozen_config"]["path"])["source_config"]["path"])
        closure=runtime_check(parent["gh_config"]["path"],parent["derivation"]["path"])
        _,plan,legacy,sources,_=prepare(c["frozen_config"]["path"])
        need(plan.id==PLAN,"original materialization Plan must remain exact")
        store=StreamingMaterializedStore(ArtifactStore(Path(FULL)/"artifacts",new_base=OLD_BASE),plan)
        store.verify_plan(force=True)
        if require_fresh:fresh_images(store,c.get("recovery"))
        for row in plan.manifest.gallery:checked(store.image_request(row.image_id).public.content)
        memory=ram();disk=fs_resources(FULL)
        need(memory["MemAvailable"]>=c["minimum_available_ram_bytes"],"actual available RAM insufficient")
        need(disk["available_bytes"]>=c["minimum_disk_bytes"] and disk["available_inodes"]>=c["minimum_inodes"],
             "disk/inode headroom insufficient")
        before=snapshot(FULL)
        need(digest(before)==text_report["output_snapshot_digest"],"Stage4D1 files changed during preflight")
    expected,old=old_pixel_reference()
    gate.checked();gate.exit_checked()
    need(not torch.cuda.is_initialized(),"CPU preflight initialized CUDA")
    gpu=gpu_available(c)
    report={"status":"G_FULL_CPU_PREFLIGHT_PASS","plan_id":plan.id,"library":library,
        "text_audit":text_report,"GH_runtime_source_count":closure["runtime_source_count"],
        "GH_executed_files":len(closure["executed_sources"]),"GH_status":closure["status"],
        "CPU_contexts_restored":gate.exited,"CUDA_initialized":False,"G_H_IRRA_loads":0,
        "claim_recovery":c.get("recovery"),"verified_preprocess_sources":c.get("preprocess_sources"),
        "old_pixel_reference":expected,"retained_reference":old,"RAM":memory,"GPU":gpu,"disk":disk,
        "gallery_order_sha256":digest([g.image_id for g in plan.manifest.gallery]),
        "query_order_sha256":digest([q.query_id for q in plan.manifest.queries])}
    return c,plan,legacy,store,report,before

def verify_subset(before):
    for path,v in before.items():need(checked(FileRef(path,v["sha256"]))==v,"old Stage4D1 file changed: "+path)
    return len(before)

def child(path,token):
    import torch
    c=control(path);frozen_identity();record=records();record.verify("attempt")
    pre=read_json(Path(OUTPUT)/"attempt/preflight.json")
    ctl=read_json(Path(OUTPUT)/"attempt/control.json")
    need(ctl["token_sha256"]==digest(token) and ctl["config"]==asdict(ref(path)) and
         pre["status"]=="G_FULL_CPU_PREFLIGHT_PASS" and pre["plan_id"]==PLAN
         and pre["CPU_contexts_restored"] and pre["GH_status"]=="PASS","immutable CPU launch gate")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit GPU0 child")
    need(not (Path(OUTPUT)/"gpu-start").exists(),"unique GPU Attempt already consumed")
    gate=CPUStartupGate()
    with cpu_only():
        before=read_json(Path(OUTPUT)/"attempt/old_files.json")
        verify_subset(before)
        _,plan,legacy,_,_=prepare(c["frozen_config"]["path"])
        need(plan.id==PLAN,"child original Plan identity")
        store=StreamingMaterializedStore(ArtifactStore(Path(FULL)/"artifacts",new_base=OLD_BASE),plan)
        fresh_images(store,c.get("recovery"))
        need(ram()["MemAvailable"]>=c["minimum_available_ram_bytes"],"RAM changed before GPU")
    gate.checked();gate.exit_checked()
    gpu=gpu_available(c)
    record.publish("gpu-start",{"claim.json":json_bytes({"plan_id":PLAN,"may_have_loaded_or_generated":True})},
        {"kind":"single_G_full_GPU_attempt"},sources=(ref(Path(OUTPUT)/"attempt/manifest.json"),ref(__file__)))
    counts={"G_load_attempts":0,"G_loads":0,"pixel_tensor_loads":0}
    def mismatch(image_id,value,comparison):
        arrays={"adversarial":value.payload.detach().cpu().clone(),
                "effective_delta":value.effective_delta.detach().cpu().clone()}
        stream=io.BytesIO();torch.save(arrays,stream)
        record.publish("pixel-mismatch",{"pixels.pt":stream.getvalue(),
            "comparison.json":json_bytes({"image_id":image_id,**comparison,"trace":value.trace()})},
            {"kind":"failed_pixel_comparison_evidence"},sources=(ref(Path(OUTPUT)/"attempt/manifest.json"),))
    options={}
    if c.get("preprocess_sources"):
        options={"image_sources":{g.image_id:store.image_request(g.image_id).public.content
                                  for g in plan.manifest.gallery},"preprocess_sources":c["preprocess_sources"]}
    observation=GObservation([g.image_id for g in plan.manifest.gallery],pre["old_pixel_reference"],mismatch,**options)
    report={"status":"RUNNING","plan_id":PLAN,"H_loads":0,"H_forwards":0,"IRRA_loads":0,
            "IRRA_forwards":0,"backward":0,"optimizer":0,"JVP":0,"GPU_before":gpu}
    guard=FullGuard(c,FULL);guard.seconds=max(.01,c["max_seconds"]-(time.time()-ctl["started_unix"]))
    last=[time.perf_counter()];last_gpu=[0]
    try:
        with guard:
            torch.set_num_threads(2);torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            gate.initialize_cuda()
            # CPU门禁通过后立即安装限制；两项均成功才可进入G工厂。
            with no_optimization() as blocked,g_load_only(plan.g.checkpoint.path,store.store.root,counts):
                @contextmanager
                def factory():
                    with source_execution_audit(legacy) as imports,load_legacy_plugins(
                         legacy,components=("gv2",),device="cuda:0",allow_model_load=True) as plugins:
                        counts["G_loads"]+=1;need(counts["G_loads"]==1,"one restored G")
                        plugin=plugins["gv2"];report["model_before"]=model_snapshot(plugin.backend.model)
                        with observation.attach(plugin):
                            yield plugin
                        report["model_after"]=model_snapshot(plugin.backend.model)
                        need(report["model_before"]==report["model_after"],"G parameters/buffers changed")
                    report["runtime_import_files"]=sorted(imports)
                def resource():
                    guard.check()
                    if time.monotonic()-last_gpu[0]>=10:
                        own_gpu_only();last_gpu[0]=time.monotonic()
                def progress(n,elapsed,stats):
                    now=time.perf_counter()
                    need(observation.started==observation.completed==stats["generated"],"actual hook/readback accounting")
                    q=store.plan.manifest.gallery[n-1];unit=store.reference(store.image_request(q.image_id).binding)
                    md=read_json(unit.metadata.path)
                    metrics={"completed":n,"G_forward_started":observation.started,
                        "G_forward_completed":observation.completed,"image_id":q.image_id,
                        "unit_seconds":now-last[0],"elapsed_seconds":elapsed,"readback_SHA_verified":True,
                        "linf":md["pixels"]["linf"],"pixel_file_sha256":unit.data.sha256,
                        "adversarial_pixels_sha256":unit.content_sha256,"RSS_bytes":guard.peak,
                        "GPU_allocated_bytes":torch.cuda.memory_allocated(),
                        "GPU_reserved_bytes":torch.cuda.memory_reserved(),"disk":guard.disk}
                    print("G_FULL_PROGRESS",json.dumps(metrics,separators=(",",":")),flush=True);last[0]=now
                with fp32_runtime():
                    if c.get("recovery"):
                        report["stream"]=stream_with_recovery(store,factory,
                            FirstClaimRecovery(FileRef(**c["recovery"])),record,resource_check=resource,progress=progress)
                    else:
                        report["stream"]=stream_images(store,factory,allow_real_generation=True,
                                                       resource_check=resource,progress=progress)
                need(observation.started==observation.completed==3074 and report["stream"]["generated"]==3074
                     and report["stream"]["skipped"]==0 and counts["G_load_attempts"]==counts["G_loads"]==1,
                     "exact 3074 generation / one restoration")
                need(set(observation.comparisons)==set(GALLERY) and
                     all(v["exact"] for v in observation.comparisons.values()),"two historical comparisons")
                report["forbidden_operations"]=dict(blocked)
                verify_subset(before);store.verify_plan(force=True)
                report["status"]="IMAGE_FULL_MATERIALIZED"
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
    finally:
        report["counts"]=counts;report["actual_G_forwards"]=observation.report()
        report["resources"]=guard.snapshot()
        if torch.cuda.is_initialized():
            report["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
            report["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        record.publish("gpu-result",{"report.json":json_bytes(report)},{"kind":"G_full_GPU_result","status":report["status"]},
            sources=(ref(Path(OUTPUT)/"gpu-start/manifest.json"),ref(__file__)))
    print(report["status"],flush=True);return int(report["status"]=="FAIL")

def seal_joint(path,pre):
    """GPU子进程退出后CPU封存；只引用完成单位，不再次调用G/H。"""
    from tbps.materialization.generation import seal_generated_collection
    from tbps.materialization.legacy_text import HistoricalTextSource
    from tbps.materialization.full_audit import verify_text_units,EXPECTED_ACTION_DIGEST
    with cpu_only():
        c=control(path);_,plan,legacy,sources,_=prepare(c["frozen_config"]["path"])
        store=StreamingMaterializedStore(ArtifactStore(Path(FULL)/"artifacts",new_base=OLD_BASE),plan)
        need(plan.id==PLAN and not store.store.pending(),"same complete Plan/store")
        store.store.verify("image-full");store.store.verify("text-full")
        text_body=read_json(store.store.root/"text-full/text_collection.json")
        checked(FileRef(**text_body["collection"]))
        text_core=read_json(text_body["collection"]["path"])
        texts=tuple(unit_ref(v) for v in text_core["texts"])
        image_body=read_json(store.store.root/"image-full/image_collection.json")
        checked(FileRef(**image_body["collection"]))
        image_index=read_json(image_body["collection"]["path"])
        checked(FileRef(**image_index["collection"]))
        images=tuple(unit_ref(v) for v in read_json(image_index["collection"]["path"])["images"])
    with native_action_validator(legacy.config_ref) as native:
        validate=ValidatedActions(native)
        with cpu_only():
            text_check=verify_text_units(store,HistoricalTextSource(sources,plan.h,validate),validate,
                                        {"edited":6116,"NOOP":40,"edits":13537})
            need(text_check["ordered_text_action_digest"]==EXPECTED_ACTION_DIGEST,"H text/action changed")
        # 仅允许本Collection内pixels.pt Tensor读取，不允许任何模型。
        with feature_only_cpu({r.data.path for r in images}),BulkSession(store):
            collection=seal_generated_collection(store,"joint",images=images,texts=texts,validate=validate)
    store.verify_plan(force=True)
    verify_subset(read_json(Path(OUTPUT)/"attempt/old_files.json"))
    body={"schema":"tbps-joint-full-materialized-1","status":"JOINT_FULL_MATERIALIZED","plan_id":PLAN,
        "gallery":3074,"query":6156,"collection":asdict(collection),"text_statistics":text_check["stats"],
        "G_forwards_during_composition":0,"H_loads":0,"H_forwards":0,"victim_model":None,
        "historical_text_receipt_preserved":True}
    store.store.publish("joint-full",{"joint_collection.json":json_bytes(body)},
        {"kind":"joint_full_reference_collection","plan_id":PLAN},sources=(collection,
        ref(store.store.root/"image-full/manifest.json"),ref(store.store.root/"text-full/manifest.json"),
        ref(Path(OUTPUT)/"gpu-result/manifest.json"),ref(__file__)))
    return body

def wait_owned_child(command,env,log,seconds,*,spawn=subprocess.Popen):
    """仅管理由本入口创建的Popen，不按外部PID终止任务。"""
    process=spawn(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:return process.wait(timeout=seconds)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill();process.wait(timeout=10)
        raise

def main(argv=None):
    p=argparse.ArgumentParser(description="default CPU preflight; unique explicitly authorized G full generation")
    p.add_argument("--config",required=True);p.add_argument("--allow-gpu-generation",action="store_true")
    p.add_argument("--private-child-token",help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.private_child_token:return child(a.config,a.private_child_token)
    if a.allow_gpu_generation:need(not Path(OUTPUT).exists(),"unique Attempt already exists; no retry/resume")
    started_unix=time.time()
    c,plan,legacy,store,pre,before=preflight(a.config)
    print("GH_RUNTIME_PREFLIGHT_PASS\nG_FULL_CPU_PREFLIGHT_PASS",flush=True)
    if not a.allow_gpu_generation:
        print(json.dumps({k:v for k,v in pre.items() if k not in ("text_audit","retained_reference")},indent=2));return 0
    record=records();token=secrets.token_hex(32)
    sources=(ref(a.config),*(FileRef(k,v) for k,v in c["entry_sources"].items()),
        ref(FULL+"/artifacts/protocol/manifest.json"),ref(OLD_BASE+"/full_audit_v1/result/manifest.json"))
    record.publish("attempt",{"preflight.json":json_bytes(pre),"old_files.json":json_bytes(before),
        "control.json":json_bytes({"token_sha256":digest(token),"config":asdict(ref(a.config)),"started_unix":started_unix})},
        {"kind":"one_authorized_G_full_generation","plan_id":PLAN},sources=sources)
    report={"status":"RUNNING","plan_id":PLAN}
    parent_guard=FullGuard(c,FULL)
    parent_guard.seconds=max(.01,c["max_seconds"]-(time.time()-started_unix))
    try:
        with parent_guard:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",PYTHONDONTWRITEBYTECODE="1")
            with (Path(OUTPUT)/"gpu.log").open("xb") as log:
                exitcode=wait_owned_child([sys.executable,"-B",str(Path(__file__).resolve()),"--config",a.config,
                    "--private-child-token",token],env,log,max(.01,c["max_seconds"]-(time.time()-started_unix)))
            record.verify("gpu-result")
            need(exitcode==0 and read_json(Path(OUTPUT)/"gpu-result/report.json")["status"]=="IMAGE_FULL_MATERIALIZED",
                 "GPU Attempt failed; preserve site and do not retry")
            report["joint"]=seal_joint(a.config,pre)
            report["status"]="JOINT_FULL_MATERIALIZED"
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
    report["supervisor_resources"]=parent_guard.snapshot()
    record.publish("result",{"report.json":json_bytes(report)},{"kind":"full_materialization_result","status":report["status"]},
        sources=(ref(Path(OUTPUT)/"attempt/manifest.json"),ref(Path(OUTPUT)/"gpu.log"),ref(__file__),
        *(ref(p) for p in (Path(OUTPUT)/"gpu-result/manifest.json",Path(FULL)/"artifacts/joint-full/manifest.json") if p.exists())))
    print(report["status"],flush=True)
    return int(report["status"]=="FAIL")
if __name__=="__main__":raise SystemExit(main())
