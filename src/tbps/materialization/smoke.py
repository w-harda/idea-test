"""Stage4C2一次Attempt，G/CPU文本/IRRA分别在独立子进程；默认严格CPU预检。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import io
import json
import os
import secrets
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
from ..artifacts import ArtifactStore, FileRef, checked, sha_file, json_bytes, read_json, digest
from ..engines.replay_control import ResourceGuard, historical_snapshot, unchanged
from .contracts import need
from .smoke_support import (BASE,OUTPUT,GALLERY,QUERY,LIMITS,RSS,SECONDS,config,preflight,
                            materialized,compare,selected_features,Ledger)

def tensor_bytes(value):
    import torch
    stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()

def record_store(output):return ArtifactStore(output,new_base=BASE)

def stage_g(c,output,report,ledger):
    import torch
    from ..attacks.legacy_runtime import load_legacy_plugins
    from ..runtime_closure import source_execution_audit
    from ..models.irra import fp32_runtime
    from ..smoke_irra import model_snapshot
    from .generation import generate_image_once
    store,legacy=materialized(c,output);store.publish_plan()
    results=[];report["G_load_attempts"]=1
    with source_execution_audit(legacy),load_legacy_plugins(legacy,components=("gv2",),device="cuda:0",allow_model_load=True) as plugins:
        report["G_loads"]=1;plugin=plugins["gv2"];model=plugin.backend.model
        report["model_before"]=model_snapshot(model);original=plugin.backend.generate_once
        captures={}
        def observe(g,image):
            values=original(g,image)
            captures.clear()
            captures.update({k:v.detach().cpu().clone() for k,v in values.items()})
            captures["original"]=image.detach().cpu().clone()
            return values
        with patch.object(model,"forward",new=ledger.wrap("G",model.forward)),patch.object(plugin.backend,"generate_once",new=observe),fp32_runtime(),torch.inference_mode():
            for image_id in GALLERY:
                ref=generate_image_once(store,image_id,plugin,lambda x:plugin.backend.load_pixels(x.path),allow_real_generation=True)
                arrays,md=store.load(ref)
                checks={k:compare(arrays[k],captures[k],exact=True) for k in arrays}
                raw=captures["delta"].detach().clone()
                need(float(raw.double().abs().max())<=8/255,"raw delta bound")
                evidence=record_store(output).publish("raw-delta-"+str(len(results)),{"delta.pt":tensor_bytes(raw)},
                    {"kind":"smoke_raw_generator_delta","image_id":image_id},sources=(ref.manifest,))
                results.append({"unit":asdict(ref),"pixels":md["pixels"],"readback":checks,
                    "raw_delta_sha256":__import__("tbps.attacks.base",fromlist=["tensor_sha"]).tensor_sha(raw),
                    "raw_delta_linf":float(raw.double().abs().max()),"raw_delta_file_sha256":evidence["files_sha256"]["delta.pt"]})
        report["model_after"]=model_snapshot(model)
        need(report["model_before"]==report["model_after"],"G model changed")
    need(ledger.completed["G"]==2,"exactly two G samples")
    report["images"]=results
    store.verify_plan(force=True)

def stage_text(c,output,report,ledger):
    from ..engines.historical_sources import JointFullSources
    from ..engines.historical_loader import native_action_validator
    from ..runtime_closure import cpu_only
    from .legacy_text import HistoricalTextSource
    store,legacy=materialized(c,output);sources=JointFullSources(c["replay_config"]["path"])
    selected={}
    with cpu_only(),native_action_validator(legacy.config_ref) as validate:
        for request,result in HistoricalTextSource(sources,store.plan.h,validate).iter_outputs():
            if request.binding.sample_id in QUERY:
                ref=store.publish_text(request,result,validate)
                text,md=store.load(ref,validate=validate)
                need(text==result.payload and md["action"]==result.trace()["action"],"H history exact")
                selected[request.binding.sample_id]={"unit":asdict(ref),"action":md["action"],
                    "original_utf8_sha256":md["original_utf8_sha256"],"adversarial_utf8_sha256":md["adversarial_utf8_sha256"]}
            if len(selected)==2:break
    need(selected[QUERY[0]]["action"]["edits"] and not selected[QUERY[1]]["action"]["edits"],"edit and NOOP")
    report["texts"]=[selected[k] for k in QUERY];report["H_loads"]=0
    store.verify_plan(force=True)

def stage_irra(c,output,report,ledger):
    import torch
    from ..models.irra import IRRAAdapter,fp32_runtime
    from ..models.irra_source import IRRASource
    from ..engines.historical_loader import native_action_validator
    from ..smoke_irra import model_snapshot
    from .irra_startup import prepare_inputs
    events=report.setdefault("startup_events",[])
    inputs,gate=prepare_inputs(c,output,store_factory=materialized,feature_reader=selected_features,emit=events.append)
    historical,feature_refs=inputs.historical,inputs.feature_refs
    images,texts,source,store=inputs.images,inputs.texts,inputs.source,inputs.store
    gate.initialize_cuda()
    events.append("irra_model_load_entry")
    report["IRRA_load_attempts"]=1
    adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
    report["IRRA_loads"]=1;backend=adapter._backend
    report["model_before"]=model_snapshot(backend);actual={}
    with patch.object(backend,"encode_image",new=ledger.wrap("IRRA_image",backend.encode_image)), \
         patch.object(backend,"encode_text",new=ledger.wrap("IRRA_text",backend.encode_text)),fp32_runtime(),torch.inference_mode():
        for family,pixel_key in (("clean_gallery","original"),("image_gallery","adversarial")):
            rows=[adapter.inference().encode_images(x[pixel_key].to("cuda:0")).payload.cpu().clone() for x in images]
            actual[family]=torch.cat(rows).detach().clone()
        for family,key in (("clean_text","original_caption"),("adv_text","adversarial_caption")):
            rows=[]
            for text in texts:
                batch,_=adapter.inference().encode_texts((text[key],))
                rows.append(batch.payload.cpu().clone())
            actual[family]=torch.cat(rows).detach().clone()
    report["model_after"]=model_snapshot(backend)
    need(report["model_before"]==report["model_after"],"IRRA parameters/buffers changed")
    report["feature_comparisons"]={}
    for family in actual:
        x=actual[family]
        record_store(output).publish("features-"+family,{"features.pt":tensor_bytes(x)},
            {"kind":"small_smoke_native_representation","family":family,"model_space":asdict(adapter.space)},
            sources=tuple(feature_refs))
        report["feature_comparisons"][family]=compare(x,historical[family])
        report["feature_comparisons"][family]["norms"]=x.norm(dim=1).tolist()
        need(torch.allclose(x.norm(dim=1),torch.ones(2),atol=1e-6,rtol=0),"unit feature norms")
    report["scores"]={}
    for mode,q,g in (("clean","clean_text","clean_gallery"),("image_only","clean_text","image_gallery"),
                     ("text_only","adv_text","clean_gallery"),("joint","adv_text","image_gallery")):
        # 独立路径：新物化像素/Unicode原生编码，与只读历史对应行分别形成2x2分数。
        a=actual[q]@actual[g].T;b=historical[q]@historical[g].T
        report["scores"][mode]={**compare(a,b),"materialized":a.tolist(),"historical":b.tolist()}
    need(ledger.completed["IRRA_image"]==ledger.completed["IRRA_text"]==4,"exact 4/4 IRRA samples")
    store.verify_plan(force=True)

def child(stage,path,output,token):
    need(str(Path(output).resolve())==OUTPUT and stage in ("g","text","irra"),"fixed private child scope")
    root=record_store(output);root.verify("attempt")
    control=read_json(Path(output)/"attempt/control.json")
    need(control["token_sha256"]==digest(token) and control["config"]==asdict(FileRef(str(Path(path).resolve()),sha_file(path))),
         "child must belong to authorized immutable Attempt")
    gate=read_json(Path(output)/"attempt/preflight.json")
    need(gate["status"]=="MATERIALIZED_SMOKE_CPU_PREFLIGHT_PASS" and gate["GH"]["status"]=="PASS"
         and gate["GH"]["runtime_source_count"]==75 and gate["IRRA"]["namespace_restored"],"strict CPU launch gate")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")==("" if stage=="text" else "0"),"fixed child device scope")
    previous={"text":"g","irra":"text"}.get(stage)
    if previous:
        root.verify("stage-"+previous)
        need(read_json(Path(output)/("stage-"+previous)/"report.json")["status"]=="PASS","prior session must complete")
    need(not (Path(output)/("stage-"+stage)).exists(),"child cannot retry")
    # 每阶段单次claim先于模型；即使失败也不可重新加载/生成。
    root.publish("claim-"+stage,{"claim.json":json_bytes({"stage":stage,"may_have_executed":True})},
        {"kind":"single_stage_claim"},sources=(FileRef(str(Path(output)/"attempt/manifest.json"),sha_file(Path(output)/"attempt/manifest.json")),))
    ledger=Ledger();report={"stage":stage,"status":"RUNNING","G_loads":0,"H_loads":0,"IRRA_loads":0,
        "backward":0,"optimizer":0,"JVP":0,"IRRA_model_loaded_in_G_session":False}
    started=time.monotonic();guard=ResourceGuard(RSS,max(.01,SECONDS-(time.time()-control["started_unix"])))
    try:
        with guard:
            c=config(path)
            import torch
            torch.set_num_threads(2);torch.manual_seed(42)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if stage!="text":
                from ..smoke_irra import gpu_available
                report["gpu_before"]=gpu_available(c)
                if stage=="g":torch.cuda.reset_peak_memory_stats()
            {"g":stage_g,"text":stage_text,"irra":stage_irra}[stage](c,output,report,ledger)
            guard.check();report["status"]="PASS"
    except BaseException as exc:
        report.update(status="FAIL",exception_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        report["forwards"]=ledger.report();report["resources"]=guard.snapshot()
        report["elapsed_seconds"]=time.monotonic()-started
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            t=sys.modules["torch"]
            report["gpu_peak_allocated_mib"]=t.cuda.max_memory_allocated()/2**20
            report["gpu_peak_reserved_mib"]=t.cuda.max_memory_reserved()/2**20
        root.publish("stage-"+stage,{"report.json":json_bytes(report)},{"kind":"small_smoke_stage_result"},
            sources=(FileRef(str(Path(output)/("claim-"+stage)/"manifest.json"),sha_file(Path(output)/("claim-"+stage)/"manifest.json")),))
    print("MATERIALIZED_STAGE_"+stage.upper()+"_"+report["status"],flush=True)
    if report["status"]!="PASS":print(report.get("error"),flush=True)
    return int(report["status"]!="PASS")

def launch(path,*,allow_gpu=False,output=OUTPUT):
    need(str(Path(output).resolve())==OUTPUT,"fixed new Attempt output")
    if allow_gpu:need(not Path(output).exists(),"single Attempt already used; no retry")
    pre=preflight(path)
    print("GH_RUNTIME_PREFLIGHT_PASS",flush=True);print("MATERIALIZED_SMOKE_CPU_PREFLIGHT_PASS",flush=True)
    if not allow_gpu:
        print(json.dumps(pre,ensure_ascii=False,indent=2));return 0
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit CUDA_VISIBLE_DEVICES=0")
    from ..smoke_irra import gpu_available
    resource=gpu_available(config(path))
    protected=historical_snapshot("/home/lzf/ldx/tmp/tbps-framework-stage2/protected_before.json")
    root=record_store(output);token=secrets.token_hex(32);started=time.time()
    producers=tuple(FileRef(str(p),sha_file(p)) for p in
        (Path(__file__),Path(__file__).with_name("smoke_support.py"),Path(path).resolve()))
    root.publish("attempt",{"preflight.json":json_bytes(pre),"resources_before.json":json_bytes(resource),
        "control.json":json_bytes({"config":asdict(producers[-1]),"token_sha256":digest(token),"started_unix":started})},
        {"kind":"single_authorized_materialized_smoke","limit":LIMITS},sources=producers)
    report={"status":"RUNNING","scope":"2 Test Gallery / 2 Test Query; no ranking; no full collection",
        "H_loads":0,"H_forwards":0,"backward":0,"optimizer":0,"JVP":0,"stages":{}}
    guard=ResourceGuard(RSS,SECONDS)
    process=None
    try:
        with guard:
            for stage in ("g","text","irra"):
                # G子进程完全退出后才能进入文本/IRRA；会话之间仅传不可变文件。
                command=[sys.executable,"-B","-m","tbps.materialization.smoke","--config",path,
                    "--private-stage",stage,"--token",token]
                env=dict(os.environ);env["CUDA_VISIBLE_DEVICES"]="" if stage=="text" else "0"
                with (Path(output)/(stage+".log")).open("xb") as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                    try:
                        code=process.wait(timeout=max(.01,SECONDS-(time.time()-started)))
                    except BaseException:
                        if process.poll()is None:process.terminate();process.wait(timeout=10)
                        raise
                process=None
                p=Path(output)/("stage-"+stage)/"report.json"
                if p.exists():report["stages"][stage]=read_json(p)
                need(code==0 and report["stages"].get(stage,{}).get("status")=="PASS","stage failed: "+stage)
                print("MATERIALIZED_STAGE_"+stage.upper()+"_PASS",flush=True)
                guard.check()
            report["historical_files_sha_mtime_unchanged"]=unchanged(protected)
            before=pre["historical_feature_snapshot"]
            need(before=={p:checked(FileRef(p,v["sha256"])) for p,v in before.items()},"historical Full files changed")
            report["historical_full_files_unchanged"]=len(before)
            report["status"]="MATERIALIZED_END_TO_END_SMOKE_VERIFIED"
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
    finally:
        report["resources"]=guard.snapshot();report["elapsed_seconds"]=time.time()-started
        # 失败仍核验保护，不覆盖失败现场。
        try:report["historical_files_sha_mtime_unchanged"]=unchanged(protected)
        except BaseException as exc:report["historical_protection_error"]=str(exc);report["status"]="FAIL"
        refs=tuple(FileRef(str(p),sha_file(p)) for p in Path(output).glob("stage-*/manifest.json"))
        root.publish("result",{"report.json":json_bytes(report)},{"kind":"small_materialized_smoke","status":report["status"]},
            sources=(FileRef(str(Path(output)/"attempt/manifest.json"),sha_file(Path(output)/"attempt/manifest.json")),*refs))
    print(json.dumps(report,ensure_ascii=False,indent=2));print(report["status"],flush=True)
    return int(report["status"]!="MATERIALIZED_END_TO_END_SMOKE_VERIFIED")

def main(argv=None):
    p=argparse.ArgumentParser(description="Fixed Stage4C2 CPU gate; explicit once-only GPU authorization")
    p.add_argument("--config",required=True);p.add_argument("--allow-gpu-smoke",action="store_true")
    p.add_argument("--private-stage",choices=("g","text","irra"),help=argparse.SUPPRESS)
    p.add_argument("--token",help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.private_stage:
        need(not a.allow_gpu_smoke and a.token,"private child requires parent authorization")
        return child(a.private_stage,a.config,OUTPUT,a.token)
    need(a.token is None,"no public token")
    return launch(a.config,allow_gpu=a.allow_gpu_smoke)
if __name__=="__main__":raise SystemExit(main())
