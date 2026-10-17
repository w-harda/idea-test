"""Stage4C3：默认严格CPU预检；显式授权后唯一IRRA-only Attempt，不生成G/H。"""
from dataclasses import asdict
from pathlib import Path
from contextlib import ExitStack
import argparse
import json
import os
import time
import traceback
from unittest.mock import patch
from ..artifacts import ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes,digest
from ..engines.replay_control import ResourceGuard,rss_bytes,historical_snapshot,unchanged
from ..runtime_closure import cpu_only
from ..models.irra_io import PREPROCESS
from .contracts import need
from .retained_inputs import RetainedInputs,ORIGINAL
from .irra_startup import prepare_inputs
from .smoke_support import (preflight as source_preflight,config as source_config,selected_features,
                            GALLERY,QUERY,Ledger,THRESHOLD)
from .smoke import stage_irra
from .audit_retained import snapshot

BASE="/home/lzf/ldx/outputs/tbps-framework-stage4c3"
OUTPUT=BASE+"/irra_only_v1"
LIMITS={"G":0,"H":0,"IRRA_image":4,"IRRA_text":4}
SECONDS=300
MIN_RAM=8*1024**3
RESERVE_RAM=2*1024**3

def ram():
    result={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k,v,*_=line.split()
        if k.rstrip(":") in ("MemTotal","MemAvailable","SwapTotal","SwapFree"):
            result[k.rstrip(":")]=int(v)*1024
    need("MemTotal" in result and "MemAvailable" in result,"Linux RAM availability required")
    return result

class SystemGuard(ResourceGuard):
    """仅按实际系统可用RAM/耗时停止，记录RSS；没有固定每进程RSS上限。"""
    def __init__(self,*,memory=ram,sample=rss_bytes,clock=time.monotonic):
        self.memory=memory;self.minimum_available=None
        super().__init__(0,SECONDS,sample=sample,clock=clock)
    def check(self):
        self.peak=max(self.peak,self.sample())
        available=self.memory()["MemAvailable"]
        self.minimum_available=available if self.minimum_available is None else min(available,self.minimum_available)
        if available<RESERVE_RAM:self.reason="system available RAM below safety reserve"
        if self.clock()-self.started>=self.seconds:self.reason="time budget reached"
        if self.reason:
            from ..engines.replay_control import ResourceLimit
            raise ResourceLimit(self.reason)
    def snapshot(self):
        return {"peak_rss_bytes":self.peak,"elapsed_seconds":self.clock()-self.started,
            "rss_hard_limit_bytes":None,"minimum_system_available_bytes":self.minimum_available,
            "system_ram_reserve_bytes":RESERVE_RAM,"max_seconds":self.seconds,"stop_reason":self.reason}

def read_control(path):
    c=read_json(path,"tbps-irra-only-smoke-1")
    need(set(c)=={"schema","source_config","cpu_repair_receipt","original_root","gallery","query",
         "limits","threshold","precision","seed","rss_hard_limit_bytes","minimum_available_ram_bytes",
         "system_ram_reserve_bytes","max_seconds","physical_gpu","minimum_free_mib","maximum_utilization_percent"},
         "IRRA-only configuration fields")
    need(c["original_root"]==ORIGINAL and c["gallery"]==list(GALLERY) and c["query"]==list(QUERY),"fixed retained Test inputs")
    need(c["limits"]==LIMITS and c["threshold"]==THRESHOLD and c["precision"]=="fp32/math/no_tf32"
         and c["seed"]==42,"fixed numerical/count contract")
    need(c["rss_hard_limit_bytes"]is None and c["minimum_available_ram_bytes"]==MIN_RAM
         and c["system_ram_reserve_bytes"]==RESERVE_RAM and c["max_seconds"]==SECONDS,"new system RAM policy")
    need(c["physical_gpu"]==0 and c["minimum_free_mib"]==8192 and c["maximum_utilization_percent"]==10,"GPU resource policy")
    for key in ("source_config","cpu_repair_receipt"):checked(FileRef(**c[key]))
    parent=Path(c["cpu_repair_receipt"]["path"]).parent
    ArtifactStore(parent.parent,new_base=str(parent.parent.parent)).verify(parent.name)
    need(read_json(parent/"report.json")["status"]=="CPU_REPAIR_AND_RETAINED_INPUTS_VERIFIED","audited CPU repair required")
    return c

def tokenizer_check(inputs):
    from ..models.irra_io import official_tokenizer
    tokenizer=official_tokenizer(inputs.source);reports=[]
    for text in inputs.texts:
        action=text["action"]
        for side,key in (("original","original_caption"),("adversarial","adversarial_caption")):
            actual=tokenizer.inspect(text[key])
            expected=action["tokenizer_audit"]["before" if side=="original" else "after"]["model_tokens"]
            need(list(actual.ids)==expected,"frozen historical Unicode/Tokenizer ids mismatch")
            reports.append({"query":action["original"],"side":side,"token_ids":list(actual.ids),
                "untruncated_length":actual.untruncated_length,"truncated":actual.truncated,"exact":True})
    return {"fingerprint":tokenizer.fingerprint,"records":reports,"historical_tokens_exact":True}

def preflight(path):
    c=read_control(path)
    with cpu_only():old=source_preflight(c["source_config"]["path"])
    retained=RetainedInputs(c["original_root"])
    source=source_config(c["source_config"]["path"]);events=[]
    inputs,gate=prepare_inputs(source,ORIGINAL,store_factory=lambda c,o:(retained,retained.legacy),
                              feature_reader=selected_features,emit=events.append)
    need(gate.completed and gate.exited,"CPU contexts restored")
    with cpu_only():tokens=tokenizer_check(inputs)
    import torch
    need(not torch.cuda.is_initialized(),"CPU preflight must not initialize CUDA")
    files=snapshot(ORIGINAL)
    report={"status":"IRRA_ONLY_CPU_PREFLIGHT_PASS","GH_runtime_source_count":old["GH"]["runtime_source_count"],
        "GH_executed_files":len(old["GH"]["executed_sources"]),"IRRA_executed_files":len(old["IRRA"]["executed_sources"]),
        "retained_inputs":retained.descriptor(),"CPU_events":events,"CPU_contexts_restored":True,
        "feature_rows":{k:list(v.shape) for k,v in inputs.historical.items()},"tokenizer":tokens,
        "preprocess":PREPROCESS,"fixed_limits":LIMITS,"max_abs_error":THRESHOLD,
        "new_resource_policy":{"rss_hard_limit_bytes":None,"minimum_available_ram_bytes":MIN_RAM,
             "system_ram_reserve_bytes":RESERVE_RAM,"max_seconds":SECONDS},
        "historical_feature_snapshot":old["historical_feature_snapshot"],"retained_file_snapshot":files,
        "RAM":ram(),"model_loads":0,"neural_forwards":0,"CUDA_initialized":False}
    refs={r.path:r for r in retained.plan.sources}
    for ref in (FileRef(str(Path(path).resolve()),sha_file(path)),FileRef(**c["source_config"]),
                FileRef(**c["cpu_repair_receipt"]),*inputs.feature_refs,
                inputs.source.checkpoint,inputs.source.config,inputs.source.clip_bootstrap,inputs.source.bpe,
                *(r for n,r in inputs.source.code),*(FileRef(p,v["sha256"]) for p,v in files.items())):
        refs[ref.path]=ref
    for relative in ("materialization/irra_only.py","materialization/smoke.py","materialization/smoke_support.py",
                     "materialization/irra_startup.py","materialization/retained_inputs.py","runtime_closure.py",
                     "artifacts.py","contracts.py","models/irra.py","models/irra_io.py","models/irra_source.py","models/base.py"):
        p=Path(__file__).parents[1]/relative;ref=FileRef(str(p),sha_file(p));refs[ref.path]=ref
    return c,source,retained,report,tuple(refs.values())

def run(path,*,allow_gpu=False):
    if allow_gpu:need(not Path(OUTPUT).exists(),"single IRRA-only Attempt already used; no retry")
    c,source,retained,pre,refs=preflight(path)
    print("GH_RUNTIME_PREFLIGHT_PASS",flush=True);print("IRRA_ONLY_CPU_PREFLIGHT_PASS",flush=True)
    if not allow_gpu:
        print(json.dumps(pre,ensure_ascii=False,indent=2));return 0
    import torch
    from ..smoke_irra import gpu_available
    need(not torch.cuda.is_initialized() and os.environ.get("CUDA_VISIBLE_DEVICES")=="0","isolated CUDA_VISIBLE_DEVICES=0")
    memory=ram();need(memory["MemAvailable"]>=MIN_RAM,"insufficient available system RAM")
    gpu=gpu_available(c)
    before={r.path:checked(r) for r in refs}
    protected=historical_snapshot("/home/lzf/ldx/tmp/tbps-framework-stage2/protected_before.json")
    store=ArtifactStore(OUTPUT,new_base=BASE)
    store.publish("attempt",{"protocol.json":json_bytes(c),"preflight.json":json_bytes(pre),
                  "resources_before.json":json_bytes({"RAM":memory,"GPU":gpu})},
                  {"kind":"one_authorized_IRRA_only_materialized_smoke","old_attempt_status":"FAIL","new_G_H_forwards":0},sources=refs)
    ledger=Ledger(LIMITS);guard=SystemGuard()
    report={"status":"RUNNING","scope":"fixed 2 Test Gallery / 2 Test Query only","G_loads":0,"H_loads":0,
        "IRRA_loads":0,"IRRA_load_attempts":0,"backward":0,"optimizer":0,"JVP":0,
        "retained_inputs":retained.descriptor(),"resources_before":{"RAM":memory,"GPU":gpu},
        "pixel_preprocess":PREPROCESS,"tokenizer":pre["tokenizer"],"precision":"fp32/math/no_tf32",
        "old_attempt_status":"FAIL","new_G_H_forwards":0}
    loads={"attempted":0,"restored":0};forbidden={}
    start=time.monotonic()
    try:
        with guard:
            torch.set_num_threads(2);torch.manual_seed(42)
            from ..models import irra as adapter_module
            from ..attacks.base import tensor_sha
            original_load=adapter_module.load_official_model
            original_normalize=adapter_module.normalize_pixels
            expected_pixels=[];pixel_records=[]
            def loaded(*args,**kw):
                need(loads["attempted"]==0,"only one IRRA restoration allowed")
                loads["attempted"]+=1
                model=original_load(*args,**kw)
                loads["restored"]+=1
                return model
            def normalize(pixels):
                need(expected_pixels and torch.equal(pixels.detach().cpu(),expected_pixels.pop(0)),"IRRA must consume exact saved FP32 pixels")
                value=original_normalize(pixels)
                pixel_records.append({"input_sha256":tensor_sha(pixels),"normalized_sha256":tensor_sha(value),
                                      "shape":list(pixels.shape),"dtype":str(pixels.dtype)})
                return value
            def ready(inputs,gate):
                need(inputs.store.plan.id==retained.plan.id and inputs.store.descriptor()==retained.descriptor(),"old input reference changed")
                need(not torch.cuda.is_initialized() and gate.exited,"CPU isolation must precede CUDA")
                need(tokenizer_check(inputs)==pre["tokenizer"],"Tokenizer changed after preflight")
                expected_pixels.extend(x[key].detach().clone() for key in ("original","adversarial") for x in inputs.images)
                report["CPU_contexts_restored_before_CUDA"]=True
            def stop(name):
                def fail(*a,**kw):
                    forbidden[name]=forbidden.get(name,0)+1
                    raise RuntimeError("IRRA-only forbids "+name)
                return fail
            with ExitStack() as stack:
                stack.enter_context(patch.object(adapter_module,"load_official_model",new=loaded))
                stack.enter_context(patch.object(adapter_module,"normalize_pixels",new=normalize))
                for owner,name in ((torch.Tensor,"backward"),(torch.autograd,"backward"),(torch.autograd,"grad"),
                                   (torch.optim.Optimizer,"__init__"),(torch.autograd.functional,"jvp")):
                    stack.enter_context(patch.object(owner,name,new=stop(name)))
                stage_irra(source,OUTPUT,report,ledger,store_factory=lambda c,o:(retained,retained.legacy),
                           result_store=store,on_cpu_ready=ready)
            need(loads=={"attempted":1,"restored":1} and not expected_pixels and len(pixel_records)==4,"actual loading/pixel sample counts")
            need(ledger.started==ledger.completed==LIMITS and not forbidden,"exact IRRA-only forward budget")
            report["pixel_consumption"]=pixel_records;report["IRRA_source"]=asdict(__import__("tbps.models.irra_source",fromlist=["IRRASource"]).IRRASource.from_dict(
                 read_json(source["dataset_sources"]["path"])["irra"]))
            report["historical_files_unchanged"]=unchanged(protected)
            old=pre["historical_feature_snapshot"]
            need(old=={p:checked(FileRef(p,v["sha256"])) for p,v in old.items()},"historical Full changed")
            need(pre["retained_file_snapshot"]==snapshot(ORIGINAL),"old failed Attempt changed")
            need(before=={r.path:checked(r) for r in refs},"source SHA/mtime changed")
            report["historical_feature_files_unchanged"]=len(old)
            report["retained_files_unchanged"]=len(pre["retained_file_snapshot"])
            guard.check();report["status"]="IRRA_MATERIALIZED_SMOKE_VERIFIED"
    except BaseException as exc:
        report.update(status="FAIL",exception_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        report["IRRA_load_attempts"]=loads["attempted"];report["IRRA_loads"]=loads["restored"]
        report["forwards"]=ledger.report();report["forbidden_attempts"]=forbidden
        report["resources"]=guard.snapshot();report["elapsed_seconds"]=time.monotonic()-start
        report["RAM_after"]=ram()
        if torch.cuda.is_initialized():
            report["gpu_peak_allocated_mib"]=torch.cuda.max_memory_allocated()/2**20
            report["gpu_peak_reserved_mib"]=torch.cuda.max_memory_reserved()/2**20
        try:
            report["historical_files_unchanged"]=unchanged(protected)
            need(pre["retained_file_snapshot"]==snapshot(ORIGINAL),"old FAIL input files changed")
            old=pre["historical_feature_snapshot"]
            need(old=={p:checked(FileRef(p,v["sha256"])) for p,v in old.items()},"old Full cache changed")
            report["historical_feature_files_unchanged"]=len(old);report["retained_files_unchanged"]=len(pre["retained_file_snapshot"])
        except BaseException as exc:
            report["status"]="FAIL";report["protection_error"]=str(exc)
        # 失败也保留此前已完成的小样本Tensor，不重试或更改阈值。
        files={"report.json":json_bytes(report)}
        saved=tuple(FileRef(str(p),sha_file(p)) for p in Path(OUTPUT).glob("features-*/manifest.json"))
        report["feature_artifact_manifests"]=[asdict(r) for r in saved]
        files={"report.json":json_bytes(report)}
        store.publish("result",files,{"kind":"IRRA-only small materialized validation","status":report["status"]},
                      sources=(FileRef(str(Path(OUTPUT)/"attempt/manifest.json"),sha_file(Path(OUTPUT)/"attempt/manifest.json")),*saved))
    print(json.dumps(report,ensure_ascii=False,indent=2));print(report["status"],flush=True)
    return int(report["status"]!="IRRA_MATERIALIZED_SMOKE_VERIFIED")

def main(argv=None):
    p=argparse.ArgumentParser(description="Default strict CPU; one authorized IRRA-only 2x2 GPU smoke")
    p.add_argument("--config",required=True);p.add_argument("--allow-gpu-smoke",action="store_true")
    a=p.parse_args(argv);return run(a.config,allow_gpu=a.allow_gpu_smoke)
if __name__=="__main__":raise SystemExit(main())
