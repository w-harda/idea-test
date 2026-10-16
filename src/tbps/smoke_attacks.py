"""Stage 3B一次性真实G/H输出等价；默认CPU预检，不含IRRA、训练或排名。"""
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import os
import random
import signal
import sys
import time
import traceback
from .artifacts import ArtifactError, ArtifactStore, checked, digest, json_bytes, read_json, sha_file
from .contracts import FileRef, DatasetManifest
from .datasets.cuhk_pedes import CuhkPedes
from .datasets.historical_membership import load_historical_partitions
from .attacks.base import AttackBinding, ImageRequest, TextRequest, need
from .attacks.provenance import LegacyAttackSources
from .attacks.legacy_runtime import legacy_imports, load_legacy_plugins
from .smoke_irra import gpu_available, compare, model_snapshot

BASE="/home/lzf/ldx/outputs/tbps-framework-stage3b"
ROW_IDS=("cuhk:9955:0","cuhk:26530:0")
LIMITS={"G":4,"H":4,"IRRA":0}

def exact_action(native, plugin):
    """JSON规范化仅消除tuple/list表示差异，所有字段/浮点值仍要求exact。"""
    a,b=json.loads(json_bytes(native)),json.loads(json_bytes(plugin))
    need(a==b, "H complete action mismatch; native/plugin records retained in report")
    return {"exact":True,"normalized_sha256":digest(a),"fields":sorted(a),
            "action":a,"H_forward":a["H_forward"]}

def rng_summary():
    import numpy as np
    import torch
    n=np.random.get_state()
    return {"python":digest(repr(random.getstate())),
            "numpy":digest([n[0],n[1].tolist(),n[2],n[3],n[4]]),
            "torch_cpu":hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "cuda":[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                    for x in torch.cuda.get_rng_state_all()]}

class ForwardLedger:
    def __init__(self):
        self.started=dict.fromkeys(LIMITS,0)
        self.completed=dict.fromkeys(LIMITS,0)
        self.phase="unset";self.sample="unset";self.records=[]
        self.g_captures={}
    def attach(self, model, kind, stack):
        def before(module,args):
            import torch
            need(len(args)==1, "one public input only")
            if kind=="G":
                count=len(args[0]);need(count==1,"batch=1 only")
            else:
                need(set(vars(args[0]))=={"caption","attributes"},"Caption-only H TextExample")
                count=1
            need(self.started[kind]+count<=LIMITS[kind],"actual neural forward budget exceeded")
            self.started[kind]+=count
            self.records.append({"kind":kind,"phase":self.phase,"sample":self.sample,
                                 "samples":count,"returned":False,
                                 "inference_mode":torch.is_inference_mode_enabled(),
                                 "grad_enabled":torch.is_grad_enabled()})
        def after(module,args,value):
            self.completed[kind]+=1
            record=next(r for r in reversed(self.records) if r["kind"]==kind and not r["returned"])
            record["returned"]=True
            if kind=="G":
                # 最多4组紧凑CPU副本，不持有历史GPU Tensor/计算图。
                self.g_captures[(self.phase,self.sample)]={
                    "pixels":args[0].detach().cpu().clone(),"delta":value.detach().cpu().clone()}
        stack.callback(model.register_forward_pre_hook(before).remove)
        stack.callback(model.register_forward_hook(after).remove)
    def report(self):
        return {"started":dict(self.started),"completed":dict(self.completed),"records":list(self.records)}

def preflight(path):
    cfgref=FileRef(str(Path(path).resolve()),sha_file(path))
    c=read_json(path,"tbps-gh-equivalence-smoke-1")
    need(set(c)=={"schema","dataset_sources","attack_sources","row_ids","expected_eligible","limits",
                 "max_abs_error","seed","precision","physical_gpu","minimum_free_mib","maximum_utilization_percent"},
         "smoke config fields")
    need(tuple(c["row_ids"])==ROW_IDS and c["expected_eligible"]==[True,False] and c["limits"]==LIMITS,
         "fixed eligible/NOOP Train samples and 4/4/0 limits")
    need(c["max_abs_error"]==1e-6 and c["seed"]==42 and c["precision"]=="fp32/math/no_tf32",
         "fixed threshold/seed/precision")
    need(c["physical_gpu"]==0 and type(c["minimum_free_mib"]) is int and c["minimum_free_mib"]>=8192 and
         type(c["maximum_utilization_percent"]) is int and 0<=c["maximum_utilization_percent"]<=10,
         "GPU guard cannot be weakened")
    dsref,aref=FileRef(**c["dataset_sources"]),FileRef(**c["attack_sources"])
    checked(dsref);checked(aref)
    data=read_json(dsref.path,"tbps-stage2-cpu-sources-1")
    sources=LegacyAttackSources(aref.path)
    dataset=CuhkPedes(FileRef(**data["annotation"]),data["image_root"])
    partitions=load_historical_partitions(dataset,{k:FileRef(**v) for k,v in data["membership"].items()})
    qualification=read_json(data["membership"]["h_train_qualification"]["path"])
    entries=qualification["entries"]
    chosen=tuple(next(e for e in entries if e["eligible"] is wanted)["identity"]["row_id"]
                 for wanted in (True,False))
    need(chosen==ROW_IDS,"first eligible/first fixed NOOP selection changed")
    qs={q.query_id:q for q in partitions.train.queries};gs={g.image_id:g for g in partitions.train.gallery}
    selected_q=tuple(partitions.require_member(qs[i],"attack_train") for i in ROW_IDS)
    selected_g=tuple(partitions.require_member(gs[q.paired_image_id],"attack_train") for q in selected_q)
    manifest=DatasetManifest("cuhk_pedes","train",dataset.annotation,selected_g,selected_q)
    images=tuple(ImageRequest(dataset.public_image(manifest,g.image_id),
                   AttackBinding.from_manifest(manifest,g.image_id,"image")) for g in selected_g)
    texts=tuple(TextRequest(dataset.public_text(manifest,q.query_id),
                  AttackBinding.from_manifest(manifest,q.query_id,"text")) for q in selected_q)
    samples=[]
    # CPU预检禁止权重反序列化及任何神经模型forward；只检查旧加载函数/前端/Tokenizer路径。
    import torch
    from unittest.mock import patch
    need(not torch.cuda.is_initialized(),"CPU preflight must not initialize CUDA")
    with patch.object(torch,"load",side_effect=AssertionError("preflight weight load forbidden")), \
         patch.object(torch.nn.Module,"_call_impl",side_effect=AssertionError("preflight neural forward forbidden")):
        with legacy_imports(sources,allow_import=True) as old:
            native=old("inference_once");module=old("phase5_joint.backend")
            frontend=old("text_frontend").CaptionFrontend()
            probe=old("tokenizer_probe").official_probe()
            qualify=old("formal_training.data").qualify_caption
            for i,(q,g) in enumerate(zip(selected_q,selected_g)):
                saved=next(e for e in entries if e["identity"]["row_id"]==q.query_id)
                current=qualify(q.caption,frontend,probe)
                for key in ("eligible","reasons","attributes","BPE_length","utf8_bytes","structural_K0"):
                    need(json_bytes(current[key])==json_bytes(saved[key]),"qualification/source drift: "+key)
                need(current["eligible"] is c["expected_eligible"][i],"qualification role mismatch")
                samples.append({"row_id":q.query_id,"image_id":g.image_id,"image_rel_path":g.image_rel_path,
                    "image":asdict(images[i].public.content),"caption":q.caption,"caption_index":q.caption_index,
                    "caption_sha256":texts[i].binding.input_sha256,"pid":asdict(q.pid),
                    "official_split":q.split,"role":q.role,
                    "qualification":{k:current[k] for k in ("eligible","reasons","attributes","BPE_length","structural_K0")},
                    "native_G_function":native.generate_once.__module__+".generate_once",
                    "native_H_class":module.TextAttacker.__module__+".TextAttacker"})
    need(not torch.cuda.is_initialized(),"preflight initialized CUDA")
    refs=[cfgref,dsref,aref,sources.catalog_ref,dataset.annotation,*partitions.sources,*partitions.metadata_refs,
          *sources.runtime_refs,*(i.public.content for i in images)]
    for asset in sources.assets.values():
        refs.extend(FileRef(v["path"],v["sha256"]) for v in asset["files"].values())
    refs.extend(FileRef(p,h) for p,h in sources.framework_sources.items())
    refs.extend(FileRef(str(p),sha_file(p)) for p in (Path(__file__),Path(__file__).with_name("smoke_irra.py"),
                                                    Path(__file__).parent/"models/irra.py"))
    refs=tuple({r.path:r for r in refs}.values())
    before={r.path:checked(r) for r in refs}
    report={"schema":"tbps-gh-smoke-preflight-1","samples":samples,"limits":LIMITS,
            "config":asdict(cfgref),"plugin_source_sha256":sources.fingerprint,
            "membership_sha256":partitions.fingerprint,"manifest_sha256":digest(asdict(manifest)),
            "CPU_import_isolation_and_loading_paths_passed":True,"weights_restored":0,"neural_forwards":0}
    return c,sources,images,texts,refs,before,report

def run_once(c,sources,images,texts,refs,before,pre,output):
    store=ArtifactStore(output,new_base=BASE)
    need(not store.root.exists() or not any(store.root.iterdir()),"one-attempt output already used; no retry")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","set CUDA_VISIBLE_DEVICES=0")
    resource=gpu_available(c)
    store.publish("attempt",{"preflight.json":json_bytes(pre),"resources.json":json_bytes(resource)},
                  {"purpose":"single authorized G/H equivalence","config_sha256":pre["config"]["sha256"]},
                  sources=refs)
    report={"schema":"tbps-gh-smoke-result-1","status":"RUNNING","preflight":pre,
            "resources_before":resource,"weight_load_attempts":{"G":0,"H":0},
            "weight_load_completed":{"G":0,"H":0},"IRRA_model_loads":0,
            "backward":0,"optimizer_steps":0,"JVP":0,"G_comparisons":[],"H_comparisons":[]}
    ledger=ForwardLedger();start=time.perf_counter();models={}
    old_signal=signal.getsignal(signal.SIGTERM)
    def stop(*_):raise InterruptedError("SIGTERM: preserve attempt and stop; no automatic retry")
    signal.signal(signal.SIGTERM,stop)
    try:
        import torch
        import numpy as np
        from PIL import Image
        from unittest.mock import patch
        from .models.irra import fp32_runtime
        need(not torch.cuda.is_initialized(),"isolated process required")
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        report["resources_at_load"]=gpu_available(c)
        torch.cuda.reset_peak_memory_stats()
        original_load=torch.load
        allowed={str(Path(sources.ref(k,"checkpoint").path).resolve()):v for k,v in (("gv2","G"),("hv1","H"))}
        def counted_load(path,*args,**kwargs):
            key=allowed.get(str(Path(path).resolve()))
            need(key is not None,"unapproved weight load; IRRA/other checkpoints forbidden")
            need(report["weight_load_attempts"][key]==0,"weights may only be restored once")
            report["weight_load_attempts"][key]+=1
            value=original_load(path,*args,**kwargs)
            report["weight_load_completed"][key]+=1
            return value
        original_call=torch.nn.Module._call_impl
        def neural_guard(module,*args,**kwargs):
            need(not type(module).__module__.startswith("model.") and
                 "IRRA" not in type(module).__name__,"IRRA neural forward forbidden")
            return original_call(module,*args,**kwargs)
        print("GH_SMOKE_STAGE one_G_one_H_weight_restore",flush=True)
        with patch.object(torch,"load",new=counted_load),patch.object(torch.nn.Module,"_call_impl",new=neural_guard):
            with load_legacy_plugins(sources,device="cuda:0",allow_model_load=True) as plugins:
                g,h,joint=plugins["gv2"],plugins["hv1"],plugins["joint"]
                models={"G":g.backend.model,"H":h.backend.model}
                need(next(models["G"].parameters()).device.type=="cuda" and
                     next(models["H"].parameters()).device.type=="cpu","G GPU/H CPU placement")
                report["model_before"]={k:model_snapshot(m) for k,m in models.items()}
                report["rng_before_inference"]=rng_summary()
                old=importlib.import_module("inference_once")
                old_h=importlib.import_module("phase5_joint.backend")
                need(g.backend.generate_once is old.generate_once and
                     type(h.backend.attacker) is old_h.TextAttacker,"must reuse actual legacy implementation")
                native_g={};native_h={};plugin_g={};plugin_h={}
                with ExitStack() as stack, fp32_runtime(),torch.inference_mode():
                    for k,m in models.items():ledger.attach(m,k,stack)
                    print("GH_SMOKE_STAGE native_two_images_two_captions",flush=True)
                    ledger.phase="native"
                    for i,sample in enumerate(pre["samples"]):
                        ledger.sample=sample["row_id"]
                        with Image.open(sample["image"]["path"]) as im:
                            pixels=g.backend.pixel_reader(im)[None].to("cuda:0")
                        value=old.generate_once(models["G"],pixels)
                        native_g[i]={k:v.detach().cpu().clone() for k,v in value.items()}
                        native_h[i]=json.loads(json_bytes(h.backend.attacker.attack(texts[i].public.caption)))
                    report["rng_after_native"]=rng_summary()
                    print("GH_SMOKE_STAGE plugins_three_joint_modes",flush=True)
                    ledger.phase="plugin"
                    # Joint输入故意交叉身份：Gallery0 + Query1，无正确配对依赖。
                    ledger.sample=ROW_IDS[0]
                    first=joint.generate(mode="joint",image=images[0],text=texts[1])
                    plugin_g[0]=first.image;plugin_h[1]=first.text
                    ledger.sample=ROW_IDS[1]
                    second=joint.generate(mode="image_only",image=images[1]);plugin_g[1]=second.image
                    ledger.sample=ROW_IDS[0]
                    third=joint.generate(mode="text_only",text=texts[0]);plugin_h[0]=third.text
                    report["joint"]={"modes":["joint","image_only","text_only"],
                        "shared_instances":joint.image_plugin is g and joint.text_plugin is h,
                        "cross_identity":{"image":first.image.binding.sample_id,"text":first.text.binding.sample_id},
                        "single_modal_absent_outputs":second.text is None and third.image is None,
                        "extra_forward_samples":0}
                    need(report["joint"]["shared_instances"] and report["joint"]["single_modal_absent_outputs"],
                         "Joint composition mismatch")
                    for i,row in enumerate(ROW_IDS):
                        a=ledger.g_captures[("native",row)];b=ledger.g_captures[("plugin",row)]
                        metrics={k:compare(a[k],b[k],exact=k=="pixels") for k in ("pixels","delta")}
                        metrics["adversarial"]=compare(native_g[i]["adversarial"],plugin_g[i].payload)
                        metrics["effective_delta"]=compare(native_g[i]["effective_delta"],plugin_g[i].effective_delta)
                        metrics["native_raw_delta_hook"]=compare(native_g[i]["delta"],a["delta"],exact=True)
                        for name,value in (("native",native_g[i]["adversarial"]),("plugin",plugin_g[i].payload)):
                            cap=float((value.detach().cpu().double()-a["pixels"].double()).abs().max())
                            need(bool(torch.isfinite(value).all()) and bool(((value>=0)&(value<=1)).all()) and cap<=8/255,
                                 "raw RGB budget violation")
                            metrics[name+"_linf"]=cap
                        report["G_comparisons"].append({"row_id":row,**metrics})
                        b_record=plugin_h[i].trace()["action"]
                        # 先保留两条原始动作证据；任何字段差异均拒绝。
                        evidence={"row_id":row,"native":native_h[i],"plugin":b_record}
                        report["H_comparisons"].append(evidence)
                        evidence["comparison"]=exact_action(native_h[i],b_record)
                    report["rng_after_plugin"]=rng_summary()
                torch.cuda.synchronize()
                report["model_after"]={k:model_snapshot(m) for k,m in models.items()}
                need(report["model_before"]==report["model_after"],"model parameters/buffers changed")
                need(report["rng_before_inference"]==report["rng_after_native"]==report["rng_after_plugin"],
                     "inference RNG changed; stop and investigate")
                need(ledger.started==ledger.completed and ledger.completed=={"G":4,"H":2,"IRRA":0},
                     "unexpected actual forward counts")
                need(report["weight_load_completed"]=={"G":1,"H":1},"unexpected weight restore counts")
                report["source_files_unchanged"]=len(refs)
                need(before=={r.path:checked(r) for r in refs},"source SHA/mtime changed")
                report["status"]="PASS"
    except BaseException as exc:
        report.update(status="FAIL",exception_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        if models:
            try:report["model_after_failure"]={k:model_snapshot(m) for k,m in models.items()}
            except BaseException as state_error:report["state_check_error"]=str(state_error)
    finally:
        signal.signal(signal.SIGTERM,old_signal)
        report["elapsed_seconds"]=time.perf_counter()-start
        report["forwards"]=ledger.report()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            report["gpu_peak_allocated_mib"]=sys.modules["torch"].cuda.max_memory_allocated()/2**20
            report["gpu_peak_reserved_mib"]=sys.modules["torch"].cuda.max_memory_reserved()/2**20
        store.publish("result",{"report.json":json_bytes(report)},
                      {"attempt_manifest_sha256":sha_file(store.root/"attempt/manifest.json")})
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("GH_PLUGIN_EQUIVALENCE_"+report["status"],flush=True)
    return 0 if report["status"]=="PASS" else 1

def main(argv=None):
    p=argparse.ArgumentParser(description="Default CPU preflight; one explicit bounded G/H GPU attempt")
    p.add_argument("--config",required=True);p.add_argument("--output",default=BASE+"/equivalence_v1")
    p.add_argument("--allow-gpu-smoke",action="store_true")
    a=p.parse_args(argv)
    try:
        result=preflight(a.config)
        if not a.allow_gpu_smoke:
            print(json.dumps(result[-1],ensure_ascii=False,indent=2));print("GH_SMOKE_CPU_PREFLIGHT_PASS");return 0
        return run_once(*result,output=a.output)
    except (ValueError,TypeError,OSError,KeyError,StopIteration) as exc:
        p.exit(1,"GH_SMOKE_REJECTED "+str(exc)+"\n")

if __name__=="__main__":raise SystemExit(main())
