"""G60×H1–H10探索性Validation；显式授权才运行128次G/IRRA图像前向。"""
from pathlib import Path
from contextlib import ExitStack
import argparse, importlib.util, json, time, resource, traceback
import torch
from unittest.mock import patch
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,digest,checked
from tbps.materialization.store import image_checks
from tbps.materialization.gpu_safety import no_optimization
from tbps.models.irra import IRRAAdapter,fp32_runtime
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.ablations.gv2_sweep import generator_state,ARCH,resources
from tbps.ablations.gv2_epoch_state import tree_hash
from .g60_h_bridge import (BASE,KEYS,need,ref,configuration,prepare,verify_unchanged,
                          normalized,load_tensors,tensor_bytes,tensor_sha,metrics_csv)

def official_metrics(scores,pids,path):
    need(scores.dtype==torch.float32 and scores.shape==(len(pids),len(pids))
         and bool(torch.isfinite(scores).all()),"native score shape/finite")
    need(bool((pids[:,None]==pids[None,:]).any(1).all()),"all same-PID positives")
    spec=importlib.util.spec_from_file_location("g60_validation_official_metrics",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with torch.inference_mode():
        cmc,ap,_,_=module.rank(scores,pids,pids,max_rank=10,get_mAP=True)
    return {"R@1":float(cmc[0]),"R@5":float(cmc[4]),"R@10":float(cmc[9]),"mAP":float(ap)}

def frozen(model):
    need(all(not m.training for m in model.modules()) and all(not p.requires_grad and p.grad is None
         for p in model.parameters()),"model eval/freeze/grad")
    return tree_hash(model.state_dict())

def prefix_cpu_audit(ctx,store,counts):
    """只读前两张已提交产物的CPU读回审计；失败时不得继续第3张。"""
    need(counts=={"G_started":2,"G_completed":2,"IRRA_images":2},"prefix exact forward budget")
    for i in (0,1):
        store.verify(f"claim-{i:03d}");store.verify(f"image-{i:03d}")
        unit=store.root/f"image-{i:03d}"
        md=read_json(unit/"metadata.json");claim=read_json(store.root/f"claim-{i:03d}/claim.json")
        need(md["identity"]==claim["identity"]==ctx["identities"][i] and
             md["G60"]==claim["G60"]==ctx["c"]["G60"] and md["actual_forward_index"]==i+1,
             "prefix provenance/order")
        x=load_tensors(unit/"tensors.pt")
        need(image_checks(x["original"],x["adversarial"],x["effective_delta"],(384,128))==md["pixels"],
             "prefix CPU pixel SHA/budget")
        normalized(x["feature"],1)
        need(tensor_sha(x["feature"])==md["feature_sha256"],"prefix CPU feature SHA")
    return {"status":"PREFIX_CPU_AUDIT_PASS","images":2,"no_new_forwards":True}

def stream(ctx,store,generate,encode,read_pixels,counts,*,limit=128,smoke_check=lambda:None):
    """先Claim后前向；前两张即Smoke，保留并计入128总预算，不支持自动恢复/重试。"""
    need(limit==len(ctx["identities"]) and limit>=2,"fixed unique Gallery budget")
    features=[];maximum=0.
    for i,row in enumerate(ctx["identities"]):
        key=f"image-{i:03d}";binding={"index":i,"identity":row,"G60":ctx["c"]["G60"]}
        need(not (store.root/key).exists() and not (store.root/f"claim-{i:03d}").exists(),"existing image/Claim")
        store.publish(f"claim-{i:03d}",{"claim.json":json_bytes(binding)},
                      {"kind":"G60-before-forward"},sources=(ref(store.root/"protocol/manifest.json"),))
        pixels=read_pixels(ctx["image_refs"][i])
        with torch.inference_mode():
            result=generate(pixels);adv=result["adversarial"].detach().cpu().clone()
            saved={"original":pixels.cpu().clone(),"adversarial":adv,
                   "effective_delta":result["effective_delta"].detach().cpu().clone()}
            stats=image_checks(saved["original"],adv,saved["effective_delta"],(384,128))
            x=encode(adv).detach().cpu().clone();normalized(x,1)
        need(counts["G_started"]==counts["G_completed"]==counts["IRRA_images"]==i+1,"one G/IRRA sample per image")
        md={**binding,"pixels":stats,"feature_sha256":tensor_sha(x),"actual_forward_index":i+1,
            "raw_RGB_input":"original PIL RGB Resize384x128 ToTensor; Adv never re-quantized"}
        store.publish(key,{"tensors.pt":tensor_bytes({**saved,"feature":x}),"metadata.json":json_bytes(md)},
                      {"kind":"G60-Validation-RGB-and-IRRA"},
                      sources=(ref(store.root/f"claim-{i:03d}/manifest.json"),ctx["image_refs"][i]))
        reread=load_tensors(store.root/key/"tensors.pt")
        need(all(torch.equal(v,reread[k]) for k,v in {**saved,"feature":x}.items()),"materialized bitwise readback")
        features.append(x);maximum=max(maximum,stats["linf"])
        if i==1:
            smoke_check()
            prefix_report=prefix_cpu_audit(ctx,store,counts)
            store.publish("prefix-smoke",{"report.json":json_bytes({"status":"PREFIX_SMOKE_PASS",
                "cpu_readback_audit":prefix_report,"indices":[0,1],"retained_as_part_of_128":True,"counts":dict(counts),"max_linf":maximum})},
                {"kind":"two-image-prefix-smoke"},sources=tuple(ref(store.root/f"image-{j:03d}/manifest.json") for j in (0,1)))
            print("G60_PREFIX_SMOKE_PASS 2 / 128 (retained)",flush=True)
        if (i+1)%16==0:print("G60_VALIDATION_PROGRESS",i+1,"/ 128",flush=True)
    return torch.cat(features).clone(),maximum

def gpu_run(ctx):
    c=ctx["c"];root=Path(c["output"])
    need(not root.exists(),"Attempt exists; no retry/overwrite")
    # 资源读取不初始化CUDA；只观察其他任务，不终止/等待。
    rc={**c,"minimum_disk_bytes":{"smoke":c["minimum_disk_bytes"]}}
    snapshot=resources(rc,"smoke")
    store=ArtifactStore(root,new_base=BASE)
    store.publish("protocol",{"protocol.json":json_bytes({k:v for k,v in c.items() if not k.startswith("_")}),
        "preflight.json":json_bytes(ctx["report"])},{"kind":"exploratory-internal-validation"},
        sources=(ref(c["_config_path"]),))
    report={"status":"RUNNING","resources":snapshot,"G_loads":0,"IRRA_loads":0,"H_loads":0,"H_forwards":0,
            "IRRA_text_forwards":0,"G_started":0,"G_completed":0,"IRRA_images":0,
            "backward":0,"optimizer":0,"JVP":0}
    counts={k:0 for k in ("G_started","G_completed","IRRA_images")};begin=time.monotonic()
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        ctx["gate"].initialize_cuda()  # 必须在安全wrapper安装之前，CPU快照/命名空间已恢复。
        with no_optimization() as attempts,fp32_runtime(),ExitStack() as stack:
            with legacy_imports(ctx["legacy"],allow_import=True) as old:
                cls=old("image_only_generator").ImageOnlyGenerator
                fn=old("inference_once").generate_once
                pixel_reader=old("irra_image_adapter").pixels_from_pil
                payload=torch.load(c["G60"]["path"],map_location="cpu",weights_only=False)
                state=generator_state(payload,60,read_json(c["sources"]["original_contract"]["path"]),c["G10"])
                g=cls(**ARCH);g.load_state_dict(state,strict=True)
                g=g.to("cuda:0").eval().requires_grad_(False);del state,payload
            report["G_loads"]=1;gbefore=frozen(g)
            adapter=IRRAAdapter.from_source(ctx["source"],device="cuda:0",allow_model_load=True)
            report["IRRA_loads"]=1;model=adapter._backend;ibefore=frozen(model)
            def pre(_,args):
                need(not torch.is_grad_enabled() and len(args)==1 and len(args[0])==1,"G inference single RGB only")
                counts["G_started"]+=1;need(counts["G_started"]<=128,"G forward limit")
            def post(*_):counts["G_completed"]+=1
            stack.callback(g.register_forward_pre_hook(pre).remove)
            stack.callback(g.register_forward_hook(post).remove)
            def observe_image(_,args):
                need(not torch.is_grad_enabled() and len(args[0])==1,"IRRA image inference batch1")
                counts["IRRA_images"]+=1;need(counts["IRRA_images"]<=128,"IRRA image limit")
            stack.callback(model.base_model.visual.register_forward_pre_hook(observe_image).remove)
            def forbid_text(*a,**k):raise ValueError("IRRA text forward forbidden: reuse historical features")
            stack.enter_context(patch.object(model,"encode_text",forbid_text))
            def pixels(item):
                from PIL import Image
                checked(item)
                with Image.open(item.path) as image:return pixel_reader(image).unsqueeze(0)
            gallery,maximum=stream(ctx,store,lambda x:fn(g,x.to("cuda:0")),
                lambda x:adapter.inference().encode_images(x.to("cuda:0")).payload,pixels,counts,
                smoke_check=lambda:need(frozen(g)==gbefore and frozen(model)==ibefore,"prefix smoke model frozen"))
            need(frozen(g)==gbefore and frozen(model)==ibefore,"G/IRRA parameters or buffers changed")
            need(counts=={"G_started":128,"G_completed":128,"IRRA_images":128},"exact GPU budget")
            report.update(**counts,G_before=gbefore,G_after=frozen(g),IRRA_before=ibefore,IRRA_after=frozen(model),
                          max_linf=maximum,forbidden_attempts=attempts,models_frozen=True,
                          model_space=__import__("dataclasses").asdict(adapter.space),
                          GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        del g,model,adapter
        torch.cuda.empty_cache()
        pids=torch.tensor([x["pid"] for x in ctx["identities"]])
        groups={};rows=[];official=ctx["source"].ref("utils/metrics.py").path
        for name,text in [("G60 Image-only",ctx["raw"]),*[(f"G60 + H{e} Joint",ctx["texts"][str(e)]) for e in range(1,11)]]:
            # 评分表达式与历史formal_training.validation.metrics完全相同：CPU FP32 normalized dot。
            scores=text@gallery.T;metrics=official_metrics(scores,pids,official);groups[name]=metrics
            rows.append({"Group":name,**metrics})
            store.publish("score-"+str(len(rows)).zfill(2),
                {"scores.pt":tensor_bytes({"scores":scores}),"metrics.json":json_bytes(metrics)},
                {"kind":"IRRA-native-validation-score","group":name,"gallery_sha256":tensor_sha(gallery),
                 "text_feature_sha256":tensor_sha(text),"ordered_identities_sha256":digest(ctx["identities"])},
                sources=(ref(store.root/"protocol/manifest.json"),ctx["source"].ref("utils/metrics.py")))
        store.publish("features",{"gallery.pt":tensor_bytes({"adversarial":gallery})},
                      {"kind":"shared-G60-gallery","content_sha256":tensor_sha(gallery)},
                      sources=tuple(ref(root/f"image-{i:03d}/manifest.json") for i in range(128)))
        verify_unchanged(ctx["report"])
        report.update(status="G60_H_GPU_AND_SCORING_PASS",**counts,elapsed_seconds=time.monotonic()-begin,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,Gallery=128,Query=128,
            same_gallery_shared_by_all_10_H=True,all_queries_retained=True,
            history_results_not_recomputed=True,diagnostic_not_new_official_selection=True)
        store.publish("completed",{"summary.json":json_bytes({"groups":groups,"execution":report}),
                "metrics_raw.csv":metrics_csv(rows),"metrics_2dp.csv":metrics_csv(rows,True)},
                {"kind":"complete-exploratory-validation"},
                sources=(ref(root/"features/manifest.json"),*(ref(root/f"score-{i:02d}/manifest.json") for i in range(1,12))))
        print("G60_H_GPU_AND_SCORING_PASS",json.dumps(report),flush=True)
    except BaseException as exc:
        report.update(status="FAIL",**counts,error=str(exc),traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-begin)
        store.publish("failure",{"report.json":json_bytes(report)},{"kind":"failed-attempt-preserved"})
        raise

def audit(ctx):
    need(not torch.cuda.is_initialized(),"new audit CPU only")
    root=Path(ctx["c"]["output"]);store=ArtifactStore(root,new_base=BASE)
    need(root.exists() and not store.pending() and not (root/"failure").exists(),"pending/FAIL/missing Attempt")
    expected={"protocol","prefix-smoke","features","completed"}|{f"claim-{i:03d}" for i in range(128)}|{f"image-{i:03d}" for i in range(128)}|{f"score-{i:02d}" for i in range(1,12)}
    need({p.name for p in root.iterdir() if p.is_dir()}==expected,"missing/orphan/unexpected artifacts")
    for name in expected:store.verify(name)
    saved=read_json(root/"protocol/preflight.json")
    need(saved["identities"]==ctx["identities"] and saved["text_bindings"]==ctx["report"]["text_bindings"],
         "new audit original sources/identities")
    verify_unchanged(saved)
    gallery=[];maximum=0.;image_index=[]
    for i,row in enumerate(ctx["identities"]):
        unit=root/f"image-{i:03d}";md=read_json(unit/"metadata.json");claim=read_json(root/f"claim-{i:03d}/claim.json")
        need(md["identity"]==claim["identity"]==row and md["index"]==claim["index"]==i
             and md["G60"]==claim["G60"]==ctx["c"]["G60"] and md["actual_forward_index"]==i+1,"128 ordered claims/proofs")
        data=load_tensors(unit/"tensors.pt");stats=image_checks(data["original"],data["adversarial"],data["effective_delta"],(384,128))
        need(stats==md["pixels"] and tensor_sha(data["feature"])==md["feature_sha256"],"pixel/feature content")
        normalized(data["feature"],1);gallery.append(data["feature"]);maximum=max(maximum,stats["linf"])
        image_index.append({"index":i,"row_id":row["row_id"],"manifest":str(unit/"manifest.json"),"manifest_sha256":ref(unit/"manifest.json").sha256,**stats})
    gallery=torch.cat(gallery)
    need(torch.equal(gallery,load_tensors(root/"features/gallery.pt")["adversarial"]),"shared gallery exact")
    summary=read_json(root/"completed/summary.json");report=summary["execution"]
    need(report["status"]=="G60_H_GPU_AND_SCORING_PASS" and report["models_frozen"]
         and report["G_before"]==report["G_after"] and report["IRRA_before"]==report["IRRA_after"]
         and all(report[k]==128 for k in ("G_started","G_completed","IRRA_images"))
         and report["G_loads"]==report["IRRA_loads"]==1
         and all(report[k]==0 for k in ("H_loads","H_forwards","IRRA_text_forwards","backward","optimizer","JVP"))
         and report["max_linf"]==maximum and report["forbidden_attempts"]=={},"actual counts/freeze/budget")
    pids=torch.tensor([x["pid"] for x in ctx["identities"]]);rows=[]
    for idx,(name,group) in enumerate(summary["groups"].items(),1):
        # JSON键排序不代表模式顺序：显式找到对应的不可变score身份。
        number=1 if name=="G60 Image-only" else int(name.split("H")[1].split()[0])+1
        unit=root/f"score-{number:02d}";manifest=read_json(unit/"manifest.json")
        text=ctx["raw"] if number==1 else ctx["texts"][str(number-1)]
        scores=load_tensors(unit/"scores.pt")["scores"]
        need(torch.equal(scores,text@gallery.T),"native score exact from bound sources")
        m=official_metrics(scores,pids,ctx["source"].ref("utils/metrics.py").path)
        need(m==group==read_json(unit/"metrics.json") and manifest["provenance"]["gallery_sha256"]==tensor_sha(gallery)
             and manifest["provenance"]["text_feature_sha256"]==tensor_sha(text),"official metrics/source exact")
    rows=[{"Group":name,**summary["groups"][name]} for name in ["G60 Image-only",*[f"G60 + H{i} Joint" for i in range(1,11)]]]
    need((root/"completed/metrics_raw.csv").read_bytes()==metrics_csv(rows)
         and (root/"completed/metrics_2dp.csv").read_bytes()==metrics_csv(rows,True),"CSV summary exact")
    result={"status":"G60_H_VALIDATION_CPU_AUDIT_PASS","images":128,"queries":128,"H_candidates":10,
            "verified_artifacts":len(expected),"max_linf":maximum,"counts":{k:report[k] for k in ("G_started","G_completed","IRRA_images","H_forwards","IRRA_text_forwards")},
            "historical_sources_SHA_stat_unchanged":True,"old_audit_not_rerun":True,
            "G60":ctx["c"]["G60"],"shared_gallery_content_sha256":tensor_sha(gallery),
            "official_score_and_metrics_exact":True,"selection_not_modified":True}
    out=ArtifactStore(BASE/"audit_v1",new_base=BASE)
    need(not out.root.exists(),"independent audit already published; no rerun")
    out.publish("result",{"report.json":json_bytes(result),"image_index.json":json_bytes({"images":image_index})},
                {"kind":"independent-new-Validation-CPU-audit"},sources=(ref(root/"completed/manifest.json"),))
    print("G60_H_VALIDATION_CPU_AUDIT_PASS",json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--action",choices=("preflight","run","audit"),default="preflight")
    p.add_argument("--allow-validation-gpu",action="store_true");a=p.parse_args()
    torch.set_num_threads(2)
    c=configuration(a.config);ctx=prepare(c)
    if a.action=="preflight":
        print("G60_H_CPU_PREFLIGHT_PASS",json.dumps({k:v for k,v in ctx["report"].items() if k not in ("sources","identities","text_bindings","closure")}),flush=True)
    elif a.action=="run":
        need(a.allow_validation_gpu,"explicit GPU authorization");gpu_run(ctx)
    else:audit(ctx)
if __name__=="__main__":main()
