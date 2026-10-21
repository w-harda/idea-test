"""G60引导的H10→60独立接线；原H/SCST/奖励/Checkpoint函数保持只读复用。"""
from pathlib import Path
from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import argparse,json,os,time,signal,resource,traceback
import torch
from tbps.artifacts import ArtifactStore,FileRef,read_json,json_bytes,digest,checked
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.models.irra import IRRAAdapter,fp32_runtime
from tbps.materialization.store import image_checks
from tbps.materialization.gpu_safety import no_optimization
from .gv2_sweep import resources,generator_state,ARCH
from .gv2_epoch_state import tree_hash
from .g60_h_bridge import tensor_bytes,load_tensors,tensor_sha,metrics_csv
from .g60_h_validation import official_metrics,frozen
from .hv1_g60_state import *

def resource_check(c):
    return resources({**c,"minimum_disk_bytes":{"smoke":c["minimum_disk_bytes"]}},"smoke")

def gallery(ctx):
    c=ctx["c"];root=BASE/"gallery_train";need(not root.exists(),"Gallery exists; no automatic generation replay")
    snapshot=resource_check(c)
    store=ArtifactStore(root,new_base=BASE)
    store.publish("protocol",{"protocol.json":json_bytes({"G60":c["G60"],"IRRA":asdict(ctx["source"].checkpoint),
        "role":"train","identities":ctx["report"]["identities"],"precision":c["precision"],"epsilon":8/255}),
        "preflight.json":json_bytes(ctx["report"])},{"kind":"G60-Train-gallery"},
        sources=(ref(c["_path"]),))
    begin=time.monotonic();counts={"G_loads":0,"IRRA_loads":0,"G_started":0,"G_completed":0,"IRRA_images":0,
                                 "H_forwards":0,"IRRA_text_forwards":0,"backward":0,"optimizer":0,"JVP":0}
    report={"status":"RUNNING","counts":counts,"resources":snapshot}
    try:
        ctx["gate"].initialize_cuda();torch.manual_seed(42)
        with fp32_runtime(),no_optimization(),ExitStack() as stack:
            with legacy_imports(ctx["legacy"],allow_import=True) as old:
                cls=old("image_only_generator").ImageOnlyGenerator;fn=old("inference_once").generate_once
                pixel_reader=old("irra_image_adapter").pixels_from_pil
                value=torch.load(c["G60"]["path"],map_location="cpu",weights_only=False)
                weights=generator_state(value,60,read_json(c["sources"]["original_G_contract"]["path"]),c["G10"])
                g=cls(**ARCH);g.load_state_dict(weights,strict=True);g=g.to("cuda:0").eval().requires_grad_(False)
                del value,weights
            counts["G_loads"]=1;g_before=frozen(g)
            adapter=IRRAAdapter.from_source(ctx["source"],device="cuda:0",allow_model_load=True)
            counts["IRRA_loads"]=1;i_before=frozen(adapter._backend)
            def pre(module,args):
                need(len(args)==1 and len(args[0])==1 and not torch.is_grad_enabled(),"G single inference")
                counts["G_started"]+=1;need(counts["G_started"]<=2048,"2048 unique G budget")
            def post(*_):counts["G_completed"]+=1
            def enc(module,args):
                need(not torch.is_grad_enabled() and len(args[0])==1,"IRRA image inference only")
                counts["IRRA_images"]+=1
            stack.callback(g.register_forward_pre_hook(pre).remove);stack.callback(g.register_forward_hook(post).remove)
            stack.callback(adapter._backend.base_model.visual.register_forward_pre_hook(enc).remove)
            stack.enter_context(patch.object(adapter._backend,"encode_text",side_effect=ValueError("gallery text forbidden")))
            maximum=0.;features=[]
            for i,row in enumerate(ctx["report"]["identities"]):
                need(row["role"]=="train" and row["split"]=="train","Train-only generation")
                source=FileRef(str(Path(c["image_root"])/row["image_rel_path"]),row["image_sha256"]);checked(source)
                store.publish(f"claim-{i:04d}",{"claim.json":json_bytes({"index":i,"identity":row,"G60":c["G60"]})},
                      {"kind":"single-G-before-forward"},sources=(ref(root/"protocol/manifest.json"),source))
                from PIL import Image
                with Image.open(source.path) as image:pixels=pixel_reader(image).unsqueeze(0)
                with torch.inference_mode():
                    result=fn(g,pixels.to("cuda:0"));adv=result["adversarial"].cpu().clone()
                    arrays={"original":pixels.clone(),"adversarial":adv,
                            "effective_delta":result["effective_delta"].cpu().clone()}
                    stats=image_checks(arrays["original"],adv,arrays["effective_delta"],(384,128))
                    x=adapter.inference().encode_images(adv.to("cuda:0")).payload.cpu().clone()
                normalized(x,1);need(counts["G_started"]==counts["G_completed"]==counts["IRRA_images"]==i+1,"exact one forward")
                arrays["feature"]=x
                store.publish(f"image-{i:04d}",{"tensors.pt":tensor_bytes(arrays),"metadata.json":json_bytes({
                    "index":i,"identity":row,"pixels":stats,"feature_content_sha256":tensor_sha(x)})},
                    {"kind":"immutable-G60-train-RGB-feature"},sources=(ref(root/f"claim-{i:04d}/manifest.json"),))
                readback=load_tensors(root/f"image-{i:04d}/tensors.pt")
                need(all(torch.equal(v,readback[k]) for k,v in arrays.items()),"compact Tensor bitwise readback")
                maximum=max(maximum,stats["linf"]);features.append(x)
                if (i+1)%64==0:print("G60_TRAIN_GALLERY_PROGRESS",i+1,"/ 2048",flush=True)
            need(frozen(g)==g_before and frozen(adapter._backend)==i_before,"frozen G/IRRA state changed")
            report.update(status="G60_TRAIN_GALLERY_PASS",images=2048,max_linf=maximum,G_before=g_before,G_after=frozen(g),
                          IRRA_before=i_before,IRRA_after=frozen(adapter._backend),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        unchanged(ctx["report"])
        feature=torch.cat(features).clone();normalized(feature,2048)
        report.update(elapsed_seconds=time.monotonic()-begin,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        store.publish("completed",{"features.pt":tensor_bytes({"adversarial":feature}),"report.json":json_bytes(report)},
            {"kind":"complete-G60-Train-gallery","identities_sha256":digest(ctx["report"]["identities"]),
             "feature_content_sha256":tensor_sha(feature)},
            sources=tuple(ref(root/f"image-{i:04d}/manifest.json") for i in range(2048)))
        print("G60_TRAIN_GALLERY_PASS",json.dumps(report),flush=True)
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
        store.publish("failure",{"report.json":json_bytes(report)},{"kind":"preserved-gallery-failure"});raise

def audit_gallery(ctx):
    need(not torch.cuda.is_initialized(),"Gallery audit CPU only")
    root=BASE/"gallery_train";store=ArtifactStore(root,new_base=BASE)
    need(not store.pending() and not (root/"failure").exists(),"uncommitted/failed Gallery")
    expected={"protocol","completed"}|{f"{kind}-{i:04d}" for kind in ("claim","image") for i in range(2048)}
    need({p.name for p in root.iterdir() if p.is_dir()}==expected,"2048 complete Gallery claims/units")
    for name in expected:store.verify(name)
    features=[];maximum=0.
    for i,row in enumerate(ctx["report"]["identities"]):
        unit=root/f"image-{i:04d}";md=read_json(unit/"metadata.json");claim=read_json(root/f"claim-{i:04d}/claim.json")
        need(md["identity"]==claim["identity"]==row and md["index"]==claim["index"]==i and claim["G60"]==ctx["c"]["G60"],"Gallery member proof")
        x=load_tensors(unit/"tensors.pt")
        stats=image_checks(x["original"],x["adversarial"],x["effective_delta"],(384,128))
        need(stats==md["pixels"] and tensor_sha(x["feature"])==md["feature_content_sha256"],"pixel/feature SHA")
        normalized(x["feature"],1);features.append(x["feature"]);maximum=max(maximum,stats["linf"])
    gathered=torch.cat(features);saved=load_tensors(root/"completed/features.pt")["adversarial"]
    report=read_json(root/"completed/report.json")
    need(torch.equal(gathered,saved) and report["max_linf"]==maximum and report["counts"]["G_started"]==2048
         and report["counts"]["G_completed"]==2048 and report["counts"]["IRRA_images"]==2048
         and report["G_before"]==report["G_after"] and report["IRRA_before"]==report["IRRA_after"],"aggregate exact/frozen/count")
    unchanged(read_json(root/"protocol/preflight.json"))
    audit={"status":"G60_TRAIN_GALLERY_CPU_AUDIT_PASS","images":2048,"max_linf":maximum,
           "same_train_members":True,"G60":ctx["c"]["G60"],"feature_content_sha256":tensor_sha(saved)}
    ArtifactStore(BASE/"gallery_audit",new_base=BASE).publish("result",{"report.json":json_bytes(audit)},
        {"kind":"independent-G60-gallery-CPU-audit"},sources=(ref(root/"completed/manifest.json"),))
    print("G60_TRAIN_GALLERY_CPU_AUDIT_PASS",json.dumps(audit),flush=True)

class TextRuntime:
    """只有冻结IRRA文本推理；不持有G，图像前向硬拒绝。"""
    def __init__(self,source,probe):
        self.adapter=IRRAAdapter.from_source(source,device="cuda:0",allow_model_load=True)
        self.model=self.adapter._backend;self.probe=probe;self.before=frozen(self.model)
        self.versions=tuple((n,v._version) for n,v in self.model.state_dict(keep_vars=True).items())
        self.samples={k:0 for k in ("G","clean","adv","text")};self.calls=0
        self.original_image=self.model.encode_image;self.original_text=self.model.encode_text
        def no_image(*a,**k):raise ValueError("training/Validation cannot encode images; fixed caches only")
        def observed(tokens):
            need(not torch.is_grad_enabled() and torch.is_inference_mode_enabled(),"IRRA text inference only")
            self.samples["text"]+=len(tokens);self.calls+=1
            return self.original_text(tokens)
        self.model.encode_image=no_image;self.model.encode_text=observed
    def assert_frozen(self):
        need(all(not m.training for m in self.model.modules()) and
             all(not p.requires_grad and p.grad is None for p in self.model.parameters()) and
             tuple((n,v._version) for n,v in self.model.state_dict(keep_vars=True).items())==self.versions,"IRRA freeze/version changed")
    def encode_text(self,texts):
        self.assert_frozen()
        reports=[self.probe.inspect(t) for t in texts]
        need(not any(r.truncated for r in reports),"new text truncation forbidden")
        tokens=torch.tensor([r.model_tokens for r in reports],dtype=torch.long)
        result=self.adapter.inference().encode_tokens(tokens).payload.cpu().clone()
        normalized(result,len(texts));self.assert_frozen();return result
    def finish(self):
        self.assert_frozen();after=frozen(self.model);need(after==self.before,"IRRA full SHA changed")
        return {"IRRA_state_before":self.before,"IRRA_state_after":after,"samples":self.samples,
                "calls":self.calls,"IRRA_loads":1,"G_loads":0,"G_forwards":0,"parameters_buffers_unchanged":True,
                "GPU_peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                "GPU_peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}

class CachedBank:
    role="train"
    def __init__(self,clean,adv):
        normalized(clean,2048);normalized(adv,2048);self.features={"clean":clean,"adversarial":adv}
    def gather(self):return self.features

def load_bank(ctx):
    root=BASE/"gallery_train";store=ArtifactStore(root,new_base=BASE);store.verify("completed")
    proof=read_json(BASE/"gallery_audit/result/report.json")
    need(proof["status"]=="G60_TRAIN_GALLERY_CPU_AUDIT_PASS" and proof["G60"]==ctx["c"]["G60"],"audited G60 bank required")
    adv=load_tensors(root/"completed/features.pt")["adversarial"]
    need(tensor_sha(adv)==proof["feature_content_sha256"],"bank content")
    return CachedBank(read_old_features(ctx,"train","clean"),adv)

def validation(ctx,old,policy,runtime,epoch,checkpoint_sha):
    root=BASE/"validation"/f"epoch_{epoch:04d}";store=ArtifactStore(root,new_base=BASE)
    need(not root.exists(),"Validation immutable candidate already exists")
    identities=ctx["report"]["validation_identities"];probe=runtime.probe
    original=read_old_features(ctx,"validation","text");clean=read_old_features(ctx,"validation","clean")
    adv=load_tensors(Path(ctx["c"]["validation_root"])/"features/gallery.pt")["adversarial"]
    normalized(adv,128);features=[];actions=[];before=tree_hash(policy.state_dict())
    policy.eval().requires_grad_(False)
    for i,entry in enumerate(ctx["val_qualification"]["entries"]):
        if entry["eligible"]:
            with torch.inference_mode():result=policy.attack(old("formal_training.data").example_from(entry))
            text=result.adversarial;action=result.record()
        else:text=identities[i]["caption"];action={"NOOP":True,"reasons":entry["reasons"],"edits":[],"budget_k":0}
        x=original[i:i+1].clone() if text==identities[i]["caption"] else runtime.encode_text((text,))
        normalized(x,1);features.append(x)
        record={"identity":identities[i],"adversarial":text,"action":action,
                "checkpoint_sha256":checkpoint_sha,"IRRA_model_sha256":IRRA}
        store.publish(f"query-{i:03d}",{"text.pt":tensor_bytes({"text":x}),"record.json":json_bytes(record)},
                    {"kind":"new-H-Validation-caption-only"})
        actions.append(record)
    edited=torch.cat(features);pids=torch.tensor([s["pid"] for s in identities])
    groups={}
    for name,text,image in (("Clean",original,clean),("Image-only",original,adv),
                             ("Text-only",edited,clean),("Joint",edited,adv)):
        if name in ("Clean","Image-only"):
            prior=read_json(Path(ctx["c"]["validation_root"])/"completed/summary.json")
            hist=read_json(ctx["c"]["sources"]["history_evidence"]["path"])
            groups[name]=({k:hist["candidates"][0]["groups"]["Clean"][k] for k in KEYS} if name=="Clean"
                          else prior["groups"]["G60 Image-only"])
        else:
            scores=text@image.T;groups[name]=official_metrics(scores,pids,ctx["source"].ref("utils/metrics.py").path)
            store.publish(name.lower(),{"scores.pt":tensor_bytes({"scores":scores})},
                 {"kind":"IRRA-native-validation-score","text_sha256":tensor_sha(edited),"image_sha256":tensor_sha(image)})
    need(tree_hash(policy.state_dict())==before,"Validation changed H")
    summary={"epoch":epoch,"visit":epoch*2048,"checkpoint_sha256":checkpoint_sha,
             "gallery":128,"query":128,"groups":groups,"diagnostic_reused_validation":True,
             "G60":ctx["c"]["G60"],"G60_image_feature_sha256":tensor_sha(adv),
             "all_queries_retained":True,"H_unchanged":True,
             "extra_drop_pp":groups["Image-only"]["mAP"]-groups["Joint"]["mAP"]}
    store.publish("completed",{"summary.json":json_bytes(summary),
          "metrics_raw.csv":metrics_csv([{"Group":k,**v} for k,v in groups.items()]),
          "metrics_2dp.csv":metrics_csv([{"Group":k,**v} for k,v in groups.items()],True)},
          {"kind":"complete-new-H-Validation"},sources=tuple(ref(root/f"query-{i:03d}/manifest.json") for i in range(128)))
    policy.requires_grad_(True).train()
    print("HV1_G60_VALIDATION_COMPLETE",epoch,groups["Joint"]["mAP"],flush=True)

def execute(ctx,*,smoke=False,resume=False):
    c=ctx["c"];root=BASE/("smoke_v1" if smoke else "training")
    need(resume or not root.exists(),"independent run already exists")
    need(not (root/"pending").exists(),"uncommitted Step may exist; no automatic rollback")
    resources_snapshot=resource_check(c)
    bankdata=load_bank(ctx)
    begin=time.monotonic();ctx["gate"].initialize_cuda()
    with fp32_runtime():
        # 官方加载在legacy导入隔离之外完成，避免model命名空间冲突。
        source_probe=None
        with legacy_imports(ctx["legacy"],allow_import=True) as old:
            source_probe=old("tokenizer_probe").official_probe()
        runtime=TextRuntime(ctx["source"],source_probe)
        with legacy_imports(ctx["legacy"],allow_import=True) as old:
            catalog=old("training_inputs").load_training_catalog()
            frontend=old("text_frontend").CaptionFrontend();probe=old("tokenizer_probe").official_probe();runtime.probe=probe
            protocol=make_protocol(ctx,old)
            cfg=protocol["config"];ck=old("formal_training.checkpoint")
            policy=old("policy").CharacterPolicy(probe,**cfg["architecture"])
            optimizer=old("phase4_training").optimizer_for(policy,cfg);sampler=torch.Generator()
            bank=old("formal_training.runtime").RewardBank(catalog,bankdata,frontend)
            if resume:
                payload,evidence=audit_versions(root,protocol,old);need(payload is not None,"no committed child")
                ck.restore(payload,policy,optimizer,sampler);cursor,updates=payload["cursor"],payload["updates"];parent=evidence["parent"]
            else:
                payload=restore_parent(old,c,policy,optimizer,sampler,allow_cuda=True)
                cursor,updates,parent=20480,20370,H10
            restoration={"H_exact":True,"Adam_exact":True,"RNG_exact":True,"cursor":cursor,"updates":updates,
                "H_state_sha256":tree_hash(policy.state_dict()),"Adam_state_sha256":tree_hash(optimizer.state_dict()),
                "RNG_state_sha256":tree_hash(ck.capture_rng(sampler)),"parent_checkpoint_sha256":parent,
                "not_new_Adam":True,"resources":resources_snapshot}
            root.mkdir(parents=True,exist_ok=resume)
            old("formal_training.common").immutable_json(root/"protocol.json",protocol)
            # 独立锁，不调用旧safe_root/rollback/current.json。
            with old("formal_training.common").run_lock(root),old("formal_training.common").Stop() as stop:
                old("formal_training.common").immutable_json(root/("restore_"+str(cursor)+".json"),restoration)
                if not (root/"started.json").exists():
                    old("formal_training.common").immutable_json(root/"started.json",{"start_time":time.time()})
                samples={x.row_id:x for x in catalog.train}
                entries={e["identity"]["row_id"]:e for e in ctx["train_qualification"]["entries"]}
                work=None;initial=cursor;epoch_begin=time.monotonic()
                while cursor<len(protocol["query_order"]):
                    if smoke and cursor-initial>=c["smoke_visits"]:break
                    if work is None:work=ck.begin(root,protocol,parent);start=cursor;start_updates=updates
                    q=samples[protocol["query_order"][cursor]];catalog.require_train(q);entry=entries[q.row_id]
                    tic=time.monotonic()
                    if entry["eligible"]:
                        ex=old("formal_training.data").example_from(entry)
                        log,_=old("phase4_training").train_step(policy,optimizer,ex,sampler,
                                  lambda k:bank.bind(q,runtime,ex,k),updates+1,q.row_id)
                        updates+=1
                    else:
                        log={"row_id":q.row_id,"optimizer_steps_this_step":0,"skip_reasons":entry["reasons"],
                             "H_state_sha256_after":tree_hash(policy.state_dict()),"NOOP":True,"IRRA_text_samples":0}
                    cursor+=1;epoch=(cursor-1)//2048+1
                    log.update(visit=cursor,epoch=epoch,epoch_position=(cursor-1)%2048+1,
                               committed_update_candidate=updates,seconds=time.monotonic()-tic,
                               GPU_peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                               GPU_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                    ck.append(work,log);runtime.assert_frozen()
                    boundary=cursor%2048==0;probe_end=smoke and cursor-initial>=c["smoke_visits"]
                    if boundary or updates-start_updates>=cfg["checkpoint_interval_updates"] or probe_end:
                        _,parent=ck.publish(root,work,policy,optimizer,sampler,protocol,cursor,updates,parent,start,start_updates,
                                            "smoke_probe" if smoke else "epoch_complete" if boundary else "periodic")
                        work=None
                        print("HV1_G60_CHECKPOINT_COMMITTED",cursor,updates,parent,flush=True)
                    if cursor%32==0 or probe_end:
                        print("HV1_G60_TRAIN_PROGRESS",epoch,(cursor-1)%2048+1,"/ 2048","visit",cursor,"Adam",updates,flush=True)
                    if boundary:
                        need(updates==epoch*2037,"absolute eligible update count")
                        print("HV1_G60_EPOCH_COMPLETE",epoch,"Adam",updates,"seconds",time.monotonic()-epoch_begin,flush=True)
                        epoch_begin=time.monotonic()
                        if epoch in (20,30,40,50,60):validation(ctx,old,policy,runtime,epoch,parent)
                        if stop.requested:break
                last,ledger=audit_versions(root,protocol,old)
                need(ck.same(last["policy"],policy.state_dict()) and ck.same(last["optimizer"],optimizer.state_dict()),"last H/Adam exact")
                saved_rng=ck.capture_rng(sampler)
                other=old("policy").CharacterPolicy(probe,**cfg["architecture"])
                opt2=old("phase4_training").optimizer_for(other,cfg);gen2=torch.Generator()
                ck.restore(last,other,opt2,gen2);ck.restore_rng(saved_rng,sampler)
                result={"status":"HV1_G60_SMOKE_PASS" if smoke else "HV1_G60_TRAINING_PASS" if cursor==122880 else "PARTIAL",
                        "cursor":cursor,"updates":updates,"initial_cursor":initial,"new_visits":cursor-20480,
                        "new_updates":updates-20370,"attempt_new_visits":cursor-initial,
                        "attempt_new_updates":updates-restoration["updates"],"final_epoch":cursor//2048,
                        "parent_H10":c["parent"],"last_checkpoint_sha256":parent,"runtime":runtime.finish(),
                        "H_Adam_RNG_reload_exact":True,"H_changed":tree_hash(policy.state_dict())!=restoration["H_state_sha256"],
                        "elapsed_seconds":time.time()-read_json(root/"started.json")["start_time"],
                        "attempt_elapsed_seconds":time.monotonic()-begin,"ledger":ledger,
                        "resume_count":max(0,len(list(root.glob("restore_*.json")))-1),
                        "smoke_not_formal_Epoch11":smoke,"model_updates_only_H":True}
                if smoke:
                    logs=ck.logs(Path(ledger["published"][-1]["path"])/"steps.jsonl")
                    need(result["H_changed"] and any(x.get("gradient_norm",0)>0 for x in logs),"Smoke H update/gradient not verified")
                target=root/("partial_attempts/visit_"+str(cursor)+".json") if result["status"]=="PARTIAL" else root/"completed.json"
                old("formal_training.common").immutable_json(target,result)
                print(result["status"],json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--action",choices=["preflight","gallery","audit-gallery","smoke","train","audit","audit-smoke"],default="preflight")
    p.add_argument("--allow-gpu",action="store_true");p.add_argument("--allow-training",action="store_true");p.add_argument("--resume",action="store_true")
    a=p.parse_args();torch.set_num_threads(2);c=config(a.config);ctx=preflight(c,check_restore=a.action=="preflight")
    if a.action=="preflight":
        print("HV1_G60_CPU_PREFLIGHT_PASS",json.dumps({k:v for k,v in ctx["report"].items() if k not in ("sources","identities","validation_identities","closure")}),flush=True)
    elif a.action=="gallery":need(a.allow_gpu,"explicit GPU authorization");gallery(ctx)
    elif a.action=="audit-gallery":audit_gallery(ctx)
    elif a.action in ("smoke","train"):
        need(a.allow_gpu and a.allow_training,"explicit H-only update authorization")
        execute(ctx,smoke=a.action=="smoke",resume=a.resume)
    else:
        from .hv1_g60_audit import audit_training
        audit_training(ctx,smoke=a.action=="audit-smoke")
if __name__=="__main__":main()
