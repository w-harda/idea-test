"""官方SGA+IRRA源的薄运行器；默认CPU，Source与三个Target分进程。"""
from pathlib import Path
from dataclasses import asdict, replace
from contextlib import contextmanager
import argparse, csv, io, json, os, random, subprocess, sys, time, traceback
import numpy as np
import torch
from tbps.artifacts import ArtifactStore, FileRef, checked, read_json, json_bytes, digest, sha_file
from tbps.baselines.sga_official_io import official_modules, dataset, source_pixels, official_projection_check, OFFICIAL_COMMIT
from tbps.models.sga_irra_source import IRRASGASource, AuditedBert, require
from tbps.models.irra_source import IRRASource
from tbps.models.irra import IRRAAdapter
from tbps.models.irra_io import MEAN, STD
from tbps.engines.materialized_cache import ref, FeatureCache, CachedIRRA
from tbps.engines.native_cache import NativeFeatureCache, compact
from tbps.engines.provenance import AttackOrigin
from tbps.contracts import DatasetManifest, GalleryItem, QueryItem, PersonID, AttackBudget, ImageBudget, TextBudget
from tbps.models.base import InputBinding
from tbps.engines.evaluation import EvaluationEngine
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.engines.aptm_score_cache import APTMSetScorer
from tbps.engines.provenance import payload_digest
from tbps.smoke_irra import model_snapshot
from importlib.metadata import version

METRICS=("R@1","R@5","R@10","mAP")

def configuration(path):
    c=read_json(path,"tbps-sga-irra-source-1")
    require(c["official_commit"]==OFFICIAL_COMMIT and c["source_model"]=="irra", "fixed official SGA / IRRA source")
    require((c["epsilon"],c["steps"],c["step_size"],c["batch_size"],c["seed"],c["max_length"])==
            (2/255,10,.5/255,8,42,30) and c["scales"]==[.5,.75,1.25,1.5], "frozen official actual defaults")
    for x in c["official"].values(): checked(FileRef(**x))
    for p,h in c["producers"].items(): checked(FileRef(p,h))
    for p,h in c["official_files"].items(): checked(FileRef(p,h))
    require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=c["official_root"],text=True).strip()==OFFICIAL_COMMIT,
            "official SGA Git revision")
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=c["official_root"],text=True), "official source modified")
    for name,h in c["bert_files"].items():checked(FileRef(str(Path(c["bert"])/name),h))
    return c

def manifest_from_dict(x):
    return DatasetManifest(x["dataset_id"],x["official_split"],FileRef(**x["annotation"]),
        tuple(GalleryItem(**{**g,"pid":PersonID(**g["pid"])}) for g in x["gallery"]),
        tuple(QueryItem(**{**q,"pid":PersonID(**q["pid"])}) for q in x["queries"]), x["relevance"])

def preflight(path,scope):
    c=configuration(path)
    IRRASource.from_dict(c["irra"]).verify()
    ds,raw,matched=dataset(c,scope)
    require(not torch.cuda.is_initialized(),"CPU preflight cannot initialize CUDA")
    print("SGA_IRRA_CPU_PREFLIGHT_PASS",scope,len(raw.gallery),len(raw.queries),flush=True)
    return c,ds,raw,matched

@contextmanager
def tensor_archive(store,key,tensors,metadata,sources):
    # compact clone avoids old shared Storage files; no attacker calculation here.
    b=io.BytesIO();torch.save(compact(tensors),b)
    store.publish(key,{"pixels.pt":b.getvalue(),"metadata.json":json_bytes(metadata)},
                  {"kind":"SGA_IRRA_source_pixels"},sources=sources)
    store.verify(key)
    got=torch.load(store.root/key/"pixels.pt",map_location="cpu",weights_only=True)
    require(all(torch.equal(v,got[k]) for k,v in tensors.items()),"source Tensor bitwise readback")
    yield

class OfficialObservation:
    """委托原方法并观测真实调用，不替换Loss/候选/更新。"""
    def __init__(self,attacker,source,write):
        self.source=source;self.write=write;self.events=[];self.stage=0;self.selections=[]
        text=attacker.txt_attacker;image=attacker.img_attacker
        text_fn=text.img_guided_attack;image_fn=image.txt_guided_attack;loss_fn=text.loss_func
        def observed_text(net,texts,img_embeds=None):
            self.stage+=1
            require(self.stage in (1,3) and list(texts)==self.original,"T1 and T2 start from original pre_caption")
            self.events.append("text1" if self.stage==1 else "text2")
            result=text_fn(net,texts,img_embeds=img_embeds)
            if self.stage==1:self.intermediate=list(result)
            self.write({"event":self.events[-1],"input_texts":list(texts),"output_texts":result})
            return result
        def observed_image(model,imgs,txt2img,device,scales=None,txt_embeds=None):
            self.stage+=1;require(self.stage==2,"official text-image-text order")
            # SGAttacker刚编码的T1确实是图像攻击所收到的监督。
            require(source.last_text_requests==self.intermediate,"image guidance must encode actual T1")
            require(torch.equal(source.last_text[1],txt_embeds.detach().cpu()),"actual T1 feature supervision exact")
            self.events.append("image")
            self.write({"event":"image","T1":self.intermediate,"supervision_sha256":payload_digest(txt_embeds.detach().cpu())})
            return image_fn(model,imgs,txt2img,device,scales=scales,txt_embeds=txt_embeds)
        def observed_loss(txt_embeds,img_embeds,label):
            value=loss_fn(txt_embeds,img_embeds,label)
            row={"event":"candidate_choice","stage":self.stage,"query_local_index":label,
                 "candidate_texts":source.last_text_requests,"scores":value.detach().cpu().tolist(),
                 "selected_index":int(value.argmax())}
            self.selections.append(row);self.write(row)
            return value
        text.img_guided_attack=observed_text;image.txt_guided_attack=observed_image;text.loss_func=observed_loss
    def reset(self,texts):
        self.stage=0;self.original=list(texts);self.events=[];self.selections=[]

def source_run(path,scope):
    c,ds,raw,matched=preflight(path,scope)
    require({k:version(k) for k in c["source_versions"]}==c["source_versions"],"fixed original source runtime versions")
    root=Path(c["output_base"])/scope/"source";require(not root.exists(),"source Attempt exists; no overwrite/retry")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit GPU0")
    from tbps.engines.materialized_irra import gpu_available
    gpu=gpu_available({"physical_gpu":0});require(gpu["free_mib"]>=c["minimum_free_mib"],"insufficient shared GPU memory")
    store=ArtifactStore(root,new_base=c["output_base"])
    store.publish("protocol",{"protocol.json":json_bytes({"config":asdict(ref(path)),"source":"irra",
        "raw_manifest":asdict(raw),"matched_manifest":asdict(matched),"targets_feedback":False,
        "source_preprocess":"PIL RGB bilinear384x128; official SGA Normalize; pre_caption30",
        "SGAttacker_max_length":30,"TextAttacker_max_length":30})},{"kind":"SGA_IRRA_source"},
        sources=(ref(path),FileRef(**c["irra"]["checkpoint"])))
    report={"status":"RUNNING","scope":scope,"source_model_loads":0,"MLM_loads":0,"target_models_in_generation":[],
            "optimizer":0,"G_H_loads":0,"official_commit":OFFICIAL_COMMIT,"GPU_start":gpu,
            "runtime_versions":{k:version(k) for k in c["source_versions"]}}
    start=time.monotonic()
    try:
        from torchvision import transforms
        from transformers import BertForMaskedLM
        torch.set_num_threads(2);torch.manual_seed(c["seed"]);np.random.seed(c["seed"]);random.seed(c["seed"])
        torch.backends.cudnn.benchmark=True
        # 仅关闭IRRA明确禁止的TF32，不替换SGA的增强、backward或投影。
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        a=IRRAAdapter.from_source(IRRASource.from_dict(c["irra"]),device="cuda:0",allow_model_load=True)
        report["source_model_loads"]=1
        official,token_module,_=official_modules(c)
        tokenizer=AuditedBert(token_module.BertTokenizer.from_pretrained(c["bert"],local_files_only=True,
                                                                       clean_up_tokenization_spaces=True))
        mlm=BertForMaskedLM.from_pretrained(c["bert"],local_files_only=True).eval().requires_grad_(False).to("cuda:0")
        report["MLM_loads"]=1
        trace_path=root/"bridge_trace.jsonl";events_path=root/"official_flow.jsonl"
        with trace_path.open("x") as trace,events_path.open("x") as events:
            def bridge(row):trace.write(json.dumps(row,ensure_ascii=False)+"\n")
            def event(row):events.write(json.dumps(row,ensure_ascii=False)+"\n")
            s=IRRASGASource(a._backend,a._tokenizer,tokenizer,write_trace=bridge)
            mlm_count={"calls":0,"samples":0}
            def mlm_hook(mod,args,kwargs):
                mlm_count["calls"]+=1;mlm_count["samples"]+=len(args[0])
            hook=mlm.register_forward_pre_hook(mlm_hook,with_kwargs=True)
            before=model_snapshot(s.backend);mlm_before=model_snapshot(mlm)
            report["model_before"]=before;report["MLM_before"]=mlm_before
            norm=transforms.Normalize(MEAN,STD)
            image=official.ImageAttacker(norm,eps=2/255,steps=10,step_size=.5/255)
            text=official.TextAttacker(mlm,tokenizer,cls=False,max_length=30,number_perturbation=1,topk=10,threshold_pred_score=.3)
            attacker=official.SGAttacker(s,image,text)
            observation=OfficialObservation(attacker,s,event)
            # 单次输入梯度检查，与正式攻击调用分开记录。
            original=source_pixels(ds.image_path(raw.gallery[0].image_rel_path)).unsqueeze(0).to("cuda:0").requires_grad_()
            inputs=tokenizer([matched.queries[0].caption],padding="max_length",truncation=True,max_length=30,return_tensors="pt").to("cuda:0")
            z=s.inference_text(inputs)["text_feat"].detach()
            grad,=torch.autograd.grad((s.inference_image(norm(original))["image_feat"]*z).sum(),original)
            require(torch.isfinite(grad).all() and grad.norm()>0,"finite nonzero native IRRA input gradient")
            report["input_gradient_check"]={"norm":float(grad.norm()),"finite":True,"shape":list(grad.shape)}
            del original,inputs,z,grad
            report["gradient_check_counts"]=dict(s.counts)
            units=[];final_texts={};max_linf=0.
            for start_index in range(0,len(raw.gallery),8):
                gs=raw.gallery[start_index:start_index+8];ids=[g.image_id for g in gs]
                qs=[q for q in matched.queries if q.paired_image_id in ids]
                texts=[q.caption for q in qs];mapping=[ids.index(q.paired_image_id) for q in qs]
                key="batch-"+str(start_index//8).zfill(6)
                store.publish("claim-"+key,{"claim.json":json_bytes({"gallery_ids":ids,"query_ids":[q.query_id for q in qs]})},
                              {"kind":"SGA_IRRA_claim"},sources=(ref(root/"protocol/manifest.json"),))
                pixels=torch.stack([source_pixels(ds.image_path(g.image_rel_path)) for g in gs]).to("cuda:0")
                observation.reset(texts);timer=time.monotonic()
                # 唯一攻击入口：固定官方SGAttacker.attack本体。
                adv,adv_texts=attacker.attack(pixels,texts,mapping,device="cuda:0",max_length=30,scales=c["scales"])
                require(observation.events==["text1","image","text2"],"official complete flow")
                check=official_projection_check(pixels,adv);max_linf=max(max_linf,check["real_linf"])
                md={"gallery_ids":ids,"query_ids":[q.query_id for q in qs],"clean_texts":texts,"adv_texts":adv_texts,
                    "T1":observation.intermediate,"flow":observation.events,"budget":check,
                    "seconds":time.monotonic()-timer,
                    "no_edit_reason":[None if x!=y else "official argmax kept original or all positions/candidates filtered"
                                      for x,y in zip(texts,adv_texts)],
                    "selected_word_actions":sum(row["selected_index"]>0 for row in observation.selections if row["stage"]==3)}
                cpu={"original":pixels.detach().cpu().clone(),"adversarial":adv.detach().cpu().clone(),
                     "effective_delta":(adv-pixels).detach().cpu().clone()}
                with tensor_archive(store,key,cpu,md,(ref(root/"protocol/manifest.json"),ref(root/("claim-"+key)/"manifest.json"))):pass
                units.append(key);final_texts.update(zip([q.query_id for q in qs],adv_texts))
                print("SGA_OFFICIAL_SOURCE_PROGRESS",scope,start_index+len(gs),"/",len(raw.gallery),flush=True)
                del pixels,adv,cpu
            hook.remove()
            after=model_snapshot(s.backend);mlm_after=model_snapshot(mlm)
            require(before==after and mlm_before==mlm_after,"IRRA/MLM weights or buffers changed")
            require(all(not p.requires_grad and p.grad is None for net in (s.backend,mlm) for p in net.parameters()),"source parameter freeze/gradients")
            trace.flush();events.flush()
            report.update(status="PASS",model_after=after,MLM_after=mlm_after,source_counts=s.counts,MLM_counts=mlm_count,
                          source_frozen=True,max_real_linf=max_linf)
            collection={"manifest":asdict(matched),"raw_manifest":asdict(raw),"units":units,"adv_texts":final_texts,
                        "official_source_calls":len(units),"Gallery":len(raw.gallery),"Query":len(raw.queries)}
        store.publish("completed",{"collection.json":json_bytes(collection)},{"kind":"SGA_IRRA_source_complete"},
                      sources=(ref(root/"protocol/manifest.json"),ref(trace_path),ref(events_path),
                               *(ref(root/key/"manifest.json") for key in units)))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc());print(report["traceback"],flush=True)
    finally:
        from tbps.engines.replay_control import rss_bytes
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss_bytes())
        if torch.cuda.is_initialized():
            report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"SGA_IRRA_execution"},sources=(ref(root/"protocol/manifest.json"),))
    print("SGA_OFFICIAL_IRRA_SOURCE_"+report["status"],scope,flush=True)
    return int(report["status"]!="PASS")

def source_inputs(c,scope):
    root=Path(c["output_base"])/scope/"source";store=ArtifactStore(root,new_base=c["output_base"])
    for key in ["completed","execution","protocol"]:store.verify(key)
    require(read_json(root/"execution/report.json")["status"]=="PASS","source not complete")
    x=read_json(root/"completed/collection.json")
    return store,x,manifest_from_dict(x["manifest"])

def encode_target(c,scope,model):
    root=Path(c["output_base"])/scope/model;require(not root.exists(),"target Attempt exists")
    from tbps.models.irra import fp32_runtime
    from tbps.baselines.comparison import load_model
    src,collection,manifest=source_inputs(c,scope);store=ArtifactStore(root,new_base=c["output_base"])
    # Existing model loader needs only these source references; no B3 attack/config is used.
    loader_config={"inputs":{"dataset_sources":c["dataset_sources"]},"victim_sources":c["victim_sources"]}
    start=time.monotonic();report={"status":"RUNNING","scope":scope,"model":model,"target_loads":0}
    try:
        with fp32_runtime(),load_model(loader_config,model,scope) as (adapter,backend,checkpoint,counts):
            report["target_loads"]=1;before=model_snapshot(backend)
            protocol={"schema":"tbps-materialized-irra-cache-1" if model=="irra" else "tbps-SGA-native-target-1",
                      "model_space":asdict(adapter.space),"G_H_generation":False,"ranking_feedback_to_attack":False,
                      "source_collection":asdict(ref(src.root/"completed/manifest.json")),"manifest":asdict(manifest)}
            store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"kind":"SGA_target"},
                          sources=(checkpoint,ref(src.root/"completed/manifest.json"),ref(c["path"])))
            if model=="irra":cache=FeatureCache(store,protocol,adapter.space,chunk=16)
            else:
                mod=__import__("tbps.models."+model,fromlist=["validate_payload"])
                cache=NativeFeatureCache(store,protocol,adapter.space,mod.validate_payload,chunk=16)
            bindings={};inputs={}
            for family in ("clean_gallery","adv_gallery","clean_text","adv_text"):
                image=family.endswith("gallery");adv=family.startswith("adv")
                ids=tuple(g.image_id for g in manifest.gallery) if image else tuple(q.query_id for q in manifest.queries)
                originals=tuple(g.image_sha256 for g in manifest.gallery) if image else tuple(__import__("hashlib").sha256(q.caption.encode()).hexdigest() for q in manifest.queries)
                hashes=[];parts=[]
                if image:
                    for unit in collection["units"]:
                        src.verify(unit);raw=torch.load(src.root/unit/"pixels.pt",map_location="cpu",weights_only=True)
                        xs=raw["adversarial" if adv else "original"].detach().clone();del raw
                        for x in xs:
                            hashes.append(payload_digest(x.unsqueeze(0).clone()) if adv else originals[len(hashes)])
                            value=adapter.inference().encode_images(x.unsqueeze(0).to("cuda:0")).payload.cpu() if model=="irra" else adapter.encode_images(x.unsqueeze(0))
                            parts.append(compact(value))
                        del xs
                else:
                    texts=[collection["adv_texts"][q.query_id] if adv else q.caption for q in manifest.queries]
                    hashes=[__import__("hashlib").sha256(t.encode()).hexdigest() for t in texts]
                    for ti in range(0,len(texts),16):
                        value=adapter.inference().encode_texts(texts[ti:ti+16])[0].payload.cpu() if model=="irra" else adapter.encode_texts(texts[ti:ti+16])
                        parts.append(compact(value))
                from tbps.baselines.comparison import payload_cat,payload_select
                all_values=payload_cat(parts);del parts
                binding=InputBinding(adapter.space.id,digest(asdict(manifest)),tuple(hashes),"image" if image else "text",ids)
                for start_i in range(0,len(ids),16):
                    end=min(start_i+16,len(ids));cache.publish(family,binding,start_i,end,payload_select(all_values,list(range(start_i,end))))
                del all_values
                budget=AttackBudget(image=ImageBudget("linf",2/255,"rgb_0_1")) if image else AttackBudget(text=TextBudget("word",1,None,"SGA_official_"+OFFICIAL_COMMIT))
                origin=AttackOrigin("sga_official_irra_"+("image" if image else "text"),FileRef(**c["irra"]["checkpoint"]),
                                   ref(src.root/"protocol/manifest.json"),budget,sha_file(c["path"])) if adv else None
                inputs[family]=cache.gather(family,binding,origin=origin,originals=originals if adv else ())
            scorer=CachedIRRA(adapter.space,checkpoint,cache_protocol=ref(root/"protocol/manifest.json")) if model=="irra" else (
                   ChunkCachedScorer if model=="rasa" else APTMSetScorer)(adapter,inputs,store,ref(root/"protocol/manifest.json"))
            engine=EvaluationEngine(manifest,scorer,direction="higher_better",
                        execution="readonly_cached" if model=="irra" else "materialized_native",model_sources=(checkpoint,))
            p,result,scores=engine.run(inputs,query_chunk=128)
            groups={mode:{k:value[k] for k in METRICS} for mode,value in result["groups"].items()}
            b=io.StringIO();writer=csv.writer(b,lineterminator="\n");writer.writerow(("Mode",*METRICS))
            for mode in ("Clean","Image-only","Text-only","Joint"):writer.writerow((mode,*(format(groups[mode][k],".2f") for k in METRICS)))
            out=io.BytesIO();torch.save(compact(scores),out)
            require(model_snapshot(backend)==before,"target changed")
            store.publish("completed",{"summary.json":json_bytes({"scope":scope,"Gallery":len(manifest.gallery),"Query":len(manifest.queries),
                "groups":groups,"matched_clean":True,"model":model,"source":"irra","metric_unit":"percent"}),
                "metrics_2dp.csv":b.getvalue().encode(),"scores.pt":out.getvalue()},
                {"kind":"SGA_IRRA_target_complete"},sources=(ref(root/"protocol/manifest.json"),))
            report.update(status="PASS",model_before=before,model_after=model_snapshot(backend),counts=dict(counts),
                          neural_counts=dict(counts if model=="irra" else adapter.neural_counts),
                          source_regeneration=0)
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc());print(report["traceback"],flush=True)
    finally:
        from tbps.engines.replay_control import rss_bytes
        report.update(elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=rss_bytes())
        if torch.cuda.is_initialized():report.update(GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if not (root/"protocol").exists():store.publish("protocol",{"protocol.json":json_bytes({"model":model})},{"kind":"failed_target"},sources=(ref(c["path"]),))
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"SGA_target_execution"},sources=(ref(root/"protocol/manifest.json"),))
    print("SGA_OFFICIAL_TARGET_"+report["status"],scope,model,flush=True)
    return int(report["status"]!="PASS")

def audit(c,scope):
    require(not torch.cuda.is_initialized(),"audit CPU-only")
    base=Path(c["output_base"])/scope
    for name in ("source","irra","rasa","aptm"):
        root=base/name;store=ArtifactStore(root,new_base=c["output_base"])
        require(not store.pending(),"pending SGA output")
        for path in root.glob("*/manifest.json"):store.verify(path.parent.name)
        require(read_json(root/"execution/report.json")["status"]=="PASS","incomplete stage")
    src,x,manifest=source_inputs(c,scope)
    gallery_ids=[];query_ids=[]
    for key in x["units"]:
        md=read_json(src.root/key/"metadata.json");raw=torch.load(src.root/key/"pixels.pt",map_location="cpu",weights_only=True)
        official_projection_check(raw["original"],raw["adversarial"])
        require(torch.equal(raw["adversarial"]-raw["original"],raw["effective_delta"]),"source effective delta")
        require(md["flow"]==["text1","image","text2"],"official source flow proof")
        claim=read_json(src.root/("claim-"+key)/"claim.json")
        require(claim["gallery_ids"]==md["gallery_ids"] and claim["query_ids"]==md["query_ids"],"claim correspondence")
        gallery_ids.extend(md["gallery_ids"]);query_ids.extend(md["query_ids"])
        for q,t in zip(md["query_ids"],md["adv_texts"]):require(x["adv_texts"][q]==t,"frozen final text binding")
    require(gallery_ids==[g.image_id for g in manifest.gallery] and query_ids==[q.query_id for q in manifest.queries],"complete official ordered members")
    bridge=[json.loads(line) for line in (src.root/"bridge_trace.jsonl").read_text().splitlines()]
    require(bridge and all(t["wordpiece_roundtrip_exact"] and not t["irra_truncated"] for t in bridge),"candidate bridge exact / no added BPE truncation")
    for model in ("irra","rasa","aptm"):
        s=read_json(base/model/"completed/summary.json");require(s["Gallery"]==len(manifest.gallery) and s["Query"]==len(manifest.queries) and s["matched_clean"],"matched identity")
    audit_store=ArtifactStore(base/"audit",new_base=c["output_base"])
    audit_store.publish("completed",{"audit.json":json_bytes({"status":"PASS","scope":scope,"Gallery":len(manifest.gallery),
        "Query":len(manifest.queries),"bridge_rows":len(bridge),"source_models": ["irra","BERT_MLM"],
        "target_feedback":False,"GPU":False})},{"kind":"SGA_IRRA_independent_CPU_audit"},
        sources=tuple(ref(base/name/"completed/manifest.json") for name in ("source","irra","rasa","aptm")))
    print("SGA_OFFICIAL_IRRA_CPU_AUDIT_PASS",scope,flush=True)
    return 0

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--scope",choices=("smoke","full"),default="smoke")
    p.add_argument("--action",choices=("preflight","source","evaluate","audit","pipeline"),default="preflight")
    p.add_argument("--model",choices=("irra","rasa","aptm"));p.add_argument("--allow-gpu-smoke",action="store_true");p.add_argument("--allow-full",action="store_true")
    a=p.parse_args(argv);c=configuration(a.config);c["path"]=str(Path(a.config).resolve())
    if a.action=="preflight":preflight(a.config,a.scope);return 0
    if a.action=="audit":return audit(c,a.scope)
    require((a.scope=="smoke" and a.allow_gpu_smoke and not a.allow_full) or (a.scope=="full" and a.allow_full and not a.allow_gpu_smoke),"explicit GPU scope")
    if a.action=="source":return source_run(a.config,a.scope)
    if a.action=="evaluate":require(a.model is not None,"target model");return encode_target(c,a.scope,a.model)
    base=Path(c["output_base"])/a.scope
    require(not base.exists(),"existing pipeline; no automatic retry")
    base.mkdir(parents=True)
    commands=[("source",c["source_python"],"source",None)]+[(m,c["target_python"],"evaluate",m) for m in ("irra","rasa","aptm")]+[("audit",c["target_python"],"audit",None)]
    for name,python,action,model in commands:
        cmd=[python,"-B","-m","tbps.baselines.sga_irra","--config",a.config,"--scope",a.scope,"--action",action]
        if action!="audit":cmd += ["--allow-gpu-smoke" if a.scope=="smoke" else "--allow-full"]
        if model:cmd += ["--model",model]
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"" if action=="audit" else "0"}
        with (base/(name+".log")).open("x") as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        print("SGA_PIPELINE_STAGE",name,result.returncode,flush=True)
        if result.returncode:return result.returncode
    print("SGA_OFFICIAL_IRRA_PIPELINE_PASS",a.scope,flush=True)
    return 0

if __name__=="__main__":raise SystemExit(main())
