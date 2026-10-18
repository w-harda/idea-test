"""SGA TBPS白盒：每目标独立生成Gallery，再用既有原生Engine排名。默认CPU。"""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,csv,io,json,os,time,traceback,subprocess
from unittest.mock import patch
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest,sha_file
from tbps.baselines.sga_attack import (need,official_image_attacker,attack_image,caption_groups,
    sample_seed,OFFICIAL_COMMIT,SCALES,EPSILON,STEP_SIZE,STEPS)
from tbps.models.sga_surrogate import SGASurrogate,OBJECTIVES
# 已验证的只读Full桥、原生加载及payload工具；不调用B3配置、像素或攻击。
from tbps.baselines.comparison import FullReference,load_model,payload_select,payload_cat,four
from tbps.engines.materialized_inputs import load_inputs
from tbps.engines.materialized_irra import gpu_available
from tbps.models.irra import fp32_runtime
from tbps.models.base import InputBinding
from tbps.engines.materialized_cache import ref,FeatureCache,CachedIRRA
from tbps.engines.native_cache import NativeFeatureCache
from tbps.engines.native_score_cache import ChunkCachedScorer
from tbps.engines.aptm_score_cache import APTMSetScorer
from tbps.engines.evaluation import EvaluationEngine
from tbps.engines.provenance import AttackOrigin
from tbps.contracts import AttackBudget,ImageBudget
from tbps.materialization.store import image_checks
from tbps.materialization.bulk_session import BulkSession
from tbps.smoke_irra import model_snapshot

KEYS=("R@1","R@5","R@10","mAP")
METHODS=("Clean","SGA-TBPS White-box")
BUDGET=AttackBudget(image=ImageBudget("linf",EPSILON,"rgb_0_1"))

def configuration(path):
    c=read_json(path,"tbps-sga-whitebox-1")
    need(c["models"]==["irra","rasa","aptm"] and c["steps"]==STEPS and c["step_size"]==STEP_SIZE
         and c["epsilon"]==EPSILON and tuple(c["scales"])==SCALES and c["seed"]==42
         and c["precision"]=="fp32/math/no_tf32" and c["metrics"]==list(KEYS)
         and c["caption_policy"]=="all_original_paired_captions_in_official_order"
         and c["image_batch"]==1 and c["query_chunk"]==128,"frozen SGA TBPS protocol")
    need(c["official_commit"]==OFFICIAL_COMMIT,"pinned official SGA commit")
    need(subprocess.check_output(["git","rev-parse","HEAD"],cwd=c["official_root"],text=True).strip()==OFFICIAL_COMMIT
         and not subprocess.check_output(["git","status","--porcelain"],cwd=c["official_root"],text=True),
         "official SGA checkout changed")
    expected={str(p) for p in Path(c["official_root"]).rglob("*") if p.is_file() and p.suffix==".py"}
    expected|={str(Path(c["official_root"])/"LICENSE"),str(Path(c["official_root"])/"README.md")}
    need(set(c["official_files"])==expected,"complete pinned SGA sources")
    root=Path(__file__).resolve().parents[3]
    producers={str(p) for d in ("models","engines","baselines","materialization","attacks","datasets") for p in (root/"src/tbps"/d).rglob("*.py")}
    producers|={str(root/"src/tbps"/n) for n in ("sga_eval.py","artifacts.py","contracts.py","config.py","runtime_closure.py","smoke_irra.py")}
    need(set(c["producers"])==producers,"complete explicit SGA runtime sources")
    for p,h in {**c["official_files"],**c["producers"]}.items():checked(FileRef(p,h))
    need(c["attacker"]=={"path":str(Path(c["official_root"])/"attacker.py"),"sha256":c["official_files"][str(Path(c["official_root"])/"attacker.py")]},"official ImageAttacker source")
    need(c["objectives"]==OBJECTIVES and c["gpu_policy"]=="shared_observe_only"
         and c["minimum_free_mib"]==3584,"explicit native objectives/shared resource floor")
    return c

def preflight(path,scope,model):
    c=configuration(path)
    from torchvision import transforms
    official=official_image_attacker(c["attacker"])
    # 先完成官方依赖/Tokenizer导入再建立CPUStartupGate快照，不加载真实权重。
    def forbidden(*a,**k):raise RuntimeError("SGA CPU preflight forbids model deserialization/CUDA")
    with patch.object(torch,"load",forbidden),patch.object(torch.cuda,"_lazy_init",forbidden):
        if model!="irra":
            mod=__import__("tbps.models."+model,fromlist=["native_modules"])
            source=(mod.RasaSource if model=="rasa" else mod.APTMSource)(c["victim_sources"][model]["path"])
            checked(FileRef(**c["victim_sources"][model]))
            with mod.native_modules(source) as (_,tok,utils):
                tokenizer=mod.official_tokenizer(tok,source)
                need(tokenizer(["A person in a shirt."],return_tensors="pt").input_ids.ndim==2,"native Tokenizer")
        else:
            from tbps.models.irra_source import IRRASource
            IRRASource.from_dict(read_json(c["inputs"]["dataset_sources"]["path"])["irra"]).verify()
    ctx=load_inputs(c["inputs"],scope);ctx["c"]=c
    reference=FullReference(c,ctx,model)
    groups=caption_groups(ctx["plan"].manifest)
    # Smoke的攻击Caption集合来自完整annotation；评测Query仍固定10条。
    selected_ids={g.image_id for g in ctx["manifest"].gallery}
    attack_queries=tuple(q for q in ctx["plan"].manifest.queries if q.paired_image_id in selected_ids)
    ctx.update(reference=reference,official=official,groups=groups,
        attack_manifest=replace(ctx["manifest"],queries=attack_queries))
    need(len(ctx["manifest"].gallery)==(10 if scope=="smoke" else 3074)
         and len(ctx["manifest"].queries)==(10 if scope=="smoke" else 6156)
         and not torch.cuda.is_initialized(),"fixed CPU scope/no CUDA")
    print("SGA_CPU_PREFLIGHT_PASS",scope,model,json.dumps({
        "Gallery":len(ctx["manifest"].gallery),"Query":len(ctx["manifest"].queries),
        "attack_Caption_set":len(attack_queries),"official_commit":OFFICIAL_COMMIT,"model_loads":0,"CUDA":False}),flush=True)
    return ctx

class SGAPixels:
    """SGA为受害者专属迭代攻击，不能伪装为原G/H模型无关Plan；复用其像素校验与Store。"""
    def __init__(self,store,protocol_ref):
        self.store,self.protocol_ref=store,protocol_ref
    @staticmethod
    def key(image_id):return "image-"+digest(image_id)
    def claim(self,row,queries,seed):
        key="claim-"+digest(row.image_id)
        need(not (self.store.root/key).exists(),"orphan/consumed SGA Claim: preserve, no automatic forward retry")
        self.store.publish(key,{"claim.json":json_bytes({"row":asdict(row),"query_ids":[q.query_id for q in queries],"seed":seed})},
            {"kind":"SGA_generation_claim","image_id":row.image_id},sources=(self.protocol_ref,))
        return ref(self.store.root/key/"manifest.json")
    def publish(self,row,clean,adv,rounds,claim,queries):
        arrays={k:x.detach().cpu().clone() for k,x in
            (("original",clean),("adversarial",adv),("effective_delta",adv-clean))}
        stats=image_checks(*arrays.values(),(384,128));stream=io.BytesIO();torch.save(arrays,stream)
        md={"row":asdict(row),"pixels":stats,"rounds":rounds,"steps":len(rounds),
            "query_ids":[q.query_id for q in queries],"caption_sha256":[__import__("hashlib").sha256(q.caption.encode("utf-8")).hexdigest() for q in queries],
            "victim_specific_whitebox":True,"text_modified":False}
        self.store.publish(self.key(row.image_id),{"pixels.pt":stream.getvalue(),"metadata.json":json_bytes(md)},
            {"kind":"SGA_RGB","image_id":row.image_id},sources=(self.protocol_ref,claim))
        got,gotmd=self.load(row)
        need(all(torch.equal(got[k],arrays[k]) for k in arrays) and gotmd==json.loads(json_bytes(md)),"SGA bitwise readback")
        return md
    def load(self,row):
        key=self.key(row.image_id);self.store.verify(key,expected_provenance={"kind":"SGA_RGB","image_id":row.image_id})
        raw=torch.load(self.store.root/key/"pixels.pt",map_location="cpu",weights_only=True)
        x={k:v.detach().clone() for k,v in raw.items()};del raw
        md=read_json(self.store.root/key/"metadata.json")
        need(md["row"]==json.loads(json_bytes(asdict(row))) and
             md["pixels"]==image_checks(x["original"],x["adversarial"],x["effective_delta"],(384,128)),
             "SGA input/content/budget mismatch")
        return x,md

def generate(ctx,adapter,model,root,checkpoint,condition):
    store=ArtifactStore(root/"pixels",new_base=ctx["c"]["output_base"])
    p={"schema":"tbps-sga-pixels-1","model":model,"victim_checkpoint":asdict(checkpoint),
       "protocol":asdict(ref(root/"protocol/manifest.json")),"manifest":asdict(ctx["manifest"]),
       "attack_manifest":asdict(ctx["attack_manifest"]),"algorithm":ctx["c"]["objectives"][model],
       "official_commit":OFFICIAL_COMMIT,"whitebox":True,"text_modified":False}
    store.publish("protocol",{"protocol.json":json_bytes(p)},{"kind":"SGA_pixel_protocol"},sources=(ref(root/"protocol/manifest.json"),))
    pixels=SGAPixels(store,ref(store.root/"protocol/manifest.json"))
    position={q.query_id:i for i,q in enumerate(ctx["attack_manifest"].queries)}
    counts={"image_samples":0,"fusion_pairs":0,"input_gradient_scales":0};ordered=[];maximum=0.
    with BulkSession(ctx["store"]):
        for index,row in enumerate(ctx["manifest"].gallery):
            queries=ctx["groups"][row.image_id];seed=sample_seed(row.image_id,ctx["c"]["seed"])
            claim=pixels.claim(row,queries,seed)
            raw,_=ctx["store"].load(ctx["images"][row.image_id])
            clean=raw["original"].to("cuda:0");del raw
            supervision=payload_select(condition.batch.payload,[position[q.query_id] for q in queries])
            surrogate=SGASurrogate(adapter,model,supervision)
            start=time.monotonic()
            adv,rounds=attack_image(clean,surrogate,ctx["official"],seed=seed)
            md=pixels.publish(row,clean,adv,rounds,claim,queries);maximum=max(maximum,md["pixels"]["linf"])
            for k in counts:counts[k]+=surrogate.counts[k]
            ordered.append(row.image_id)
            print("SGA_IMAGE_COMPLETE",model,index+1,"/",len(ctx["manifest"].gallery),
                  json.dumps({"image_id":row.image_id,"Captions":len(queries),"seconds":time.monotonic()-start,
                   "max_linf":md["pixels"]["linf"],"first_objective":rounds[0]["objective"],
                   "last_objective":rounds[-1]["objective"],"min_gradient":min(r["gradient_l2"] for r in rounds)}),flush=True)
            del clean,adv,supervision,surrogate,rounds
    store.publish("collection",{"collection.json":json_bytes({"ordered_ids":ordered,"max_linf":maximum,
        "count":len(ordered),"whitebox_model":model,"text_modified":False,"counts":counts})},
        {"kind":"SGA_RGB_collection"},sources=(pixels.protocol_ref,))
    return pixels,counts

def encode_gallery(ctx,adapter,model,store,pixels,checkpoint,protocol):
    rows=ctx["manifest"].gallery
    hashes=tuple(pixels.load(g)[1]["pixels"]["adversarial_pixels_sha256"] for g in rows)
    binding=InputBinding(adapter.space.id,digest(asdict(ctx["manifest"])),hashes,"image",tuple(g.image_id for g in rows))
    if model=="irra":cache=FeatureCache(store,protocol,adapter.space,chunk=16)
    else:
        validate=__import__("tbps.models."+model,fromlist=["validate_payload"]).validate_payload
        cache=NativeFeatureCache(store,protocol,adapter.space,validate,chunk=16)
    for start in range(0,len(rows),16):
        parts=[]
        for row in rows[start:start+16]:
            x,_=pixels.load(row)
            if model=="irra":value=adapter.inference().encode_images(x["adversarial"].to("cuda:0")).payload.cpu()
            else:value=adapter.encode_images(x["adversarial"])
            parts.append(value);del x
        end=min(start+16,len(rows));cache.publish("adv_gallery",binding,start,end,payload_cat(parts));del parts
        print("SGA_EVAL_FEATURES",model,end,"/",len(rows),flush=True)
    origin=AttackOrigin("sga_tbps_whitebox_"+model,checkpoint,ref(pixels.store.root/"protocol/manifest.json"),BUDGET,digest(ctx["c"]["objectives"][model]))
    return cache.gather("adv_gallery",binding,origin=origin,originals=tuple(g.image_sha256 for g in rows))

def evaluate(ctx,adapter,model,store,inputs,mode):
    local=ArtifactStore(store.root/("ranking-"+mode),new_base=ctx["c"]["output_base"])
    pref=ref(store.root/"protocol/manifest.json")
    local.publish("protocol",{"protocol.json":json_bytes({"session":asdict(pref),"mode":mode})},{"kind":"SGA_ranking"},sources=(pref,))
    if model=="irra":
        scorer=CachedIRRA(adapter.space,ctx["checkpoint"],cache_protocol=ctx["reference"].protocol_ref)
        execution="readonly_cached"
    else:
        scorer=(ChunkCachedScorer if model=="rasa" else APTMSetScorer)(adapter,inputs,local,ref(local.root/"protocol/manifest.json"))
        execution="materialized_native"
    engine=EvaluationEngine(ctx["manifest"],scorer,direction="higher_better",execution=execution,model_sources=(ctx["checkpoint"],))
    p,result,scores=engine.run(inputs,modes=(mode,),query_chunk=128)
    value=four(result["groups"][mode]);stream=io.BytesIO();torch.save(scores[mode].detach().clone(),stream)
    local.publish("result",{"protocol.json":json_bytes(p),"metrics.json":json_bytes(value),"score.pt":stream.getvalue()},
        {"kind":"SGA_four_metrics"},sources=(ref(local.root/"protocol/manifest.json"),))
    return value

def table(groups,rounded=False):
    buf=io.StringIO();writer=csv.writer(buf,lineterminator="\n");writer.writerow(("Attack Method",*KEYS))
    for method in METHODS:writer.writerow((method,*(format(groups[method][k],".2f") if rounded else groups[method][k] for k in KEYS)))
    return buf.getvalue().encode()

def run(path,scope,model):
    ctx=preflight(path,scope,model);c=ctx["c"];root=Path(c["output_base"])/scope/model
    need(not root.exists(),"existing SGA Attempt; no overwrite or automatic retry")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit GPU0")
    gpu=gpu_available({"physical_gpu":0})
    print("SGA_GPU_RESOURCE",json.dumps(gpu),flush=True)
    need(gpu["free_mib"]>=c["minimum_free_mib"],"insufficient GPU memory for SGA gradient Smoke; no waiting/retry")
    store=ArtifactStore(root,new_base=c["output_base"])
    store.publish("attempt",{"preflight.json":json_bytes(ctx["report"]),"gpu.json":json_bytes(gpu)},{"kind":"SGA_attempt"},sources=(ref(path),))
    report={"status":"RUNNING","scope":scope,"model":model,"victim_loads":0,"optimizer":0,"G_H_loads":0,"G_H_forwards":0,"JVP":0}
    counts={};start=time.monotonic()
    try:
        # CPU常量读取先于CUDA；严格读原文本，不使用历史H对抗文本。
        condition=ctx["reference"].read("clean_text",{**ctx,"manifest":ctx["attack_manifest"]})
        cleantext=condition if scope=="full" else ctx["reference"].read("clean_text",ctx)
        torch.set_num_threads(2);torch.manual_seed(c["seed"])
        ctx["gate"].initialize_cuda()
        with fp32_runtime(),load_model(c,model,scope) as (adapter,backend,checkpoint,counts):
            ctx["checkpoint"]=checkpoint;report["victim_loads"]=1;before=model_snapshot(backend);report["model_before"]=before
            need(adapter.space==ctx["reference"].space,"same native historical ModelSpace required")
            protocol={"schema":"tbps-sga-native-1","config":asdict(ref(path)),"manifest":asdict(ctx["manifest"]),
              "model_space":asdict(adapter.space),"whitebox_target":model,"objective":c["objectives"][model],
              "text_modified":False,"phase_order":["generation","native_evaluation"],"budget":asdict(BUDGET)}
            protocol=json.loads(json_bytes(protocol))
            store.publish("protocol",{"protocol.json":json_bytes(protocol)},{"kind":"SGA_native_session"},sources=(ref(path),checkpoint,ctx["reference"].protocol_ref))
            pixels,attack_counts=generate(ctx,adapter,model,root,checkpoint,condition)
            del condition
            adv=encode_gallery(ctx,adapter,model,store,pixels,checkpoint,protocol)
            attack=evaluate(ctx,adapter,model,store,{"clean_text":cleantext,"adv_gallery":adv},"Image-only");del adv
            if scope=="full":clean=four(ctx["reference"].summary["groups"]["Clean"])
            else:clean=evaluate(ctx,adapter,model,store,{"clean_text":cleantext,"clean_gallery":ctx["reference"].read("clean_gallery",ctx)},"Clean")
            after=model_snapshot(backend)
            need(after==before and all(p.grad is None and not p.requires_grad for p in backend.parameters()),"victim freeze changed")
            actual=dict(counts if model=="irra" else adapter.neural_counts)
            expected_images=attack_counts["image_samples"]+len(ctx["manifest"].gallery)
            need(actual["image_samples"]==expected_images and actual["text_samples"]==0,"actual image/text hook counts")
            if model!="irra":need(actual["fusion_pairs"]==attack_counts["fusion_pairs"]+adapter.counts["fusion_pairs"],"actual native fusion counts")
            report.update(status="PASS",model_before=before,model_after=after,model_frozen=True,
                actual_neural_counts=actual,attack_counts=attack_counts,
                input_gradient_calls=attack_counts["input_gradient_scales"]+STEPS*len(ctx["manifest"].gallery),
                clean_origin="historical audited Full" if scope=="full" else "new subset ranking of historical Clean features",
                GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            groups=dict(zip(METHODS,(clean,attack)))
            store.publish("completed",{"summary.json":json_bytes({"schema":"tbps-sga-four-metrics-1","groups":groups,
                "Gallery":len(ctx["manifest"].gallery),"Query":len(ctx["manifest"].queries),"metric_unit":"percent",
                "checkpoint":asdict(checkpoint),"whitebox_target":model,"text_modified":False,"SGA_TBPS_adaptation":True}),
                "metrics.csv":table(groups),"metrics_2dp.csv":table(groups,True)},{"kind":"SGA_complete"},sources=(ref(root/"protocol/manifest.json"),))
    except BaseException as exc:
        report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    finally:
        if "adapter" in locals():
            report.setdefault("actual_neural_counts",dict(counts if model=="irra" else adapter.neural_counts))
        report["elapsed_seconds"]=time.monotonic()-start
        from tbps.engines.replay_control import rss_bytes
        report["peak_RSS_bytes"]=rss_bytes()
        if torch.cuda.is_initialized():
            report["GPU_peak_allocated_bytes"]=torch.cuda.max_memory_allocated();report["GPU_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        store.publish("execution",{"report.json":json_bytes(report)},{"kind":"SGA_execution"},sources=(ref(root/"attempt/manifest.json"),))
    print("SGA_TBPS_EVALUATION_"+report["status"],scope,model,flush=True)
    return int(report["status"]!="PASS")

def audit(path,scope,model):
    c=configuration(path);root=Path(c["output_base"])/scope/model
    store=ArtifactStore(root,new_base=c["output_base"]);need(not store.pending(),"SGA pending publication")
    for p in root.rglob("manifest.json"):
        ArtifactStore(p.parent.parent,new_base=c["output_base"]).verify(p.parent.name)
    s=read_json(root/"completed/summary.json");r=read_json(root/"execution/report.json")
    need(r["status"]=="PASS" and r["model_frozen"] and r["model_before"]==r["model_after"] and r["victim_loads"]==1,"successful frozen SGA")
    need(s["whitebox_target"]==model and s["text_modified"] is False and
         s["Gallery"]==(10 if scope=="smoke" else 3074) and s["Query"]==(10 if scope=="smoke" else 6156),"SGA scope")
    pixels=SGAPixels(ArtifactStore(root/"pixels",new_base=c["output_base"]),ref(root/"pixels/protocol/manifest.json"))
    p=read_json(root/"protocol/protocol.json");collection=read_json(root/"pixels/collection/collection.json")
    from tbps.contracts import GalleryItem,PersonID
    # 原像素绑定用生产契约对象；无原生模型、无排名重算。
    for row in p["manifest"]["gallery"]:
        item=GalleryItem(**{**row,"pid":PersonID(**row["pid"])})
        x,md=pixels.load(item)
        need(md["steps"]==STEPS and len(md["rounds"])==STEPS and all(a["gradient_l2"]>0 for a in md["rounds"]),"SGA rounds/nonzero gradient")
        del x
    need(collection["ordered_ids"]==[g["image_id"] for g in p["manifest"]["gallery"]],"SGA Gallery order")
    ordered={k:s["groups"][k] for k in METHODS}
    need((root/"completed/metrics.csv").read_bytes()==table(ordered) and (root/"completed/metrics_2dp.csv").read_bytes()==table(ordered,True),"four metric CSV/summary")
    print("SGA_TBPS_CPU_AUDIT_PASS",scope,model,flush=True);return 0

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True);p.add_argument("--action",choices=("preflight","run","audit"),default="preflight")
    p.add_argument("--scope",choices=("smoke","full"),default="smoke");p.add_argument("--model",choices=("irra","rasa","aptm"),required=True)
    p.add_argument("--allow-smoke",action="store_true");p.add_argument("--allow-full",action="store_true")
    a=p.parse_args(argv)
    if a.action=="preflight":
        need(not a.allow_smoke and not a.allow_full,"CPU preflight no GPU flags");preflight(a.config,a.scope,a.model);return 0
    if a.action=="audit":
        need(not a.allow_smoke and not a.allow_full,"CPU audit no GPU flags");return audit(a.config,a.scope,a.model)
    need(a.allow_smoke!=a.allow_full and (a.allow_smoke if a.scope=="smoke" else a.allow_full),"explicit scope-specific GPU permission")
    return run(a.config,a.scope,a.model)
