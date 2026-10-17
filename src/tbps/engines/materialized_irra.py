"""已封存Stage4D产物的官方IRRA评估；默认CPU预检，真实编码须用户授权。"""
from dataclasses import asdict,replace
from pathlib import Path
from contextlib import ExitStack
import argparse,csv,io,json,os,time,traceback
from unittest.mock import patch
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,json_bytes,digest,sha_file
from tbps.materialization.contracts import need
from tbps.materialization.full_prepare import prepare,FULL,BASE as MATERIAL_BASE
from tbps.materialization.bulk_session import StreamingMaterializedStore,BulkSession
from tbps.materialization.retained_inputs import unit_ref
from tbps.materialization.full_runner import ValidatedActions
from tbps.materialization.irra_startup import CPUStartupGate
from tbps.engines.historical_loader import native_action_validator,feature_only_cpu
from tbps.runtime_closure import cpu_only
from tbps.models.irra_source import IRRASource
from tbps.models.irra import IRRAAdapter
from tbps.models.base import ModelSpace,InputBinding
from tbps.engines.evaluation import EvaluationEngine,MODES,publish_result,load_result
from tbps.engines.provenance import AttackOrigin
from tbps.smoke_irra import model_snapshot,gpu_available
if __package__:
    from .materialized_cache import FeatureCache,CachedIRRA,FAMILIES,ref
else:
    from materialized_cache import FeatureCache,CachedIRRA,FAMILIES,ref

FROZEN="/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017"
PLAN="7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5"

def config(path):
    c=read_json(path,"tbps-materialized-irra-evaluation-1")
    fixed={"gallery":3074,"query":6156,"plan_id":PLAN,"precision":"fp32/math/no_tf32",
           "seed":42,"image_batch":1,"text_batch":1,"cache_chunk":64,"query_chunk":128,
           "physical_gpu":0,"minimum_free_mib":8192,"maximum_utilization_percent":10,"clean_metric_atol_pp":.001}
    need(set(c)==set(fixed)|{"schema","materialization_config","generation_result","materialization_audit",
         "dataset_sources","output_base","entry_sources","legacy_reference"} and all(c[k]==v for k,v in fixed.items()),
         "fixed official Test evaluation configuration")
    for key in ("materialization_config","generation_result","materialization_audit","dataset_sources"):
        checked(FileRef(**c[key]))
    need(c["materialization_config"]["path"]==FROZEN+"/configs/framework/stage4d1_materialization.server.json",
         "original frozen Plan producer config")
    need(c["generation_result"]["path"]=="/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3/result/manifest.json"
         and c["materialization_audit"]["path"]=="/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3_cpu_audit_v1/result/manifest.json",
         "only sealed generation_v3 and its independent audit")
    expected={str(Path(__file__).resolve()),str(Path(__file__).with_name("materialized_cache.py").resolve())}
    need(set(c["entry_sources"])==expected,"explicit current entry sources")
    for p,h in c["entry_sources"].items():checked(FileRef(p,h))
    need(c["output_base"]=="/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation",
         "new isolated evaluation output")
    need(set(c["legacy_reference"])=={"summary","protocol","completed"},"legacy Clean reference evidence")
    for value in c["legacy_reference"].values():checked(FileRef(**value))
    return c

def verify_clean_reference(c,manifest,source):
    refs=c["legacy_reference"]
    p=read_json(refs["protocol"]["path"]);s=read_json(refs["summary"]["path"]);done=read_json(refs["completed"]["path"])
    need(done["summary_sha256"]==refs["summary"]["sha256"] and done["protocol_sha256"]==digest(p)
         and done["method_audit_passed"] is True and s["protocol_sha256"]==digest(p)
         and p["model_sha256"]["IRRA"]==source.checkpoint.sha256,"historical official Clean source")
    from tbps.datasets.cuhk_pedes import CuhkPedes
    ds=read_json(c["dataset_sources"]["path"])
    CuhkPedes(FileRef(**ds["annotation"]),ds["image_root"]).match_test_protocol(manifest,p)
    return s["groups"]["Clean"]

def clean_check(actual,expected,tolerance=.001):
    errors={k:abs(actual[k]-expected[k]) for k in expected}
    need(set(actual)==set(expected) and all(v<=tolerance for v in errors.values()),
         "official Clean protocol mismatch: "+str(errors))
    return errors

def frozen_library():
    import subprocess
    import tbps.materialization.full_prepare as original
    import tbps.engines.evaluation as engine
    import tbps.models.irra as model
    for m in (original,engine,model):
        need(Path(m.__file__).resolve().is_relative_to(Path(FROZEN)/"src"),"production uses frozen Stage4D1 library")
    need(subprocess.check_output(["git","rev-parse","HEAD"],cwd=FROZEN,text=True).strip()==
         "82fe88a5a74955838fc6341c1ed9a46921c94d7c" and
         not subprocess.check_output(["git","status","--porcelain"],cwd=FROZEN,text=True),
         "frozen library identity/clean")

def collection(store):
    """只读解析原封存指针，仍由原ArtifactStore校验全部声明SHA。"""
    store.store.verify("joint-full")
    body=read_json(store.store.root/"joint-full/joint_collection.json")
    need(body["status"]=="JOINT_FULL_MATERIALIZED" and body["plan_id"]==store.plan.id
         and body["victim_model"]is None,"sealed model-independent Joint")
    pointer=FileRef(**body["collection"]);checked(pointer)
    need(Path(pointer.path).is_relative_to(store.store.root),"collection reference escape")
    store.store.verify(Path(pointer.path).parent.name)
    index=read_json(pointer.path)
    need(index["mode"]=="joint" and index["new_G_H_forwards"]==0,"composition did not generate attacks")
    pointer=FileRef(**index["collection"]);checked(pointer)
    need(Path(pointer.path).is_relative_to(store.store.root),"collection core escape")
    store.store.verify(Path(pointer.path).parent.name)
    core=read_json(pointer.path)
    need(core["plan_id"]==store.plan.id and core["mode"]=="joint" and core["victim_model"]is None,
         "Joint collection core identity")
    images,texts=tuple(unit_ref(v) for v in core["images"]),tuple(unit_ref(v) for v in core["texts"])
    need(tuple(u.binding.sample_id for u in images)==tuple(g.image_id for g in store.plan.manifest.gallery),
         "complete official Gallery order; never zip to Query")
    need(tuple(u.binding.sample_id for u in texts)==tuple(q.query_id for q in store.plan.manifest.queries),
         "complete official Query order including NOOP/multiple Captions")
    for u in (*images,*texts):
        need(u.binding==store.plan.binding(u.binding.sample_id,u.modality) and
             u.plan_id==store.plan.id and u.unit_id==store.unit_id(u.binding),"collection input identity")
    return images,texts,pointer

def select(manifest,scope):
    need(scope in ("full","smoke"),"explicit full/smoke scope")
    if scope=="full":return manifest
    gallery=manifest.gallery[:10];ids={g.image_id for g in gallery}
    queries=tuple(q for q in manifest.queries if q.paired_image_id in ids)[:10]
    need(len(gallery)==len(queries)==10,"fixed positive-covered 10+10 smoke")
    return replace(manifest,gallery=gallery,queries=queries)

def preflight(path,scope):
    """完整顺序/来源/Unicode检查；首尾RGB抽查，所有RGB在编码前原生读回验证。"""
    import torch
    gate=CPUStartupGate()
    with cpu_only():
        c=config(path);frozen_library()
        for key in ("generation_result","materialization_audit"):
            p=Path(c[key]["path"]).parent
            ArtifactStore(p.parent,new_base=p.parent.parent).verify(p.name)
        done=read_json(Path(c["generation_result"]["path"]).with_name("report.json"))
        audit=read_json(Path(c["materialization_audit"]["path"]).with_name("report.json"))
        need(done["status"]=="JOINT_FULL_MATERIALIZED" and audit["status"]=="JOINT_FULL_CPU_AUDIT_PASS"
             and done["plan_id"]==audit["plan_id"]==PLAN and audit["generated"]==3074
             and audit["text"]["units"]==6156 and not audit["pending"],"sealed and CPU audited generation_v3")
        _,plan,legacy,_,_=prepare(c["materialization_config"]["path"])
        need(plan.id==PLAN and (len(plan.manifest.gallery),len(plan.manifest.queries))==(3074,6156),"original full Plan")
        from tbps.datasets.cuhk_pedes import CuhkPedes
        ds=read_json(c["dataset_sources"]["path"])
        dataset=CuhkPedes(FileRef(**ds["annotation"]),ds["image_root"])
        need(dataset.manifest("test",expected_counts=(3074,6156))==plan.manifest,"official ordered manifest differs")
        source=IRRASource.from_dict(ds["irra"]);source.verify()
        expected_clean=verify_clean_reference(c,plan.manifest,source)
        store=StreamingMaterializedStore(ArtifactStore(Path(FULL)/"artifacts",new_base=MATERIAL_BASE),plan)
        need(not store.store.pending(),"pending materialized data")
        texts_by_id={};statistics={"edited":0,"NOOP":0,"edits":0}
        with BulkSession(store),native_action_validator(legacy.config_ref) as native:
            images,texts,core=collection(store)
            validate=ValidatedActions(native)
            for u in texts:
                adv,md=store.load(u,validate=validate)
                original=md["original_caption"];texts_by_id[u.binding.sample_id]=(original,adv)
                statistics["NOOP" if original==adv else "edited"]+=1
                statistics["edits"]+=len(md["action"]["edits"])
            need(statistics=={"edited":6116,"NOOP":40,"edits":13537},"all historical H actions retained")
        manifest=select(plan.manifest,scope)
    samples=(images[0],images[-1])
    with feature_only_cpu({u.data.path for u in samples}):
        for u in samples:
            arrays,_=store.load(u);del arrays
    gate.checked();gate.exit_checked()
    need(not torch.cuda.is_initialized(),"preflight must remain CPU only")
    img={u.binding.sample_id:u for u in images};txt={u.binding.sample_id:u for u in texts}
    report={"status":"MATERIALIZED_IRRA_CPU_PREFLIGHT_PASS","plan_id":PLAN,"scope":scope,
            "gallery":len(manifest.gallery),"query":len(manifest.queries),"text_statistics":statistics,
            "ordered_manifest_sha256":digest(asdict(manifest)),"sampled_RGB_verified":2,
            "RGB_validation_before_each_encoding":True,"CUDA_initialized":False,"G_H_IRRA_loads":0}
    return dict(c=c,plan=plan,legacy=legacy,store=store,manifest=manifest,source=source,gate=gate,
                images=img,text_units=txt,texts=texts_by_id,core=core,report=report,expected_clean=expected_clean)

def bindings(ctx,space):
    m=ctx["manifest"];mh=digest(asdict(m))
    image_ids=tuple(g.image_id for g in m.gallery);query_ids=tuple(q.query_id for q in m.queries)
    originals={"image":tuple(g.image_sha256 for g in m.gallery),
               "text":tuple(hashlib_sha(q.caption) for q in m.queries)}
    return {
        "clean_gallery":InputBinding(space.id,mh,originals["image"],"image",image_ids),
        "adv_gallery":InputBinding(space.id,mh,tuple(ctx["images"][i].content_sha256 for i in image_ids),"image",image_ids),
        "clean_text":InputBinding(space.id,mh,originals["text"],"text",query_ids),
        "adv_text":InputBinding(space.id,mh,tuple(ctx["text_units"][i].content_sha256 for i in query_ids),"text",query_ids)
    },originals

def hashlib_sha(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def encoded_batch(ctx,view,family,ids):
    import torch
    rows=[]
    for key in ids:
        if family.endswith("gallery"):
            arrays,_=ctx["store"].load(ctx["images"][key])
            x=arrays["original" if family=="clean_gallery" else "adversarial"]
            value=view.encode_images(x.to("cuda:0")).payload
            del arrays,x
        else:
            pair=ctx["texts"][key]
            value,_=view.encode_texts((pair[0 if family=="clean_text" else 1],))
            value=value.payload
        rows.append(value.detach().cpu().clone())
    return torch.cat(rows).detach().clone()

def finish(cache,ctx,space,checkpoint):
    bound,originals=bindings(ctx,space);inputs={}
    for family,binding in bound.items():
        kind="image" if family.endswith("gallery") else "text";origin=None
        if family.startswith("adv"):
            p=ctx["plan"].g if kind=="image" else ctx["plan"].h
            origin=AttackOrigin(p.attack_id,p.checkpoint,ctx["core"],
                __import__("tbps.attacks.gv2" if kind=="image" else "tbps.attacks.hv1",fromlist=["BUDGET"]).BUDGET,
                p.sources_sha256)
        inputs[family]=cache.gather(family,binding,origin=origin,
                                   originals=originals[kind] if origin else ())
    import torch
    noop=[i for i,(a,b) in enumerate(zip(bound["clean_text"].ordered_sha256,bound["adv_text"].ordered_sha256)) if a==b]
    need(torch.equal(inputs["clean_text"].batch.payload[noop],inputs["adv_text"].batch.payload[noop]),
         "NOOP native features must equal clean exactly")
    scorer=CachedIRRA(space,checkpoint,cache_protocol=cache.protocol_ref)
    engine=EvaluationEngine(ctx["manifest"],scorer,direction="higher_better",execution="readonly_cached",
                            model_sources=(checkpoint,))
    if (cache.store.root/"result").exists():
        prior=read_json(cache.store.root/"result/protocol.json")
        need(prior["manifest_sha256"]==digest(asdict(ctx["manifest"])) and
             prior["model_space"]==asdict(space) and
             prior["inputs"]==json.loads(json_bytes({k:v.descriptor() for k,v in inputs.items()})),
             "prior result input/model protocol mismatch")
        _,report,scores=load_result(cache.store,"result",expected_protocol_sha256=digest(prior))
        if ctx["report"]["scope"]=="full":
            clean_check(report["groups"]["Clean"],ctx["expected_clean"],ctx["c"]["clean_metric_atol_pp"])
        del scores
        return report,0
    protocol,report,scores=engine.run(inputs,query_chunk=128)
    if ctx["report"]["scope"]=="full":
        clean_check(report["groups"]["Clean"],ctx["expected_clean"],ctx["c"]["clean_metric_atol_pp"])
    publish_result(cache.store,"result",protocol,report,scores,sources=(cache.protocol_ref,))
    _,verified,loaded=load_result(cache.store,"result",expected_protocol_sha256=report["protocol_sha256"])
    need(verified==report,"published result mismatch")
    del scores,loaded
    return report,scorer.score_calls

def csv_bytes(report):
    out=io.StringIO(newline="");w=csv.writer(out)
    keys=("R@1","R@5","R@10","mAP","mINP")
    w.writerow(["Group",*keys,*["DR_"+k+"_percent" for k in keys]])
    for mode in MODES:
        v=report["groups"][mode]
        w.writerow([mode,*[v[k] for k in keys],*[report["drop_rate_percent"][mode][k] for k in keys]])
    return out.getvalue().encode()

def run(path,scope,*,authorized=False,resume=False):
    need(authorized is True,"explicit --allow-full-evaluation/--allow-smoke required before any model load")
    import torch
    ctx=preflight(path,scope);c=ctx["c"];root=Path(c["output_base"])/scope
    outputs=ArtifactStore(root,new_base=c["output_base"])
    if (root/"completed").exists():
        audit_outputs(path,scope);print("MATERIALIZED_IRRA_ALREADY_COMPLETE",flush=True);return 0
    need(resume or not root.exists(),"existing evaluation requires explicit --resume")
    need(not outputs.pending(),"uncommitted feature publication; preserve and inspect")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="0","explicit physical GPU0 required")
    gpu=gpu_available(c)
    previous=list(root.glob("attempt-*"))
    for p in previous:need((root/("execution-"+p.name.split("-")[-1])).exists(),"unclosed previous Attempt")
    number=len(previous)+1;attempt="attempt-"+str(number).zfill(6);execution="execution-"+str(number).zfill(6)
    outputs.publish(attempt,{"preflight.json":json_bytes(ctx["report"])},
        {"kind":"explicit_user_IRRA_evaluation","scope":scope},
        sources=(ref(path),FileRef(**c["generation_result"]),FileRef(**c["materialization_audit"]),
                 *(FileRef(p,h) for p,h in c["entry_sources"].items())))
    counts={"IRRA_load_attempts":0,"IRRA_loads":0,"IRRA_image_started":0,"IRRA_image_completed":0,
            "IRRA_text_started":0,"IRRA_text_completed":0,"G_loads":0,"H_loads":0,"G_forwards":0,"H_forwards":0}
    started=time.monotonic();execution_report={"status":"RUNNING","GPU_before":gpu,"counts":counts}
    try:
        torch.set_num_threads(2);torch.manual_seed(42)
        ctx["gate"].initialize_cuda()
        counts["IRRA_load_attempts"]=1
        adapter=IRRAAdapter.from_source(ctx["source"],device="cuda:0",allow_model_load=True);counts["IRRA_loads"]=1
        space=adapter.space;before=model_snapshot(adapter._backend)
        bound,_=bindings(ctx,space)
        protocol={"schema":"tbps-materialized-irra-cache-1","config":asdict(ref(path)),
            "scope":scope,"plan_id":PLAN,"collection":asdict(ctx["core"]),"model_space":asdict(space),
            "manifest":asdict(ctx["manifest"]),"inputs":{k:asdict(v) for k,v in bound.items()},
            "G_H_generation":False,"ranking_feedback_to_attack":False}
        if (root/"protocol").exists():
            outputs.verify("protocol");need(read_json(root/"protocol/protocol.json")==json.loads(json_bytes(protocol)),"resume protocol/model/inputs mismatch")
        else:
            outputs.publish("protocol",{"protocol.json":json_bytes(protocol)},{"kind":"IRRA_materialized_cache"},
                sources=(ref(path),ctx["core"],ctx["source"].checkpoint))
        cache=FeatureCache(outputs,protocol,space,chunk=c["cache_chunk"])
        def observed(kind,fn):
            def call(x):
                counts["IRRA_"+kind+"_started"]+=len(x)
                value=fn(x);counts["IRRA_"+kind+"_completed"]+=len(x);return value
            return call
        expected_new={"image":0,"text":0}
        with ExitStack() as stack:
            for name,kind in (("encode_image","image"),("encode_text","text")):
                stack.enter_context(patch.object(adapter._backend,name,new=observed(kind,getattr(adapter._backend,name))))
            stack.enter_context(BulkSession(ctx["store"]))
            for family,binding in bound.items():
                n=len(binding.ordered_ids)
                for start in range(0,n,cache.chunk):
                    end=min(start+cache.chunk,n);key=cache.key(family,start)
                    if (root/key).exists():cache.load(family,binding,start,end);status="REUSED"
                    else:
                        x=encoded_batch(ctx,adapter.inference(),family,binding.ordered_ids[start:end])
                        cache.publish(family,binding,start,end,x,stats={"IRRA_samples":end-start});del x;status="ENCODED"
                        expected_new["image" if family.endswith("gallery") else "text"]+=end-start
                    print("IRRA_FEATURE_PROGRESS",family,end,"/",n,status,flush=True)
        need(all(counts["IRRA_"+kind+"_started"]==counts["IRRA_"+kind+"_completed"]==expected_new[kind]
                 for kind in ("image","text")),"actual IRRA sample count / committed feature mismatch")
        need(model_snapshot(adapter._backend)==before,"IRRA parameters/buffers changed")
        execution_report["model_before"]=execution_report["model_after"]=before
        execution_report["gpu_peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
        execution_report["gpu_peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
        del adapter
        import gc
        gc.collect();torch.cuda.empty_cache()
        with BulkSession(ctx["store"]):collection(ctx["store"])
        report,calls=finish(cache,ctx,space,ctx["source"].checkpoint)
        outputs.publish("completed",{"summary.json":json_bytes(report),"metrics.csv":csv_bytes(report)},
            {"kind":"IRRA_materialized_evaluation_complete","scope":scope},
            sources=(ref(root/"result/manifest.json"),cache.protocol_ref))
        execution_report.update(status="PASS",score_calls=calls,IRRA_frozen=True,backward=0,optimizer=0,JVP=0)
    except BaseException as exc:
        execution_report.update(status="FAIL",error=str(exc),traceback=traceback.format_exc())
    finally:
        execution_report["elapsed_seconds"]=time.monotonic()-started
        from tbps.engines.replay_control import rss_bytes
        execution_report["peak_RSS_bytes"]=rss_bytes()
        outputs.publish(execution,{"report.json":json_bytes(execution_report)},{"kind":"IRRA_evaluation_execution"},
            sources=(ref(root/attempt/"manifest.json"),ref(path)))
    print("MATERIALIZED_IRRA_EVALUATION_"+execution_report["status"],scope,flush=True)
    return int(execution_report["status"]!="PASS")

def audit_outputs(path,scope):
    """CPU只读SHA/完成一致性，不重新排名、不执行编码。"""
    c=config(path);root=Path(c["output_base"])/scope
    store=ArtifactStore(root,new_base=c["output_base"]);need(not store.pending(),"pending evaluation")
    store.verify("completed");store.verify("protocol")
    summary=read_json(root/"completed/summary.json")
    _,report,scores=load_result(store,"result",expected_protocol_sha256=summary["protocol_sha256"])
    need(report==summary and (root/"completed/metrics.csv").read_bytes()==csv_bytes(report),"summary/CSV/result conflict")
    del scores
    need(report["gallery_count"]==(3074 if scope=="full" else 10) and
         report["query_count"]==(6156 if scope=="full" else 10),"completed member counts")
    # 核验特征文件本身，而非只验证其manifest SHA；不重算排名。
    cached_protocol=read_json(root/"protocol/protocol.json")
    space=ModelSpace(**cached_protocol["model_space"])
    cache=FeatureCache(store,cached_protocol,space,chunk=c["cache_chunk"])
    import torch
    from tbps.engines.provenance import payload_digest
    for family,d in cached_protocol["inputs"].items():
        binding=InputBinding(d["model_space_id"],d["dataset_manifest_sha256"],
            tuple(d["ordered_sha256"]),d["modality"],tuple(d["ordered_ids"]))
        x=torch.cat([cache.load(family,binding,start,min(start+cache.chunk,len(binding.ordered_ids)))
                     for start in range(0,len(binding.ordered_ids),cache.chunk)])
        need(payload_digest(x)==read_json(root/"result/protocol.json")["inputs"][family]["payload_sha256"],
             "cache / evaluated feature content differs")
        del x
    attempts=sorted(root.glob("attempt-*"));need(attempts,"missing execution evidence")
    for p in attempts:
        name="execution-"+p.name.split("-")[-1];store.verify(p.name);store.verify(name)
    last=read_json(root/("execution-"+attempts[-1].name.split("-")[-1])/"report.json")
    need(last["status"]=="PASS" and last["IRRA_frozen"] is True
         and last["model_before"]==last["model_after"] and
         all(last[k]==0 for k in ("backward","optimizer","JVP")) and
         all(last["counts"][k]==0 for k in ("G_loads","H_loads","G_forwards","H_forwards")),
         "missing final successful frozen inference execution")
    print("MATERIALIZED_IRRA_CPU_AUDIT_PASS",scope,flush=True)
    return report

def main(argv=None):
    p=argparse.ArgumentParser(description="sealed RGB/Unicode -> official IRRA; default CPU only")
    p.add_argument("--config",required=True);p.add_argument("--action",choices=("preflight","run","audit"),default="preflight")
    p.add_argument("--scope",choices=("full","smoke"),default="full")
    p.add_argument("--allow-full-evaluation",action="store_true");p.add_argument("--allow-smoke",action="store_true")
    p.add_argument("--resume",action="store_true");a=p.parse_args(argv)
    if a.action=="preflight":
        need(not(a.allow_full_evaluation or a.allow_smoke or a.resume),"CPU preflight has no execution flags")
        ctx=preflight(a.config,a.scope);print(json.dumps(ctx["report"],indent=2));return 0
    if a.action=="audit":
        need(not(a.allow_full_evaluation or a.allow_smoke or a.resume),"read-only audit has no GPU flags")
        ctx=preflight(a.config,a.scope);c=ctx["c"]
        root=Path(c["output_base"])/a.scope
        allowed={str(x) for pattern in ("result/scores/*.pt","*-*/features.pt") for x in root.glob(pattern)}
        with feature_only_cpu(allowed):audit_outputs(a.config,a.scope)
        return 0
    need(a.allow_full_evaluation!=a.allow_smoke and
         (a.allow_full_evaluation if a.scope=="full" else a.allow_smoke),"scope-specific explicit authorization")
    return run(a.config,a.scope,authorized=True,resume=a.resume)
if __name__=="__main__":raise SystemExit(main())
