"""CPU只读物化来源预检；没有真实生成、模型加载或排名选项。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import ast
import json
from ..artifacts import ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes,digest
from ..attacks.provenance import LegacyAttackSources
from ..engines.historical_sources import JointFullSources
from ..engines.historical_loader import native_action_validator
from ..engines.replay_control import historical_snapshot,unchanged
from ..runtime_closure import cpu_only
from .contracts import MaterializationPlan,need
from .legacy_text import HistoricalTextSource

def audit_g_call_path(sources,legacy):
    g=legacy.root/"prototypes/attribute_image_generator_v2_20261014"
    files=[g/"inference_once.py",g/"evaluation_phase29a/eval_backend.py",g/"image_only_generator.py",g/"budget_projection.py"]
    pins={**sources.protocol["source_sha256"],**legacy.config["runtime_sources"]}
    for p in files:
        need(str(p) in pins,"G runtime audit source not pinned")
        checked(FileRef(str(p),pins[str(p)]))
    tree=ast.parse(files[0].read_text())
    fn=next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=="generate_once")
    need([x.arg for x in fn.args.args]==["generator","image"],"original image-only signature changed")
    calls=[x for x in ast.walk(fn) if isinstance(x,ast.Call)]
    need(any(isinstance(c.func,ast.Name) and c.func.id=="generator" and
             len(c.args)==1 and isinstance(c.args[0],ast.Name) and c.args[0].id=="image" for c in calls),"original G(I) call missing")
    need(not any(isinstance(c.func,ast.Name) and c.func.id in ("open","load","fresh_score","prepare_condition") for c in calls),
         "unexpected G inference preparation/condition access")
    backend=ast.parse(files[1].read_text())
    real=next(x for x in backend.body if isinstance(x,ast.ClassDef) and x.name=="RealBackend")
    image=next(x for x in real.body if isinstance(x,ast.FunctionDef) and x.name=="image")
    need(any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=="generate_once"
             for x in ast.walk(image)),"official G2 Test generation route changed")
    return files

def audit(config):
    sources=JointFullSources(config)
    legacy=LegacyAttackSources(FileRef(**sources.config["attack_sources"]).path)
    protected=historical_snapshot("/home/lzf/ldx/tmp/tbps-framework-stage2/protected_before.json")
    call_sources=audit_g_call_path(sources,legacy)
    manifest_ref=legacy.ref("gv2","cache_manifest");checked(manifest_ref)
    cache=read_json(manifest_ref.path)
    need(cache["counts"]=={"train":2048,"validation":128} and len(cache["entries"])==2176,"original training cache counts")
    need(all(e["split"]=="train" and e["role"] in ("train","validation") for e in cache["entries"]),"unexpected official Test supervision")
    test_ids={g.image_id for g in sources.manifest.gallery}
    overlap=test_ids&{e["row_id"] for e in cache["entries"]}
    need(not overlap,"training cache overlaps official Test")
    rows=sources.protocol["gallery"]
    ready=0
    for row in rows:
        p=(Path(sources.base["image_root"])/row["image"]).resolve()
        checked(FileRef(str(p),row["image_sha256"]));ready+=1
    # G-v1确有完整条件，但它不是G-v2输入，也不读取其张量内容。
    v1=Path("/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/outputs/test_full")
    old_m=FileRef(str(v1/"conditions/manifest.json"),read_json(v1/"prepare_completed.json")["manifest_sha256"]);checked(old_m)
    old=read_json(old_m.path);need(len(old["entries"])==len(rows),"G1 historical metadata count")
    v1refs=[old_m]
    for e,r in zip(old["entries"],rows):
        need((e["row_id"],e["image"],e["image_sha256"],e["pid"])==(r["row_id"],r["image"],r["image_sha256"],r["pid"]),
             "G1 conditions actual Test member identity")
        for field in ("tensor","metadata"):
            ref=FileRef(e[field+"_path"],e[field+"_sha256"]);checked(ref);v1refs.append(ref)
    provenance=legacy.provenance("hv1")
    with native_action_validator(FileRef(**sources.config["attack_sources"])) as validate:
        text=HistoricalTextSource(sources,provenance,validate)
        stats={"edited_queries":0,"NOOP_queries":0,"eligible_queries":0,"edits":0}
        content=[];examples={}
        for request,result in text.iter_outputs():
            action=result.trace()["action"]
            stats["eligible_queries"]+=int(action["eligible"]);stats["edits"]+=len(action["edits"])
            key="NOOP_queries" if result.payload==request.public.caption else "edited_queries"
            stats[key]+=1
            examples.setdefault("first_"+key,request.binding.sample_id)
            content.append([request.binding.sample_id,result.trace()["original_utf8_sha256"],
                            result.trace()["adversarial_utf8_sha256"],digest(action)])
        for key in ("edited_queries","NOOP_queries","edits"):
            need(stats[key]==sources.summary["query_statistics"][key],"original H summary/action count mismatch")
        text_report={**text.report(),**stats,"examples":examples,"ordered_text_action_digest":digest(content)}
        text_refs=[FileRef(p,v["sha256"]) for p,v in text.file_snapshot.items()]
    # 这里只构造计划，不发布3074/6156攻击单元或调用Plugin.generate。
    refs={r.path:r for r in sources.refs.values()}
    for ref in (legacy.config_ref,legacy.catalog_ref,*legacy.runtime_refs,
                legacy.ref("gv2","protocol"),legacy.ref("hv1","protocol"),legacy.ref("hv1","selection"),
                *(FileRef(p,h) for p,h in legacy.framework_sources.items()),*text_refs):
        refs[ref.path]=ref
    plan=MaterializationPlan(sources.manifest,legacy.provenance("gv2"),provenance,tuple(refs.values()),
                            sources.base["image_root"],source_config=legacy.config_ref)
    need(unchanged(protected)==7991,"historical protection count")
    report={"schema":"tbps-materialization-source-audit-1","status":"MATERIALIZATION_CPU_SOURCE_PASS",
       "G_generation":{"required_gallery":len(rows),"original_images_sha_verified":ready,
         "required_test_conditions":0,"missing_required_test_conditions":0,"conditions_required":False,
         "G2_training_supervision_entries":2176,"G2_training_cache_official_test_entries":len(overlap),
         "G1_test_condition_entries":len(old["entries"]),"G1_conditions_used_for_G2":False,
         "Test_G2_RGB_artifacts_available":0,"G_input":"RGB image only",
         "original_call_path":[asdict(FileRef(str(p),sha_file(p))) for p in call_sources]},
       "H_historical_text":text_report,"plan_id":plan.id,"dataset_manifest_sha256":plan.manifest_sha256,
       "gallery_order_sha256":digest([g.image_id for g in plan.manifest.gallery]),
       "query_order_sha256":digest([q.query_id for q in plan.manifest.queries]),
       "historical_files_sha_mtime_unchanged":7991,
       "model_loads":0,"G_forwards":0,"H_forwards":0,"IRRA_forwards":0,"JVPs":0,"backwards":0,
       "feature_tensors_loaded":0,"rankings":0,"real_units_published":0,
       "text_reuse_scope":"fixed H7 Unicode strings/actions only; no IRRA embedding or new policy decision",
       "pixel_artifacts_status":"NOT_MATERIALIZED","text_source_status":"TEXT_SOURCE_VERIFIED"}
    return report,plan,tuple({r.path:r for r in (*refs.values(),*v1refs)}.values())

def main(argv=None):
    p=argparse.ArgumentParser(description="Stage4C1 CPU sources only; no generation/model/GPU option")
    p.add_argument("--config",default=str(Path(__file__).resolve().parents[3]/"configs/framework/stage4b1_replay.server.json"))
    p.add_argument("--output")
    a=p.parse_args(argv)
    with cpu_only():report,plan,refs=audit(a.config)
    if a.output:
        root=Path(a.output).resolve()
        store=ArtifactStore(root,new_base="/home/lzf/ldx/outputs/tbps-framework-stage4c1")
        need(not root.exists(),"immutable CPU audit output already exists")
        producers=tuple(FileRef(str(Path(__file__).with_name(n)),sha_file(Path(__file__).with_name(n)))
                        for n in ("audit_sources.py","contracts.py","store.py","pipeline.py","legacy_text.py"))
        refs=tuple({r.path:r for r in (*refs,*producers)}.values())
        store.publish("result",{"report.json":json_bytes(report),"plan.json":json_bytes(plan.descriptor())},
                      {"purpose":"CPU source checks only; no real materialization"},sources=refs)
    print(json.dumps(report,ensure_ascii=False,indent=2));print("MATERIALIZATION_CPU_SOURCE_PASS")
    return 0
if __name__=="__main__":raise SystemExit(main())
