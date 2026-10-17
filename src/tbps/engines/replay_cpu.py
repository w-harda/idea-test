"""Stage4B-2单次CPU Full重放；默认预检，真实计算需独立显式授权。"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import sys
import traceback
from ..artifacts import ArtifactError,ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes,digest
from .historical_sources import JointFullSources,safe_file,need
from .historical_loader import JointFullLoader,feature_only_cpu,native_action_validator
from .historical_contracts import ENGINE_FAMILIES
from .replay_acceptance import validate_plan
from .replay_control import ResourceGuard,historical_snapshot,unchanged,compact_heap
from .replay_oracle import official_rank,RankOracle,compare_metrics,compare_drops,verify_feature_values
from .evaluation import EvaluationEngine,publish_result,load_result,MODES
from .metrics import drops
from ..models.irra_cached import IRRACachedScorer

BASE=Path("/home/lzf/ldx/outputs/tbps-framework-stage4b2")
ATTEMPT=BASE/"replay_v1"
PROOF=Path("/home/lzf/ldx/outputs/tbps-framework-stage4b1/features_final")
PROOF_MANIFEST="5cd8dcf0e8ccbc61b58adc7dae37e26e51f361369e15f6fefa77c9465bfc1370"
PROOF_REPORT="7d8dd8364df3537471658a0d3ce3c30f5e4609f700bb75a68e2f68e0fdc979bf"
PROTECTED=Path("/home/lzf/ldx/tmp/tbps-framework-stage2/protected_before.json")
ROOT=Path(__file__).resolve().parents[3]

def producer_refs():
    files=["engines/replay_cpu.py","engines/replay_control.py","engines/replay_oracle.py",
           "engines/historical_sources.py","engines/historical_loader.py","engines/historical_contracts.py",
           "engines/evaluation.py","engines/metrics.py","engines/provenance.py",
           "engines/replay_acceptance.py","models/irra_cached.py","artifacts.py","runtime_closure.py"]
    return tuple(FileRef(str(ROOT/"src/tbps"/f),sha_file(ROOT/"src/tbps"/f)) for f in files)

def verify_prior_identity(identity,stored):
    need(json_bytes(asdict(identity))==json_bytes(stored),"prior/new historical replay identity mismatch")

def preflight(config,plan_path,*,check_stop=lambda:None):
    import torch
    plan=validate_plan(plan_path);check_stop()
    need(str(torch.__version__)==plan["torch"] and torch.get_num_threads()==plan["cpu_threads"],
         "frozen Torch CPU environment mismatch")
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not torch.cuda.is_initialized(),"explicit CPU-only environment")
    proof_refs=(FileRef(str(PROOF/"result/manifest.json"),PROOF_MANIFEST),
                FileRef(str(PROOF/"result/report.json"),PROOF_REPORT))
    for ref in proof_refs:checked(ref)
    prior=ArtifactStore(PROOF,new_base=PROOF.parent).verify("result");check_stop()
    report=read_json(PROOF/"result/report.json")
    need(report["status"]=="FEATURES_VERIFIED" and report["scores_computed"]==0 and
         report["feature_files_verified"]==3472 and report["metrics_replay_status"]=="PENDING",
         "Stage4B1 immutable FEATURES_VERIFIED receipt required")
    sources=JointFullSources(config);need(not sources.fixture,"real Joint Full only")
    verify_prior_identity(sources.identity,report["identity"])
    protected=historical_snapshot(PROTECTED);check_stop()
    producers=producer_refs()
    refs={ref.path:ref for ref in (*proof_refs,FileRef(str(PROOF/"result/completed.json"),sha_file(PROOF/"result/completed.json")),
          FileRef(str(PROTECTED),sha_file(PROTECTED)),FileRef(str(Path(plan_path).resolve()),sha_file(plan_path)),
          *sources.refs.values(),*producers)}
    for v in prior["sources"]:
        ref=FileRef(**v)
        need(ref.path not in refs or refs[ref.path]==ref,"prior/current source SHA conflict")
        refs[ref.path]=ref
    rank_fn=official_rank(sources.legacy_space.ranking_source)
    old_score=Path(sources.root).parents[3]/"attribute_image_generator_v2_20261014/evaluation_phase29a/eval_metrics.py"
    # 原评分表达式来源必须由历史协议锁定，而不是只依赖聊天说明。
    need(str(old_score) in sources.refs,"original scoring source not bound")
    result={"status":"CPU_REPLAY_PREFLIGHT_PASS","proof_manifest_sha256":PROOF_MANIFEST,
        "proof_report_sha256":PROOF_REPORT,"plan":plan,"historical_identity":asdict(sources.identity),
        "historical_protected_files":len(protected),"source_files":len(refs),
        "original_scoring_source":asdict(sources.refs[str(old_score)]),
        "official_ranking_source":asdict(sources.legacy_space.ranking_source),
        "actual_full_computations":0,"CUDA_initialized":False}
    return result,sources,rank_fn,tuple(refs.values()),protected

def execute_once(config,plan_path,*,authorized=False,output=ATTEMPT):
    need(type(authorized)is bool and authorized,"explicit --authorize-cpu-full-replay required")
    need(Path(output).resolve()==ATTEMPT,"single fixed new attempt required; no alternate retry directory")
    need(not ATTEMPT.exists(),"single attempt already claimed; preserve evidence and stop")
    plan=validate_plan(plan_path);store=ArtifactStore(ATTEMPT,new_base=BASE)
    owned=False;refs=();oracle=None;guard=ResourceGuard(plan["max_rss_bytes"],plan["max_seconds"])
    outcome={"status":"FAILED","metrics_replay_status":"PENDING","neural_forwards":0,"GPU_initializations":0,
             "optimizer_steps":0,"backwards":0,"JVP":0,"adversarial_pixel_sha256":"UNKNOWN"}
    try:
        with guard:
            pre,sources,rank_fn,refs,protected=preflight(config,plan_path,check_stop=guard.check)
            # 排他目录先于评分；任何失败都保留这个单次Claim，禁止新目录自动重试。
            ATTEMPT.mkdir(parents=True,exist_ok=False);owned=True
            store.publish("attempt",{"preflight.json":json_bytes(pre)},
                          {"authorization":"single CPU Full replay; frozen plan unchanged"},sources=refs)
            allowed={str(safe_file(sources.root,n).resolve()) for n in sources.declared if n.endswith("/features.pt")}
            allowed.update(str(ATTEMPT/"evaluation/scores"/(m+".pt")) for m in MODES)
            with feature_only_cpu(allowed) as forbidden,native_action_validator(FileRef(**sources.config["attack_sources"])) as validate:
                feature_snapshot={str(safe_file(sources.root,n)):checked(FileRef(str(safe_file(sources.root,n)),h))
                                  for n,h in sources.declared.items()}
                loader=JointFullLoader(sources,validate)
                feature_report,arrays=loader.verify_features(retain=True,check_stop=guard.check,
                    progress=lambda f,b,n:print("REPLAY_FEATURE_PROGRESS",f,b,n,flush=True) if b%64==0 else None)
                exact=verify_feature_values(loader,arrays,guard.check);guard.check();compact_heap()
                inputs={name:arrays[family] for name,family in ENGINE_FAMILIES.items()}
                scorer=IRRACachedScorer(sources.space,sources.identity,allow_full_scoring=True)
                engine=EvaluationEngine(sources.manifest,scorer,direction="higher_better",
                                        execution="historical_feature_replay",model_sources=tuple(sources.refs.values()))
                oracle=RankOracle(inputs,sources.protocol,rank_fn,plan,check_stop=guard.check)
                with oracle.instrument(tuple(MODES)):
                    protocol,metrics,scores=engine.run(inputs,query_chunk=128,check_stop=guard.check)
                metric_errors={m:compare_metrics(v,sources.summary["groups"][m],plan["metric_atol_percentage_points"])
                               for m,v in metrics["groups"].items()}
                dr_errors=compare_drops(metrics["drop_rate_percent"],sources.summary["drop_rate_percent"],
                                       plan["DR_atol_percentage_points"])
                expected_calls=4*((len(sources.manifest.queries)+127)//128)
                need(scorer.score_calls==oracle.score_calls==expected_calls,"actual score chunk counts")
                guard.check();compact_heap()
                publish_result(store,"evaluation",protocol,metrics,scores,sources=refs,
                               fault=lambda phase:guard.check())
                guard.check();compact_heap()
                _,saved,readback=load_result(store,"evaluation",expected_protocol_sha256=metrics["protocol_sha256"])
                import torch
                need(saved==metrics and all(torch.equal(scores[m],readback[m]) for m in MODES),"published Score readback not exact")
                del readback;guard.check()
                sources.verify_unchanged()
                # 缓存文件及旧树/原有验收来源均再核对SHA/mtime。
                need(feature_snapshot=={name:checked(FileRef(name,v["sha256"])) for name,v in feature_snapshot.items()},
                     "historical feature SHA/mtime changed during replay")
                protected_count=unchanged(protected)
                for ref in refs:checked(ref)
                need(not forbidden and not torch.cuda.is_initialized(),"forbidden operation attempted")
                outcome.update(status="PASS",metrics_replay_status="METRICS_REPLAY_VERIFIED",
                    groups=metrics["groups"],drop_rate_percent=metrics["drop_rate_percent"],
                    Joint_minus_ImageOnly_mAP_pp=metrics["Joint_minus_ImageOnly_mAP_pp"],
                    metric_errors_vs_original_summary_pp=metric_errors,DR_errors_vs_original_summary_pp=dr_errors,
                    comparisons=oracle.records,score_calls=scorer.score_calls,official_score_calls=oracle.score_calls,
                    new_ranking_calls=oracle.calls,official_ranking_calls=oracle.calls,feature_content=exact,
                    feature_batches=feature_report["feature_batches"],feature_files_verified=3472,
                    historical_files_unchanged=protected_count,source_files=len(refs),forbidden_attempts=dict(forbidden),
                    evaluation_manifest_sha256=sha_file(ATTEMPT/"evaluation/manifest.json"),
                    evaluation_protocol_sha256=metrics["protocol_sha256"],all_queries_retained=True,
                    interruption=False,not_a_new_attack=True)
            guard.check()
            outcome["resource"]=guard.snapshot()
            outcome["resource"]["timing_scope"]="through verification; excludes final acceptance publication"
            store.publish("result",{"report.json":json_bytes(outcome)},{"stage":"4B-2","purpose":"historical IRRA metric replay only"},sources=refs)
            guard.check()
    except BaseException as exc:
        outcome.update(status="INTERRUPTED" if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else "FAILED",
                       metrics_replay_status="PENDING",error=repr(exc),traceback=traceback.format_exc(),
                       comparisons={} if oracle is None else oracle.records,resource=guard.snapshot())
        if owned:
            # 不覆盖已发布evaluation/result；失败回执绝不表示指标验收通过。
            store.publish("failure",{"report.json":json_bytes(outcome)},{"stage":"4B-2","purpose":"failed single attempt evidence"})
        else:print(json.dumps(outcome,ensure_ascii=False,indent=2),flush=True)
        return 1
    outcome["resource_after_result_publication"]=guard.snapshot()
    print(json.dumps(outcome,ensure_ascii=False,indent=2),flush=True)
    print("METRICS_REPLAY_VERIFIED",flush=True);return 0

def main(argv=None):
    p=argparse.ArgumentParser(description="Stage4B2 CPU preflight by default; single explicit Full replay only")
    p.add_argument("--config",default=str(ROOT/"configs/framework/stage4b1_replay.server.json"))
    p.add_argument("--plan",default=str(ROOT/"configs/framework/stage4b2_acceptance.json"))
    p.add_argument("--authorize-cpu-full-replay",action="store_true")
    a=p.parse_args(argv)
    if a.authorize_cpu_full_replay:return execute_once(a.config,a.plan,authorized=True)
    from ..runtime_closure import cpu_only
    with cpu_only():report,*_=preflight(a.config,a.plan)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("CPU_REPLAY_PREFLIGHT_PASS",flush=True);return 0

if __name__=="__main__":raise SystemExit(main())
