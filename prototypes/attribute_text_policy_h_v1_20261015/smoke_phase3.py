"""H-v1 Phase 3：默认CPU preflight；真实Smoke必须显式授权且≤10+10。"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import signal
import subprocess
import time
import traceback
from unittest.mock import patch
import torch
from phase2_sources import ROOT,digest,sha,read_json,new_json
from phase3_inputs import preflight,safe_output,verify_sources,validate_selection
from real_feedback import load_real_runtime
from real_feature_cache import RealFeatureCache,feature_stats
from feedback_backend import state_sha
from text_frontend import CaptionFrontend
from policy import CharacterPolicy
from constraints import candidates,make_edit,replay

class StopBoundary:
    def __init__(self):self.requested=False;self.old={}
    def request(self,*args):self.requested=True
    def check(self):
        if self.requested:raise InterruptedError("safe stop at feature publication boundary")
    def __enter__(self):
        for sig in (signal.SIGINT,signal.SIGTERM):
            self.old[sig]=signal.getsignal(sig);signal.signal(sig,self.request)
        return self
    def __exit__(self,*args):
        for sig,handler in self.old.items():signal.signal(sig,handler)

def forbidden(*args,**kwargs):
    raise AssertionError("Phase 3 forbids optimizer/backward/JVP/training")

def gpu_status():
    return subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                                    "--format=csv,noheader"],text=True).strip()

def reward_stats(rows):
    result={}
    for mode in ("sampled","greedy"):
        result[mode]={}
        for k in ("R_sim","R_rank","total","synergy_increment"):
            values=[r[mode]["reward"][k] for r in rows]
            result[mode][k]={"mean":sum(values)/len(values),"min":min(values),"max":max(values)}
    return result

def run_smoke(catalog,gallery,queries,probe,protocol,output):
    validate_selection(catalog,gallery,queries)
    if protocol["execution_backend"]!="official_irra_real" or len(gallery)!=10 or len(queries)!=10:
        raise ValueError("real Phase 3 runner is fixed at 10 Gallery + 10 train Query")
    output=safe_output(output)
    # 不自动恢复失败GPU Smoke，也不重复任何已完成图像；已有输出由只读verify检查。
    output.mkdir(parents=True,exist_ok=False)
    attempt=output/"attempts/attempt_0001"
    attempt.mkdir(parents=True)
    new_json(output/"protocol.json",protocol)
    runtime=None
    started=time.perf_counter()
    context={"stage":"initialization"}
    with StopBoundary() as stop,ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor,"backward",side_effect=forbidden))
        stack.enter_context(patch.object(torch.autograd,"grad",side_effect=forbidden))
        stack.enter_context(patch.object(torch.optim.Optimizer,"__init__",side_effect=forbidden))
        try:
            verify_sources(protocol)
            resources_before=gpu_status()
            torch.cuda.reset_peak_memory_stats()
            frontend=CaptionFrontend()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                policy=CharacterPolicy(probe).eval().requires_grad_(False)
            h_before=state_sha(policy)
            sampler=torch.Generator(device="cpu").manual_seed(42)
            runtime=load_real_runtime(protocol,probe)
            loaded=time.perf_counter()
            stop.check()
            cache=RealFeatureCache(output/"real_features",catalog,protocol,frontend,probe)
            image_rows=[]
            with torch.inference_mode():
                for index,sample in enumerate(gallery):
                    context={"stage":"image","row_id":sample.row_id,"index":index}
                    stop.check()
                    t=time.perf_counter()
                    data,md=cache.image(sample,runtime)
                    torch.cuda.synchronize()
                    image_rows.append({"row_id":sample.row_id,"linf":md["linf"],
                                       "features":feature_stats(data),"seconds":time.perf_counter()-t})
                    print("REAL_IMAGE_COMPLETE",index+1,"/",len(gallery),sample.row_id,
                          "linf",md["linf"],flush=True)
                    stop.check()
                features=cache.gather(gallery)
                rows=[]
                for index,query in enumerate(queries):
                    context={"stage":"query","row_id":query.row_id,"index":index}
                    stop.check()
                    t=time.perf_counter()
                    example=frontend(query.caption)
                    # 唯一传给H的对象是Caption-only TextExample。
                    greedy=policy.rollout(example,"greedy")
                    sampled=policy.rollout(example,"sample",generator=sampler)
                    if greedy.budget_k!=sampled.budget_k:
                        raise ValueError("sampled/greedy initial Dynamic-K mismatch")
                    for out in (sampled,greedy):
                        if replay(example,out.edits,out.budget_k)!=out.adversarial:
                            raise ValueError("H action record does not replay")
                    bound=cache.bind_reward(query,gallery,runtime,sampled.budget_k)
                    sr,gr=bound.evaluate_many(query.caption,(sampled.adversarial,greedy.adversarial))
                    _,text_md=cache.text(query,(query.caption,sampled.adversarial,greedy.adversarial))
                    positive=[s.row_id for s in gallery if s.pid==query.pid]
                    negative=[s.row_id for s in gallery if s.pid!=query.pid]
                    rows.append({"row_id":query.row_id,"training_identity":asdict(query),
                        "positives":positive,"negatives":negative,"paired_index":gallery.index(query),
                        "K":sampled.budget_k,
                        "sampled":{"actions":sampled.record(),"reward":asdict(sr)},
                        "greedy":{"actions":greedy.record(),"reward":asdict(gr)},
                        "SCST_advantage_diagnostic_only":sr.total-gr.total,
                        "text_features":text_md["feature_stats"],"seconds":time.perf_counter()-t})
                    print("REAL_QUERY_COMPLETE",index+1,"/",len(queries),query.row_id,"K",sampled.budget_k,
                          "edits",len(sampled.edits),len(greedy.edits),
                          "reward",sr.total,gr.total,flush=True)
                    stop.check()
                diagnostic=None
                if all(not row[m]["actions"]["edits"] for row in rows for m in ("sampled","greedy")):
                    # 仅在H全部NOOP时允许一个单字符诊断；绝不计入H输出。
                    for query in queries:
                        example=frontend(query.caption)
                        for i in range(len(example.attributes)):
                            support=candidates(example,query.caption,i,set(),probe)
                            if not support:continue
                            pos=next(iter(support))
                            edit=make_edit(example,i,pos,support[pos][0])
                            variant=replay(example,(edit,),1)
                            bound=cache.bind_reward(query,gallery,runtime,1)
                            reward=bound.evaluate_many(query.caption,(variant,))[0]
                            diagnostic={"source":"legal single-character diagnostic; NOT H action",
                                        "row_id":query.row_id,"edit":asdict(edit),"text":variant,
                                        "tokenizer":probe.pair_report(query.caption,variant),"reward":asdict(reward)}
                            break
                        if diagnostic is not None:break
                    if diagnostic is None:
                        raise ValueError("all H NOOP and no legal diagnostic edit; Unicode chain unverified")
            torch.cuda.synchronize()
            final=runtime.finish()
            h_after=state_sha(policy)
            if h_after!=h_before or any(p.requires_grad or p.grad is not None for p in policy.parameters()):
                raise ValueError("H unexpectedly updated during Smoke")
            expected={"G":10,"clean":10,"adv":10,"text":30+(2 if diagnostic else 0)}
            if final["samples"]!=expected or final["calls"]["G"]!=10:
                raise ValueError("actual forward sample totals mismatch")
            if len(list((cache.root/"image").glob("*.pending"))) or len(list((cache.root/"text").glob("*.pending"))):
                raise ValueError("pending cache unit remains")
            verify_sources(protocol)
            summary={"method_audit_passed":True,"scope":"train-only GPU Smoke; no retrieval metrics",
                "gallery_count":10,"query_count":10,"training_catalog":catalog.audit,
                "protocol_sha256":digest(protocol),"checkpoint_sha256":protocol["checkpoint_sha256"],
                "IRRA_sha256":protocol["IRRA_sha256"],"model_checks":final,
                "H_state_sha256_before":h_before,"H_state_sha256_after":h_after,
                "H_parameters_unchanged":True,"H_forward_calls":20,
                "gallery_features":feature_stats(features),
                "maximum_linf":max(r["linf"] for r in image_rows),
                "images":image_rows,"queries":rows,"reward_statistics":reward_stats(rows),
                "legal_diagnostic_separate":diagnostic,"optimizer_steps":0,"backwards":0,"JVPs":0,
                "resources":{"GPU":torch.cuda.get_device_name(0),"peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
                             "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20,
                             "model_load_seconds":loaded-started,"total_seconds_excluding_CPU_preflight":time.perf_counter()-started,
                             "gpu_status_before":resources_before},
                "cache_file_sha256":cache.evidence(),"NaN_or_anomalies":[],
                "limitations":["H untrained","small train Gallery reward only","no mAP/ASR","no optimization","no synergy efficacy claim"]}
            new_json(output/"summary.json",summary)
            new_json(attempt/"execution.json",{"runtime":final,"resources":summary["resources"],"context":"completed"})
            new_json(output/"completed.json",{"method_audit_passed":True,"summary_sha256":sha(output/"summary.json"),
                "protocol_sha256":digest(protocol),"execution_sha256":sha(attempt/"execution.json"),"attempt":str(attempt)})
            print("HV1_PHASE3_REAL_SMOKE_COMPLETE",summary["resources"],final["samples"],flush=True)
            return summary
        except BaseException as exc:
            evidence={"error":repr(exc),"traceback":traceback.format_exc(),"context":context,
                      "runtime_samples":dict(runtime.samples) if runtime else None,
                      "runtime_calls":dict(runtime.calls) if runtime else None}
            new_json(attempt/("interruption.json" if isinstance(exc,InterruptedError) else "failure.json"),evidence)
            print("HV1_PHASE3_STOPPED",evidence,flush=True)
            raise
        finally:
            if runtime is not None:runtime.close()

def verify_completed(output):
    """只核对已有证据、SHA和小缓存，不做任何前向或奖励/指标重算。"""
    output=safe_output(output)
    protocol=read_json(output/"protocol.json")
    verify_sources(protocol)
    done=read_json(output/"completed.json")
    if not done["method_audit_passed"] or done["protocol_sha256"]!=digest(protocol):
        raise ValueError("completion protocol mismatch")
    if sha(output/"summary.json")!=done["summary_sha256"]:
        raise ValueError("summary SHA mismatch")
    summary=read_json(output/"summary.json")
    if (sha(Path(done["attempt"])/"execution.json")!=done["execution_sha256"]
            or summary["protocol_sha256"]!=digest(protocol)
            or summary["gallery_count"]!=10 or summary["query_count"]!=10
            or summary["maximum_linf"]>8/255 or summary["NaN_or_anomalies"]):
        raise ValueError("completed Smoke integrity mismatch")
    for name,expected in summary["cache_file_sha256"].items():
        if sha(name)!=expected:raise ValueError("cached feature/source SHA mismatch")
    if list((output/"real_features").rglob("*.pending")):
        raise ValueError("uncommitted cache unit exists")
    print("HV1_PHASE3_CPU_EVIDENCE_VERIFIED",summary["model_checks"]["samples"],flush=True)
    return summary

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action",choices=("preflight","smoke","verify"),default="preflight")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/phase3/smoke10")
    p.add_argument("--allow-gpu-smoke",action="store_true")
    return p

def main(argv=None):
    a=parser().parse_args(argv)
    if a.action=="verify":
        verify_completed(a.output)
        return
    if a.action=="smoke" and not a.allow_gpu_smoke:
        raise ValueError("explicit --allow-gpu-smoke required")
    t=time.perf_counter()
    catalog,gallery,queries,probe,protocol=preflight()
    print("HV1_PHASE3_CPU_PREFLIGHT",len(gallery),len(queries),catalog.audit,
          "protocol",digest(protocol),"seconds",time.perf_counter()-t,flush=True)
    if a.action=="preflight":return
    run_smoke(catalog,gallery,queries,probe,protocol,a.output)
if __name__=="__main__":main()
