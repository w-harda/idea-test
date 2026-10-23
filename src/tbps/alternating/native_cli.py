"""真实数值探针/完整两轮入口：默认仅CPU预检，不启动后台任务。"""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from tbps.artifacts import ArtifactStore, FileRef, checked, digest, json_bytes, read_json, sha_file
from .contracts import require
from .native_sources import verify_config, source_key, resource_protocol

def cpu_preflight(path, *, parents=False):
    # 父状态检查走与GPU入口相同的真实CPU准备流程，验证隔离退出与启动时序。
    if parents:
        c,bundle,gate,report=prepare_cpu_gate(path)
        report.update(source_sha256=source_key(c),status="GH_NATIVE_CPU_PREFLIGHT_PASS")
        return c,report,bundle
    # 完整来源核验必须发生于Torch导入之前。
    from .preflight import preflight
    c=verify_config(path)
    report={"sources":preflight(c["prototype"])}
    from .native_sources import strict_import_preflight
    report["imports"]=strict_import_preflight(c)
    bundle=None
    report["source_sha256"]=source_key(c)
    report["status"]="GH_NATIVE_CPU_PREFLIGHT_PASS"
    return c,report,bundle

def prepare_cpu_gate(path):
    from .preflight import preflight
    c=verify_config(path);sources=preflight(c["prototype"])
    import torch
    from tbps.materialization.irra_startup import CPUStartupGate
    from .native_sources import strict_import_preflight
    from .parents import restore_cpu_parents
    events=[];gate=CPUStartupGate(events.append)
    imports=strict_import_preflight(c)
    bundle=restore_cpu_parents(c)
    gate.checked();gate.exit_checked()
    require(not torch.cuda.is_initialized(),"CPU gate initialized CUDA")
    return c,bundle,gate,{"sources":sources,"imports":imports,"parent_restore":bundle.report,"startup_events":events}

def numerical_probe(backend):
    from .native_numeric import compare_gradients
    from .state import tree_sha
    from .parents import no_optimizer_steps
    c=backend.c
    before={k:tree_sha(v) for k,v in {
        "G":backend.g.state_dict(),"H":backend.h.state_dict(),
        "G_Adam":backend.ga.state_dict(),"H_Adam":backend.ha.state_dict()}.items()}
    rows=[]
    with no_optimizer_steps():
        for i,row in enumerate(c["probe"]["rows"]):
            trace=backend.h_rollout(backend.public_text(row))
            require((trace.original==trace.adversarial)==(i==0),
                    "locked H10 editable control produced NOOP; stop, do not replace sample")
            rows.append({"row":row,"trace":trace,"noop":i==0,"original":backend.original[row],
                         "image":backend.image(row)})
        import torch
        with torch.no_grad():
            original_feature=backend.victim.encode_texts([rows[1]["trace"].original]).detach().cpu()[0]
        original_error=float((original_feature-backend.original[rows[1]["row"]].payload["z_t"]).abs().max())
        require(original_error<=1e-6,"original text/cache numerical mismatch")
        result=compare_gradients(backend.g,backend.encoder,rows,backend.native,
                                 lambda row:backend.condition("B",row["row"],row["trace"]))
    for name,state in (("G",backend.g.state_dict()),("H",backend.h.state_dict()),
                       ("G_Adam",backend.ga.state_dict()),("H_Adam",backend.ha.state_dict())):
        require(tree_sha(state)==before[name],"numerical probe changed model/Adam: "+name)
    require(result["nonnoop"]["max_abs"]>0,"non-NOOP produced no G gradient change")
    require(result["noop"]["max_abs"]<=c["probe"]["noop_gradient_atol"],"NOOP G gradient differs")
    return {"status":"GH_NATIVE_NUMERIC_PROBE_PASS","comparisons":result,
            "nonnoop_gradient_max_abs":result["nonnoop"]["max_abs"],
            "noop_gradient_max_abs":result["noop"]["max_abs"],"optimizer_steps":0,
            "full_gallery_generations":0,"frozen_sources":backend.final_frozen(),
            "counts":dict(backend.counts),"conditions":backend.conditions_log,
            "original_text_cache_max_abs":original_error,"rollouts":backend.rollout_log,
            "scope":"fixed two Train captions; original first G-step Loss gradient only; not complete K+1 update"}

def run_gpu(path,action,*,allow_probe=False,allow_smoke=False):
    from .native_gate import authorize
    c=verify_config(path)
    # 在任何Attempt/CUDA/真实模型加载之前检查显式授权及数值探针回执。
    grant=authorize(c,action,allow_probe=allow_probe,allow_smoke=allow_smoke)
    c,bundle,gate,pre=prepare_cpu_gate(path)
    from .preflight import resource_snapshot
    resources=resource_snapshot(resource_protocol(c,action))
    require(resources["resource_eligible"],"RESOURCE_BUSY: no Attempt/CUDA created")
    root=Path(c[action+"_output"]);store=ArtifactStore(root,new_base=c["output_base"])
    require(not root.exists(),"Attempt already exists")
    store.publish("claim",{"preflight.json":json_bytes(pre),"resources.json":json_bytes(resources)},
                  {"source":source_key(c),"action":action})
    start=time.monotonic()
    try:
        import torch
        import resource
        from tbps.models.irra import fp32_runtime
        from .native_backend import loaded_backend
        torch.set_num_threads(2)
        # 不重置seed；两个父Checkpoint各自保存的完整RNG在相应训练阶段恢复。
        gate.initialize_cuda()
        with fp32_runtime(),loaded_backend(c,bundle,grant) as backend:
            if action=="probe":
                report=numerical_probe(backend)
            else:
                from .runner import Prototype
                from .state import RoundStore
                backend.load_reward_inputs()
                journal=RoundStore(root/"rounds",digest(c["prototype"]),new_base=c["output_base"])
                results=Prototype(c["prototype"],backend,journal).run(bundle.state)
                report={"status":"GH_NATIVE_TWO_ROUND_SMOKE_GENERATED",
                        "rounds":len(results),"counts":backend.counts,
                        "frozen_sources":backend.final_frozen(),"conditions":backend.conditions_log,
                        "independent_CPU_audit":"PENDING; run audit-smoke separately"}
            report.update(source_sha256=source_key(c),protocol_sha256=digest(c["prototype"]),
                          elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          model_restores={"G":1,"H":1,"IRRA":1,"reader":1},startup_events=pre["startup_events"])
        # 新旧全部来源再次SHA核验；不覆盖任何旧产物。
        verify_config(path)
        store.publish("completed",{"report.json":json_bytes(report)},{"source":source_key(c),"protocol":digest(c["prototype"])})
        print(json.dumps(report,ensure_ascii=False,indent=2));print(report["status"])
    except BaseException:
        error=traceback.format_exc()
        # 独立FAIL文件保留；禁止自动重试、回退或重用该Attempt。
        with (root/"FAIL").open("x") as f:f.write(error)
        print(error,flush=True)
        raise

def audit_smoke(c):
    """只读独立审计；不导入旧模型、不前向、不重算奖励/排名。"""
    import torch
    from .state import RoundStore, tree_sha
    from .native_gate import verify_probe
    verify_probe(c)
    root=Path(c["smoke_output"]);store=ArtifactStore(root,new_base=c["output_base"])
    require(not (root/"FAIL").exists() and not store.pending(),"Smoke failed/incomplete")
    store.verify("completed",expected_provenance={"source":source_key(c),"protocol":digest(c["prototype"])})
    report=read_json(root/"completed/report.json")
    require(report["counts"]["G"]==report["counts"]["IRRA_image"]==8300,"actual image forward accounting")
    require(report["status"]=="GH_NATIVE_TWO_ROUND_SMOKE_GENERATED" and report["rounds"]==4
            and report["frozen_sources"],"generated Smoke required")
    journal=RoundStore(root/"rounds",digest(c["prototype"]),new_base=c["output_base"])
    states={};ledgers={}
    for arm in ("A1","B"):
        for index in (1,2):
            round_id=f"{arm}-r{index}"
            manifest=journal.store.verify(round_id)
            state=torch.load(root/"rounds"/round_id/"state.pt",map_location="cpu",weights_only=False)
            require(tree_sha(state)==manifest["provenance"]["state_sha256"]
                    and state["parents"]=={"G":c["sources"]["G60"]["sha256"],"H":c["sources"]["H10"]["sha256"],
                                           "protocol":digest(c["prototype"])},"full state/source mismatch")
            ledger=read_json(root/"rounds"/round_id/"ledger.json")
            require(state["cursor"]=={"round":index,"g_updates":484080+27*index,
                    "h_visits":20480+8*index,"h_updates":20370+7*index},"round absolute cursor")
            require(ledger["all_stages_verified"] and sum(x["optimizer_steps"] for x in ledger["G"])==27
                    and len(ledger["H"])==8,"round native update ledger")
            bankroot=root/arm/f"bank_{index}"
            bs=ArtifactStore(bankroot,new_base=c["output_base"]);bs.verify("G_state");bs.verify("completed")
            bid=read_json(bankroot/"completed/identity.json")
            require(bid["ordered_train_ids"]==c["prototype"]["train_ids"],"complete ordered bank")
            require(bid["g_state_sha256"]==tree_sha(state["G"]) and bid["arm"]==arm and bid["round"]==index,
                    "bank/round G binding")
            rows=[]
            for position,row in enumerate(c["prototype"]["train_ids"]):
                bs.verify(f"claim_{position:04d}")
                md=bs.verify(f"image_{position:04d}")["provenance"]
                value=torch.load(bankroot/f"image_{position:04d}/data.pt",map_location="cpu",weights_only=True)
                from .native_backend import assert_pixels
                assert_pixels(value["original"],value["adversarial"],value["effective_delta"])
                require(md["content_sha256"]==tree_sha(value) and md["row"]==row and md["index"]==position,
                        "bank unit/order/content mismatch")
                rows.append(value["feature"])
            require(tree_sha(torch.cat(rows))==bid["adversarial_sha256"],"full bank content mismatch")
            require(not bs.pending(),"pending bank")
            ledgers[(arm,index)]=ledger
            states[(arm,index)]={"G":tree_sha(state["G"]),"H":tree_sha(state["H"]),"cursor":state["cursor"]}
    a,b=ledgers[("A1",1)]["G"],ledgers[("B",1)]["G"]
    require(a[0]["G_before"]==b[0]["G_before"] and a[0]["G_after"]==b[0]["G_after"]
            and a[0]["gradient_observations"]==b[0]["gradient_observations"],"first NOOP control differs")
    require(a[1]["G_before"]==b[1]["G_before"],"first editable control parent differs")
    effects={"first_nonNOOP_gradient_diff":a[1]["gradient_observations"][0]["gradient_sha256"]!=b[1]["gradient_observations"][0]["gradient_sha256"],
        "first_nonNOOP_G_update_diff":a[1]["G_after"]!=b[1]["G_after"],
        "same_text_Reward_changed":any(any(abs(x)>0 for x in h.get("reward_delta_same_text",[])) for v in ledgers.values() for h in v["H"]),
        "same_text_Advantage_changed":any(abs(h["reward_delta_same_text"][0]-h["reward_delta_same_text"][1])>0 for v in ledgers.values() for h in v["H"] if "reward_delta_same_text" in h)}
    # 零Advantage不改变原Reward；单独记录信息差是否真正可测，不靠调参变成PASS。
    effects["bidirectional_update_evidence_verified"]=all(effects.values())
    audit=ArtifactStore(root/"audit",new_base=c["output_base"])
    result={"status":"GH_NATIVE_TWO_ROUND_CPU_AUDIT_PASS","states":{f"{a}:{r}":v for (a,r),v in states.items()},
            "source":source_key(c),"feedback_effects":effects,"real_model_forwards":0,"CUDA":False}
    audit.publish("completed",{"report.json":json_bytes(result)},{"source":source_key(c)})
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--action",choices=("preflight","probe","smoke","audit-smoke","resources"),default="preflight")
    ap.add_argument("--check-parent-states",action="store_true")
    ap.add_argument("--allow-gpu-probe",action="store_true")
    ap.add_argument("--allow-gpu-smoke",action="store_true")
    a=ap.parse_args()
    if a.action in ("probe","smoke"):
        run_gpu(a.config,a.action,allow_probe=a.allow_gpu_probe,allow_smoke=a.allow_gpu_smoke)
    elif a.action=="audit-smoke":
        print(json.dumps(audit_smoke(verify_config(a.config)),ensure_ascii=False,indent=2))
    else:
        c,report,_=cpu_preflight(a.config,parents=a.check_parent_states)
        if a.action=="resources":
            from .preflight import resource_snapshot
            report["resources"]=resource_snapshot(c["prototype"])
        print(json.dumps(report,ensure_ascii=False,indent=2));print(report["status"])
if __name__=="__main__":main()
