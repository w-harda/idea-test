"""B正式训练/Pilot入口；默认CPU预检，正式训练必须匹配独立READY回执。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time
import traceback
from tbps.artifacts import ArtifactStore,FileRef,checked,digest,json_bytes,read_json,sha_file
from .contracts import require
from .formal_sources import load_formal,source_preflight,prepare_cpu,verified_native,FormalGrant
from .native_sources import source_key
from .formal_state import FormalJournal,run_rounds

def locked_ref(value):
    # 历史证据索引额外包含metadata，不属于FileRef构造参数；SHA校验仍然完整执行。
    return FileRef(value["path"],value["sha256"])

def history(f):
    """读取原151378条快照，不重写原日志/元数据或旧审计回执。"""
    from .preflight import metadata
    p=f["references"]["history_index"];checked(FileRef(**p));record=read_json(p["path"])
    path=record["history_snapshot"];checked(locked_ref(path));rows=read_json(path["path"])
    require(len(rows)==151378,"protected history cardinality")
    for name,stat in rows.items():require(Path(name).is_file() and metadata(name)==stat,"history metadata changed: "+name)
    checked(locked_ref(record["old_FAIL_Watcher_index"]))
    old=read_json(record["old_FAIL_Watcher_index"]["path"])
    for name,ref in old.items():
        require(metadata(name)==ref["metadata"] and sha_file(name)==ref["sha256"],"old failure/watcher changed")
    for ref in record["parent_checkpoints"].values():checked(locked_ref(ref))
    for name,ref in record["artifact_files"].items():
        require(metadata(name)==ref["metadata"] and sha_file(name)==ref["sha256"],"original probe evidence changed")
    return {"history_metadata":len(rows),"FAIL_Watcher":len(old),"parent_probe_preserved":True}

def ready_for_training(f):
    path=Path(f["ready_receipt"])
    if not path.is_file():return False
    r=read_json(path)
    checked(FileRef(**r["pilot_audit"]))
    require(Path(r["pilot_audit"]["path"])==Path(f["audits"]["pilot"])/"completed/report.json","foreign Pilot audit")
    ArtifactStore(f["audits"]["pilot"],new_base=f["output_base"]).verify("completed",
        expected_provenance={"formal_configuration":sha_file(f["_config_path"]),"scope":"pilot"})
    a=read_json(r["pilot_audit"]["path"])
    require(r["formal_configuration_sha256"]==sha_file(f["_config_path"])
        and r["producer_sha256"]==digest(f["producers"])
        and a["status"]=="GH_B_FORMAL_PILOT_CPU_AUDIT_PASS" and a["scope"]=="pilot"
        and a["source"]["formal_config_sha256"]==sha_file(f["_config_path"]),"READY source/Pilot conflict")
    return True

def run(path,scope,*,allow_pilot=False,allow_training=False,resume_round=None):
    f=load_formal(path)
    require(allow_pilot if scope=="pilot" else allow_training and ready_for_training(f),
            "explicit GPU capability and completed Pilot READY required")
    control=Path(f["controls"][scope]);control.mkdir(parents=True,exist_ok=True)
    with (control/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        f,c,bundle,gate,pre=prepare_cpu(path,scope)
        pre["history"]=history(f)
        root=Path(f["outputs"][scope]);journal=FormalJournal(root/"rounds",sha_file(path),f["output_base"])
        previous=None
        if resume_round is None:
            require(not root.exists(),"Attempt consumed; only explicit committed boundary resume allowed")
            parent=bundle.state
        else:
            require(scope=="train","Pilot never automatically resumed/repeated")
            parent,previous=journal.load_boundary(resume_round,bundle.state["parents"])
            require(parent["cursor"]["round"]==resume_round,"resume absolute round")
            require(all(int(p.name.split("_")[-1])<=resume_round for p in (root/"round_io").glob("round_*"))
                and all(int(p.name.split("_")[-1])<=resume_round for p in (root/"B").glob("bank_*")),
                "later uncommitted IO/bank blocks resume")
            bank=ArtifactStore(root/"B"/f"bank_{resume_round}",new_base=f["output_base"])
            bank.verify("G_state");bank.verify("completed")
            require(read_json(bank.root/"completed/identity.json")==previous["bank"],"resume bank binding")
            for j in range(len(c["prototype"]["train_ids"])):
                bank.verify(f"claim_{j:04d}");bank.verify(f"image_{j:04d}")
            require(not bank.pending(),"incomplete committed bank")
        from .preflight import resource_snapshot
        resources=resource_snapshot(c["prototype"])
        require(resources["resource_eligible"],"RESOURCE_BUSY: no GPU Attempt created")
        grant=FormalGrant(c,scope,allow_pilot=allow_pilot,allow_training=allow_training,
                          ready=scope=="pilot" or ready_for_training(f))
        store=ArtifactStore(root,new_base=f["output_base"])
        if resume_round is None:
            store.publish("claim",{"preflight.json":json_bytes(pre),"resources.json":json_bytes(resources)},
                          {"formal_config":sha_file(path),"native_source":source_key(c),"scope":scope})
            journal.initialize({"formal":f,"native_source":source_key(c),"scope":scope},parent)
        else:
            failures=tuple(FileRef(str(p),sha_file(p)) for p in root.glob("FAIL*"))
            resume_id=journal.record_resume(resume_round,failures)
        run_id="initial" if resume_round is None else "resume_"+str(resume_id)
        identity={"pid":os.getpid(),"ppid":os.getppid(),"pgid":os.getpgid(0),"sid":os.getsid(0),
                  "tty":os.ttyname(0) if os.isatty(0) else None,"scope":scope}
        (control/("process_"+run_id+".json")).write_bytes(json_bytes(identity))
        stop=[False]
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
        start=time.monotonic()
        def phase(label,**fields):
            (control/"status.json").write_bytes(json_bytes({"phase":label,**identity,**fields}))
        try:
            import torch,resource
            from tbps.models.irra import fp32_runtime
            from .formal_backend import loaded_formal_backend
            torch.set_num_threads(2)
            gate.initialize_cuda()
            with fp32_runtime(),loaded_formal_backend(c,bundle,grant) as backend:
                backend.load_reward_inputs()
                def progress(label,index,visited,total):
                    import resource
                    info={"round":index,"visited":visited,"total":total,"cursor":dict(backend.cursor),
                        "counts":dict(backend.counts),"GPU_allocated":torch.cuda.memory_allocated(),
                        "GPU_reserved":torch.cuda.memory_reserved(),
                        "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
                    phase(label,**info);print("GH_B_STAGE",label,index,visited,"/",total,info["cursor"],flush=True)
                backend.progress=progress
                backend.resume_ledger=previous
                phase("RUNNING_B",round=(resume_round or 0)+1)
                state,complete=run_rounds(backend,journal,parent,c["prototype"]["rounds"],
                    backend.original_orders,start=resume_round or 0,stop_requested=lambda:stop[0],
                    on_round=lambda i,s,l:phase("ROUND_COMMITTED",round=i,cursor=s["cursor"]))
                report={"status":"GH_B_FORMAL_PILOT_GENERATED" if scope=="pilot" else "GH_B_FORMAL_GENERATED",
                    "complete":complete,"scope":scope,"cursor":state["cursor"],"rounds":state["cursor"]["round"],
                    "counts":backend.counts,"frozen_sources":backend.final_frozen(),
                    "elapsed_seconds":time.monotonic()-start,"peak_allocated":torch.cuda.max_memory_allocated(),
                    "peak_reserved":torch.cuda.max_memory_reserved(),
                    "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    "native_source":source_key(c),"independent_CPU_audit":"PENDING",
                    "not_attack_performance_evaluation":True}
            verified_native(f,scope);report["history"]=history(f)
            if complete:
                store.publish("generated",{"report.json":json_bytes(report)},{"formal_config":sha_file(path),"scope":scope})
                phase("GPU_GENERATED_CPU_AUDIT_PENDING",cursor=state["cursor"])
            else:
                store.publish("stopped_"+str(state["cursor"]["round"]),{"report.json":json_bytes(report)},
                              {"formal_config":sha_file(path),"scope":scope})
                phase("STOPPED_COMMITTED_BOUNDARY",cursor=state["cursor"])
            print(json.dumps(report,ensure_ascii=False,indent=2));print(report["status"],flush=True)
            return complete
        except BaseException:
            error=traceback.format_exc()
            with (root/("FAIL_"+run_id)).open("x") as out:out.write(error)
            phase("FAIL",error=error);print(error,flush=True);raise

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--action",choices=("preflight","resources","pilot","train","audit"),default="preflight")
    ap.add_argument("--scope",choices=("pilot","train"),default="pilot")
    ap.add_argument("--allow-gpu-pilot",action="store_true")
    ap.add_argument("--allow-training",action="store_true")
    ap.add_argument("--resume-round",type=int)
    a=ap.parse_args()
    if a.action in ("pilot","train"):
        done=run(a.config,a.action,allow_pilot=a.allow_gpu_pilot,allow_training=a.allow_training,resume_round=a.resume_round)
        if not done:raise SystemExit(3)
    elif a.action=="audit":
        from .formal_audit import audit
        print(json.dumps(audit(a.config,a.scope),ensure_ascii=False,indent=2))
    else:
        f,c,b,g,r=prepare_cpu(a.config,a.scope);r["history"]=history(f)
        if a.action=="resources":
            from .preflight import resource_snapshot
            r["resources"]=resource_snapshot(c["prototype"])
        print(json.dumps(r,ensure_ascii=False,indent=2));print(r["status"])
        if a.action=="resources" and not r["resources"]["resource_eligible"]:raise SystemExit(75)
if __name__=="__main__":main()
