"""只读启动/完成核验；不导入模型或运行训练，不改冻结配置。"""
import argparse
import json
import subprocess
from pathlib import Path
from gh_cuda_rng_validation import FROZEN_ROOT,CONFIG,CONFIG_SHA,SOURCE_KEY,PROBE_MANIFEST,RNG_OUTPUT,verify_history

def check_sources():
    from tbps.artifacts import sha_file
    from tbps.alternating.native_sources import verify_config,source_key
    from tbps.alternating.native_gate import verify_probe
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=FROZEN_ROOT,text=True).strip()=="aafc7715ddafa78f9a30410b6a6ed8852e81780a"
    assert not subprocess.check_output(["git","status","--porcelain"],cwd=FROZEN_ROOT,text=True).strip()
    c=verify_config(CONFIG)
    assert sha_file(CONFIG)==CONFIG_SHA and source_key(c)==SOURCE_KEY
    assert verify_probe(c)==PROBE_MANIFEST
    return c

def rng_receipt(c):
    from tbps.artifacts import ArtifactStore,sha_file
    store=ArtifactStore(RNG_OUTPUT,new_base=c["output_base"])
    assert not store.pending() and not (RNG_OUTPUT/"FAIL").exists()
    for unit in ("claim","state_capture","completed"):store.verify(unit)
    r=json.loads((RNG_OUTPUT/"completed/report.json").read_text())
    assert r["status"]=="CUDA_RNG_PASS" and r["source"]==SOURCE_KEY and r["probe_manifest_sha256"]==PROBE_MANIFEST
    assert r["G_H_Adam_unchanged"] and r["counts"]["model_forwards"]==r["counts"]["JVPs"]==r["counts"]["optimizer_steps"]==0
    return sha_file(RNG_OUTPUT/"completed/manifest.json")

def lifecycle(c):
    """当前生命周期：已有同来源探针PASS，Smoke身份尚未消费。"""
    from tbps.alternating.native_gate import verify_probe
    assert verify_probe(c)==PROBE_MANIFEST,"foreign/missing probe receipt"
    assert not Path(c["smoke_output"]).exists(),"Smoke already consumed; no replay"
    return PROBE_MANIFEST

def ready():
    from tbps.alternating.preflight import resource_snapshot
    c=check_sources()
    rng_sha=rng_receipt(c)
    lifecycle(c)
    resources=resource_snapshot(c["prototype"])
    assert resources["resource_eligible"],"RESOURCE_BUSY"
    assert c["prototype"]["proposed_gpu_limits"]["minimum_free_mib"]==10240
    return {"status":"SMOKE_READY","source":SOURCE_KEY,"probe_manifest":PROBE_MANIFEST,
        "CUDA_RNG_manifest":rng_sha,"resources":resources,"history":verify_history()}

def final():
    from tbps.artifacts import ArtifactStore,sha_file
    c=check_sources()
    rng_sha=rng_receipt(c)
    root=Path(c["smoke_output"])
    store=ArtifactStore(root,new_base=c["output_base"])
    assert not store.pending() and not (root/"FAIL").exists()
    store.verify("completed")
    ArtifactStore(root/"audit",new_base=c["output_base"]).verify("completed")
    report=json.loads((root/"completed/report.json").read_text())
    audit=json.loads((root/"audit/completed/report.json").read_text())
    assert report["status"]=="GH_NATIVE_TWO_ROUND_SMOKE_GENERATED" and report["rounds"]==4 and report["frozen_sources"]
    assert audit["status"]=="GH_NATIVE_TWO_ROUND_CPU_AUDIT_PASS" and audit["source"]==SOURCE_KEY
    assert report["source_sha256"]==SOURCE_KEY
    summary={"status":"GH_TWO_ROUND_SMOKE_AND_CPU_AUDIT_PASS",
        "smoke_control_flow":"PASS","CPU_audit":"PASS","feedback_effects":audit["feedback_effects"],
        "counts":report["counts"],"resource_statistics":{k:report[k] for k in [
        "elapsed_seconds","peak_allocated_bytes","peak_reserved_bytes","peak_RSS_bytes"]},
        "source":SOURCE_KEY,"probe_manifest":PROBE_MANIFEST,"CUDA_RNG_manifest":rng_sha,
        "smoke_manifest":sha_file(root/"completed/manifest.json"),
        "audit_manifest":sha_file(root/"audit/completed/manifest.json"),
        "history":verify_history(),"rounds":[]}
    for arm in ("A1","B"):
        for i in (1,2):
            roundroot=root/"rounds"/f"{arm}-r{i}"
            store2=ArtifactStore(root/"rounds",new_base=c["output_base"])
            store2.verify(f"{arm}-r{i}-claim");store2.verify(f"{arm}-r{i}")
            ledger=json.loads((roundroot/"ledger.json").read_text())
            summary["rounds"].append({"arm":arm,"round":i,
                "G_Adam_updates":sum(g["optimizer_steps"] for g in ledger["G"]),
                "H_visits":len(ledger["H"]),
                "H_Adam_updates":audit["states"][f"{arm}:{i}"]["cursor"]["h_updates"]-(20370 if i==1 else audit["states"][f"{arm}:1"]["cursor"]["h_updates"]),
                "bank":ledger["bank"],"G":ledger["G"],"H":ledger["H"]})
    assert sum(x["G_Adam_updates"] for x in summary["rounds"])==108
    assert sum(x["H_visits"] for x in summary["rounds"])==32
    assert sum(x["H_Adam_updates"] for x in summary["rounds"])<=28
    return summary

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--action",choices=["ready","final"],required=True)
    a=ap.parse_args()
    print(json.dumps(ready() if a.action=="ready" else final(),ensure_ascii=False,indent=2))
