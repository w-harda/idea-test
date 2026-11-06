"""GH-Alternating-v1独立配置/来源；旧146源码及实验身份不修改。"""
import copy
from pathlib import Path
from tbps.artifacts import FileRef, checked, digest, read_json, sha_file
from .contracts import require
from .native_sources import verify_config, strict_import_preflight
from .formal_sources import full_members, historical_receipts, FormalGrant
from .formal_cli import history

SCHEMA = "gh-alternating-v1"

def positive(value, name):
    require(type(value) is int and value >= 1, "positive integer " + name)

def load(path, scope):
    require(scope in ("pilot", "train"), "explicit execution scope")
    f = read_json(path, SCHEMA)
    require(f["name"] == "GH-Alternating-v1" and f["word_budget"] == 1, "variant identity")
    for k in ("rounds", "g_passes_per_round", "h_passes_per_round", "checkpoint_interval_rounds"):
        positive(f[k], k)
    require(f["pilot_rounds"] == 2 and f["long_training_authorized"] is False, "pilot-only authorization")
    require(type(f["seed"]) is int and f["image_epsilon"] == 8/255, "seed/image budget")
    for k in ("g_learning_rate", "h_learning_rate"):
        import math
        require(type(f[k]) in (float, int) and math.isfinite(f[k]) and f[k] > 0, "finite learning rate")
    for k in ("g_scheduler", "h_scheduler"):
        s=f[k];require(s["kind"] in ("constant", "step"), "supported independent Scheduler")
        positive(s["step_size"], k)
        require(0 < s["gamma"] <= 1 and (s["kind"] != "constant" or s["gamma"] == 1), "Scheduler gamma")
    root = Path(__file__).resolve().parents[3]
    require(f["producers"] == {str(p):sha_file(p) for p in (root/"src/tbps").rglob("*.py")}, "complete current producer SHA")
    for p,h in f["tools"].items():checked(FileRef(p,h))
    for v in f["references"].values():checked(FileRef(**v))
    old=read_json(f["references"]["formal_parent"]["path"])
    oldroot=Path(f["references"]["formal_parent"]["path"]).parents[2]
    for p,h in old["producers"].items():
        require(sha_file(root/Path(p).relative_to(oldroot)) == h, "frozen 146 producer changed")
    require(read_json(f["native_configs"][scope]) == derive(f, scope, path), "derived configuration mismatch")
    c=verify_config(f["native_configs"][scope])
    return f,c

def derive(f, scope, path):
    old=read_json(f["references"]["formal_parent"]["path"])
    c=read_json(old["native_configs"][scope])
    root=Path(__file__).resolve().parents[3]
    oldroot=Path(f["references"]["formal_parent"]["path"]).parents[2]
    p=c["prototype"]
    p["samples"]=list(old["pilot"]["rows"] if scope=="pilot" else p["train_ids"])
    p["rounds"]=f["pilot_rounds"] if scope=="pilot" else f["rounds"]
    p["output_base"]=f["outputs"][scope]
    p["proposed_gpu_limits"]={"policy":"shared_observe_only", "minimum_ram_bytes":f["resources"][scope]["minimum_ram_bytes"],
        "minimum_disk_bytes":f["resources"][scope]["minimum_disk_bytes"],"minimum_free_inodes":f["resources"][scope]["minimum_free_inodes"]}
    c["output_base"]=f["output_base"];c["smoke_output"]=f["outputs"][scope]
    c["probe_output"]=f["outputs"][scope]+"-UNAUTHORIZED-probe"
    c["producers"]=copy.deepcopy(f["producers"]);p["new_producers"]=copy.deepcopy(f["producers"])
    c["managed_roots"]=[str(root/"src") if x==str(oldroot/"src") else x for x in c["managed_roots"]]
    gm={e["row_id"]:e for e in read_json(p["g_manifest"])["entries"] if e["role"]=="train"}
    entries=read_json(Path(p["history_root"])/"qualification/train.json")["entries"]
    eligible={e["identity"]["row_id"]:e["eligible"] for e in entries}
    k=sum(gm[row]["K"] for row in p["samples"]);n=len(p["samples"]);r=p["rounds"]
    gp,hp=f["g_passes_per_round"],f["h_passes_per_round"]
    # 不重算固定Clean/初始G60图库；额度仅覆盖计划训练与完整新Bank。
    c["smoke"]={"rows":p["samples"],"max_G_training_updates":r*gp*(k+n),
        "max_H_updates":r*hp*sum(eligible[x] for x in p["samples"]),"full_bank_images":2048,
        "max_G_forwards":r*(gp*(k+n)+2048),"max_IRRA_images":r*(gp*(k+n)+2048),
        "max_JVPs":r*gp*k,"max_H_forwards":r*sum(eligible[x] for x in p["samples"])*(gp+2*hp),
        "max_reader_forwards":r*gp*n,"max_IRRA_texts":r*(10*gp*k+10*(gp+hp)*n)}
    c["condition_space_sha256"]=digest({"IRRA":p["source_space_sha256"],"reader":c["sources"]["reader"],
        "basis":c["sources"]["basis"],"byte_mapping":c["byte_alignment_policy"],
        "producer_sources":{q:h for q,h in c["producers"].items() if "/alternating/" in q}})
    p["condition_space_sha256"]=c["condition_space_sha256"]
    c["formal_binding"]={"scope":scope,"configuration":{"path":str(Path(path).resolve()),"sha256":sha_file(path)},
        "name":"GH-Alternating-v1; new WordBudget1 support and configurable scheduling"}
    c["derivation"]={"variant":"GH-Alternating-v1","original_G_math_changed":False,"original_SCST_changed":False,
        "H_action_support_changed":"WordBudget1; natural K retained; original one character per word",
        "GPU_policy":"shared_observe_only_no_VRAM_floor","formal_parent":f["references"]["formal_parent"]}
    c["alternating_configuration"]={k:f[k] for k in ("name","rounds","g_passes_per_round","h_passes_per_round",
        "seed","g_learning_rate","h_learning_rate","g_scheduler","h_scheduler","checkpoint_interval_rounds","word_budget")}
    return c

def prepare(path, scope):
    # 标准库来源预检必须先于Torch；原八样本投影不冒充完整新语境验证。
    from .preflight import preflight
    f,c=load(path,scope)
    projection=copy.deepcopy(c["prototype"])
    old=read_json(f["references"]["formal_parent"]["path"])
    projection.update(arms=["A1","B"],rounds=2,samples=old["pilot"]["rows"],output_base=f["output_base"]+"/source_projection_unused")
    report={"legacy_source_projection":preflight(projection),"members":full_members(c),
            "history":history(old),"old_receipts":historical_receipts(old)}
    from tbps.materialization.irra_startup import CPUStartupGate
    from .parents import restore_cpu_parents
    import torch
    gate=CPUStartupGate(lambda *a:None)
    report["imports"]=strict_import_preflight(c);bundle=restore_cpu_parents(c)
    from .word_budget import install
    install(bundle.h,bundle.bindings.get("policy"))
    report["parent_restore"]=bundle.report
    gate.checked();gate.exit_checked()
    require(not torch.cuda.is_initialized(),"CPU preparation initialized CUDA")
    report.update(status="GH_ALTERNATING_V1_CPU_PREFLIGHT_PASS",configuration_sha256=sha_file(path),
                  scope=scope,word_budget=1,legacy_H_protocol="unchanged source; new action support explicitly versioned")
    return f,c,bundle,gate,report
