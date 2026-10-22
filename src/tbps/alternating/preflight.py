"""纯标准库只读来源预检；不导入模型、不反序列化权重、不创建实验Attempt。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tbps.artifacts import FileRef, checked, digest, read_json, sha_file
from .contracts import Anchor, require

BASE_COMMIT = "0b87fa55caf04e7b636bd8d0bf74b13ef3a759ff"
PARENTS = {
    "G": "9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049",
    "H": "4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd",
    "IRRA": "9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009",
}

def verify_refs(refs):
    require(len({r["path"] for r in refs}) == len(refs), "duplicate source path")
    return [checked(FileRef(**ref)) for ref in refs]

def metadata(path):
    s=Path(path).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino]

def resource_snapshot(config):
    """只检查一次；nvidia-smi不初始化本进程CUDA，不杀进程、不轮询。"""
    gpu = subprocess.check_output([
        "nvidia-smi","--id=0","--query-gpu=name,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"],text=True,timeout=20).strip()
    parts=[p.strip() for p in gpu.split(",")]
    require(len(parts)==4 and len(gpu.splitlines())==1, "GPU status malformed")
    processes=subprocess.check_output([
        "nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits"],text=True,timeout=20).strip()
    mem=dict((line.split(":")[0],int(line.split()[1])*1024)
             for line in Path("/proc/meminfo").read_text().splitlines())
    disk=shutil.disk_usage(config["resource_probe_directory"])
    ino=os.statvfs(config["resource_probe_directory"])
    limits=config["proposed_gpu_limits"]
    checks={"gpu":int(parts[1])>=limits["minimum_free_mib"],
            "ram":mem["MemAvailable"]>=limits["minimum_ram_bytes"],
            "disk":disk.free>=limits["minimum_disk_bytes"],
            "inodes":ino.f_favail>=limits["minimum_free_inodes"]}
    return {"GPU":gpu,"compute_processes":processes,"available_ram_bytes":mem["MemAvailable"],
            "free_disk_bytes":disk.free,"free_inodes":ino.f_favail,
            "checks":checks,"resource_eligible":all(checks.values()),
            "GPU_authorized":False,"policy":"single shared_observe_only snapshot; no launch"}

def preflight(config):
    require("torch" not in sys.modules, "standard-library CPU preflight required")
    require(config["schema"]=="gh-alternating-prototype-1"
            and config["base_commit"]==BASE_COMMIT
            and config["arms"]==["A1","B"] and config["rounds"]==2
            and config["gpu_execution_authorized"] is False, "fixed CPU-only design")
    require(config["frozen_rules"]=={
        "G":"original K+1 Global/Local/Retain",
        "H":"original SCST R_sim+R_rank weights 1/1",
        "K":"unchanged; original G selection; H natural K",
        "inference":"G(I),H(T)", "precision":"fp32/math/no_tf32",
        "budget":"8/255", "gallery_refresh":"complete Train 2048 after each G phase"},
        "algorithm/config change not authorized")
    require(config["parent_sha256"]==PARENTS, "fixed complete G60/H10 parents")
    for arm in config["arms"]:
        root=Path(config["output_base"])/arm
        require(not root.exists(), "no duplicate/unfinished Attempt; preserve existing output")
    locks=verify_refs(config["sources"])
    locked={r["path"]:r["sha256"] for r in config["sources"]}
    inherited=read_json(config["historical_config"])
    require(config["historical_config"] in locked, "historical configuration must be locked")
    old_producers=verify_refs([dict(path=p,sha256=h) for p,h in inherited["producers"].items()])
    for ref in inherited["sources"].values():
        checked(FileRef(**ref))
    gp=read_json(str(Path(config["g_manifest"]).with_name("protocol.json")))
    g_locks=verify_refs([dict(path=p,sha256=h) for p,h in gp["producer_sha256"].items()])
    upstream=verify_refs([dict(path=p,sha256=h) for p,h in gp["upstream_sha256"].items()])
    hp=read_json(Path(config["history_root"])/"training/protocol.json")
    h_locks=verify_refs([dict(path=p,sha256=h) for p,h in hp["source_sha256"].items()])
    for k in ("parent","G60"):
        checked(FileRef(**inherited[k]))
    require(inherited["G60"]["sha256"]==PARENTS["G"]
            and inherited["parent"]["sha256"]==PARENTS["H"], "historical parent conflict")
    root=Path(__file__).resolve().parents[3]
    require(set(config["new_producers"])=={str(p) for p in (root/"src/tbps").rglob("*.py")},
            "complete current producer inventory required")
    for p,h in config["new_producers"].items():
        checked(FileRef(p,h))
    history=Path(config["history_root"])
    train=read_json(history/"qualification/train.json")["entries"]
    val=read_json(history/"qualification/validation.json")["entries"]
    require(len(train)==2048 and len(val)==128, "locked Train/Validation cardinality")
    raw=json.loads(Path(config["annotation"]).read_text())
    by_image={x["file_path"]:x for x in raw}
    require(len(by_image)==len(raw), "ambiguous annotation image")
    for role, entries in (("train",train),("validation",val)):
        for e in entries:
            i=e["identity"];a=by_image[i["image_rel_path"]]
            require(i["role"]==role and i["split"]==a["split"]=="train"
                    and i["pid"]==a["id"] and i["caption"]==e["caption"]==a["captions"][i["caption_index"]],
                    "actual member/PID/caption mismatch")
    for e in train:
        identity=e["identity"]
        checked(FileRef(str(Path(config["image_root"])/identity["image_rel_path"]),identity["image_sha256"]))
    ids=[e["identity"]["row_id"] for e in train]
    require(ids==config["train_ids"] and len(set(ids))==2048, "full ordered Train identity")
    clean_protocol=read_json(history/"gallery_train/protocol.json")
    clean_done=read_json(history/"gallery_train/completed.json")
    require(clean_done["image"]==2048 and clean_done["protocol_sha256"]==digest(clean_protocol)
            and clean_done["content_sha256"]==digest(clean_done["files"])
            and clean_protocol["identities"]==[e["identity"] for e in train], "fixed Clean bank receipt/order")
    clean_refs=verify_refs([dict(path=p,sha256=h) for p,h in clean_done["files"].items()])
    clean_binding=digest({"kind":"immutable original Clean gallery source receipt binding",
                         "protocol":clean_done["protocol_sha256"],"files":clean_done["files"]})
    require(config["clean_gallery_sha256"]==clean_binding, "fixed Clean bank binding mismatch")
    native_source=read_json(inherited["sources"]["dataset_sources"]["path"])["irra"]
    space=digest({"native_irra_source":native_source,
                  "preprocessing":"original FP32 RGB384x128 -> official IRRA normalization; unchanged",
                  "precision":"fp32/math/no_tf32"})
    require(config["source_space_sha256"]==space, "IRRA ModelSpace mismatch")
    tp={e["identity"]["pid"] for e in train};vp={e["identity"]["pid"] for e in val}
    test={a["id"] for a in raw if a["split"]=="test"}
    require(not tp&vp and not tp&test and not vp&test, "Train/Validation/Test PID overlap")
    sample_by_id={e["identity"]["row_id"]:e for e in train}
    gentries=read_json(config["g_manifest"])["entries"]
    gm={e["row_id"]:e for e in gentries if e["role"]=="train"}
    require(set(gm)==set(ids), "G/H Train membership")
    selected=[]
    for row in config["samples"]:
        e=sample_by_id[row];g=gm[row];i=e["identity"]
        require(e["caption"]==g["caption"] and i["pid"]==g["pid"], "G/H sample identity")
        checked(FileRef(g["tensor_path"],g["tensor_sha256"]))
        md=read_json(g["metadata_path"]);checked(FileRef(g["metadata_path"],g["metadata_sha256"]))
        anchors=md["source_selection"]["selected_attributes"]
        h_attrs=[Anchor(**a) for a in e["attributes"]]
        for a in anchors:
            ga=Anchor(**{k:a[k] for k in Anchor.__dataclass_fields__})
            ga.check(e["caption"])
            require(sum(ga==ha for ha in h_attrs)==1, "unmatched/AMBIGUOUS selected G anchor")
        checked(FileRef(str(Path(config["image_root"])/i["image_rel_path"]),i["image_sha256"]))
        selected.append({"row_id":row,"G_K":g["K"],"H_eligible":e["eligible"],
                         "H_reasons":e["reasons"],"original_caption":e["caption"]})
    require(len(selected)==8 and len(set(config["samples"]))==8
            and any(not e["H_eligible"] for e in selected)
            and any(e["H_eligible"] for e in selected), "eight samples with fixed NOOP control")
    hmd=read_json(inherited["parent"]["path"].replace("checkpoint.pt","checkpoint.json"))
    require(hmd["checkpoint_sha256"]==PARENTS["H"] and hmd["cursor"]==20480
            and hmd["updates"]==20370 and hmd["reason"]=="epoch_complete", "full H10 metadata")
    gmd=read_json(config["g_checkpoint_metadata"])
    require(gmd["epoch"]==60, "full G60 metadata")
    # 本轮校验字节与来源，不伪称已再次反序列化/恢复真实双Adam/CUDA RNG。
    require("torch" not in sys.modules, "no implicit model/Torch import")
    return {"status":"GH_ALTERNATING_CPU_SOURCE_PREFLIGHT_PASS", "parent_sha256":PARENTS,
            "source_files":len(locks),"legacy_producers":len(old_producers),
            "new_producers":len(config["new_producers"]), "G_producers":len(g_locks),
            "G_upstream":len(upstream),"H_training_sources":len(h_locks),
            "Clean_gallery_source_files":len(clean_refs), "samples":selected,
            "G_updates_per_round":sum(e["G_K"]+1 for e in selected),
            "H_updates_max_per_round":sum(e["H_eligible"] for e in selected),
            "Train":2048,"Validation":128,"PID_isolation":True,
            "model_loads":0,"model_forwards":0,"CUDA_initialized":False,
            "real_parent_restore":"PENDING GPU phase; original complete parent metadata verified",
            "new_JVP_numeric_equivalence":"PENDING explicit GPU authorization",
            "config_sha256":digest(config)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--resource-snapshot",action="store_true")
    args=ap.parse_args()
    c=read_json(args.config)
    result=preflight(c)
    if args.resource_snapshot:
        result["resources"]=resource_snapshot(c)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(result["status"])

if __name__=="__main__":
    main()
