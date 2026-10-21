"""G60与历史H1–H10内部Validation的显式只读桥；不加载模型或重算旧指标。"""
from pathlib import Path
from contextlib import ExitStack
import io, csv
import torch
from tbps.artifacts import FileRef, checked, read_json, digest, sha_file, json_bytes
from tbps.engines.historical_loader import feature_only_cpu, native_action_validator
from tbps.models.irra_cached import normalized
from tbps.materialization.irra_startup import CPUStartupGate
from tbps.attacks.provenance import LegacyAttackSources
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.runtime_closure import cpu_only, source_execution_audit, inspect_imports

KEYS=("R@1","R@5","R@10","mAP")
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-g60-h-validation")
G60="9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049"
G10="f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e"
IRRA="9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"
def need(ok,message):
    if not ok: raise ValueError(message)
def ref(path): return FileRef(str(Path(path).resolve()),sha_file(path))
def tensor_bytes(value):
    stream=io.BytesIO()
    torch.save({k:v.detach().cpu().clone() for k,v in value.items()},stream)
    return stream.getvalue()
def load_tensors(path):
    values=torch.load(path,map_location="cpu",weights_only=True)
    return {k:v.detach().clone() for k,v in values.items()}
def tensor_sha(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def configuration(path):
    c=read_json(path,"g60-h-candidates-validation-1")
    need(c["base_commit"]=="be30206f3d6170c06cb34716e3b111f52945c0af"
         and c["gallery"]==c["query"]==128 and c["H_epochs"]==list(range(1,11))
         and c["precision"]=="fp32/math/no_tf32" and c["seed"]==42 and c["image_batch"]==1
         and c["epsilon"]==8/255 and c["prefix_smoke"]==[0,1]
         and c["output"]==str(BASE/"validation_v1"),"fixed internal Validation protocol")
    need(c["G60"]["sha256"]==G60 and c["G10"]["sha256"]==G10,"locked G checkpoints")
    need(c["minimum_free_mib"]==6144 and c["minimum_ram_bytes"]==32*2**30
         and c["minimum_disk_bytes"]==10*2**30 and c["gpu_policy"]=="shared_observe_only","fixed resource policy")
    expected={str(p.resolve()) for p in Path(__file__).resolve().parents[1].rglob("*.py")}
    need(set(c["producers"])==expected,"complete fixed Validation runtime sources")
    for entry in c["sources"].values(): checked(FileRef(**entry))
    for key in ("G60","G10"): checked(FileRef(**c[key]))
    for p,h in c["producers"].items(): checked(FileRef(p,h))
    c["_config_path"]=str(Path(path).resolve())
    return c

def membership(identities,train,annotation,cache):
    need(len(identities)==128 and len({x["row_id"] for x in identities})==128
         and len({x["image_rel_path"] for x in identities})==128,"128 unique identities/images")
    pids={x["pid"] for x in identities}; tpids={x["pid"] for x in train}
    test={a["id"] for a in annotation if a["split"]=="test"}
    need(not pids & tpids and not pids & test,"Train/Validation/Test PID isolation")
    ce=[x for x in cache["entries"] if x["role"]=="validation"]
    need([x["row_id"] for x in ce]==[x["row_id"] for x in identities],"cache ordered validation membership")
    for row,entry in zip(identities,ce):
        idx=int(row["row_id"].split(":")[1]); a=annotation[idx]
        need(row["role"]=="validation" and row["split"]==a["split"]=="train"
             and row["caption_index"]==0 and a["captions"][0]==row["caption"]
             and a["file_path"]==row["image_rel_path"] and a["id"]==row["pid"],"actual annotation identity")
        for k in ("row_id","caption","image_rel_path","pid","caption_index","split","role"):
            need(row[k]==entry[k],"cache identity mismatch")
        need(row["metadata_sha256"]==entry["metadata_sha256"],"cache metadata identity")
    return {"validation":128,"validation_PIDs":len(pids),"train_PIDs":len(tpids),
            "train_validation_intersection":0,"test_intersection":0}

def legacy_text_record(md,entry):
    action=md["action"];caption=md["identity"]["caption"]
    if not entry["eligible"]:
        need(md["adversarial"]==caption and action=={"NOOP":True,"reasons":entry["reasons"],"edits":[],"budget_k":0},"retained NOOP")
        return None
    need(action["original"]==caption and action["adversarial"]==md["adversarial"],"action content")
    return {**action,"eligible":True,"reasons":entry["reasons"],"H_forward":1}

def prepare(c):
    gate=CPUStartupGate(); before={}
    def check_file(p,h=None):
        p=str(Path(p).resolve());item=FileRef(p,h or sha_file(p))
        result=checked(item);before[p]=result
        return item
    for item in c["sources"].values():check_file(**dict(p=item["path"],h=item["sha256"]))
    check_file(c["_config_path"])
    for p,h in c["producers"].items():check_file(p,h)
    for key in ("G60","G10"):check_file(c[key]["path"],c[key]["sha256"])
    evidence=read_json(c["sources"]["history_evidence"]["path"])
    for item in evidence["evidence"]+evidence["original_data_sources"]:check_file(item["path"],item["sha256"])
    root=Path(c["history_root"]); gp=read_json(root/"gallery_validation/protocol.json")
    done=read_json(root/"gallery_validation/completed.json")
    identities=gp["identities"]
    need(gp["G_sha256"]==G10 and gp["IRRA_sha256"]==IRRA and gp["role"]=="validation"
         and gp["execution_backend"]=="official_irra_real" and done["image"]==done["original_text"]==128
         and done["protocol_sha256"]==digest(gp) and done["content_sha256"]==digest(done["files"]),
         "historical gallery/model/completion")
    for p,h in done["files"].items():check_file(p,h)
    train=read_json(root/"gallery_train/protocol.json")["identities"]
    annotation=__import__("json").loads(Path(c["sources"]["annotation"]["path"]).read_text())
    cache=read_json(c["sources"]["condition_manifest"]["path"])
    members=membership(identities,train,annotation,cache)
    image_refs=[]
    for row in identities:
        image_refs.append(check_file(Path(c["image_root"])/row["image_rel_path"],row["image_sha256"]))
    qualification=read_json(root/"qualification/validation.json")
    need([q["identity"] for q in qualification["entries"]]==identities,"qualification ordered identity")
    candidates=evidence["candidates"]
    need([a["epoch"] for a in candidates]==list(range(1,11)),"all 10 H epochs, fixed order")
    allowed=set(); old=[]
    for row in identities:
        for family in ("image","original_text"):
            allowed.add(str(root/"gallery_validation"/family/digest(row)/"features.pt"))
    for candidate in candidates:
        check_file(candidate["checkpoint_path"],candidate["checkpoint_sha256"])
        folder=Path(candidate["validation_summary_path"]).parent
        s=read_json(folder/"summary.json");p=read_json(folder/"protocol.json");d=read_json(folder/"completed.json")
        need(sha_file(folder/"summary.json")==candidate["validation_summary_sha256"]
             and sha_file(folder/"protocol.json")==candidate["validation_protocol_sha256"]
             and d=={"summary_sha256":sha_file(folder/"summary.json"),"protocol_sha256":digest(p),
                      "checkpoint_sha256":candidate["checkpoint_sha256"],"queries":128}
             and s["groups"]==candidate["groups"] and s["gallery"]==s["queries"]==128
             and s["all_queries_retained"] and s["H_unchanged"]
             and p["identities"]==identities and p["validation_gallery_protocol_sha256"]==digest(gp)
             and p["validation_content_sha256"]==done["content_sha256"]
             and p["qualification_sha256"]==digest(qualification),"H validation completion/provenance")
        need(p["training_protocol_sha256"]==evidence["training_protocol"],"H training source")
        for name,h in p["source_sha256"].items():check_file(name,h)
        for name,h in s["query_files_sha256"].items():check_file(folder/name,h)
        actual={str(f.relative_to(folder)) for f in (folder/"queries").rglob("*") if f.is_file()}
        need(actual==set(s["query_files_sha256"]) and not list(folder.rglob("*.pending")),"cache exact files/no pending")
        allowed.update(str(folder/"queries"/digest(row)/"text.pt") for row in identities)
        old.append({"epoch":candidate["epoch"],"checkpoint":{"path":candidate["checkpoint_path"],"sha256":candidate["checkpoint_sha256"]},
                    "root":str(folder),"protocol":p,"summary":s})
    legacy=LegacyAttackSources(c["sources"]["attack_sources"]["path"])
    with cpu_only(),source_execution_audit(legacy):
        closure=inspect_imports(legacy)
    from tbps.models.irra_source import IRRASource
    source=IRRASource.from_dict(read_json(c["sources"]["dataset_sources"]["path"])["irra"])
    source.verify()
    need(source.checkpoint.sha256==gp["IRRA_sha256"],"IRRA ModelSpace bridge")
    for item in (source.checkpoint,source.config,source.clip_bootstrap,source.bpe,*(r for _,r in source.code)):
        check_file(item.path,item.sha256)
    for item in legacy.runtime_refs:check_file(item.path,item.sha256)
    clean=[];raw=[];features={};text_bindings={}
    with feature_only_cpu(allowed):
        for row in identities:
            for family,output,key in (("image",clean,"clean"),("original_text",raw,"text")):
                unit=root/"gallery_validation"/family/digest(row)
                md=read_json(unit/"metadata.json");receipt=read_json(unit/"receipt.json")
                need(md["identity"]==row and md["kind"]==family and md["protocol_sha256"]==digest(gp)
                     and receipt=={"protocol_sha256":digest(gp),"metadata_sha256":sha_file(unit/"metadata.json"),
                                  "features_sha256":sha_file(unit/"features.pt")},"original feature binding")
                data=load_tensors(unit/"features.pt")
                need(set(data)==({"clean","adversarial"} if family=="image" else {"text"}),"feature keys")
                for x in data.values():normalized(x,1)
                output.append(data[key])
        with native_action_validator(FileRef(**c["sources"]["attack_sources"])) as validate:
            for item in old:
                blocks=[];bindings=[];folder=Path(item["root"]);noops=0
                for row,entry in zip(identities,qualification["entries"]):
                    unit=folder/"queries"/digest(row);md=read_json(unit/"metadata.json");d=read_json(unit/"receipt.json")
                    need(md["identity"]==row and md["protocol_sha256"]==digest(item["protocol"]) and md["H_caption_only"] is True
                         and d=={"metadata_sha256":sha_file(unit/"metadata.json"),"feature_sha256":sha_file(unit/"text.pt")}
                         and md["feature_sha256"]==d["feature_sha256"],"H query feature SHA/order")
                    action=legacy_text_record(md,entry)
                    if action is not None:validate(row["caption"],action)
                    else:noops+=1
                    x=torch.load(unit/"text.pt",map_location="cpu",weights_only=True).detach().clone()
                    normalized(x,1);blocks.append(x)
                    bindings.append({"row_id":row["row_id"],"feature":str(unit/"text.pt"),
                        "feature_sha256":d["feature_sha256"],"original_utf8_sha256":__import__("hashlib").sha256(row["caption"].encode()).hexdigest(),
                        "adversarial_utf8_sha256":__import__("hashlib").sha256(md["adversarial"].encode()).hexdigest(),
                        "token_sha256":md["token_sha256"],"action_sha256":digest(md["action"])})
                features[str(item["epoch"])]=torch.cat(blocks).clone()
                text_bindings[str(item["epoch"])]=bindings
                need(noops==1,"fixed ineligible NOOP retained")
    # 完整CPU上下文退出后才给CUDA许可；不改变旧G10结果或选择记录。
    gate.checked();gate.exit_checked()
    need(not torch.cuda.is_initialized(),"CPU preflight side effect")
    report={"status":"G60_H_CPU_PREFLIGHT_PASS","membership":members,"H_candidates":10,"IRRA_text_features_reused":1280,
            "G_forward":0,"IRRA_forward":0,"CUDA":False,"sources":before,"closure":closure,
            "identities":identities,"text_bindings":text_bindings,
            "historical_gallery_protocol_sha256":digest(gp),"historical_content_sha256":done["content_sha256"],
            "existing_history_audit":"prior READ_ONLY_HV1_HISTORY_EVIDENCE_PASS; not rerun",
            "protection":"all consumed file SHA/stat rechecked by new audit"}
    return dict(c=c,report=report,gate=gate,source=source,legacy=legacy,identities=identities,old=old,
                raw=torch.cat(raw).clone(),clean=torch.cat(clean).clone(),texts=features,image_refs=image_refs)

def verify_unchanged(report):
    for p,record in report["sources"].items():
        need(checked(FileRef(p,record["sha256"]))==record,"historical/source file changed: "+p)

def metrics_csv(rows,rounded=False):
    out=io.StringIO(newline="");w=csv.DictWriter(out,fieldnames=["Group",*KEYS],lineterminator="\n");w.writeheader()
    for row in rows:
        w.writerow({k:(f"{v:.2f}" if rounded and type(v) is float else v) for k,v in row.items()})
    return out.getvalue().encode()
