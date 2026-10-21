"""H10父状态、G60奖励条件与独立扩展契约；不修改任何历史协议。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
import copy,json,os
import torch
from tbps.artifacts import ArtifactStore,FileRef,checked,sha_file,digest,read_json,json_bytes
from tbps.attacks.provenance import LegacyAttackSources
from tbps.attacks.legacy_runtime import legacy_imports
from tbps.runtime_closure import source_execution_audit,cpu_only,inspect_imports
from tbps.models.irra_source import IRRASource
from tbps.models.irra_cached import normalized
from tbps.materialization.irra_startup import CPUStartupGate
from .gv2_epoch_state import tree_hash

BASE=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
H10="4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd"
G60="9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049"
IRRA="9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009"
KEYS=("R@1","R@5","R@10","mAP")
def need(ok,message):
    if not ok:raise ValueError(message)
def ref(p):return FileRef(str(Path(p).resolve()),sha_file(p))
def stat(p):
    s=Path(p).stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
def config(path):
    c=read_json(path,"hv1-g60-extension-config-1")
    fixed={"base_commit":"807fdc63b0cd7546e76192870aabd19acc925c82","start_epoch":11,"target_epochs":60,
           "validation_epochs":[10,20,30,40,50,60],"seed":42,"batch":1,"smoke_visits":4,
           "precision":"fp32/math/no_tf32","gpu_policy":"shared_observe_only","minimum_free_mib":6144,
           "minimum_ram_bytes":32*2**30,"minimum_disk_bytes":35*2**30,"output_base":str(BASE)}
    need(all(c[k]==v for k,v in fixed.items()),"fixed adaptation protocol")
    need(c["parent"]["sha256"]==H10 and c["G60"]["sha256"]==G60,"fixed H10/G60")
    for d in (c["parent"],c["G60"],*c["sources"].values()):checked(FileRef(**d))
    root=Path(__file__).resolve().parents[3]
    expected={str(p) for p in (root/"src/tbps").rglob("*.py")}
    need(set(c["producers"])==expected,"complete frozen adaptation producers")
    for p,h in c["producers"].items():checked(FileRef(p,h))
    c["_path"]=str(Path(path).resolve())
    return c

def extension_contract(parent,bank,query_order,source_sha):
    need(parent["schema"]=="hv1-formal-periodic-checkpoint-1" and parent["cursor"]==20480
         and parent["updates"]==20370 and parent["reason"]=="epoch_complete","complete H10 parent required")
    original=parent["protocol"]
    need(parent["protocol_sha256"]==digest(original) and original["config"]["epochs"]==10
         and original["epoch_size"]==2048 and len(original["query_order"])==20480
         and query_order[:20480]==original["query_order"] and len(query_order)==122880,
         "original epoch order/config")
    cfg=copy.deepcopy(original["config"]);cfg["epochs"]=60
    return {"schema":"hv1-g60-adaptation-extension-1","execution_backend":"official_irra_real",
        "original_training_protocol":original,"original_protocol_sha256":digest(original),
        "parent_checkpoint_sha256":H10,"start_epoch":11,"target_epochs":60,
        "allowed_changes":["config.epochs:10->60","reward_adversarial_gallery:G10->G60","independent output/source identity"],
        "not_from_scratch_G60_60_epochs":True,"config":cfg,"query_order":query_order,
        "epoch_size":2048,"eligible_by_row":original["eligible_by_row"],
        "train_identities":original["train_identities"],"catalog_sha256":original["catalog_sha256"],
        "qualification_sha256":original["qualification_sha256"],"validation_feedback":False,
        "reward_gallery":bank,"source_sha256":source_sha,
        "validation_epochs":[10,20,30,40,50,60],
        "selection":"lowest repeated internal G60 Validation Joint mAP; tie earliest visit; adaptation diagnostic only"}

def restore_parent(old,c,policy=None,optimizer=None,sampler=None,allow_cuda=False):
    path=Path(c["parent"]["path"]);checked(FileRef(**c["parent"]))
    md=read_json(path.with_name("checkpoint.json"))
    ck=old("formal_training.checkpoint")
    p,_=ck.verify(path.parent,read_json(Path(c["history_root"])/"training/protocol.json"),
          md["parent_checkpoint_sha256"],md["start_cursor"],md["start_updates"])
    need(md["checkpoint_sha256"]==H10 and p["cursor"]==20480 and p["updates"]==20370
         and p["reason"]=="epoch_complete","verified actual H10 identity")
    required={"python","numpy","torch_cpu","sample_generator","torch_cuda"}
    need(set(p["rng"])==required and p["optimizer"]["state"] and len(p["query_order"])==20480,"full Adam/RNG/order")
    if policy is not None:ck.restore(p,policy,optimizer,sampler,allow_cuda=allow_cuda)
    return p

def preflight(c,*,check_restore=False):
    gate=CPUStartupGate();history=Path(c["history_root"])
    legacy=LegacyAttackSources(c["sources"]["attack_sources"]["path"])
    before={}
    for p,h in c["producers"].items():before[p]=checked(FileRef(p,h))
    for v in (*c["sources"].values(),c["parent"],c["G60"]):before[v["path"]]=checked(FileRef(**v))
    before[c["_path"]]=checked(ref(c["_path"]))
    # 绑定既有独立审计及上一阶段G60 Validation，不重跑旧审计或旧排名。
    past=read_json(c["sources"]["history_evidence"]["path"])
    need(past["status"]=="READ_ONLY_HV1_HISTORY_EVIDENCE_PASS"
         and past["completed"]["visits"]==20480 and past["completed"]["updates"]==20370,
         "historical completed H audit")
    for v in past["evidence"]+past["original_data_sources"]:before[v["path"]]=checked(FileRef(**v))
    gval=Path(c["validation_root"])
    audit=read_json(c["sources"]["G60_validation_audit"]["path"])
    need(audit["status"]=="G60_H_VALIDATION_CPU_AUDIT_PASS" and audit["G60"]==c["G60"]
         and audit["images"]==audit["queries"]==128,"prior G60 Validation audited")
    oldstore=ArtifactStore(gval,new_base=gval.parent)
    for key in ("protocol","features","completed"):oldstore.verify(key)
    before[str(gval/"features/gallery.pt")]=checked(ref(gval/"features/gallery.pt"))
    before[str(gval/"protocol/preflight.json")]=checked(ref(gval/"protocol/preflight.json"))
    oldpref=read_json(gval/"protocol/preflight.json")
    for p,s in oldpref["sources"].items():before[p]=checked(FileRef(p,s["sha256"]))
    gp=read_json(history/"gallery_train/protocol.json")
    gd=read_json(history/"gallery_train/completed.json")
    need(gd["image"]==2048 and gd["protocol_sha256"]==digest(gp) and gd["content_sha256"]==digest(gd["files"]),
         "completed original Clean Train Gallery")
    for p,h in gd["files"].items():before[p]=checked(FileRef(p,h))
    for p,h in gp["source_sha256"].items():before[p]=checked(FileRef(p,h))
    for role in ("train","validation"):
        q=read_json(history/f"qualification/{role}.json")
        before[str(history/f"qualification/{role}.json")]=checked(ref(history/f"qualification/{role}.json"))
        need(q["summary"]["samples"]==(2048 if role=="train" else 128),"locked qualification counts")
    with cpu_only(),source_execution_audit(legacy):closure=inspect_imports(legacy)
    # 原数据入口核对annotation、缓存、真实成员及PID，不反序列化JVP。
    with legacy_imports(legacy,allow_import=True) as old:
        catalog=old("training_inputs").load_training_catalog()
        tr=old("formal_training.data").read_qualification(history,catalog,"train")
        val=old("formal_training.data").read_qualification(history,catalog,"validation")
        identities=[asdict(x) for x in catalog.train]
        need(gp["identities"]==identities and tr["summary"]["eligible"]==2037
             and tr["summary"]["fixed_NOOP_ineligible"]==11,"same 2048/2037/11")
        need([asdict(x) for x in catalog.validation]==oldpref["identities"],"same reused128 Validation")
        raw=json.loads(Path(c["sources"]["annotation"]["path"]).read_text())
        testpids={x["id"] for x in raw if x["split"]=="test"}
        need(not {x.pid for x in catalog.train}&testpids
             and not {x.pid for x in catalog.validation}&testpids,"official Test PID isolation")
        for x in catalog.train:
            item=FileRef(str(Path(c["image_root"])/x.image_rel_path),x.image_sha256);before[item.path]=checked(item)
        original=read_json(history/"training/protocol.json")
        need(original["train_identities"]==identities and original["qualification_sha256"]==digest(tr),"parent training identity")
        for p,h in original["source_sha256"].items():before[p]=checked(FileRef(p,h))
        original_order=old("formal_training.training").order(catalog.train,60,42)
        need(original_order[:20480]==original["query_order"],"absolute epoch11 order prefix")
        restore_evidence=None
        if check_restore:
            probe=old("tokenizer_probe").official_probe()
            policy=old("policy").CharacterPolicy(probe,**original["config"]["architecture"])
            optimizer=old("phase4_training").optimizer_for(policy,original["config"])
            sampler=torch.Generator()
            p=restore_parent(old,c,policy,optimizer,sampler,allow_cuda=False)
            restore_evidence={"H_exact":True,"Adam_exact":True,"CPU_RNG_exact":True,
                "CUDA_RNG_saved_devices":len(p["rng"]["torch_cuda"]),"CUDA_RNG_restored":False,
                "H_state":tree_hash(policy.state_dict()),"Adam_state":tree_hash(optimizer.state_dict()),
                "RNG_state":tree_hash(p["rng"]),"cursor":p["cursor"],"updates":p["updates"],
                "parameter_Adam_steps":sorted({float(x["step"]) for x in p["optimizer"]["state"].values()})}
            del policy,optimizer,p
    source=IRRASource.from_dict(read_json(c["sources"]["dataset_sources"]["path"])["irra"])
    source.verify();need(source.checkpoint.sha256==IRRA==gp["IRRA_sha256"],"original IRRA identity")
    for r in (source.checkpoint,source.config,source.clip_bootstrap,source.bpe,*(r for _,r in source.code)):
        before[r.path]=checked(r)
    for r in legacy.runtime_refs:before[r.path]=checked(r)
    gate.checked();gate.exit_checked()
    report={"status":"HV1_G60_CPU_PREFLIGHT_PASS","sources":before,"identities":identities,
        "validation_identities":oldpref["identities"],"eligible":2037,"skipped":11,"Train":2048,"Validation":128,
        "PID_isolation":True,"parent_restore":restore_evidence,"closure":closure,"CUDA":False,"model_forward":0}
    return dict(c=c,legacy=legacy,source=source,gate=gate,report=report,gp=gp,gd=gd,
                train_qualification=tr,val_qualification=val,query_order=original_order)

def read_old_features(ctx,role,key):
    """只取经原完成回执绑定的Clean或原文；不将旧G10 Adv用于新奖励。"""
    root=Path(ctx["c"]["history_root"])/("gallery_"+role)
    gp=read_json(root/"protocol.json");done=read_json(root/"completed.json");blocks=[]
    family="original_text" if key=="text" else "image"
    for row in gp["identities"]:
        unit=root/family/digest(row)
        for name in ("features.pt","metadata.json","receipt.json"):checked(FileRef(str(unit/name),done["files"][str(unit/name)]))
        md=read_json(unit/"metadata.json");rc=read_json(unit/"receipt.json")
        need(md["identity"]==row and md["protocol_sha256"]==digest(gp)
             and rc=={"metadata_sha256":sha_file(unit/"metadata.json"),"features_sha256":sha_file(unit/"features.pt"),
                      "protocol_sha256":digest(gp)},"readonly Clean cache binding")
        data=torch.load(unit/"features.pt",map_location="cpu",weights_only=True)
        x=data[key].detach().clone();del data;normalized(x,1);blocks.append(x)
    return torch.cat(blocks).clone()

def unchanged(report):
    for p,s in report["sources"].items():need(checked(FileRef(p,s["sha256"]))==s,"source/history changed: "+p)

def audit_versions(root,protocol,old,*,load_last=True):
    root=Path(root);need(not (root/"pending").exists(),"uncommitted segment; no automatic replay")
    parent=H10;cursor=20480;updates=20370;items=[];last=None
    for path in sorted((root/"checkpoints").glob("visit_*")):
        value,md=old("formal_training.checkpoint").verify(path,protocol,parent,cursor,updates)
        need(md["cursor"]>cursor,"strictly advancing immutable checkpoint")
        items.append({"path":str(path),"sha256":md["checkpoint_sha256"],"cursor":md["cursor"],
                      "updates":md["updates"],"reason":md["reason"]})
        cursor,updates,parent=md["cursor"],md["updates"],md["checkpoint_sha256"];last=value
    return last,{"published":items,"cursor":cursor,"updates":updates,"parent":parent}

def make_protocol(ctx,old):
    bank=ArtifactStore(BASE/"gallery_train",new_base=BASE);bank.verify("completed")
    parent=restore_parent(old,ctx["c"])
    sources={p:s["sha256"] for p,s in ctx["report"]["sources"].items()}
    sources[str(BASE/"gallery_train/completed/manifest.json")]=sha_file(BASE/"gallery_train/completed/manifest.json")
    return extension_contract(parent,{"G60":ctx["c"]["G60"],
        "completed_manifest":str(BASE/"gallery_train/completed/manifest.json"),
        "completed_sha256":sources[str(BASE/"gallery_train/completed/manifest.json")]},ctx["query_order"],sources)
