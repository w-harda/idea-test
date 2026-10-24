"""B正式协议：保留冻结数学实现，独立绑定调度、完整Train与输出身份。"""
import copy
from pathlib import Path
from dataclasses import asdict
from tbps.artifacts import ArtifactStore, FileRef, checked, digest, read_json, sha_file
from .contracts import Anchor, require
from .native_sources import verify_config, source_key

SCHEMA="gh-b-formal-1"
PARENT_COMMIT="a7597f9c964dda0c8c48973e4678f0b9d3fa095a"

def ref(p):
    return {"path":str(Path(p).resolve()),"sha256":sha_file(p)}

def load_formal(path):
    f=read_json(path,SCHEMA)
    require(f["production_commit"]==PARENT_COMMIT and f["arm"]=="B","fixed native B protocol")
    require(f["train_rounds"]>=1 and type(f["train_rounds"]) is int,"positive round budget")
    require(f["training_authorized"] is False,"long training requires separate READY authorization")
    require(f["evaluation"]["scope"]=="official_test_after_training_only"
            and f["evaluation"]["query_chunk"]==128 and f["evaluation"]["precision"]=="fp32/math/no_tf32",
            "frozen IRRA evaluation")
    for v in f["references"].values():checked(FileRef(**v))
    for p,h in f["tools"].items():checked(FileRef(p,h))
    old=read_json(f["references"]["native_parent"]["path"])
    root=Path(__file__).resolve().parents[3]
    all_py={str(p):sha_file(p) for p in (root/"src/tbps").rglob("*.py")}
    require(f["producers"]==all_py,"complete formal producer SHA inventory")
    frozen_root=Path(f["references"]["native_parent"]["path"]).parents[2]
    for p,h in old["producers"].items():
        rel=Path(p).relative_to(frozen_root)
        require(sha_file(root/rel)==h,"frozen native producer changed: "+str(rel))
    require(f["pilot"]["rows"]==old["prototype"]["samples"] and f["pilot"]["rounds"]==2,"locked eight-row pilot")
    require(f["resources"]["pilot"]==old["prototype"]["proposed_gpu_limits"],"Pilot keeps original Smoke resource gates")
    require(f["resources"]["train"]["minimum_free_mib"]==10240,"formal GPU limit unchanged")
    f["_config_path"]=str(Path(path).resolve())
    return f

def derived_native(f,scope):
    require(scope in ("pilot","train"),"explicit scope")
    c=read_json(f["references"]["native_parent"]["path"])
    p=copy.deepcopy(c["prototype"]);old_p=copy.deepcopy(p)
    p["arms"]=["B"];p["rounds"]=f["pilot"]["rounds"] if scope=="pilot" else f["train_rounds"]
    p["samples"]=list(f["pilot"]["rows"] if scope=="pilot" else p["train_ids"])
    p["output_base"]=f["outputs"][scope]
    p["proposed_gpu_limits"]=f["resources"][scope]
    p["new_producers"]=dict(f["producers"])
    c["producers"]=dict(f["producers"])
    root=str(Path(__file__).resolve().parents[3]/"src")
    c["managed_roots"]=[root if x.endswith("idea-test-tbps-gh-probe-shared-20261023/src") else x for x in c["managed_roots"]]
    c["prototype"]=p;c["output_base"]=f["output_base"]
    c["smoke_output"]=f["outputs"][scope]
    c["probe_output"]=f["outputs"][scope]+"-UNAUTHORIZED-probe"
    gm={e["row_id"]:e for e in read_json(p["g_manifest"])["entries"] if e["role"]=="train"}
    hs=read_json(Path(p["history_root"])/"qualification/train.json")["entries"]
    eligible={e["identity"]["row_id"]:e["eligible"] for e in hs}
    k=sum(gm[row]["K"] for row in p["samples"]);r=p["rounds"];n=len(p["samples"])
    c["smoke"]={"rows":p["samples"],"max_G_training_updates":r*(k+n),
        "max_H_updates":r*sum(eligible[row] for row in p["samples"]),"full_bank_images":2048,
        "max_G_forwards":r*(k+n+2048),"max_IRRA_images":r*(k+n+2048),
        "max_JVPs":r*k,"max_H_forwards":r*sum(eligible[row] for row in p["samples"])*3,
        "max_reader_forwards":r*n,"max_IRRA_texts":r*(10*k+10*n)}
    c["condition_space_sha256"]=digest({"IRRA":p["source_space_sha256"],"reader":c["sources"]["reader"],
        "basis":c["sources"]["basis"],"byte_mapping":c["byte_alignment_policy"],
        "producer_sources":{q:h for q,h in c["producers"].items() if "/alternating/" in q}})
    p["condition_space_sha256"]=c["condition_space_sha256"]
    c["formal_binding"]={"configuration":ref(f["_config_path"]),"scope":scope,
        "name":"independent B formal controller; legacy smoke method names are internal only",
        "parent_native":f["references"]["native_parent"],"prior_exact_audit_not_reclassified":True}
    c["derivation"]={"parent_execution_commit":PARENT_COMMIT,"scope":scope,
        "allowed_changes":["independent B round/member schedule","new source identity","atomic round resume"],
        "math_changed":False,"original_prototype_sha256":digest(old_p)}
    return c

def verified_native(f,scope):
    path=f["native_configs"][scope]
    require(read_json(path)==derived_native(f,scope),"derived native configuration differs")
    return verify_config(path)

def full_members(c):
    """只按原文Canonical字段匹配；不做模糊回退或删除不兼容样本。"""
    p=c["prototype"]
    train=read_json(Path(p["history_root"])/"qualification/train.json")["entries"]
    gm={x["row_id"]:x for x in read_json(p["g_manifest"])["entries"] if x["role"]=="train"}
    require([e["identity"]["row_id"] for e in train]==p["train_ids"] and len(train)==2048,"full Train order")
    bad=[];total_k=0;distribution={}
    for e in train:
        row=e["identity"]["row_id"];g=gm[row]
        require(e["caption"]==g["caption"] and e["identity"]["pid"]==g["pid"],"full G/H identity")
        checked(FileRef(g["tensor_path"],g["tensor_sha256"]))
        checked(FileRef(g["metadata_path"],g["metadata_sha256"]))
        selected=read_json(g["metadata_path"])["source_selection"]["selected_attributes"]
        hattrs=[Anchor(**a) for a in e["attributes"]]
        require(len(selected)==g["K"],"K/selected cardinality")
        for a in selected:
            anchor=Anchor(**{k:a[k] for k in Anchor.__dataclass_fields__})
            anchor.check(e["caption"])
            if sum(anchor==ha for ha in hattrs)!=1:bad.append({"row":row,"anchor":asdict(anchor)})
        total_k+=g["K"];distribution[str(g["K"])]=distribution.get(str(g["K"]),0)+1
    require(not bad,"full Train ambiguous/unmatched canonical anchors: "+str(bad[:4]))
    return {"Train":2048,"eligible_H":sum(e["eligible"] for e in train),"G_K_total":total_k,
        "G_updates_per_round":2048+total_k,"K_distribution":distribution,"strict_anchor_failures":bad}

def validate_budget(f,members):
    r=f["train_rounds"];b=f["budget"];n=members["Train"];g=members["G_updates_per_round"];h=members["eligible_H"]
    wanted={"G_samples_per_round":n,"G_updates_per_round":g,"H_visits_per_round":n,
        "H_valid_updates_per_round":h,"gallery_images_per_round":n,"new_G_updates":g*r,
        "new_H_visits":n*r,"new_H_updates":h*r,"final_G_updates":484080+g*r,
        "final_H_visits":20480+n*r,"final_H_updates":20370+h*r,
        "full_gallery_refreshes":r,"gallery_images":n*r,"new_JVP_upper_bound":members["G_K_total"]*r}
    require(all(b[k]==v for k,v in wanted.items()),"declared budget differs from actual Train/round count")
    require(b["seed"]==42 and b["G_parent_absolute_epoch"]==60 and b["H_parent_absolute_epoch"]==10,
            "absolute epoch/seed changed")
    return wanted

def historical_receipts(f):
    """旧FAIL与新范围分开；不把原exact审计改判为PASS。"""
    for key,status in (("probe","GH_NATIVE_NUMERIC_PROBE_PASS"),("rng","CUDA_RNG_PASS")):
        mr=FileRef(**f["references"][key]);checked(mr)
        root=Path(mr.path).parent.parent
        store=ArtifactStore(root,new_base=root.parent);store.verify("completed")
        report=read_json(root/"completed/report.json")
        require(report["status"]==status,"original "+key+" completion")
    checked(FileRef(**f["references"]["old_smoke_report"]))
    require(read_json(f["references"]["old_smoke_report"]["path"])["status"]=="GH_NATIVE_TWO_ROUND_SMOKE_GENERATED",
            "prior native generated receipt")
    checked(FileRef(**f["references"]["old_smoke_audit_fail"]))
    return {"probe":"PASS original source only","CUDA_RNG":"PASS original parents",
        "old_smoke_native":"GENERATED","old_exact_audit":"FAIL preserved; not formal B acceptance"}

def source_preflight(path,scope):
    from .preflight import preflight
    f=load_formal(path);c=verified_native(f,scope)
    projection=copy.deepcopy(c["prototype"])
    projection["arms"]=["A1","B"];projection["rounds"]=2
    projection["samples"]=f["pilot"]["rows"]
    projection["output_base"]=f["output_base"]+"/source_audit_unused"
    legacy=preflight(projection)
    members=full_members(c);budget=validate_budget(f,members);receipts=historical_receipts(f)
    return f,c,{"status":"GH_B_FORMAL_SOURCE_PREFLIGHT_PASS","legacy_source_audit":legacy,
        "legacy_projection_sha256":digest(projection),"formal_native_source":source_key(c),
        "formal_config_sha256":sha_file(path),"scope":scope,"members":members,"budget":budget,"old_evidence":receipts,
        "model_forwards":0,"CUDA":False}

def prepare_cpu(path,scope):
    f,c,report=source_preflight(path,scope)
    import torch
    from tbps.materialization.irra_startup import CPUStartupGate
    from .native_sources import strict_import_preflight
    from .parents import restore_cpu_parents
    events=[];gate=CPUStartupGate(events.append)
    report["imports"]=strict_import_preflight(c)
    bundle=restore_cpu_parents(c)
    # 真正导入原Epoch排序函数并检查完整范围；仅临时ID对象，不加载图库或神经前向。
    from .native_sources import native_session
    from .parents import cpu_restore_only
    from .state import capture_rng,tree_sha
    from types import SimpleNamespace
    before=tree_sha(capture_rng(bundle.sampler))
    order_evidence={}
    with native_session(c,reuse=bundle.modules) as session:
        with cpu_restore_only(()):
            eg=session.import_module("epoch_training")
            ht=session.import_module("formal_training.training")
            gm=[e for e in read_json(c["prototype"]["g_manifest"])["entries"] if e["role"]=="train"]
            samples=[SimpleNamespace(row_id=row) for row in c["prototype"]["train_ids"]]
            for index in (1,f["train_rounds"]):
                g=[e["row_id"] for e in eg.epoch_order(gm,42,60+index)]
                h=ht.order(samples,10+index,42)[-2048:]
                require(len(g)==len(h)==2048 and set(g)==set(h)==set(c["prototype"]["train_ids"]),"absolute epoch member order")
                order_evidence[str(index)]={"G_absolute_epoch":60+index,"H_absolute_epoch":10+index,
                    "G_order_sha":digest(g),"H_order_sha":digest(h)}
    bundle.modules=session.modules
    require(before==tree_sha(capture_rng(bundle.sampler)),"schedule construction changed global RNG")
    report["native_orders"]=order_evidence
    gate.checked();gate.exit_checked()
    require(not torch.cuda.is_initialized(),"CPU preparation initialized CUDA")
    report["parent_restore"]=bundle.report;report["startup_events"]=events
    report["status"]="GH_B_FORMAL_CPU_PREFLIGHT_PASS"
    return f,c,bundle,gate,report

class FormalGrant:
    """独立权限，不复用probe_v2/Smoke的授权票据；旧API的action名称无新授权含义。"""
    action="smoke"
    def __init__(self,c,scope,*,allow_pilot=False,allow_training=False,ready=False):
        require(scope in ("pilot","train"),"grant scope")
        require(allow_pilot if scope=="pilot" else allow_training and ready,"explicit formal GPU authorization/READY required")
        require(c["formal_binding"]["scope"]==scope,"scope mismatch")
        self.key=source_key(c);self.scope=scope
    def validate(self,c,action):
        require(action=="smoke" and source_key(c)==self.key and c["formal_binding"]["scope"]==self.scope,
            "formal capability/source mismatch")
