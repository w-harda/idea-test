"""G官方test只读协议与指标；不加载victim，不重评分。"""
import importlib.util
import json
import sys
from pathlib import Path
from collections import Counter
import torch
import common as c
import checkpoint_io as ck
from generator_model import ConditionalGenerator,generate_once

FULL=c.REPO/"prototypes/attribute_selection_full_test_20261011"
REFERENCE=FULL/"outputs/full"
CHECKPOINT=c.ROOT/"outputs/train/last.pt"
TRAIN_CACHE=c.ROOT/"outputs/conditions"
sys.path.insert(0,str(FULL))
import core as official
sys.path.insert(0,str(c.REPO/"src"))
from attributes.tbps_retrieval_metrics import text_to_image_metrics,aggregate_rows
sys.path.insert(0,str(c.REPO/"prototypes/attribute_selection_pid_metrics_20261011"))
from pid_metrics import drop_rate

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    return obj

def normalized(t,n):
    if t.shape!=(n,512) or t.dtype!=torch.float32 or not torch.isfinite(t).all() or not torch.allclose(t.norm(dim=1),torch.ones(n),atol=2e-4,rtol=0):
        raise ValueError("invalid normalized CPU feature")

def check_checkpoint():
    cp,side,file=ck.resolve_checkpoint(CHECKPOINT,"cpu")
    done=c.load(c.ROOT/"outputs/train/completed.json")
    manifest=c.checked_cache(TRAIN_CACHE)
    if not done["method_audit_passed"] or done["epochs"]!=10 or cp["epoch"]!=10 or cp["scope"]!="training":
        raise ValueError("completed final epoch10 required")
    if side["sha256"]!=done["checkpoint_sha256"] or cp["cache_manifest_sha256"]!=done["cache_manifest_sha256"] or cp["cache_manifest_sha256"]!=c.sha(TRAIN_CACHE/"manifest.json"):
        raise ValueError("trained checkpoint/cache SHA")
    if cp["config"]!=c.load(c.ROOT/"config.json") or not done["test_used"] is False:
        raise ValueError("trained config/test provenance")
    if not all(torch.isfinite(t).all() for t in cp["model"].values()):raise ValueError("nonfinite G parameters")
    c.verify(cp["inference_source_sha256"]);c.verify(manifest["frozen_sha256"])
    return cp,side,file,manifest

def validate_reference(protocol,m):
    if protocol["scope"]!="full" or protocol["score_gallery"]!="clean official test gallery; paired index of conditioned source image":
        raise ValueError("unknown score reference gallery")
    for key,value in (("gamma",3),("beta",1),("kmax",None),("tau",.02),("precision","fp32")):
        if protocol[key]!=value:raise ValueError("frozen test score setting: "+key)
    if (m["gallery_count"],m["query_count"])!=(3074,6156):raise ValueError("official test coverage")
    if len({r["image"] for r in m["gallery"]})!=3074:raise ValueError("nonunique gallery")
    for g in m["gallery"]:
        if g["caption_index"]!=0 or not g["row_id"].endswith(":0"):raise ValueError("not first annotation caption")
    for q in m["queries"]:
        if m["gallery"][q["paired_index"]]["pid"]!=q["pid"]:raise ValueError("query/PID mapping")

def validate_record(saved,entry,row,protocol,select_fn):
    official.validate_score(saved["record"],saved["payload"],row,select_fn)
    if saved["protocol_sha256"]!=c.identity(protocol) or saved["image_sha256"]!=entry["image_sha256"]:
        raise ValueError("score source protocol/image")
    if saved["record"]["gallery_scope"]!="clean test shared gallery":raise ValueError("not test scores")
    if saved["payload"]["selection_ids"]!=entry["selection_ids"] or saved["payload"]["natural_k"]!=entry["natural_k"]:
        raise ValueError("saved test Top-K differs")

def audit():
    cp,side,file,train=check_checkpoint()
    old=module("g_official_test_run",FULL/"run.py")
    m,labels=old.dataset()
    protocol=c.load(REFERENCE/"protocol.json")
    manifest=old.verify_prepared(REFERENCE,m,protocol)
    completed=c.load(REFERENCE/"completed.json")
    official.require_complete(completed,m,"full")
    validate_reference(protocol,m)
    c.verify(protocol["frozen_sha256"])
    shared=set(train["frozen_sha256"])&set(protocol["frozen_sha256"])
    if any(train["frozen_sha256"][p]!=protocol["frozen_sha256"][p] for p in shared):
        raise ValueError("trained/test reader/basis/source mismatch")
    essential=[c.REPO/"prototypes/attribute_unit_v4_learned_reader_20261003/outputs/reader_final.pt",
        c.REPO/"prototypes/attribute_unit_v55_interface_20261003/outputs/full/semantic_bases.json",
        Path("/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth")]
    if not set(map(str,essential))<=shared:raise ValueError("missing shared reader/basis/IRRA SHA")
    sys.path.insert(0,str(c.REPO/"prototypes/attribute_tta_step74_dynamic_k_20261004"))
    from selection_audit import selection
    statuses=Counter()
    for row,entry in zip(m["gallery"],manifest["rows"]):
        saved=c.load(entry["score_path"]);validate_record(saved,entry,row,protocol,selection)
        statuses.update(a["status"] for a in saved["record"]["attributes"])
    source=c.load(Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/prototype.json"))
    train_paths=set(source["gallery_spec"]["gallery_paths"])
    if train_paths&{g["image"] for g in m["gallery"]}:raise ValueError("test image appears in train reference")
    train_pids={e["pid"] for role in ("train","validation") for e in train[role]}
    if train_pids&{g["pid"] for g in m["gallery"]}:raise ValueError("G train/test PID leakage")
    locks={**train["frozen_sha256"],**protocol["frozen_sha256"]}
    for path in [file,Path(str(file)+".json"),c.ROOT/"outputs/train/completed.json",
                 c.ROOT/"outputs/train/protocol.json",TRAIN_CACHE/"manifest.json",TRAIN_CACHE/"completed.json",
                 *[REFERENCE/n for n in ("protocol.json","selection_manifest.json","prepare_completed.json","completed.json","clean_features.pt","summary.json")],
                 c.REPO/"src/attributes/tbps_retrieval_metrics.py",
                 c.REPO/"prototypes/attribute_selection_pid_metrics_20261011/pid_metrics.py",
                 *sorted(c.ROOT.glob("g_test*.py"))]:
        locks[str(path)]=c.sha(path)
    locks.update({e["score_path"]:e["score_sha256"] for e in manifest["rows"]})
    report={"method_audit_passed":True,"checkpoint_epoch":10,"checkpoint_sha256":side["sha256"],
        "gallery_count":3074,"query_count":6156,"score_cache_reused":3074,"new_scoring_backward":0,
        "test_score_reference_gallery":3074,"training_score_reference_gallery":34054,
        "test_images_in_train_reference":0,"paired_index_exact":True,"first_caption_exact":True,
        "reader_basis_IRRA_common_SHA":len(shared),"instance_status_counts":dict(statuses),
        "K_distribution":manifest["inventory"]["K_distribution"],
        "K0_count":manifest["inventory"]["K0_clean_passthrough"],
        "test_G_K0_behavior":"global-conditioned G forward; not TTA clean passthrough",
        "clean_expected":c.load(REFERENCE/"summary.json")["clean"],
        "reference_change":"existing full-test protocol reused; no new score formula/gallery substitution",
        "conditioning_threat_model":"known first caption plus offline surrogate gradient-derived condition"}
    return report,m,manifest,locks

def scoped(m,scope):
    if scope=="full":return m
    # 固定先遇到的两个不同PID，不按K/score/攻击效果选样本。
    selected=[];pids=set()
    for g in m["gallery"]:
        if g["pid"] not in pids:selected.append(g);pids.add(g["pid"])
        if len(selected)==2:break
    indices={g["index"] for g in selected}
    queries=[q for q in m["queries"] if q["paired_index"] in indices]
    return {**m,"gallery":selected,"queries":queries,"gallery_count":len(selected),"query_count":len(queries)}

def make_protocol(report,m,locks,scope):
    return {"schema":"G-official-test-1","scope":scope,"gallery_count":m["gallery_count"],"query_count":m["query_count"],
        "checkpoint_sha256":report["checkpoint_sha256"],"checkpoint_epoch":10,
        "trained_cache_manifest_sha256":c.sha(TRAIN_CACHE/"manifest.json"),
        "query_set_sha256":c.identity(m["queries"]),"gallery_layout_sha256":c.identity(m["gallery"]),
        "source_sha256":locks,"reference_score_manifest":str(REFERENCE/"selection_manifest.json"),
        "reference_score_gallery":"3074 clean official test images; exact cached paired index",
        "training_score_gallery":"34054 clean train images; test has no paired positive there",
        "score_formula":"frozen S_raw=||Q^T w^T G||2; exact old test scores",
        "gamma":3,"beta":1,"kmax":None,"tau":.02,"epsilon":8/255,
        "condition_formula":"unchanged aggregate_condition / original IRRA TextTransport JVP",
        "condition_rule":m["conditioning_rule"],"G_forwards_per_image":1,"K0":"global-only G",
        "new_score_backward":0,"no_training":True,"test_time_optimization_steps":0,
        "generation_backward":0,"generation_JVP":0,"victim":"IRRA-official-CUHK-fp32",
        "relevance":"all same-PID positives; text-to-image; no camera filtering",
        "formal_test_evaluation":scope=="full","shared_adversarial_gallery":True}

def checked_conditions(out,protocol):
    done=c.load(out/"prepare_completed.json");manifest=c.load(out/"conditions/manifest.json")
    if not done["method_audit_passed"] or done["manifest_sha256"]!=c.sha(out/"conditions/manifest.json"):
        raise ValueError("incomplete/modified test conditions")
    if manifest["protocol_sha256"]!=c.identity(protocol) or len(manifest["entries"])!=protocol["gallery_count"]:
        raise ValueError("condition scope/protocol")
    if len({e["index"] for e in manifest["entries"]})!=len(manifest["entries"]):raise ValueError("duplicate condition")
    reference=c.load(REFERENCE/"selection_manifest.json")
    expected=scoped(reference,protocol["scope"])
    if c.identity(expected["gallery"])!=protocol["gallery_layout_sha256"] or c.identity(expected["queries"])!=protocol["query_set_sha256"]:
        raise ValueError("official test layout changed")
    if [e["index"] for e in manifest["entries"]]!=[r["index"] for r in expected["gallery"]]:
        raise ValueError("test condition order/coverage")
    for e,row in zip(manifest["entries"],expected["gallery"]):
        condition=c.read_condition(e);metadata=c.load(e["metadata_path"])
        validate_condition_binding(e,metadata,protocol,row,reference["rows"][e["index"]],condition)
    return manifest

def validate_condition_binding(entry,metadata,protocol,row,source,condition):
    keys=("index","row_id","pid","image","caption")
    if any(entry[k]!=row[k] or metadata[k]!=row[k] for k in keys):raise ValueError("test condition identity")
    if metadata["protocol_sha256"]!=c.identity(protocol) or metadata["tensor_sha256"]!=entry["tensor_sha256"]:
        raise ValueError("condition metadata/tensor binding")
    if metadata["score_sha256"]!=source["score_sha256"] or entry["score_sha256"]!=source["score_sha256"] or metadata["score_path"]!=source["score_path"]:
        raise ValueError("condition score source")
    if entry["image_sha256"]!=source["image_sha256"] or metadata["image_sha256"]!=source["image_sha256"]:
        raise ValueError("condition image source")
    if entry["K"]!=source["natural_k"] or metadata["selected_instance_ids"]!=source["selection_ids"]["S_TOPK"]:
        raise ValueError("condition frozen Top-K")
    if bool(condition["has_local"])!=(entry["K"]>0):raise ValueError("K0/local binding")
    weights=metadata["score_normalized_weights"]
    if len(weights)!=entry["K"] or (entry["K"] and abs(sum(weights)-1)>1e-12):
        raise ValueError("frozen score aggregation weights")
    if metadata["source_scope"]!="official test score cache":raise ValueError("train/validation conditions forbidden")

def load_generator(protocol,device):
    cp,side,_,_=check_checkpoint()
    if side["sha256"]!=protocol["checkpoint_sha256"] or cp["cache_manifest_sha256"]!=protocol["trained_cache_manifest_sha256"]:
        raise ValueError("G/test protocol binding")
    g=ConditionalGenerator(**cp["architecture"]).to(device)
    g.load_state_dict(cp["model"],strict=True);g.eval().requires_grad_(False)
    return g

def condition_entry(manifest,row_id):
    e=next((e for e in manifest["entries"] if e["row_id"]==row_id),None)
    if e is None:raise ValueError("uncached test caption; no online scoring fallback")
    return e

def metrics_for(query,features,queries,gallery):
    normalized(query,len(queries));normalized(features,len(gallery))
    allrows=[]
    for start in range(0,len(queries),128):
        scores=query[start:start+128]@features.T
        _,rows=text_to_image_metrics(scores,[q["pid"] for q in queries[start:start+128]],[g["pid"] for g in gallery])
        allrows.extend({**q,**r} for q,r in zip(queries[start:start+128],rows))
    return aggregate_rows(allrows),allrows

def assert_clean_match(actual,expected):
    errors={k:abs(actual[k]-expected[k]) for k in ("mAP","R@1","R@5","R@10","mINP")}
    if any(v>1e-8 for v in errors.values()):raise ValueError("clean baseline mismatch: "+json.dumps(errors))
    return errors

def validate_archive(value,clean,entry):
    for key in ("pixels","delta","effective_delta"):
        x=value[key]
        if x.shape!=clean.shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():raise ValueError("archive pixels/dtype")
    if value["pixels"].min()<0 or value["pixels"].max()>1:raise ValueError("image range")
    if not torch.equal(value["pixels"]-clean,value["effective_delta"]):raise ValueError("effective delta mismatch")
    if not torch.equal((clean+value["delta"]).clamp(0,1),value["pixels"]):raise ValueError("G output composition mismatch")
    budget=float(value["effective_delta"].abs().max())
    if max(budget,float(value["delta"].abs().max()))>8/255+1e-6:raise ValueError("budget")
    normalized(value["IRRA_feature"],1)
    return budget
