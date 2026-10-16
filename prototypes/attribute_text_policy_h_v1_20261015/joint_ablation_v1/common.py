"""冻结Phase5输入，不重算正式结果、不重新选H或运行图像模型。"""
from pathlib import Path
import copy
import torch
from phase2_sources import ROOT as HROOT,digest,sha,checked,read_json
from formal_training.common import immutable_json,run_lock,Stop,cpu_only
from phase5_joint.common import OUTPUT as FULL,legacy
from phase5_joint.storage import Store
from phase5_joint.evaluate import audit as audit_full
HERE=Path(__file__).resolve().parent
OUTPUT=HROOT/"outputs/joint_ablation_v1/full"
SOURCES=("common.py","strategies.py","backend.py","storage.py","evaluate.py","config.json")
def safe_output(path):
    allowed=(HROOT/"outputs").resolve()/"joint_ablation_v1/full"
    if Path(path).resolve()!=allowed:raise ValueError("ablation outputs must be isolated from all history")
    return allowed
def config():
    c=read_json(HERE/"config.json")
    if (c["schema"]!="hv1-joint-ablation-config-1" or c["random_seeds"]!=[42,43,44,45,46]
            or c["untrained_seed"]!=42 or c["search_max_nodes"]!=256 or c["sample_standard_deviation_ddof"]!=1):
        raise ValueError("committed ablation seeds/search/statistical definition changed")
    return c
def trials(c):
    return [{"key":kind+"_seed"+str(seed),"kind":kind,"seed":seed}
            for kind in ("random_attribute","h_selected") for seed in c["random_seeds"]]+[
            {"key":"untrained_seed42","kind":"untrained","seed":42}]
def pins():
    c=config();p=dict(c["frozen_sources"])
    p.update({str(HERE/n):sha(HERE/n) for n in SOURCES})
    for name,h in p.items():checked(name,h)
    return p
def verify(protocol):
    if protocol["execution_backend"]=="official_irra_real" and protocol["config"]!=config():
        raise ValueError("real ablation configuration differs from committed fixed definition")
    for name,h in protocol["source_sha256"].items():checked(name,h)
    for name,h in protocol["input_files_sha256"].items():checked(name,h)
    if protocol["trials"]!=trials(protocol["config"]):raise ValueError("trial definitions changed")
    if protocol["Test_selection"] is not False:raise ValueError("Test model selection forbidden")

def full_inputs():
    """CPU只核验正式Full及回执SHA；不会重新计算Full指标或执行任何模型。"""
    p=read_json(FULL/"protocol.json")
    from contextlib import redirect_stdout
    import io
    with redirect_stdout(io.StringIO()):summary=audit_full(FULL,p,recompute=False)
    baseline=read_json(FULL/"baseline_verified.json")
    checked(FULL/"image_reuse.json",baseline["image_reuse_sha256"])
    reuse=read_json(FULL/"image_reuse.json")
    if (not 0<=reuse["maximum_linf"]<=8/255
            or reuse["G_historical_samples"]!=3074 or reuse["IRRA_image_historical_samples"]!=3074
            or reuse["G_new_samples"]!=0 or reuse["IRRA_image_new_samples"]!=0
            or summary["all_queries_retained"] is not True):
        raise ValueError("frozen Gallery budget/coverage/retention evidence mismatch")
    store=Store(FULL,p)
    arrays={};proof={}
    for family in ("clean_gallery","image_gallery","clean_text","adv_text"):
        x,stats,files=store.gather(family);arrays[family]=x;proof.update({str(FULL/n):h for n,h in files.items()})
        if family=="adv_text":
            actions=[a for s in stats for a in s["actions"]]
    if len(actions)!=6156 or (len(p["gallery"]),len(p["queries"]))!=(3074,6156):
        raise ValueError("frozen Full coverage changed")
    for n in ("protocol.json","summary.json","completed.json","metrics.csv","baseline_verified.json","image_reuse.json"):
        proof[str(FULL/n)]=sha(FULL/n)
    proof.update(p["source_sha256"]);proof.update(p["evidence_sha256"])
    for name,h in read_json(FULL/"completed.json")["attempt_files_sha256"].items():
        proof[str(FULL/name)]=h
    return p,summary,arrays,actions,proof

def preflight(text_batch=16):
    if type(text_batch)is not int or not 1<=text_batch<=64:raise ValueError("text batch 1..64")
    c=config();source=pins()
    p,summary,arrays,actions,proof=full_inputs()
    from tokenizer_probe import official_probe
    from text_frontend import CaptionFrontend
    from .strategies import validate_reference,build_untrained,reference_statistics
    from feedback_backend import state_sha
    probe,frontend=official_probe(),CaptionFrontend()
    for q,action in zip(p["queries"],actions):validate_reference(q["caption"],action,probe,frontend)
    initial=state_sha(build_untrained(p["H_selection"]["architecture"],probe))
    cpu_only()
    return {"schema":"hv1-joint-ablation-1","execution_backend":"official_irra_real","config":c,
        "trials":trials(c),"gallery":p["gallery"],"queries":p["queries"],"text_batch":text_batch,
        "gallery_count":3074,"query_count":6156,"architecture":p["H_selection"]["architecture"],"untrained_initial_state_sha256":initial,
        "model_sha256":p["model_sha256"],"full_protocol_sha256":digest(p),
        "source_sha256":source,"input_files_sha256":proof,"Test_selection":False,
        "fixed_baseline_groups":summary["groups"],"learned_edit_statistics":reference_statistics(actions),"seed_scope":"SHA256(seed|kind|Caption); no PID/pair/feedback",
        "runtime":{"torch":str(torch.__version__),"python":__import__("sys").version,"cpu_threads":torch.get_num_threads()},
        "G_new_forwards":0,"IRRA_image_new_forwards":0,
        "reuse":"Phase5 all Gallery/Clean-text/features; learned H result without new H/IRRA forward"}

def load_inputs(protocol):
    p=read_json(FULL/"protocol.json")
    if digest(p)!=protocol["full_protocol_sha256"]:raise ValueError("frozen Full protocol changed")
    store=Store(FULL,p)
    arrays={};actions=[]
    for family in ("clean_gallery","image_gallery","clean_text","adv_text"):
        x,stats,_=store.gather(family);arrays[family]=x
        if family=="adv_text":actions=[a for s in stats for a in s["actions"]]
    return arrays,actions

def trial_protocol(parent,trial):
    # 兼容已冻结Phase5原子Store，仅使用adv_text family；不复制任何图库。
    return {"schema":"hv1-ablation-text-trial-1","parent_sha256":digest(parent),"trial":trial,
            "gallery":parent["gallery"],"queries":parent["queries"],"text_batch":parent["text_batch"]}
