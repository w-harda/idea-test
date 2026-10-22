"""独立审计版本入口：原生产者仍从冻结执行树加载；禁止GPU或训练重入。"""
from pathlib import Path
import argparse,importlib.util,json,os,sys,subprocess

OLD=Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021")
ROOT=Path(__file__).resolve().parents[3]
# 必须先选旧包，不能让新树的__file__改变113项来源身份。
sys.path.insert(0,str(OLD/"src"))
from tbps.artifacts import ArtifactStore,FileRef,checked,sha_file,read_json,json_bytes
from tbps.ablations.hv1_g60_state import config,preflight,BASE,ref,need,unchanged
import torch

def rename_only(text):
    return (text.replace('stat=per_epoch.setdefault','epoch_stat=per_epoch.setdefault')
                .replace('stat["visits"]','epoch_stat["visits"]')
                .replace('stat["updates"]','epoch_stat["updates"]')
                .replace('stat["skipped"]','epoch_stat["skipped"]'))

def load_fixed(path):
    spec=importlib.util.spec_from_file_location("tbps.ablations.hv1_g60_fixed_audit",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def verify_binding(path):
    b=read_json(path,"hv1-g60-independent-auditor-1")
    need(Path(b["execution_root"])==OLD and Path(b["repair_root"])==ROOT,"explicit runtime roots")
    need(b["execution_commit"]=="9765a536a11a9a0e9111f2f834230c53d41bf2d8"
         and b["snapshot_head"]=="1b16d653d17573d3dafaa7a26bd84bf8d31949ac","fixed original commits")
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=OLD,text=True).strip()
    need(head==b["snapshot_head"],"frozen execution tree HEAD")
    for x in (b["configuration"],b["old_auditor"],*b["repair_sources"],*b["failure_evidence"]):checked(FileRef(**x))
    expected={str(ROOT/x) for x in ("src/tbps/ablations/hv1_g60_audit.py",
        "src/tbps/ablations/hv1_g60_audit_repair.py","src/tbps/ablations/run_hv1_g60_adaptation_repaired.sh",
        "src/tbps/ablations/hv1_g60_repaired_results.py","tests/framework/test_hv1_g60_audit_repair.py")}
    need({x["path"] for x in b["repair_sources"]}==expected,"complete independent repair sources")
    need(b["old_auditor"]["path"]==str(OLD/"src/tbps/ablations/hv1_g60_audit.py"),"original auditor path")
    original=Path(b["old_auditor"]["path"]).read_text()
    tracked=subprocess.check_output(["git","show",b["execution_commit"]+":src/tbps/ablations/hv1_g60_audit.py"],cwd=OLD).decode()
    need(original==tracked,"original audit bytes at execution commit")
    fixed=ROOT/"src/tbps/ablations/hv1_g60_audit.py"
    need(fixed.read_text()==rename_only(original) and fixed.read_text()!=original,"only stat -> epoch_stat allowed")
    need('need(stat(p)==expected' in fixed.read_text(),"historical stat guard retained")
    c=config(b["configuration"]["path"])
    need(len(c["producers"])==113,"all frozen producers retained")
    need(Path(sys.modules["tbps.ablations.hv1_g60_state"].__file__).resolve().is_relative_to(OLD),
         "original runtime namespace required")
    return b,c

def sources_for(binding,b):
    return tuple([ref(binding),*map(lambda x:FileRef(**x),
        [b["configuration"],b["old_auditor"],*b["repair_sources"],*b["failure_evidence"]])])

def smoke_gate(binding):
    b,c=verify_binding(binding)
    store=ArtifactStore(BASE/"audit_repair_v1",new_base=BASE)
    store.verify("smoke")
    proof=read_json(BASE/"audit_repair_v1/smoke/report.json")
    need(proof["status"]=="HV1_G60_SMOKE_CPU_AUDIT_PASS"
         and proof["binding_sha256"]==sha_file(binding),"new independent auditor provenance")
    ArtifactStore(BASE/"smoke_audit",new_base=BASE).verify("result")
    ArtifactStore(BASE/"gallery_audit",new_base=BASE).verify("result")
    ArtifactStore(BASE/"gallery_train",new_base=BASE).verify("completed")
    need(not (BASE/"training").exists(),"formal training directory must be fresh")
    return b,c

def retained_snapshot():
    from tbps.ablations.hv1_g60_state import stat
    return {str(p):stat(p) for name in ("gallery_train","gallery_audit","smoke_v1")
            for p in (BASE/name).rglob("*") if p.is_file()}

def verify_retained():
    store=ArtifactStore(BASE/"gallery_train",new_base=BASE)
    need(not store.pending(),"retained gallery pending")
    dirs={p.name for p in (BASE/"gallery_train").iterdir() if p.is_dir()}
    expected={"protocol","completed"}|{f"{kind}-{i:04d}" for kind in ("claim","image") for i in range(2048)}
    need(dirs==expected,"exact 2048 gallery Claim/image pairs")
    for name in sorted(dirs):store.verify(name)
    ArtifactStore(BASE/"gallery_audit",new_base=BASE).verify("result")
    need(not (BASE/"smoke_v1/pending").exists(),"retained Smoke pending")

def run_audit(binding,smoke):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","explicit CPU-only environment")
    b,c=verify_binding(binding)
    preserved=retained_snapshot();verify_retained()
    ctx=preflight(c,check_restore=True)
    # 原G60图库完成回执、内容SHA和旧Clean来源继续严格验证；不重新生成/编码。
    from tbps.ablations.hv1_g60_adaptation import load_bank
    bank=load_bank(ctx);del bank
    root=BASE/("smoke_v1" if smoke else "training")
    need(not (root/"pending").exists(),"unfinished training transaction")
    fixed=load_fixed(ROOT/"src/tbps/ablations/hv1_g60_audit.py")
    fixed.audit_training(ctx,smoke=smoke)
    need(retained_snapshot()==preserved,"retained gallery/Smoke metadata unchanged")
    # 原审计结果与新审计器版本另行绑定，不改原执行协议或旧FAIL。
    auditroot=BASE/("smoke_audit" if smoke else "training_audit")
    ArtifactStore(auditroot,new_base=BASE).verify("result")
    report=read_json(auditroot/"result/report.json")
    result={**report,"execution_commit":b["execution_commit"],"original_snapshot_head":b["snapshot_head"],
        "repair_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "binding_sha256":sha_file(binding),"auditor_sha256":sha_file(ROOT/"src/tbps/ablations/hv1_g60_audit.py"),
        "original_producers":113,"old_FAIL_preserved":True,"retained_files_unchanged":len(preserved),"CPU_only":not torch.cuda.is_initialized()}
    need(result["CPU_only"],"CPU audit initialized CUDA")
    sources=sources_for(binding,b)+(ref(auditroot/"result/manifest.json"),ref(root/"completed.json"),
        ref(BASE/"gallery_audit/result/manifest.json"))
    ArtifactStore(BASE/"audit_repair_v1",new_base=BASE).publish("smoke" if smoke else "training",
        {"report.json":json_bytes(result)},{"kind":"independent-auditor-version-binding"},sources=sources)
    print("HV1_G60_VERSIONED_CPU_AUDIT_PASS",json.dumps(result),flush=True)

def launch_gate(binding):
    need(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CPU gate environment")
    b,c=smoke_gate(binding)
    ctx=preflight(c,check_restore=True)
    from tbps.ablations.hv1_g60_adaptation import load_bank,resource_check
    bank=load_bank(ctx);del bank
    resources=resource_check(c)
    unchanged(ctx["report"])
    protected=read_json(c["sources"]["protection_snapshot"]["path"])
    from tbps.ablations.hv1_g60_state import stat
    for p,s in protected.items():need(stat(p)==s,"historical protection changed: "+p)
    need(not torch.cuda.is_initialized(),"CPU launch gate must not initialize CUDA")
    print("HV1_G60_REPAIRED_LAUNCH_GATE_PASS",json.dumps({"parent_restore":ctx["report"]["parent_restore"],
        "resources":resources,"protected":len(protected),"producers":113,"CUDA":False}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--binding",required=True)
    p.add_argument("--action",choices=("audit-smoke","audit-training","launch-gate"),required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.action=="launch-gate":launch_gate(a.binding)
    else:run_audit(a.binding,a.action=="audit-smoke")
if __name__=="__main__":main()
