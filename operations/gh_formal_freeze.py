"""开发期显式冻结新B协议；不会更新旧配置，不覆盖已存在的新协议或创建GPU Attempt。"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from tbps.artifacts import json_bytes,sha_file,read_json
from tbps.alternating.contracts import require
from tbps.alternating.formal_sources import ref,derived_native
OLD=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-probe-shared-20261023")
NATIVE=OLD/"configs/framework/gh_bidirectional_backend.probe_v2.shared.server.json"
OUT="/home/lzf/ldx/outputs/tbps-framework-gh-formal-b"
def freeze(rebind=False):
    cfg=ROOT/"configs/framework/gh_b_formal.server.json"
    native_paths={s:str(ROOT/f"configs/framework/gh_b_formal.{s}.native.server.json") for s in ("pilot","train")}
    exists=cfg.exists() or any(Path(p).exists() for p in native_paths.values())
    require(not exists or rebind,"new immutable configs already exist")
    for scope in ("pilot_v1","train_v1"):
        require(not Path(OUT,scope).exists(),"Attempt already consumed; no source rebinding")
    if exists:
        import shutil
        archive=Path("/home/lzf/ldx/tmp/gh_formal_b_cpu_20261024/config_revisions")
        n=len(list(archive.glob("v*")))+1;dest=archive/f"v{n}";dest.mkdir(parents=True)
        for p in [cfg,*map(Path,native_paths.values())]:
            if p.exists():shutil.copy2(p,dest/p.name)
    old=read_json(NATIVE)
    b=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
    original=Path("/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend")
    f={"schema":"gh-b-formal-1","production_commit":"a7597f9c964dda0c8c48973e4678f0b9d3fa095a",
       "arm":"B","train_rounds":50,"training_authorized":False,
       "pilot":{"rows":old["prototype"]["samples"],"rounds":2,"not_formal_training":True},
       "output_base":OUT,"outputs":{"pilot":OUT+"/pilot_v1","train":OUT+"/train_v1"},
       "controls":{"pilot":"/home/lzf/ldx/tmp/gh_formal_b_pilot_v1","train":"/home/lzf/ldx/tmp/gh_formal_b_train_v1"},
       "audits":{"pilot":OUT+"/pilot_cpu_audit_v1","train":OUT+"/train_cpu_audit_v1"},
       "ready_receipt":OUT+"/readiness/ready.json","native_configs":native_paths,
       "resources":{"pilot":old["prototype"]["proposed_gpu_limits"],
          "train":{"minimum_free_mib":10240,"minimum_ram_bytes":32*1024**3,
                   "minimum_disk_bytes":250*1024**3,"minimum_free_inodes":2000000}},
       "evaluation":{"scope":"official_test_after_training_only","authorized":False,"query_chunk":128,
          "precision":"fp32/math/no_tf32","output_base":OUT+"/unified_irra_evaluation_v1",
          "baseline":"true G60 + adapted H60 visit122880","modes":["Clean","Image-only","Text-only","Joint"],
          "ASR_definition":"Clean-correct Queries that become attack Top1 wrong / Clean-correct Queries",
          "Test_checkpoint_selection":False},
       "budget":{"G_parent_absolute_epoch":60,"H_parent_absolute_epoch":10,"seed":42,
          "G_samples_per_round":2048,"G_updates_per_round":8068,"H_visits_per_round":2048,
          "H_valid_updates_per_round":2037,"gallery_images_per_round":2048,
          "new_G_updates":403400,"new_H_visits":102400,"new_H_updates":101850,
          "final_G_updates":887480,"final_H_visits":122880,"final_H_updates":122220,
          "full_gallery_refreshes":50,"gallery_images":102400,"new_JVP_upper_bound":301000,
          "compute_matched_to_fixed_G_baseline":False,
          "reason":"H adaptation budget matched; B additionally trains G and refreshes all Train banks"},
       "references":{
          "native_parent":ref(NATIVE),"H60":ref(b/"training/checkpoints/visit_00122880/checkpoint.pt"),
          "H60_training_protocol":ref(b/"training/protocol.json"),
          "H60_training_complete":ref(b/"training/completed.json"),
          "H60_training_audit":ref(b/"training_audit/result/report.json"),
          "baseline_identity":ref(ROOT/"docs/framework/results/gh_formal_b/BASELINE_AUDIT.json"),
          "probe":ref(original/"probe_v2/completed/manifest.json"),
          "rng":ref(original/"cuda_rng_v1/completed/manifest.json"),
          "old_smoke_report":ref(original/"smoke_v2/completed/report.json"),
          "old_smoke_audit_fail":ref(Path("/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023/cpu_audit.log")),
          "history_index":ref(OLD/"docs/framework/results/gh_probe_shared/CPU_RECEIPT_AUDIT.json")},
       "tools":{str(p):sha_file(p) for p in [Path(__file__),ROOT/"operations/gh_formal_baseline_audit.py",
                 ROOT/"operations/run_gh_formal_b.sh",ROOT/"operations/gh_formal_b.py"]},
       "producers":{str(p):sha_file(p) for p in (ROOT/"src/tbps").rglob("*.py")}}
    cfg.write_bytes(json_bytes(f));f["_config_path"]=str(cfg)
    for scope,p in native_paths.items():Path(p).write_bytes(json_bytes(derived_native(f,scope)))
    print(json.dumps({"formal":ref(cfg),"native":{s:ref(p) for s,p in native_paths.items()}},indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--write-new",action="store_true");ap.add_argument("--rebind-unconsumed",action="store_true");a=ap.parse_args()
    require(a.write_new,"explicit development freeze required");freeze(rebind=a.rebind_unconsumed)
