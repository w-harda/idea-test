"""只读核实真实G60/H10/H60；只向新文档目录写派生证据，不重新审计训练。"""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from tbps.artifacts import FileRef,checked,sha_file,digest,json_bytes,read_json
from tbps.alternating.contracts import require
from tbps.alternating.state import tree_sha
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
G=Path("/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0060/checkpoint.pt")
H10=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoints/visit_00020480/checkpoint.pt")
H60=BASE/"training/checkpoints/visit_00122880/checkpoint.pt"
SHAS=["9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049",
"4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd",
"c0de26658c5f36083aaeb04dde3948cdd5bf941d30b1b5c97f7bb2c796404094"]
def ref(p):return {"path":str(p),"sha256":sha_file(p)}
def inspect():
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    from tbps.alternating.parents import cpu_restore_only
    paths=[G,H10,H60];before={str(p):[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino] for p in paths}
    for p,h in zip(paths,SHAS):checked(FileRef(str(p),h))
    with cpu_restore_only(paths):
        g,h10,h60=[torch.load(p,map_location="cpu",weights_only=False) for p in paths]
    protocol=read_json(BASE/"training/protocol.json")
    require(h60["protocol"]==protocol and h60["protocol_sha256"]==digest(protocol),"H60 actual protocol identity")
    require(protocol["parent_checkpoint_sha256"]==SHAS[1] and protocol["start_epoch"]==11
        and protocol["target_epochs"]==60,"H10 to H60 extension")
    require(g["completed_epoch"]==60 and h10["cursor"]==20480 and h10["updates"]==20370
        and h60["cursor"]==122880 and h60["updates"]==122220,"real checkpoint cursors")
    done=read_json(BASE/"training/completed.json");audit=read_json(BASE/"training_audit/result/report.json")
    require(done["status"]=="HV1_G60_TRAINING_PASS" and audit["status"]=="HV1_G60_TRAINING_CPU_AUDIT_PASS",
            "existing train/audit pass")
    require(done["last_checkpoint_sha256"]==SHAS[2] and done["attempt_new_updates"]==101850
        and done["attempt_new_visits"]==102400,"actual completed budget")
    require(set(audit["epochs"])==set(map(str,range(11,61)))
        and all(v=={"skipped":11,"updates":2037,"visits":2048} for v in audit["epochs"].values()),"50 epoch accounting")
    md=read_json(H60.with_name("checkpoint.json"))
    require(md["checkpoint_sha256"]==SHAS[2] and md["protocol_sha256"]==digest(protocol)
        and h60["sample_log_sha256"]==sha_file(H60.with_name("steps.jsonl")),"H60 metadata/log SHA")
    result={"status":"G60_H60_BASELINE_IDENTITY_AUDIT_PASS","G60":ref(G),"H10":ref(H10),"H60":ref(H60),
        "G":{"epoch":60,"cumulative_Adam":484080,"state":tree_sha(g["generator"]),
             "epoch_Train":2048,"epoch_Validation":128,"attribute_updates":6020,"joint_updates":2048},
        "H":{"initial_visits":20480,"initial_updates":20370,"new_epochs":[11,60],
             "new_visits":102400,"new_updates":101850,"final_visits":122880,"final_updates":122220,
             "state":tree_sha(h60["policy"]),"Adam":tree_sha(h60["optimizer"]),"RNG":tree_sha(h60["rng"]),
             "optimizer":h60["optimizer"]["param_groups"],"protocol":ref(BASE/"training/protocol.json"),
             "fixed_reward_gallery":protocol["reward_gallery"],"elapsed_seconds":done["elapsed_seconds"]},
        "validation":read_json(BASE/"validation/epoch_0060/completed/summary.json"),
        "official_Test_G60_H60":{"status":"NOT_FOUND","searched_roots":[str(BASE)],
            "reason":"仅找到128内部Validation；历史EpochSweep G60 Full绑定H7，不能冒充H60"},
        "historical_Test_G60_H7":read_json("/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full/epoch_0060/irra/completed/summary.json"),
        "sources":[ref(p) for p in [G,H10,H60,H60.with_name("checkpoint.json"),H60.with_name("steps.jsonl"),
            BASE/"training/completed.json",BASE/"training_audit/result/report.json",BASE/"training/protocol.json",
            BASE/"gallery_train/completed/manifest.json",BASE/"gallery_audit/result/report.json",
            BASE/"validation/epoch_0060/completed/summary.json"]],
        "native_budget":{"image":"RGB FP32 L∞<=8/255","text":"原H Natural-K gamma3 beta1; per-attribute/per-shared-word<=1 Unicode substitution"},
        "CUDA":torch.cuda.is_initialized(),"new_model_forwards":0,"new_optimizer_steps":0}
    require(not result["CUDA"] and before=={str(p):[p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino] for p in paths},
            "parent metadata/CUDA protection")
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    v=inspect()
    with (out/"BASELINE_AUDIT.json").open("xb") as f:f.write(json_bytes(v))
    print(v["status"])
if __name__=="__main__":main()
