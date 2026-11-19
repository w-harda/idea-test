"""显式GPU接线验证：克隆G/H状态的小样本，不提交正式训练账本。"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path("/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116")
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116")
sys.path.insert(0,str(SOURCE/"src"))
import tbps
tbps.__path__.append(str(ROOT/"src/tbps"))
import torch
from tbps.performance.runtime import install,CONFIG,MANIFEST
from tbps.artifacts import read_json,sha_file,checked,FileRef
from tbps.alternating.state import tree_sha
from tbps.models.irra import fp32_runtime
def main():
 if "--allow-gpu" not in sys.argv:raise ValueError("explicit GPU validation required")
 identity=install()
 from tbps.attribute_response.sources import preflight,Grant
 from tbps.attribute_response.backend import loaded
 f,c,bundle,gate,proof=preflight(CONFIG,"train");gate.initialize_cuda()
 archive=BASE/"recovery_evidence/20261119/committed_round30"
 statefile=archive/"boundaries/round_0030/state.pt"
 checked(FileRef(str(statefile),read_json(statefile.parent/"manifest.json")["files_sha256"]["state.pt"]))
 state=torch.load(statefile,map_location="cpu",weights_only=False)
 validation=BASE/"performance_validation/integration_v1"
 if validation.exists():raise RuntimeError("integration scope consumed; inspect before another attempt")
 result={"manifest_sha256":sha_file(MANIFEST),"test_source_sha256":sha_file(Path(__file__)),
   "historical_artifacts_written":False,"historical_JVP_recomputed":0,"not_formal_training":True}
 with fp32_runtime(),loaded(c,bundle,Grant(c,True)) as b:
  b.load_reward_inputs()
  assert hasattr(b,"_performance_images")
  b.resume_ledger=read_json(BASE/"train/ledgers/round_0030/ledger.json")
  b.root=archive;b.restore_complete(state)
  row=next(r for r in b.p["samples"] if b.original[r].payload["K"]>0)
  b.root=validation/"G_clone";b.begin_round(31)
  b.enter_phase_rng("G");before=b.h_fingerprint()
  b.probe_cursor=1;b.response_visits["G"]=1
  old_steps=b.cursor["g_updates"];raw=b.g_update(row,b.original[row])
  assert b.h_fingerprint()==before
  assert raw["optimizer_steps"]==b.original[row].payload["K"]+1
  assert b.cursor["g_updates"]-old_steps==raw["optimizer_steps"]
  assert all(x["finite_gradients"] and x["cumulative_linf"]<=8/255 for x in raw["records"])
  result["G_clone"]={"row":row,"updates":raw["optimizer_steps"],"all_finite":True,
    "image_budget_pass":True,"inactive_H_exact":True,"real_original_cached_JVP_no_recomputation":True}
  print("G_RUNTIME_INTEGRATION_PASS",flush=True)
  b.root=archive;b.restore_complete(state)
  b.root=validation/"H_clone";b.begin_round(31);b.enter_phase_rng("H")
  row=next(r for r in b.p["samples"] if b.entries[r]["eligible"])
  before=b.g_fingerprint();b.h_probe_cursor=1;b.response_visits["H"]=1
  raw=b.h_update(row,b.last_bank)
  assert b.g_fingerprint()==before and raw["optimizer_steps_this_step"]==1
  result["H_clone"]={"row":row,"updates":1,"inactive_G_exact":True,
    "real_original_RewardBank_binding":True,"feature_artifact_published":raw["attribute_response"]["counterfactual_features"]}
  result["frozen"]=b.final_frozen()
 result["status"]="GH_PERFORMANCE_RUNTIME_INTEGRATION_PASS"
 validation.mkdir(exist_ok=True)
 with (validation/"report.json").open("x") as out:json.dump(result,out,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
