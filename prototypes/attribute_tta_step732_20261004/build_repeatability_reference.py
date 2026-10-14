"""从已完成5次重复结果建立保守条件回归，不运行IRRA或攻击。"""
import json,sys,hashlib,itertools
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from repeatability_gate import historical_signature
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
out=ROOT/"outputs/repeatability_35"
all_data=[json.loads((out/f"run{i}.json").read_text()) for i in range(1,6)]
prior=json.loads((ROOT/"outputs/diagnostic_round1_fixed/regression_failure.json").read_text())["saved_NEW"]
condition=[i for i,d in enumerate(all_data,1)
           if d["rounds"][0]["post_projection_image_sha256"]==prior["rounds"][0]["post_projection_image_sha256"]
           and d["states"][:2]==prior["states"][:2]]
assert condition==[3,4,5]
certified=[i for i in condition if all_data[i-1]["states"]==prior["states"] and
           [e["post_projection_image_sha256"] for e in all_data[i-1]["rounds"]]==[e["post_projection_image_sha256"] for e in prior["rounds"]]]
assert certified==[3]
data=[torch.load(out/f"run{i}_tensors.pt",weights_only=True,map_location="cpu") for i in condition]
reference=[data[0]["initial_image"]]+data[0]["round_outputs"]
lasts=[d["round_outputs"][1] for d in data]
stack=torch.stack(lasts)
metrics={k:{"min":min(all_data[i-1]["states"][2][k] for i in condition),
            "max":max(all_data[i-1]["states"][2][k] for i in condition),
            "max_abs":max(abs(all_data[i-1]["states"][2][k]-all_data[j-1]["states"][2][k]) for i,j in itertools.combinations(condition,2))}
         for k in ("paired_cosine","paired_soft_rank_F")}
images=[];directions=[]
for a,b in itertools.combinations(lasts,2):
 delta=a.double()-b.double()
 images.append({"max_abs":float(delta.abs().max()),"l2":float(delta.norm()),"changed":int((a!=b).sum())})
 directions.append(float(F.cosine_similarity((a-reference[0]).double().flatten(),(b-reference[0]).double().flatten(),dim=0)))
tensor_file="repeatability_reference_35.pt"
torch.save({"reference":reference,"lower":stack.min(0).values,"upper":stack.max(0).values},ROOT/tensor_file)
manifest={"kind":"observed_conditional_repeatability","caption_id":"cuhk:35:1",
 "all_repeated_runs":[1,2,3,4,5],"conditioned_runs":condition,"historical_SHA_certified_reconstruction_run":3,
 "excluded_from_gate_calibration":[{"run":i,"reason":"round1 differs from historical condition; remains reported and rejected"} for i in (1,2)],
 "historical_signature":historical_signature(prior),"tensor_file":tensor_file,
 "execution_contract":{"steps":10,"transforms":6,"scales":[.5,.75,1.25,1.5],"epsilon":8/255,"step_size":2/255,"momentum":.9,"seed":42},
 "metric_bounds":metrics,"image_limits":{"max_abs":max(x["max_abs"] for x in images),"max_l2":max(x["l2"] for x in images),
     "max_changed_elements":max(x["changed"] for x in images),"min_perturbation_cosine":min(directions)},
 "provenance":{str(out/f):digest(out/f) for f in ["report.json","summary.json","run3_tensors.pt","run4_tensors.pt","run5_tensors.pt","run3.json","run4.json","run5.json"]},
 "algorithm_unchanged":True,"no_global_tolerance":True,
 "limitation":"five repeats do not calibrate other captions; larger early numerical branches remain rejected"}
(ROOT/"repeatability_reference_35.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({k:v for k,v in manifest.items() if k!="provenance"},indent=2))
