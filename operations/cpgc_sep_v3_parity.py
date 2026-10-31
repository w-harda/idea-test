"""独立CPU数值对照；不推理、不重新排名Full、不修改任何旧产物。"""
from pathlib import Path
import json
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from tbps.baselines.cpgc_sep.dataset import require
from tbps.artifacts import sha_file
OLD=Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2")
def compare(c,dataset,victim):
 import torch
 require(not torch.cuda.is_initialized(),"independent CPU parity only")
 roots=[OLD/"smoke/cpgc"/dataset/victim,Path(c["output_base"])/"smoke/cpgc"/dataset/victim]
 stores=[VerifiedArtifactStore(root,new_base=base) for root,base in zip(roots,[OLD,Path(c["output_base"])])]
 for store in stores:store.verify("completed")
 protocols=[json.loads((root/"protocol/protocol.json").read_text()) for root in roots]
 for key in("manifest","matched_manifest","source_completion","native_input","Raw_Clean","Matched_Clean"):
  require(protocols[0][key]==protocols[1][key],"input/Source/order difference: "+key)
 reports=[json.loads((root/"completed/summary.json").read_text()) for root in roots]
 require(reports[0]["model_before"]==reports[1]["model_before"]==reports[1]["model_after"],"same frozen model state")
 require(reports[0]["noise"]==reports[1]["noise"],"exact UAP source, pixel fingerprints and budget")
 n=0
 for family in("pixels-","clean_gallery-","adv_gallery-","clean_text-","adv_text-","matched_text-"):
  old={p.parent.name for p in roots[0].glob(family+"*/details.json")}
  new={p.parent.name for p in roots[1].glob(family+"*/details.json")}
  require(old==new,"same immutable member blocks: "+family)
  for key in old:
   for store in stores:store.verify(key)
   detail=[json.loads((root/key/"details.json").read_text()) for root in roots]
   require(detail[0]==detail[1],"pixel/feature Tensor payload digest difference: "+key);n+=1
 scores={}
 for mode in("clean","joint","matched"):
  key="scores-"+mode
  for store in stores:store.verify(key)
  tensors=[torch.load(root/key/"scores.pt",map_location="cpu",weights_only=True) for root in roots]
  d=(tensors[1]-tensors[0]).double()
  order=torch.equal(tensors[0].argsort(dim=1,descending=True),tensors[1].argsort(dim=1,descending=True))
  require(order and float(d.abs().max())<=1e-5,"native scoring ranking or FP32 error mismatch")
  scores[mode]={"exact":torch.equal(*tensors),"max_abs":float(d.abs().max()),"L2_difference":float(d.norm()),"ranking_exact":order}
 require(reports[0]["groups"]==reports[1]["groups"],"native Smoke retrieval metrics differ")
 return {"status":"PASS","dataset":dataset,"victim":victim,"pixel_feature_blocks_exact":n,"scores":scores,"metrics_exact":True,
  "old":str(roots[0]),"new":str(roots[1]),"old_manifest_sha":sha_file(roots[0]/"completed/manifest.json"),"new_manifest_sha":sha_file(roots[1]/"completed/manifest.json"),"GPU":False}
if __name__=="__main__":
 import sys
 c=json.loads(Path(sys.argv[1]).read_text());r=compare(c,sys.argv[2],sys.argv[3])
 with Path(sys.argv[4]).open("x")as f:json.dump(r,f,indent=2);f.write("\n")
 print("V2_V3_SMOKE_PARITY_PASS",json.dumps(r),flush=True)
