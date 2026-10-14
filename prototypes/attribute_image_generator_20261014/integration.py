"""仅离线prepare/训练导入；在线generate.py不导入此文件。"""
import sys,copy,json
from pathlib import Path
from unittest.mock import patch
import common as c
FULL=c.REPO/"prototypes/attribute_selection_full_test_20261011"
sys.path.insert(0,str(FULL))
import runtime as frozen
import torch
from attribute_tta import TextTransport
from paired_prefix import prepare_uncapped
from selection_audit import selection
GALLERY=frozen.score.GALLERY
VAL_REFERENCE=frozen.frozen.P74/"outputs/clean_scores/selection_manifest.json"
def input_locks():
 lock=frozen.locks()
 ref=c.load(VAL_REFERENCE)
 if not ref["full_preparation"]:raise ValueError("completed original validation score preparation required")
 for name,sha in [(ref["score_cache_path"],ref["score_cache_sha256"]),(ref["score_protocol_path"],ref["score_protocol_sha256"])]:
  c.verify({name:sha});lock[name]=sha
 prior=c.load(frozen.OLD/"outputs/diagnostic/protocol.json")["frozen_sha256"]
 if str(GALLERY) not in prior:raise ValueError("missing authoritative gallery SHA")
 lock[str(GALLERY)]=prior[str(GALLERY)];c.verify({str(GALLERY):lock[str(GALLERY)]})
 lock.update(c.own_hashes())
 return lock,ref
def dataset_rows(split,raw,role,limit):
 indices=split["train_indices" if role=="train" else "validation_indices"]
 by_image={}
 for i in indices:
  rec=split["records"][i];_,a,j=rec["row_id"].split(":");source=raw[int(a)]
  if source["split"]!="train" or source["id"]!=rec["pid"] or source["captions"][int(j)]!=rec["caption"]:raise ValueError("raw/v4 split mismatch")
  row={**rec,"image":source["file_path"]}
  old=by_image.get(row["image"])
  if old is None or int(j)<int(old["row_id"].split(":")[-1]):by_image[row["image"]]=row
 rows=sorted(by_image.values(),key=lambda x:(c.identity(["generator-v1",42,role,x["row_id"]]),x["row_id"]))
 if len(rows)<limit:raise ValueError("insufficient split images")
 return rows[:limit]
def fresh_score(victim,localizer,bases,features,row,label,index):
 probe=frozen.score.AttributeUnitV2Probe(victim,features,index,tau=.02)
 with patch("torch.autograd.grad",wraps=torch.autograd.grad) as calls:
  z,H,G,F,cos,zero=frozen.score.baseline_gradient(probe,row["caption"])
  with torch.no_grad():
   reference=victim.encode_texts([row["caption"]])
   explicit,_,zero2=probe.encode(row["caption"],8,torch.zeros_like(H))
  if H.dtype!=torch.float32 or G.dtype!=torch.float32 or zero.any() or zero2.any() or not torch.equal(z,reference) or not torch.equal(z,explicit):
   raise ValueError("frozen block8/zero contract")
  attrs=frozen.score.convert_record(label)
  attrs,weights,_,_=localizer.attach(attrs,row["caption"],H,None)
  scored=frozen.score.score_instances(attrs,row["caption"],G,weights,bases,None)
  if calls.call_count!=1:raise ValueError("new score needs exactly one clean backward")
 probe.assert_frozen()
 rec={"row_id":row["row_id"],"pid":row["pid"],"caption":row["caption"],"paired_image":row["image"],"paired_gallery_index":index,
 "F":F,"paired_cosine":cos,"retrieval_backward_count":1,"zero_intervention_exact":True,
 "attributes":[{k:copy.deepcopy(a[k]) for k in ("instance","instance_id","status","rank","S_raw","w") if k in a} for a in scored]}
 return rec
def prepare_condition(victim,row,record,bases):
 from conditioning import aggregate_condition
 if (record["row_id"],record["caption"],record["pid"],record["paired_image"])!=(row["row_id"],row["caption"],row["pid"],row["image"]):raise ValueError("cached condition identity")
 payload=selection(record)
 plan=prepare_uncapped(payload,record,bases,"cuda")
 transport=TextTransport(victim,row["caption"]);vectors={};checks=[]
 for a in plan:
  v,check=transport.read(a)
  if not check["zero_exact"] or check["central_fd_relative_error"]>.02:raise ValueError("frozen JVP/zero gate")
  vectors[a.instance_id]=v;checks.append(check)
 condition,weights=aggregate_condition(transport.clean,plan,vectors)
 return condition,{"payload":payload,"selected_instance_ids":[a.instance_id for a in plan],
 "score_normalized_weights":weights,"JVP_checks":checks,"K":len(plan),"S_raw_unchanged":True,
 "pooling":"sum_i (S_raw_i/sum_selected_S_raw) * unit(projected_context_i)",
 "local_target":"same weights * original unnormalized IRRA retrieval JVP_i"}
