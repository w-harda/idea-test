"""手动离线condition准备；默认CPU preflight，不自动做全量backward。"""
import argparse,time,copy,json
from collections import Counter
from pathlib import Path
import common as c
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--allow-preparation",action="store_true")
 p.add_argument("--output",type=Path,default=c.ROOT/"outputs/conditions")
 return p
def prepare_conditions(out,*,smoke=False):
 import integration as it
 from integration import torch
 out=c.safe(out);config=c.load(c.ROOT/"config.json")
 lock,ref=it.input_locks()
 split=c.load(it.frozen.frozen.V4/"outputs/split.json");raw=c.load(it.frozen.score.v4.RAW)
 counts={"train":2,"validation":2} if smoke else {"train":config["preparation"]["train_images"],"validation":config["preparation"]["validation_images"]}
 rows={role:it.dataset_rows(split,raw,role,n) for role,n in counts.items()}
 if set(r["pid"] for r in rows["train"])&set(r["pid"] for r in rows["validation"]):raise ValueError("PID leakage")
 protocol={"scope":"smoke" if smoke else "prepared","schema":"frozen-generator-condition-1","config":config,
 "row_ids":{role:[r["row_id"] for r in rs] for role,rs in rows.items()},"frozen_sha256":lock,
 "score_gallery":"34054 clean train images","test_split_used":False,"new_score_rule":"one clean retrieval backward per newly prepared training caption",
 "validation_S":"reuse exact original Step7.4 validation cache","fresh_training_e":"unchanged FrozenLocalization.attach(cache_index=None)",
 "online_inference_backward":False,"online_inference_frontend":"precomputed condition only; no arbitrary raw-caption S predictor"}
 out.mkdir(parents=True,exist_ok=True)
 if (out/"protocol.json").exists() and c.load(out/"protocol.json")!=protocol:raise ValueError("cache protocol changed; use new output")
 c.save(out/"protocol.json",protocol)
 if (out/"completed.json").exists():
  c.checked_cache(out);print("CONDITIONS_ALREADY_COMPLETE",flush=True);return
 wanted={r["row_id"] for rs in rows.values() for r in rs};labels={}
 for line in it.frozen.score.v4.LABELS.open():
  a=json.loads(line)
  if a["row_id"] in wanted:labels[a["row_id"]]=a
 if set(labels)!=wanted:raise ValueError("missing source provenance")
 cached={};valwanted={r["row_id"] for r in rows["validation"]}
 for line in Path(ref["score_cache_path"]).open():
  a=json.loads(line)
  if a["row_id"] in valwanted:cached[a["row_id"]]=a
 if set(cached)!=valwanted:raise ValueError("missing frozen validation score")
 victim=it.frozen.setup();versions={n:p._version for n,p in victim.named_parameters()}
 reader,bases,localizer=it.frozen.fresh_localizer(victim);reader_state={k:v.detach().cpu().clone() for k,v in reader.state_dict().items()}
 gallery=torch.load(it.GALLERY,weights_only=False,map_location="cpu")
 if gallery["spec"]!=c.load(it.frozen.score.BASE)["gallery_spec"] or len(gallery["features"])!=34054 or gallery["spec"]["torch_version"]!=str(torch.__version__):raise ValueError("gallery provenance")
 features=gallery["features"].cuda();paths={x:i for i,x in enumerate(gallery["spec"]["gallery_paths"])}
 entries={"train":[],"validation":[]};new=0;reuse=0;jvp=[];start=time.time();statuses=Counter()
 for role,rs in rows.items():
  for row in rs:
   cid=row["row_id"];name=cid.replace(":","_");meta_path=out/"records"/(name+".json");tensor_path=out/"tensors"/(name+".pt")
   image=c.IMAGE_ROOT/row["image"];image_sha=c.sha(image)
   if meta_path.exists():
    saved=c.load(meta_path)
    if saved["protocol_sha256"]!=c.identity(protocol) or saved["image_sha256"]!=image_sha or saved["row_id"]!=cid:raise ValueError("resume provenance")
    c.verify({str(tensor_path):saved["tensor_sha256"]})
   else:
    if labels[cid]["caption"]!=row["caption"] or labels[cid]["image"]!=row["image"] or labels[cid]["split"]!="train":raise ValueError("Stage01 mismatch")
    if role=="validation":
     record=copy.deepcopy(cached[cid]);origin="exact_frozen_validation_score";reuse+=1
    else:
     record=it.fresh_score(victim,localizer,bases,features,row,labels[cid],paths[row["image"]]);origin="once_clean_train_score";new+=1
    if record["paired_gallery_index"]!=paths[row["image"]] or record["retrieval_backward_count"]!=1 or not record["zero_intervention_exact"]:raise ValueError("score source")
    condition,metadata=it.prepare_condition(victim,row,record,bases)
    c.tensor_save(tensor_path,condition)
    saved={"row_id":cid,"pid":row["pid"],"caption":row["caption"],"image":row["image"],"role":role,
     "image_sha256":image_sha,"protocol_sha256":c.identity(protocol),"tensor_sha256":c.sha(tensor_path),
     "score_origin":origin,"score_record":record,**metadata}
    c.save(meta_path,saved)
   statuses.update(a["status"] for a in saved["score_record"]["attributes"])
   jvp.extend(x["central_fd_relative_error"] for x in saved["JVP_checks"])
   entry={"row_id":cid,"pid":row["pid"],"caption":row["caption"],"image":row["image"],"image_sha256":image_sha,
   "K":saved["K"],"tensor_path":str(tensor_path),"tensor_sha256":saved["tensor_sha256"],
   "metadata_path":str(meta_path),"metadata_sha256":c.sha(meta_path)}
   c.read_condition(entry);entries[role].append(entry)
   print("CONDITION_READY",role,cid,"K",saved["K"],flush=True)
 it.frozen.assert_frozen(victim,versions,reader,reader_state);c.verify(lock);it.frozen.frozen.assert_formal_clean()
 manifest={"schema":"generator-condition-manifest-1","scope":protocol["scope"],"counts":counts,**entries,
 "protocol_path":str(out/"protocol.json"),"protocol_sha256":c.sha(out/"protocol.json"),"config_sha256":c.sha(c.ROOT/"config.json"),
 "train_validation_PID_disjoint":True,"test_used":False,"frozen_sha256":lock}
 c.save(out/"manifest.json",manifest)
 c.save(out/"completed.json",{"method_audit_passed":True,"scope":protocol["scope"],"counts":counts,"manifest_sha256":c.sha(out/"manifest.json"),
 "new_scoring_backwards_this_run":new,"validation_scores_reused_this_run":reuse,
 "IRRA_reader_frozen":True,"max_JVP_FD_relative_error":max(jvp,default=0),"status_counts":dict(statuses),"seconds":time.time()-start})
 print("CONDITIONS_COMPLETE",json.dumps(c.load(out/"completed.json")),flush=True)
def main(argv=None):
 args=parser().parse_args(argv)
 if not args.allow_preparation:
  print("PREFLIGHT_ONLY: manual --allow-preparation prepares configured train/validation conditions; no GPU started");return
 prepare_conditions(args.output)
if __name__=="__main__":main()
