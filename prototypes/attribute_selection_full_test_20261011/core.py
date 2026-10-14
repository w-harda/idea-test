"""完整 test 的共享图库协议、确定性选择及安全写入；不改变冻结评分。"""
import hashlib,json,random,math,os
from pathlib import Path
from collections import Counter
MODES=("S_TOPK","RANDOM_K","S_BOTTOMK")
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
RAW=Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json")
IMAGES=RAW.parent/"imgs"
LABELS=Path("/home/lzf/ldx/outputs/idea-TBPS-test1/stage03/cuhk_all_scores.jsonl")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
def safe(p):
 p=Path(p).resolve()
 if not p.is_relative_to(Path("/home/lzf/ldx").resolve()):raise ValueError("write outside /home/lzf/ldx")
 return p
def save(p,x):
 p=safe(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+".tmp")
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
 os.replace(tmp,p)
def verify(locks):
 for p,h in locks.items():
  if sha(p)!=h:raise ValueError("SHA changed: "+p)
def manifest(raw):
 gallery=[];queries=[];seen=set()
 for i,a in enumerate(raw):
  if a["split"]!="test":continue
  path=a["file_path"]
  if path in seen:raise ValueError("duplicate test image")
  seen.add(path)
  if not (IMAGES/path).resolve().is_relative_to(IMAGES.resolve()):raise ValueError("unsafe image path")
  captions=a["captions"]
  if not captions or any(not isinstance(c,str) or not c.strip() for c in captions):
   raise ValueError("test empty caption; no silent omission")
  ids=[f"cuhk:{i}:{j}" for j in range(len(captions))]
  index=len(gallery)
  gallery.append({"index":index,"image":path,"pid":a["id"],"row_id":ids[0],
                  "caption":captions[0],"annotation_index":i,"caption_index":0})
  queries.extend({"row_id":rid,"caption":c,"pid":a["id"],"paired_index":index}
                 for rid,c in zip(ids,captions))
 if not gallery:raise ValueError("empty test split")
 return {"schema":"shared-test-selection-v1","gallery":gallery,"queries":queries,
  "gallery_count":len(gallery),"query_count":len(queries),
  "conditioning_rule":"first annotation caption (index=0) per image; fixed before scoring; no result selection",
  "all_test_images_included":True,"all_test_captions_included":True}
def selected(payload):
 ranked=payload["ranked_attributes"];k=payload["natural_k"];m=len(ranked)
 if not 0<=k<=m or m!=payload["candidate_count"]:raise ValueError("invalid K/m")
 ids=[a["instance_id"] for a in ranked]
 if len(set(ids))!=m or any(a["S_raw"]<b["S_raw"] for a,b in zip(ranked,ranked[1:])):raise ValueError("invalid frozen rank")
 seed=int.from_bytes(hashlib.sha256(("random-k|42|"+payload["caption_id"]).encode()).digest()[:8],"big")
 indices=sorted(random.Random(seed).sample(range(m),k))
 choices={MODES[0]:ids[:k],MODES[1]:[ids[i] for i in indices],MODES[2]:ids[m-k:] if k else []}
 overlap={}
 for a,b in ((MODES[0],MODES[1]),(MODES[0],MODES[2]),(MODES[1],MODES[2])):
  x,y=set(choices[a]),set(choices[b]);overlap[a+"_vs_"+b]={"intersection":len(x&y),"identical":x==y,"jaccard":len(x&y)/len(x|y) if x|y else 1.0}
 return {**payload,"selection_ids":choices,"random_seed_hex":hex(seed),"set_overlap":overlap}
def inventory(rows):
 return {"image_count":len(rows),"K_distribution":dict(Counter(r["natural_k"] for r in rows)),
 "m_distribution":dict(Counter(r["candidate_count"] for r in rows)),
 "m_gt_K":sum(r["candidate_count"]>r["natural_k"] for r in rows),
 "K0_clean_passthrough":sum(r["natural_k"]==0 for r in rows),
 "logical_rounds_per_mode":sum(r["natural_k"] for r in rows),
 "physical_rounds":sum(r["natural_k"]*len({tuple(v) for v in r["selection_ids"].values()}) for r in rows)}
def validate_score(record,payload,source,select_fn):
 if (record["row_id"],record["caption"],record["pid"],record["paired_image"],record["paired_gallery_index"])!=(source["row_id"],source["caption"],source["pid"],source["image"],source["index"]):
  raise ValueError("score source mismatch")
 if record["retrieval_backward_count"]!=1 or not record["zero_intervention_exact"]:raise ValueError("one-clean-backward/zero contract")
 if selected(select_fn(record))!=payload:raise ValueError("frozen score/selection changed")
def require_complete(done,manifest,scope):
 if done.get("scope")!=scope or not done.get("method_audit_passed"):raise ValueError("incomplete/wrong-scope stage")
 if done.get("gallery_count")!=manifest["gallery_count"] or done.get("query_count")!=manifest["query_count"]:
  raise ValueError("incomplete gallery/query coverage")
def estimated_hours(inv,step_seconds=.8):
 return inv["physical_rounds"]*10*step_seconds/3600
