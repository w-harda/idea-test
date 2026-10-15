"""CPU来源/官方Test布局审计，不构造模型或编码全量数据。"""
import sys,hashlib
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path
import torch
from eval_common import *
from eval_metrics import normalized
import epoch_checkpoint as ck
import irra_image_adapter as adapter
from eval_compat import accept_legacy_protocol
def authorize(mode,action,gallery_count,query_count,user100=False,userfull=False):
 sizes={"smoke":(10,10),"diagnostic100":(100,100),"full":(3074,6156)}
 if mode not in sizes or (gallery_count,query_count)!=sizes[mode]:raise ValueError("fixed mode sample limits")
 if action!="preflight" and ((mode=="diagnostic100" and not user100) or (mode=="full" and not userfull)):
  raise ValueError("100+100 and Full are USER MANUAL ONLY; explicit confirmation required")
 return sizes[mode]
def layout(raw):
 gallery=[];queries=[];seen=set()
 for ri,row in enumerate(raw):
  if row["split"]!="test":continue
  image=row["file_path"];pid=int(row["id"])
  path=(IMAGES/image).resolve()
  if not path.is_relative_to(IMAGES.resolve()) or image in seen:raise ValueError("unsafe/duplicate Test image")
  seen.add(image);index=len(gallery)
  if not row["captions"] or any(not isinstance(c,str) or not c.strip() for c in row["captions"]):
   raise ValueError("missing Test caption")
  gallery.append({"index":index,"row_id":f"cuhk:{ri}:0","annotation_index":ri,"image":image,"pid":pid})
  for ci,caption in enumerate(row["captions"]):
   queries.append({"index":len(queries),"row_id":f"cuhk:{ri}:{ci}","caption_index":ci,"caption":caption,"pid":pid,"paired_index":index})
 return gallery,queries
def subset(gallery,queries,count,seed=42):
 by_pid=defaultdict(list)
 for g in gallery:by_pid[g["pid"]].append(g)
 eligible=[pid for pid,gs in by_pid.items() if len(gs)>=2]
 eligible.sort(key=lambda pid:hashlib.sha256(f"gv2-test|{seed}|{pid}".encode()).hexdigest())
 if count%2 or len(eligible)<count//2:raise ValueError("insufficient paired PID subset")
 chosen=[]
 for pid in eligible[:count//2]:
  chosen.extend(sorted(by_pid[pid],key=lambda g:hashlib.sha256(f"{seed}|{g['row_id']}".encode()).hexdigest())[:2])
 gs=sorted(chosen,key=lambda g:g["index"]);indices={g["index"] for g in gs}
 qs=[q for q in queries if q["paired_index"] in indices and q["caption_index"]==0]
 if len(qs)!=count or not set(q["pid"] for q in qs)<=set(g["pid"] for g in gs):raise ValueError("subset positive pairing")
 return gs,qs
def reference(gallery,queries):
 io.verify({str(REFERENCE/n):h for n,h in REF_SHA.items()})
 manifest=io.load(REFERENCE/"selection_manifest.json");old=io.load(REFERENCE/"protocol.json")
 if manifest["clean_features_sha256"]!=REF_SHA["clean_features.pt"] or old["precision"]!="fp32":raise ValueError("clean reference provenance")
 for key,actual in (("gallery",gallery),("queries",queries)):
  prior=manifest[key]
  fields=("row_id","image","pid","index") if key=="gallery" else ("row_id","caption","pid","paired_index")
  if len(prior)!=len(actual) or any(any(a[k]!=b[k] for k in fields) for a,b in zip(actual,prior)):
   raise ValueError("reference official layout mismatch")
 for name,sha in ((str(RAW),io.sha(RAW)),(str(adapter.CHECKPOINT),adapter.CHECKPOINT_SHA)):
  if old["frozen_sha256"].get(name)!=sha:raise ValueError("reference IRRA/annotation mismatch")
 for path,h in old["frozen_sha256"].items():
  if path.startswith(str(OFFICIAL)+"/") or path==str(adapter.CONFIG):io.verify({path:h})
 if io.sha(OFFICIAL/"utils/metrics.py")!="94910252a9e42f0820d49c11203b31d3b67951f58624f66abff4bf662500d5fb":raise ValueError("official rank source changed")
 return io.load(REFERENCE/"summary.json")["clean"]
def preflight(args):
 authorize(args.mode,args.action,args.gallery_count,args.query_count,args.user_run_100,args.user_run_full)
 output=safe(args.output)
 gallery,queries=layout(io.load(RAW))
 if (len(gallery),len(queries))!=(3074,6156):raise ValueError("official Test counts changed")
 CUHK,_,_=official_imports()
 official=CUHK(root=str(RAW.parent.parent),verbose=False).test
 if (official["img_paths"]!=[str(IMAGES/g["image"]) for g in gallery]
     or official["image_pids"]!=[g["pid"] for g in gallery]
     or official["captions"]!=[q["caption"] for q in queries]
     or official["caption_pids"]!=[q["pid"] for q in queries]):
  raise ValueError("official CUHKPEDES reader identity mismatch")
 expected=reference(gallery,queries)
 prior_images=io.load(REFERENCE/"selection_manifest.json")["rows"]
 for g in gallery:
  g["image_sha256"]=io.sha(IMAGES/g["image"])
  if g["image_sha256"]!=prior_images[g["index"]]["image_sha256"]:raise ValueError("Test image changed since pinned Clean reference")
 contract=io.load(TRAIN/"training_protocol.json")
 if io.digest(contract["config"])!=io.digest(io.load(ROOT/"config_phase26.json")):raise ValueError("frozen training config changed")
 io.verify(contract["training_source_sha256"]);io.verify(contract["config"]["entry_source_sha256"])
 for epoch,path in ((1,args.checkpoint1),(3,args.checkpoint3),(10,args.checkpoint10)):
  if Path(path).resolve()!=TRAIN/f"checkpoints/epoch_{epoch:04d}/checkpoint.pt":raise ValueError("fixed trained checkpoint path required")
  io.verify({str(path):CP_SHA[epoch]})
  value=ck.verify_version(Path(path).parent,contract)
  if value["completed_epoch"]!=epoch:raise ValueError("wrong Epoch")
  del value
 done=io.load(ROOT/"outputs/phase28/completed.json")
 io.verify({str(ROOT/"outputs/phase28/summary.json"):done["summary_sha256"]})
 known={e["epoch"]:e["checkpoint_sha256"] for e in io.load(ROOT/"outputs/phase28/summary.json")["epochs"]}
 locks=dict(io.load(ROOT/"outputs/phase28/protected_sources.json"))
 # 保护全部10 Epoch及日志；只读核验不重新准备属性缓存。
 for n in range(1,11):
  path=TRAIN/f"checkpoints/epoch_{n:04d}"
  io.verify({str(path/"checkpoint.pt"):known[n]});value=ck.verify_version(path,contract)
  locks.update({str(p):io.sha(p) for p in path.iterdir()})
  locks[value["progress"]["sample_log_path"]]=value["progress"]["sample_log_sha256"]
  locks.update(value["progress"]["supplementary_log_sha256"]);del value
 locks.update({str(p):io.sha(p) for p in [*OFFICIAL.rglob("*.py"),adapter.CONFIG,adapter.CHECKPOINT,
  adapter.CLIP_CACHE/"ViT-B-16.pt",OFFICIAL/"data/bpe_simple_vocab_16e6.txt.gz",RAW,
  ROOT/"outputs/phase28/completed.json",ROOT/"outputs/phase28/summary.json",ROOT/"config_phase26.json",TRAIN/"training_protocol.json",
  ROOT/"image_only_generator.py",ROOT/"budget_projection.py",ROOT/"inference_once.py",ROOT/"irra_image_adapter.py",
  *HERE.glob("*.py"),HERE/"storage_fix_compatibility.json",*[REFERENCE/n for n in REF_SHA]]})
 io.verify(locks)
 gs,qs=(gallery,queries) if args.mode=="full" else subset(gallery,queries,args.gallery_count,42)
 locks.update({str(IMAGES/g["image"]):g["image_sha256"] for g in gs})
 if args.reuse_clean_reference and args.mode=="smoke":raise ValueError("Smoke must freshly encode only selected10+10")
 protocol={"schema":"gv2-official-evaluation-1","mode":args.mode,"gallery":gs,"queries":qs,
  "official_counts":{"gallery":3074,"query":6156},"gallery_count":len(gs),"query_count":len(qs),
  "user_manual_run_confirmed":bool(args.user_run_100 if args.mode=="diagnostic100" else args.user_run_full) if args.mode!="smoke" else False,
  "seed":42,"image_batch":args.image_batch,"text_batch":args.text_batch,
  "reuse_clean_reference":args.reuse_clean_reference,"checkpoint_sha256":{str(n):CP_SHA[n] for n in (1,3,10)},
  "checkpoint_paths":{str(n):str(getattr(args,f"checkpoint{n}")) for n in (1,3,10)},
  "source_sha256":locks,"expected_full_clean":expected,"feature_atol":FEATURE_ATOL,
  "clean_metric_atol_percentage_points":CLEAN_METRIC_ATOL_PP,"precision":"fp32","image_size":[384,128],
  "relevance":"all same-PID positives; no camera exclusion; official rank",
  "G_input":"image only","epsilon":8/255,"single_victim":"official IRRA; no transfer claim",
  "runtime":{"torch":str(torch.__version__),"python":sys.version,
   "packages":{name:version(name) for name in ("torchvision","numpy","ftfy","regex","prettytable","easydict")}}}
 if type(args.image_batch) is not int or not 1<=args.image_batch<=32 or not 1<=args.text_batch<=128:
  raise ValueError("invalid bounded batch sizes")
 if getattr(args,"allow_storage_fix_compat",False):
  if not (output/"protocol.json").exists():raise ValueError("storage fix requires the existing locked Full protocol")
  return accept_legacy_protocol(io.load(output/"protocol.json"),protocol,output)
 if (output/"protocol.json").exists() and io.digest(io.load(output/"protocol.json"))!=io.digest(protocol):
  raise ValueError("output protocol/source/checkpoint changed; locked Full repair requires --allow-storage-fix-compat")
 return protocol
def read_reference(protocol):
 v=torch.load(REFERENCE/"clean_features.pt",map_location="cpu",weights_only=True)
 normalized(v["image"],3074);normalized(v["text"],6156)
 return v["image"][[g["index"] for g in protocol["gallery"]]],v["text"][[q["index"] for q in protocol["queries"]]]
