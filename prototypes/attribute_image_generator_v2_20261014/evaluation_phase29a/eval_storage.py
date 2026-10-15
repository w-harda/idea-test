"""不可变特征批次；原子发布、SHA、同输出进程锁和未提交计算回执。"""
import os,uuid,math
from pathlib import Path
import torch
from eval_common import io,safe
from eval_metrics import normalized
def validate_stats(family,stats,n):
 reused=stats.get("reference_reused",False)
 if reused and family not in ("text","clean"):raise ValueError("adversarial reference reuse forbidden")
 if stats["G_samples"]!=(n if family.startswith("epoch") else 0):raise ValueError("G sample count mismatch")
 if stats["text_samples"]!=(n if family=="text" and not reused else 0):raise ValueError("text sample count mismatch")
 if stats["IRRA_image_samples"]!=(n if family!="text" and not reused else 0):raise ValueError("IRRA image count mismatch")
 if family!="text":
  values=stats["linfs"]
  if len(values)!=n or any(not math.isfinite(v) or not 0<=v<=8/255 for v in values):
   raise ValueError("cached strict budget violation")
  if family=="clean" and any(v!=0 for v in values):raise ValueError("clean pixels must be unperturbed")
def unit_path(out,family,index):return safe(Path(out)/"features"/family/f"{index:06d}")
def load_unit(out,family,index,protocol,rows):
 target=unit_path(out,family,index)
 if not target.exists():return None
 meta=io.load(target/"metadata.json")
 if (meta["protocol_sha256"]!=io.digest(protocol) or meta["family"]!=family or meta["rows"]!=rows):
  raise ValueError("feature cache identity/protocol mismatch")
 io.verify({str(target/"features.pt"):meta["sha256"]})
 value=torch.load(target/"features.pt",map_location="cpu",weights_only=True)
 normalized(value["features"],len(rows))
 if value["stats"]!=meta["stats"]:raise ValueError("feature stats mismatch")
 validate_stats(family,value["stats"],len(rows))
 return value
def begin_unit(out,family,index):
 claim=safe(Path(out)/"receipts"/family/f"{index:06d}.json")
 if claim.exists():raise ValueError("uncommitted computation receipt; inspect evidence; never automatically repeat G")
 io.save_json(claim,{"family":family,"batch_index":index,"status":"started_before_forward"})
def publish(out,family,index,protocol,rows,features,stats):
 target=unit_path(out,family,index)
 if target.exists():raise ValueError("immutable feature batch already exists")
 normalized(features,len(rows));validate_stats(family,stats,len(rows))
 target.parent.mkdir(parents=True,exist_ok=True)
 stage=safe(target.parent/(".pending-"+uuid.uuid4().hex));stage.mkdir()
 with (stage/"features.pt").open("xb") as f:
  torch.save({"features":features.detach().cpu(),"stats":stats},f);f.flush();os.fsync(f.fileno())
 io.save_json(stage/"metadata.json",{"family":family,"rows":rows,"protocol_sha256":io.digest(protocol),
  "sha256":io.sha(stage/"features.pt"),"stats":stats})
 io.fsync_dir(stage);os.rename(stage,target);io.fsync_dir(target.parent)
 return load_unit(out,family,index,protocol,rows)
def shards(rows,size):
 return [(index,rows[start:start+size]) for index,start in enumerate(range(0,len(rows),size))]
def gather(out,family,protocol,rows,size):
 values=[];stats=[];locks={}
 for index,part in shards(rows,size):
  v=load_unit(out,family,index,protocol,part)
  if v is None:raise ValueError("incomplete feature cache "+family)
  values.append(v["features"]);stats.append(v["stats"])
  p=unit_path(out,family,index)
  locks.update({str(p/n):io.sha(p/n) for n in ("features.pt","metadata.json")})
 return torch.cat(values),stats,locks
