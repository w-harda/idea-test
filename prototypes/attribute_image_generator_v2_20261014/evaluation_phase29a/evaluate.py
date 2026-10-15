"""G-v2官方IRRA Test；默认CPU，Smoke≤10+10，100/Full仅用户手动确认。"""
import argparse,json,time,os,signal,uuid,traceback,csv
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
from eval_common import *
from eval_protocol import preflight,read_reference,authorize
from eval_storage import *
from eval_metrics import metrics,check_clean,drops,normalized
from eval_backend import RealBackend,ForwardBudget

class SafeStop(Exception):pass
class SignalBoundary:
 def __init__(self):self.requested=False;self.old={}
 def request(self,*args):self.requested=True
 def check(self):
  if self.requested:raise SafeStop("requested stop after atomic feature batch publication")
 def __enter__(self):
  for sig in (signal.SIGINT,signal.SIGTERM):
   self.old[sig]=signal.getsignal(sig);signal.signal(sig,self.request)
  return self
 def __exit__(self,*args):
  for sig,handler in self.old.items():signal.signal(sig,handler)

def evaluate_cached(protocol,out,reference,write=True):
 text,ts,locks=gather(out,"text",protocol,protocol["queries"],protocol["text_batch"])
 results={};group_stats={}
 for family in GROUPS:
  features,stats,files=gather(out,family,protocol,protocol["gallery"],protocol["image_batch"]);locks.update(files)
  result,sim=metrics(text,features,[q["pid"] for q in protocol["queries"]],[g["pid"] for g in protocol["gallery"]])
  if family=="clean":
   expected=protocol["expected_full_clean"] if protocol["mode"]=="full" else metrics(reference[1],reference[0],
    [q["pid"] for q in protocol["queries"]],[g["pid"] for g in protocol["gallery"]])[0]
   try:errors=check_clean(result,expected)
   except ValueError as exc:
    io.save_json(safe(out/"clean_mismatch.json"),{"actual":result,"expected":expected,"error":str(exc)});raise
  results[family]={"metrics":result,"drop_rate_percent":drops(results["clean"]["metrics"],result) if family!="clean" else {k:0. for k in METRICS}}
  group_stats[family]={"G_samples":sum(s["G_samples"] for s in stats),
   "IRRA_image_samples":sum(s["IRRA_image_samples"] for s in stats),"maximum_linf":max(v for s in stats for v in s["linfs"])}
  if family!="clean" and group_stats[family]["G_samples"]!=protocol["gallery_count"]:raise ValueError("exactly once G cache coverage")
 summary={"method_audit_passed":True,"mode":protocol["mode"],"formal_test":protocol["mode"]=="full",
  "gallery_count":protocol["gallery_count"],"query_count":protocol["query_count"],
  "groups":results,"group_stats":group_stats,"clean_metric_abs_error":errors,"metric_unit":"percent",
  "drop_definition":"(clean-adversarial)/clean*100; negative allowed; null if clean=0",
  "single_victim":"official IRRA; no transfer claim","Text_Query_unchanged":True,"text_features_shared":True,
  "optimizer_steps":0,"backwards":0,"JVPs":0,"feature_file_sha256":locks,
  "checkpoint_sha256":protocol["checkpoint_sha256"]}
 if not write:return summary
 io.save_json(safe(out/"summary.json"),summary)
 with safe(out/"metrics.csv").open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f);w.writerow(["group",*METRICS,*["drop_"+k for k in METRICS]])
  for family in GROUPS:w.writerow([family,*[results[family]["metrics"][k] for k in METRICS],*[results[family]["drop_rate_percent"][k] for k in METRICS]])
 return summary

@torch.inference_mode()
def run_pipeline(protocol,out,backend_factory=RealBackend,reference=None,stop=None):
 out=safe(out);started=time.perf_counter();budget=ForwardBudget(protocol["gallery_count"],protocol["query_count"])
 reference=read_reference(protocol) if reference is None else reference
 backend=None;context={};files={}
 if stop is None:stop=SignalBoundary()
 with io.output_lock(out):
  if (out/"protocol.json").exists():
   if io.digest(io.load(out/"protocol.json"))!=io.digest(protocol):raise ValueError("output evaluation protocol changed")
  else:io.save_json(out/"protocol.json",protocol)
  if (out/"completed.json").exists():
   done=io.load(out/"completed.json")
   if not done.get("method_audit_passed") or done["protocol_sha256"]!=io.digest(protocol):raise ValueError("completion protocol mismatch")
   io.verify({str(out/"summary.json"):done["summary_sha256"]})
   summary=io.load(out/"summary.json");io.verify(summary["feature_file_sha256"])
   print("ALREADY_COMPLETE_CPU_ONLY",flush=True);return summary
  attempt=safe(out/"attempts"/uuid.uuid4().hex);attempt.mkdir(parents=True)
  try:
   for family in ("text",*GROUPS):
    rows=protocol["queries"] if family=="text" else protocol["gallery"]
    size=protocol["text_batch"] if family=="text" else protocol["image_batch"]
    for index,part in shards(rows,size):
     context={"family":family,"batch_index":index}
     stop.check()
     saved=load_unit(out,family,index,protocol,part)
     if saved is not None:continue
     begin_unit(out,family,index)
     if protocol["reuse_clean_reference"] and family in ("text","clean"):
      start=index*size;feature=reference[1 if family=="text" else 0][start:start+len(part)]
      stats={"text_samples":0,"G_samples":0,"IRRA_image_samples":0,"reference_reused":True}
      if family=="clean":stats["linfs"]=[0.]*len(part)
     else:
      if backend is None:backend=backend_factory(protocol,budget)
      feature,stats=backend.text(part) if family=="text" else backend.image(family,part)
      if family in ("text","clean"):
       start=index*size;ref=reference[1 if family=="text" else 0][start:start+len(part)]
       error=float((feature-ref).abs().max())
       if error>FEATURE_ATOL:raise ValueError("fresh clean/text feature mismatch; never relax tolerance")
       stats["fresh_reference_max_abs"]=error
     publish(out,family,index,protocol,part,feature,stats)
     print("FEATURE_COMPLETE",family,min((index+1)*size,len(rows)),"/",len(rows),flush=True)
     stop.check()
    if family=="clean":
     text,_,_=gather(out,"text",protocol,protocol["queries"],protocol["text_batch"])
     clean,_,_=gather(out,"clean",protocol,protocol["gallery"],protocol["image_batch"])
     actual,_=metrics(text,clean,[q["pid"] for q in protocol["queries"]],[g["pid"] for g in protocol["gallery"]])
     expected=protocol["expected_full_clean"] if protocol["mode"]=="full" else metrics(reference[1],reference[0],
      [q["pid"] for q in protocol["queries"]],[g["pid"] for g in protocol["gallery"]])[0]
     try:errors=check_clean(actual,expected)
     except ValueError as exc:
      io.save_json(out/"clean_mismatch.json",{"actual":actual,"expected":expected,"error":str(exc)});raise
     io.save_json(out/"clean_verified.json",{"errors":errors,"passed":True})
   resources=backend.finish() if backend is not None else {"no_model_required":True}
   summary=evaluate_cached(protocol,out,reference)
   io.verify(protocol["source_sha256"])
   io.save_json(attempt/"execution.json",{"seconds":time.perf_counter()-started,"samples":budget.samples,"calls":budget.calls,"resources":resources})
   io.save_json(out/"completed.json",{"method_audit_passed":True,"summary_sha256":io.sha(out/"summary.json"),
    "protocol_sha256":io.digest(protocol),"attempt":str(attempt)})
   print("EVALUATION_COMPLETE",protocol["mode"],budget.samples,resources,flush=True)
   return summary
  except BaseException as exc:
   io.save_json(attempt/"interruption.json" if isinstance(exc,SafeStop) else attempt/"failure.json",
    {"error":repr(exc),"traceback":traceback.format_exc(),"context":context,"samples":budget.samples,"calls":budget.calls})
   raise
  finally:
   if backend is not None:backend.finish()

def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--action",choices=("preflight","run","audit"),default="preflight")
 p.add_argument("--mode",choices=("smoke","diagnostic100","full"),default="smoke")
 p.add_argument("--gallery-count",type=int,default=10);p.add_argument("--query-count",type=int,default=10)
 p.add_argument("--checkpoint1",type=Path,default=TRAIN/"checkpoints/epoch_0001/checkpoint.pt")
 p.add_argument("--checkpoint3",type=Path,default=TRAIN/"checkpoints/epoch_0003/checkpoint.pt")
 p.add_argument("--checkpoint10",type=Path,default=TRAIN/"checkpoints/epoch_0010/checkpoint.pt")
 p.add_argument("--output",type=Path,default=EVAL_OUTPUTS/"smoke10")
 p.add_argument("--image-batch",type=int,default=1);p.add_argument("--text-batch",type=int,default=1)
 p.add_argument("--reuse-clean-reference",action="store_true")
 p.add_argument("--user-run-100",action="store_true");p.add_argument("--user-run-full",action="store_true")
 return p
def main(argv=None):
 a=parser().parse_args(argv)
 authorize(a.mode,a.action,a.gallery_count,a.query_count,a.user_run_100,a.user_run_full)
 protocol=preflight(a)
 print("CPU_PREFLIGHT",protocol["official_counts"],"selected",protocol["gallery_count"],protocol["query_count"],
  "mode",a.mode,"checkpoint_sha256",protocol["checkpoint_sha256"],flush=True)
 if a.action=="preflight":return
 if a.action=="audit":
  done=io.load(safe(a.output)/"completed.json")
  io.verify({str(safe(a.output)/"summary.json"):done["summary_sha256"]})
  with torch.inference_mode():summary=evaluate_cached(protocol,safe(a.output),read_reference(protocol),write=False)
  if io.digest(summary)!=io.digest(io.load(safe(a.output)/"summary.json")):raise ValueError("cached result metric mismatch")
  io.verify(protocol["source_sha256"]);print("CPU_CACHE_AUDIT",summary["groups"]);return
 with SignalBoundary() as stop,ExitStack() as stack:
  for name in ("backward",):stack.enter_context(patch.object(torch.Tensor,name,side_effect=AssertionError("inference backward forbidden")))
  stack.enter_context(patch.object(torch.autograd,"grad",side_effect=AssertionError("no scoring/JVP")))
  stack.enter_context(patch.object(torch.optim.Optimizer,"__init__",side_effect=AssertionError("no optimizer")))
  try:run_pipeline(protocol,a.output,stop=stop)
  except SafeStop:print("SAFE_STOP; resume same exact command/output",flush=True)
if __name__=="__main__":main()
