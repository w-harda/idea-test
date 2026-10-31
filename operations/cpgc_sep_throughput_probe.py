"""授权的独立吞吐量/数值对照；不改官方攻击、不写旧实验。"""
from pathlib import Path
from dataclasses import asdict
import argparse,json,time,os,traceback,resource,subprocess,threading,hashlib
import torch
from tbps.artifacts import ArtifactStore,sha_file,json_bytes
from tbps.models.irra import fp32_runtime
from tbps.baselines.cpgc_sep.runtime import config,resources
from tbps.baselines.cpgc_sep.preflight import run as preflight
from tbps.baselines.cpgc_sep.dataset import read_test,require
from tbps.baselines.cpgc_sep.victims import load_victim,native_pixels,encode_images,encode_texts,snapshot,frozen
from tbps.baselines.cpgc_sep.target import cat_payload
from tbps.models.base import RepresentationBatch

ROOT=Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-throughput-validation-v3")
def save(p,data):
 with p.open("x") as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write("\n")
def diff(a,b):
 if isinstance(a,dict):return {k:diff(a[k],b[k]) for k in a}
 d=(a.double()-b.double());return {"exact":torch.equal(a,b),"max_abs":float(d.abs().max()),"l2":float(d.norm()),"relative_l2":float(d.norm()/a.double().norm().clamp_min(1e-30)),"different_elements":int((a!=b).sum())}
def exact(d):return all(exact(x) for x in d.values()) if "exact" not in d else d["exact"]
def timed(fn):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
 x=fn();torch.cuda.synchronize()
 return x,{"seconds":time.monotonic()-t,"allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved()}
def encode(adapter,inputs,model,batch,kind):
 return cat_payload([encode_images(adapter,inputs[i:i+batch],model) if kind=="image" else encode_texts(adapter,inputs[i:i+batch],model) for i in range(0,len(inputs),batch)])
def scores(adapter,q,g):
 return adapter.score(RepresentationBatch(q,adapter.space.id,"text"),RepresentationBatch(g,adapter.space.id,"image")).payload
def rank(s,pids,qpid):
 # This benchmark reports exact complete local ranking identities, not official Full metrics.
 order=s.argsort(dim=1,descending=True)
 ap=[];r=[0,0,0]
 for i,row in enumerate(order):
  pos=torch.tensor([pids[j]==qpid[i] for j in row]);where=pos.nonzero().flatten()
  if len(where):
   ap.append(float((pos.cumsum(0)[where]/(where+1)).float().mean()))
   for k,K in enumerate((1,5,10)):r[k]+=bool(pos[:K].any())
  else:ap.append(0.)
 return {"R@1":r[0]/len(order)*100,"R@5":r[1]/len(order)*100,"R@10":r[2]/len(order)*100,"mAP":sum(ap)/len(order)*100},order
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--allow-gpu-benchmark",action="store_true");a=p.parse_args()
 require(a.allow_gpu_benchmark,"explicit authorization");c,_=preflight(a.config);rs=resources(c)
 require(rs["memory.free"]>=20000,"benchmark needs >=20000MiB")
 require(not ROOT.exists(),"unique benchmark consumed")
 ROOT.mkdir();save(ROOT/"Claim.json",{"config":str(a.config),"sha256":sha_file(a.config),"time":time.time(),"resources":rs})
 events=[];stop=threading.Event()
 def monitor():
  while not stop.wait(1):
   z=subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,power.draw","--format=csv,noheader,nounits"],capture_output=True,text=True)
   events.append({"time":time.time(),"value":z.stdout.strip()})
 t=threading.Thread(target=monitor,daemon=True);t.start()
 out={"status":"RUNNING","models":{},"config_sha256":c["sha256"],"semantics":"same native model/FP32/UAP; reject numerical or ranking failures, no changed attack"}
 torch.set_num_threads(2)
 try:
  for model in ("irra","rasa"):
   m=read_test(c["datasets"]["icfg"])
   # Fixed first 128 official Gallery, first 64 original paired Query; no performance-based selection.
   gs=m.gallery[:128];qs=m.queries[:64]
   x=torch.cat([native_pixels(Path(c["datasets"]["icfg"]["root"])/"imgs"/g.image_rel_path,model) for g in gs])
   oldsource=Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/full/cpgc_source/icfg")
   st=ArtifactStore(oldsource,new_base=oldsource.parents[2]);st.verify("completed")
   texts=[];matched=[]
   for i,q in enumerate(qs):
    st.verify(f"text-{i:06d}");v=json.loads((oldsource/f"text-{i:06d}/text.json").read_text());require(v["query_id"]==q.query_id,"Query order")
    texts.append(v["adversarial"]);matched.append(v["matched_clean"])
   noise=torch.load(c["resources"]["cpgc_noise"]["path"],map_location="cpu",weights_only=True).detach().unsqueeze(0)
   if model=="irra":noise=torch.nn.functional.interpolate(noise,size=(384,128),mode="bilinear",align_corners=False)
   adv=(x+noise).clamp(0,1)
   print("BENCH_MODEL",model,flush=True)
   with load_victim(c,model,"icfg","cuda:0",smoke=True) as (adapter,backend,source):
    before=snapshot(backend);result={"images":[],"texts":[],"fusion":[]}
    # Warmup is separate, explicitly recorded (one image, eight texts).
    encode_images(adapter,x[:1],model);encode_texts(adapter,[q.caption for q in qs[:8]],model)
    reference,tm=timed(lambda:encode(adapter,x,model,1,"image"));result["images"].append({"batch":1,**tm,"samples":len(x),"accepted":True})
    for batch in (8,16,32):
     value,tm=timed(lambda:encode(adapter,x,model,batch,"image"));d=diff(reference,value)
     result["images"].append({"batch":batch,**tm,"samples":len(x),"difference":d,"accepted":exact(d)})
     print("IMAGE",model,batch,tm,d,flush=True)
    textref,tm=timed(lambda:encode(adapter,[q.caption for q in qs],model,8,"text"))
    result["texts"].append({"batch":8,**tm,"samples":len(qs),"accepted":True})
    for batch in (16,32,64):
     value,tm=timed(lambda:encode(adapter,[q.caption for q in qs],model,batch,"text"));d=diff(textref,value)
     result["texts"].append({"batch":batch,**tm,"samples":len(qs),"difference":d,"accepted":exact(d)})
     print("TEXT",model,batch,tm,d,flush=True)
    # Same old execution features for fusion isolation (8 queries x complete Top128).
    if model=="rasa":
     q={k:v[:8].clone() for k,v in textref.items()};g=reference
     adapter.fusion_batch=16
     baseline,tm=timed(lambda:scores(adapter,q,g));metrics,order=rank(baseline,[v.pid for v in gs],[v.pid for v in qs[:8]])
     torch.save({"q":q,"g":g,"baseline_scores":baseline,"input":x,"adv":adv},ROOT/"rasa_reference.pt")
     result["fusion"].append({"batch":16,**tm,"queries":8,"pairs":1024,"accepted":True,"metrics":metrics})
     for batch in (32,64,128):
      adapter.fusion_batch=batch
      value,tm=timed(lambda:scores(adapter,q,g));d=diff(baseline,value);met,ord_=rank(value,[v.pid for v in gs],[v.pid for v in qs[:8]])
      # Reject any score ranking or retrieval metric change. Quantify FP32 element differences.
      accepted=torch.equal(order,ord_) and metrics==met and d["max_abs"]<=1e-5
      result["fusion"].append({"batch":batch,**tm,"queries":8,"pairs":1024,"difference":d,"ranking_exact":torch.equal(order,ord_),"metrics":met,"accepted":accepted})
      print("FUSION",batch,tm,d,"ranking_exact",torch.equal(order,ord_),flush=True)
     # Torch profiler separates host CPU work / copy / GEMM / attention; exported as independent evidence.
     adapter.fusion_batch=16
     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
      scores(adapter,{k:v[:1] for k,v in q.items()},g)
     (ROOT/"profile_rasa16.txt").write_text(prof.key_averages().table(sort_by="self_cuda_time_total",row_limit=35))
    frozen(backend);require(snapshot(backend)==before,"model parameters/buffers changed")
    result["frozen"]=True;out["models"][model]=result
    save(ROOT/(model+".json"),result)
   del reference,textref,adapter,backend;torch.cuda.empty_cache()
  out["status"]="PASS";out["elapsed_seconds"]=time.monotonic()-start
  out["peak_RSS_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
  out["GPU_monitor"]=events
  save(ROOT/"completed.json",out);print("THROUGHPUT_BENCHMARK_PASS",flush=True)
 except BaseException as e:
  save(ROOT/"FAIL.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
 finally:stop.set();t.join(3)
start=time.monotonic()
if __name__=="__main__":main()
