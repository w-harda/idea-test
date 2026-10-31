"""真实完整ICFG Gallery上的Top128、chunk、融合分数与精确排名对照。"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,time,resource,hashlib
import torch
from tbps.artifacts import sha_file
from tbps.baselines.cpgc_sep.runtime import config,resources
from tbps.baselines.cpgc_sep.verified_store import VerifiedArtifactStore
from tbps.baselines.cpgc_sep.cache import FeatureBank,BlockCache,NativeScorer
from tbps.baselines.cpgc_sep.dataset import read_test,require
from tbps.baselines.cpgc_sep.victims import load_victim,snapshot,frozen
from tbps.models.base import RepresentationBatch
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-throughput-validation-v3")
def write(p,x):
 with p.open("x")as f:json.dump(x,f,indent=2);f.write("\n")
def metrics(s,m):
 order=s.argsort(dim=1,descending=True);r=[0,0,0];aps=[]
 gp=[g.pid for g in m.gallery]
 for q,row in zip(m.queries,order):
  hits=torch.tensor([gp[j]==q.pid for j in row]);ix=hits.nonzero().flatten()
  aps.append(float((hits.cumsum(0)[ix]/(ix+1)).float().mean()) if len(ix) else 0.)
  for j,k in enumerate((1,5,10)):r[j]+=bool(hits[:k].any())
 return dict(zip(("R@1","R@5","R@10","mAP"),[x/len(order)*100 for x in r]+[sum(aps)/len(order)*100])),order
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--allow-gpu-benchmark",action="store_true");a=p.parse_args()
 require(a.allow_gpu_benchmark,"explicit GPU benchmark authorization");c=config(a.config);rs=resources(c)
 require(rs["memory.free"]>=20000,"candidate benchmark resources")
 out=BASE/"full_candidate_check";require(not out.exists(),"unique diagnostic consumed");out.mkdir()
 write(out/"Claim.json",{"configuration":a.config,"config_sha":sha_file(a.config),"resources":rs,"time":time.time()})
 torch.set_num_threads(2);t=time.monotonic()
 try:
  file=BASE/"cache_profile/full_global_reference.pt"
  require(file.is_file(),"read-only full Gallery globals must already be verified")
  data=torch.load(file,map_location="cpu",weights_only=True)
  old=Path("/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-uap-detach-20261029")
  cp=old/"configs/framework/cpgc_sep_tbps.uap_detach.server.json";parent=json.loads(cp.read_text());parent["sha256"]=sha_file(cp)
  m=read_test(c["datasets"]["icfg"]);root=Path(parent["output_base"])/"full/cpgc/icfg/rasa"
  with load_victim(c,"rasa","icfg","cuda:0",smoke=False)as(adapter,model,source):
   before=snapshot(model)
   store=VerifiedArtifactStore(root,new_base=parent["output_base"])
   # Parent feature bindings are retained as original identities; this diagnostic does not republish them.
   gb=FeatureBank(parent,store,"clean_gallery",adapter,m,tuple(g.image_id for g in m.gallery),tuple(g.image_sha256 for g in m.gallery),modality="image")
   qb=FeatureBank(parent,store,"clean_text",adapter,m,tuple(q.query_id for q in m.queries),tuple(hashlib.sha256(q.caption.encode()).hexdigest() for q in m.queries),modality="text")
   gb.globals=data["gallery"];qb.globals=data["queries"]
   require(len(gb.globals)==19848 and len(qb.globals)==128,"full original Gallery and first 128 Queries")
   qdev=qb.globals.to("cuda:0");gdev=gb.globals.to("cuda:0")
   choices={}
   for chunk in(32,64,128):
    torch.cuda.synchronize();tic=time.monotonic()
    values=torch.cat([qdev[i:i+chunk]@gdev.T for i in range(0,128,chunk)])
    ids=values.topk(128,dim=1).indices.cpu()
    torch.cuda.synchronize();choices[chunk]={"seconds":time.monotonic()-tic,"ids":ids,"global":values.cpu()}
   reference=choices[32];chunk_report={}
   for chunk,v in choices.items():
    chunk_report[chunk]={"seconds":v["seconds"],"top128_identity_and_order_exact":torch.equal(v["ids"],reference["ids"]),
     "different_candidate_positions":int((v["ids"]!=reference["ids"]).sum()),"global_max_abs":float((v["global"]-reference["global"]).abs().max())}
   results=[];reference_scores=None;order_ref=None;metrics_ref=None
   for batch in(16,32,64,128):
    adapter.fusion_batch=batch;cache=BlockCache(64*1024**3);gb.block_cache=cache;qb.block_cache=cache
    scorer=NativeScorer(adapter,"rasa",{"q":qb,"g":gb})
    parts=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tic=time.monotonic()
    for i in range(0,128,32):
     q=scorer.slice_query(RepresentationBatch(qb.globals,adapter.space.id,"text"),i,i+32)
     parts.append(scorer.score(q,RepresentationBatch(gb.globals,adapter.space.id,"image")).payload)
    matrix=torch.cat(parts);torch.cuda.synchronize();sec=time.monotonic()-tic
    met,order=metrics(matrix,m)
    if reference_scores is None:reference_scores=matrix;order_ref=order;metrics_ref=met
    error=(matrix-reference_scores).double()
    report={"fusion_batch":batch,"seconds":sec,"queries":128,"Gallery":19848,"pairs":128*128,
     "max_abs":float(error.abs().max()),"l2_difference":float(error.norm()),"relative_l2":float(error.norm()/reference_scores.double().norm()),
     "ranking_exact":torch.equal(order,order_ref),"metrics":met,"metrics_exact":met==metrics_ref,
     "allocated":torch.cuda.max_memory_allocated(),"reserved":torch.cuda.max_memory_reserved(),"cache":cache.statistics()}
    report["accepted"]=report["ranking_exact"]and report["metrics_exact"]and report["max_abs"]<=1e-5
    results.append(report);write(out/f"fusion_{batch}.json",report);print("FULL_CANDIDATE",json.dumps(report),flush=True)
    if batch==16:
     # Scores already completed by old native execution: independent CPU reference.
     orig=torch.cat([torch.load(root/f"scorechunk-clean_text--clean_gallery-{i:06d}/scores.pt",map_location="cpu",weights_only=True) for i in range(0,128,32)])
     require(torch.equal(orig,matrix),"original v2 score chunks not exactly reproduced")
     report["v2_score_tensor_exact"]=True
    del scorer,parts
   require(snapshot(model)==before,"model changed");frozen(model)
   # Retain existing query chunk unless the whole complete-Gallery Top128 sequence is exact.
   accepted=[v for v in results if v["accepted"]]
   best=min(accepted,key=lambda v:v["seconds"]-v["cache"]["read_seconds"])
   # Cache warmup/I/O dominates the cold first entry; selection additionally considers compute-only first benchmark.
   result={"status":"PASS","chunks":chunk_report,"fusion":results,"old_scores_exact":True,
    "selected_fusion_batch":best["fusion_batch"],"query_chunk":32,"elapsed_seconds":time.monotonic()-t,
    "RSS":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"config_sha256":sha_file(a.config),
    "notes":"first run cold cache; rank changes reject an optimization; keep chunk32 mathematical path"}
   write(out/"completed.json",result);torch.save(reference_scores,out/"reference_scores.pt")
   print("FULL_CANDIDATE_PARITY_PASS",flush=True)
 except BaseException as e:
  import traceback
  write(out/"FAIL.json",{"error":repr(e),"traceback":traceback.format_exc()});raise
if __name__=="__main__":main()
