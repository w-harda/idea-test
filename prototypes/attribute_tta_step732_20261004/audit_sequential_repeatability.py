"""仅35:1 SEQUENTIAL重复性审计，3--5次，无PREFIX。"""
import sys, os, json, copy, itertools, argparse, hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run as runner, prefix_tta as p, controlled_tta as c
from attribute_tta import prepared_plan,TextTransport,seed_all,IMAGE_EPSILON,digest
from attributes.irra_surrogate import IRRASurrogate
from PIL import Image
import torch
def diff(a,b):
 a=a.detach().cpu();b=b.detach().cpu()
 delta=a.double()-b.double();x=delta.abs();rel=x/b.double().abs().clamp_min(1e-30)
 ids=(a!=b).nonzero();first=ids[0].tolist() if len(ids) else None
 norms=a.double().norm(),b.double().norm(),delta.norm()
 cos=float(torch.nn.functional.cosine_similarity(a.double().flatten(),b.double().flatten(),dim=0)) if min(norms[:2])>0 else None
 return {"exact":torch.equal(a,b),"max_abs_diff":float(x.max()),"max_rel_diff":float(rel.max()),
   "l2_diff":float(norms[2]),"relative_l2_diff":float(norms[2]/norms[1]) if norms[1]>0 else None,"cosine":cos,
   "changed_elements":len(ids),"first_index":first,
   "first_current":float(a[tuple(first)]) if first else None,"first_reference":float(b[tuple(first)]) if first else None}
class Observer(c.UpdateObserver):
 instances=[]
 def __init__(self):
  super().__init__();self.tensors=[];self.initial=None;self.instances.append(self)
 def pre_step(self,before,sign_step,ig_grad,momentum):
  self.tensors.append({"input":before.detach(),"IG_gradient":ig_grad.detach(),"momentum":momentum.detach(),"pre_projection_update":sign_step.detach()})
  return super().pre_step(before,sign_step,ig_grad,momentum)
 def after_clamp(self,after,incoming,original):
  if self.before is None:self.initial=after.detach()
  else:self.tensors[-1]["post_projection_image"]=after.detach()
  return super().after_clamp(after,incoming,original)
class Capture(c.ControlledTTA):
 instances=[]
 def __init__(self,*args,**kw):
  super().__init__(*args,**kw);self.inputs=[];self.outputs=[];self.momenta=[];self.losses=[];self.lowlevel=[]
  self.instances.append(self)
  for name in ("loss_func1","ig","transform_ig_batch"):
   fn=getattr(self.attack,name)
   def wrap(*args,_fn=fn,_name=name,**kwargs):
    v=_fn(*args,**kwargs)
    if _name=="loss_func1":self.losses.append(v.detach())
    elif _name=="ig":self.lowlevel.append({"kind":"sole_IG","value":v.detach()})
    else:self.lowlevel.append({"kind":"transformed_IG","value":v[0].detach()})
    return v
   setattr(self.attack,name,wrap)
 def step(self,current,attribute,features,labels):
  self.inputs.append(current.detach())
  y=super().step(current,attribute,features,labels)
  self.outputs.append(y.detach());self.momenta.append(self.momentum.detach())
  return y
def cpu_tree(x):
 if torch.is_tensor(x):return x.detach().cpu()
 if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu_tree(v) for v in x]
 return x
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--runs",type=int,choices=(3,4,5),default=5);a=ap.parse_args()
 out=a.output.resolve()
 if out.exists() or not out.is_relative_to(Path("/home/lzf/ldx")):raise ValueError("safe new output")
 hashes,rows,payloads,bases,protocol,history=runner.inputs();out.mkdir()
 sources={str(x):digest(x) for x in [ROOT/"prefix_tta.py",ROOT/"run.py",Path(c.__file__)]}
 sys.path.insert(0,str(p.REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
 import learned_reader as v4
 v4.setup();torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False)
 torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim);versions={n:t._version for n,t in victim.named_parameters()}
 gallery,idx,_=runner.old.load_gallery(hashes)
 cid="cuhk:35:1";row=next(r for r in rows if r["row_id"]==cid);payload=next(x for x in payloads if x["caption_id"]==cid)
 plan=prepared_plan(payload,row,bases,"cuda")
 with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/row["paired_image"]) as im:image=victim.pixels(im).unsqueeze(0).cuda()
 tr=TextTransport(victim,row["caption"]);vec={a.instance_id:tr.read(a)[0] for a in plan}
 cached=p.cached_round_guidance("SEQUENTIAL_NEW",plan,vec)
 ev=runner.old.RetrievalMetrics(source,tr.clean,gallery,idx[row["paired_image"]],tau=.02)
 p.ControlledTTA=Capture;c.UpdateObserver=Observer
 # 所有run均走同一个record-mode代码路径，独立reset RNG/engine/momentum。
 runs=[];ts=[]
 for i in range(a.runs):
  Capture.instances=[];Observer.instances=[]
  seed_all(42);tape=c.CategoricalTape();cp=[]
  data,inputs,m=p.execute(source,image,plan,cached,ev,steps=10,transforms=6,scales=tuple(protocol["scales"]),tape=tape,first_round_capture=cp)
  en=Capture.instances[0]
  tensors=cpu_tree({"round_inputs":en.inputs,"round_outputs":en.outputs,"round_momentum":en.momenta,"losses":en.losses,
     "lowlevel":en.lowlevel,"steps":[o.tensors for o in Observer.instances],
     "initializations":[o.initial for o in Observer.instances],"initial_image":image,"JVP":vec,"tape_records":tape.records})
  torch.save(tensors,out/f"run{i+1}_tensors.pt");runner.save(out/f"run{i+1}.json",data)
  runs.append(data);ts.append(tensors)
  print("RUN_DONE",i+1,"states",data["states"],"image_SHA",[x["post_projection_image_sha256"] for x in data["rounds"]],flush=True)
  del en,m,cp,inputs;Capture.instances=[];Observer.instances=[]
 pairs=[]
 for i,j in itertools.combinations(range(a.runs),2):
  r={"runs":[i+1,j+1],"controls":{k:runs[i][k]==runs[j][k] for k in ("fixed_sequence","noise_sha256","random_trace","rng_end_sha256")},"rounds":[]}
  assert all(r["controls"].values()),r["controls"]
  for k in range(len(plan)):
   rr={"round":k+1,"input_image":diff(ts[j]["round_inputs"][k],ts[i]["round_inputs"][k]),
       "output_image":diff(ts[j]["round_outputs"][k],ts[i]["round_outputs"][k]),
       "momentum":diff(ts[j]["round_momentum"][k],ts[i]["round_momentum"][k]),"steps":[]}
   for step,(sa,sb) in enumerate(zip(ts[j]["steps"][k],ts[i]["steps"][k]),1):
    rr["steps"].append({"step":step,**{key:diff(sa[key],sb[key]) for key in sa}})
   rr["metrics"]={key:{"max_abs_diff":abs(runs[j]["states"][k+1][key]-runs[i]["states"][k+1][key]),
                        "max_rel_diff":abs(runs[j]["states"][k+1][key]-runs[i]["states"][k+1][key])/max(abs(runs[i]["states"][k+1][key]),1e-30)}
                   for key in ("paired_cosine","paired_soft_rank_F","positive_image_rank")}
   r["rounds"].append(rr)
  r["losses"]={"count":len(ts[i]["losses"]),"first_difference":next(({"index":k,**diff(a,b)} for k,(a,b) in enumerate(zip(ts[j]["losses"],ts[i]["losses"])) if not torch.equal(a,b)),None)}
  r["lowlevel"]={"first_difference":next(({"index":k,"kind":aa["kind"],**diff(aa["value"],bb["value"])} for k,(aa,bb) in enumerate(zip(ts[j]["lowlevel"],ts[i]["lowlevel"])) if not torch.equal(aa["value"],bb["value"])),None)}
  pairs.append(r)
 prior=history[cid]["modes"]["NEW"]
 failed=runner.load(ROOT/"outputs/diagnostic_round1_fixed/regression_failure.json")["current_sequential"]
 historical=[]
 for i,data in enumerate(runs):
  historical.append({"run":i+1,"controls_exact":all(data[k]==prior[k] for k in ("fixed_sequence","noise_sha256","random_trace","rng_end_sha256")),
      "state_differences":[{key:data["states"][r][key]-prior["states"][r][key] for key in prior["states"][r]} for r in range(3)],
      "image_sha_matches_history":[data["rounds"][r]["post_projection_image_sha256"]==prior["rounds"][r]["post_projection_image_sha256"] for r in range(2)],
      "image_sha_matches_failed":[data["rounds"][r]["post_projection_image_sha256"]==failed["rounds"][r]["post_projection_image_sha256"] for r in range(2)]})
 # 仅诊断deterministic mode可用性，不替换正常重复实验、不给正式方法设flag。
 det={"diagnostic_only":True};seed_all(42)
 olddet=torch.are_deterministic_algorithms_enabled()
 torch.use_deterministic_algorithms(True)
 try:
  p.execute(source,image,plan[:1],cached[:1],ev,steps=1,transforms=6,scales=tuple(protocol["scales"]),tape=c.CategoricalTape())
  det["one_step_completed"]=True
 except RuntimeError as e:
  det["one_step_completed"]=False;det["error"]=str(e)
 finally:torch.use_deterministic_algorithms(olddet)
 runner.old.verify(hashes);runner.old.verify(sources)
 assert all(not t.requires_grad and t.grad is None and t._version==versions[n] for n,t in victim.named_parameters())
 report={"caption_id":cid,"runs":a.runs,"current_vs_current":pairs,"current_vs_saved":historical,"deterministic_probe":det,
         "frozen_SHA":hashes,"audit_source_SHA":sources,"frozen":True,"prefix_runs":0,"full25_run":False,
         "initial_image_sha":c.tensor_hash(image),"JVP_SHA":{k:c.tensor_hash(v) for k,v in vec.items()}}
 runner.save(out/"report.json",report)
 compact={"runs":a.runs,"saved_comparisons":historical,"deterministic_probe":det,
   "rounds":[{"round":r+1,"image_max_abs":max(x["rounds"][r]["output_image"]["max_abs_diff"] for x in pairs),
             "image_changed_max":max(x["rounds"][r]["output_image"]["changed_elements"] for x in pairs),
             "gradient_max_abs":max(s["IG_gradient"]["max_abs_diff"] for x in pairs for s in x["rounds"][r]["steps"]),
             "momentum_max_abs":max(s["momentum"]["max_abs_diff"] for x in pairs for s in x["rounds"][r]["steps"]),
             "metrics_max_abs":{k:max(x["rounds"][r]["metrics"][k]["max_abs_diff"] for x in pairs) for k in ("paired_cosine","paired_soft_rank_F","positive_image_rank")}} for r in range(2)],
   "first_lowlevel_differences":[x["lowlevel"] for x in pairs],"first_loss_differences":[x["losses"] for x in pairs]}
 runner.save(out/"summary.json",compact);print("REPEATABILITY_SUMMARY",json.dumps(compact),flush=True)
if __name__=="__main__":main()
