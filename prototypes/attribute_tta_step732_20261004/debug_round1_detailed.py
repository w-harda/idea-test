"""仅失败35:1及授权短round1回归；不运行25句。"""
import sys, os, json, argparse, copy
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run as runner
import prefix_tta as p
import controlled_tta as c
from attribute_tta import prepared_plan,TextTransport,seed_all,IMAGE_EPSILON
from attributes.irra_surrogate import IRRASurrogate
from PIL import Image
import torch

def diff(a,b):
 a=a.detach().cpu();b=b.detach().cpu()
 x=(a.double()-b.double()).abs(); rel=x/b.double().abs().clamp_min(1e-30)
 where=(a!=b).nonzero(); first=where[0].tolist() if len(where) else None
 return {"exact":torch.equal(a,b),"max_abs_diff":float(x.max()),"max_rel_diff":float(rel.max()),
         "allclose_existing":torch.allclose(a,b,atol=2e-6,rtol=1e-6),"changed_elements":len(where),
         "first_index":first,"first_current":float(a[tuple(first)]) if first else None,
         "first_reference":float(b[tuple(first)]) if first else None}
class DebugObserver(c.UpdateObserver):
 instances=[]
 def __init__(self):
  super().__init__();self.tensors=[];self.initial_tensor=None;self.instances.append(self)
 def pre_step(self,before,sign_step,ig_grad,momentum):
  self.tensors.append({"before":before.detach(),"sign_step":sign_step.detach(),
                       "ig_gradient":ig_grad.detach(),"momentum":momentum.detach()})
  return super().pre_step(before,sign_step,ig_grad,momentum)
 def after_clamp(self,after,incoming,original):
  if self.before is None:self.initial_tensor=after.detach()
  else:self.tensors[-1]["after"]=after.detach()
  return super().after_clamp(after,incoming,original)
class Captured(c.ControlledTTA):
 instances=[]
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.debug_images=[];self.debug_momentum=[];self.debug_features=[];self.debug_inputs=[]
  self.instances.append(self);self.losses=[]
  for name in ("loss_func","loss_func1"):
   fn=getattr(self.attack,name)
   def wrapper(*args,_fn=fn,_name=name,**kwargs):
    y=_fn(*args,**kwargs);self.losses.append((_name,y.detach()));return y
   setattr(self.attack,name,wrapper)
 def step(self,current,attribute,features,labels):
  self.debug_inputs.append(current.detach());self.debug_features.append(features.detach())
  value=super().step(current,attribute,features,labels)
  self.debug_images.append(value.detach());self.debug_momentum.append(self.momentum.detach())
  return value

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);opt=ap.parse_args()
 out=opt.output.resolve()
 if out.exists() or not out.is_relative_to(Path("/home/lzf/ldx")):raise ValueError("safe new output required")
 hashes,rows,payloads,bases,protocol,history=runner.inputs();out.mkdir()
 sys.path.insert(0,str(p.REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
 import learned_reader as v4
 v4.setup();torch.backends.mha.set_fastpath_enabled(False)
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim)
 versions={n:t._version for n,t in victim.named_parameters()}
 gallery,idx,_=runner.old.load_gallery(hashes)
 cid="cuhk:35:1"
 row=next(r for r in rows if r["row_id"]==cid);payload=next(r for r in payloads if r["caption_id"]==cid)
 plan=prepared_plan(payload,row,bases,"cuda")
 with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/row["paired_image"]) as im:image=victim.pixels(im).unsqueeze(0).cuda()
 tr=TextTransport(victim,row["caption"]);vectors={a.instance_id:tr.read(a)[0] for a in plan}
 cached={m:p.cached_round_guidance(m,plan,vectors) for m in p.MODES}
 ev=runner.old.RetrievalMetrics(source,tr.clean,gallery,idx[row["paired_image"]],tau=.02)
 p.ControlledTTA=Captured;c.UpdateObserver=DebugObserver
 plan=plan[:1];cached={m:cached[m][:1] for m in p.MODES}
 results={};tapes=None
 for m in p.MODES:
  seed_all(42);tape=c.CategoricalTape(tapes)
  results[m],_,_=p.execute(source,image,plan,cached[m],ev,steps=10,transforms=6,scales=tuple(protocol["scales"]),tape=tape)
  if m=="SEQUENTIAL_NEW":tapes=copy.deepcopy(tape.records)
 seq,pre=Captured.instances
 fields={key:results[p.MODES[0]][key]==results[p.MODES[1]][key] for key in ("random_trace","noise_sha256","rng_end_sha256","fixed_sequence")}
 rd={"caption_id":cid,"k":len(plan),"controls":fields,"round1":{},"states":results}
 for key,a,b in [("input_image",pre.debug_inputs[0],seq.debug_inputs[0]),
                 ("JVP",pre.debug_features[0],seq.debug_features[0]),
                 ("output_image",pre.debug_images[0],seq.debug_images[0]),
                 ("momentum",pre.debug_momentum[0],seq.debug_momentum[0])]:
  rd["round1"][key]=diff(a,b)
 rd["round1"]["metrics"]={key:{"exact":a==results[p.MODES[0]]["states"][1][key],
                   "current":a,"reference":results[p.MODES[0]]["states"][1][key],
                   "max_abs_diff":abs(a-results[p.MODES[0]]["states"][1][key]),
                   "max_rel_diff":abs(a-results[p.MODES[0]]["states"][1][key])/max(abs(results[p.MODES[0]]["states"][1][key]),1e-30)}
                     for key,a in results[p.MODES[1]]["states"][1].items()}
 prior=history[cid]["modes"]["NEW"]
 rd["saved_round1_image_exact"]=results[p.MODES[0]]["rounds"][0]["post_projection_image_sha256"]==prior["rounds"][0]["post_projection_image_sha256"]
 rd["steps"]=[]
 oa,ob=DebugObserver.instances
 for i,(a,b) in enumerate(zip(ob.tensors,oa.tensors),1):
  rd["steps"].append({"step":i,**{key:diff(a[key],b[key]) for key in a}})
 rd["initialization_tensor"]=diff(ob.initial_tensor,oa.initial_tensor)
 rd["loss_calls"]=len(seq.losses)
 rd["loss_comparisons"]=[{"index":i,"function":a[0],**diff(a[1],b[1])} for i,(a,b) in enumerate(zip(pre.losses,seq.losses))]
 rd["round_start_gradient"]=diff(pre.previous_raw_gradient,seq.previous_raw_gradient)
 rd["budget"]={m:results[m]["final_linf"] for m in p.MODES}
 runner.old.verify(hashes)
 rd["frozen"]=all(not t.requires_grad and t.grad is None and versions[n]==t._version for n,t in victim.named_parameters())
 torch.save({m:{"images":[x.cpu() for x in en.debug_images],"inputs":[x.cpu() for x in en.debug_inputs],
               "momentum":[x.cpu() for x in en.debug_momentum],"JVP":[x.cpu() for x in en.debug_features]} for m,en in zip(p.MODES,Captured.instances)},out/"tensors.pt")
 runner.save(out/"report.json",rd)
 print(json.dumps({k:v for k,v in rd.items() if k not in ("states","steps","loss_comparisons")},ensure_ascii=False),flush=True)
 print("FIRST_STEP_DIFFERENCE",next((x for x in rd["steps"] if any(not v["exact"] for k,v in x.items() if k!="step")),None),flush=True)
 print("FIRST_LOSS_DIFFERENCE",next((x for x in rd["loss_comparisons"] if not x["exact"]),None),flush=True)
 print("MAX_GRADIENT_ABS",max(x["ig_gradient"]["max_abs_diff"] for x in rd["steps"]),flush=True)
 print("MAX_MOMENTUM_ABS",max(x["momentum"]["max_abs_diff"] for x in rd["steps"]),flush=True)
if __name__=="__main__":main()
