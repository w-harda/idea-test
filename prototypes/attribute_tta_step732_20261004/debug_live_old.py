"""单21:0旧runner NEW循环与新SEQUENTIAL的live tensor比较；只debug。"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run as c
import prefix_tta as p
from prefix_tta import *
from attribute_tta import prepared_plan, TextTransport,seed_all
from controlled_tta import tensor_hash,metric_delta,ControlledTTA,rng_fingerprint
from attributes.irra_surrogate import IRRASurrogate
from PIL import Image
import json
def td(a,b):
 d=(a-b).double();bn=float(b.double().norm())
 return {"max_abs":float(d.abs().max()),"l2":float(d.norm()),"relative_l2":float(d.norm())/bn if bn else None,
         "different_elements":int((a!=b).sum()),"existing_fp32_allclose":torch.allclose(a,b,atol=2e-6,rtol=1e-6)}
def main():
 out=ROOT/"outputs/regression_live_old_21"
 if out.exists():raise ValueError("no overwrite")
 out.mkdir()
 hashes,rows,payloads,bases,protocol,history=c.inputs();cid="cuhk:21:0"
 sys.path.insert(0,str(REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
 import learned_reader as v4
 v4.setup()
 torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False)
 torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim);gallery,idx,_=c.old.load_gallery(hashes)
 rec=next(x for x in rows if x["row_id"]==cid);plan=prepared_plan(next(x for x in payloads if x["caption_id"]==cid),rec,bases,"cuda")
 with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/rec["paired_image"]) as im:image=victim.pixels(im).unsqueeze(0).cuda()
 transport=TextTransport(victim,rec["caption"]);vec={a.instance_id:transport.read(a)[0] for a in plan}
 # 旧runner确实在NEW前编码了RAW，但NEW不使用它；这里只复现准备次序，不运行RAW。
 with torch.no_grad():unused_raw=source.encode_texts([a.metadata["raw_text"] for a in plan]).detach()
 ev=c.old.RetrievalMetrics(source,transport.clean,gallery,idx[rec["paired_image"]],tau=.02)
 clean=ev(image);seed_all(42);tape=CategoricalTape()
 eng=ControlledTTA(source,image,"NEW",steps=10,transforms=6,scales=tuple(protocol["scales"]),tape=tape)
 current=image.clone();actual_changes=[];old_images=[];old_m=[];old_states=[dict(state_index=0,**clean)]
 for a in plan:
  before=current
  current=eng.step(current,a,vec[a.instance_id],["attribute_unit:"+a.instance_id])
  actual_changes.append((a,(current-before).detach().clone()))  # 原Step73旧循环分配顺序
  metric=ev(current);old_states.append(dict(state_index=a.selection_rank,**metric))
  old_images.append(current.cpu());old_m.append(eng.momentum.cpu())
 old_trace=tape.finish();old_rng=rng_fingerprint()
 cached={m:cached_round_guidance(m,plan,vec) for m in MODES}
 class Capture(ControlledTTA):
  def __init__(self,*a,**kw):super().__init__(*a,**kw);self.images=[];self.ms=[]
  def step(self,*a,**kw):
   out=super().step(*a,**kw);self.images.append(out.cpu());self.ms.append(self.momentum.cpu());return out
 capture=[]
 def factory(*a,**kw):
  e=Capture(*a,**kw);capture.append(e);return e
 p.ControlledTTA=factory
 seed_all(42);nt=CategoricalTape()
 now,inputs,momentum=execute(source,image,plan,cached["SEQUENTIAL_NEW"],ev,steps=10,transforms=6,scales=tuple(protocol["scales"]),tape=nt)
 report={"caption_id":cid,"old_historical_images_exact":[tensor_hash(x)==r["post_projection_image_sha256"] for x,r in zip(old_images,history[cid]["modes"]["NEW"]["rounds"])],
    "old_historical_metrics_exact":old_states==history[cid]["modes"]["NEW"]["states"],
    "trace_live_equal":old_trace==now["random_trace"],"rng_live_equal":old_rng==now["rng_end_sha256"],
    "rounds":[{"round":i+1,"output_image":td(n,o),"momentum":td(nm,om),"metrics_current":now["states"][i+1],
               "metrics_old_live":old_states[i+1]} for i,(n,o,nm,om) in enumerate(zip(capture[0].images,old_images,capture[0].ms,old_m))]}
 c.save(out/"report.json",report)
 torch.save({"old_images":old_images,"new_images":capture[0].images,"old_momentum":old_m,"new_momentum":capture[0].ms},out/"tensors.pt")
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
