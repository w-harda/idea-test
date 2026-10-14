"""修复后只跑35:1、k1，及k3/k4两步round1；禁止25句。"""
import sys,copy,json,argparse
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run as runner, prefix_tta as p
from controlled_tta import CategoricalTape,tensor_hash
from attribute_tta import prepared_plan,TextTransport,seed_all,IMAGE_EPSILON
from attributes.irra_surrogate import IRRASurrogate
from PIL import Image
import torch
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);o=ap.parse_args().output.resolve()
 if o.exists() or not o.is_relative_to(Path("/home/lzf/ldx")):raise ValueError("safe new output")
 hashes,rows,payloads,bases,protocol,history=runner.inputs();o.mkdir()
 sys.path.insert(0,str(p.REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
 import learned_reader as v4
 v4.setup();torch.backends.mha.set_fastpath_enabled(False)
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim)
 versions={n:x._version for n,x in victim.named_parameters()}
 features,idx,_=runner.old.load_gallery(hashes)
 reports=[]
 for cid,full in [("cuhk:35:1",True),("cuhk:34:0",True),("cuhk:0:1",False),("cuhk:18:0",False)]:
  row=next(r for r in rows if r["row_id"]==cid);payload=next(r for r in payloads if r["caption_id"]==cid)
  original_plan=prepared_plan(payload,row,bases,"cuda");plan=original_plan if full else original_plan[:1]
  with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/row["paired_image"]) as im:image=victim.pixels(im).unsqueeze(0).cuda()
  tr=TextTransport(victim,row["caption"]);vec={};mapping=[]
  for a in plan:v,r=tr.read(a);vec[a.instance_id]=v;mapping.append(r)
  vector_hashes={k:tensor_hash(v) for k,v in vec.items()}
  cached={m:p.cached_round_guidance(m,plan,vec) for m in p.MODES}
  ev=runner.old.RetrievalMetrics(source,tr.clean,features,idx[row["paired_image"]],tau=.02)
  steps=10 if full else 2;transforms=6 if full else 1;scales=tuple(protocol["scales"]) if full else None
  seed_all(42);tape=CategoricalTape();cp=[]
  seq,_,sm=p.execute(source,image,plan,cached["SEQUENTIAL_NEW"],ev,steps=steps,transforms=transforms,scales=scales,tape=tape,first_round_capture=cp)
  # checkpoint保留只读状态；旧SEQ后续momentum无in-place修改。
  cm=cp[0]["momentum"].clone();ci=cp[0]["image"].clone()
  seed_all(42);replay=CategoricalTape(tape.records)
  pre,_,pm=p.execute(source,image,plan,cached["PREFIX_NEW"],ev,steps=steps,transforms=transforms,scales=scales,tape=replay,first_round_checkpoint=cp[0])
  checks={"round1_event_exact":pre["rounds"][0]==seq["rounds"][0],
          "round1_metrics_exact":pre["states"][:2]==seq["states"][:2],
          "common_image_unchanged":torch.equal(cp[0]["image"],ci),
          "common_momentum_unchanged":torch.equal(cp[0]["momentum"],cm),
          "JVP_unchanged":vector_hashes=={k:tensor_hash(v) for k,v in vec.items()},
          "controls_exact":all(pre[k]==seq[k] for k in ("noise_sha256","random_trace","rng_end_sha256","fixed_sequence"))}
  if len(plan)==1:checks["k1_full_exact"]=pre==seq and torch.equal(pm,sm)
  if not all(checks.values()):raise ValueError(checks)
  for mode,result in zip(p.MODES,[seq,pre]):p.budget_and_prefix_checks(mode,result,plan,IMAGE_EPSILON)
  reg=p.saved_regression(seq,history[cid]["modes"]["NEW"]) if full else {"not_same_config":"two-step round1 only"}
  r={"caption_id":cid,"original_k":len(original_plan),"executed_rounds":len(plan),"steps":steps,"transforms":transforms,
     "saved_sequential":reg,"checks":checks,"round1_max_abs_diff":0,"round1_max_rel_diff":0,
     "max_linf":max(seq["final_linf"],pre["final_linf"]),"mapping":mapping}
  reports.append(r);runner.save(o/(cid.replace(":","_")+".json"),{"report":r,"SEQUENTIAL_NEW":seq,"PREFIX_NEW":pre})
  print("CASE_PASS",json.dumps(r),flush=True)
 runner.old.verify(hashes)
 frozen=all(not t.requires_grad and t.grad is None and t._version==versions[n] for n,t in victim.named_parameters())
 if not frozen:raise ValueError("frozen state")
 runner.save(o/"completed.json",{"cases":reports,"frozen":True,"SHA_count":len(hashes),"full25_run":False})
 print("COMPLETE_SHORT_VERIFICATION",flush=True)
if __name__=="__main__":main()
