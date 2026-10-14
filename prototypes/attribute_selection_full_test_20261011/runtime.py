"""只读接入冻结实现；新 test 缓存与共享图库在本 prototype 内。"""
import sys,os,copy,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import core as c
sys.dont_write_bytecode=True
OLD=c.REPO/"prototypes/attribute_tta_selection_value_20261005"
sys.path.insert(0,str(OLD))
import deps as frozen
import torch
from PIL import Image
from engines import execute_three,plans_for
import engines
from selection_audit import selection
from controlled_tta import RetrievalMetrics,tensor_hash
from attribute_tta import TextTransport,IMAGE_EPSILON
from attributes.irra_surrogate import IRRASurrogate
import attributes
# 旧block8 probe位于已锁SHA的历史快照；只扩展模块查找路径，不修改源码。
probe_package="/home/lzf/ldx/outputs/idea-TBPS-test1/stage03-attribute-unit-v2-20261002/code/src/attributes"
if probe_package not in attributes.__path__:attributes.__path__.append(probe_package)
score=frozen.module("full_test_frozen_score_runner",frozen.P51/"run.py")
sys.path.insert(0,str(c.REPO/"prototypes/attribute_selection_pid_metrics_20261011"))
from pid_metrics import aggregate_victims,drop_rate
from attributes.tbps_retrieval_metrics import text_to_image_metrics,aggregate_rows
def locks():
 # 当前100句成功protocol锁过的来源；不读取/复用train feature/score缓存。
 prior=c.load(OLD/"outputs/diagnostic/protocol.json")["frozen_sha256"]
 essentials={str(c.RAW),str(c.LABELS),str(frozen.V4/"outputs/split.json"),
 str(frozen.V4/"outputs/reader_final.pt"),str(frozen.V55/"outputs/full/semantic_bases.json"),
 "/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth"}
 lock={p:h for p,h in prior.items() if Path(p).suffix in (".py",".yaml",".gz") or p in essentials}
 if not essentials<=set(lock):raise ValueError("essential frozen source absent: "+str(essentials-set(lock)))
 c.verify(lock)
 lock.update({str(p):c.sha(p) for p in sorted(c.ROOT.glob("*.py"))})
 return lock
def setup():
 score.v4.setup()
 torch.backends.mha.set_fastpath_enabled(False)
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 torch.backends.cuda.enable_math_sdp(True)
 victim=score.v4.load_victim()
 return victim
def assert_frozen(victim,versions,reader=None,reader_state=None):
 if any(p.requires_grad or p.grad is not None or p._version!=versions[n] for n,p in victim.named_parameters()):
  raise ValueError("IRRA state changed")
 if reader is not None:
  if any(p.requires_grad or p.grad is not None for p in reader.parameters()):raise ValueError("reader gradients")
  if any(not torch.equal(v.detach().cpu(),reader_state[k]) for k,v in reader.state_dict().items()):raise ValueError("reader state changed")
def pixels(victim,row):
 with Image.open(c.IMAGES/row["image"]) as im:return victim.pixels(im).unsqueeze(0).cuda()
def tensor_save(path,value):
 path=c.safe(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp")
 torch.save(value,tmp);os.replace(tmp,path)
def normalized(x,count):
 if x.shape!=(count,512) or x.dtype!=torch.float32 or not torch.isfinite(x).all() or not torch.allclose(x.norm(dim=1),torch.ones(count,device=x.device),atol=2e-4,rtol=0):
  raise ValueError("invalid normalized feature cache")
def fresh_localizer(victim):
 split=c.load(frozen.V4/"outputs/split.json");bases=c.load(score.BASIS)
 reader=score.v4.SlotReader([len(v) for v in split["values"]]).cuda()
 cp=torch.load(frozen.V4/"outputs/reader_final.pt",map_location="cuda",weights_only=False)
 if cp["epoch"]!=20 or cp["slots"]!=split["slots"] or cp["values"]!=split["values"]:raise ValueError("reader checkpoint metadata")
 reader.load_state_dict(cp["state_dict"]);reader.eval().requires_grad_(False)
 # FrozenLocalization.attach(cache_index=None)已有fresh路径，仅需要tokenizer/alignment接口，不实例化train memmap。
 from attributes.irra_token_alignment import align_irra_tokens,_span_token_indices
 alignment=SimpleNamespace(tokenizer=victim.tokenizer,align=align_irra_tokens,span_tokens=_span_token_indices)
 return reader,bases,score.FrozenLocalization(reader,split,bases,alignment)
def score_one(victim,reader,localizer,bases,features,row,label):
 if (label["row_id"],label["caption"],label["image"],label["split"])!=(row["row_id"],row["caption"],row["image"],"test"):
  raise ValueError("test Stage01 provenance mismatch")
 probe=score.AttributeUnitV2Probe(victim,features,row["index"],tau=.02)
 with patch("torch.autograd.grad",wraps=torch.autograd.grad) as calls:
  feature,H,G,F,cos,zero=score.baseline_gradient(probe,row["caption"])
  if H.dtype!=torch.float32 or G.dtype!=torch.float32 or zero.any():raise ValueError("block8 fp32 contract")
  with torch.no_grad():
   ref=victim.encode_texts([row["caption"]]);explicit,_,z=probe.encode(row["caption"],8,torch.zeros_like(H))
  if not torch.equal(feature,ref) or not torch.equal(explicit,ref) or z.any():raise ValueError("zero not exact")
  attrs=score.convert_record(label)
  attrs,weights,alignment,regression=localizer.attach(attrs,row["caption"],H,None)
  rows=score.score_instances(attrs,row["caption"],G,weights,bases,None)
  if calls.call_count!=1:raise ValueError("each caption exactly one clean retrieval backward")
 probe.assert_frozen()
 compact=[{k:copy.deepcopy(a[k]) for k in ("instance","instance_id","status","rank","S_raw","w") if k in a} for a in rows]
 rec={"row_id":row["row_id"],"pid":row["pid"],"caption":row["caption"],"paired_image":row["image"],
 "paired_gallery_index":row["index"],"F":F,"paired_cosine":cos,"retrieval_backward_count":1,
 "zero_intervention_exact":True,"attributes":compact,"gallery_scope":"clean test shared gallery"}
 return rec,c.selected(selection(rec))
def run_three(victim,source,image,record,payload,bases,features,limit):
 if payload["natural_k"]==0:
  feature=source.encode_images(image).detach()
  metric=RetrievalMetrics(source,TextTransport(victim,record["caption"]).clean,features,record["paired_gallery_index"],tau=.02)(image)
  return {m:{"states":[{"state_index":0,**metric}],"rounds":[],"fixed_sequence":[],"final_linf":0.,
  "execution_reuse_of":None if m==c.MODES[0] else c.MODES[0]} for m in c.MODES},{m:image.detach() for m in c.MODES},{"K0_clean_passthrough":True}
 plans=plans_for(payload,record,bases,"cuda")
 transport=TextTransport(victim,record["caption"]);vectors={};mapping=[]
 for a in {a.instance_id:a for plan in plans.values() for a in plan}.values():
  v,check=transport.read(a)
  if not check["zero_exact"] or check["central_fd_relative_error"]>.02:raise ValueError("JVP FD/zero gate")
  vectors[a.instance_id]=v;mapping.append(check)
 vh={i:tensor_hash(v) for i,v in vectors.items()}
 evaluator=RetrievalMetrics(source,transport.clean,features,record["paired_gallery_index"],tau=.02)
 captured={};original=engines.original.execute
 def capture(*args,**kwargs):
  box=[];kwargs["final_image_capture"]=box
  result=original(*args,**kwargs)
  if len(box)!=1:raise ValueError("missing final pixels")
  captured[tuple(a.instance_id for a in args[2])]=box[0].detach()
  return result
 before=copy.deepcopy(payload)
 with patch.object(engines.original,"execute",side_effect=capture):
  results,checks=execute_three(source,image,plans,vectors,evaluator,observed_limit=limit)
 if before!=payload or vh!={i:tensor_hash(v) for i,v in vectors.items()}:raise ValueError("frozen instance/JVP changed")
 images={m:captured[tuple(payload["selection_ids"][m])] for m in c.MODES}
 for m,result in results.items():
  if result["fixed_sequence"]!=payload["selection_ids"][m] or len(result["rounds"])!=payload["natural_k"]:raise ValueError("sequence/K")
  if any(len(e["steps"])!=(limit or 10) or e["configured_steps_per_round"]!=10 for e in result["rounds"]):raise ValueError("steps")
  if any(s["cumulative_perturbation_linf"]>IMAGE_EPSILON+1e-6 for e in result["rounds"] for s in e["steps"]):raise ValueError("step budget")
  if float((images[m]-image).abs().max())>IMAGE_EPSILON+1e-6:raise ValueError("final budget")
 checks["JVP_mapping"]=mapping
 return results,images,checks
