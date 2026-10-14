"""仅35:1原配置一步deterministic诊断，不作为正式方法。"""
import os,sys,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run as runner,prefix_tta as p
from attribute_tta import prepared_plan,TextTransport,seed_all
from controlled_tta import CategoricalTape
from attributes.irra_surrogate import IRRASurrogate
import torch
from PIL import Image
hashes,rows,payloads,bases,protocol,history=runner.inputs()
sys.path.insert(0,str(p.REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
import learned_reader as v4
v4.setup();torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
victim=v4.load_victim();source=IRRASurrogate(victim);gallery,idx,_=runner.old.load_gallery(hashes)
row=next(r for r in rows if r["row_id"]=="cuhk:35:1");payload=next(x for x in payloads if x["caption_id"]=="cuhk:35:1")
plan=prepared_plan(payload,row,bases,"cuda")[:1]
with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/row["paired_image"]) as im:image=victim.pixels(im).unsqueeze(0).cuda()
tr=TextTransport(victim,row["caption"]);vec={a.instance_id:tr.read(a)[0] for a in plan}
ev=runner.old.RetrievalMetrics(source,tr.clean,gallery,idx[row["paired_image"]],tau=.02)
seed_all(42);torch.use_deterministic_algorithms(True)
r={"diagnostic_only":True,"CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"caption_id":"cuhk:35:1","max_steps":1}
try:
 p.execute(source,image,plan,p.cached_round_guidance("SEQUENTIAL_NEW",plan,vec),ev,steps=1,transforms=6,scales=tuple(protocol["scales"]),tape=CategoricalTape())
 r["one_step_completed"]=True
except RuntimeError as e:
 import traceback
 r["one_step_completed"]=False;r["error"]=str(e);r["traceback"]=traceback.format_exc()
runner.old.verify(hashes);runner.save(ROOT/"outputs/repeatability_35/deterministic_configured_probe.json",r)
print(json.dumps(r),flush=True)
