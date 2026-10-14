"""同一失败case热上下文单case回归；不运行其它caption或PREFIX攻击。"""
import sys,os
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import run as current
from prefix_tta import *
from attribute_tta import prepared_plan,TextTransport,seed_all
from attributes.irra_surrogate import IRRASurrogate
from PIL import Image
from debug_regression import comparison
import json
def main():
 out=ROOT/"outputs/regression_warm_21"
 if out.exists():raise ValueError("no overwrite")
 out.mkdir()
 hashes,rows,payloads,bases,protocol,hist=current.inputs()
 cid="cuhk:21:0"
 sys.path.insert(0,str(REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
 import learned_reader as v4
 v4.setup()
 torch.backends.mha.set_fastpath_enabled(False);torch.backends.cuda.enable_flash_sdp(False)
 torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 victim=v4.load_victim();source=IRRASurrogate(victim)
 gallery,idx,_=current.old.load_gallery(hashes)
 record=next(x for x in rows if x["row_id"]==cid)
 plan=prepared_plan(next(x for x in payloads if x["caption_id"]==cid),record,bases,"cuda")
 with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/record["paired_image"]) as im:image=victim.pixels(im).unsqueeze(0).cuda()
 transport=TextTransport(victim,record["caption"]);vec={a.instance_id:transport.read(a)[0] for a in plan}
 saved=torch.load(ROOT/"outputs/regression_debug_21/debug_tensors.pt",weights_only=False,map_location="cpu")
 # 只对同一caption既有round输入补算prefix梯度诊断，复现上一case结束后的工作区状态。
 for j,x in enumerate(saved["round_inputs"],1):gradient_decomposition(source,x.cuda(),plan[:j],vec)
 ev=current.old.RetrievalMetrics(source,transport.clean,gallery,idx[record["paired_image"]],tau=.02)
 seed_all(42);tape=CategoricalTape()
 now,inputs,momentum=execute(source,image,plan,cached_round_guidance("SEQUENTIAL_NEW",plan,vec),ev,
                            steps=10,transforms=6,scales=tuple(protocol["scales"]),tape=tape)
 report=comparison(now,hist[cid]["modes"]["NEW"])
 current.save(out/"current_sequential.json",now);current.save(out/"report.json",report)
 print(json.dumps({"state_differences":report["state_differences"],"sequence_exact":report["sequence_exact"],
     "noise_exact":report["noise_exact"],"trace_exact":report["trace_exact"],"rng_end_exact":report["rng_end_exact"],
     "round_output_sha_exact":[x["output_image_sha_exact"] for x in report["rounds"]]},indent=2))
if __name__=="__main__":main()
