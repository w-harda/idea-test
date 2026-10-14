"""单失败case、相同tensor字节的CUDA memory-alignment数值复核；不运行攻击。"""
import os,sys,json
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run as current
from prefix_tta import *
from controlled_tta import tensor_hash
from attributes.irra_surrogate import IRRASurrogate
from attribute_tta import TextTransport
def main():
    out=ROOT/"outputs/regression_metric_rechecks_21"
    if out.exists():raise ValueError("no overwrite")
    out.mkdir()
    hashes,rows,_,_,_,hist=current.inputs()
    record=next(r for r in rows if r["row_id"]=="cuhk:21:0")
    sys.path.insert(0,str(REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
    import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim=v4.load_victim();source=IRRASurrogate(victim)
    gallery,idx,_=current.old.load_gallery(hashes)
    saved=torch.load(ROOT/"outputs/regression_debug_21/debug_tensors.pt",weights_only=False,map_location="cpu")
    images=[x.cuda() for x in saved["round_inputs"]]
    trans=TextTransport(victim,record["caption"])
    reports=[]
    for offset in [0,1,2,4,8,16]:
        # 完全相同fp32值，仅改变contiguous tensor起始地址；不改任何模型/数据数值或超参。
        gbuffer=torch.empty(gallery.numel()+offset,device="cuda",dtype=gallery.dtype)
        g=gbuffer[offset:].view_as(gallery);g.copy_(gallery)
        assert torch.equal(g,gallery)
        evaluator=current.old.RetrievalMetrics(source,trans.clean,g,idx[record["paired_image"]],tau=.02)
        for i,image in enumerate(images):
            now=evaluator(image);old=hist[record["row_id"]]["modes"]["NEW"]["states"][i]
            differences={key:{"current":now[key],"saved":old[key],"absolute_error":abs(now[key]-old[key]),
                             "relative_error":abs(now[key]-old[key])/abs(old[key]) if old[key] else None}
                         for key in now if now[key]!=old[key]}
            if differences:reports.append({"offset_floats":offset,"state":i,"image_sha256":tensor_hash(image),"differences":differences})
    current.save(out/"report.json",reports)
    print(json.dumps(reports,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
