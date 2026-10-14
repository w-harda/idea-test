"""只复现首次失败的单case SEQUENTIAL，不运行PREFIX或固定25句。"""
import sys,os
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import run as current
from prefix_tta import *
from attribute_tta import prepared_plan,TextTransport,seed_all,IMAGE_EPSILON
from controlled_tta import tensor_hash
from attributes.irra_surrogate import IRRASurrogate
from PIL import Image
import json,math, argparse
def error(a,b):
    ab=abs(a-b);return {"current":a,"saved":b,"absolute_error":ab,"relative_error":ab/abs(b) if b else None}
def comparison(now,previous):
    report={"sequence_exact":now["fixed_sequence"]==previous["fixed_sequence"],
            "noise_exact":now["noise_sha256"]==previous["noise_sha256"],
            "trace_exact":now["random_trace"]==previous["random_trace"],
            "rng_end_exact":now["rng_end_sha256"]==previous["rng_end_sha256"],
            "state_differences":[],"rounds":[]}
    for i,(a,b) in enumerate(zip(now["states"],previous["states"])):
        for key in a:
            if a[key]!=b[key]:report["state_differences"].append({"state":i,"field":key,**error(a[key],b[key])})
    for i,(a,b) in enumerate(zip(now["rounds"],previous["rounds"]),1):
        stepdiff=[]
        for j,(sa,sb) in enumerate(zip(a["steps"],b["steps"]),1):
            for key in ["ig_gradient_l2","momentum_l2","cumulative_perturbation_linf"]:
                if sa[key]!=sb[key]:stepdiff.append({"step":j,"field":key,**error(sa[key],sb[key])})
        report["rounds"].append({"round":i,"output_image_sha_exact":a["post_projection_image_sha256"]==b["post_projection_image_sha256"],
              "input_image_sha_exact":True if i==1 else now["rounds"][i-2]["post_projection_image_sha256"]==previous["rounds"][i-2]["post_projection_image_sha256"],
              "initialization":a["initialization"],"saved_initialization":b["initialization"],
              "momentum_gradient_norm_differences":stepdiff})
    return report
def main():
    args=argparse.ArgumentParser()
    args.add_argument("--output",type=Path,required=True)
    opt=args.parse_args()
    out=opt.output.resolve()
    if not out.is_relative_to(Path("/home/lzf/ldx")) or out.exists():raise ValueError("new safe output only")
    hashes,rows,payloads,bases,protocol,history=current.inputs()
    done=current.load(ROOT/"outputs/diagnostic/results.json")
    cid=protocol["row_ids"][len(done)]
    if cid!="cuhk:21:0":raise ValueError("only first failed case authorized")
    out.mkdir()
    sys.path.insert(0,str(REPO/"prototypes/attribute_unit_v4_learned_reader_20261003"))
    import learned_reader as v4
    v4.setup()
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    victim=v4.load_victim();source=IRRASurrogate(victim)
    gallery,idx,_=current.old.load_gallery(hashes)
    record=next(r for r in rows if r["row_id"]==cid)
    payload=next(p for p in payloads if p["caption_id"]==cid)
    plan=prepared_plan(payload,record,bases,"cuda")
    with Image.open(Path("/home/lzf/TBPS/Datasets/CUHK-PEDES/imgs")/record["paired_image"]) as im:
        image=victim.pixels(im).unsqueeze(0).cuda()
    trans=TextTransport(victim,record["caption"]);vec={};mapping=[]
    for a in plan:
        v,r=trans.read(a);vec[a.instance_id]=v;mapping.append(r)
    vector_hashes={a.instance_id:tensor_hash(vec[a.instance_id]) for a in plan}
    both={mode:cached_round_guidance(mode,plan,vec) for mode in MODES}
    evaluator=current.old.RetrievalMetrics(source,trans.clean,gallery,idx[record["paired_image"]],tau=.02)
    cached=both["SEQUENTIAL_NEW"]
    seed_all(42);tape=CategoricalTape()
    now,images,momentum=execute(source,image,plan,cached,evaluator,steps=10,transforms=6,scales=tuple(protocol["scales"]),tape=tape)
    previous=history[cid]["modes"]["NEW"]
    report={"caption_id":cid,"case_count":1,"round_count":len(plan),
            "initial_image_sha256":tensor_hash(image),"per_round_input_sha256":[tensor_hash(x) for x in images],
            "comparison":comparison(now,previous),"JVP_mapping":mapping,
            "historical_JVP_mapping":history[cid]["JVP_mapping"],
            "historical_tensor_limitation":"Step73 stores round image hashes and step momentum/gradient norms, no image/momentum tensors. SHA mismatch cannot establish elementwise tensor error.",
            "saved_regression_pass":None}
    try:report["saved_regression_pass"]=saved_regression(now,previous,current.load(ROOT/"regression_reference_variants.json")["variants"].get(cid,()))
    except ValueError as e:report["saved_regression_error"]=str(e)
    current.save(out/"current_sequential.json",now)
    current.save(out/"report.json",report)
    torch.save({"initial_image":image.cpu(),"round_inputs":[x.cpu() for x in images],"final_momentum":momentum.cpu(),
                "JVP_vectors":{k:v.cpu() for k,v in vec.items()},"categorical_tape":tape.records},out/"debug_tensors.pt")
    current.old.verify(hashes)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
