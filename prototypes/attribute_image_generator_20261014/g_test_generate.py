"""独立Test条件推理：只加载G和已准备的Test条件；不初始化victim。"""
import argparse,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from torchvision import transforms
import common as c
import g_test_core as g
from generator_model import generate_once

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation",type=Path,required=True);p.add_argument("--row-id",required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args(argv);out=c.safe(args.output)
    if out.exists():raise FileExistsError("refuse overwrite")
    protocol=c.load(args.evaluation/"protocol.json")
    if protocol["schema"]!="G-official-test-1":raise ValueError("not official test conditions")
    c.verify(protocol["source_sha256"])
    manifest=g.checked_conditions(args.evaluation,protocol)
    entry=g.condition_entry(manifest,args.row_id)
    if c.sha(c.IMAGE_ROOT/entry["image"])!=entry["image_sha256"]:raise ValueError("source image SHA")
    condition=c.read_condition(entry);model=g.load_generator(protocol,args.device)
    with Image.open(c.IMAGE_ROOT/entry["image"]) as im:
        image=transforms.ToTensor()(transforms.Resize((384,128))(im.convert("RGB"))).unsqueeze(0).to(args.device)
    calls=[];handle=model.register_forward_hook(lambda *args:calls.append(1))
    try:
        with patch("torch.autograd.grad",side_effect=AssertionError("no backward online")):
            result=generate_once(model,image,condition["z_t"][None].to(args.device),condition["c_attr"][None].to(args.device))
    finally:handle.remove()
    if len(calls)!=1 or any(name in sys.modules for name in ("integration","runtime","attribute_tta")):
        raise ValueError("not independent one-forward generation")
    budget=float(result["effective_delta"].abs().max())
    if budget>8/255+1e-6:raise ValueError("budget")
    c.tensor_save(out,{k:v.cpu() for k,v in result.items()})
    c.save(str(out)+".json",{"row_id":entry["row_id"],"scope":protocol["scope"],"G_forward_calls":1,
       "victim_initialized":False,"score_backward":0,"JVP_computations":0,"optimization_steps":0,
       "linf":budget,"checkpoint_sha256":protocol["checkpoint_sha256"],"output_sha256":c.sha(out)})
    print("TEST_GENERATION_COMPLETE",json.dumps(c.load(str(out)+".json")),flush=True)

if __name__=="__main__":main()
