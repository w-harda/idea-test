"""在线只读取准备好的text condition和G；不导入IRRA、scorer、JVP或TTA。"""
import argparse,json,sys
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import common as c
from generator_model import ConditionalGenerator,generate_once
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--cache",type=Path,required=True)
 p.add_argument("--row-id",required=True);p.add_argument("--caption")
 p.add_argument("--image",type=Path);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
 return p
def main(argv=None):
 args=parser().parse_args(argv);out=c.safe(args.output)
 if out.exists():raise FileExistsError("refuse overwrite")
 manifest=c.checked_cache(args.cache)
 entry=next((e for role in ("train","validation") for e in manifest[role] if e["row_id"]==args.row_id),None)
 if entry is None:raise ValueError("caption condition is uncached; cannot silently run S_raw backward online")
 if args.caption is not None and args.caption!=entry["caption"]:raise ValueError("raw caption differs from frozen condition")
 source=args.image or c.IMAGE_ROOT/entry["image"]
 if c.sha(source)!=entry["image_sha256"]:raise ValueError("cached condition belongs to a different source image")
 condition=c.read_condition(entry)
 side=c.load(str(args.checkpoint)+".json");c.verify({str(args.checkpoint):side["sha256"]})
 cp=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
 if cp["schema"]!="conditional-G-image-1" or cp["cache_manifest_sha256"]!=c.sha(args.cache/"manifest.json"):raise ValueError("checkpoint/cache provenance")
 c.verify(cp["inference_source_sha256"])
 model=ConditionalGenerator(**cp["architecture"]).to(args.device);model.load_state_dict(cp["model"]);model.eval()
 with Image.open(source) as im:image=transforms.ToTensor()(transforms.Resize((384,128))(im.convert("RGB"))).unsqueeze(0).to(args.device)
 calls=[];handle=model.register_forward_hook(lambda *unused:calls.append(1))
 try:result=generate_once(model,image,condition["z_t"][None].to(args.device),condition["c_attr"][None].to(args.device))
 finally:handle.remove()
 if len(calls)!=1 or any(v.requires_grad for v in result.values()):raise ValueError("not one inference-only G forward")
 linf=float(result["effective_delta"].abs().max())
 if linf>8/255+1e-6 or any(name in sys.modules for name in ("integration","runtime","attribute_tta")):raise ValueError("budget or victim frontend initialized")
 c.tensor_save(out,{k:v.cpu() for k,v in result.items()})
 c.save(str(out)+".json",{"row_id":entry["row_id"],"caption":entry["caption"],"generator_forward_calls":1,
 "victim_initialized":False,"score_backward":0,"JVP_computations":0,"test_time_optimization_steps":0,
 "condition_precomputed":True,"linf":linf,"checkpoint_sha256":side["sha256"],"output_sha256":c.sha(out)})
 print("GENERATION_COMPLETE",json.dumps(c.load(str(out)+".json"),ensure_ascii=False),flush=True)
if __name__=="__main__":main()
