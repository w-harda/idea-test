"""最小G训练：每batch一次G forward、一次frozen IRRA image forward，只有G optimizer更新。"""
import argparse,time,json,os
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
import common as c
from generator_model import ConditionalGenerator,compose,training_loss
import checkpoint_io as ck
class ConditionDataset(Dataset):
 def __init__(self,entries):
  self.entries=entries;self.transform=transforms.Compose([transforms.Resize((384,128)),transforms.ToTensor()])
  self.conditions=[c.read_condition(e) for e in entries]
  for e in entries:
   path=(c.IMAGE_ROOT/e["image"]).resolve()
   if not path.is_relative_to(c.IMAGE_ROOT.resolve()) or c.sha(path)!=e["image_sha256"]:raise ValueError("image source changed")
 def __len__(self):return len(self.entries)
 def __getitem__(self,i):
  e=self.entries[i]
  with Image.open(c.IMAGE_ROOT/e["image"]) as im:image=self.transform(im.convert("RGB"))
  return {"image":image,**self.conditions[i],"row_id":e["row_id"]}
def batch_to(batch,device):
 return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
def weights(config):return {k:config["training"][k] for k in ("global_weight","local_weight","perturbation_weight","tv_weight")}
def step(generator,victim,batch,optimizer,config,*,diagnose=False):
 optimizer.zero_grad(set_to_none=True)
 delta=generator(batch["image"],batch["z_t"],batch["c_attr"]);adv=compose(batch["image"],delta)
 feature=victim.encode_images(adv) # 不用no_grad：梯度经frozen IRRA回到G。
 total,terms=training_loss(feature,batch["z_t"],batch["v_local"],batch["has_local"],adv-batch["image"],**weights(config))
 diagnostic={}
 if diagnose:
  for key in ("global","local"):
   grad,=torch.autograd.grad(terms[key],adv,retain_graph=True)
   diagnostic[key+"_input_gradient_norm"]=float(grad.norm())
 total.backward()
 norm=sum(float(p.grad.detach().square().sum()) for p in generator.parameters() if p.grad is not None)**.5
 if not norm>0 or not all(p.grad is None or torch.isfinite(p.grad).all() for p in generator.parameters()):raise ValueError("G gradient zero/nonfinite")
 optimizer.step()
 linf=float((adv-batch["image"]).abs().max())
 if linf>8/255+1e-6 or any(p.requires_grad or p.grad is not None for p in victim.parameters()):raise ValueError("budget/frozen victim")
 return {"total":float(total.detach()),**{k:float(v.detach()) for k,v in terms.items()},
 "G_gradient_norm":norm,"linf":linf,**diagnostic}
@torch.no_grad()
def validate(generator,victim,loader,config):
 generator.eval();totals={};n=0;max_linf=0.
 for batch in loader:
  b=batch_to(batch,"cuda");adv=compose(b["image"],generator(b["image"],b["z_t"],b["c_attr"]))
  clean=victim.encode_images(b["image"]);feature=victim.encode_images(adv)
  loss,parts=training_loss(feature,b["z_t"],b["v_local"],b["has_local"],adv-b["image"],**weights(config))
  vals={"total":float(loss),**{k:float(v) for k,v in parts.items()},
   "clean_caption_cosine":float((clean*b["z_t"]).sum(1).mean()),"adv_caption_cosine":float((feature*b["z_t"]).sum(1).mean()),
   "caption_cosine_change":float(((feature-clean)*b["z_t"]).sum(1).mean()),
   "local_dot_change":float((((feature-clean)*b["v_local"]).sum(1)*b["has_local"]).mean())}
  for k,v in vals.items():totals[k]=totals.get(k,0)+len(b["image"])*v
  n+=len(b["image"]);max_linf=max(max_linf,float((adv-b["image"]).abs().max()))
 if max_linf>8/255+1e-6:raise ValueError("validation budget")
 return {"count":n,**{k:v/n for k,v in totals.items()},"max_linf":max_linf,"not_formal_retrieval_evaluation":True}
def checkpoint(path,generator,optimizer,epoch,config,cache_sha,scope):
 payload={"schema":"conditional-G-image-1","architecture":generator.architecture,
 "model":{k:v.detach().cpu() for k,v in generator.state_dict().items()},
 "optimizer":optimizer.state_dict(),"epoch":epoch,"config":config,"cache_manifest_sha256":cache_sha,"scope":scope,
 "inference_source_sha256":{str(c.ROOT/name):c.sha(c.ROOT/name) for name in ("generator_model.py","common.py","generate.py","checkpoint_io.py")}}
 ck.save_checkpoint(path,payload)
def load_checkpoint(path,device):
 return ck.restore_checkpoint(path,device)
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--cache",type=Path,default=c.ROOT/"outputs/conditions")
 p.add_argument("--output",type=Path,default=c.ROOT/"outputs/train")
 p.add_argument("--allow-training",action="store_true");p.add_argument("--resume",action="store_true")
 return p
def main(argv=None):
 args=parser().parse_args(argv)
 if not args.allow_training:raise SystemExit("manual training requires --allow-training")
 import integration as it
 config=c.load(c.ROOT/"config.json");manifest=c.checked_cache(args.cache)
 if manifest["scope"]!="prepared":raise ValueError("full training cannot silently use smoke conditions")
 if manifest["config_sha256"]!=c.sha(c.ROOT/"config.json"):raise ValueError("config changed after preparation")
 c.verify(manifest["frozen_sha256"]);it.frozen.frozen.assert_formal_clean()
 smoke=c.load(c.ROOT/"outputs/smoke/completed.json")
 ck.verify_training_smoke(smoke)
 out=c.safe(args.output)
 with ck.training_lock(out):
  protocol={"config":config,"cache_manifest_sha256":c.sha(args.cache/"manifest.json"),"frozen_sha256":manifest["frozen_sha256"],"checkpoint_selection":"fixed final epoch"}
  if (out/"protocol.json").exists() and c.load(out/"protocol.json")!=protocol:raise ValueError("training provenance changed")
  c.save(out/"protocol.json",protocol)
  torch.manual_seed(config["seed"]);victim=it.frozen.setup();versions={n:p._version for n,p in victim.named_parameters()}
  generator=ConditionalGenerator(**config["architecture"]).cuda();opt=torch.optim.Adam(generator.parameters(),lr=config["training"]["lr"],betas=tuple(config["training"]["betas"]))
  start=1
  if args.resume:
   cp=load_checkpoint(out/"last.pt","cuda")
   if cp["cache_manifest_sha256"]!=protocol["cache_manifest_sha256"] or cp["config"]!=config or cp["scope"]!="training":raise ValueError("resume metadata")
   generator.load_state_dict(cp["model"]);opt.load_state_dict(cp["optimizer"]);start=cp["epoch"]+1
  elif ck.checkpoint_exists(out/"last.pt"):raise FileExistsError("use --resume or new output")
  trainset=ConditionDataset(manifest["train"]);valset=ConditionDataset(manifest["validation"])
  val=DataLoader(valset,batch_size=config["training"]["batch_size"],shuffle=False,num_workers=0)
  for epoch in range(start,config["training"]["epochs"]+1):
   generator.train();shuffle=torch.Generator().manual_seed(config["seed"]+epoch)
   loader=DataLoader(trainset,batch_size=config["training"]["batch_size"],shuffle=True,generator=shuffle,num_workers=0)
   t=time.time();values=[]
   for i,batch in enumerate(loader):
    stats=step(generator,victim,batch_to(batch,"cuda"),opt,config);values.append((len(batch["image"]),stats))
    if i%50==0:print("TRAIN_STEP",epoch,i,json.dumps(stats),flush=True)
   report=validate(generator,victim,val,config)
   it.frozen.assert_frozen(victim,versions);c.verify(manifest["frozen_sha256"])
   denom=sum(n for n,x in values);train_mean={k:sum(n*x[k] for n,x in values)/denom for k in values[0][1]}
   event={"epoch":epoch,"train":train_mean,"validation":report,"seconds":time.time()-t}
   c.save(out/f"epoch_{epoch:03d}.json",event);checkpoint(out/"last.pt",generator,opt,epoch,config,protocol["cache_manifest_sha256"],"training")
   print("EPOCH_COMPLETE",json.dumps(event),flush=True)
  it.frozen.frozen.assert_formal_clean()
  c.save(out/"completed.json",{"method_audit_passed":True,"epochs":config["training"]["epochs"],"checkpoint_sha256":c.sha(out/"last.pt"),
  "cache_manifest_sha256":protocol["cache_manifest_sha256"],"IRRA_frozen":True,"only_G_trained":True,"test_used":False})
  print("TRAINING_COMPLETE",flush=True)
if __name__=="__main__":main()
