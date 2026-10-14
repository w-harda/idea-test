"""仅2 train+2 validation条件，恰好2次G optimizer step；不启动正式训练。"""
import json,time,sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
import common as c
from prepare import prepare_conditions
from generator_model import ConditionalGenerator,generate_once
from train import ConditionDataset,batch_to,step,validate,checkpoint
def main():
 out=c.ROOT/"outputs/smoke";cache=c.ROOT/"outputs/smoke_conditions"
 if out.exists():raise FileExistsError("refuse overwrite smoke")
 out.mkdir(parents=True)
 prepare_conditions(cache,smoke=True)
 import integration as it
 config=c.load(c.ROOT/"config.json");manifest=c.checked_cache(cache)
 c.verify(manifest["frozen_sha256"])
 torch.manual_seed(config["seed"]);victim=it.frozen.setup();versions={n:p._version for n,p in victim.named_parameters()}
 model=ConditionalGenerator(**config["architecture"]).cuda();opt=torch.optim.Adam(model.parameters(),lr=config["training"]["lr"],betas=tuple(config["training"]["betas"]))
 dataset=ConditionDataset(manifest["train"]);batch=batch_to(next(iter(DataLoader(dataset,batch_size=2))),"cuda")
 before={k:v.detach().clone() for k,v in model.named_parameters()}
 image_calls=[];g_calls=[];handle=model.register_forward_hook(lambda *a:g_calls.append(1))
 events=[]
 try:
  for index in range(2):
   torch.cuda.synchronize();t=time.perf_counter()
   with patch.object(victim,"encode_images",wraps=victim.encode_images) as calls:
    event=step(model,victim,batch,opt,config,diagnose=True)
    if calls.call_count!=1:raise ValueError("more than one surrogate image forward per training step")
   torch.cuda.synchronize();event["seconds"]=time.perf_counter()-t;events.append(event)
 finally:handle.remove()
 if len(g_calls)!=2 or not any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters()):raise ValueError("G did not update once/step")
 if not all(e["global_input_gradient_norm"]>0 and e["local_input_gradient_norm"]>0 for e in events):raise ValueError("global/local gradient disconnected")
 cond_grad=sum(float(p.grad.square().sum()) for p in model.condition.parameters() if p.grad is not None)**.5
 if not cond_grad>0:raise ValueError("conditioning MLP not trainable")
 it.frozen.assert_frozen(victim,versions)
 val=validate(model,victim,DataLoader(ConditionDataset(manifest["validation"]),batch_size=2),config)
 # 测试生成阶段明确禁止autograd.grad/反向；不使用victim。
 inference_calls=[];h=model.register_forward_hook(lambda *a:inference_calls.append(1))
 try:
  with patch("torch.autograd.grad",side_effect=AssertionError("inference backward forbidden")):
   result=generate_once(model,batch["image"],batch["z_t"],batch["c_attr"])
 finally:h.remove()
 if len(inference_calls)!=1 or result["adversarial"].requires_grad:raise ValueError("online contract")
 repeat=generate_once(model,batch["image"],batch["z_t"],batch["c_attr"])
 if not torch.equal(result["adversarial"],repeat["adversarial"]):raise ValueError("inference nondeterministic")
 different_z=generate_once(model,batch["image"],batch["z_t"].roll(1,0),batch["c_attr"])
 different_a=generate_once(model,batch["image"],batch["z_t"],batch["c_attr"].roll(1,0))
 zdiff=float((different_z["delta"]-result["delta"]).abs().max());adiff=float((different_a["delta"]-result["delta"]).abs().max())
 if zdiff==0 or adiff==0:raise ValueError("generator ignores one condition branch")
 torch.cuda.synchronize();t=time.perf_counter()
 for _ in range(10):generate_once(model,batch["image"],batch["z_t"],batch["c_attr"])
 torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000/10
 checkpoint(out/"checkpoint.pt",model,opt,0,config,c.sha(cache/"manifest.json"),"smoke")
 it.frozen.assert_frozen(victim,versions);c.verify(manifest["frozen_sha256"]);it.frozen.frozen.assert_formal_clean()
 report={"method_audit_passed":True,"training_optimizer_steps":2,"G_forward_count_training":2,"train_condition_count":2,"validation_condition_count":2,
 "train_validation_PID_disjoint":True,"test_used":False,"IRRA_reader_frozen":True,"only_G_updated":True,"condition_MLP_gradient_norm":cond_grad,
 "training_steps":events,"validation_diagnostic":val,"online_G_forward_calls":1,"online_victim_forward_backward":0,
 "online_scoring_JVP_TTA_calls":0,"online_requires_grad":False,"deterministic_repeat_exact":True,
 "caption_condition_delta_max_difference":zdiff,"attribute_condition_delta_max_difference":adiff,
 "inference_batch2_mean_milliseconds":ms,"parameters":sum(p.numel() for p in model.parameters()),
 "maximum_linf":float(result["effective_delta"].abs().max()),"peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
 "source_sha256":manifest["frozen_sha256"],"manual_long_training_started":False}
 c.save(out/"completed.json",report)
 print("SMOKE_COMPLETE",json.dumps(report),flush=True)
if __name__=="__main__":main()
