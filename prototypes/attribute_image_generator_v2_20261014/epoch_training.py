"""G-v2单样本Epoch调度；默认preflight，正式执行必须显式allow-training。"""
import argparse,json,os,random,sys,time,uuid,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import cache_io as io
from training_data import checked_cache,file_contract,make_loader
from image_only_generator import ImageOnlyGenerator
from serial_training import train_single_sample,LossWeights
from losses import joint_losses
from condition_schema import validate
from epoch_checkpoint import save_epoch,load_latest
from inference_once import generate_once

TRAIN_FILES=("epoch_training.py","epoch_checkpoint.py","training_data.py","serial_training.py",
             "image_only_generator.py","budget_projection.py","losses.py","condition_schema.py",
             "cache_io.py","irra_image_adapter.py","inference_once.py")

def epoch_order(entries,seed,epoch):
    ordered=sorted(entries,key=lambda e:e["row_id"])
    random.Random(f"gv2|{seed}|{epoch}").shuffle(ordered)
    return ordered

def configuration(scope="formal",epochs=10,seed=42,device="cuda"):
    return {"schema":"gv2-epoch-config-1","scope":scope,"epochs":epochs,"seed":seed,"device":device,
        "architecture":{"width":32,"blocks":4},"optimizer":{"lr":2e-4,"betas":[.5,.999]},
        "weights":{"lambda_global":1.,"lambda_local":1.,"lambda_retain":1.}}

def contract_for(manifest,cache,config):
    return {"config":config,"cache_manifest_sha256":io.sha(Path(cache)/"manifest.json"),
        "cache_protocol_sha256":manifest["protocol_sha256"],"training_source_sha256":file_contract(TRAIN_FILES),
        "runtime":{"python":sys.version,"torch":str(torch.__version__),"numpy":np.__version__}}

def train_epoch(g,encoder,opt,entries,loader,seed,epoch,weights,log_path=None):
    if any(e["role"]!="train" for e in entries): raise ValueError("validation cannot enter optimizer scheduling")
    ordered=epoch_order(entries,seed,epoch);logs=[];by_k=Counter();updates=0
    handle=None
    if log_path is not None:
        path=io.safe(log_path);path.parent.mkdir(parents=True,exist_ok=True);handle=path.open("x",encoding="utf-8")
    try:
        for position,entry in enumerate(ordered,1):
            image,condition=loader(entry)
            validate(condition,entry)
            if condition["row_id"]!=entry["row_id"] or condition["role"]!="train":
                raise ValueError("loaded sample identity/role mismatch")
            result=train_single_sample(g,encoder,opt,image,condition,weights=weights)
            if result["optimizer_steps"]!=entry["K"]+1: raise ValueError("K+1 update count")
            record={"epoch":epoch,"sample_position":position,"row_id":entry["row_id"],"role":"train","K":entry["K"],
                    "optimizer_steps":result["optimizer_steps"],"rounds":result["records"]}
            logs.append(record);by_k[str(entry["K"])]+=result["optimizer_steps"];updates+=result["optimizer_steps"]
            if handle:
                handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n");handle.flush();os.fsync(handle.fileno())
    finally:
        if handle:handle.close()
    return {"epoch":epoch,"sample_order":[e["row_id"] for e in ordered],"samples":len(ordered),
            "optimizer_steps":updates,"updates_by_K":dict(by_k),"sample_logs":logs}

@torch.no_grad()
def validate_epoch(g,encoder,entries,loader):
    if any(e["role"]!="validation" for e in entries): raise ValueError("validation role mismatch")
    g.eval();encoder.eval();records=[]
    for e in entries:
        image,c=loader(e)
        validate(c,e)
        if c["row_id"]!=e["row_id"] or c["role"]!="validation": raise ValueError("validation loaded identity mismatch")
        adv=generate_once(g,image)["adversarial"]
        z=c["z_t"][None].to(image.device);v=c["v_local"][None].to(image.device)
        terms=joint_losses(encoder(adv),z,v,torch.tensor([c["has_local"]],device=image.device))
        records.append({"row_id":e["row_id"],"global":float(terms["global"]),"local":float(terms["local"]),
                        "optimizer_steps":0,"cumulative_linf":float((adv.double()-image.double()).abs().max())})
    return records

def run_epochs(cache,output,config,*,resume=False,stop_after_epochs=None,generator=None,encoder=None,loader=None):
    if config["scope"] not in ("formal","smoke","synthetic") or type(config["epochs"]) is not int or config["epochs"]<1:
        raise ValueError("invalid training config")
    m=checked_cache(cache,require_full=config["scope"]=="formal")
    contract=contract_for(m,cache,config)
    with io.output_lock(output) as out:
        if (out/"training_protocol.json").exists() and io.digest(io.load(out/"training_protocol.json"))!=io.digest(contract):
            raise ValueError("training protocol changed")
        existing=list((out/"checkpoints").glob("epoch_*"))
        if existing and not resume: raise ValueError("existing checkpoints require explicit resume")
        # 写入产物前已持有锁；校验后才加载GPU。
        io.save_json(out/"training_protocol.json",contract)
        random.seed(config["seed"]);np.random.seed(config["seed"]);torch.manual_seed(config["seed"])
        if generator is None: generator=ImageOnlyGenerator(**config["architecture"]).to(config["device"])
        if encoder is None:
            from irra_image_adapter import load_official
            encoder=load_official(config["device"])
        opt=torch.optim.Adam(generator.parameters(),**config["optimizer"])
        start=1
        if resume:
            restored=load_latest(out,contract,generator,opt);start=restored["next_epoch"]
        if loader is None:loader=make_loader(m,config["device"])
        weights=LossWeights(**config["weights"])
        train=[e for e in m["entries"] if e["role"]=="train"]
        validation=[e for e in m["entries"] if e["role"]=="validation"]
        summaries=[]
        for epoch in range(start,config["epochs"]+1):
            if stop_after_epochs is not None and len(summaries)>=stop_after_epochs:break
            attempt=io.safe(out/"epoch_attempts"/f"epoch_{epoch:04d}_{uuid.uuid4().hex}");attempt.mkdir(parents=True)
            progress=train_epoch(generator,encoder,opt,train,loader,config["seed"],epoch,weights,attempt/"samples.jsonl")
            progress["validation"]=validate_epoch(generator,encoder,validation,loader)
            progress["sample_log_sha256"]=io.sha(attempt/"samples.jsonl")
            progress["sample_log_path"]=str(attempt/"samples.jsonl")
            io.verify(contract["training_source_sha256"])
            # 每epoch末再次核验缓存；数据变化拒绝发布checkpoint。
            checked=checked_cache(cache,require_full=config["scope"]=="formal")
            if io.sha(Path(cache)/"manifest.json")!=contract["cache_manifest_sha256"]:raise ValueError("cache changed during epoch")
            save_epoch(out,generator,opt,epoch,contract,progress)
            io.save_json(attempt/"completed.json",{"checkpoint_published":True,"epoch":epoch})
            summaries.append(progress)
            print("EPOCH_COMPLETE",epoch,"samples",progress["samples"],"steps",progress["optimizer_steps"],flush=True)
        return generator,opt,{"completed_epoch":start-1+len(summaries),"epochs_this_run":summaries,"contract":contract}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",type=Path,default=io.OUTPUTS/"conditions_full")
    p.add_argument("--output",type=Path,default=io.OUTPUTS/"train_v2")
    p.add_argument("--epochs",type=int,default=10);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--resume",action="store_true");p.add_argument("--allow-training",action="store_true")
    p.add_argument("--config",type=Path,help="完整训练配置JSON；resume要求配置严格不变")
    a=p.parse_args(argv)
    if not a.allow_training:
        if (a.cache/"completed.json").exists():checked_cache(a.cache,require_full=True)
        print("PREFLIGHT_ONLY; no model/GPU/optimizer started");return
    config=io.load(a.config) if a.config else configuration(epochs=a.epochs,seed=a.seed)
    if config["scope"]!="formal":raise ValueError("CLI training requires formal scope/full cache")
    run_epochs(a.cache,a.output,config,resume=a.resume)
if __name__=="__main__":main()
