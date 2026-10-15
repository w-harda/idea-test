"""完整训练只读审计：逐Epoch加载后释放权重，不持有历史Tensor。"""
import argparse,json,math,subprocess,sys
from pathlib import Path
import torch
import cache_io as io
import epoch_training as et
import epoch_checkpoint as ck
from phase27_analysis import audit_epoch,read_rows,validation_comparison
from phase26_support import state_hash
from budget_projection import DEFAULT_EPSILON
from training_data import checked_cache

def scalar_only(value):
    if value is None or type(value) in (str,int,bool):return
    if type(value) is float:
        if not math.isfinite(value):raise ValueError("nonfinite monitoring scalar")
        return
    if isinstance(value,dict):
        for k,v in value.items():
            if type(k) is not str:raise ValueError("monitor keys must be strings")
            scalar_only(v)
        return
    if isinstance(value,(tuple,list)):
        for v in value:scalar_only(v)
        return
    raise ValueError("monitoring must not retain Tensor/array/graph")

def published_epochs(output):
    names=sorted(p.name for p in (Path(output)/"checkpoints").glob("epoch_*"))
    if not names:raise ValueError("existing formal checkpoints required")
    try:epochs=[int(n.split("_")[1]) for n in names]
    except ValueError as exc:raise ValueError("unexpected checkpoint directory") from exc
    if epochs!=list(range(1,max(epochs)+1)) or names!=[f"epoch_{n:04d}" for n in epochs]:
        raise ValueError("checkpoint sequence must be contiguous")
    if not 3<=epochs[-1]<=10:raise ValueError("authorized boundary is Epoch3–10")
    return epochs

def round_path(value,output):
    epoch=value["completed_epoch"]
    if epoch==1:return Path(output)/"round_audit.jsonl"
    paths=[Path(p) for p in value["progress"]["supplementary_log_sha256"] if p.endswith("_round_audit.jsonl")]
    if len(paths)!=1:raise ValueError("exactly one sealed per-Epoch round log required")
    return paths[0]

def analyze(cache,output,stage,*,last_epoch=10):
    m=checked_cache(cache,require_full=True);output=Path(output);stage=Path(stage)
    contract=io.load(output/"training_protocol.json")
    if io.digest(contract)!=io.digest(et.contract_for(m,cache,io.load(io.ROOT/"config_phase26.json"))):
        raise ValueError("live formal contract mismatch")
    io.verify(contract["training_source_sha256"]);io.verify(contract["config"]["entry_source_sha256"])
    epochs=published_epochs(output)
    if epochs!=list(range(1,last_epoch+1)):raise ValueError("expected completed Epoch range mismatch")
    summaries=[];validations=[];final_hash=None
    for epoch in epochs:
        version=output/"checkpoints"/f"epoch_{epoch:04d}"
        value=ck.verify_version(version,contract)
        steps=sorted({int(s["step"]) for s in value["optimizer"]["state"].values()})
        if steps!=[8068*epoch]:raise ValueError("cumulative Adam steps mismatch")
        for state in value["optimizer"]["state"].values():
            if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in state.values()):
                raise ValueError("nonfinite saved Adam")
        if any(not torch.isfinite(v).all() for v in value["generator"].values()):
            raise ValueError("nonfinite saved G")
        result=audit_epoch(value["progress"],m["entries"],read_rows(round_path(value,output)),epoch)
        if result["optimizer_steps"]!=8068 or result["attribute_updates"]!=6020 or result["joint_updates"]!=2048:
            raise ValueError("formal Epoch counts mismatch")
        result.update(checkpoint_sha256=io.sha(version/"checkpoint.pt"),
            sample_log_sha256=value["progress"]["sample_log_sha256"],Adam_steps=steps,
            supplementary_log_sha256=value["progress"]["supplementary_log_sha256"])
        summaries.append(result);validations.append(value["progress"]["validation"])
        if epoch==last_epoch:final_hash=state_hash(value["generator"])
        del value
    deltas={f"{n-1}_to_{n}":validation_comparison(validations[n-2],validations[n-1]) for n in range(2,last_epoch+1)}
    deltas[f"1_to_{last_epoch}"]=validation_comparison(validations[0],validations[-1])
    result={"method_audit_passed":True,"epochs":summaries,"total_optimizer_steps":sum(s["optimizer_steps"] for s in summaries),
        "final_Adam_step":8068*last_epoch,"validation_deltas":deltas,"final_G_state_sha256":final_hash,
        "new_scoring_backwards":0,"new_JVPs":0,"gallery_evaluation":False,"Epoch11_started":False}
    scalar_only(result)
    if (stage/"protected_sources.json").exists():io.verify(io.load(stage/"protected_sources.json"))
    if (stage/"execution_protocol.json").exists():io.verify(io.load(stage/"execution_protocol.json")["new_source_sha256"])
    return result

def inference_check(checkpoint):
    # 全新CPU进程只导入image-only推理模块；不读取文本、属性、JVP或IRRA。
    script=r'''
import json,sys,torch
from inference_once import load_generator,generate_once
torch.set_num_threads(2)
g=load_generator(sys.argv[1],"cpu")
calls=[]
h=g.register_forward_pre_hook(lambda m,args:calls.append(len(args)))
image=torch.linspace(0,1,3*384*128).reshape(1,3,384,128)
r=generate_once(g,image);h.remove()
assert calls==[1] and not r["adversarial"].requires_grad
assert torch.isfinite(r["adversarial"]).all()
assert ((r["adversarial"]>=0)&(r["adversarial"]<=1)).all()
linf=float((r["adversarial"].double()-image.double()).abs().max())
assert linf<=8/255
assert all(n not in sys.modules for n in ("cache_io","training_data","irra_image_adapter","serial_training","integration","deps"))
assert not torch.cuda.is_initialized()
print(json.dumps({"G_forwards":1,"shape":list(r["adversarial"].shape),"maximum_linf":linf,"CPU_image_only":True}))
'''
    p=subprocess.run([sys.executable,"-c",script,str(checkpoint)],cwd=io.ROOT,capture_output=True,text=True)
    if p.returncode:raise ValueError("final image-only inference check failed: "+p.stderr)
    result=json.loads(p.stdout);scalar_only(result);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--audit",action="store_true");a=p.parse_args(argv)
    if a.audit:
        result=analyze(io.OUTPUTS/"conditions_full",io.OUTPUTS/"train_v2",io.OUTPUTS/"phase28")
        result["final_inference"]=inference_check(io.OUTPUTS/"train_v2/checkpoints/epoch_0010/checkpoint.pt")
        io.save_json(io.OUTPUTS/"phase28/independent_audit.json",result)
        print("READ_ONLY_EPOCH1_10_AUDIT",result["total_optimizer_steps"],result["final_inference"])
    else:print("CPU analysis only; --audit requires completed Epoch10. No training.")
if __name__=="__main__":main()
