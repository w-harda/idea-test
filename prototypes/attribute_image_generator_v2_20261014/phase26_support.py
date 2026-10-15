"""正式Epoch只读统计与状态摘要；不包含训练算法。"""
import hashlib,math
from collections import defaultdict
import numpy as np
import torch
FIELDS=("global","local","retain","total","G_gradient_l2","IRRA_input_gradient_l2",
        "cumulative_linf","budget_saturation_ratio","strict_projection_clipping_ratio","changed_parameter_count")
def state_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()
def stats(rows):
    out={"rounds":len(rows)}
    for key in FIELDS:
        values=[float(r[key]) for r in rows]
        if any(not math.isfinite(v) for v in values):raise ValueError("nonfinite logged metric "+key)
        out[key]={"mean":sum(values)/len(values),"min":min(values),"max":max(values)} if values else None
    out["zero_G_gradient_rounds"]=sum(r["G_gradient_l2"]==0 for r in rows)
    out["unchanged_parameter_rounds"]=sum(r["changed_parameter_count"]==0 for r in rows)
    return out
def summarize(rows):
    groups={"all":rows}
    for phase in ("attribute","joint"):groups[phase]=[r for r in rows if r["phase"]==phase]
    for k in range(8):
        groups[f"K{k}"]=[r for r in rows if r["K"]==k]
        for phase in ("attribute","joint"):groups[f"K{k}_{phase}"]=[r for r in groups[f"K{k}"] if r["phase"]==phase]
    groups["first256_joint"]=[r for r in rows if r["phase"]=="joint" and r["sample_position"]<=256]
    groups["last256_joint"]=[r for r in rows if r["phase"]=="joint" and r["sample_position"]>1792]
    return {k:stats(v) for k,v in groups.items()}
def validation_stats(rows):
    if not rows:raise ValueError("empty validation")
    if any(r["optimizer_steps"]!=0 for r in rows):raise ValueError("validation optimizer update")
    return {"samples":len(rows),"global_mean":sum(r["global"] for r in rows)/len(rows),
        "local_mean":sum(r["local"] for r in rows)/len(rows),
        "maximum_linf":max(r["cumulative_linf"] for r in rows),"optimizer_steps":0}
