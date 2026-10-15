"""Epoch 1–3只读审计；不加载IRRA、不执行优化或重新准备监督。"""
import argparse,json,math
from collections import Counter
from pathlib import Path
import cache_io as io
import epoch_checkpoint as ck
import epoch_training as et
from budget_projection import DEFAULT_EPSILON
from phase26_support import summarize,stats,validation_stats

def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]

def detailed_stats(rows):
    result=summarize(rows)
    for index in range(1,8):
        result[f"attribute_round{index}"]=stats([r for r in rows if r["phase"]=="attribute" and r["update_index"]==index])
    # 同一K内比较轮次，避免不同K的样本组成造成混淆。
    for k in range(1,8):
        for index in range(1,k+1):
            result[f"K{k}_attribute_round{index}"]=stats(
                [r for r in rows if r["K"]==k and r["phase"]=="attribute" and r["update_index"]==index])
    return result

def audit_epoch(progress,entries,rows,epoch):
    train=[e for e in entries if e["role"]=="train"]
    expected=et.epoch_order(train,42,epoch)
    if (progress["epoch"]!=epoch or progress["samples"]!=len(train)
        or progress["sample_order"]!=[e["row_id"] for e in expected]):
        raise ValueError("epoch identity/order mismatch")
    if len(progress["sample_logs"])!=len(train):raise ValueError("sample log count mismatch")
    expected_steps=sum(e["K"]+1 for e in train)
    if len(rows)!=expected_steps or progress["optimizer_steps"]!=expected_steps:
        raise ValueError("epoch update counts mismatch")
    cursor=0
    for pos,(entry,sample) in enumerate(zip(expected,progress["sample_logs"]),1):
        if (sample["row_id"]!=entry["row_id"] or sample["role"]!="train"
            or sample["K"]!=entry["K"] or sample["sample_position"]!=pos):
            raise ValueError("sample identity/K mismatch")
        ids=entry.get("selected_instance_ids")
        if ids is None:ids=io.read_entry(entry)[0]["selected_instance_ids"]
        block=rows[cursor:cursor+entry["K"]+1];cursor+=len(block)
        if len(sample["rounds"])!=len(block):raise ValueError("sample round count mismatch")
        for index,(record,extra) in enumerate(zip(sample["rounds"],block),1):
            if any(extra.get(key)!=value for key,value in record.items()):
                raise ValueError("round log/checkpoint mismatch")
            if (extra["row_id"]!=entry["row_id"] or extra["K"]!=entry["K"]
                or extra["sample_position"]!=pos or extra["update_index"]!=index):
                raise ValueError("round identity mismatch")
            attr=index<=entry["K"]
            if extra["phase"]!=("attribute" if attr else "joint"):
                raise ValueError("round phase mismatch")
            if attr and extra["instance_id"]!=ids[index-1]:
                raise ValueError("attribute order mismatch")
            if (extra["cumulative_linf"]>DEFAULT_EPSILON or extra["cumulative_linf"]<0
                or not extra["finite_gradients"]):
                raise ValueError("budget/gradient audit failed")
            for key in ("budget_saturation_ratio","strict_projection_clipping_ratio"):
                if not 0<=extra[key]<=1:raise ValueError("invalid projection ratio")
    detailed=detailed_stats(rows) # 所有关键数值必须有限。
    val=progress["validation"]
    if ({r["row_id"] for r in val}!={e["row_id"] for e in entries if e["role"]=="validation"}
        or len(val)!=len({r["row_id"] for r in val})):
        raise ValueError("validation identity mismatch")
    for r in val:
        if (r["cumulative_linf"]>DEFAULT_EPSILON or
            any(not math.isfinite(r[k]) for k in ("global","local","cumulative_linf"))):
            raise ValueError("invalid validation")
    return {"epoch":epoch,"samples":len(train),"attribute_updates":sum(e["K"] for e in train),
        "joint_updates":len(train),"optimizer_steps":expected_steps,
        "updates_by_K":dict(Counter({str(k):sum(e["K"]+1 for e in train if e["K"]==k) for k in range(8)})),
        "statistics":detailed,"validation":validation_stats(val)}

def validation_comparison(previous,current):
    a={r["row_id"]:r for r in previous};b={r["row_id"]:r for r in current}
    if a.keys()!=b.keys() or len(a)!=len(previous) or len(b)!=len(current):
        raise ValueError("fixed validation IDs differ")
    out={"samples":len(a)}
    for field in ("global","local"):
        values=[b[key][field]-a[key][field] for key in sorted(a)]
        out[field]={"mean_delta":sum(values)/len(values),"min_delta":min(values),"max_delta":max(values),
                    "decreased":sum(v<0 for v in values),"increased":sum(v>0 for v in values),"equal":sum(v==0 for v in values)}
    return out

def analyze(cache,output,stage):
    from training_data import checked_cache
    m=checked_cache(cache,require_full=True);contract=io.load(Path(output)/"training_protocol.json")
    epochs=[];values=[]
    for epoch in (1,2,3):
        version=Path(output)/"checkpoints"/f"epoch_{epoch:04d}"
        value=ck.verify_version(version,contract);values.append(value)
        path=Path(output)/"round_audit.jsonl" if epoch==1 else Path(stage)/f"epoch_{epoch:04d}_round_audit.jsonl"
        result=audit_epoch(value["progress"],m["entries"],read_rows(path),epoch)
        steps=sorted({int(s["step"]) for s in value["optimizer"]["state"].values()})
        if steps!=[epoch*8068]:raise ValueError("Adam step mismatch")
        result.update(checkpoint_sha256=io.sha(version/"checkpoint.pt"),
            sample_log_sha256=value["progress"]["sample_log_sha256"],Adam_steps=steps)
        epochs.append(result)
    if any((Path(output)/"checkpoints").glob("epoch_000[4-9]")) or (Path(output)/"checkpoints/epoch_0010").exists():
        raise ValueError("unauthorized later Epoch")
    result={"method_audit_passed":True,"epochs":epochs,
        "validation_deltas":{f"{a}_to_{b}":validation_comparison(values[a-1]["progress"]["validation"],values[b-1]["progress"]["validation"])
                             for a,b in ((1,2),(2,3),(1,3))},
        "new_scoring_backwards":0,"new_JVPs":0,"Epoch4_started":False,"gallery_evaluation":False}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--audit",action="store_true")
    a=p.parse_args()
    if a.audit:
        result=analyze(io.OUTPUTS/"conditions_full",io.OUTPUTS/"train_v2",io.OUTPUTS/"phase27")
        io.save_json(io.OUTPUTS/"phase27/independent_audit.json",result)
        print("READ_ONLY_EPOCH1_3_AUDIT",[(e["epoch"],e["optimizer_steps"],e["validation"]) for e in result["epochs"]])
    else:print("No training; --audit reads completed Epochs 1–3 only.")
if __name__=="__main__":main()
