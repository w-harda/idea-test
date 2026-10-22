"""补充只读资源统计和奖励/排名诊断；不导入Torch或调用旧审计。"""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,math,io
import numpy as np
ROOT=Path(__file__).resolve().parents[3];O=ROOT/"docs/framework/results/hv1_g60_posthoc_analysis"
def rows(n):return list(csv.DictReader((O/n).open()))
def write(n,rs):
    keys=list(dict.fromkeys(k for r in rs for k in r))
    stream=io.StringIO(newline="")
    w=csv.DictWriter(stream,keys,lineterminator="\n");w.writeheader();w.writerows(rs)
    value=stream.getvalue().encode("utf-8");p=O/n
    if p.exists():
        assert p.read_bytes()==value,"existing derived CSV differs; refuse overwrite"
    else:p.write_bytes(value)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
def pearson(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
def ranks(x):
    x=np.asarray(x);return np.array([np.mean(np.where(np.sort(x)==v)[0]) for v in x])
s=json.loads((O/"statistics.json").read_text());tr={x["epoch"]:x for x in s["training_epochs"]}
resource=defaultdict(lambda:dict(visits=0,updates=0,step_seconds=0.,max_step_seconds=0.,
                               GPU_allocated_MiB=0.,GPU_reserved_MiB=0.))
for item in s["training_completed"]["ledger"]["published"]:
    p=Path(item["path"]);md=json.loads((p/"checkpoint.json").read_text());h=hashlib.sha256()
    with (p/"steps.jsonl").open("rb") as f:
        for line in f:
            h.update(line);d=json.loads(line);e=d["epoch"];v=resource[e];v["visits"]+=1
            if d["optimizer_steps_this_step"]:
                v["updates"]+=1;v["step_seconds"]+=d["seconds"]
                v["max_step_seconds"]=max(v["max_step_seconds"],d["seconds"])
                v["GPU_allocated_MiB"]=max(v["GPU_allocated_MiB"],d["GPU_peak_allocated_MiB"])
                v["GPU_reserved_MiB"]=max(v["GPU_reserved_MiB"],d["GPU_peak_reserved_MiB"])
    assert h.hexdigest()==md["sample_log_sha256"]
resource_rows=[dict(epoch=e,**v,mean_step_seconds=v["step_seconds"]/v["updates"]) for e,v in sorted(resource.items())]
write("training_resource_epoch_statistics.csv",resource_rows)
q=rows("per_query_analysis.csv");base={int(x["query_index"]):x for x in q if x["candidate"]=="Image-only"}
vr=[];by={}
for key in ["H"+str(e) for e in list(range(1,11))+[20,30,40,50,60]]:
    rr=[x for x in q if x["candidate"]==key and x["group"]=="Joint"];by[key]=rr
    AP=np.array([float(x["AP"]) for x in rr]);bAP=np.array([float(base[int(x["query_index"])]["AP"]) for x in rr])
    R=np.array([float(x["R_sim_posthoc"])+float(x["R_rank_posthoc"]) for x in rr])
    delta=AP-bAP
    text=[x for x in q if x["candidate"]==key and x["group"]=="Text-only"]
    nsuccess=sum(int(base[int(x["query_index"])]["rank1"])==1 and int(x["rank1"])==0 for x in rr)
    repair=sum(int(base[int(x["query_index"])]["rank1"])==0 and int(x["rank1"])==1 for x in rr)
    vr.append(dict(candidate=key,R_sim_mean=np.mean([float(x["R_sim_posthoc"]) for x in rr]),
        R_rank_mean=np.mean([float(x["R_rank_posthoc"]) for x in rr]),R_total_mean=float(R.mean()),
        R_total_positive=int((R>0).sum()),R_total_negative=int((R<0).sum()),R_total_zero=int((R==0).sum()),
        positive_R_but_AP_increased=int(((R>0)&(delta>0)).sum()),
        negative_R_but_AP_decreased=int(((R<0)&(delta<0)).sum()),
        Rsim_negative_but_AP_decreased=sum(float(x["R_sim_posthoc"])<0 and float(x["AP"])<float(base[int(x["query_index"])]["AP"]) for x in rr),
        paired_R_AP_damage_Pearson=pearson(R,-delta),
        Joint_margin_median=float(np.median([float(x["margin"]) for x in rr])),
        Text_only_margin_median=float(np.median([float(x["margin"]) for x in text])) if text else None,
        new_Rank1_attack_success=nsuccess,restored_Rank1_match=repair))
write("validation_reward_diagnostics.csv",vr)
phases=[]
for lo,hi in [(11,20),(21,30),(31,40),(41,50),(51,60)]:
    subset=[tr[e] for e in range(lo,hi+1)]
    phases.append(dict(epoch_begin=lo,epoch_end=hi,
        **{k:float(np.mean([x[k] for x in subset])) for k in ["sampled_total_mean","greedy_total_mean","zero_gradient_rate","same_text_rate","greedy_mean_edits","greedy_NOOP_rate","greedy_STOP_rollout_rate"]}))
write("training_phase_statistics.csv",phases)
selected=[20,30,40,50,60]
vmap={x["H_epoch"]:x for x in s["validation"]}
x=[tr[e]["sampled_total_mean"] for e in selected];y=[vmap[e]["Joint mAP"] for e in selected]
last=[tr[e]["sampled_total_mean"] for e in range(51,61)]
facts=dict(train_reward_Joint_mAP_Pearson_n5=pearson(x,y),train_reward_Joint_mAP_Spearman_n5=pearson(ranks(x),ranks(y)),
  late_E51_E60_sampled_reward_slope_per_epoch=float(np.polyfit(range(51,61),last,1)[0]),
  phases=phases,validation_reward=vr,
  same_text_equals_zero_gradient_all_50=all(tr[e]["same_text_rate"]==tr[e]["zero_gradient_rate"] for e in range(11,61)),
  all_eligible_H_updates_have_nonzero_parameter_delta=all(tr[e]["parameter_delta_l2_zero_count"]==0 for e in range(11,61)),
  conditional_nonzero_gradient_mean={str(e):tr[e]["gradient_norm_mean"]/(1-tr[e]["zero_gradient_rate"]) for e in [11,20,30,40,50,60]},
  total_recorded_step_seconds=sum(x["step_seconds"] for x in resource_rows),
  total_step_seconds_excludes_Checkpoint_Validation_IO=True,
  image_only_correct_margin_median=float(np.median([float(x["margin"]) for x in base.values() if x["rank1"]=="1"])),
  image_only_correct_margin_below_001=sum(float(x["margin"])<.01 for x in base.values() if x["rank1"]=="1"),
  summary_vs_perquery_float64_rounding_note="Official summary uses FP32 mean; paired AP differences use FP64 means of stored-native FP32 AP, error <1e-5pp.")
# 原Full摘要/CSV按真实原精度核对；这里只读标量，不加载Full分数。
for row in rows("historical_full_context.csv"):
    p=Path(row["source"]);d=json.loads(p.read_text());groups=d.get("groups",d.get("metrics"))
    if row["method"] in groups:g=groups[row["method"]]
    elif row["method"]=="G-v2 Image-only":g=groups["Image-only"]
    elif row["method"]=="G-v2 + H-v1 Joint":g=groups["Joint"]
    elif row["method"]=="H-v1 Text-only":g=groups["Text-only"]
    else:g=groups["Clean"]
    for k in ["R@1","R@5","R@10","mAP"]:assert float(row[k])==float(g[k]),(p,k)
# 将已经消费且也在原协议中声明的源码/缓存进一步逐项匹配；不放宽原SHA。
protocol=json.loads(Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/training/protocol.json").read_text())
manifest=json.loads((O/"INPUT_MANIFEST.json").read_text());matches=0
for item in manifest:
    if item["path"] in protocol["source_sha256"]:
        assert item["sha256"]==protocol["source_sha256"][item["path"]];matches+=1
facts["consumed_sources_matching_original_protocol"]=matches
facts["historical_full_scalar_CSV_matches_actual_summaries"]=True
(O/"supplementary_statistics.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2)+"\n")
print("SUPPLEMENTARY_STATISTICS_PASS")
print(json.dumps(facts,ensure_ascii=False))
