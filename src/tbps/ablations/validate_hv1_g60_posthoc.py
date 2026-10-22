"""交付核验与轻量来源登记：只读原实验，写入本轮独立分析目录。"""
from pathlib import Path
import csv,json,hashlib,xml.etree.ElementTree as ET
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];O=ROOT/"docs/framework/results/hv1_g60_posthoc_analysis"
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
def state(p):
    s=Path(p).stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
def ref(p):return dict(path=str(p),sha256=sha(p),metadata=state(p))
def rows(n):return list(csv.DictReader((O/n).open()))
def dump(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
source=read(O/"SOURCE_EVIDENCE.json");manifest=read(O/"INPUT_MANIFEST.json")
historical=read(Path("/home/lzf/ldx/projects/idea-test-tbps-g60-h-validation-20261021/docs/framework/results/hv1_epoch7_training_selection_audit/SOURCE_EVIDENCE.json"))
clean_controls=[]
for candidate in historical["candidates"]:
    clean_controls.append(read(candidate["validation_summary_path"])["groups"]["Clean"])
for epoch in (20,30,40,50,60):
    clean_controls.append(read(Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/validation")/f"epoch_{epoch:04d}/completed/summary.json")["groups"]["Clean"])
assert all(all(c[k]==100. for k in ("R@1","R@5","R@10","mAP")) for c in clean_controls)
source["Clean_control_actual_summary_verified_count"]=len(clean_controls)
source["Clean_control_from_actual_summaries"]={k:clean_controls[0][k] for k in ("R@1","R@5","R@10","mAP")}
assert sha(O/"INPUT_MANIFEST.json")==source["input_manifest_sha256"]
for item in manifest:assert state(item["path"])==[item["size"],item["mtime_ns"],item["inode"]],item["path"]
assert sha(ROOT/"src/tbps/ablations/hv1_g60_posthoc.py")==source["analysis_script_sha256"]
t=rows("training_epoch_statistics.csv");old=rows("historical_training_epoch_statistics.csv")
assert [int(x["epoch"]) for x in t]==list(range(11,61)) and len(old)==10
assert sum(int(x["visits"]) for x in t)==102400 and sum(int(x["updates"]) for x in t)==101850
assert all(int(x["visits"])==2048 and int(x["updates"])==2037 and int(x["skips"])==11 for x in t)
v=rows("validation_comparison.csv");expected=list(range(1,11))+[20,30,40,50,60]
assert [int(x["H_epoch"]) for x in v]==expected
q=rows("per_query_analysis.csv");assert len(q)==2688
for candidate,group in {(x["candidate"],x["group"]) for x in q}:
    rr=[x for x in q if x["candidate"]==candidate and x["group"]==group]
    assert len(rr)==128 and len({x["row_id"] for x in rr})==128
    assert [int(x["query_index"]) for x in rr]==list(range(128))
    assert all(0<=float(x["AP"])<=1 for x in rr)
ck=rows("checkpoint_inventory.csv");assert len(ck)==481
new=[x for x in ck if x["phase"]=="G60 adaptation"]
assert len(new)==400 and sum(x["reason"]=="epoch_complete" for x in new)==50
fixed=[dict(H_epoch=int(x["cursor"])//2048,visit=int(x["cursor"]),checkpoint_path=x["path"]+"/checkpoint.pt",sha256=x["checkpoint_sha256"])
       for x in ck if x["reason"]=="epoch_complete" and int(x["cursor"]) in [20480,40960,61440,81920,102400,122880]]
assert len(fixed)==6
with (O/"fixed_checkpoint_sources.csv").open("x",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,list(fixed[0]),lineterminator="\n");w.writeheader();w.writerows(fixed)
figures=sorted((O/"figures").glob("*.png"));assert len(figures)==12 and len(list((O/"figures").glob("*.svg")))==12
title={
"G10_G60_validation_context":"G10/G60候选排序与额外下降","paired_PID_bootstrap":"配对PID聚类Bootstrap区间",
"parameter_updates":"实际参数更新及非零梯度条件幅度","per_query_AP_changes":"逐Query AP变化分布",
"per_query_AP_heatmap":"相对Image-only的AP变化热图","replacement_attribute_distribution":"Unicode字符与原始属性标签",
"reward_validation_alignment":"训练Reward与Validation分离","training_loss_gradient":"Loss、Advantage与梯度",
"training_policy":"训练编辑/STOP/NOOP及采样趋同","training_rewards":"Reward各分项与诊断synergy",
"validation_behavior":"Validation预算与实际动作","validation_metrics":"全部15候选Text-only/Joint指标"}
index="# 图表索引\n\n全部1920×1080 PNG和矢量SVG，横坐标为真实Epoch/Query index，仅连接实测节点。\n\n| 图 | PNG | SVG |\n| --- | --- | --- |\n"
for p in figures:
    with Image.open(p) as im:assert im.size==(1920,1080);im.verify()
    ET.parse(p.with_suffix(".svg"))
    index+="| "+title[p.stem]+" | [PNG](figures/"+p.name+") | [SVG](figures/"+p.stem+".svg) |\n"
(O/"FIGURES.md").write_text(index)
# 仅补充本轮新报告的证据解释，不改变原报告失败现场。
p=O/"REPORT.md";txt=p.read_text()
txt+="""

## 交付补充

[全部12图PNG/SVG索引](FIGURES.md)；[六个固定节点的Checkpoint来源](fixed_checkpoint_sources.csv)。

训练E11–20的Greedy NOOP十轮均值约2.84%，E51–60约1.16%，
平均编辑数约1.849→1.832；这不支持“后期NOOP持续增多”的强断言。
相对历史H9存在位置NOOP和较少编辑的变化，但H20之后的更弱Joint不能简单归因于不断提前STOP。
图中显示的是事件频率，绝不是未保存的策略熵。

Validation派生R_total在validation_reward_diagnostics.csv中使用Python浮点相加，
更贴近RewardEvaluator将两个FP32差值转成Python float后求和的语义；
per_query_analysis.csv的R_total_posthoc保留初始FP32相加诊断值，可能有约1e−8尾差。
两者都只是本轮只读派生量，原始训练Reward和原实验指标完全没有修改。

Source、原始控制FAIL和报告残留均保持；本交付仅表示CPU分析通过，不将旧监督器重标为SUCCESS。
"""
p.write_text(txt)
config=read(Path("/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021/configs/framework/hv1_g60_adaptation.server.json"))
protected=read(config["sources"]["protection_snapshot"]["path"])
for p,s in protected.items():assert state(p)==s,p
for p,h in config["producers"].items():assert sha(p)==h,p
selection=Path("/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/selections/30f573098d28b6af6afe146330823986f724378fc7bd4df9d257fdd2c6f20dd1.json")
assert sha(selection)=="584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757"
# 只读复核原Full审计文件和已有审计日志，不运行审计入口。
full=read(ROOT/"docs/framework/results/gv2_epoch_sweep_full_analysis/SOURCE_EVIDENCE.json")
full_refs=[]
base=Path("/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full")
for a in full["audits"]:
    if a["epoch"] not in [10,60]:continue
    # 实际审计目录由对应Epoch输出定位，不假设新旧审计的模式名称。
    e=base/f"epoch_{a['epoch']:04d}";paths=list(e.glob("**/manifest.json"))
    matches=[p for p in paths if sha(p)==a["manifest_sha256"]]
    assert len(matches)==1,(a["epoch"],a["model"])
    md=read(matches[0]);rc=read(matches[0].with_name("completed.json"))
    assert rc["manifest_sha256"]==a["manifest_sha256"]
    assert read(matches[0].with_name("report.json"))==a["report"]
    full_refs.append(ref(matches[0]));full_refs.append(ref(matches[0].with_name("report.json")))
    if "historical_audit_referenced" in a["report"]:
        z=a["report"]["historical_audit_referenced"];assert sha(z["log"]["path"])==z["log"]["sha256"]
        assert z["marker"] in Path(z["log"]["path"]).read_text()
        full_refs.append(ref(z["log"]["path"]))
tmp=Path("/home/lzf/ldx/tmp/hv1_g60_posthoc_analysis_20261022")
assert read(tmp/"cpu_statistics_tests_exit.json")["exit_code"]==0
assert read(tmp/"analysis_exit.json")["exit_code"]==0
assert read(tmp/"supplementary_confirmed_exit.json")["exit_code"]==0
assert read(tmp/"plots_exit.json")["exit_code"]==0
assert read(tmp/"report_exit.json")["exit_code"]==0
additional=[ref(p) for p in sorted((ROOT/"src/tbps/ablations").glob("*hv1_g60_posthoc*.py"))]
additional+=[ref(ROOT/"tests/framework/test_hv1_g60_posthoc.py"),ref(selection)]
additional+=[ref(p) for p in sorted(tmp.glob("*.log"))]+[ref(p) for p in sorted(tmp.glob("*exit.json"))]+full_refs
source.update(status="H60_POSTHOC_ANALYSIS_AND_DELIVERY_PASS",additional_sources=additional,
              historical_Full_audits_read_only_verified=6,H7_Selection_unchanged=True,
              CPU_statistics_tests=10,visual_review="12-panel contact sheet reviewed; no clipping or empty charts",
              first_supplementary_format_failure_preserved=True,old_controller_FAIL_not_modified=True)
check=dict(status="PASS",new_epochs=50,visits=102400,new_Adam_updates=101850,checkpoint_records=481,
 new_checkpoints=400,epoch_boundaries=50,Validation_candidates=15,per_query_records=len(q),
 saved_score_matrices_checked=21,ranking_and_metrics_native_exact=True,PNG=12,SVG=12,
 no_model_forward=True,no_GPU=True,no_training=True,no_old_audit_entry=True,original_selection_sha256=sha(selection))
dump("DELIVERABLE_CHECK.json",check)
# 清单不自引用；SOURCE_EVIDENCE绑定清单SHA，故也从该清单排除。
derived=[dict(path=str(p.relative_to(O)),sha256=sha(p),bytes=p.stat().st_size)
         for p in sorted(O.rglob("*")) if p.is_file() and p.name not in ["SOURCE_EVIDENCE.json","DELIVERABLE_MANIFEST.json"]]
dump("DELIVERABLE_MANIFEST.json",derived)
source["derived_manifest_sha256"]=sha(O/"DELIVERABLE_MANIFEST.json")
dump("SOURCE_EVIDENCE.json",source)
print("H60_POSTHOC_DELIVERY_PASS",json.dumps(check))
