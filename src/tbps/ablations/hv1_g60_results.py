"""只读整理已完成适配训练与Validation；不运行任何模型或排名。"""
from pathlib import Path
import csv,json,hashlib,subprocess,os,sys
BASE=Path("/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60")
ROOT=Path(__file__).resolve().parents[3]
def sha(p):return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def csv_write(path,rows,rounded=False):
    with path.open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader()
        writer.writerows({k:format(v,".2f") if rounded and type(v)is float else v for k,v in row.items()} for row in rows)
def main():
    audit=json.loads((BASE/"training_audit/result/report.json").read_text())
    assert audit["status"]=="HV1_G60_TRAINING_CPU_AUDIT_PASS"
    done=json.loads((BASE/"training/completed.json").read_text());assert done["cursor"]==122880 and done["updates"]==122220
    out=ROOT/"docs/framework/results/hv1_g60_adaptation_epoch60";out.mkdir(parents=True,exist_ok=False)
    c=json.loads((ROOT/"configs/framework/hv1_g60_adaptation.server.json").read_text())
    old=json.loads(Path(c["sources"]["history_evidence"]["path"]).read_text())
    previous=json.loads((Path(c["validation_root"])/"completed/summary.json").read_text())
    summaries={}
    summaries[10]={"groups":{"Clean":{k:old["candidates"][9]["groups"]["Clean"][k] for k in ("R@1","R@5","R@10","mAP")},
        "Image-only":previous["groups"]["G60 Image-only"],
        "Text-only":{k:old["candidates"][9]["groups"]["Text-only"][k] for k in ("R@1","R@5","R@10","mAP")},
        "Joint":previous["groups"]["G60 + H10 Joint"]},"checkpoint_sha256":c["parent"]["sha256"],"visit":20480}
    for e in (20,30,40,50,60):summaries[e]=json.loads((BASE/f"validation/epoch_{e:04d}/completed/summary.json").read_text())
    rows=[];comparison=[]
    for e,s in summaries.items():
        for name in ("Clean","Image-only","Text-only","Joint"):
            rows.append({"H Epoch":e,"Training image guide":"G10 (historical)" if e==10 else "G60 adaptation",
                         "Group":name,**s["groups"][name]})
        comparison.append({"H Epoch":e,"Training image guide":"G10" if e==10 else "G60 (Epoch11 onwards)",
            "Text-only R@1":s["groups"]["Text-only"]["R@1"],"Text-only mAP":s["groups"]["Text-only"]["mAP"],
            "Joint R@1":s["groups"]["Joint"]["R@1"],"Joint mAP":s["groups"]["Joint"]["mAP"],
            "Joint extra drop pp":s["groups"]["Image-only"]["mAP"]-s["groups"]["Joint"]["mAP"]})
    csv_write(out/"validation_raw.csv",rows);csv_write(out/"validation_2dp.csv",rows,True)
    csv_write(out/"comparison_raw.csv",comparison);csv_write(out/"comparison_2dp.csv",comparison,True)
    # 逐Epoch只累计原提交日志标量；不调用训练/audit/模型。
    scalars={}
    for version in sorted((BASE/"training/checkpoints").glob("visit_*")):
        for line in (version/"steps.jsonl").open():
            r=json.loads(line);e=r["epoch"]
            a=scalars.setdefault(e,{"visits":0,"updates":0,"skips":0,"sums":{}})
            a["visits"]+=1
            if r["optimizer_steps_this_step"]:
                a["updates"]+=1
                values={k:r[k] for k in ("SCST_loss","gradient_norm","advantage","parameter_delta_l2","seconds")}
                for mode in ("sampled","greedy"):
                    values.update({mode+"_"+k:r[mode+"_reward"][k] for k in ("R_sim","R_rank","total","synergy_increment")})
                for k,v in values.items():a["sums"][k]=a["sums"].get(k,0.)+v
            else:a["skips"]+=1
    trainrows=[{"Epoch":e,"visits":a["visits"],"updates":a["updates"],"skips":a["skips"],
                **{k:v/a["updates"] for k,v in a["sums"].items()}} for e,a in sorted(scalars.items())]
    csv_write(out/"training_epoch_raw.csv",trainrows)
    chosen=min(summaries,key=lambda e:(summaries[e]["groups"]["Joint"]["mAP"],e))
    selection={"criterion":"minimum repeated internal G60 Validation Joint mAP; earliest visit tie",
       "diagnostic_not_fresh_independent_selection":True,"epoch":chosen,"visit":chosen*2048,
       "checkpoint_sha256":summaries[chosen]["checkpoint_sha256"],"Joint_mAP":summaries[chosen]["groups"]["Joint"]["mAP"],
       "candidates":{str(e):{"checkpoint_sha256":s["checkpoint_sha256"],"Joint_mAP":s["groups"]["Joint"]["mAP"]} for e,s in summaries.items()},
       "old_H7_selection_untouched":True,"official_Test_used":False}
    (out/"adaptation_selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2)+"\n")
    # Selection单独派生，不改历史Selection或训练产物。
    sources={"training_completed":{"path":str(BASE/"training/completed.json"),"sha256":sha(BASE/"training/completed.json")},
             "new_CPU_audit":{"path":str(BASE/"training_audit/result/report.json"),"sha256":sha(BASE/"training_audit/result/report.json")},
             "checkpoints":done["ledger"]["published"],"parent_H10":c["parent"],"G60":c["G60"],
             "runtime":done["runtime"],"elapsed_seconds":done["elapsed_seconds"],"audit":audit,"selection":selection}
    (out/"SOURCE_EVIDENCE.json").write_text(json.dumps(sources,ensure_ascii=False,indent=2)+"\n")
    h7=previous["groups"]["G60 + H7 Joint"]["mAP"];h9=previous["groups"]["G60 + H9 Joint"]["mAP"]
    table="| H Epoch | 引导G | Text-only R@1 | Text-only mAP | Joint R@1 | Joint mAP | 额外下降pp |\n|---|---|---|---|---|---|---|\n"
    for row in comparison:table+="| "+" | ".join(str(v) if not isinstance(v,float) else format(v,".2f") for v in row.values())+" |\n"
    trend=[summaries[e]["groups"]["Joint"]["mAP"] for e in summaries]
    textbest=min(summaries,key=lambda e:summaries[e]["groups"]["Text-only"]["mAP"])
    report=f"""# G60引导H10→H60适配续训与Validation消融

最终状态：HV1_G60_TRAINING_CPU_AUDIT_PASS。新增Epoch11–60完整50轮，102400 visits、101850次更新；
最终122880 visits、122220次累计Adam更新。原H10完整H/Adam/RNG恢复，不重跑H1–H10。
独立新输出：{BASE}。原算法、SCST、奖励权重、Adam和资格/顺序不变；唯一研究变化为延长训练并将奖励图像G10切换G60。
实际训练耗时{done["elapsed_seconds"]/3600:.3f}小时，运行资源及SHA见SOURCE_EVIDENCE.json。

## 固定六节点（IRRA内部Validation，指标百分数）

{table}

所有四指标R@1/R@5/R@10/mAP的原精度及两位小数见CSV。
H10使用历史已审计G60适配结果；其他5节点真实新H输出/IRRA文本编码，全部共用原128 G60 Gallery。
Clean及Image-only只读引用，不重新生成G或重新编码图像；保留全部128条Query/NOOP。

## 分析

预设六节点最低Joint mAP为H{chosen}：{selection["Joint_mAP"]}%；
Text-only最低为H{textbest}。Joint序列是否单调下降：{all(b<=a for a,b in zip(trend,trend[1:]))}。
最低候选相对历史G60+H7（{h7}%）进一步下降{h7-selection["Joint_mAP"]}个百分点，
相对G60+H9（{h9}%）下降{h9-selection["Joint_mAP"]}个百分点。
H60 Joint为{summaries[60]["groups"]["Joint"]["mAP"]}%，不以最终节点替代实际Validation排序。
训练奖励各分量、Loss、梯度及synergy均逐轮保存于training_epoch_raw.csv，不能只依据总奖励判断联合攻击有效。

本次同时延长训练并改变奖励图像条件，没有G10延长训练对照，无法将变化完全归因于G60引导。
内部128 Validation已重复使用，单seed/单分支，无独立确认集或置信区间；最低节点只是适配诊断。
不将其宣称为Test最优/跨模型普遍有效，也不能据小幅差异断定收敛或过拟合。
若收益趋缓，应先审查原始曲线、选择偏差与G10控制分支价值；任何新交互奖励需独立预注册和授权。
本轮未修改历史H7 Selection，未运行官方Test或RaSa/APTM评估，未自动创建G10控制续训。

## 完整性

原父H10、新扩展协议、G60图库2048图、全部Epoch/Checkpoint父子链、Adam/RNG、Unicode回放及来源由新独立CPU审计通过。
使用原train_step、RewardBank/FreshReward、Checkpoint publish/verify，未复制或调整SCST/奖励数学。
G60每图仅生成一次，缓存审计后训练G前向0；IRRA全程冻结，仅H参与优化器。
原Checkpoint/Selection/缓存/结果保护统计见审计文件。异常不自动恢复/重试。

[原精度指标](validation_raw.csv) · [两位小数](validation_2dp.csv) · [逐轮训练标量](training_epoch_raw.csv) · [来源与Checkpoint索引](SOURCE_EVIDENCE.json)。
"""
    (out/"REPORT.md").write_text(report,encoding="utf-8")
    state=ROOT/"docs/PROJECT_STATE.md"
    current=state.read_text()
    pos=current.index("## 当前阶段：")
    nextpos=current.find("\n## ",pos+3)
    tail=current[nextpos:] if nextpos!=-1 else ""
    state.write_text(current[:pos]+f"""## 当前阶段：G60引导H10→H60适配训练及内部Validation完成

新增50轮、102400 visits/101850更新，累计122880 visits/122220更新；新独立CPU审计PASS。
原H10完整Adam/RNG恢复；仅H更新，G60图库固定、IRRA冻结，旧H7 Selection不变。
预设六节点最低Joint为H{chosen}，mAP{selection["Joint_mAP"]:.5f}%，仅重复内部Validation诊断。
[正式训练/消融/原精度CSV](framework/results/hv1_g60_adaptation_epoch60/REPORT.md)。
未运行官方Test、RaSa/APTM或G10训练长度控制分支。完成后停止等待审查。
""" + tail)
    registry=ROOT/"docs/EXPERIMENT_REGISTRY.md"
    registry.write_text(registry.read_text()+f"""
## H10→H60的G60适配续训完成

独立输出{BASE}；训练及新独立CPU审计PASS，累计122220次Adam更新。
六节点最低Joint为H{chosen}，仅重复内部Validation适配诊断；历史Selection不变。
来源和Checkpoint索引见framework/results/hv1_g60_adaptation_epoch60/SOURCE_EVIDENCE.json。
""")
    # 使用已有CPU Matplotlib环境，不给训练环境安装依赖。
    script=out/"plot_results.py"
    script.write_text('''from pathlib import Path
import csv,sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=Path(__file__).parent
rows=list(csv.DictReader((r/"comparison_raw.csv").open()))
epochs=[int(x["H Epoch"]) for x in rows]
assert epochs==[10,20,30,40,50,60] and "torch" not in sys.modules
for key in ["Text-only mAP","Joint mAP","Text-only R@1","Joint R@1","Joint extra drop pp"]:
 fig,ax=plt.subplots(figsize=(12.8,7.2),layout="constrained")
 ax.plot(epochs,[float(x[key]) for x in rows],"o-")
 ax.set_xticks(epochs);ax.set_xlabel("Absolute H epoch")
 ax.set_ylabel(key+(" (percentage points)" if key.endswith("pp") else " (%)"))
 ax.set_title("G60 adaptation: "+key);ax.grid(alpha=.2)
 name=key.replace(" ","_").replace("@","")
 fig.savefig(r/(name+".png"),dpi=180);fig.savefig(r/(name+".svg"))
 p=r/(name+".svg");p.write_text("\\n".join(x.rstrip() for x in p.read_text().splitlines())+"\\n")
 plt.close(fig)
t=list(csv.DictReader((r/"training_epoch_raw.csv").open()))
for key in ["sampled_R_sim","sampled_R_rank","sampled_total","SCST_loss","sampled_synergy_increment"]:
 fig,ax=plt.subplots(figsize=(12.8,7.2),layout="constrained")
 ax.plot([int(x["Epoch"]) for x in t],[float(x[key]) for x in t],"o-",markersize=3)
 ax.set_xlabel("Absolute H epoch");ax.set_ylabel(key+" (epoch mean)");ax.grid(alpha=.2)
 fig.savefig(r/(key+".png"),dpi=180);fig.savefig(r/(key+".svg"))
 p=r/(key+".svg");p.write_text("\\n".join(x.rstrip() for x in p.read_text().splitlines())+"\\n")
 plt.close(fig)
''')
    subprocess.run(["/home/lzf/ldx/envs/apattack/bin/python","-B",str(script)],
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},check=True)
    print("HV1_G60_FINAL_REPORT_PASS",out,flush=True)
if __name__=="__main__":main()
