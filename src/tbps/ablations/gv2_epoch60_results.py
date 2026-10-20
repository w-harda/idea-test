"""完整Epoch60及原生CPU审计成功后，读取标量并生成曲线/报告；无模型加载或评分。"""
from pathlib import Path
from dataclasses import asdict
import argparse,csv,json,os,platform,subprocess
from tbps.artifacts import ArtifactStore,FileRef,checked,read_json,sha_file

ROOT=Path("/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020")
MANAGER=Path("/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-launch-20261020")
COLUMNS=("epoch","train_global","train_local","train_retain","train_total","validation_global",
    "validation_local","validation_retain","validation_total","G_gradient_l2","lr","seconds")

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_epochs(rows):
    require([r["epoch"] for r in rows]==list(range(1,61)),"complete ordered Epoch1-60 required")
    import math
    for r in rows:
        require(all(math.isfinite(float(r[k])) for k in COLUMNS if k!="seconds"),"nonfinite training curve")
        require(r["lr"]==.0002,"frozen lr")
    return rows

def collect(config_path):
    config_path=Path(config_path);c=read_json(config_path)
    root=Path(c["formal_output"]);store=ArtifactStore(root,new_base=c["output_base"])
    require(not store.pending() and not (root/"failure").exists(),"failed/pending training")
    for k in ("protocol","completed","cpu_audit"):store.verify(k)
    protocol=read_json(root/"protocol/protocol.json")
    checked(FileRef(**protocol["extension_configuration"]))
    require(protocol["extension_configuration"]["path"]==str(config_path),"frozen extension configuration path")
    require(protocol["parent_checkpoint"]==c["parent_checkpoint"],"same parent")
    done=read_json(root/"completed/report.json");audit=read_json(root/"cpu_audit/report.json")
    require(done["status"]=="PASS" and done["completed_epoch"]==60 and done["Adam_step"]==484080,"complete Epoch60")
    require(audit["status"]=="PASS" and audit["epochs"]==50 and audit["Adam_step"]==484080,"independent training audit")
    checked(FileRef(**c["legacy_refs"]["history"]))
    old=read_json(c["history_summary"])
    require(old["method_audit_passed"] and old["final_Adam_step"]==80680,"original Epoch1-10")
    rows=[]
    for e in old["epochs"]:
        t=e["statistics"]["all"];v=e["validation"]
        rows.append({"epoch":e["epoch"],**{"train_"+k:t[k]["mean"] for k in ("global","local","retain","total")},
            "validation_global":v["global_mean"],"validation_local":v["local_mean"],"validation_retain":0.,
            "validation_total":v["global_mean"]+v["local_mean"],"G_gradient_l2":t["G_gradient_l2"]["mean"],
            "lr":.0002,"seconds":e.get("seconds","")})
    cp=ArtifactStore(root/"checkpoints",new_base=c["output_base"])
    expected=[f"epoch_{e:04d}" for e in range(11,61)]
    require(sorted(p.name for p in cp.root.glob("epoch_[0-9][0-9][0-9][0-9]"))==expected and not cp.pending(),"50 immutable checkpoints")
    points={"10":c["parent_checkpoint"]};sources={str(config_path):sha_file(config_path)}
    for e in range(11,61):
        key=f"epoch_{e:04d}";store.verify(f"claim_{e:04d}");store.verify(key);cp.verify(key)
        s=read_json(root/key/"summary.json")
        require(s["absolute_epoch"]==e and s["samples"]==2048 and s["steps"]==8068 and
            s["Adam_step"]==e*8068 and s["IRRA_parameters_buffers_exact"] is True,"Epoch stats/source")
        require(s["maximum_linf"]<=8/255,"budget")
        r=FileRef(**s["checkpoint"])
        require(Path(r.path)==cp.root/key/"checkpoint.pt","checkpoint location")
        checked(r);sources[r.path]=r.sha256
        sources[str(root/key/"summary.json")]=sha_file(root/key/"summary.json")
        rows.append({"epoch":e,**{"train_"+k:s["train"][k]["mean"] for k in ("global","local","retain","total")},
            **{"validation_"+k:s["validation"][k] for k in ("global","local","retain","total")},
            "G_gradient_l2":s["train"]["G_gradient_l2"]["mean"],"lr":s["lr"],"seconds":s["seconds"]})
        if e in (20,30,40,50,60):points[str(e)]=asdict(r)
    for p in (root/"completed/report.json",root/"cpu_audit/report.json",root/"protocol/protocol.json",Path(c["history_summary"])):
        sources[str(p)]=sha_file(p)
    checked(FileRef(**c["parent_checkpoint"]))
    return validate_epochs(rows),points,done,audit,sources

def plot(rows,dest):
    # 训练环境不安装新依赖；只读复用已有绘图Python，独立CPU子进程只读已生成CSV。
    plot_python="/home/lzf/ldx/envs/apattack/bin/python"
    code="""import csv,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
dest=Path(sys.argv[1])
with (dest/"loss_raw.csv").open() as f:rows=list(csv.DictReader(f))
x=[int(r["epoch"]) for r in rows]
fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True)
for key in ("train_global","train_total","validation_global","validation_total"):
 axes[0].plot(x,[float(r[key]) for r in rows],label=key)
for key in ("train_local","train_retain","validation_local"):
 axes[1].plot(x,[float(r[key]) for r in rows],label=key)
for a in axes:
 for e in (10,20,30,40,50,60):a.axvline(e,color="grey",alpha=.2,linestyle="--")
 a.set_ylabel("Loss (native units)");a.grid(alpha=.2);a.legend()
axes[1].set_xlabel("Absolute Epoch");fig.tight_layout()
fig.savefig(dest/"loss_curves.png",dpi=180);fig.savefig(dest/"loss_curves.svg");plt.close(fig)
print("CPU_LOSS_CURVES_PASS")
"""
    subprocess.run([plot_python,"-B","-c",code,str(dest)],check=True,
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1",
            "MPLCONFIGDIR":"/home/lzf/ldx/tmp/gv2_epoch60_train_v1/matplotlib"})

def publish(control):
    control=Path(control)
    rows,points,done,audit,sources=collect(ROOT/"configs/framework/gv2_epoch60.server.json")
    dest=MANAGER/"docs/framework/results/gv2_epoch60_training"
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/"loss_raw.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows(rows)
    plot(rows,dest)
    table="\n".join(f'| {r["epoch"]} | {r["train_total"]:.10f} | {r["validation_total"]:.10f} | {r["G_gradient_l2"]:.10f} |'
        for r in rows)
    paths="\n".join(f'- Epoch{e}：{r["path"]}；SHA-256：{r["sha256"]}' for e,r in points.items())
    report=f"""# G-v2 Epoch11→60正式连续训练与独立CPU审计

冻结训练Commit：635471b6d8a722616f87df16c300a8358861cd10。
完整50个新增Epoch及独立CPU审计均PASS，最终Adam step=484080，新增403400次更新。
父Epoch10只读引用；各Epoch2048 Train、128 Validation零更新、8068次Adam。
模型、Global/Local/Retain、逐属性K+1、原图Joint、8/255、Adam2e-4/(.5,.999)保持。
没有Scheduler、FiLM、TV或扰动惩罚，没有生成Gallery或执行三模型Full。

## 曲线与训练统计

![Loss curves](loss_curves.png)

原精度数据见loss_raw.csv；Epoch1–10来自原不可变训练摘要，Epoch11–60来自已审计扩展摘要。
这些是训练诊断量，不能代替Test检索效果；不使用Test选择Checkpoint。

| Epoch | Train Total | Validation Total | G gradient L2 mean |
| --- | --- | --- | --- |
{table}

## 六个固定Checkpoint

{paths}

每个Epoch均保留完整不可变Checkpoint；10/20/30/40/50/60为预设消融点。
原始数据、SHA、完成回执及CPU审计来源见evidence.json。
受控训练总耗时{done["seconds"]:.3f}秒（含恢复、逐Epoch缓存验证和保存），
GPU peak allocated/reserved为{done["GPU_peak_allocated_bytes"]}/{done["GPU_peak_reserved_bytes"]} bytes，
Peak RSS为{done["peak_RSS_bytes"]} bytes。启动和阶段耗时另见{control}/status.log。

历史保护：{audit["protection"]}。原G10/H7、缓存和既有Full不变。
只声称完整状态恢复及固定算法/数据顺序，不声称历史CUDA轨迹bitwise等价。
本轮训练/CPU审计已完成，后续攻击样本生成与IRRA/RaSa/APTM评估待用户单独授权。
"""
    (dest/"REPORT.md").write_text(report)
    evidence={"schema":"gv2-epoch60-completed-results-1","frozen_training_commit":"635471b6d8a722616f87df16c300a8358861cd10",
        "points":points,"sources_sha256":sources,"completed":done,"audit":audit,
        "control":str(control),"reporter_sha256":sha_file(__file__),"python":platform.python_version(),
        "new_gpu_evaluation":False}
    (dest/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    p=MANAGER/"docs/PROJECT_STATE.md";text=p.read_text();start=text.index("## 当前阶段：");end=text.index("\n## ",start+4)
    text=text[:start]+"""## 当前阶段：G-v2 Epoch11→60正式训练和独立CPU审计完成

冻结训练入口635471b，50个新增Epoch/403400更新完成，Adam最终484080。
原G10只读，2048 Train/128 Validation、K+1/原图Joint及全部原算法和来源保持。
原生CPU训练审计PASS；50份新Epoch Checkpoint及六个固定消融点有SHA索引。
本轮未生成新的对抗Gallery，未执行三模型Full，后续评估等待用户授权。
[完整训练报告与曲线](framework/results/gv2_epoch60_training/REPORT.md)；
[原始精度数据](framework/results/gv2_epoch60_training/loss_raw.csv)。

"""+text[end:]
    p.write_text(text)
    p=MANAGER/"docs/EXPERIMENT_REGISTRY.md"
    p.write_text(p.read_text()+"\n## G-v2 Epoch11→60正式训练完成\n\n50个新增Epoch与原生独立CPU审计PASS，最终Adam484080；六点Checkpoint及Loss曲线见"
        "[训练报告](framework/results/gv2_epoch60_training/REPORT.md)。没有新Gallery生成或三模型评估。\n")
    print("GV2_EPOCH60_TRAINING_REPORT_PASS",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--control",required=True)
    publish(parser.parse_args().control)
