# SGA的TBPS三模型白盒适配

代码基准8254c6db06203d9ba7ac99eda9a7206000e36878；独立分支codex/tbps-sga-whitebox-fixed，工作树/home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018。SGA正式Full尚未执行；本轮仅开发、CPU验证与资源允许时固定10图真实Smoke。

## 官方依据与算法边界

- [SGA作者仓库](https://github.com/Zoky-2020/SGA)，固定Commit f49e6926cd13148e49bdbc2c3e537660e096230c。服务器/home/lzf/ldx/projects/SGA-official，18项官方Python/README/LICENSE来源SHA锁定；许可证原文见SGA_LICENSE.txt。HTTPS clone首次TLS失败，改用GitHub SSH获取同一官方仓库成功，未使用非官方镜像。
- [SGA论文](https://arxiv.org/abs/2307.14061)，ICCV2023。原方法利用多Caption与多尺度集合引导，并包含图文交替扰动；本项目先固定只攻击Gallery图像的适配分支。
- 官方attacker.py::ImageAttacker默认eps=2/255、10轮、step_size=0.5/255。这里仅将预算改为8/255，其他参数保持；不因统一预算将步长或轮数额外扩大。
- get_scaled_imgs直接调用固定官方代码：原图尺度加0.5/0.75/1.25/1.5，每个增强尺度加入std0.05的Gaussian噪声，BICUBIC缩放、clip、再BICUBIC回原大小。回放可能轻微越过[0,1]的中间增强值保留官方行为；最终保存的攻击RGB必须严格[0,1]及8/255。
- 保留多Caption之和与多尺度之和的集合梯度、梯度mean-abs归一化、sign上升和随机起点；不是普通单Caption/单尺度PGD冠名SGA。按图像ID与seed42派生固定NumPy种子，增强结束恢复外部RNG。
- Image-only保持全部Query原文，不执行原SGAttacker两次BERT文本替换；明确标为SGA-TBPS White-box适配。
- 为控制显存逐尺度计算输入梯度，再回传官方增强图；不保存历史模型图。CPU对官方独立原循环做数值对照，不能据此宣称GPU批5与microbatch完全bitwise一致。

每张Gallery只生成一张RGB。Caption集合为完整官方annotation中该图paired_image_id对应的全部Caption，按Query原序；不选择同PID其他图的文本，也不按攻击效果挑选。Smoke评估10 Gallery+10 Query，攻击指导使用这10图完整20条原Caption；这些是只读文本特征缓存，不是额外真实文本前向。

## 模型可微目标与原生评估

| Target | 白盒攻击目标（最大化） | 最终评估 |
| --- | --- | --- |
| IRRA | 负的原生L2图像/原文点积之和 | 原生L2 dot排序 |
| RaSa | 负的原生ITM正匹配logit之和 | 原生global Top-128候选→ITM融合 |
| APTM | 负的原生ITM正匹配logit之和 | 原生Top-128→ITM/full-matrix minmax+0.002 global |

RaSa/APTM攻击梯度真正穿过各自冻结视觉/融合层，不把IRRA cosine用作其融合目标。逐图ITM目标不包含离散Top-K，APTM也不做逐图伪造全矩阵minmax；因此不是其完整原生排名分数的精确导数。这是明确的白盒适配差异，详见[ADR-0008](../decisions/0008-sga-tbps-whitebox.md)。

受害者参数始终eval/requires_grad=false，无Optimizer；只对RGB求导。每个目标独立产生像素，不能跨目标混用并称为白盒。G/H/B3不加载、不训练、不修改，也不用于SGA初始化。

## 复用与新增

复用Dataset Manifest、generation_v3原RGB、只读FullReference（仅Clean文本及Clean控制指标）、三个官方加载器、ArtifactStore、MaterializedStore的image_checks、FeatureCache/NativeFeatureCache、ChunkCachedScorer/APTMSetScorer及EvaluationEngine。未改任何旧Adapter、B3、G/H或Ranking文件。

新增：
- baselines/sga_attack.py：官方增强加载、集合迭代、严格FP32预算、固定Caption分组/种子。
- models/sga_surrogate.py：目标专属可微前向，原生视觉/融合模块。
- baselines/sga_whitebox.py：CPU门禁、生成Claim/紧凑像素/读回、特征、原生排名与CPU审计。
- sga_eval.py：冻结旧输入命名空间，保持原Plan；SGA逻辑在独立baselines模块。
- baselines/run_sga_full.sh：仅用户手动Full的轻量顺序执行器，不轮询、不部署、不自动重试。
- configs/framework/sga_whitebox.server.json、tests/framework/test_sga_whitebox.py及本报告/ADR。

SGA Store独立于原固定G/H MaterializationPlan，按模型、scope、像素内容及模型SHA绑定。原Store类型不放宽，历史Plan/产物不改。每图先Claim再迭代，保存原RGB/Adv RGB/effective_delta及逐轮Loss/梯度/预算，立即bitwise读回；未提交Claim拒绝自动再攻击。

## 实际CLI：用户复制执行

统一环境：
~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/src
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/home/lzf/ldx/tmp TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
~~~

默认CPU预检（将irra替换成rasa/aptm即可）：
~~~bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.sga_eval --config /home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/configs/framework/sga_whitebox.server.json --action preflight --scope full --model irra
~~~
预期SGA_CPU_PREFLIGHT_PASS full irra；不得触发GPU、模型加载或Full排名。

真实10图Smoke（已有成功输出后不要重复执行）：
~~~bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.sga_eval --config /home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/configs/framework/sga_whitebox.server.json --action run --scope smoke --model irra --allow-smoke
~~~
成功标志SGA_TBPS_EVALUATION_PASS smoke irra。每目标10图、10轮、5尺度，500个攻击视觉样本前向+10个最终编码；Clean文本和Clean图像特征从已审计历史缓存读取。模型只恢复1次。

独立CPU审计：
~~~bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.sga_eval --config /home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/configs/framework/sga_whitebox.server.json --action audit --scope smoke --model irra
~~~
成功标志SGA_TBPS_CPU_AUDIT_PASS smoke irra；不重排完整指标。其他两个目标相同参数替换。

**用户审阅后手动启动三模型Full的一条命令；Codex本轮不执行：**
~~~bash
mkdir -p /home/lzf/ldx/tmp/sga_whitebox_full_v2 && { nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/src/tbps/baselines/run_sga_full.sh --allow-full >> /home/lzf/ldx/tmp/sga_whitebox_full_v2/launcher.log 2>&1 < /dev/null & echo "启动器PID=$!"; }
~~~

脚本首先要求三个Smoke的完整CPU审计均PASS，再依次IRRA Full→审计、RaSa Full→审计、APTM Full→审计；每目标生成自己的3074图像。仅四指标R@1/R@5/R@10/mAP，两位小数。Full Clean显式复用同模型已审计Full数值及原Clean文本表示，不谎称重新编码控制组。尚无成功Smoke的目标不能绕过检查进入Full。

查看阶段/日志/结果：
~~~bash
cat /home/lzf/ldx/tmp/sga_whitebox_full_v2/phase
tail -n 30 /home/lzf/ldx/tmp/sga_whitebox_full_v2/status.log
tail -n 30 /home/lzf/ldx/tmp/sga_whitebox_full_v2/irra-full.log
for m in irra rasa aptm; do
  cat "/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/full/$m/completed/metrics_2dp.csv"
done
~~~

SSH关闭后nohup会继续。需要停止时仅终止本任务独立进程组；先核实supervisor.pid==supervisor.pgid和命令行是run_sga_full.sh，再对该负PGID发TERM。不停止其他GPU用户。不提供自动恢复/重试；失败保留Claim、错误及pending，必须人工审查后再决定后续方案。

## 输出与资源约束

新输出/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/{smoke,full}/{irra,rasa,aptm}/，所有旧结果只读。GPU共享只观察一次即时资源，不等待Ollama退出、不抢占；最低空闲3584MiB，不足明确停止，实际OOM保留完整错误，不调精度、batch或算法。

源配置、官方Commit、完整ModelSpace、文本表示缓存来源、Caption集合、原图与对抗内容SHA全部绑定。CPU门禁/输入核验先于CUDA，模型加载次数、真实视觉/融合Hook、梯度次数、峰值显存/RSS/耗时在execution/report.json；逐图10轮记录在pixels/image-*/metadata.json。

固定Full攻击视觉样本数每目标153700（3074×10×5），另3074次最终编码；融合目标攻击pair总数307800（6156×10×5），还需原生最终Top128评分。CPU文本特征可复用，真实文本前向0。Full像素每目标约5.066GiB（三FP32 Tensor），原生特征/Score会增加约数GiB，三目标需预留至少35GiB并根据实际缓存核算。完成时间/峰值需以本轮Smoke实测更新，不能用Generator一次前向耗时估计迭代攻击。

## 原始运行环境差异

官方requirements.txt为Torch2.0.0、Torchvision0.15.0、Transformers4.28.1；本适配沿用已验证TBPS环境{"torch": "2.5.1+cu124", "torchvision": "0.20.1+cu124", "numpy": "1.26.4", "Pillow": "11.3.0", "transformers": "4.44.2", "timm": "0.9.8"}。官方Resize未显式传antialias，Tensor默认行为可能随Torchvision版本变化；这里直接使用当前固定环境调用官方源码，不能声称与原requirements环境或论文数值bitwise等价。CPU官方对照也限于同一当前环境，不修改三个Victim的已冻结环境。

## 初始数值失败与派生修复

初始IRRA尝试因FP32低端预算舍入FAIL，原树/Commit80361ec1fdb7c4c96e4ec68fa256742cc8188566、原输出tbps-framework-sga-whitebox/smoke/irra/和Claim保持。详见SGA_INITIAL_FAIL.md。派生树只将边界先以FP64计算再转FP32、完善失败计数；最终RGB/模型仍FP32、8/255无容差。配置和输出改为独立v2，不覆盖旧来源。121项CPU测试通过（42SGA+79相关；5项旧路径测试筛选未运行）。当前正式Full目录仍不存在。

三个目标真实Smoke及CPU审计均PASS，数值与边界见[Smoke报告](SGA_SMOKE_REPORT.md)。仅当前固定10+10范围，正式Full仍PENDING。
