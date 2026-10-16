# H-v1 Phase 4 正式训练扩展：用户手动执行说明

本目录独立于 Phase 3 Smoke 与 Phase 4 的 32 Step 实验。Codex 本阶段仅开发代码、运行 CPU 测试与 2176 条 Caption 的 CPU 资格检查。真实缓存生成、GPU 奖励、正式训练、Validation、监控、停止与恢复全部由用户执行。默认入口仅 CPU preflight，任何命令均没有后台调度或自动重试。

## 数据、方法与固定处理

原 G-v2 Manifest 锁定 2048 Train（1861 PID）与 128 内部 Validation（121 PID），PID 交集 0；每张图使用原协议的唯一 Caption/row_id/caption_index。Train Gallery 固定保留全部 2048 张，Validation 固定保留全部 128 张及 128 条 Query，不扩展到原 annotation 的其他 Caption，也不使用官方 Test。

CPU 资格统计：

| 项目 | Train | Validation |
|---|---:|---:|
| 样本数 | 2048 | 128 |
| 有 Canonical 属性 | 2043 | 127 |
| 有合法、改变 Token 的字符替换且满足长度限制 | 2037 | 127 |
| 无 Canonical 属性，H 的自然 K=0 | 5 | 1 |
| 固定 NOOP 的不合格 Caption | 11 | 1 |
| 原文 BPE 截断 | 5 | 0 |
| 超过 1024 UTF-8 bytes | 0 | 0 |
| 提取 Span 全部准确 | 是 | 是 |

Train 的 11 条不合格 Caption 保留 Epoch 访问记录，固定 NOOP，不获取奖励、不执行 optimizer.step；其图像仍在训练 Gallery 中。Validation 的 1 条不合格 Query 原文 NOOP，仍参加全部指标。截断原文若出现在 Validation，固定 NOOP 并沿用官方 77-token 截断编码，绝不按攻击效果过滤。原始 BPE 截断训练文本不进入奖励。没有合法字符编辑的统计不能冒充 H 学到的 Dynamic-K 分布，后者每个有效训练 Step 重新预测并记入日志。

因此本数据协议每 Epoch **2048 visits / 2037 optimizer updates / 11 skipped visits**。有效样本即使 H 自选 STOP/NOOP，也仍按原 SCST 做一次更新；这与资格不合格的固定跳过不同。网络、三个 Head、gamma3 Dynamic-K、Unicode Mask、每属性/共享词最多一字符、原奖励 R=R_sim+R_rank、权重 1/1 以及贪心 Baseline 均不变。synergy_increment 仅诊断，不要求其为正。

H 以 seed42 新初始化，Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、batch1。**不加载旧 Step32 权重**。配置构造器候选默认 10 Epoch；CLI 新运行必须显式给出 --epochs，候选默认不代表任何实验授权。

## 独立输出与成功标志

所有新资产位于：
~~~
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/
├── qualification/           # CPU资格账本、来源协议及完成SHA
├── gallery_train/           # 2048 Clean/Adv图像特征
├── gallery_validation/      # 128 Clean/Adv图像特征与128原文特征
├── training/
│   ├── protocol.json
│   ├── checkpoints/visit_*/ # 周期完整H/Adam/RNG + 已提交steps.jsonl
│   ├── pending/             # 未提交段；崩溃后禁止静默恢复
│   ├── abandoned/           # 显式确认回滚后永久保留的尾段与回执
│   ├── attempts/
│   └── completed.json
├── validation/<H-checkpoint-SHA>/ # 每个Checkpoint独立Query缓存、summary及completed
└── selections/              # 不可变Validation选优记录
~~~

首次缓存准备每个唯一图像一次 G、一次 Clean IRRA image、一次 Adv IRRA image。有效已发布单元按身份/协议/原图SHA/特征SHA读取，不重复生成。**不复用 Phase 3/4 的协议或文本动作缓存**，也不把 Mock/Test 缓存当作正式缓存。两套正式图库各自绑定 role，Validation 不能构造训练奖励。

单元使用紧凑 clone Storage；features、metadata、receipt 在同一临时目录完成并 fsync 后一起原子发布。若发现损坏、未知单元或 pending，立即拒绝，不用零向量替代。模型SHA、官方预处理、来源与显式 producer 文件集合均锁定；新增无关源码/文档不因目录枚举使旧协议失效。运行只保留标量调用计数，不累计 forward Tensor。

## 直接执行命令

以下均为服务器 Bash 命令，复制相应完整代码块即可。CPU 命令不加载真实网络。GPU 命令仅供用户亲自决定并执行；本交付未运行任何一条 GPU 命令。

### 1. CPU Preflight 与资格检查

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action preflight --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_CPU_PREFLIGHT PASS 2048 128。核验模型 SHA、原来源、原图、划分；无模型前向。

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action qualify --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_QUALIFICATION_COMPLETE。资格账本本阶段已由 Codex CPU 生成；该命令可重复只读计算后核对原不可变文件，不覆盖。只查看既有账本可运行：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action audit --section qualification --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_CPU_AUDIT qualification PASS。

### 2. 正式 Train / Validation Gallery 缓存（仅用户）

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m formal_training.cli --action prepare --role train --confirm-gallery-generation --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_GALLERY_COMPLETE train 2048。缓存 /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/gallery_train/。

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m formal_training.cli --action prepare --role validation --confirm-gallery-generation --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_GALLERY_COMPLETE validation 128 128。后一个 128 为原文文本特征数，供所有 H Checkpoint 复用。验证集完全独立，不给 SCST 反馈。

Ctrl+C 或 SIGTERM 在当前单元结束后安全停止，打印 FORMAL_GALLERY_SAFE_STOP。恢复时执行相同命令，逐条校验并跳过有效单元。SIGKILL/OOM 后若存在 pending，保留现场，程序拒绝自动再次 G 前向；需针对证据人工定位，**不要删除 pending、回执或缓存来强行重跑**。本版不承诺恢复未发布且无法证明是否完成过 G 的单元。

两套缓存完成后 CPU 只读核验：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action audit --section gallery --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_CPU_AUDIT gallery PASS。读取并校验原有特征，不做图像/文本编码或指标计算。

### 3. 正式训练（仅用户选择轮数后执行）

下面显式采用候选 **10 Epoch**；只有用户决定实际执行时才授权这次运行。若选择其他轮数，在第一次运行前修改 --epochs 数字；首次冻结后不能在恢复时修改轮数、lr、网络、来源、缓存或资格协议。

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m formal_training.cli --action train --epochs 10 --checkpoint-every 256 --confirm-training --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_TRAIN_COMPLETE 20480 20370。每 32 visits 打印 SCST_PROGRESS；周期提交打印 H_CHECKPOINT_COMMITTED。每 Epoch 边界强制提交，默认每 256 个 H 更新额外提交一次完整 Checkpoint；CLI 周期间隔至少 32，不允许正式每步全量保存。训练中 G / IRRA image forward 都为 0，每有效更新新编码 original/sampled/greedy 三条文本，10 Epoch 上限 61110 个真实文本样本。

正常 Ctrl+C / SIGTERM：当前样本完成后提交完整状态，输出 FORMAL_TRAIN_SAFE_STOP。然后由用户恢复：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m formal_training.cli --action train --resume --confirm-training --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

如果 SIGKILL/OOM/异常留下未提交尾段，默认上述恢复命令拒绝。先 CPU 审计已发布版本并保存现场，确认接受从最后完整提交状态重新执行尾段后，由用户显式运行：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m formal_training.cli --action train --resume --rollback-uncommitted --confirm-training --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

该选项将同协议的 pending 整段移入 abandoned 并保存文件 SHA/回滚回执；绝不删除。尾段可能包含已经在内存执行、尚未写完日志的更新，这些**从未计为 committed**。恢复 H/完整 Adam/全部 RNG/样本游标后，重新执行这些访问，独立区分已提交计数与废弃尝试；旧 committed 不重复计入。未知来源的 pending 仍拒绝。**不是从任意崩溃瞬间进行 Step 级无损恢复**；精确恢复承诺针对已发布周期/安全停止边界，未来 CUDA 执行轨迹不承诺跨环境 bitwise 一致。

CPU 只读审计：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action audit --section training --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_CPU_AUDIT training PASS。检查完整版本链、Checkpoint 与段日志 SHA、样本顺序/资格跳过/累计更新、完整 Adam 与来源；输出各 Epoch 的 R_sim、R_rank、synergy_increment、Loss、梯度、K、NOOP 等分项统计，不重算奖励。

### 4. Validation 四组评估与模型选择（仅用户GPU评估）

训练与评测为分开的手动命令，没有训练进程自动调度 Validation。下式评估全部已发布 Epoch 边界 Checkpoint：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m formal_training.cli --action validate --all-epoch-checkpoints --confirm-validation --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：每个 H Checkpoint 出现 VALIDATION_COMPLETE <SHA> <Joint mAP>。比较 Clean / Image-only / Text-only / Joint，全部用同一固定 128 Gallery + 128 Query、同 PID 所有正例、官方 IRRA rank 与 FP32 余弦特征。缓存已准备，G/image 前向都是 0；原文文本特征复用，H 仅拿 caption-only TextExample，贪心 Unicode 变化才新编码。无效/NOOP Query 复用原文特征，不删除。

Ctrl+C / SIGTERM 在 Query 边界停止；重执行相同命令读取已有有效单元，完整结果也只读复用。硬中断残留 pending 会拒绝，需保留证据定位。也支持 --checkpoint 指向某个已发布 checkpoint.pt，替代 --all-epoch-checkpoints；仅接受当前训练版本链中的文件。

CPU 只读核验并选优：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action audit --section validation --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action select --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

第一条成功：FORMAL_CPU_AUDIT validation PASS。只读取完成证据与特征 SHA，不重新编码或重算检索矩阵。第二条成功：VALIDATION_SELECTION_COMPLETE，新增不可变 selections/<SHA>.json，选 **Joint mAP 最低**，并列选更早 visit，报告 Joint-minus-Image-only 百分点差与四组指标。select 是 CPU 写选优回执操作，audit 始终只读。128 规模统计有限，不与 3074 Gallery 官方 Test 的绝对数值直接比较，不据训练奖励或正式 Test 选择模型。

最终全部 CPU 审计：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m formal_training.cli --action audit --section all --root /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1
~~~

成功：FORMAL_CPU_AUDIT all PASS。

## 进度、证据和成本

所有命令前台运行。需要持久日志时由用户在终端启用 tee，先 mkdir -p /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/logs，然后把对应命令末尾加上 | tee -a <新日志路径>，并先 set -o pipefail 保留 Python 退出码；不要覆盖旧日志。原始进度可直接看终端；只读查看已提交样本日志：

~~~
find /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoints -name steps.jsonl -print
find /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoints -name checkpoint.pt -exec sha256sum {} +
pgrep -af 'formal_training.cli'
nvidia-smi
~~~

上面进程/GPU查看命令由用户使用，Codex 不持续监控。停止优先按当前前台终端 Ctrl+C；其他终端发送 SIGTERM 前须先核实自己的 PID。确认 pgrep 中目标进程消失、nvidia-smi 中该 PID 消失后再恢复或启动下一任务。不要并发写同一输出，入口有排他锁。

缓存 completion 绑定全体 feature/metadata/receipt SHA；H Checkpoint 的 payload 与 sidecar 双向关联日志 SHA，版本链锁定父 Checkpoint。模型 SHA固定：
- G-v2 E10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
- 官方 IRRA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。

仅成本估计，没有本阶段 GPU实测：2176 图缓存 2176 次 G + 4352 次 IRRA image，另 128 原文 text；依据旧10图耗时粗估约 4–10 分钟，加来源检查和 I/O。旧32 Step真实更新均值约 0.05948 秒，但其图库仅10图，不能线性承诺正式吞吐；建议预算每 Epoch 3–15 分钟，10 Epoch 约 30–150 分钟，实际以用户日志为准。预计 GPU 显存约 1–2 GiB，须用户检查其他占用，不自动调整规模或方法。

纯图像特征原始值 2176×2×512×4 = 8.5 MiB，另原文 text 0.25 MiB；实际序列化/metadata/文件系统开销另计。10 Epoch约81个完整周期/初始 Checkpoint（安全停止会额外增加），含全体顺序/身份协议估计 0.3–0.5 GiB；依据旧逐步日志平均约15KB，20370次更新约0.3 GiB日志。建议预留总计 1–2 GiB，不保存对抗像素，也不使用历史大 Storage 视图写盘。

本阶段 CPU 测试：
~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest formal_training.test_formal test_phase4 test_phase3 test_phase2 test_policy test_scst test_tokenizer_real -v
~~~

所有缓存、数据、Checkpoint、logs、outputs 均不进入 Git。正式网络效果及 GPU长程稳定性尚待用户手动实验；本交付没有新的攻击效果结论。
