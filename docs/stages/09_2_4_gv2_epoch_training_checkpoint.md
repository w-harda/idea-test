# G-v2 Phase 2.4：完整缓存入口、Epoch 调度与 Checkpoint 恢复

2026-10-15：源码与验证完成。219 项 CPU tests 通过；只准备独立的 5 条小缓存，并完成 4 train + 1 validation 的单 Epoch GPU Smoke。未执行 2176 条完整缓存准备、正式多 Epoch 训练或 Gallery 评测。按用户授权发布后停止等待独立审查，不进入 Phase 2.5。

## 位置与文件

服务器发布工作树：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014
分支：codex/gv2-per-attribute-cache
本阶段基准：3f480a653ea37edc629657234669912f3013c97b

所有新增实现位于 prototypes/attribute_image_generator_v2_20261014/：

| 文件 | 职责 |
|---|---|
| prepare_full.py | 原 2048+128 来源审计、独立新缓存准备、逐条恢复与显式 full/small 门禁 |
| prep_text_frontend.py | 仅离线使用的原 JVP 接线，保留原 z_t、S_raw、顺序与权重 |
| training_data.py | 新 Manifest 校验、依赖清单、真实图像/Caption/PID/role 接线 |
| epoch_training.py | 确定性 Epoch 调度、train/validation 隔离、样本日志与训练入口 |
| epoch_checkpoint.py | 不可变 Epoch 版本、SHA/原子发布、Adam/RNG 恢复 |
| inference_once.py | 独立 G 权重加载与 image-only 一次前向推理 |
| smoke_phase24.py | 固定五条受控 Epoch、恢复/冻结/预算/推理检查 |
| test_phase24.py | 42 项新增 CPU tests |
| README_PHASE24.md | 接口、协议、手动命令和边界 |

本中文报告在 Windows 的 docs/stages/09_2_4_gv2_epoch_training_checkpoint.md 编写，再同步至服务器同一提交。Windows 的 AGENTS.md、docs/PROJECT_STATUS.md 仅更新本地状态，不发布。原有 Windows 未提交代码与文档保持，不混入本次提交。

G-v1、官方 IRRA、Phase 1/2.1/2.2/2.3 已验证源码与资产没有修改；54 份保护文件 SHA 保持。训练模块不导入 G-v1、integration 或离线 JVP 前端。

## 新缓存与来源锁

新 Manifest 使用 gv2-training-cache-2；逐样本张量仍严格使用原 gv2-attribute-supervision-1，未修改 Phase 1 schema、cache_io 或历史 Smoke 协议。

prepare_full 默认只做 CPU 来源审计。原 2048 train / 128 validation 的 Manifest、tensor/metadata SHA、冻结原评分记录、raw annotation、图像 SHA、内部 PID 划分、原降序选择与权重均核验；本次来源审计包含 4450 项上游 SHA。没有重新调用评分或计算 Dynamic Top-K。

producer_sha256 只列出 7 个实际准备依赖：prepare_full.py、prep_text_frontend.py、training_data.py、condition_schema.py、cache_io.py、irra_image_adapter.py、budget_projection.py。不枚举 G-v2 全目录。新增 Epoch、测试或文档文件不会改变缓存锁；实际 producer 文件变化仍拒绝复用。

upstream_sha256 保留原评分、缓存和冻结数学实现的历史出处。准备阶段校验这些来源；训练阶段只校验 G-v2 条件/协议、producer 文件以及独立 dataset_sha256 中的 annotation/split，不要求 G-v1 缓存或其 Python 模块继续可用。每次加载图像另外检查源图 SHA、完整身份与内部 role。

使用 Phase 1 原 write_entry 的逐样本 SHA、fsync、目录原子发布和进程锁。恢复读取并核验已经发布的样本，不重新生成、不覆盖；pending 不能作为已完成样本。协议变化、来源指纹不匹配或已完成样本损坏均拒绝。

每条缓存保留 row_id、image_rel_path、caption_index、split、role、PID、caption、K、原降序 ID/S_raw/score_weights、[K,512] JVP、z_t/v_local。K=0 固定空 ID/分数/权重、[0,512] JVP、零 v_local、has_local=false。z_t 仅作为 Global Loss 监督，永远不传入 G。

## 少量真实准备结果

独立输出 outputs/phase24/conditions_small/，原 Phase 1 Smoke 不变。

- 固定 train K=0/1/3/7 和 validation K=4，共 5 条。
- 复用 4 条原 G-v2 独立 JVP 缓存；K=1 样本按原 TextTransport.read 补算 1 个 JVP，用于验证新入口的真实生产路径。
- new_scoring_backwards_this_run=0；new_JVPs_this_run=1；实际 JVP 内部 torch.autograd.grad 调用为 2，单列记录。
- 原 z_t 按值复用；JVP 计算所需的原 caption 文本前向属于离线线性化/一致性检查，不生成新的文本监督。
- K=1 新 JVP zero_exact=true，有限差分相对误差 0.0602028%；原 caption feature 对照 exact。
- 五条 v_local 由单属性向量与原权重重建，max_abs_error/l2_error 全部为 0。
- 原 S_raw 与 CUDA float64 权重原样保留。CPU 来源核验使用现有 Phase 1 权重公式容差 atol=1e-15/rtol=1e-14，避免把不同求和实现的最后一位当成协议变化；没有重新写权重。
- 本次只发布 5 条新缓存。outputs/conditions_full 不存在，未执行完整准备。

首次 CPU 来源 preflight 的 Python sum 与原 CUDA float64 权重精确列表比较触发末位误判；修正为原 schema 已有的公式核验容差后通过。该失败发生在 GPU 准备之前，日志保留。首轮新 CPU tests 的三处断言字段名错误也已修正，失败记录保留；未修改历史测试或算法。

## Epoch 调度与 validation 隔离

正式 CLI 必须显式 --allow-training，且只接受 scope=formal 和已完成原 2048+128 的 full-2176 缓存；小缓存无法进入正式入口。默认只 preflight，不初始化模型/GPU/optimizer。训练配置可通过 --config JSON 提供，默认模板为 10 Epoch、seed42、width32/4-block、Adam lr=2e-4/betas=(0.5,0.999)、三个 lambda=1；该模板不是正式训练已获批准的含义。

每个 Epoch 先按 row_id 排序，再由独立 random.Random("gv2|seed|epoch") shuffle；顺序不依赖外部随机调用或 Manifest 排列。每个 train 样本恰好调用一次未修改的 train_single_sample，更新数检查为 K+1，不设 K=4 cap。逐属性立即 step、上一轮 detach 图、紧邻上一轮 Retain、原图联合步和全局预算均沿 Phase 2.2。

每个样本写 JSONL 并 fsync，记录 Epoch/位置/row_id/K、全部轮次 Global/Local/Retain/总损失、G 梯度、累计 L∞、预算饱和与严格投影裁剪率。Epoch progress 保存样本顺序、样本数、总更新数和按 K 汇总的更新数。

validation 在 no_grad 下只执行一次 G/IRRA 前向，以原 z_t/v_local 给出 Global/Local 诊断，无逐属性更新、无 Retain 更新、无 optimizer.step，也不是 TBPS Gallery 指标。不根据 validation 自动选权重。测试与真实 Smoke 均确认 validation 前后 G/optimizer 状态不变。

## Checkpoint 契约

独立路径为 outputs/<本次运行>/checkpoints/epoch_0001/checkpoint.pt + checkpoint.json；current.json 只是可修复指针，完整历史版本不覆盖。

Checkpoint 保存 G、Adam、completed_epoch/next_epoch、配置、数据 Manifest/协议 SHA、明确训练依赖 SHA、Python/Torch/NumPy 版本、Python/NumPy/Torch CPU/CUDA RNG，以及 Epoch progress/样本日志 SHA。

持有运行锁后先写 pending，fsync、核验真实 payload 与 SHA sidecar，再原子发布不可变版本，最后原子更新指针。恢复扫描最高已发布 Epoch；pending 忽略。最高发布版本损坏直接拒绝，不静默回退。配置/数据/来源/环境契约不匹配也拒绝。

只支持完整 Epoch 边界恢复。中途失败会保留独立 attempt 日志，但必须从最后成功 Epoch 重做未完成 Epoch；没有样本中途或 Step 级无损恢复。CPU 同环境已验证连续训练与边界恢复逐值一致；GPU 只验证边界 G/Adam/RNG 精确恢复，不承诺后续 CUDA 轨迹 bitwise 重复。

## 单步推理

inference_once.generate_once(generator,image) 的输入仅为图像；一次 G(I) 后调用已有全局投影。没有 Caption、属性/JVP/IRRA 输入，不读取条件缓存。

load_generator 仅读取独立 Checkpoint 的架构与 G 权重字段，并先核验旁路 checkpoint.json SHA。不会导入训练、缓存或 IRRA 模块，其余训练状态不参与生成。

## CPU 验证

42 项新增 + 177 项历史回归 = 219 项，全部通过，CUDA 未初始化：

- K=0/1/3/7 的 Epoch 合计 15 次更新；每个样本只调用一次既有训练函数。
- 确定性 shuffle、完整轮日志、预算、有限梯度、encoder 冻结和 train/validation 隔离。
- 连续 2 个合成 Epoch 对比第 1 个 Epoch 保存后恢复：G 参数、完整 Adam 状态、第二个 Epoch 顺序/逐样本日志、Python/NumPy/Torch RNG 全部逐值一致。
- Checkpoint SHA/sidecar/配置/来源/数据不匹配、不可变版本、最高版本损坏拒绝、pending 忽略、指针修复、发布前后注入失败。
- 真实子进程退出后缓存逐条恢复与 Checkpoint 恢复，不覆盖已完成样本；并发写锁。
- 缓存 producer 实际相关文件变化拒绝，无关训练文件新增不影响锁。
- 原始 caption/PID/路径/内部 role/图像 SHA、符号链接越界与 Manifest/协议损坏拒绝。
- 推理一次前向、预算/像素范围与无监督/IRRA 导入。

测试命令（已执行）：

    cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014
    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase24 test_phase23 test_phase22 test_phase21 test_conditions -v

## 真实 GPU Smoke

单个共享 G/Adam 状态，1 个小 Epoch，seed42，方案 A 与三个默认 lambda 保持。

| 实际顺序 | row_id | role | K | optimizer.step | 训练函数秒数 |
|---:|---|---|---:|---:|---:|
| 1 | cuhk:26196:0 | train | 7 | 8 | 0.7954 |
| 2 | cuhk:26747:0 | train | 1 | 2 | 0.0743 |
| 3 | cuhk:9955:0 | train | 3 | 4 | 0.1306 |
| 4 | cuhk:26530:0 | train | 0 | 1 | 0.0497 |
| validation | cuhk:17663:0 | validation | 4 | 0 | 无梯度诊断 |

训练 G/IRRA/backward/step 各 15 次，validation G/IRRA 各 1 次；另外单步推理 G 1 次、IRRA 0 次。没有训练文本前向、新评分或 JVP。

- 每轮 G 输入与 detach/即时更新参数/上一轮 Retain/原图联合重启核验通过。
- 15 轮 G 参数均改变，零 G 梯度轮为空。
- G 梯度 L2 0.4593905–4.5353989；IRRA 输入梯度 L2 0.2897703–0.7725985。
- 最大累计 L∞=0.03137254901960784=8/255，无预算超限。
- 最高预算饱和率 57.9888%；最高严格投影裁剪率 58.0458%。饱和分量按既有真实 clamp 阻断梯度，未引入 STE；本次没有整轮零梯度。
- IRRA 完整参数/buffers SHA exact，参数无梯度。
- GPU Checkpoint 恢复后的 G/Adam/CPU+CUDA RNG exact，next_epoch=2；没有执行第 2 个真实 Epoch。
- 独立推理 1 次 G，L∞=0.03131995350122452；validation L∞=0.03133860230445862。
- 训练峰值 allocated=1035.3745 MiB / reserved=1160 MiB；含 checkpoint/恢复/推理的已加载模型诊断段约 2.5182 秒，不含官方模型加载与缓存准备。
- Smoke Checkpoint 仅为本阶段恢复诊断证据，不是正式训练模型，不进入 Git。
- GPU 进程已退出，其他既有进程未修改。

完成态另在禁止 CUDA 初始化、模型加载、backward、JVP 的环境下重入小缓存准备与 Smoke，均 CPU-only 返回；54 份保护 SHA 仍保持。

## 证据与手动命令

服务器证据仅在新 outputs/phase24/：protected_sources.json、cpu_final.log、source_preflight_verified.log、prepare_small.log、conditions_small/、smoke.log、smoke/summary.json、smoke/epoch_run/ 与 reentry_verified.json。不提交输出、缓存、权重或日志。

本阶段已执行的 GPU 命令：

    PYTHONDONTWRITEBYTECODE=1 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python prepare_full.py --allow-small
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python smoke_phase24.py --allow-smoke

后续完整准备/正式训练的入口示例，**本阶段未执行，需后续用户批准任务范围**：

    python prepare_full.py --allow-full-preparation --output outputs/conditions_full
    python epoch_training.py --cache outputs/conditions_full --output outputs/train_v2 --epochs 10 --allow-training
    python epoch_training.py --cache outputs/conditions_full --output outputs/train_v2 --epochs 10 --allow-training --resume

正式运行须使用上述服务器 Python 环境及 cache 环境变量。--config 配置文件优先于 CLI epochs/seed；恢复须使用完全相同配置、数据与相关源码。不可用修改配置文件来延长一个既有契约的总 Epoch 数。

## 正式训练前待办

1. 等待独立源码审查，确认本阶段恢复和缓存协议。
2. 获得完整缓存准备授权后，准备原 2176 条并核验完整覆盖、K 分布、逐条 SHA/聚合与 train/validation PID 隔离。
3. 确认正式 Epoch 数、lambda/optimizer/seed/输出目录和同环境恢复计划；冻结配置后再启动正式训练。
4. 明确后续评测协议，单独授权 Gallery 评测。当前 paired validation 与 Smoke 不能证明正式攻击效果。

发布仅包含本阶段 8 个 Python 文件、README 与本中文报告。git diff --check、提交白名单及远端 SHA 核对通过后停止，不进入后续阶段。
