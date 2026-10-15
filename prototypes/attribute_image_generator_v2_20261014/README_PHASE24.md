# G-v2 Phase 2.4：Epoch 训练系统

基于 Phase 1–2.3 的独立新增实现；前阶段源码未修改，不导入 G-v1/integration。219 项 CPU tests 与固定 4 train + 1 validation 的单 Epoch GPU Smoke 通过。完整 2176 条缓存准备、正式训练和 Gallery 评测均未执行。

## 接口

- prepare_full.py：默认 CPU 审计原 2048+128 来源；--allow-small 仅固定五条，--allow-full-preparation 才允许全量。独立 outputs，逐条原子发布与恢复。
- training_data.py：新 gv2-training-cache-2 Manifest，沿用 Phase 1 张量 schema；完整身份、图像 SHA、原 caption/PID/role 核验。
- prep_text_frontend.py：仅离线调用冻结原 TextTransport.read/prepare_uncapped，不评分或改选择；保留原 z_t/S_raw/权重。
- epoch_training.run_epochs：每个 train 样本调用一次原 train_single_sample，K+1 更新；确定性 Epoch shuffle。validation 仅一次 no_grad G/IRRA 诊断，零 optimizer.step。
- epoch_checkpoint：SHA/fsync/原子发布不可变 Epoch，保存 G/Adam/配置/数据与来源契约/进度/RNG；仅 Epoch 边界恢复。
- inference_once.generate_once(g,image)：只输入图像，一次 G 前向和既有全局投影；不读文本/属性/JVP/IRRA。load_generator 从配有 SHA sidecar 的独立 Epoch Checkpoint 加载架构/G。
- smoke_phase24.py：默认 CPU 检查缓存，--allow-smoke 才执行固定单 Epoch 诊断。

z_t 始终仅为 Global Loss 监督，不作为生成器输入。算法、三个 Loss、相对原图 8/255 预算与饱和梯度行为均保持。

## 来源与恢复

缓存 producer 只锁定 CACHE_FILES 中明确的 7 个相关文件，不枚举目录。训练另锁定 TRAIN_FILES 的 11 个实际依赖文件、配置和缓存协议。新增无关文件不使缓存失效；相关源码、配置、数据变化拒绝恢复。

旧 G-v1 缓存/冻结来源在准备期只读审计，作为历史 provenance 保存；训练运行不读取 G-v1 缓存或模块。annotation/split 与每张源图 SHA 仍校验。

训练持有输出锁。每 Epoch 写入独立 attempt 日志，完成后发布 checkpoints/epoch_0001/ 与 current.json。完整版本不覆盖；最高发布版本损坏拒绝，不静默回退，pending 忽略，指针可恢复。中途失败从最后完整 Epoch 重做，**没有 Step 或样本中途无损恢复**。GPU 后续轨迹不承诺 bitwise 重复。

## 命令

在本目录，使用 /home/lzf/ldx/envs/tbps-image/bin/python；GPU 使用既有 /home/lzf/ldx/cache 下环境变量。完整命令和结果见[中文阶段报告](../../docs/stages/09_2_4_gv2_epoch_training_checkpoint.md)。

CPU（已执行）：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase24 test_phase23 test_phase22 test_phase21 test_conditions -v

少量验证（已执行；完成态重入只核验，不再计算）：

    python prepare_full.py --allow-small
    python smoke_phase24.py --allow-smoke

后续完整入口示例（本阶段未执行，等待后续任务授权）：

    python prepare_full.py --allow-full-preparation --output outputs/conditions_full
    python epoch_training.py --cache outputs/conditions_full --output outputs/train_v2 --epochs 10 --allow-training
    python epoch_training.py --cache outputs/conditions_full --output outputs/train_v2 --epochs 10 --allow-training --resume

默认配置模板：seed42，width32/4-block，Adam lr=2e-4/betas=(0.5,0.999)，lambda_global/local/retain=1，10 Epoch。可传 --config JSON；该文件优先于 --epochs/--seed，恢复时全部配置须保持。正式入口要求 scope=formal 与 full-2176 完成缓存；小缓存只能由受控 Smoke API 运行。

## 验证摘要

新增42+历史177=219项 CPU tests。连续2个合成 Epoch 与第1个 Epoch后恢复的 G/Adam/RNG、第二个 Epoch顺序和样本日志逐值一致；真实子进程退出、损坏拒绝、原子发布、并发锁及隔离均通过。

真实 Smoke：K7/1/3/0依序8/2/4/1步，合计15；validation K4零更新。全部G更新，IRRA参数/buffers exact；G梯度0.4593905–4.5353989，无零梯度轮。最大L∞=8/255，预算饱和率最高57.9888%，严格裁剪58.0458%；peak allocated1035.3745MiB。GPU边界恢复G/Adam/RNG exact；独立推理仅1次G。少量缓存4条复用、1个新JVP/2次内部autograd.grad、0次新评分。

所有证据位于本目录 outputs/phase24/，不进入Git。停止等待独立审查，不进入Phase2.5。
