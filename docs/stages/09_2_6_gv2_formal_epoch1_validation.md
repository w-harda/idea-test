# G-v2 Phase 2.6：正式Epoch 1训练与恢复验证

2026-10-15：正式Epoch 1完成。2048 Train、6020次属性更新+2048次联合更新=8068次optimizer.step；128 Validation只诊断、零更新。Checkpoint的G/Adam/RNG恢复exact，关联样本及补充日志SHA复核通过。总目标仍为10，未启动真实Epoch 2～10或Gallery评测，停止等待独立审查。

## 工作树、配置与文件

仓库：w-harda/idea-test，分支：codex/gv2-per-attribute-cache。
本阶段基准：79fb440ee7506bc1a90d6539cd02358c2b59f78b。
服务器：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014。
实现目录：prototypes/attribute_image_generator_v2_20261014/。

| 文件 | 本阶段变更 |
|---|---|
| epoch_checkpoint.py | 最小补强恢复前关联日志SHA/内容/进度校验，正式Epoch强制提供样本日志 |
| phase26_train.py | 固定仅Epoch 1的受控正式入口，复用run_epochs(stop_after_epochs=1)，观测与恢复验证 |
| phase26_support.py | 状态摘要与逐轮/K/窗口统计，不含训练算法 |
| test_phase26.py | 19项新增CPU测试 |
| config_phase26.json | 正式不可变配置，总目标10，入口源码和缓存SHA绑定 |
| README_PHASE26.md | 入口、结果、Checkpoint与后续恢复边界 |

本中文报告在Windows docs/stages/09_2_6_gv2_formal_epoch1_validation.md编写，并同步服务器同一提交。Windows状态管理文档不发布。G-v1、官方IRRA、属性/JVP/选择、G架构、Loss、Retain和方案A实现没有修改；除明确授权的epoch_checkpoint.py外，67份保护文件SHA保持。

## 训练前检查与正式冻结

直接复用Phase 2.5完成的2176条缓存，不重新准备JVP：

- Manifest SHA：506caf30b8f82b8a29fc35e06df7bb04a5b2bcddc58a1f159b524bb107e7511e。
- Protocol SHA：9ad4903a6de6dc221df1a15af48d9e63e6332d7fcd3a9a62ecd3f11a2ed91c9d。
- 新旧缓存协议、全部tensor/metadata SHA、完整caption/PID/row_id/图像SHA/role与原始annotation逐条核验。
- 2048 Train/1861 PID与128 Validation/121 PID交集0。正式训练只读取Train。
- 原S_raw的34054 clean train评分参考图库仍包含内部Validation图；保留历史定义，不声称gallery-disjoint评分设置。
- 未运行新评分、Top-K或JVP，训练入口禁止torch.autograd.grad；图像训练仅使用合法Tensor.backward。

Phase 1～2.5全部240项CPU回归先通过；补强后240+19=259项CPU tests通过，CUDA未初始化。原连续/中断合成恢复测试继续通过，真实训练范围仍仅Epoch 1。

正式配置在运行前冻结为：

| 项目 | 固定值 |
|---|---|
| Generator | ImageOnlyGenerator(width=32, blocks=4)，1,377,411参数，image-only |
| Surrogate | 官方IRRA，fp32，eval/参数冻结 |
| Epoch总目标/本次上限 | 10 / 1 |
| Optimizer | Adam，lr=2e-4，betas=(0.5,0.999) |
| lambda_global/local/retain | 1 / 1 / 1 |
| Seed | 42 |
| epsilon | 8/255 |
| 数据 | 2048 Train，128 Validation仅诊断 |
| 算法 | 原方案A，每轮立即step，最后从原图联合更新 |

配置源文件与outputs/train_v2/frozen_config.json SHA完全相同：

    7e87bd272abfee440afc13109d439507a34b140506a19a132a8f740afe093bc4

环境：/home/lzf/ldx/envs/tbps-image/bin/python；Python3.10.21、Torch2.5.1+cu124、NumPy1.26.4；官方IRRA checkpoint SHA仍9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。无AMP、TF32、augmentation或超参数变更。

启动前正式outputs/train_v2不存在。GPU基线已使用20226 MiB，空闲4338 MiB，既有PID3507659占用20178 MiB；/home磁盘可用约6 TiB。没有停止或修改其他进程。

## 日志完整性补强

保持原schema、不可变Epoch版本、SHA sidecar、fsync/原子发布及current.json恢复机制。

verify_version在加载任何G/Adam/RNG前调用verify_associated_logs：

1. 正式Epoch必须同时具备sample_log_path与sample_log_sha256，禁止缺字段。
2. 日志路径必须留在G-v2 outputs；核验SHA并读取JSONL。
3. 核对日志与Checkpoint内sample_logs、样本数/顺序/位置/Epoch/train角色、K+1、轮数及总step。
4. 可选supplementary_log_sha256同样逐文件核验，本次绑定逐轮审计、训练前验证、Epoch 1验证与资源记录。

保存发布前也执行相同检查。缺失、损坏、内容与progress不匹配时拒绝，不加载状态，不静默回退最高损坏版本。旧通用无关联日志的合成快照仍兼容；正式快照不能省略日志。只能Epoch边界恢复，没有Step级无损恢复。

新增CPU tests覆盖篡改/删除/截断、SHA一致但内容错配、路径越界、顺序/数量错误、补充日志、正式缺日志拒绝、兼容性及固定10/1入口。对正式日志的独立副本再次做篡改和缺失测试，均拒绝；真实日志/Checkpoint未修改。

## 正式执行行为与计数

受控入口固定调用已有run_epochs(...,stop_after_epochs=1)，持久化config.epochs=10。本入口拒绝已有正式输出，不带自动resume选项，不能用它继续Epoch 2。

每个样本只调用一次原train_single_sample；逐属性G输入为上一轮detach图，下一轮使用立即更新后的参数，Retain只用紧邻上一轮JVP/stop-gradient分数。联合步重新输入原图；每轮全局预算相对原图。全部轮次检查通过。

| 项目 | 实际数量 |
|---|---:|
| Train样本 | 2048 |
| 属性轮 | 6020 |
| 原图联合轮 | 2048 |
| Train G/IRRA forward各自 | 8068 |
| backward/zero_grad/optimizer.step各自 | 8068 |
| Epoch 1 Validation G/IRRA forward各自 | 128 |
| Validation optimizer.step | 0 |
| 额外训练前Validation G forward | 128 |
| 额外训练前Validation IRRA forward | 256（128 clean+128初始G） |
| 新评分/JVP/Gallery评测 | 0 |

额外验证只作同一128条样本的clean/初始化G/训练后对照；无梯度，无参数或optimizer变化，不根据Validation选权重。初始诊断不改变RNG。

训练完成进度点为受控计时660.211秒；受控训练/诊断段总计699.414秒，含内层缓存复核、模型加载、训练前/后验证、Checkpoint保存/恢复及统计，不含外层CPU preflight和最终独立复核，不称为纯kernel耗时。GPU峰值allocated1033.164 MiB / reserved1196 MiB；其他进程活跃度变化保留在10秒资源采样中。

## Loss及梯度统计

下面是每轮等权平均；属性轮的Local为单属性JVP，联合轮为原S_raw权重聚合Local，两类均值不能当作同一目标直接优劣比较。

| 轮次 | 数量 | Global | Local | Retain | Total | G梯度L2均值 |
|---|---:|---:|---:|---:|---:|---:|
| 属性 | 6020 | 0.41296133 | 0.0000555900 | 0.0000884943 | 0.41310541 | 0.55966907 |
| 联合 | 2048 | 0.39484565 | 0.0001161897 | 0 | 0.39496184 | 0.64589916 |
| 全部 | 8068 | 0.40836280 | 0.0000709728 | 0.0000660307 | 0.40849980 | 0.58155792 |

有上一轮参考的3977个属性更新中，1879次Retain>0，eligible均值0.0001339542，最大0.006536603。第一属性轮、K0和联合轮遵循既定零Retain协议。Local/Retain的均值尺度小，不能由均值推断梯度贡献，更不能据此认定属性模块有效；没有调节权重。

| 联合步窗口 | Global | Local | Total | G梯度L2均值 | 预算饱和率均值 |
|---|---:|---:|---:|---:|---:|
| 前256样本 | 0.38573436 | 0.0003335750 | 0.38606793 | 0.85517979 | 9.8960% |
| 后256样本 | 0.38880811 | 0.0000008431 | 0.38880895 | 0.43045813 | 76.8237% |

训练窗口Global没有单调下降，前后窗口包含不同样本，不能视为相同图像的收敛对照。固定Validation的前后对照另见下表；只完成一个Epoch，不宣称收敛或最终检索性能。

所有Loss、梯度及图像有限。G梯度L2范围0.03504761–7.88458928；IRRA输入梯度L2范围0.14085796–21.51284407，说明冻结编码器保留输入梯度。零梯度轮次0，参数未变化轮次0。每轮数值变化的G参数元素数范围1,377,218–1,377,410，均值1,377,383.14；这是元素级比较，不只是参数版本号。

## 各K表现

下表Global/Local/Retain/Total、梯度和饱和率均按该K全部属性+联合轮平均；n较少的K6/K7不支持效果推断。

| K | 样本 | 更新 | Global | Local | Retain | Total | G梯度L2 | 预算饱和率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 5 | .36946203 | 0 | 0 | .36946203 | .87781063 | 54.0206% |
| 1 | 34 | 68 | .42580354 | -.0001387110 | 0 | .42566483 | .65424804 | 43.1728% |
| 2 | 639 | 1917 | .41748410 | .0000008604 | .0000571119 | .41754207 | .58356961 | 67.2105% |
| 3 | 842 | 3368 | .40551854 | .0001295370 | .0000671803 | .40571526 | .58605071 | 72.9592% |
| 4 | 463 | 2315 | .40260028 | -.0000024201 | .0000730123 | .40267087 | .57176869 | 76.7273% |
| 5 | 61 | 366 | .41465172 | .0004346621 | .0000734783 | .41515986 | .58811138 | 81.4261% |
| 6 | 3 | 21 | .45249314 | .0000391829 | .0000502594 | .45258258 | .32343840 | 82.1690% |
| 7 | 1 | 8 | .56014093 | -.0012745064 | .0000021258 | .55886854 | .61551777 | 86.8014% |

完整各K的min/max、属性/联合分项及严格裁剪率均在phase26_summary.json。无固定K=4 cap。

## 预算、投影与Validation

最大累计L∞=0.03137254901960784=8/255，无额外容差、无越界。

| 轮次 | 预算饱和率mean/max | 严格投影裁剪率mean/max |
|---|---:|---:|
| 属性 | 78.9069% / 99.1143% | 62.5241% / 99.3510% |
| 联合 | 54.9809% / 87.1840% | 2.5808% / 37.7943% |
| 全部 | 72.8335% / 99.1143% | 47.3079% / 99.3510% |

预算端点饱和与严格越界裁剪不是同一统计。高饱和分量按原真实clamp阻断梯度，未引入STE；本轮全局G梯度均非零。

固定128条内部Validation，以下Global是完整caption配对cosine；Local保持原JVP尺度，不是检索指标。

| 图像状态 | Global mean | 聚合Local mean | optimizer.step |
|---|---:|---:|---:|
| Clean | 0.4990569244 | 0.0005109839 | 0 |
| 初始化G单步 | 0.4969101984 | 0.0005186336 | 0 |
| Epoch 1 G单步 | 0.3991928893 | 0.0001416418 | 0 |

训练后Validation最大L∞=0.0313725471496582；G/Adam状态在Validation前后不变。IRRA全参数/buffers起止SHA exact、eval/参数冻结持续保持。没有Gallery、mAP/mDR或跨模型结果。

## Checkpoint与独立恢复复核

正式Checkpoint位置：

    /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0001/checkpoint.pt

同目录checkpoint.json及outputs/train_v2/current.json已原子发布。仅epoch_0001，无epoch_0002目录或attempt。

- Checkpoint SHA：876547ffeb827cbb4691bbf5b5d473da495590d15df2adba12c9e1415e52a27a。
- 样本日志SHA：c37611250443677a1d6fb1bd0238059ca72ddac4fb167a774524b88a348fa2b8。
- 逐轮审计SHA：419f489c3a856a2d493bac10dc15f3fa766161c05ecbd6f261b38cbdd233fcb5。
- G初始state摘要be3fc4d83a60b85c598c300274899f628a64f2508ca16031c06c9a58f55f8ffa；最终9ba521cd9367198fa9bf1bb646f3c659bc0952ab97f6e690531f10b54a4fd832。

在GPU新建G/Adam，仅加载Epoch 1，不执行前向或step：G、完整Adam状态、Python/NumPy/Torch CPU+CUDA RNG摘要全部exact；所有Adam step=8068，状态有限。completed_epoch=1，next_epoch=2，config.epochs=10。next_epoch只是恢复位置，未执行Epoch 2。

另在禁CUDA的独立进程逐条核对2048样本顺序、缓存属性ID、K+1、6020/2048轮数及8068条补充审计与Checkpoint内原记录一致。正式日志副本篡改/缺失拒绝，真实日志和权重保持；所有样本和补充日志SHA复核通过。

仅支持Epoch边界恢复，不承诺Step级恢复或后续CUDA轨迹bitwise重复。GPU进程已退出，既有PID3507659保留。

## 命令、证据与后续边界

已执行CPU：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_conditions test_phase21 test_phase22 test_phase23 test_phase24 test_phase25 test_phase26 -v

已执行正式单Epoch：

    cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python phase26_train.py --allow-epoch1

证据：outputs/phase26/下的CPU/preflight/正式log、authorized_checkpoint_change.json、protected_sources.json与independent_epoch1_audit.json；outputs/train_v2/下的frozen_config.json、training_protocol.json、execution_limit.json、epoch_attempts/、round_audit.jsonl、Validation与资源日志、Checkpoint、phase26_summary.json和phase26_completed.json。

训练/恢复过程中没有非有限值、预算越界、来源SHA错误、Checkpoint错误或OOM，没有重启正式训练、调整超参数或继续Epoch 2。关联日志副本的破坏测试属于独立负向验证，没有破坏正式资产。

后续如获授权继续Epoch 2～10，须保持同一配置、缓存、环境与相关源码，并保留全部关联日志，不能只复制checkpoint.pt。可使用原epoch_training.py的--resume和--config config_phase26.json；本阶段未运行该命令。先等待独立审查，不进入下一阶段。

仅发布本阶段源码/配置/tests/中文文档。数据、缓存、权重、Checkpoint、outputs及临时文件不提交；发布前核对diff --check、文件白名单、Windows/服务器报告SHA及远端SHA。
