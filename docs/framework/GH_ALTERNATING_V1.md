# GH-Alternating-v1 运行说明

本版本复用 `a820278` 的 GH-B；原146份框架源码、G/H prototypes、IRRA及历史Checkpoint保持。新增文件仅实现预算支持集、可配置调度、Scheduler、诊断与执行/审计。

## 方法与预算

每轮：冻结当前H生成攻击文本 → 重算非NOOP语境/定位/JVP → 原G K+1训练 → 最新G完整刷新2048张Train Bank → 原SCST更新H → 保存完整Round。

G保留原S_raw/Canonical属性集合/K及顺序；不按H重新选择，不重评分。非NOOP重新计算整句监督、reader定位、真实JVP及原FD门禁；NOOP显式复用原条件。G推理仍为单次G(I)。

H保留原Byte Transformer、Natural-K gamma3/beta1及SCST。WordBudget1按原文连续非空白片段计数（等价原文split()的位置），不规范化文本；连字符、标点不切词。原每属性/共享词一次字符替换保留，因此新版本每条Caption最多1词且该词最多1字符替换。候选支持集在一次实际编辑后只允许STOP/NOOP；不将Natural-K改成1，不截断输出。超预算、非法Unicode、Span或回放错误直接拒绝。

适配在隔离legacy会话中有界替换候选函数，调用原rollout，finally恢复函数；网络Parameter对象与Adam动量不换。训练、H→G rollout和greedy共享该策略。旧H7/H60的多词协议不改签WordBudget1。

## 初始化和完整训练配置

`configs/framework/gh_alternating_v1.server.json` 是唯一新配置，另有pilot/train派生native来源配置。新父状态采用原G60/H10，完整恢复双Adam、各参数历史step、Python/NumPy/Torch CPU/CUDA及H采样RNG；不是从旧失败Smoke续训。

| 参数 | 配置 |
|---|---|
| rounds | 50（正式候选配置，本次不执行） |
| pilot_rounds | 2 |
| g_passes_per_round / h_passes_per_round | 1 / 1，可独立正整数 |
| G / H learning rate | 0.0002 / 0.0001 |
| Optimizer | 原G Adam(.5,.999) / 原H Adam(.9,.999)，完整动量/step恢复 |
| Scheduler | 独立StepLR状态；默认gamma=1保持常数，亦支持显式step方案 |
| 数据 | 完整训练固定2048 Train；Pilot沿原8条更新样本，Bank始终2048 |
| seed | 42 |
| 图像 / 文本 | 严格8/255 / 最多1原词位置 |
| checkpoint_interval_rounds | 1，支持正整数里程碑间隔；每个完整Round始终保存恢复状态 |
| 精度 | 原fp32/math/no_tf32；H策略CPU，G/IRRA GPU |

G/H每遍分别调用原排序函数；绝对Epoch分别从61/11前进，多pass推进各自绝对Epoch。Scheduler每个模块阶段结束后前进一次。配置LR显式覆盖父参数组LR但保留Adam动量/step；新Scheduler从本任务Round0开始，不能描述成旧训练Scheduler连续恢复。

50轮默认预计新增403400次G更新、102400次H访问；旧资格对应101850次H更新，实际按账本计数。50份完整Bank共102400张，三FP32 RGB纯数据约168.75GiB，另有状态/来源/目录。新增语境JVP上限301000。不是与固定G训练H等计算量对比。

## CPU与Pilot命令

工作树 `/home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v1-20261106`，Python `/home/lzf/ldx/envs/tbps-image/bin/python`。输出 `/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v1-20261106`，控制 `/home/lzf/ldx/tmp/gh_alternating_v1_pilot_20261106`。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v1-20261106
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -B operations/gh_alternating_v1.py --config configs/framework/gh_alternating_v1.server.json --action preflight --scope train
```

GPU只记录，不设置固定空闲显存阈值。RAM/磁盘/inode仍检查。真实OOM/预算/SHA/审计错误立即保留停止，不自动重试、不终止外部进程。

授权的唯一Pilot采用后台独立Supervisor：

```bash
nohup setsid /home/lzf/ldx/envs/tbps-image/bin/python -u -B operations/gh_alternating_supervisor.py --allow-gpu-pilot > /home/lzf/ldx/tmp/gh_alternating_v1_pilot_20261106_launcher.log 2>&1 < /dev/null &
```

监督器执行Round1→正常完整边界退出（exit3）→新进程显式恢复Round1→Round2→独立CPU审计。此恢复是预定验证，不是失败重试。启动标记永久保留；再次启动拒绝。审计PASS才发布READY，不自动启动训练。

## 后续正式训练与恢复（需另外明确授权）

```bash
nohup setsid /home/lzf/ldx/envs/tbps-image/bin/python -u -B operations/gh_alternating_supervisor.py --scope train --allow-training > /home/lzf/ldx/tmp/gh_alternating_v1_train_20261106_launcher.log 2>&1 < /dev/null &
```

仅完整Round边界可显式恢复，先核对来源、最近状态及后续没有Claim/IO/Bank/pending。恢复参数为 `--scope train --allow-training --resume-round N`，使用独立恢复控制目录与日志。真实FAIL不自动恢复；本版保守拒绝含FAIL的Attempt。SIGTERM只向已核实的本任务Python PID发送，完成当前Round后停止；监督器将停止作为保留边界的状态，不应自动重试。CUDA未来轨迹不承诺bitwise一致。

状态包含G/H、双Adam、双Scheduler、双阶段全RNG、累计游标、配置和Bank引用。每轮Bank另存生成它的G完整state.pt并绑定G SHA和2048有序成员。恢复先核验全部Bank文件再安装当前版本，不把初始G60 Bank作为恢复后的奖励。

`status.json`、`resource_stages.jsonl`记录阶段、CPU累计时间/峰值RSS、GPU利用率/显存；Round ledger记录各阶段耗时、Loss、四组合配对相似度、sampled/greedy奖励、梯度和新旧Bank同文本Reward差。相似度或奖励变化不等于正式mAP改善。

## 结果边界

本次真实状态见 `results/gh_alternating_v1/REPORT.md`。Pilot不属于官方Test、正式Full或严格A1/B因果对照。旧 `first NOOP control differs` 和旧v1失败均保留。本任务不运行正式长训练、Test评分、APTM或跨数据集新实验。
