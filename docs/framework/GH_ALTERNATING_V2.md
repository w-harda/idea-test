# GH-Alternating-v2 从零60轮与六节点Joint Test

## 版本与授权

独立分支 `codex/gh-alternating-v2`，基于 `b8c45ec`。原v1源码/配置/资产和所有Baseline保持；本次授权短Smoke与审计通过后自动60轮，再自动执行CUHK-PEDES × IRRA六节点Joint Test，不扩展其他模式、Victim或Dataset。真实失败停止保留，不自动重试。

## 数据流

G/H从本次seed42分别按原训练规则创建，双Adam为空、计数为0。没有历史G/H权重、动量、RNG或动态Bank恢复；Source IRRA、Reader、basis、Tokenizer和固定属性/clean资源仍使用已锁定版本。

每轮固定2048 Train Manifest顺序：冻结H生成攻击Caption → 非NOOP重算原Canonical锚点的新语境/reader定位/真实JVP → 原G K+1训练 → 最新G完整2048图库 → 原H sampled/greedy SCST → 提交完整边界。当前H与G分别沿上一轮状态前进，不重新初始化。

原G选择的S_raw/K/属性顺序不改，不重新评分。H按原预测重要性/Natural-K/动作规则，多词可分别编辑1字符；共享词不重复、每属性最多1编辑、STOP/NOOP合法，不强凑K。采用原H `word_at` 的Unicode词位定义，非BERT Subword；不安装v1 WordBudget1，其他Baseline预算不改。

| 参数 | 设置 |
|---|---|
| rounds | 60 |
| 每轮G/H遍数 | 1 / 1 |
| Train | 固定2048，无新增Validation |
| G Adam | lr2e-4，betas(.5,.999)，原默认eps/weight_decay |
| H Adam | lr1e-4，betas(.9,.999)，eps1e-8，weight_decay0 |
| Scheduler | G=None，H=None；原脚本均无Scheduler |
| seed | 42；独立G/H阶段RNG及H采样器 |
| 图像 | 原FP32 RGB，严格L∞≤8/255 |
| 文本 | 原Natural-K gamma3/beta1，每个合法原词位最多1字符 |
| 保存与Test节点 | 10、20、30、40、50、60 |
| Test | Joint only，3074 Gallery × 6156 Query，全部同PID正例 |
| 计算 | 原生FP32/math/no_tf32，图像/文本逐样本编码，Query chunk128 |

历史S_raw固定参考34054张官方Train图（含既有内部Validation），不把它误称为2048奖励图库；本次不使用128内部Validation执行训练/评价。

## 文件与入口

- 配置：`configs/framework/gh_alternating_v2.server.json`。
- 入口：`operations/gh_alternating_v2.py`。
- 单次监督器：`operations/gh_alternating_v2_supervisor.py`。
- 新模块：`fresh_sources/training/store/run/audit/evaluation/report.py`。
- 复用：NativeBackend原G/H/条件/JVP/图库函数、configured_training原调度循环、ArtifactStore、原IRRA Adapter/FeatureCache/EvaluationEngine。
- 没有重写G、H、SCST、Loss、Reward或模型编码器；v1文件保持。

## 运行与状态

工作树 `/home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v2-20261107`，Python `/home/lzf/ldx/envs/tbps-image/bin/python`。
输出 `/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107`：`smoke/`、`train/`、`test/round_10`至`round_60/`分开。
控制 `/home/lzf/ldx/tmp/gh_alternating_v2_supervisor_20261107`，训练阶段进度 `/home/lzf/ldx/tmp/gh_alternating_v2_train_20261107/status.json`。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v2-20261107
/home/lzf/ldx/envs/tbps-image/bin/python -B operations/gh_alternating_v2.py --config configs/framework/gh_alternating_v2.server.json --action preflight --scope train
nohup setsid /home/lzf/ldx/envs/tbps-image/bin/python -u -B operations/gh_alternating_v2_supervisor.py --allow-authorized-pipeline > /home/lzf/ldx/tmp/gh_alternating_v2_launcher.log 2>&1 < /dev/null &
```

监督顺序：1轮原8条Train Smoke（完整2048 Bank）→独立CPU审计→10 Gallery/10 Query Joint执行Smoke及独立审计→全新初始化正式60轮→训练审计→六组独立Joint生成/编码/排名及各CPU审计→生成表格/两曲线→自动Git报告发布并核对远端SHA。

Joint Smoke仅验证执行协议，不用指标门槛决定训练，不把它当正式结果。正式训练与Smoke分别重新初始化，绝不延续Smoke权重。

GPU利用率/显存只观察，无最低空闲显存准入；RAM/磁盘/inode检查保留。没有等待资源Watcher，不停止外部任务。实际OOM/来源/预算/语境或审计错误立即停止本任务。

## Checkpoint与恢复

`train/checkpoints/round_10/{G.pt,H.pt}`等六组不可变权重只用于正式评估；包含本次版本/轮次/配置/账本来源身份，不与历史G10/H10同名资产混用。

每轮完整双模型/双Adam/Scheduler(None)/分阶段Python、NumPy、Torch CPU/CUDA、H采样RNG及游标保存在最新 `train/boundaries/round_N/state.pt`，原子更新latest.json。先发布新完整边界及六节点导出，再按退休回执删除仅属于本次旧边界、旧Bank和旧条件IO；全部小型账本和六组模型永久保留。最新Bank完整保留且绑定当前G及2048顺序。

仅已提交完整边界、没有后续Claim/IO/pending/FAIL时支持显式恢复：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -u -B operations/gh_alternating_v2.py --config configs/framework/gh_alternating_v2.server.json --action train --allow-gpu --resume
```

正常SIGTERM只发给已核实的本任务训练Python PID，在完整Round边界退出3；Supervisor记录停止，不自动重放。恢复须使用独立控制日志，原日志保留。真实失败不静默恢复。CUDA未来轨迹不承诺bitwise一致；CPU连续/恢复状态有定向验证。

## 执行成本与结果

正式60轮预定484080次G Adam更新、122880次H访问、按固定资格122220次H更新，60份完整图库累计122880次图像生成；新JVP上限361200，实际由NOOP比例决定。旧Bank随完整边界退休，磁盘不会永久累积60份RGB图库。

六组Test在60轮及训练审计全部结束后进行，各组独立G/H产物、目标冻结、无Test反馈。原生Clean可只读参考，不额外重算。当前真实状态见 `results/gh_alternating_v2/REPORT.md`；未完成时不填正式指标或曲线。
