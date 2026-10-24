# B正式入口运行协议

本轮实际只完成CPU开发和只读核验。Pilot资源不足未启动；以下GPU命令为后续资源满足后的操作入口，不代表已经运行。没有Watcher或自动长训练链。

## 固定位置

- 工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024
- 分支：codex/gh-formal-b
- 配置：/home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_formal.server.json
- Pilot：/home/lzf/ldx/outputs/tbps-framework-gh-formal-b/pilot_v1/
- 正式训练：/home/lzf/ldx/outputs/tbps-framework-gh-formal-b/train_v1/
- 原H60基线与父Checkpoint位置见BASELINE_AUDIT.md，全部只读。
- 配置及SHA索引见SOURCE_EVIDENCE.json；FORMAL_TRAINING_CONFIG.json是同字节文档副本。

使用operations入口，不能改成src模块的python -m调用：冻结legacy导入器会正确拒绝受管理src内的__main__身份冲突。入口不关闭该保护。

## CPU预检

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/src
export PYTHONDONTWRITEBYTECODE=1
/home/lzf/ldx/envs/tbps-image/bin/python -B operations/gh_formal_b.py --config /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_formal.server.json --action preflight --scope train
```

这会核验当前完整producer、原140份数学源码、496项运行来源、完整2048映射、父G/H双Adam与CPU RNG、历史151378文件和28旧FAIL/Watcher。不会初始化CUDA或创建训练Attempt。

一次性资源检查：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/gh_formal_b.py --config /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_formal.server.json --action resources --scope pilot
```

CPU PASS但资源不足时退出75，不是GPU Pilot失败。GPU利用率只记录；最低空闲10240MiB、RAM32GiB、Pilot磁盘25GiB、inode80000；正式训练磁盘250GiB、inode200万。不要覆盖配置或通过修改SHA门禁绕过。

## 独立B两轮Pilot

固定沿原八条样本，不更换样本、原K、Loss、Reward或预算。B单臂两Round，预计G更新54、H访问16/合法更新14、完整图库2份共4096张；JVP上限38。不同于原A1/B四Round Smoke，不复用其Attempt。

资源通过时执行一次：

```bash
mkdir -p /home/lzf/ldx/tmp/gh_formal_b_pilot_v1
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/run_gh_formal_b.sh pilot >> /home/lzf/ldx/tmp/gh_formal_b_pilot_v1/launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
```

脚本CPU/资源预检通过后才消费启动标记，持有flock；Python执行层再次验证来源/资源/唯一Attempt。GPU完成后自动运行新范围的独立CPU审计；不运行旧audit-smoke或改变旧exact FAIL。

Pilot真实审计成功才发布readiness/ready.json，状态READY_FOR_USER_AUTHORIZATION。该回执不会自动启动长训练。

## 正式长训练

**当前BLOCKED，不能现在执行：Pilot未运行/未审计，无READY回执；本轮不启动长训练。**

后续Pilot通过且获得长训练明确授权后，原入口：

```bash
mkdir -p /home/lzf/ldx/tmp/gh_formal_b_train_v1
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/run_gh_formal_b.sh train --allow-training >> /home/lzf/ldx/tmp/gh_formal_b_train_v1/launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
```

等价Python训练能力入口为operations/gh_formal_b.py --action train --allow-training。缺少匹配配置SHA、producer摘要和Pilot审计的READY，入口拒绝，不消费GPU训练身份。

拟定50 Round从G60/H10出发，G绝对Epoch61–110、H绝对Epoch11–60；不是重训旧G60/H60，也不是把Smoke状态续成正式状态。

## 进度与正常停止

```bash
cat /home/lzf/ldx/tmp/gh_formal_b_train_v1/status.json
cat /home/lzf/ldx/tmp/gh_formal_b_train_v1/supervisor.pid
cat /home/lzf/ldx/tmp/gh_formal_b_train_v1/process.txt
cat /home/lzf/ldx/tmp/gh_formal_b_train_v1/process_initial.json
tail -n 30 /home/lzf/ldx/tmp/gh_formal_b_train_v1/train.log
tail -n 20 /home/lzf/ldx/tmp/gh_formal_b_train_v1/status.log
tail -n 20 /home/lzf/ldx/tmp/gh_formal_b_train_v1/cpu_audit.log
```

阶段包含H_ROLLOUT、NEW_CONTEXT_JVP、G_TRAIN、FULL_TRAIN_BANK、H_SCST_TRAIN、ROUND_COMMITTED。每32训练样本更新阶段状态，图库逐图输出原生GH_BANK_PROGRESS；每Round输出GH_B_ROUND_COMPLETE、绝对游标、累计前向和资源。

需要正常停止时，仅对process_initial.json记录的本任务Python PID发送SIGTERM；它请求完整Round边界停止，不能kill其他用户进程。不要将监督器PID当成Python PID，也不要使用kill -9来假装可恢复。

显式Round边界恢复接口：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -u -B /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/gh_formal_b.py --config /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_formal.server.json --action train --allow-training --resume-round 1
```

这里只展示参数格式，不授权或自动恢复。必须先人工核验最近完整Round和后续没有Claim/IO/bank；不能随意填1回退。保持后台运行时应使用新的控制日志文件nohup+setsid调用Python恢复入口，原控制/FAIL保留。

## 统一IRRA正式比较（后续单独授权）

只有正式B训练及独立CPU审计完成后，CPU配置准备器才能冻结一个新评估配置，不改训练配置：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/gh_formal_prepare_eval.py --training-config /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_formal.server.json --output /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_comparison.evaluation_v1.json
```

它要求最终Round50及正式CPU审计PASS，绑定其FileRef、原训练配置和真实H60，产生新配置及新native来源文件。当前缺少正式训练产物，运行会拒绝，不生成Test协议。

获Test评估授权后分别调用同一入口，B默认只接受新评估配置预先绑定的最终Checkpoint：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -u -B /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/gh_formal_b.py --evaluate --config /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_comparison.evaluation_v1.json --method baseline --allow-test-evaluation
/home/lzf/ldx/envs/tbps-image/bin/python -u -B /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/operations/gh_formal_b.py --evaluate --config /home/lzf/ldx/projects/idea-test-tbps-gh-formal-b-20261024/configs/framework/gh_b_comparison.evaluation_v1.json --method B --allow-test-evaluation
```

新输出为/home/lzf/ldx/outputs/tbps-framework-gh-formal-b/unified_irra_evaluation_v1/baseline/和B/。两套攻击生成与评分串行运行，Clean/Image-only/Text-only/Joint共用各自固定攻击输入；同一IRRA/Manifest/Raw Caption/预算/Tokenizer/评分与Query chunk128。

results/completed保存metrics.csv、metrics_2dp.csv、summary.json；results/engine保存原Engine评分与协议，features保存紧凑特征与输入绑定。当前没有这些真实结果，不以旧G60/H7或Smoke数据填充。
