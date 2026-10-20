# G-v2 Epoch11→60：独立扩展协议与续训验证

本轮仅授权CPU验证和一次最小真实GPU Smoke；正式训练、后台任务和三模型Full均未启动。
父模型只读，原epochs=10、contract、Checkpoint、current.json及历史结果不修改。

## 调用关系

- gv2_epoch_state.py：父SHA/contract、G/Adam/RNG恢复、扩展schema、样本日志SHA、原子Checkpoint与边界恢复。
- gv2_epochs.py：默认CPU预检，调用原epoch_training.train_epoch / validate_epoch、
  原serial_training.train_single_sample；仅增加外部日志、计数及独立事务保存。
- 原G/Global/Local/Retain/8/255投影均直接复用；没有FiLM/TV/扰动惩罚/Scheduler。
- seed42与绝对Epoch编号调用原epoch_order；Epoch11首样本预先固定cuhk:211:0、K=4。
- 每正式Epoch2048 Train、6020属性+2048 Joint=8068次Adam；128 Validation零更新。
- 六个计划评估点10/20/30/40/50/60固定。历史E10只引用；新协议每个完整Epoch保存恢复状态。
  本轮未实现或执行三模型消融Full。

## 输出与恢复语义

正式路径：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train（本轮不得创建）。
Smoke路径：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/resume_smoke_v1。
Smoke探针只做首个训练样本K+1及一条内部Validation诊断；不代表完整Epoch11。
probe_state/state.pt使用独立schema，completed_epoch=null，resume_as_formal_allowed=false，
不能作为正式训练Checkpoint。正式训练未来仍从原E10/80680开始。

扩展contract包含完整原contract和digest、父SHA、原epochs10、新目标60、起点11、
独立输出、新代码和配置SHA；唯一有效训练配置差异为epochs。SHA检查全部保留。
G、Adam和全部RNG在模型构造/IRRA恢复后恢复，避免模型初始化消耗影响下一步随机状态。
每Epoch先Claim、写轻量样本日志，完成Train/Validation/缓存复核后原子发布不可变Checkpoint。
SIGINT/TERM请求在完整Epoch边界发布后安全退出；只有明确--resume才能继续。

未提交Claim、pending、最高Checkpoint损坏或关联日志改变均拒绝自动重跑/回退。
崩溃中断的未完成Epoch须保留现场并人工审查，不支持样本/Step级无损恢复。
连续与恢复的CPU合成轨迹逐值一致；真实GPU证明状态恢复exact，不承诺历史CUDA未来轨迹bitwise一致。

## 可复核命令

以下均在真实服务器独立工作树执行。Smoke本轮最多一次，已经执行后不得重跑。

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020/src
CUDA_VISIBLE_DEVICES= /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.ablations.gv2_epochs --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020/configs/framework/gv2_epoch60.server.json --action preflight
~~~

唯一Smoke入口（已用则拒绝，不允许恢复或重试）：

~~~bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.ablations.gv2_epochs --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020/configs/framework/gv2_epoch60.server.json --action smoke --allow-gpu-smoke
~~~

独立CPU审计（不可变发布，不重复覆盖已完成审计）：

~~~bash
CUDA_VISIBLE_DEVICES= /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.ablations.gv2_epochs --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020/configs/framework/gv2_epoch60.server.json --action smoke-audit
~~~

成功标志分别为GV2_EPOCH60_CPU_PREFLIGHT_PASS、GV2_EPOCH60_GPU_SMOKE_PASS、
GV2_EPOCH60_SMOKE_CPU_AUDIT_PASS。原始运行日志在/home/lzf/ldx/tmp/gv2_epoch60_validation/。

**下列为后续授权的正式命令模板，本轮没有执行，不可因文档存在视为已授权：**

~~~bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.ablations.gv2_epochs --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch60-ablation-20261020/configs/framework/gv2_epoch60.server.json --action train --allow-training
~~~

边界恢复需同配置同工作树、额外--resume；不修改任何冻结来源。正式后台监督器及Full评估不在本轮启动。
资源检查仅即时观察，不轮询、不终止其他任务；不足即停。GPU异常保留failure及Claim，不自动重试。

## CPU验证

tests/framework/test_gv2_epoch_extension.py使用合成父状态、原G小尺寸实例和模拟编码器，
覆盖完整Runner控制流、原serial调度、恢复、日志与原子提交；Mock设备运行不算真实GPU证据。
历史Phase2.1/2.2与ArtifactStore一起回归158项；真实父/日志/2176缓存另做CPU来源预检。
实际Smoke结果及证据索引见GV2_EPOCH_EXTENSION_VALIDATION_REPORT.md。
