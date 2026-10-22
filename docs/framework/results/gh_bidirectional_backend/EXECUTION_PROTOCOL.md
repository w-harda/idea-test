# 真实数值探针与A1/B两轮Smoke执行协议（本轮均未执行）

## 授权顺序

1. 本轮仅完成CPU代码、恢复、来源和资源快照。
2. 用户另行授权后，才能执行一次固定数值探针。
3. 探针PASS并经审查后，再另行授权A1/B两轮Smoke。
4. 不修改配置中的gpu_execution_default=false或继承协议的gpu_execution_authorized=false。
   这两个字段不是运行授权；显式命令参数和绑定当前SHA的探针回执共同构成门禁。
5. 不启动60轮联合训练，不运行Validation/Test或任何RaSa/APTM实验。

## 固定数值探针

- cuhk:26530:0：资格固定NOOP，原K0。
- cuhk:26747:0：预先固定的合法H候选，原G K1。
- 父模型：G60/H10；IRRA与reader来源见配置。
- H10从原Caption生成一次greedy动作；若第二条实际NOOP，停止并报告，不能换样本。
- NOOP条件必须内容exact，新增文本编码/JVP为0。
- 非NOOP重新编码/定位/JVP/FD；保存真实condition.pt及完整动作，不重新标记旧向量。
- 与原缓存整句特征的独立原文编码对照max_abs≤1e-6。
- 原第一G属性步Loss梯度：原条件/新条件各一次；NOOP原条件/复用条件各一次。
- NOOP梯度max_abs≤1e-6；非NOOP梯度max_abs>0。只验证数值变化，不判攻击性能。

| 项目 | 上限 |
| --- | --- |
| G图像样本前向 | 4 |
| IRRA图像样本前向 | 4 |
| IRRA文本样本前向 | 6：原文对照1、新语境TextTransport初始化2、JVP/FD3 |
| reader调用 | 1 |
| H整策略样本前向 | 1，CPU greedy |
| 属性JVP | 1，直接使用原TextTransport.read |
| Optimizer.step | 0，全局禁止包括Adam/SGD等 |
| 新图库生成 | 0 |

G参数梯度用autograd.grad取得，不调用原包含Adam更新的完整训练函数。
JVP内部自动求导不等于模型参数更新；原IRRA/reader参数始终冻结。
G/H权重、buffers、双Adam前后内容一致。IRRA/reader核对Tensor版本及完整state摘要。
GPU显存allocated/reserved、进程Peak RSS、耗时、真实计数和失败Trace均保留。
当前没有真实显存/耗时结果；估计数十秒到数分钟，不能保证共享GPU下一时刻仍有足够资源。

成功标志：GH_NATIVE_NUMERIC_PROBE_PASS。
输出：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/probe_v1/。
完成回执：completed/manifest.json、completed/completed.json。
结果：completed/report.json；动作rollout_*；实际条件condition_*。
失败：FAIL及已有Claim/pending保留，不重新使用probe_v1。

## CPU命令（可执行，不启动GPU）

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/home/lzf/ldx/tmp/gh_bidirectional_backend_20261022
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/src
export CUDA_VISIBLE_DEVICES=""
/home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.alternating.native_cli \
  --config /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/configs/framework/gh_bidirectional_backend.server.json \
  --action preflight --check-parent-states
~~~

PASS：GH_NATIVE_CPU_PREFLIGHT_PASS；
parent_restore为GH_NATIVE_PARENT_CPU_RESTORE_PASS；
imports为GH_NATIVE_IMPORT_CLOSURE_PASS；
startup_events应只有cpu_sources_verified、cpu_isolation_exited，不能有cuda_initialization_call。
CPU恢复仅保留CUDA RNG字节，不能把它当成CUDA恢复测试PASS。

追加--action resources替代preflight可进行一次只读nvidia-smi/RAM/磁盘/inode快照；不轮询、不抢占。
运行probe/smoke自身还会在CPU门禁通过后重新检查即时资源。

## 探针命令（仅供未来明确授权后执行；本轮未运行）

在上述正确工作目录、PYTHONPATH和TMPDIR环境中：

~~~bash
export CUDA_VISIBLE_DEVICES=0
/home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.alternating.native_cli \
  --config /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/configs/framework/gh_bidirectional_backend.server.json \
  --action probe --allow-gpu-probe \
  > /home/lzf/ldx/tmp/gh_bidirectional_backend_20261022/probe_execution.log 2>&1
~~~

没有授权参数时在CUDA、真实模型加载与Attempt创建之前拒绝。
必须等该探针结果被审查，不自动接着运行Smoke。

## 两轮Smoke命令（探针PASS且用户另行授权后）

~~~bash
export CUDA_VISIBLE_DEVICES=0
/home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.alternating.native_cli \
  --config /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/configs/framework/gh_bidirectional_backend.server.json \
  --action smoke --allow-gpu-smoke \
  > /home/lzf/ldx/tmp/gh_bidirectional_backend_20261022/two_round_execution.log 2>&1
~~~

同一配置/所有producer/父来源绑定的probe_v1完成回执缺失、被篡改、FAIL或超预算，均拒绝。
输出：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v1/。
该入口不会自动启动CPU审计或下一训练阶段。

G/H轮状态：rounds/A1-r1、A1-r2、B-r1、B-r2；各自永久保留对应Claim。
新图库：A1/bank_1、A1/bank_2、B/bank_1、B/bank_2，每个2048成员。
原生进度行GH_BANK_PROGRESS给出臂、round、已完成数和2048。
完整生成状态GH_NATIVE_TWO_ROUND_SMOKE_GENERATED，不等于独立审计或双向效果已证明。

## 独立CPU审计（未来Smoke结束后执行）

~~~bash
export CUDA_VISIBLE_DEVICES=""
/home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.alternating.native_cli \
  --config /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/configs/framework/gh_bidirectional_backend.server.json \
  --action audit-smoke \
  > /home/lzf/ldx/tmp/gh_bidirectional_backend_20261022/two_round_cpu_audit.log 2>&1
~~~

审计检查完整状态、真实计数、四组2048顺序、逐图SHA/预算、bank G版本、原父来源和不可变发布。
它读取已提交round，而不是调用恢复函数回滚到有后续Claim的旧round。
GH_NATIVE_TWO_ROUND_CPU_AUDIT_PASS仅表示产物审计通过。
feedback_effects另外记录首NOOP一致、首非NOOP梯度/更新差、同文本Reward及Advantage差；
bidirectional_update_evidence_verified必须按实际数值判断，不将零差异强行写成PASS。

## 资源与停止条件

新派生配置保留6144MiB GPU空闲、32GiB可用RAM、25GiB可用磁盘；
按四组完整bank的真实文件数要求至少80000可用inode。
GPU利用率只记录，共享进程不因存在本身被终止或抢占。
现有CPU原型原配置/原资源记录保持只读，新配置绑定新的输出和producer。

真实探针需重新测定峰值，当前仅估计2–6GiB显存，不能视为容量验证。
两轮Smoke约13.5GiB RGB纯数据、约14–16GiB完整新产物，暂估15–60分钟，实际以授权运行结果为准。
没有人为2GiB RSS硬上限，没有GPU轮询、自动降精度、改batch/分辨率或替换模型。

来源不一致、CPU隔离未退出、资源不足：在CUDA和Attempt前停止。
运行中的Token/Canonical映射、FD、非有限、预算、冻结或OOM异常：保留已发布动作/条件/Claim与FAIL，停止后续阶段。
不自动重试、不换固定样本、不删除pending、不自动从旧round回放、不假称CUDA bitwise轨迹等价。
