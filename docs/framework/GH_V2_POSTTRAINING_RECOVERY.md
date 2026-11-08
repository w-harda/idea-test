# GH-v2原60轮的后处理恢复

独立操作工具位于operations/gh_v2_recovery*.py，版本绑定configs/framework/gh_v2_posttraining_recovery_20261108.json。它们在原冻结158份生产源码集合之外，原配置SHA不变；每个子进程启动前验证全部恢复工具SHA及原冻结来源。配置明确记录新审计输出路由，不静默关闭来源检查。

原训练输出、60份账本、六对权重和旧失败现场只读。新控制目录/home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108；后处理输出在原输出posttraining_recovery_20261108，正式Test在预留的test/round_10至round_60，每组禁止重生成/重试。原始Smoke保留，不作为正式结果。

## 单次启动

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v2-20261107
nohup setsid /home/lzf/ldx/envs/tbps-image/bin/python -u -B operations/gh_v2_recovery_v2_supervisor.py --allow-posttraining-recovery > /home/lzf/ldx/tmp/gh_v2_recovery_launcher_20261108_v2.log 2>&1 < /dev/null &
```

身份一旦消费，不删除started.json重启。状态见新控制目录status.json、child.json和每阶段日志。监督器阶段只有审计/Test/报告，没有训练入口；真实失败停止并保留FAIL.json，不自动重试。所有资源观察只读，共享GPU无VRAM floor，不操作外部任务。

## 读取与审计

独立分项JSON读取器先经ArtifactStore原验证，按项回调G/H/conditions，保留其余有界字段，完整消费JSON直到EOF，拒绝重复键、NaN/Infinity/浮点溢出、无效UTF-8、缺失闭合/尾随/非法分隔及类型。所有项都解析，原文件不拆写、不重存。每轮处理前后SHA检查及最后全部43907历史文件SHA/stat复核；所有恢复工具版本随回执记录。

CPU门禁禁止NN前向、CUDA、反向、autograd.grad和Optimizer.step，仅允许白名单CPU反序列化及纯规则/tokenizer操作。第1–59轮退休Tensor以原封存账本/退休Manifest审计，限制在报告显式声明。第60轮保留资产及六对权重直接读回。无训练JVP/Bank重算。

评测复用原fresh_evaluation完整Joint-only生成/评分数学。独立preflight只构建架构并严格载入当前端点权重，不创建训练初始状态、Adam或阶段RNG。内存中仅将训练审计证明读取路径改为新审计输出；原配置文件、配置SHA和生产来源不改。原CPU Test审计独立重读预算/Unicode/分数并使用原生ranking复算指标。

结果仅全部六组完成后写入独立docs/framework/results/gh_alternating_v2_recovery；表格/R@1/mAP曲线/来源说明自动commit/push并核对远端SHA。数据、权重、Ledger、缓存和原始日志不上传Git。

## 第二版摘要域兼容

第一版恢复工具的首轮逐步H检查混用了两个不同摘要函数，13秒停止；第一版控制目录和FAIL保留。当前入口为operations/gh_v2_recovery_v2_supervisor.py，版本configs/framework/gh_v2_posttraining_recovery_20261108_v2.json，控制目录后缀_v2，训练CPU审计输出train_cpu_audit_v2。

原H逐步日志使用冻结feedback_backend.state_sha，边界使用tree_sha，分别核验连续链；在原初始状态、六组Checkpoint及最终完整状态重算双摘要验证桥接。仅操作审计兼容，不修改或重新计算训练。第一版7个控制文件另有SHA/stat保护，每阶段核验。当前第二版工具仍是原训练Attempt的后处理，不是新训练Attempt。
