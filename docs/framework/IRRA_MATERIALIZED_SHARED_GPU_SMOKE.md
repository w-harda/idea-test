# IRRA物化评估共享GPU修订与10+10真实Smoke

本轮在现有独立工作树 `/home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017`、分支 `codex/tbps-irra-materialized-evaluation` 上开发，基准 `01dd2ac199b6357473959e63012ef2e265dbd325`。仅执行用户授权的一次10 Gallery + 10 Query真实Smoke；正式Full未启动。

实现Commit：`a6875e53c7581cf747e02eda1882259b7df7de6f`。代码、配置、测试及本报告同批提交；本后续文档提交登记其完整SHA。

## 最小修改

- `materialized_irra.py` 使用本入口自己的 `shared_observe_only` GPU状态查询，移除独占、8192MiB和启动利用率拒绝条件。记录GPU名称、总量/空闲/已用显存、利用率和计算进程，不修改或终止Ollama，不增加进程身份绑定或后台监控。
- 实际CUDA OOM与其他异常保留完整stderr堆栈及不可变execution报告，记录异常类型、OOM标志、真实加载/前向计数和显存峰值，不自动重试。
- 独立配置只改变资源政策及Runner源码SHA。配置文件SHA：`2c1623150b2598eacf326abdc2db331ec79f872f0c7a5d9121ee5aaf9cc99fe2`。
- 没有修改冻结Stage4D1、G/H算法、权重、Plan、IRRA架构/权重/预处理、FP32/math/no_tf32、batch1、输入数据、Ranking或指标定义。

运行前确认原评估输出目录不存在，没有Attempt。因此本次使用原 `smoke/`，仅创建首次 `attempt-000001`，没有新实验版本或GPU重试。

## CPU测试

必要回归102项通过，43.42秒。范围为物化评估/缓存、Stage4A Engine及Stage2 Model Adapter。新增3项验证已有Ollama可共存、低空闲显存仅记录、模拟CUDA OOM保留堆栈且只尝试加载一次；CPU测试不初始化真实CUDA。

日志：`/home/lzf/ldx/tmp/irra_materialized_shared_cpu_tests.log`；SHA `5f1986ff249b81ddd7194b98d0495da3fa3f37fd1bdc2326c6e78074efbf2097`。

## 真实Smoke结果

执行退出码0，标志 `MATERIALIZED_IRRA_EVALUATION_PASS smoke`。

固定官方顺序前10 Gallery（cuhk:37132:0至cuhk:37141:0），前10合法Query（cuhk:37132:0至cuhk:37136:1）；全部保留原PID多正例关系。Gallery有4个PID，Query覆盖其中2个PID。实际身份、原图SHA与Caption见不可变protocol，未按攻击效果选择样本。

| Group | R@1 | R@5 | R@10 | mAP | mINP | DR_mAP(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| Image-only | 90.0 | 100.0 | 100.0 | 93.0 | 92.5 | 7.0 |
| Text-only | 100.0 | 100.0 | 100.0 | 99.16666412353516 | 97.5 | 0.8333358764648438 |
| Joint | 90.0 | 100.0 | 100.0 | 90.95238494873047 | 88.0 | 9.047615051269531 |

指标单位百分数；DR=(Clean-Group)/Clean×100。Joint−Image-only mAP=-2.0476150512695312个百分点。上述小样本指标仅证明链路执行，不能替代Full结果或判断正式攻击效果。

IRRA加载尝试/成功均1次；图像样本started/completed=20/20，文本样本20/20，四模式CPU评分调用4次。G/H加载/前向、Backward、Optimizer、JVP均0。四类缓存各1个紧凑批次，每批10行512维FP32；原物化RGB/Unicode只读消费。

IRRA 367个参数Tensor运行前后SHA均为 `b0f0c4e01e6601eb280698f8ee5a3304d9ed117e22968db9f7be17f453330d0f`；buffers数量0，摘要exact。模型始终eval、requires_grad=False、parameter_grads_none=True。官方Checkpoint SHA仍为 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。原生加载器的“Training Model with...”为构造日志，本次Optimizer/Backward为0。

## 实际资源与Full可行性

- 运行前RTX4090总显存24564MiB，空闲3865MiB、利用率0%；Ollama PID2436556占用20178MiB。
- PyTorch峰值allocated=967184384字节（922.37890625MiB），reserved=1050673152字节（1002MiB）。
- 执行报告时CUDA上下文仍存在，评估进程剩余478MiB；进程退出后空闲恢复3865MiB，仅保留原Ollama进程。
- Peak RSS=2403823616字节（约2.239GiB）。
- execution计时36.31761821359396秒；外层完整进程计时121.65875400602818秒，后者包含CPU来源预检。
- 未出现OOM、NaN、异常或重试；完整stdout/stderr日志保留。

相同batch1流式编码下，Full增加样本数量，不需要同时把所有样本留在GPU。当前3865MiB空闲结合本次约1002MiB reserved峰值，支持按既定实现尝试Full；这不是显存保证，CUDA上下文及其他进程的后续占用变化仍可能触发正常OOM。未修改batch/精度，也未运行Full来探测资源。

Full原入口与命令见 `IRRA_MATERIALIZED_EVALUATION.md`，仍需用户自行启动。首次Full预期6148图像样本＋12312文本样本前向，紧凑特征约36.05MiB、四组FP32 Score约288.75MiB纯数据。没有自动后台任务、资源轮询、恢复或Full启动。

## 证据与保护

输出根：
`/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/smoke/`

- `execution-000001/report.json` SHA：`6a4752556488ec311f9856e1d07311cc287430ef811f792110a9e550fca1a84d`。
- `completed/summary.json` SHA：`d9499227e2f785b96af640f8b9bf6ec88285b0c5aac6f2664a5c68c827067f28`。
- `completed/metrics.csv` SHA：`9fafcc15329dbc3c09a0db10e7550a2f027b7ca9301359a72eb66d7c0df23c7b`。
- `completed/manifest.json` SHA：`0cf3c7863460290f39157e76c821fef2380c66703ba7b0d980471e4079f6d639`。
- 完整日志 `/home/lzf/ldx/tmp/irra_materialized_smoke_shared.log` SHA：`2dc695edc523997f79f6e39147216cac1ad2705200500a6cc67aa1e25e290396`。

沿用原开发保护快照，26个Stage4D生成/审计文件SHA、size、mtime不变；61555个物化文件成员、size、mtime不变。冻结Stage4D1及generation_v3生产树HEAD与clean状态不变；没有额外重跑历史Full审计。新Smoke不是正式Full独立审计，未运行额外CPU audit CLI。

本次只提交独立Runner、配置、必要测试和文档；日志、缓存及结果不提交。正式Full保持PENDING，不接入RaSa/APTM。
