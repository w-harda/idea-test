# 项目当前状态（唯一权威入口）

状态来自Git、实际代码和不可变证据；架构见[ARCHITECTURE](ARCHITECTURE.md)，实验来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。旧报告是历史快照，不替代本入口。

## 当前阶段

**Stage4C-2唯一真实Test小样本Attempt为FAIL，已停止。** 两图G10真实RGB/delta物化与两条历史H7文本复用通过；IRRA加载前被CPU隔离闸门拒绝，端到端特征/四组2×2 Score等价仍PENDING。不得标记Stage4C-2通过或自动重试。

- 基准：4d3bc025a7da593368ebd22cc05db1ebe787b446。
- 分支：codex/tbps-framework-stage4c2。
- 工作树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c2-20261017。
- 实现Commit由后续文档提交登记。
- [接口/命令与边界](framework/STAGE4C2.md)、[完整FAIL报告](framework/STAGE4C2_REPORT.md)、[ADR-0007](decisions/0007-model-independent-materialized-attacks.md)。

## 已确认结果

严格CPU预检通过：75项G/H来源、实际G/H与官方IRRA导入闭包、固定Test身份/顺序、C1/B2验收来源与编辑/NOOP。GPU前RTX4090无计算进程、free24051MiB。

G10恢复1次，固定Gallery cuhk:37132:0/index0、cuhk:37147:0/index15各1次G前向；原/Adv/effective_delta为FP32 [1,3,384,128]，生成与读回bitwise exact，L∞均0.0313725471496582≤8/255，参数/buffers摘要exact，GPU peak allocated42.3857421875MiB/reserved52MiB。

H7恢复/前向0。Query cuhk:37132:0历史2次Unicode编辑与cuhk:37147:1固定NOOP完整action/UTF-8读回exact；全部来源/动作合法，未用Test标签/奖励重新选字符。四个小样本单元未调用full seal_collection，不是完整攻击集合。

新增Runner在IRRA子进程的CPU读回之前调用torch.cuda.reset_peak_memory_stats()，使CUDA提前初始化；原native_action_validator→cpu_only正确报ArtifactError: isolated CPU process required。IRRA权重恢复/图像文本前向均0，未得到新IRRA特征或Score，不能给出等价误差0。Attempt35.17172169685364秒，所有进程RSS低于2GiB，无NaN/OOM/预算超限；唯一失败是启动时序。GPU进程已退出，没有重试。

## CPU与保护

最终501框架+36历史基础=**537 CPU tests通过**；23项新增包括此次启动顺序FAIL的合成复现，验证门禁拒绝且IRRA加载不发生。测试通过不表示真实IRRA端到端通过。

7991项历史保护文件及3472个Joint Full文件SHA/mtime不变；C1/B2回执和本次生成证明通过只读校验，无pending。原发布树、Stage3B FAIL树、Stage3B2 PASS树、Stage4B1/4B2/C1树HEAD/clean保持。G/H/IRRA、Tokenizer、Checkpoint、缓存、历史结果与原框架算法均未修改。

## 不可变证据

根：/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/。

- attempt/manifest.json SHAff0c62a387da4f0f9de11ceedf9c6bfae0ad10ca7eee28a09326cc1e4b762c1a。
- result/manifest.json SHAc1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b。
- stage-g PASS、stage-text PASS、stage-irra FAIL及g.log/text.log/irra.log；artifacts/含真实两图/两文，raw-delta-0/1保留原G输出。
- 后置CPU保护：/home/lzf/ldx/outputs/tbps-framework-stage4c2/post_cpu_protection_v1/result/。
- 最终CPU验收：/home/lzf/ldx/outputs/tbps-framework-stage4c2/cpu_verified_v1/result/。
- 既有C1/B2证据不覆盖、不重算；完整SHA和Tensor数据见阶段报告及真实回执。

## 下一步与权限

等待用户审查。建议另行授权CPU-only最小修复：在IRRA子进程完成所有CPU历史/动作读取后，才初始化CUDA、设置峰值统计并恢复IRRA；补齐启动时序Mock。IRRA检查点加载期2GiB预算尚未实测，应先分析，不能自行提高预算。

本次Runner和FAIL来源保持，修复需新受控方案；新GPU Attempt须重新授权，不能恢复/覆盖materialized_smoke_v1，不能重复已经发生的G生成而不明确计数。未批准全量物化、Stage5、RaSa/APTM、训练或Full评测。

## 既往阶段索引

- Stage4C-1：4d3bc025a7da593368ebd22cc05db1ebe787b446，[来源与存储验收报告](framework/STAGE4C1_REPORT.md)；真实G2推理无需Test条件缓存、6156条历史H文本TEXT_SOURCE_VERIFIED。
- Stage4B-2：c09754e6afc8cf53a6c0c6e8715b6d4eace80946，历史IRRA Full特征METRICS_REPLAY_VERIFIED；不表示真实对抗像素已物化。
- 更早模型、实验与来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)，历史架构与决策见ARCHITECTURE及docs/decisions；本轮均保持原证据。
