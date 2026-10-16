# 项目当前状态（唯一权威入口）

状态来自Git、实际代码和不可变证据；历史文档是阶段快照，不用于推断当前授权。架构见[ARCHITECTURE](ARCHITECTURE.md)，历史资产见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

Stage 1原则通过、Stage 2已审查；**Stage 2.5唯一一次IRRA真实GPU最小等价Smoke通过**，CPU回归与历史保护核验完成，等待独立审查。未进入Stage 3。

- 本轮基准：4e2a5838693c5bc04a22b31ab058daa69a987d50。
- 本轮分支：codex/tbps-framework-stage25。
- 复用Stage 2独立开发树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016。
- 本轮实现Commit：907fdcb82de1ee8abd6df101f8e8b253c129fb88（由本次后续文档提交登记）。
- [运行入口/命令与停止语义](framework/STAGE25.md)；[数值/审计报告](framework/STAGE25_REPORT.md)。

## 本轮已完成

- 新smoke_irra入口、冻结server配置及30项CPU护栏测试；Stage 2既有Adapter、配置和测试未改。
- 仅内部Train cuhk:9955:0、cuhk:29253:0的两张不同原图与对应原始Caption。实际成员经原2048/128来源核验；无Test/Validation前向、无G/H、无Optimizer。
- 同一个真实IRRA Backend加载一次。A独立官方预处理/Tokenizer/原生编码路径，B新Adapter；共同FP32/math/no_tf32，阈值预先固定1e-6。
- Token IDs、原始/归一化像素、[2,512]图像/文本特征及2×2 Score均逐值exact，最大误差0，特征范数均1.0。
- 总5图像样本前向、4文本样本前向，1次普通输入导数。输入梯度有限非零，L2=0.3470962643623352；模型全部eval/frozen，367个参数Tensor SHA前后相同，无registered buffers。
- GPU峰值allocated930.5205078125MiB/reserved1060MiB；受控run_once段12.00397365167737s。GPU只执行一次并已退出，无自动重试或恢复。

## 验证与证据

- 213项框架CPU tests通过：原183+新增30；GPU前后均通过。
- 36项历史基础CPU回归通过；合计249项不同测试，不代表全部prototype tests重跑。
- 2236项本轮相关来源SHA/mtime_ns不变；attempt/result不可变Manifest/完成回执CPU只读核验通过。
- 7991项历史保护文件SHA/mtime_ns不变；原实验树仍40c7476、Stage 1树仍61471db，均干净。
- 输出只在/home/lzf/ldx/outputs/tbps-framework-stage25/，不提交Git。
- result/report.json SHA：b27212f576573f13d5129fcb644effb0dcb535bf13bea259ed2e55a0d09897d9。

## 已完成框架与历史边界

Stage 1实现ffd18dd、发布61471db；Stage 2实现55b83bb48307bebba3e8a3bf8881570091ef1d15、发布4e2a583。原183框架tests和36基础tests通过。Stage 2官方Test 3074 Gallery/6156 Query有序身份与三份历史协议一致；G/H实际2048 Train/128内部Validation PID隔离，成员摘要f1a54c3caa57f2b0c9e1abf5f2a40b6282a1e1b002af383d12f94b370bfbcc66。

G-v2完整训练/Full、H正式SCST/Validation Epoch7 Selection、Joint Full及Joint消融均保持冻结；8个历史资产索引见注册表。Stage 2说明/报告保留当时的CPU验收快照，不用其“GPU未执行”覆盖本轮真实证据。

## 未验证与下一步

本次只验证固定硬件/FP32/2个Train样本及输入导数，不承诺所有输入/截断/batch/精度/设备的逐值一致。没有Full排名重算、G/H Attack Plugin、统一Evaluation Engine、RaSa/APTM或ICFG/RSTP Adapter；配置仍不能执行攻击或评测。

历史IRRA特征没有重写或自动重标为新ModelSpace；跨模型不兼容。Stage 3需要另行授权；本轮停止等待用户审查，不自行开始攻击插件、GPU扩大验证或Full评测。
