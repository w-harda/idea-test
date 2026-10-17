# 项目当前状态（唯一权威入口）

状态来自Git、实际代码和不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，实验来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

**Stage4C-1真实RGB/Unicode物化基础设施CPU验收完成。** G原图输入READY（3074/3074 SHA）；历史H文本TEXT_SOURCE_VERIFIED（6156/6156）；真实G RGB及新文本单元未全量物化。没有真实模型加载/前向、GPU、JVP、训练或排名。

- 基准：c09754e6afc8cf53a6c0c6e8715b6d4eace80946。
- 独立分支：codex/tbps-framework-stage4c1。
- 独立树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c1-20261017。
- 实现Commit：256140a7ac5794174c57eb3ca2e4cfbebc50a05c（由后续文档提交登记）。
- [接口/CPU命令](framework/STAGE4C1.md)、[来源/存储/成本/下一阶段方案](framework/STAGE4C1_REPORT.md)、[ADR-0007](decisions/0007-model-independent-materialized-attacks.md)。

## 本轮结果与边界

G-v2真实Test推理是G(I)，无需Test条件/JVP；3074原图及冻结G10来源就绪，必需条件数0/缺口0。2176缓存均official train训练监督，不供Test生成输入。旧G-v1的3074 Test条件及6148数据/metadata文件SHA核验通过，仅作为历史索引，未混用于G-v2，也未反序列化。

历史Joint Full的6156条H7文本/动作及1155个metadata/intent/receipt通过原Canonical/Span/Unicode/Tokenizer校验：6116有编辑、40 NOOP、13537字符替换。可显式复用，不再运行H，不读取features.pt；本轮没有发布新的真实文本单元。

新materialization命名空间复用G/H薄插件与ArtifactStore，保存FP32原/Adv/delta和独立文件/像素SHA，保存真实UTF-8及完整动作；Joint只组合引用。可恢复入口先发布Claim，前向后未发布的中断拒绝自动重复。缓存计划摘要只计算一次，来源首/尾完整SHA及会话stat变更拒绝保持。已有源码/算法/模型/Checkpoint/缓存/正式结果不修改。

**478框架+36历史基础=514 CPU tests通过，43新增。** 真实源检查61.10218879207969秒/peakRSS566.703125MiB，模型加载及G/H/IRRA前向、JVP、Feature Tensor载入、排名、真实单元发布全部0。早期CPU源检查重复Manifest摘要导致性能问题，安全中断自有CPU进程后修复，日志保留；没有GPU或真实生成失败重试。

7991旧保护文件及3472Joint特征文件SHA/mtime不变，后者完整快照摘要与Stage4B1原记录exact；原Stage4B1/4B2完成回执/绑定来源有效。原实验发布树及4B1/4B2树保持原HEAD/clean。

## 不可变CPU来源证据

/home/lzf/ldx/outputs/tbps-framework-stage4c1/source_audit_v2/result/：

- report SHA37d244e6ec250a5d536e4a4e06832005a9e896233cc9b908f88e9b83365bf6e7。
- Manifest SHA484605d0501d94dbb4a21315ba0f61c76499e17d283fd2a51c6eb7a1d96c9ce7，10570来源FileRef。
- plan.json是准备协议，不是已经生成3074 RGB或6156新文本单元的完成证明。
- ordered_text_action_digest=afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868。
- CPU日志cpu_framework_verified.log/cpu_legacy.log及protection_result.json；前置Claim模块另由最终CPU回执绑定。

## 下一步（须另行授权）

Stage4C-2最小建议：Test Gallery cuhk:37132:0/cuhk:37147:0，Query cuhk:37132:0/cuhk:37147:1，覆盖编辑与固定NOOP；G≤2、H0（复用）、IRRA image≤4/text≤4。只验物化读回、原生编码/2×2 score与冻结，不做Full排名。GPU/资源/阈值和一次Attempt需用户另行批准。

全量三份FP32 RGB纯数据5.0657958984375GiB，建议磁盘8–10GiB、流式CPU≤2GiB；G-only显存建议空闲≥1GiB。10–60分钟只是未实测工程估算，SHA/fsync/慢磁盘可能主导，须小样本后修订。本轮没有GPU或全量命令。

RaSa/APTM将读取相同冻结RGB/Unicode，自身原生预处理/Tokenizer/评分并隔离ModelSpace缓存，不能回传反馈改样本或复用IRRA Embedding。真实Adapter、跨主机迁移及统一GPU Runner尚未实现。

## 冻结历史

Stage1 61471db；Stage2 4e2a583；Stage2.5 b7aac69；Stage3A ef835a6；Stage3B 0fb90cd保持FAIL；Stage3B-R a2988e0；Stage3B-2 0d90e79固定G/H最小等价通过；Stage4A 32f5582；Stage4B1 a4c9387完成FEATURES_VERIFIED；Stage4B2 c09754e（实现26f70a4）唯一CPU重放METRICS_REPLAY_VERIFIED。

历史IRRA特征的对抗像素SHA仍UNKNOWN，不被本轮Fixture升级成真实RGB证明。G/H/Joint/消融/AP及8个旧资产索引全部冻结。提交推送后停止，不自动Stage4C-2、GPU、Full、Stage5或训练。

最终CPU验收：/home/lzf/ldx/outputs/tbps-framework-stage4c1/cpu_verified_v1/result/；
Manifest SHA0f96f115d5a8d1b3c81a9e067b5c524041f4362c84f85483bdb3977300bf01da；
report SHAf10fbc4276e80048b59316b15ba8018824ac8a03cc2b5e34de0e2322714a10e5。
绑定包括generation.py在内的全部7个新模块、43项新测试源码、CPU日志、真实只读源检查与保护证据；不是实际攻击物化完成回执。
