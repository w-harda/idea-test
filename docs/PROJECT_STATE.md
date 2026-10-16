# 项目当前状态（唯一权威入口）

状态来自Git、代码和不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，实验来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

**Stage 4A统一Evaluation Engine CPU开发验收完成。** 仅合成CPU评分/排名和历史元数据只读检查，无GPU、真实特征聚合、Full排名或攻击数据生成；未进入Stage 4B/5。

- 基准：`0d90e79c08bf313b9085e75531a4226b3b266698`。
- 独立分支：`codex/tbps-framework-stage4a`。
- 独立树：`/home/lzf/ldx/projects/idea-test-tbps-framework-stage4a-20261016`。
- 实现Commit：47ab2581ac1cd35e39365e870d2693ddce2b6bf1（本条由后续文档提交登记）。
- [API/CPU命令](framework/STAGE4A.md)、[源码映射/验收/Stage 4B预案](framework/STAGE4A_REPORT.md)、[ADR-0005](decisions/0005-cached-evaluation-native-ranking.md)。

## 本轮实现

- engines.metrics/provenance/evaluation/legacy：四模式、原生Adapter评分、Query分块、官方全PID正例排名、Source/InputBinding隔离、新Score/指标不可变发布及旧Joint元数据索引。
- 不zip配对；保留全部有序Gallery/Query、全部同PID正例。指标百分数，DR=(Clean-Adv)/Clean×100，Clean=0→null。
- 维度/原生结构/Tokenizer不固定；Fusion Mock及不同score方向验证通过。未知/未实现组合拒绝，不能将IRRA特征重标RaSa/APTM。
- 不调用或持有攻击策略/模型加载工厂；目标评分没有反馈到G/H的通道。原parse_config仍拒绝直接真实执行，仅更新其拒绝提示，不放宽权限。
- 新ScoreRecord绑定模型、评分协议/方向、有序成员、两侧InputBinding及score文件SHA；保存/读取clone，原子Manifest/回执与pending保护。
- 真实只读桥只有METADATA_VERIFIED_FEATURES_PENDING，require_replay仍拒绝。生产级model-free cached IRRA scorer/历史Full加载器未提供。

## CPU测试与保护

**383框架CPU tests（332原有+51新增）+36历史基础=419项通过。** 小型随机、多正例和tie矩阵与独立官方rank的排序索引及五项指标exact；官方rank源码SHA固定94910252a9e42f0820d49c11203b31d3b67951f58624f66abff4bf662500d5fb。未将历史Full指标硬编码成输出。

元数据检查独立从annotation恢复Test3074/6156顺序，对照Joint的Caption/PID/paired_index及原IRRA空间。3472项历史feature文件只是声明索引，未读features.pt、未核验其字节或排名。

7991份历史保护文件SHA/mtime不变；原实验树及Stage3B/3BR/3B2树原HEAD/clean保持，原FAIL和PASS Smoke/CPU修复回执及来源继续有效。原prototypes、Dataset、Checkpoint、特征、协议、日志和正式结果均未修改。

新CPU证据仅在 `/home/lzf/ldx/outputs/tbps-framework-stage4a/`，不提交Git。包含cpu_framework_final.log、cpu_legacy.log、legacy_metadata_index.json和protection_result.json。

## 重要历史证据缺口

历史G-v2/Joint Full没有逐张对抗RGB内容SHA，仅有原图身份、feature SHA、预算/来源回执。**不得用feature SHA冒充像素SHA，也不能从feature可靠恢复该SHA。** 当前新引擎的严格materialized输入契约不能直接接纳这种未知像素身份。

Stage 4B需用户批准单独的feature-only历史重放身份：原图SHA+冻结权重+原协议/回执+feature SHA，明确pixel SHA未知、仅同IRRA用途，不供目标模型重新编码。当前只设计不执行，不调用G补齐。

## 已完成与冻结历史

Stage1 61471db；Stage2 4e2a583；Stage2.5 b7aac69官方/Adapter IRRA最小等价通过；Stage3A ef835a6完成薄插件；Stage3B 0fb90cd原equivalence_v1保持FAIL；Stage3B-R a2988e0补齐来源闭包。

Stage3B-2发布0d90e79、实现4469617：固定两Train样本/G10/H7/FP32/math/no_tf32/RTX4090-G/CPU-H的输出等价VERIFIED。G4/H2/IRRA0，G像素和H完整动作exact，Joint共享组件通过。[报告](framework/STAGE3B2_REPORT.md)。不扩成全输入/硬件/模式已验证，插件保守标签仍保持。

G-v2完整训练/Full、H SCST/Epoch7 Selection、Joint Full、Joint消融/AP及8个历史资产索引全部冻结。正式Full攻击效果不由本阶段重新计算或改写。

## 下一步与权限

完成提交推送后停止等待审查，不自动Stage4B/5。Stage4B预案优先读取Joint紧凑四family，CPU2线程、RSS≤2GiB、≤30分钟；feature exact、相同score排名索引exact、score最大误差≤1e-6、五指标及DR对对应summary绝对误差≤1e-5个百分点。执行前需明确授权历史重放身份及预算，超阈值停止，不调排序/容差。

任何GPU、Full排名、特征加载、重新生成、训练或跨模型任务均须另行授权。当前没有这些任务的可执行命令或后台任务。

本轮不可变CPU验收：/home/lzf/ldx/outputs/tbps-framework-stage4a/cpu_verified_v1/result/；Manifest SHA c489c1accc8ea08097dcf60229ff0db84d45cab799b4c5cc0e967eced80e6a95，保护记录SHA 162735d019ce739f9dc669b2376a420c66c4f94fd3f3332df18143bc7a2ac82e。仅CPU证据，不是Full重放完成回执。
