# 项目当前状态（唯一权威入口）

状态来自Git、代码和不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，实验来源见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

**Stage 4B-1历史IRRA Full特征只读桥验收完成。METADATA_VERIFIED、FEATURES_VERIFIED通过；METRICS_REPLAY_VERIFIED仍PENDING。** 未计算真实Full分数或排名，无模型加载/神经前向/GPU/攻击生成/训练；不自动进入Stage4B-2或Stage4C。

- 基准：32f55827c94c6512b81d32a9aa478b1d14c70b92。
- 独立分支：codex/tbps-framework-stage4b1。
- 独立树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage4b1-20261017。
- 实现Commit：23579855273f85f1ed3cf3ca9abad8260341bddd（由本次后续文档提交登记）。
- [接口/CPU命令](framework/STAGE4B1.md)、[验收报告](framework/STAGE4B1_REPORT.md)、[ADR-0006](decisions/0006-explicit-historical-feature-replay.md)。

## 本轮实现及验证

HistoricalReplayIdentity/FeatureBinding/Representation独立分型，adversarial_pixel_sha256固定UNKNOWN，严格materialized InputBinding/BoundRepresentation未放宽。只读JointFullSources/Loader校验原协议、Selection、固定G10/H7/IRRA SHA、原3074 Gallery/6156 Query完整顺序与全部声明文件。原生cached scorer接同一Evaluation Engine，真实score默认拒绝，当前CLI只元数据/特征检查。

真实CPU流式核验四family：clean_gallery49批/3074行、image_gallery49/3074、clean_text385/6156、adv_text385/6156；868批、3472文件SHA/回执/FP32/有限性/L2/身份均通过。全部6156条缓存H动作经原生Canonical/Span/Unicode/Tokenizer合法性核验；不恢复H或重新预测importance。载入即detach().clone，不保留完整表，没有使用旧G-v2大Storage Text/Clean缓存。

412框架（383原有+29新增）+36历史基础=448 CPU tests通过；小型合成历史路径与正常严格路径score、五指标及DR exact。真实scores_computed=0、model_loads=0、neural_forwards=0，指标重放状态PENDING。

最终特征检查段43.30987264215946秒、进程peak RSS512.78125MiB，Torch2.5.1+cu124/CPU2线程。不是Full排名资源数据。7991旧保护文件、3472Joint缓存文件SHA/mtime不变；原FAIL/PASS及Stage4A证据继续有效，旧实验树HEAD/clean保持。

证据：/home/lzf/ldx/outputs/tbps-framework-stage4b1/features_final/result/。
report SHA7d8dd8364df3537471658a0d3ce3c30f5e4609f700bb75a68e2f68e0fdc979bf；Manifest SHA5cd8dcf0e8ccbc61b58adc7dae37e26e51f361369e15f6fefa77c9465bfc1370；绑定6720个唯一来源FileRef。首次features_v1证据原样保留，最终版补强新回执producer/全部feature绑定后再次只读核验，未重写历史缓存。

## 重要边界与下一步

历史Full未保存逐图对抗RGB内容SHA，不能用feature SHA冒充，也不能从feature恢复像素身份。本桥只重放原IRRA特征，不是新的攻击结果，不供RaSa/APTM复用或编码。原engines.metrics/provenance、models.base、contracts及历史prototypes不变。

Stage4B-2需用户单独批准。预案固定CPU FP32/Torch2.5.1+cu124/2线程、Query128/完整3074Gallery与6156Query、RSS≤2GiB/≤1800秒；feature exact、score≤1e-6、官方/新Ranking索引exact、五指标及DR对原summary误差≤1e-5个百分点。固定配置execution_authorized=false，尚无本轮Full执行命令。失败停止，不改Ranking/阈值。

完成提交推送后停止等待审查，不自动GPU、真实Full评分/排名、重新生成、训练或Stage4C。

## 已完成与冻结历史

Stage1 61471db；Stage2 4e2a583；Stage2.5 b7aac69官方/Adapter IRRA最小等价通过；Stage3A ef835a6薄插件；Stage3B 0fb90cd原equivalence_v1保持FAIL；Stage3B-R a2988e0补齐来源闭包；Stage3B-2 0d90e79固定两Train样本G/H输出等价通过，不扩展到所有输入/硬件。

Stage4A发布32f5582、实现47ab258：统一四模式Evaluation Engine，419 CPU tests；只有metadata桥未读特征。[当时报告](framework/STAGE4A_REPORT.md)、[ADR-0005](decisions/0005-cached-evaluation-native-ranking.md)。此历史边界由本轮独立历史分型桥补充，不改当时证据。

G-v2完整训练/Full、H SCST/Epoch7 Selection、Joint Full、Joint消融/AP及8个历史资产索引全部冻结。正式Full攻击效果不由本轮重新计算或改写。当前代码Commit由下方发布登记补充。

本轮独立CPU验收回执：/home/lzf/ldx/outputs/tbps-framework-stage4b1/cpu_verified_v1/result/；
Manifest SHA80c3446cd9f149d54e7d3a403bfb5121f95b39b01498c8bfb851379e624e44ba，
report SHA4dc8a6d87c94830a246b84e0d99edb7ba69f2eb73653ada82ba713ad2331b8a8。
绑定本轮代码/配置/测试、测试日志、保护记录和最终特征验证回执；不包含新的真实分数或指标。
