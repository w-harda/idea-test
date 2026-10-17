# 项目当前状态（唯一权威入口）

状态来自Git、实际代码及不可变证据。历史报告为当时快照；架构见ARCHITECTURE，实验见EXPERIMENT_REGISTRY，设计边界见ADR。

## 当前阶段

**Stage4D-1完成：TEXT_FULL_MATERIALIZED，独立TEXT_FULL_CPU_AUDIT_PASS。** 历史H7的6156条CUHK官方Test文本已物化；Image Collection和Joint全量仍PENDING。本轮G/H/IRRA真实模型加载、神经前向、GPU、Backward、Optimizer、JVP、检索排名均为0。

- 基准a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2。
- 分支codex/tbps-framework-stage4d1。
- 工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017。
- 实现/证据Commit：7f50cd1922d2610703170d3a7e10d8c69ab5255a（本后续文档提交登记，代码与相关报告已共同提交）。
- [阶段报告](framework/STAGE4D1_REPORT.md)、[接口与实际CPU命令](framework/STAGE4D1.md)。

## 本轮真实产物与来源

统一Plan ID：7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。官方Test Gallery3074/Query6156完整原顺序；H文本复用原Joint Full，不调用H、不选择新字符、不用目标反馈。冻结G10/H7/Selection及C1/C3审计来源SHA。

唯一全量CPU Attempt退出0：6156/6156，6116编辑、40 NOOP、13537次替换。逐条原文/Adv UTF8、完整Span/Unicode动作/Tokenizer审计及旧来源读回通过。用时879.7607615217566秒，Peak RSS607948800 bytes，最终30797文件/89401019内容bytes；已分配blocks194134016 bytes，无pending、无重试。

根目录/home/lzf/ldx/outputs/tbps-framework-stage4d1/materialized_full_v1/：

- result/manifest.json SHA a149a57fc7e8c87a389ede80c7e2eaedafec9d3ffc81dff9a7275915040c615f。
- artifacts/text-full/text_collection.json SHA 9ab8881dab9f82bd31cc31c0a7cc478e3beca9b2f9b940443603efa54d21069b。
- protocol文件SHA ac77b8d0f8abb3918f7daa8194c8929ee3e504a29bd502b312ad812deb25a9f6。
- 真实像素没有在此全量Store生成，不能将历史IRRA特征当RGB或宣称Joint全量完成。

## CPU验收与历史保护

591 CPU tests通过（23新增+568既有，69.90秒）；新增审计保护代码的4项针对回归通过。严格GH_RUNTIME_PREFLIGHT_PASS、退出0，75来源/33旧执行文件、无模型反序列化/CUDA/神经前向。

独立全文审计退出0，110.86877850443125秒、Peak RSS607436800 bytes。6156条字符串/完整action/trace及官方顺序exact；动作摘要afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868与C1一致。30797新文件SHA/mtime不变；7991历史文件、3472Joint缓存、70原FAIL文件及8个保护工作树HEAD/clean保持。

审计证据/home/lzf/ldx/outputs/tbps-framework-stage4d1/full_audit_v1/result/：

- Manifest SHA913b87ecf0620730e537a33f96858e4dff41fbb6de8fa233b085da077fbbc9a2。
- report SHA114857db1ca8c29b7a44cf592ab11693feeab0c823dc3f87bac6a3543143866f。

原Stage3B FAIL、Stage3B2 PASS、Stage4B2指标重放、Stage4C2整体FAIL及Stage4C3小样本PASS不修改、不提升旧状态。

## G Runner与下一阶段

新G全量Runner仅CPU Fixture通过：工厂一次、16图Mock事务/读回/生成证明7.8663秒，真实G加载/前向0。先Claim后前向、已完成验证跳过，未提交Claim/pending在加载前拒绝，复用原G算法与像素读取，未实现GPU CLI。

局部优化只在新路径添加Manifest索引、有限来源SHA/stat缓存与64条完整动作验证缓存；首/集合强制边界/尾完整SHA、未知/输出完整SHA、原子fsync/不可变发布不变。Fixture减少重复SHA读取，未证明墙钟加速；fsync仍主要瓶颈。

3074图三FP32 Tensor纯数据5.0657958984375GiB，建议至少10GiB空闲及100000 inode；Mock存储调度线性量级约25.2分钟，非GPU实测。真实GPU资源/时限须Stage4D-2另行授权冻结，不设旧2GiB进程RSS上限。固定Plan绑定本轮producer路径，后续不能在新树重计算Plan并改签已完成H集合。

本轮已完成授权文本物化及CPU审计，提交推送后停止。未授权Stage4D-2真实G生成、IRRA Full、RaSa/APTM或训练。

## 既往阶段索引

- Stage4C-3：a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2；[两图两文IRRA真实物化端到端PASS](framework/STAGE4C3_REPORT.md)，仅固定小样本VERIFIED。
- Stage4C-2R：9f1f943ed8486a8c2889ab272764e5707a431552；[CPU时序修复/旧产物审计](framework/STAGE4C2R_REPORT.md)。
- Stage4C-2：fcf097d3ab31785381947289a680e5f5ccbf9d42；[原唯一Attempt FAIL](framework/STAGE4C2_REPORT.md)，两图G/两文H局部PASS保持。
- Stage4C-1：4d3bc025a7da593368ebd22cc05db1ebe787b446；[来源与物化契约](framework/STAGE4C1_REPORT.md)。
- Stage4B-2：c09754e6afc8cf53a6c0c6e8715b6d4eace80946，METRICS_REPLAY_VERIFIED仅历史IRRA特征排名重放。
