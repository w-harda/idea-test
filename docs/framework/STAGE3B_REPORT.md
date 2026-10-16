# Stage 3B只进行一次真实尝试的FAIL报告

## 结论

**FAIL：运行依赖闭包缺项，未到达输出等价性对照。** 这不是已经观测到数值不等价；图像/文本输出尚未生成，不能推断误差或标记VERIFIED。

本轮唯一GPU尝试后正常退出，没有重试、恢复、换权重、调seed、放宽阈值或扩大样本。G/H等价状态保持PENDING/legacy_unverified，IRRA Stage 2.5既有验收不受影响。

基准ef835a60a85c30910ef4a3442230882f72a05c44；独立分支codex/tbps-framework-stage3b；独立树/home/lzf/ldx/projects/idea-test-tbps-framework-stage3b-20261016。实现Commit由后续文档提交登记。

## 固定样本与来源

按已有2048 Train资格账本顺序，预先选取首条eligible=True和首条eligible=False，没有使用攻击收益筛选。复用Stage 2实际成员工厂核验annotation、G/H来源、2048/128划分和配对。

| row_id | 原图 | native PID | 资格 |
| --- | --- | ---: | --- |
| cuhk:9955:0 | train_query/p6841_s9083.jpg | 3193 | eligible，6个原文属性 |
| cuhk:26530:0 | test_query/p9613_s16106.jpg | 8578 | 固定NOOP，无Canonical属性/无合法编辑 |

两者official split=train、内部role=attack_train、caption_index=0。第二张目录名test_query不代表official Test；真实annotation及已核验划分确认它属于Train，没有Test前向。

图像SHA依次：
- 622ce009e5b1927f4a65171f9cdbb5a5e36e1070c84a0d66b3f09feb4f7e4fa9；
- 8032ef99b7c280aa2588bcdef40233675e7eee65d71e267dc74c3e034d96a3b4。

Caption UTF-8 SHA依次：
- 0c85e03c4508cdb222a9bc0480402c835cdcb2b93aada3bd09cb7acd3c73ae29；
- ab884c4bd6827f38bd7095f5b235c68366a7e86b1a87884cb2f8b06ba1709555。

原Caption保留全部空格与Unicode，在attempt/preflight.json中完整保存。长度为21/27 BPE tokens。第二条理由为no_canonical_attributes、no_legal_token_changing_edit，structural_K0=true；这是资格检查结果，不是已运行的H动作对照。

固定G10 SHA：
f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
固定H7/visit14336 SHA：
487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。

## 预检遗漏与确切失败链

原部分CPU预检执行来源/Checkpoint文件SHA、完整成员身份、资格一致性，以及inference_once、phase5_joint.backend和Caption前端导入；没有反序列化权重或初始化CUDA。该范围检查通过。

但预检**漏查了实际G像素读取器的传递导入**。这是本次预检的遗漏，不能把该PASS当作完整运行闭包通过。

唯一真实尝试：
smoke_attacks.run_once
→ load_legacy_plugins
→ 原inference_once.load_generator（G权重恢复1次并移至GPU）
→ legacy_runtime.py中old("irra_image_adapter").pixels_from_pil
→ 原irra_image_adapter.py:11 import cache_io
→ 原cache_io.py:6 from condition_schema import validate, IDENTITY
→ _PinnedFinder拒绝缺少runtime_sources锁定的condition_schema.py。

被拒绝原件：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/condition_schema.py。

真实文件SHA：
f13932c2688cab4e135d063eed6f066016f6ba26e2316cbc201f19b246b298b4。

这不是文件缺失、文件被改或数学定义冲突，而是Stage 3A显式运行来源清单漏项。严格闸门正确拒绝；未删除、绕过或放宽检查。

为保持执行证据源码SHA，smoke_attacks.py与被锁定Plugin源码执行后保持不变。失败后只新增check_attack_runtime.py，无GPU选项，检查实际G像素读取和H状态检查链；当前返回GH_RUNTIME_PREFLIGHT_REJECTED/exit1。新增CPU回归明确记录这一阻断，不把预期拒绝伪装为真实等价PASS。

## 原计划的对照与实际未完成项

计划A为历史generate_once/TextAttacker.attack；B为GV2Plugin/HV1Plugin，通过同一Joint组件分别执行joint、image_only、text_only。G/H共享一次恢复的Backend。Joint计划使用Gallery0+Query1交叉身份，不依赖正确配对，三种模式总计仍最多4 G、4 H前向。

| 验证 | 实际状态 |
| --- | --- |
| G原始RGB预处理对照 | 未到达，无误差数据 |
| G原始delta/对抗像素/effective_delta | 未生成，不能填0或认定exact |
| H完整动作与Unicode编辑对照 | 未生成 |
| Joint三种模式真实组合 | 未执行 |
| 推理前后RNG对照 | 未到达推理，未生成；失败原因明确为导入门禁 |
| 模型参数/buffers运行前后摘要 | 工厂未yield，未取得摘要，不能声明该项通过 |
| Checkpoint及历史来源文件SHA/mtime | 后续CPU只读校验通过，文件不变 |

脚本保留规范化全字段动作比较、原始delta实际forward hook、RGB/预算检查与计数，但这些计划不能替代真实结果。

## 实际计数、资源与异常

| 项目 | 实际 |
| --- | ---: |
| 正式GPU尝试 | **1** |
| G权重恢复尝试/完成 | **1/1** |
| H权重恢复尝试/完成 | **0/0** |
| G样本前向 | **0** |
| H策略样本前向 | **0** |
| IRRA模型加载/前向 | **0/0** |
| backward / optimizer.step / JVP / 新条件准备 | **全部0** |

GPU0为RTX 4090；开始及加载前均无compute进程，利用率0%，free24051MiB。失败段elapsed=1.0158903375267982秒（不含前置CPU预检和attempt发布）；peak allocated=5.2587890625MiB、reserved=6.0MiB。这是失败加载阶段资源，不是正常推理成本。

G原加载函数返回eval模型，薄加载器随后requires_grad_(False)；之后像素读取器导入失败。没有策略/生成器forward或参数更新。不能以此代替未取得的内存状态前后摘要。进程已退出，结束时nvidia-smi compute进程为空；没有kill或抢占其他进程。

## CPU回归与保护

- 尝试前296项框架CPU tests通过。
- 失败后最终**298项框架CPU tests通过**（Stage 3A原285+本轮13），另36项历史基础通过，合计**334项不同测试**。
- 新增完整动作逐字段exact、浮点末位差异拒绝、4/4/0上限、调用失败计数、紧凑Storage、默认CPU与单次attempt门禁。
- 两项CPU阻断回归禁止torch.load/任何nn.Module forward，验证当前像素读取链及新严格检查明确拒绝缺项。它们不证明真实模型等价。
- **7991份历史文件SHA/mtime_ns不变**；原实验树40c7476、Stage 2.5树b7aac69、Stage 3A树ef835a6均clean。
- 本次attempt/result不可变Manifest/回执及**2358项绑定来源SHA**全部只读核验通过；pending为空。
- 没有修改历史prototypes、G/H权重、缓存、协议、日志、正式结果或Stage 3A来源配置；没有Full排名重算。

## 证据

输出仅/home/lzf/ldx/outputs/tbps-framework-stage3b/，不提交Git：

- equivalence_v1/attempt/{preflight,resources,manifest,completed}.json；
- equivalence_v1/result/{report,manifest,completed}.json；
- gpu_smoke.log：GH_PLUGIN_EQUIVALENCE_FAIL；
- preflight.log：原部分预检结果；strict_preflight.log：完整运行闭包拒绝；
- cpu_before.log、cpu_final.log、legacy_cpu.log；
- receipt_protection_audit.log、protection_result.json。

SHA-256：
- attempt Manifest：1f1d9ebf83d7a92953166e4ca0e985ace498d0259e0d40abc008cc689df29ef7；
- result Manifest：a7ffd6273fe9b2748d2015695115a339e2353ebfb85cdc068824c13d233141e6；
- FAIL report：e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c。

完成回执只表示FAIL报告完整发布，不代表科学验证成功。准确CPU命令及已执行GPU命令归档见[说明](STAGE3B.md)。

## 下一步最小方案，尚未实施

1. 保留原Stage 3A配置及本次attempt/result/日志；新建派生来源配置，仅显式补入condition_schema.py及已核验SHA，不关闭或扩大闸门。
2. 新严格CPU检查覆盖所有实际加载助手、像素读取器和H状态检查链；来源完全通过后才具备再试前提。H加载后路径本次未验证，不能先宣称无其他问题。
3. 不改G/H算法、权重、Tokenizer、预算、阈值或seed。修复与必要CPU tests须先审查。
4. 任何新的真实尝试都需要用户重新明确授权；使用新的不可变attempt，不恢复/覆盖本次失败，不通过换目录绕过单次许可。

本轮停止，等价保持PENDING；不自动修来源清单、不重试GPU、不进入Stage 4。
