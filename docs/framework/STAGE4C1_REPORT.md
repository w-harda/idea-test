# Stage4C-1报告：真实RGB与Unicode产物准备（CPU-only）

## 结论

已建立新框架的模型无关物化接口、严格存储、历史Unicode只读桥、前置生成Claim与CPU来源预检。**G实际Test RGB仍NOT_MATERIALIZED；H历史文本来源TEXT_SOURCE_VERIFIED，尚未批量发布新的真实文本单元。**

开发基准c09754e6afc8cf53a6c0c6e8715b6d4eace80946；独立分支codex/tbps-framework-stage4c1；工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c1-20261017。实现Commit由PROJECT_STATE后续登记。没有真实G/H/IRRA模型恢复或前向、GPU、JVP、训练、Full生成或排名；原Stage4B2重放结果保持。

## 1. G条件覆盖与真实依赖

直接核对冻结源码：

1. G2/evaluation_phase29a/eval_backend.py::RealBackend.image读取原图RGB，调用inference_once.generate_once(self.g,image)。
2. generate_once只接generator,image，执行G(image)及原add_projected_delta(image,image,delta)，不接Caption、z_t、属性、S_raw、Top-K或JVP。
3. load_generator只读G不可变Checkpoint/sidecar及架构，不调用load_latest、训练恢复或条件准备。
4. 原RGB处理是PIL RGB、bilinear Resize384×128、ToTensor [0,1]，预算/投影发生在这个空间，IRRA标准化后置。

| 项目 | 真实检查结果 |
| --- | --- |
| Test原图/完整有序身份 | 3074/3074，原图SHA全部通过 |
| G-v2生成必需的Test条件 | **0，纯G(I)无需条件** |
| 缺少的必需Test条件 | **0** |
| G-v2训练监督缓存 | 2048 train+128内部validation=2176，均official train，Test成员0 |
| 历史G-v1 Test条件 | 3074，Manifest/完整成员及6148个Tensor/metadata文件SHA通过；只校验字节，不反序列化 |
| G-v1条件供G-v2生成使用 | **否**，模型/输入契约不同 |
| 可用的旧G-v2 Test真实RGB | **未由原Full保存，不能从IRRA特征恢复** |

所以本阶段不存在需要补算的G-v2 Test JVP缺口。旧训练的S_raw/Dynamic-K/JVP是训练监督，不是在线生成输入；本次均未计算。3074张真实G2 RGB仍需未来另行授权的单次G生成并保存，不能将旧feature SHA冒充像素SHA。

## 2. 历史H攻击文本复用

原来源：RELEASE/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/；RELEASE=/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014。

原protocol、Selection、H7 Checkpoint/训练协议、summary/completed及385批metadata/intent/receipt（1155文件）通过SHA与身份关联。6156条文本均经原Canonical/Span/Unicode候选/编辑重放/Tokenizer规则校验，完整顺序和计数与旧summary一致：

- 有编辑6116条、NOOP40条、实际字符替换13537次。
- 保留全部Query，包括原文本截断/不具资格案例；没有按奖励或Test指标筛选。
- 有序文本/动作摘要afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868。
- 可直接复用冻结H7已产生的UTF-8字符串和完整action，不需要重复H推理或恢复H权重。
- 显式historical_text_reuse来源，保存原metadata记录指针、原协议/完成/Selection FileRef；发布和读回再次核对动作确为该条历史记录，不能用另一条合法编辑冒充。
- 没有读取features.pt，也没有从特征反推文本。H的新预测重要性/贪心动作未重算，复用的是原SHA绑定结果。
- 新Text Attack Artifact发布API已通过Fixture；本轮真实6156条只做来源验证，没有批量复制为新单元。

冻结权重：G10 f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e；H7 487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df；历史训练代理IRRA 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。后续目标模型不会改变这组攻击器来源。

## 3. 接口、存储及异常语义

新增src/tbps/materialization/的contracts、store、pipeline、generation、legacy_text、audit_sources及无副作用__init__；新增tests/framework/test_stage4c_materialization.py、接口说明、本报告与ADR-0007。已有G/H插件、生成器、策略、投影、Loss、Tokenizer、Checkpoint、旧缓存/评测源码均不改。

计划绑定完整DatasetManifest与G/H训练来源，缓存不可变摘要；没有目标Model/Score/Tokenizer入口。请求由公开RGB/Caption与独立产物身份构成，PID/配对不传给后端。G/H插件仍调用原方法。

正式G单元保存CPU FP32原RGB、对抗RGB及effective_delta三Tensor；严格验证[1,3,384,128]、有限性、[0,1]、delta=adv-original及FP64 L∞≤8/255。原JPEG文件SHA、原/Adv/delta Tensor内容SHA、torch.save文件SHA分开记录，载入/保存clone。没有JPEG替代或预览混用；Fixture支持其他小尺寸，真实G固定原工作尺寸。

H单元保存original.txt/adversarial.txt及完整action/来源JSON；UTF-8内容SHA不做Unicode规范化，保留Span、原/替换字符/码位、Dynamic-K、STOP/NOOP和Tokenizer审计。读回必须经冻结Caption-only校验器。

复用ArtifactStore原子Manifest/回执和pending机制，不重写旧Store。可恢复生成先发布不可变Claim，再调用G/H；前向后未发布或生成证明未完成时拒绝自动再执行。完整单元校验后跳过，避免重复前向；中断不自动清理/回滚或承诺Step级恢复。Joint只封存两类产物引用，不重新生成。

为了完整规模避免CPU开销，Manifest/计划摘要只计算一次。会话首次和集合边界做完整来源SHA验证；期间检查inode/size/mtime/ctime，任何变化即拒绝。测试覆盖mtime恢复但ctime变化的篡改；不是关闭来源检查。

## 4. CPU验收及历史保护

**478框架tests（435原有+43新增）+36历史基础=514项通过。** 新增覆盖预算/像素SHA/错误Embedding拒绝、FP32与Storage紧凑clone、UTF-8/Unicode动作、NOOP保留、身份顺序、多Caption、来源篡改、不可变发布、pending/中断、前向前Claim、已完成跳过、Joint零新调用及不同ModelSpace缓存隔离。

源检查全程禁止torch.load、神经构造/前向、CUDA、Backward/JVP。实际报告model_loads、G/H/IRRA_forwards、feature_tensors_loaded、rankings、real_units_published均0。来源检查61.10218879207969秒、CPU peak RSS566.703125MiB，是CPU源检查资源，不是全量生成性能。

初版CPU源检查发现逐Query重复序列化完整Manifest的性能问题，在其自身CPU进程安全SIGINT后优化；source_audit_v1.log保留，未发布其完成回执、未涉及模型或GPU。source_audit_v2通过。开发测试另修正Inference Tensor变更测试写法及Fixture导入顺序；未放宽来源门禁。

7991个历史保护文件SHA/mtime不变。当前3472个原Joint特征文件SHA/mtime快照摘要与Stage4B1原记录exact，原Stage4B1/4B2完成回执及全部绑定来源有效；旧实验发布树、Stage4B1和4B2工作树保持原HEAD/clean。原训练、Full、Selection、消融/AP和原FAIL证据不修改。

证据：/home/lzf/ldx/outputs/tbps-framework-stage4c1/。

- source_audit_v2/result/report.json SHA37d244e6ec250a5d536e4a4e06832005a9e896233cc9b908f88e9b83365bf6e7。
- source_audit_v2/result/manifest.json SHA484605d0501d94dbb4a21315ba0f61c76499e17d283fd2a51c6eb7a1d96c9ce7，绑定10570项FileRef。
- source_audit_v2_resources.json、protection_result.json、cpu_framework_verified.log、cpu_legacy.log。
- generation模块在源检查后新增，仅增加前置Claim控制；原来源检查producer与回执保持有效，新模块由最终CPU验收另行绑定，不重新执行真实来源或模型计算。

## 5. Stage4C-2最小GPU方案（仅提案，未执行）

候选在本轮按固定来源先锁定，覆盖可编辑与NOOP，不按攻击收益选择：

| Query | Gallery身份 | 原Gallery index | 覆盖 |
| --- | --- | ---: | --- |
| cuhk:37132:0 | cuhk:37132:0 | 0 | 首个已有合法编辑案例 |
| cuhk:37147:1 | cuhk:37147:0 | 15 | 首个固定NOOP；注意caption_index=1 |

后续经用户授权新建独立Attempt/两样本协议，严格CPU闭包/来源门禁和GPU空闲检查后：

1. G10只恢复1次，两个Gallery各一次G，保存并读回FP32产物；输入/输出/delta/内容SHA与生成时bitwise exact，严格预算。
2. 两条H文本从已验证历史记录复用，H模型加载/前向0；原文、Unicode动作、Tokenizer审计exact，NOOP保持。
3. 官方IRRA只恢复1次，eval/冻结/inference；原/Adv图像样本前向≤4，原/Adv文本≤4（NOOP可复用原文特征），G样本前向≤2。
4. 只检查512维/有限/L2、原生预处理与Token ID、四组合2×2 score、参数/buffers冻结、来源、资源和产物读回。模型参数更新/Backward/JVP=0。
5. 不计算mAP/R@10等Full指标，不升级为正式Test或跨模型结果。建议FP32/math/no_tf32、1e-6数值阈值、RSS2GiB/总时长300秒候选预算，执行前需另行批准并冻结；不能事后调阈值。

本阶段没有GPU执行命令，未消费下一阶段许可。

## 6. 全量物化成本预估（未实测，不是已批准运行配置）

采用batch1流式图像、复用全部历史H文本，无IRRA前向/新JVP：

- 每个FP32 RGB Tensor589824字节；原图+Adv+delta三份共5439356928字节，即**5.0657958984375GiB纯Tensor数据**。加序列化、文本动作/索引、Manifest/回执、Claim和小文件开销，建议预留**8–10GiB**，不含原数据/已有权重。
- H历史metadata原文件约24.50MiB；物化单元会保存完整动作及trace，建议额外按100–300MiB及文件系统开销规划。新文本本身无需GPU/H前向。
- G-only GPU的历史Stage3B-2固定小样本峰值allocated45.1982421875MiB/reserved56MiB；不是本流水线Full实测，CUDA上下文和驱动另计。建议G-only空闲显存至少1GiB；后续加IRRA的GPU Smoke建议至少2GiB，仍须现场检查。
- CPU流式只留少量RGB、计划和动作元数据，建议≤2GiB工作预算；不能把全部三份RGB一次装入内存。本轮源检查峰值566.70MiB，仅供量级参考。
- 全量墙钟时间先按**10–60分钟量级**做工程预估，非基准承诺：原图解码、逐单元SHA/fsync与18,674,174字节G Checkpoint来源重复核验可能比G计算更慢，网络/慢磁盘可超过范围。最低5.07GiB数据以50–200MiB/s单次顺序写入约26–104秒只是I/O下界，不能当作完整运行时间。必须等下一阶段小样本实测再估算，不能直接把历史Full评测总时长当G生成速度。

## 7. RaSa/APTM接入与当前限制

同一MaterializedUnit/集合由目标Adapter只读加载Adv RGB及固定Unicode；自身执行原生Resize/归一化/Tokenizer/融合评分。target_input_binding对相同内容SHA建立各自ModelSpace缓存身份，不复用IRRA Embedding。目标可能改变Tokenizer长度/截断，只能记录，不能反馈改字符、选属性或调整G。

没有实现RaSa/APTM真实Adapter、跨主机资产迁移或统一GPU生成CLI；也没有测试真实Test像素端到端链路。本轮只准备接口和证据，不将Fixture通过表述为真实输出等价。

提交源码/测试/文档后停止等待审查，不自动Stage4C-2、Full、GPU、Stage5或新训练。缓存、权重、Score和outputs均不入Git。

最终CPU验收：/home/lzf/ldx/outputs/tbps-framework-stage4c1/cpu_verified_v1/result/；
Manifest SHA0f96f115d5a8d1b3c81a9e067b5c524041f4362c84f85483bdb3977300bf01da；
report SHAf10fbc4276e80048b59316b15ba8018824ac8a03cc2b5e34de0e2322714a10e5。
绑定包括generation.py在内的全部7个新模块、43项新测试源码、CPU日志、真实只读源检查与保护证据；不是实际攻击物化完成回执。
