# Stage 3A交付与审计报告

## 结论

G-v2、H-v1和Joint最小Attack Plugin机制已实现，薄兼容层复用原网络、推理、资格检查及Unicode算法。**本轮仅CPU Fixture/Mock、合成Caption前端和旧协议只读检查；真实G/H输出等价性仍待Stage 3B授权验证。**

基准b7aac6966a51adc4c64df477f725bb0710ae86e6；独立分支codex/tbps-framework-stage3a；独立工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage3a-20261016。先读根AGENTS、PROJECT_STATE、ARCHITECTURE、EXPERIMENT_REGISTRY及ADR-0001/0002/0003。实现Commit由后续文档提交登记。

## 真实旧代码调用链

### G-v2离线准备与在线生成

- 初始评分来源：G-v1条件元数据保存配对图像、原Caption、冻结IRRA block-8的clean retrieval backward、S_raw及Natural Dynamic-K序列。源评分记录每Caption retrieval_backward_count=1。
- G-v2初始prepare.py/integration.py只读G-v1已保存score/selection/z_t，复核原序列，通过原TextTransport.read补存每属性JVP。
- 正式prepare_full.py→source_rows读取固定2048 Train+128内部Validation；复用原S_raw与选择结果。缺失JVP由prep_text_frontend.Frontend→prepare_uncapped/TextTransport.read执行，涉及冻结IRRA文本梯度/JVP及有限差分/zero检查，不是无梯度准备。已保存项可复用；此步骤属于离线训练监督，不是G测试时访问权限。
- 已存条件包含原图SHA、row_id、caption_index、role、原属性ID/S_raw顺序及[K,512] JVP、z_t/v_local。它们不是G(I)输入。
- 在线：inference_once.load_generator→ImageOnlyGenerator加载固定架构/权重；generate_once→G(image)→add_projected_delta(image,image,delta)→project_image。只做一次G前向，原始RGB攻击工作像素相对原图L∞≤8/255，端点向内舍入、像素[0,1]。
- 原inference_once不导入条件缓存、IRRA、Loss或Epoch调度；插件也不调用load_latest/current/rollback/prepare。

新provenance.condition_record仅索引元数据：原图/Caption、K、selected IDs、S_raw、完整selected attributes、原score digest及metadata/tensor SHA。校验原序列、源评分配对与gamma3/beta1。**不读取条件Tensor、不重新选择、不计算评分/JVP。**

本轮真实只读样本：
- cuhk:9955:0，K=3；
- cuhk:26530:0，K=0，空ID/分数保持。

新scoring backward=0、新JVP=0、新IRRA图文神经前向=0。未来若重做离线准备，必须独立授权并记录代理侧评分/JVP/有限差分计算；不能使用目标RaSa/APTM梯度或反馈。当前插件没有准备执行器。

### H-v1训练来源与Caption-only在线路径

- 离线formal_training/data.py绑定原2048 Train、128独立Validation；奖励由训练Clean/Adv Gallery和TrainGallery/RewardEvaluator提供。Train身份/PID/paired信息只进入奖励绑定，SCST只更新H；Validation只选Checkpoint。
- 在线：CaptionFrontend.convert→src.attributes.text_extractor.extract_with_provenance→v55 attribute_interface（继承v54/v53规则），require_representation=False，保留原始字符Span，不需要评分、正确图像或PID。
- 原formal_training.data.qualify_caption负责原文77-token截断、1024 bytes、无属性/无合法编辑等资格检查；不可攻击文本保留为NOOP。
- 原CharacterPolicy.attack/rollout使用Byte Transformer和三个Head。属性softplus预测分数→原natural_k gamma3/beta1→masked选择/STOP→属性内字符/NOOP→受限Unicode替换。每属性和共享词最多1编辑，总编辑≤K。
- candidates使用冻结IRRA Byte-BPE探针检查token改变及截断。这是训练侧固定Tokenizer约束，不是受害者神经模型评分；未来目标Tokenizer不能替代此探针决定编辑。
- 复用原phase5_joint.backend.TextAttacker.attack和validate_action；回放原constraints.replay/make_edit，保留Unicode码位、UTF-8偏移、Canonical与原Span、K、选择和STOP/NOOP。

### Joint及不能直接复用的部分

历史Phase5的Backend.text输出IRRA文本特征，OfficialTextEncoder会调用IRRA；历史Gallery直接复用G-v2 Full特征。这些评测对象不能作为跨模型攻击插件输出。

新Joint只组合GV2Plugin/HV1Plugin同一实例。图像输出真实RGB像素/扰动，文本输出原始Unicode字符串/动作；没有复制网络/训练算法或Joint Loss。Gallery与Query绑定独立，不要求正确配对，不把3074/6156数量混为一一对应。

旧代码有绝对import、sys.path、按文件位置推断REPO以及冻结相对路径。薄会话仅在显式授权时导入旧原件，拒绝同名模块冲突和未锁定依赖，退出恢复模块注册/sys.path/pyc标志。不移动旧文件，不用全目录动态SHA或关闭来源检查。

CPU合成前端检查发现src.attributes.__init__还会间接导入attribute_scoring、visual_canonicalizer。初次严格门禁拒绝后，将这两个原件加入新显式来源清单；原件未改。最终74项固定旧来源闭包（继承正式Phase5协议，含审计依赖，并非全部在生成时导入）及14项实际新框架producer锁定。

## 接口、权限与产物

完整文件表、API、扩展示例、CPU命令见[STAGE3A.md](STAGE3A.md)。关键契约：

- PluginInfo分别声明历史训练方法、离线准备访问与在线访问；当前training_executable=False、offline_preparation_executable=False。没有通用train()/Optimizer要求。
- AttackBinding.from_manifest根据完整有序Manifest及真实成员生成输入摘要；Binding不含PID、正确Gallery或奖励。H后端只收到Caption字符串。
- G/H在线均不要求代理神经模型、梯度、奖励或测试时优化。配置拒绝额外source/victim反馈权限、预算改变、错误代理/目标轴及未实现Dataset/目标Model执行声明。
- Joint检查G/H同一已核验来源bundle、训练集及代理SHA，复用同一组件；三种模式共享各组件权重。
- AttackOutput与AttackArtifact分开：内存产物为像素/文本，publish_output复用新ArtifactStore原子发布、Manifest/完成回执、来源SHA，真正clone紧凑Storage；发布前拒绝被修改的像素/动作。
- 评分方没有回调进入攻击生成的通道。未知维度和Fusion Mock可以声明评分契约；不存在的RaSa/APTM不能标记为已可执行。AP历史描述符仍不能冒充已实现Attack Plugin。
- 原IRRA Embedding不进入插件返回或新攻击产物。目标模型预处理/Tokenizer/评分属于未来独立Adapter/Engine。

## Checkpoint与协议

G Epoch10 SHA：
f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。

H Epoch7/visit14336 SHA：
487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。

H不可变Selection SHA：
584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757。

训练协议canonical digest：
- G：1ada0c31ee4fd4bf35eaf02fcc5bcebc7a8276c2714f34aaa6d9a88e3501f50b；
- H：bec28bbcc7b194e44b6cbe351b2c2615418c93d12b5afdb7cf3c4da975a041a8。

代理IRRA SHA：
9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。

本轮插件来源/相关producer摘要：
96f6b2c00f604acb32d78c94534af50d9792fa2157955b0cb5d7a08b780a788d。
这些值由真实文件SHA及原协议关系核验，不反序列化权重。真实加载仍通过allow_model_load显式授权门禁，本轮未调用。

## CPU验证与历史保护

| 项目 | 实际结果 |
| --- | --- |
| 框架CPU tests | **285通过**：原213+本轮72 |
| 历史基础CPU回归 | **36通过**，未重跑全部prototype tests |
| 总计 | **321项不同CPU tests通过** |
| 原推理纯函数 | 原generate_once/投影由MockG调用；原TextAttacker/约束由MockH调用 |
| 能力/扩展 | 新Fixture插件注册无需改G/H；无train()也可实现接口 |
| 冻结/预算 | 重复/缺失G前向、状态改变、错误空间/特征输出、越界及修改后产物拒绝 |
| 文本 | STOP、position NOOP、K0、超长/截断保留、重复词Span、非法Unicode/码位、同词重复编辑、标签隔离 |
| 来源/存储 | Checkpoint常量/来源错误、条件SHA/配对/Top-K顺序/协议错误、发布/篡改拒绝与紧凑Storage |
| 合成前端 | “A man wears a red shirt and black pants.”，5属性/5精确Span；原Tokenizer对形近Unicode有变化且无新截断 |
| 无模型CPU检查 | torch.load和nn.Module forward设为禁止；旧模块导入/合成前端仍通过；CUDA未初始化 |
| 旧资产 | **7991份文件SHA及mtime_ns不变**；Stage 2.5 attempt/result回执及全部绑定来源只读校验通过 |

最初CPU Fixture误把r视作合法替换字符，旧白名单正确拒绝；修正Fixture为合法e/а/с替换。另一Fixture直接原地改inference Tensor被PyTorch拒绝，改为构造独立修改副本以测试发布篡改拒绝。没有改Unicode规则、推理或研究算法。

原实验树40c7476、Stage 1树61471db、Stage 2.5树b7aac69均clean；新树没有prototypes/官方源/旧数据变更。原7991保护范围涵盖历史tracked源码、固定模型/协议/结果、2176份G metadata及选定原图，不等于重新审计全部Full特征。

本轮G/H/IRRA真实模型加载=0、真实前向=0、GPU=0、训练/optimizer=0、评分/JVP=0、Full排名=0。CPU Mock前向不计为真实攻击输出；合成前端和Byte-BPE不是IRRA.encode_text。

## 证据

新输出：/home/lzf/ldx/outputs/tbps-framework-stage3a/（不提交Git）。

- cpu_verified.log：285 tests通过；SHA 89c09a001af040e29b73ed0767762a7465a8ab83159a8ef3fc9f5a8e79de66f7。
- legacy_cpu.log：36通过。
- source_verified.log：STAGE3A_PLUGIN_SOURCE_AUDIT_PASS；SHA 90e421118589bd35ea727c4f30658f54e882b6433c6b33f2f59239375566670a。
- import_only.log：LEGACY_IMPORT_ONLY_NO_MODELS_NO_FORWARD_PASS。
- frontend_synthetic_cpu_final.log：CAPTION_FRONTEND_SYNTHETIC_CPU_NO_MODEL_PASS。
- protection_result.json：SHA 56b57490a56d5c03c21c4334684efb89ea9cb00b31905e7e3912cc4bc22aec1c。
- 初期CPU失败日志保留，仅属本轮开发证据，未覆盖历史证据。

## Stage 3B最小建议（未授权、未执行）

1. CPU预检固定2张已核验内部Train图像及2条原始Caption，尽量覆盖一个可编辑输入和一个固定NOOP；不按攻击收益筛选。
2. 独立进程、空闲GPU预检、固定同一G10/H7权重，各只加载一次；H按原实现保持CPU，G可使用获授权GPU。无需IRRA模型、Test、Gallery排名或奖励。
3. A直接原generate_once/TextAttacker；B本轮Plugin，同一冻结G/H Backend、同一RGB预处理和FP32/math设置，官方/插件独立输入调用。
4. G实际样本前向最多4（2图×2路径），H最多4（2Caption×2路径，固定NOOP可以0），IRRA前向0、backward/Optimizer0。
5. 对比RGB/扰动shape/dtype/有限性及逐值差异，预先固定max_abs_error≤1e-6且预算严格8/255；H字符串、K、Span、Unicode、选择/STOP/NOOP及Tokenizer记录必须exact。
6. 核验G/H参数、buffers、Checkpoint/旧源/资产SHA不变，记录实际调用、显存与耗时。异常保留证据并停止，不擅自重试或放宽标准。
7. 分别验证image_only/text_only/joint复用和独立身份；不要求编辑或攻击收益为正，不用正确Test配对影响H。

尚未验证：本新加载会话真正反序列化G/H并生成输出、真实GPU G像素及H完整动作等价、不同batch/硬件/输入边界，以及未来目标Model原生图像尺寸/Tokenizer映射。CPU检查不能替代这些验证。

没有统一Evaluation/Training Engine、SGA/MTGA/AP插件或RaSa/APTM实际接线；不把已有IRRA Joint白盒结果称为迁移结果。完成本轮提交推送后停止，等待审查，不自行进入Stage 3B。
