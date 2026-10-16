# 项目当前状态（唯一权威入口）

状态来自Git、代码和不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，历史实验见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

Stage 2.5已审查；**Stage 3A G-v2/H-v1/Joint薄插件已完成CPU验收，等待独立审查**。未执行真实G/H前向、GPU或Stage 3B。

- 基准：b7aac6966a51adc4c64df477f725bb0710ae86e6。
- 独立分支：codex/tbps-framework-stage3a。
- 独立树：/home/lzf/ldx/projects/idea-test-tbps-framework-stage3a-20261016。
- 实现Commit：487584a0faa36d3b1a559b50558b9cd3179d9195（由本次后续文档提交登记）。
- [接口、扩展示例和CPU命令](framework/STAGE3A.md)；[真实旧依赖/核验/Stage 3B建议](framework/STAGE3A_REPORT.md)。

## 本轮完成

- src/tbps/attacks注册/能力、独立公开输入及成员绑定、来源/配置、薄G/H、Joint组合、隔离旧import会话、新产物发布。
- G复用原ImageOnlyGenerator/load_generator/generate_once/投影，不复制算法。在线只图像，输出原384×128 RGB工作像素/扰动，严格8/255，不读条件/JVP或调用代理模型。
- H复用原CaptionFrontend、CharacterPolicy、资格检查、TextAttacker/validate_action；保留原Dynamic-K/STOP/NOOP/Unicode/Span及固定训练侧Byte-BPE约束。身份只在产物绑定层，PID/配对/奖励不进策略。
- Joint直接共享G/H组件和来源bundle，支持image_only/text_only/joint；Gallery/Query独立处理，不强制正确配对。
- 真实加载默认拒绝；training_executable与offline_preparation_executable均False。条件缓存只读索引不等于执行准备；没有统一Evaluation/Training Engine。
- 74项明确旧运行来源+14项实际新框架producer锁定；不枚举整个源码目录。G10/H7/Selection及原训练/缓存协议SHA核验通过。
- src/tbps/config.py只更正插件可用性描述，统一执行仍拒绝；AP历史描述符明确未实现，不把RaSa/APTM声明为可执行。

## 验证与证据

- **285框架CPU tests通过**：原213+新增72；另36历史基础tests通过，合计**321项不同测试**，不是全部prototype tests重跑。
- CPU Mock复用旧纯推理/约束函数，验证预算、冻结、实际调用计数、Span/Unicode、信息隔离、组合和原子产物。
- 原Caption前端对合成句提取5属性/5精确Span，官方Byte-BPE合成Unicode检查通过；禁止torch.load和nn.Module前向的进程中完成，CUDA未初始化。
- 条件索引cuhk:9955:0(K3)、cuhk:26530:0(K0)只读通过；未读条件Tensor、未重算S_raw/Top-K/JVP。
- 7991份历史保护文件SHA/mtime_ns不变；Stage 2.5 attempt/result回执及全部关联来源通过。原实验树40c7476、Stage 1树61471db、Stage 2.5树b7aac69均clean。
- 证据仅/home/lzf/ldx/outputs/tbps-framework-stage3a/：cpu_verified.log、legacy_cpu.log、source_verified.log、import_only.log、frontend_synthetic_cpu_final.log、protection_result.json。不提交资产。
- 本轮真实G/H/IRRA模型加载、真实前向、GPU、训练、Optimizer、评分/JVP、Full排名全部0。

## 已完成基础与冻结历史

Stage 1实现ffd18dd/发布61471db；Stage 2实现55b83bb/发布4e2a583；Stage 2.5实现907fdcb/发布b7aac69。其真实IRRA最小GPU等价性仅2 Train样本，Token/预处理/图文特征/2×2 Score误差0、输入梯度非零。该结果不等于G/H Plugin等价通过。

Stage 2核验官方Test 3074/6156有序身份与历史三份协议一致，原2048 Train/128内部Validation实际成员PID隔离。G-v2完整训练/Full、H正式SCST及Epoch7 Selection、Joint Full及消融全部冻结，8个历史资产索引保留。

## 限制与下一步

真实G/H加载和攻击输出仍标legacy_unverified；CPU Fixture不能宣称新旧结果完全等价。Stage 3B仅建议2张内部Train图+2条原Caption、共享冻结G/H、最多4图像/4文本策略样本前向、IRRA/梯度/Optimizer0，需用户另行授权。

历史IRRA特征未重写/重标；跨模型攻击接口只声明像素/文本与原生目标评分隔离，不宣称RaSa/APTM或其他Dataset已实现。G工作像素到未来目标预处理的映射仍需验证。

本轮提交推送后停止等待审查，不启动Stage 3B、Full或新训练。
