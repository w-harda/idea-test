# Stage 1：基础契约、配置校验与历史资产只读桥交付

## 实施范围与Commit

基于审计Commit40c7476f92d6bc2458d86029cf7bf8317da0a086，独立服务器工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage1-20261016，分支codex/tbps-framework-stage1。两份全仓审计的必要设计结论已纳入ARCHITECTURE及ADR；未迁移任何旧攻击算法。

管理规范提交：16a5e3f3086b633b413cfe149e10bf9fc0b589dd。
实现提交：ffd18dd9029b493defc424668eb943d2ad1d5739（后续文档提交登记）。

## 新增文件及职责

| 文件/目录 | 交付 |
| --- | --- |
| AGENTS.md | 长期工作流程、命名、权限、来源保护与提交规则 |
| docs/PROJECT_STATE.md | 唯一当前状态入口，明确完成/下一阶段/限制及Commit |
| docs/ARCHITECTURE.md | 新旧模块边界、目录、迁移关系与历史文档索引 |
| docs/EXPERIMENT_REGISTRY.md | G/H/Joint/消融/AP不可变来源索引 |
| docs/decisions/0001-frozen-legacy-bridge.md | 冻结旧资产与只读桥ADR |
| docs/decisions/0002-phase-permissions-native-scoring.md | 信息权限、原生评分及分阶段执行ADR |
| src/tbps/__init__.py、compat/__init__.py | 无副作用包入口 |
| src/tbps/contracts.py | frozen/slots身份、输入、能力、预算及3类产物契约 |
| src/tbps/config.py | 严格JSON解析/权限/来源校验及唯一dry-run入口 |
| src/tbps/artifacts.py | 专属新输出不可变Manifest/回执、原子发布、SHA、紧凑CPU Tensor |
| src/tbps/compat/legacy_assets.py | 字节/JSON只读桥与缺失/冲突处理，无旧模块import |
| configs/framework/example_irra_gv2.json | 真实路径配置解析示例 |
| configs/framework/legacy_assets.server.json | 8类资产固定SHA与协议关系；不含权重/缓存文件 |
| tests/framework/framework_fixtures.py | 小型合成来源/协议Fixture |
| tests/framework/test_contracts.py | 维度/Tokenizer/fusion/身份/预算/权限契约 |
| tests/framework/test_config.py | 配置、迁移轴、能力拒绝、导入副作用与CLI |
| tests/framework/test_artifacts.py | 发布/中断/篡改/来源/并发/Storage |
| tests/framework/test_legacy_assets.py | 旧证据SHA/协议/Selection/回执/缺失/只读 |
| tests/framework/test_boundaries.py | Test角色、深层不可变、子协议、symlink及跨文件关系 |
| docs/framework/README.md、STAGE1_REPORT.md | API/配置/bridge示例、CPU命令、边界和本报告 |

配置清楚区分攻击器训练来源与代理/受害者模型自身训练来源；没有固定全框架512维/77 tokens/cosine。上述数值仅属于IRRA历史描述符。ModelSpec支持无独立embedding的融合评分，ScoreArtifact保存其原生评分协议。H公开输入无PID、正确Gallery、配对标签或奖励。

## CPU测试与真实只读描述

| 验证 | 实际结果 |
| --- | --- |
| 框架独立CPU tests | 114通过 |
| 历史基础CPU回归 | 36通过：test_dataset_adapter、test_dynamic_topk、test_generator_framework、test_joint_baseline |
| 合计 | 150通过；不是全prototype历史tests重跑 |
| 示例配置CPU dry-run | CONFIG_VALID_DRY_RUN；executable=false；路径存在但不冒充权重SHA已验证 |
| 真实历史桥 | 8类METADATA_VERIFIED，192个唯一文件；无模型/Tensor加载或排名 |
| 模块副作用 | 子进程禁止Torch/历史模块导入的检查通过；CPU Tensor tests后CUDA未初始化 |
| Storage | 合成旧view文件超过500,000字节，clone后返回单行Storage为2048字节；数值逐值一致，新紧凑文件小于10,000字节 |
| 原子性 | 写入后/回执前/发布前故障、真实子进程os._exit、并发锁、SHA篡改/来源冲突及pending保留均通过 |

真实只读检查初次对照时发现消融trial使用独立子协议，不是直接使用父digest；新catalog已依据实际子协议→parent_sha256建立绑定并加入CPU测试。AP原attack_modality为image/gallery，按原值固定。没有放宽原门禁或改写旧协议来使检查通过。

新桥位置：
/home/lzf/ldx/outputs/tbps-framework-stage1/bridges/readonly-history-v3/。
早期v1/v2仅为本轮开发期新目录描述快照，未改写；最终验收以v3及其3个producer SHA为准。目录配置只保存路径/SHA/关系，不复制原协议正文、指标表、Checkpoint或缓存。

## 历史资产保护证据

- 实验/发布树HEAD保持40c7476f92d6bc2458d86029cf7bf8317da0a086，工作树干净；没有pull/checkout/训练恢复。
- 426份初始保护记录与192份只读桥来源合并去重，共555个文件SHA/mtime_ns逐项不变；其中包括原412个tracked文件。
- 固定结果/协议/Selection/相关Checkpoint和消融日志均包含在核验范围。未声称对全部历史特征二进制重新做SHA或Full排名审计。
- 原prototypes、src/attributes、训练程序、官方IRRA、数据、权重、历史输出和Windows旧管理文档不修改。
- 新写入仅在独立开发树、/home/lzf/ldx/tmp/tbps-framework-stage1与新tbps-framework-stage1输出命名空间。

本轮CPU证据留在：
/home/lzf/ldx/tmp/tbps-framework-stage1/{cpu_framework_final.log,cpu_legacy_regression.log,bridge_check_final.log,protection_result.json}。
这些日志/运行输出不提交Git。git diff --check及提交范围核验已通过；相对审计基准仅新增22个文件，没有修改任何基准tracked文件。

## 兼容范围及限制

旧结果可只读索引/报告；G E10绑定训练契约，H Epoch7绑定Selection及候选Validation已存结果，Joint与消融绑定完成证据。消融日志含两项成功标志，永久纳入只读资产。AP维持metadata级来源索引。

METADATA_VERIFIED不等于重新审计所有特征或权重内部Adam/RNG/训练轨迹。桥不调用load_latest、rollback、prepare或旧Store；历史路径缺失返回NOT_READY，来源冲突拒绝。IRRA特征不能跨模型使用；Stage 1也不允许直接执行同模型特征复用。

新资产发布有Linux flock/原子目录/不可变回执；中断pending保留并拒绝静默重试，没有自动清理或Step恢复承诺。Tensor支持只在显式CPU方法中惰性导入Torch；旧缓存系统未修改。

当前仅JSON配置，DatasetManifest仅声明all_same_pid，未实现真实Dataset/Model/Attack Plugin。RaSa/APTM、ICFG/RSTP及未来Baseline明确拒绝；所有已知历史描述符同样executable=false。配置合法与资产已核验、算法可执行严格分开。

## 下一阶段建议与停止

Stage 2优先CUHK官方Dataset Adapter及IRRA只读Model Adapter，保留原有序身份/公开输入投影、Tokenizer/预处理/原生评分契约，之后再做经授权的等价验证。RaSa/APTM需真实源码/权重及融合评分能力，不能包装成统一512维cosine。

本轮没有GPU任务、G/H训练、正式Full重算、真实模型forward或后台监控；未自动进入Stage 2。代码及文档发布后停止等待源码审查。最小使用与可复制CPU命令见README。
