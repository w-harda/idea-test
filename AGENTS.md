# TBPS项目长期开发规范

本文件随Git维护。当前状态唯一入口是[docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)；架构见[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，实验来源见[docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md)。旧README/PROJECT_STATUS/阶段报告和Windows旧AGENTS是历史快照，不用于推断当前授权或完成状态。人的最新任务指令优先。

## 每次开始任务

1. 读本文件和PROJECT_STATE；按需读ARCHITECTURE、EXPERIMENT_REGISTRY及相关ADR。
2. 检查当前分支、HEAD、工作树状态，辨别开发树与实验树。
3. 先检索已有模块和公共能力，再新建文件。
4. 明确本轮范围、依赖、不可修改的历史源码/资产和允许的CPU/GPU工作。
5. 无证据的状态标UNKNOWN/PENDING，不据聊天记忆或旧CURRENT猜测。

## 目录与命名

- 新框架集中于src/tbps、configs/framework、tests/framework、docs/framework；稳定模块按职责命名，不继续增加按日期堆叠的公共模块。
- prototypes、src/attributes、scripts、ontology、旧configs/tests/docs均为冻结历史原件；除用户逐项授权，不修改、移动、删除、重命名或重定向其import。
- 兼容桥只读旧协议/结果；不能关闭SHA检查、重写旧证据或把IRRA特征标成其他模型可用。
- 接口能力必须真实：不强制512维、77-token、固定图像尺寸或cosine；Model Adapter须表达原生融合评分。
- Attack Plugin持有专属Loss、Optimizer和算法循环；公共层提供契约、存储及调度服务，不统一论文算法。
- Policy输入只含公开图像/Caption及Caption-only属性，PID/正确配对/奖励留在训练绑定或评测身份层。

## 运行与资产保护

- 用独立Git工作树和codex/分支开发；不要在实验树pull/checkout/merge/reset或修改运行依赖。
- 源码、测试和Git操作默认在服务器完成；Windows只维护或阅读Markdown，不维护第二套实验源码。
- 服务器新写入仅/home/lzf/ldx；原始数据只读；不改共享资源、官方模型源码、系统CUDA/Driver或其他用户目录。
- GPU、缓存准备、训练、评测、恢复、后台启动及持续监控必须有本轮明确授权；默认dry-run不得加载模型/启动CUDA/执行实验。
- Checkpoint、数据、缓存、outputs、日志、临时证据不提交Git。只提交代码、测试、必要配置及本轮影响的文档。
- Tensor view需紧凑Storage时真正clone；不要仅用contiguous。SHA/协议不匹配拒绝，不静默fallback。
- 分别记录模型训练数据、攻击训练/Validation/Test、代理/受害者权重、访问权限、提交更新和实际执行数量。
- Baseline默认复现原论文/固定官方Commit的算法、完整流程、损失和参数；接口接线不得暗改攻击。需要算法修改时先报告并另行授权、单独命名适配。执行/缓存审计PASS不等于原算法复现PASS；SGA归类以ADR-0009为准。

## 每次结束任务

1. 完成授权范围测试与核验，不越阶段运行实验。
2. 更新受影响的PROJECT_STATE、ARCHITECTURE、EXPERIMENT_REGISTRY/ADR；不复制全部日志，不新增多个CURRENT。
3. 记录真实完成、限制、测试和对应Commit；提交自指SHA用后续文档提交明确记录代码Commit。
4. git diff --check；仅暂存本轮范围，排除实验资产；代码和相关文档一起提交。
5. 用户允许发布时push并核对远端SHA，报告分支/Commit/范围/验证与下一步，然后停止。
6. 重要架构/研究决定写ADR；普通实现细节不机械创建ADR。未经授权不得进入下一阶段。

默认中文交流及Markdown；代码、API、路径、报错和SHA保持原样。不要把凭证写入代码、配置、报告或工具输出。
