# 项目当前状态（唯一权威入口）

状态来自Git、代码及不可变证据；历史文档是阶段快照。架构见[ARCHITECTURE](ARCHITECTURE.md)，历史实验见[EXPERIMENT_REGISTRY](EXPERIMENT_REGISTRY.md)。

## 当前阶段

**Stage 3B-2唯一一次真实G/H/Joint输出等价Smoke PASS。** 仅对应固定两条Train样本、G10/H7、FP32/math/no_tf32、RTX4090-G/CPU-H的等价状态为VERIFIED。没有进入Stage 4、Full或跨模型评测。

- 基准：`a2988e0ae2a77f41f5416c02e354f4651ed49646`。
- 独立分支：`codex/tbps-framework-stage3b2`。
- 独立树：`/home/lzf/ldx/projects/idea-test-tbps-framework-stage3b2-20261016`。
- 实现Commit：本次提交后由后续文档提交登记。
- [入口/执行记录/停止边界](framework/STAGE3B2.md)；[真实数值、动作与保护报告](framework/STAGE3B2_REPORT.md)。

## 本轮结果

- 新增最小gated入口，GPU执行前必须运行Stage 3B-R严格CPU门禁。原75项派生来源、7个工厂入口/11符号/33份实际导入检查均PASS；退出码0、GH_RUNTIME_PREFLIGHT_PASS。
- 固定样本cuhk:9955:0（可编辑）和cuhk:26530:0（固定NOOP/K0），均已核验official train/attack_train；不是从攻击效果挑选。
- G10/H7各加载一次成功；实际G/H/IRRA样本前向4/2/0，IRRA加载0。backward/Optimizer/JVP/新条件准备0。
- G输入像素、原始delta、对抗RGB和effective_delta均bitwise exact、max_abs_error=0；shape=[1,3,384,128]、FP32、有限。最大L∞=0.0313725471496582≤8/255。
- H的12个顶层字段及全部嵌套动作/Span/重要性/Tokenizer逐字段exact。可编辑样本K2、2编辑（e→Cyrillic е、o→Cyrillic о），BPE21→25不截断；固定K0样本保持原文、H_forward=0。
- Joint三模式共享同一G/H组件，独立图/文身份组合通过，额外前向0。原生A与Plugin B独立执行，没有复用A输出冒充对照。
- G/H参数、buffers和Python/NumPy/Torch CPU/CUDA RNG前后摘要exact；eval、requires_grad=false、grad=None。
- 受控run_once段1.8626863285899162秒；GPU peak allocated45.1982421875MiB/reserved56MiB。GPU进程已退出，计算进程为空。

## 测试、证据与保护

**332框架CPU tests（323原有+9新增门禁）+36历史基础=368项通过**；GPU结束后框架回归再次通过。没有重跑全部prototype或Full指标。

新证据：
- `/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2/{attempt,result}/`。
- report SHA `50afe401bf57ad244d74a014221825b58a1feb24f07201a42c501404dffecbb6`。
- attempt Manifest SHA `731e99aa2b8f19cf30ec1d8cdfad9ee78c1bca12d649d45e204269188bc8ec85`。
- result Manifest SHA `26185c4cad3f964c121a4d25d189d8f12dd8db3bfb5bd76dd734bde5c99d1f43`。
- CPU/GPU日志与只读保护审计在 `/home/lzf/ldx/outputs/tbps-framework-stage3b2/`。

7991份历史文件及原Stage 3B输出17份文件SHA/mtime不变；原FAIL回执2358项来源、新attempt2362项来源、Stage 3B-R CPU回执全部核验通过，无pending。原实验/Stage2.5/3A/3B/3BR树HEAD和clean状态保持。未改历史算法、Checkpoint、Tokenizer、预算、缓存、协议或正式结果；原Smoke比较脚本和工厂也不改。

## 原FAIL与CPU修复历史

Stage 3B发布0fb90cd，唯一旧尝试在G恢复后因缺少condition_schema来源pin停止；真实G/H/IRRA前向均0。原 `equivalence_v1` 保留FAIL，report SHA `e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c`。[FAIL报告](framework/STAGE3B_REPORT.md)。

Stage 3B-R发布a2988e0、实现7acadf6，仅通过独立派生配置补入condition_schema原SHA，75项来源CPU闭包修复PASS、359项CPU tests通过。[修复报告](framework/STAGE3BR_REPORT.md)。其CPU证据及原Stage 3A配置保持原样，本轮不覆盖或恢复旧attempt。

## 冻结历史与未验证范围

Stage 1发布61471db；Stage 2发布4e2a583；Stage 2.5发布b7aac69，真实IRRA最小等价通过；Stage 3A发布ef835a6完成薄插件CPU接入。

G-v2训练/Full、H正式SCST和Epoch7 Selection、Joint Full、Joint消融/AP及8个历史资产索引继续冻结。历史IRRA Embedding不得跨模型重标或复用。

本次不覆盖全Train/Test、批量、其他尺寸/硬件/精度、采样策略、真实主动STOP/位置NOOP、目标Tokenizer或RaSa/APTM。插件保守execution_kind标签仍保留legacy_unverified，不代表全局模型/数据组合均已验证；对应固定环境的结果验收由本次不可变证据限定。

## 下一步与权限

本轮唯一GPU尝试已用完；完成提交推送后停止等待审查。任何新的GPU尝试、Stage 4、Full、训练或跨模型任务均须另行授权。不得恢复/覆盖equivalence_v1或equivalence_v2，不自动重试或扩大样本。
