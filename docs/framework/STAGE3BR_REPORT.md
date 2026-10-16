# Stage 3B-R CPU-only修复报告

结果：**CPU运行闭包检查PASS；真实G/H/Joint输出等价仍PENDING。** 本轮无GPU尝试，无权重反序列化、真实神经构造/前向、backward、Optimizer、JVP或新条件准备。

基准：`0fb90cd1b280da886aec08834dc13d28eacd5848`。独立分支 `codex/tbps-framework-stage3br`，工作树 `/home/lzf/ldx/projects/idea-test-tbps-framework-stage3br-20261016`。实现Commit由PROJECT_STATE后续文档提交登记。

## 最小修复与来源

不改历史Python或Stage 3A原配置。新增派生配置，只补入：

```text
prototypes/attribute_image_generator_v2_20261014/condition_schema.py
f13932c2688cab4e135d063eed6f066016f6ba26e2316cbc201f19b246b298b4
```

源文件是已存在原件，SHA与FAIL定位一致。原74项及全部其他配置字段逐值保留，新清单75项。派生证明绑定parent/derived文件SHA，精确限制为此1项增量；来源检查未关闭。原配置SHA保持 `d2c01ea7cc70c3c4dd56dceb5de6d654069e129201b54b468e48ff7a2c896123`。

原工厂、生成器、Policy、Tokenizer、Checkpoint、预算、攻击算法与失败Smoke脚本均未修改。只增强新框架CPU检查入口，新增独立检查器、派生配置及测试。

## 实际CPU闭包

通过真实加载工厂AST核对全部7个旧模块入口，解析11个调用符号；不执行加载函数和攻击函数。实际普通/直接文件导入共33份源码：

| 分组 | 本次实际检查的源码 |
| --- | --- |
| G，6份 | inference_once、image_only_generator、budget_projection、irra_image_adapter、cache_io、condition_schema |
| H根模块，11份 | phase2_sources、policy、contracts、constraints、text_frontend、tokenizer_probe、feedback_backend、rewards、training_inputs、phase3_inputs、real_feature_cache |
| H子包，8份 | phase5_joint/{__init__,backend,common}；formal_training/{__init__,common,data,validation,storage} |
| Canonical，7份 | src/attributes/{__init__,attribute_scoring,visual_canonicalizer,text_extractor}；v55/attribute_interface、v54/axis_audit、v53/factorized |
| 官方Tokenizer，1份 | utils/simple_tokenizer.py（纯CPU规则，不是IRRA神经模型） |

总计6+11+8+7+1=33。全部绝对路径、SHA/mtime详见不可变CPU report的executed_sources。另逐条核验75项runtime来源，包括BPE、ontology、配置及未被执行的原锁定文件；不把静态来源项冒称实际导入。

预检包含原身份/资格检查，锁定Train的cuhk:9955:0（可编辑）和cuhk:26530:0（固定NOOP），不读取测试标签构建策略。H前端6项声明来源一致，官方Tokenizer/BPE SHA及77-token规则保持原样。

新增SourceFileLoader执行前校验覆盖普通import之外的直接动态文件加载。CPU硬闸门覆盖整个preflight和追加检查，阻断权重恢复、神经构造/前向、CUDA初始化和导数；禁用调用实际尝试数为空。两次导入会话退出均恢复命名空间设置，旧模块无残留。

## 测试与结果

- **323项框架CPU测试通过**：298历史框架 + 25新增。
- **36项历史基础CPU回归通过**，合计 **359项不同测试**；没有重跑全部prototype测试。
- 合成闭包15份依赖逐项删pin和逐项篡改SHA均拒绝；直接文件加载也拒绝缺项/篡改。
- 覆盖正向通过、工厂新增/动态入口拒绝、缺失符号拒绝、已占用命名空间拒绝、异常后路径/模块/锁恢复、CPU硬闸门恢复及捕获异常仍拒绝。
- 派生配置覆盖唯一增量通过、原字段变化拒绝、父配置/新增源SHA篡改拒绝、CPU配置绑定错误拒绝。
- 真实服务器严格CPU检查退出码0，`GH_RUNTIME_PREFLIGHT_PASS`；75项来源、33份实际源码、11个符号、全部入口完成，无额外缺项。
- 原Stage 3A缺项配置的拒绝回归仍保留，避免静默补齐或忽略旧证据。

日志在 `/home/lzf/ldx/outputs/tbps-framework-stage3br/{strict_preflight.log,cpu_framework.log,cpu_legacy.log,protection_result.json}`。

不可变CPU证据：
`/home/lzf/ldx/outputs/tbps-framework-stage3br/cpu_closure_v1/result/{report.json,protection.json,manifest.json,completed.json}`。

- report SHA：`b974a016a61876fa0c4ee40df8faff6d9b83413afc092cf4d1a978be252d4401`。
- Manifest SHA：`19434bdcdbda7255ba582190fa5c003b8a0fe98a58fc27e0878cbc8392217f34`。
- 新回执绑定45项直接来源（含本轮检查器/配置/tests和旧FAIL回执）；不是新GPU实验回执。

## 历史保护

- 7991份历史保护文件SHA和mtime_ns完全不变；不是所有Full特征的重新审计。
- 原Stage 3B输出17份文件SHA/mtime完全不变；attempt/result回执及2358项绑定来源只读验证通过，无pending。
- 原FAIL report SHA仍为 `e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c`。
- 原实验树40c7476、Stage 2.5树b7aac69、Stage 3A树ef835a6、Stage 3B树0fb90cd均保持原HEAD且clean。
- 新树只有src/tbps、configs/framework、tests/framework和相关统一文档变更；历史prototypes/模型/数据/结果无diff。所有新运行证据均在独立outputs命名空间，不提交Git。

## 下一步与未验证范围

CPU阻断已解除，可在审查后重新申请受控GPU等价Smoke。仍未反序列化Checkpoint、构造真实G/H、验证设备迁移、比较G像素/完整H动作或真实Joint组合；不能将状态改为VERIFIED。

未重跑失败equivalence_v1，未创建新的GPU attempt，无Full/训练/Stage 4。GPU许可必须由用户另行给出，并使用新的不可变输出；本阶段完成提交推送后停止。[CPU命令与文件职责](STAGE3BR.md)。
