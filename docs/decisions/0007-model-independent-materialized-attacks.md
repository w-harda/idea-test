# ADR-0007：攻击产物先物化，受害者编码与评分后置

状态：接受（Stage4C-1 CPU基础设施；真实生成和端到端GPU尚未执行）。

背景：历史IRRA特征不足以重建或证明对抗RGB。G-v2正式推理是image-only，不需要Test条件/JVP；已有H7 Test文本和完整动作可证明原始来源。未来RaSa/APTM应消费相同冻结样本，不允许目标反馈改变策略。

决定：

- 新materialization命名空间复用G/H薄插件和ArtifactStore，不复制网络/投影/字符替换算法。G保存FP32原RGB、Adv RGB及effective_delta；分别记录原图文件SHA、Tensor内容SHA、序列化文件SHA，不能用JPEG或Embedding冒充。
- H以UTF-8原字节SHA绑定原/攻击Caption，保存完整Unicode动作；历史H文本走显式只读来源桥和原生规则校验，不伪装成重新运行H。
- MaterializationPlan没有目标ModelSpace/Tokenizer/Score字段；训练代理/历史IRRA信息仅作为来源。Public输入没有PID/正确配对/奖励。Joint只索引已有两类产物，Gallery/Query独立有序。
- 可恢复入口先发布不可变生成Claim，再调用插件；前向后未发布或回执不完整时拒绝自动重复。已完整发布并校验的单元可跳过；pending与失败证据不删除、不回滚。
- 同一计划的不可变Manifest摘要只计算一次。会话首/集合边界完整SHA核验，期间复用已验证SHA并检查inode/size/mtime/ctime；变化即拒绝，不静默接受新来源。
- 目标Adapter后续只读取同一FP32 RGB/Unicode，用自身原生预处理/Tokenizer/评分；Representation/Score缓存按目标ModelSpace与实际输入SHA隔离，不重用IRRA特征，也不回传反馈给攻击器。

影响：本轮只完成CPU来源、Fixture和存储验证。真实RGB仍NOT_MATERIALIZED，H来源TEXT_SOURCE_VERIFIED不等于新文本单元已发布。Stage4C-2和全量物化须另行授权，现有历史特征重放状态不替代端到端验收。

## Stage4C-2实测注记

固定Test两图G RGB与两条历史H Unicode成功物化，未调用全量seal；模型无关产物与目标编码会话之间只传不可变文件。IRRA会话在读取CPU历史动作前初始化CUDA，触发原cpu_only门禁，唯一Attempt FAIL。此结果未改变本ADR的权限/来源/集合契约，IRRA端到端等价仍PENDING。后续启动顺序修复必须保持CPU校验早于CUDA，另行审查/授权，不用放宽旧门禁解决。

## Stage4C-2R原计划只读引用

原FAIL Attempt中已完整提交的G/H单位，可使用独立FrozenPlanReference引用；必须验证原canonical Plan ID、原工作树producer/source SHA、Claim/生成证明及实际像素/Unicode。新工作树Plan可不同，但不得改签或将旧单位伪装成新生成；新IRRA验证回执也不得提升旧整体FAIL。只读桥没有publish/generate接口。

CPU来源/动作/缓存读取必须先于CUDA，退出后检查原隔离函数、命名空间及legacy锁恢复；保留旧门禁。Stage4C-2R CPU通过，IRRA真实等价与资源适用性仍PENDING，新的GPU许可不能由CPU修复自动推导。

## Stage4C-3固定环境验收

明确新授权取消旧2GiB进程RSS上限后，唯一IRRA-only Attempt复用原四单位，G/H加载/前向0，IRRA一次恢复/图像4/文本4；真实保存RGB/Unicode到IRRA特征和四组2×2 Score误差≤1e-6，模型状态保持。仅固定两图两文环境标记IRRA_MATERIALIZED_SMOKE_VERIFIED，不提升原整体FAIL、不生成全量集合、不推导Full或跨模型结果。旧资源协议保持，新的系统RAM资源策略单独绑定新Attempt。

## Stage4D-1：全量集合与吞吐约束

本轮只授权历史6156条H文本CPU物化及G流式Runner的CPU Fixture。H模型加载/前向必须为0，6116编辑、40 NOOP和13537字符替换按历史原样保留。完整文本语义回执在所有单位、官方顺序、历史来源和会话最终SHA核验通过后发布；Image及Joint全量仍PENDING。

只对新全量路径添加索引与作用域来源验证缓存；原通用Store和历史代码不改。有限缓存仅接受已锁定FileRef，首/强制集合边界/尾完整SHA，直接来源每次stat，全部来源stat扫描至多间隔1秒；未知/输出仍完整SHA，异常恢复原模块和锁。相同Caption+完整action的原生验证结果有限缓存，不改变字符决策。fsync和不可变事务不以吞吐为由关闭。

G未来需同一Plan流式生成：一次冻结工厂、每图先Claim再一次G、原/Adv/effective_delta紧凑FP32保存及读回，未提交前向不得自动重做。Stage4D-2资源策略另行授权，不能据CPU Fixture推导GPU许可。H完成不代表Joint已完成或跨模型效果已验证。

## Stage4D-2：同Plan追加与独立执行授权

新入口必须从冻结Stage4D1公共库执行stream_images/Store/原G插件，以实际文件路径和旧Git身份核验；新入口及监督器另行SHA绑定，不更改旧Plan、producer路径或旧H单位。旧TEXT_FULL回执中的历史PENDING不重写，新Joint回执单独表达完整集合。

GPU会话与CPU文本核验/Joint封存分进程，防止在CUDA初始化后进入严格CPU隔离上下文。一次Attempt/权重加载/实际前向/Claim/Proof与日志分别计数，旧两图内容SHA不一致则保存现场并停止。父进程只管理自己创建的子进程。

本轮GPU资源门禁拒绝外部计算进程/显存不足，实际GPU Attempt为0；不降低条件、等待重试或伪造Image/Joint完成状态。CPU开发与来源保护证据单独发布，完整生成和独立Joint验收PENDING。

## Stage4D-2正式执行失败注记

新的单次Attempt在模型加载前，因GPU安全包装改变StartupGate快照而被拒绝。先退出CPU隔离并不够，还需在CUDA初始化检查前保持快照中的函数身份；优化/权重限制包装应在初始化门禁通过之后、模型加载之前安装。此处仅记录待修复结论，本轮不修改冻结门禁、入口或算法。

原generation_v1永久保留FAIL，G前向为0也不能自动重试。原Plan/6156文本与全部历史保护通过；新的CPU-only修复和独立GPU Attempt须用户单独授权，不能用现有611项Fixture通过升级为真实成功。

## Stage4D-2R：启动快照与安全包装的顺序

CPU-only最小修复保留原门禁：退出CPU上下文、确认函数/导入/锁恢复及资源设置后，先initialize_cuda，再立即安装全部GPU权重/优化保护，安装成功前禁止G工厂。真实child组合CPU Fixture覆盖正确/错误顺序及失败恢复，不能用取消门禁解决冲突。

派生配置固定独立generation_v2与修复入口SHA，原Plan/6156文本、generation_v1整体FAIL与原工作树脏报告保持。当前只有CPU代码与来源验收，不包含新的GPU授权；正式Image/Joint完成仍须后续独立Attempt及全量CPU验收。

## 零实际前向的唯一首图Claim例外

generation_v2已恢复G，但在首次像素读取时被Resize误拦截，实际G Hook为0/0。原Claim表示“可能已经生成”，不得删除或把所有未完成Claim改为可恢复。

仅在完整固定FAIL、来源、0/0实际前向、唯一首图Claim、无Proof/图像/pending、原库存只多3 Claim文件及准确身份均通过时，接受独立一次恢复能力。原Claim不改签，正常生成Proof照常绑定旧Claim，新的恢复Start/Proof保留此次例外依据；其他Claim维持拒绝。新Attempt固定v3，失败不自动再试。

预处理例外同样只限已核验原读取函数、固定图像和原Resize参数；其他非G网络继续禁止。该决定不改变攻击方法、预算、权重、原Plan或目标反馈隔离；CPU测试与资格不授权真实GPU执行。

## generation_v3共享资源许可的显式修订

用户明确授权仅将本次Runner改为与已核验Ollama PID2436556共享GPU；此前独占8192MiB为旧执行配置快照，旧工作树/配置/失败证据不改。新执行配置启动3584MiB且利用率≤10%，运行余量1024MiB，完整进程身份与来源绑定，未知进程或身份变化失败即停。保留generation_v3、原Plan、唯一首图恢复及全部算法契约；CPU开发不自动授予Codex正式GPU执行权限。

## generation_v3封存后的受害者消费边界

用户完成Image/Joint封存及独立审计后，新IRRA评估只读原Plan中的RGB/Unicode。CPU身份/动作核验先于CUDA，模型评分不反馈给攻击；真实像素SHA与IRRA ModelSpace分离。复用已有Dataset/Model Adapter、EvaluationEngine和官方Ranking，仅新增物化消费/分块缓存接线。其他Victim须独立原生编码同一产物，不借用IRRA特征。CPU测试不提升正式IRRA Full完成状态。
