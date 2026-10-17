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
