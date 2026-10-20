# ADR-0013：固定训练轮数消融的RGB变体与历史结果引用

日期：2026-10-20。决定：采用CPU接线；真实Smoke和正式Full分别验收/授权。

## 背景

六Checkpoint共享原G架构和算法，但新扩展Checkpoint的schema与原G10不同。
旧MaterializationPlan/Plugin故意固定G10，直接修改其SHA或能力会破坏历史来源。
研究变量只能是G训练Epoch，不能同时改变H、RGB输入桥、Victim或排名协议。

## 决定

- 旧Plan、MaterializedUnit及历史来源只读。新增gv2-epoch-rgb-variant-1引用旧Plan/原RGB、新Checkpoint及完整顺序。
  不将新Epoch改签成原Plan，不放宽旧GV2Plugin的G10保护。
- 新schema仅做外部读取/验证/提取generator，严格装入原ImageOnlyGenerator并调用原generate_once；
  不重新实现网络、投影、损失或训练，不恢复Optimizer进行更新。
- 新图像保存原FP32 original/adversarial/effective_delta、实际像素和文件SHA；先Claim、单次前向、原子发布与读回；
  未完成Claim/pending/失败证据拒绝自动重跑。不得用PIL量化或特征替代真实RGB。
- 原H7文本及完整动作不重新生成；其原生特征只能在对应Victim ModelSpace中复用。
  复用历史Full必须同时证明Raw输入、顺序、Checkpoint、原生预处理/Tokenizer/评分与文件SHA。
- 新Epoch所有Adv图像编码、Top-K候选/ITM和最终排名必须重新计算，不能复用G10 Score。
  G生成先于Victim评分，二者分进程，无反馈通道；同一变体供三个Target。
- 默认仅CPU预检。Smoke、Full图像生成、Full评分分别显式授权。
  未来整链脚本还要求真实Smoke及三份新CPU审计完成回执；CPU Fixture/旧Full不可替代。
- 不改变原生IRRA dot、RaSa ITM和APTM全集合归一化。公开四指标，底层Engine原schema保持。

## 验证边界

本次六点CPU严格结构/SHA与旧Full来源复用通过；真实Smoke因外部活跃GPU任务未开始，无新Attempt或GPU前向。
新加载器G10像素再现、G60生成及三Victim实际链路仍PENDING，不能据CPU结果宣称Full启动就绪。
未来只评估预设六点，不根据Test结果选择训练长度或修改攻击器。

## 共享策略修订及固定环境Smoke验收

用户明确允许GPU并行共享。仅资源准入改为6144MiB空闲、利用率仅监控，
RAM/磁盘/源SHA、算法与完整原生评分定义均保留；不调整batch、精度或预算以降低资源。
152498f实际运行一次固定10+10的G10/G60生成和三Victim评分，三份独立CPU审计通过，
G10像素SHA exact、G60严格8/255、全模型冻结，发布GV2_SIX_NODE_SMOKE_PASS。
旧资源阻塞阶段无Attempt，不改写旧证据；本次仅一个smoke_v1，无重试。
此验收满足技术Smoke门禁，不是Top128 Full或六点训练轮数趋势结论；
正式Full仍需用户最终授权，不依据小样本数值调整任何配置。
