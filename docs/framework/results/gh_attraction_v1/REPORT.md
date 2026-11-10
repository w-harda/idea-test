# GH-Adversarial-Attraction-v1：从零60轮与六节点Joint正式Test

**60轮训练、独立CPU审计与六组CUHK-PEDES × IRRA Joint Test全部PASS。**

G/H均从原始seed42初始化，双Adam状态从空开始；未加载历史G10/G60、H7/H10/H60或旧动态Bank。IRRA、Reader、属性空间及固定clean资源沿用。双Optimizer仍无Scheduler，数值权重在训练前固定。

G/H优化共同Train候选KL：靠近当前对方攻击表示，远离自身干净表示。G保留K+1、属性Local/Retain及8/255；H保留原三头SCST和Natural-K每词最多1字符，原R_sim/R_rank只作诊断。每轮固定2048 Train顺序，先G后完整当前G Bank再H，无Validation优化/评测。

第10/20/30/40/50/60轮分别导出本次G/H；它们不是历史同名Checkpoint。每十轮经前缀CPU审计后执行该节点Test，再恢复下一十轮，3074 Gallery/6156原Query、全部同PID正例、FP32 RGB、图像/文本各逐样本原生编码、Query chunk128。各组独立生成自己的图像和文本，仅Joint评分，没有额外模态/模型/数据集实验。

| 训练轮次 | G | H | R@1 | R@5 | R@10 | mAP |
|---|---|---|---:|---:|---:|---:|
| 10 | G10 | H10 | 17.87 | 32.52 | 40.76 | 14.98 |
| 20 | G20 | H20 | 15.20 | 28.02 | 35.54 | 12.36 |
| 30 | G30 | H30 | 14.13 | 26.20 | 33.79 | 11.44 |
| 40 | G40 | H40 | 13.34 | 24.87 | 32.49 | 10.88 |
| 50 | G50 | H50 | 13.53 | 25.63 | 32.52 | 10.85 |
| 60 | G60 | H60 | 14.72 | 26.95 | 33.61 | 11.36 |

数值单位%。越低攻击越有效；原精度见METRICS.csv。

六个预设节点R@1并非单调不升，mAP并非单调不升。仅一条seed42轨迹，不据此声称统计显著，也不额外用Test选点改变方法。

![R@1](R1_by_round.png)

![mAP](mAP_by_round.png)

每10轮的G/H权重不可变；每轮保存最新完整双模型/Adam/RNG/Bank身份边界并原子切换，旧新产物按退休回执清理，保留全部小型账本。真实失败不自动重试。

来源、预算和原生独立排名审计见SOURCE_EVIDENCE.json及服务器各组cpu_audit。源码/运行说明见[GH_ATTRACTION_V1.md](../../GH_ATTRACTION_V1.md)。权重、Tensor、原始数据和日志不发布Git。
