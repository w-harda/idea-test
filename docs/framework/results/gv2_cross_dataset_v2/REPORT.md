# G-v2 E60：CUHK → RSTPReid / ICFG-PEDES 跨数据集迁移

两个真实Smoke、两个Smoke独立CPU审计、两个正式Full评分和两个Full独立CPU重新排名审计全部PASS。

只有冻结的CUHK训练G-v2 E60；ImageOnlyGenerator(width32, blocks4)，原generate_once与8/255严格FP32 RGB投影不变。每张目标图库图重新一次G前向；不复用CUHK对抗图或扰动，不读目标PID/Caption作为G输入，无Backward/Optimizer/JVP，无目标域训练或微调，无H/文本攻击。

目标Victim复用C-PGC/SEP已验证的各数据集官方IRRA加载、Tokenizer与原生convert_weights混合参数/FP32输出特征、FP32 dot评分；G保持原fp32/math/no_tf32。该精度身份明确记录，未静默切换为CUHK权重或其他评分空间。Clean在当前协议下重新编码和计算，没有复制旧数值。

本实验同时改变目标数据域和IRRA checkpoint；不是只更换测试图像而Victim权重不变，也不证明多seed显著性或所有跨域泛化。E60由用户预先指定，不以目标Test结果选模型。

## 正式指标

单位为百分数，不另计算下降率。完整固定共享adversarial gallery；全部原始Test Caption、全部同PID正例、无camera过滤。

| Dataset | Victim | Mode | R@1 | R@5 | R@10 | mAP |
|---|---|---|---:|---:|---:|---:|
| RSTPReid | IRRA | Clean | 60.25 | 81.55 | 88.20 | 47.19 |
| RSTPReid | IRRA | Image-only (G-v2 E60) | 28.10 | 53.15 | 64.10 | 18.83 |
| ICFG-PEDES | IRRA | Clean | 63.43 | 80.35 | 85.77 | 38.04 |
| ICFG-PEDES | IRRA | Image-only (G-v2 E60) | 25.78 | 43.71 | 52.61 | 9.29 |

RSTP：1000 Gallery × 2000 Query；ICFG：19848 × 19848，完整集合、官方顺序和原始文本均保持。各组FP32原图/对抗图/有效扰动、特征、完整评分矩阵、全部排名索引、不可变回执与独立CPU审计保存在服务器输出，未向Git提交大型资产。

## 来源与运行

参考执行Commit：6021f3bfdec181785d1baccd7ce4f80a1428b069；本轮执行Commit：618cc19f6e7f42522b9057ce5e3834a7dab81bc8。

G checkpoint：`/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0060/checkpoint.pt`，SHA `9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049`。

输出根：`/home/lzf/ldx/outputs/tbps-framework-gv2-cross-dataset-v2-20261102`。控制与逐阶段日志：`/home/lzf/ldx/tmp/gv2_cross_dataset_v2_20261102`。

共享GPU观察模式，无固定最低空闲显存门槛；RAM/磁盘/inode检查保留，无精度或Batch Size降级，不干预外部C-PGC/SEP进程。独立nohup+setsid监督器单次启动，阶段失败保留并停止，不自动重试。

详细权重SHA、manifest/Tokenizer身份、真实调用数、原始指标、预算、审计SHA和历史保护计数见SOURCE_EVIDENCE.json；METRICS.csv保留原精度，METRICS_2DP.csv仅作展示舍入。

## v2审计边界修复与回归

基于1ed89e4，CPU审计仅允许无参数/缓冲区/子模块/Hook的精确Resize类型处理CPU图像；ToTensor走原纯CPU路径。其余nn.Module神经前向、CUDA、Backward、真实Optimizer step及两种JVP/forward AD入口仍禁止，异常后完整恢复。G、generate_once、8/255、IRRA Adapter、Tokenizer、打分和排名算法未改。

原112+新增29共141个CPU用例PASS；真实v1 RSTP Artifact仅在测试目录隔离副本中验证完整审计，旧FAIL未重写。新Attempt重新生成两目标Smoke与全部Full，不复用旧失败的生成/评分作为新执行。原v1输出/控制264文件SHA与stat再次逐项核对一致，仍保留FAIL。
