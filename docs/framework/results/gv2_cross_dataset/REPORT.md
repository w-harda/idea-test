# G-v2 E60跨数据集迁移：失败停止报告

最终状态：**FAIL_STOPPED_NO_RETRY**。没有Full正式指标，不宣称整项任务SUCCESS。

## 真实执行状态

| 阶段 | RSTPReid | ICFG-PEDES |
|---|---|---|
| CPU Preflight | PASS | PASS |
| 真实G-v2 E60图像生成 | 完成，10张，10次G前向 | NOT_RUN |
| 真实IRRA Clean/Image-only评分 | 完成，10 Gallery / 10 Query | NOT_RUN |
| 独立CPU Smoke审计 | FAIL | NOT_RUN |
| 完整Full生成/评分/独立审计 | NOT_RUN | NOT_RUN |

RSTP生成与评分的阶段完成回执不等于Smoke整体PASS。两个真实Smoke及对应CPU审计未全部PASS，因此监督器未进入Full。用户要求任何真实阶段失败保留现场并停止；本次严格停止，没有重试、覆盖、删除旧Attempt、改精度/Batch Size或启动后续链路。

## 失败定位

失败在 `src/tbps/transfer/gv2_execute.py:153` 的独立CPU原图像素复核。新增 `no_backward(cpu=True)` 把所有 `torch.nn.Module._call_impl` 调用都禁止；现有原生预处理中的纯CPU torchvision `Resize` 同样继承nn.Module，被误当成神经模型前向拦截。此护栏实现过宽，属于本轮新增审计接线错误。

原始报错：`RuntimeError: inference/audit forbids CUDA or neural forward in CPU mode; backward/optimizer/JVP always forbidden`。

它不是已观测到的G/IRRA训练、CUDA OOM、非有限值或扰动预算错误。但原图像素及完整评分重新排名审计未执行完，不能因此将FAIL改为PASS。未修改该失败源码、放宽门禁或重跑审计。后续若修复，应在新的授权和独立版本中让纯CPU图像预处理通过，同时继续禁止真实模型/CUDA/Backward/JVP；先补覆盖该边界的CPU端到端回归，保留本次原始FAIL。

## 模型与协议

G为CUHK训练的冻结E60，真实Checkpoint SHA `9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049`。严格恢复ImageOnlyGenerator(width32, blocks4)，1,377,411参数，原generate_once、384×128 FP32 RGB和8/255投影保持；原G/投影/预处理/Engine/排名文件与参考Commit6021f3bfdec181785d1baccd7ce4f80a1428b069逐字节一致。

目标IRRA使用各数据集官方权重；预检均核验SHA和分类头，不误用CUHK权重。RSTP实际恢复官方专用IRRA，真实执行20个image编码样本和10个原始Caption编码样本；冻结参数/buffers前后SHA一致。G阶段10次前向、无Victim输入或反馈；Backward/Optimizer/JVP均0。原生IRRA沿用C-PGC/SEP已验证convert_weights参数与FP32输出特征/打分；G保留fp32/math/no_tf32。

本实验设计同时变化目标数据域与IRRA checkpoint，不能描述成只更换测试图片而权重相同。没有目标域训练/微调、H/文本攻击或Test选模型。本次实际完成的RSTP小样本读数只在SOURCE_EVIDENCE.json中作为未审计Smoke记录，不作为正式性能结论。

## 产物与发布

`METRICS.csv`、`METRICS_2DP.csv`只有列名、0条正式结果；不得用小样本Smoke值填Full表格。`SOURCE_EVIDENCE.json`含真实阶段回执、两目标来源、调用数、FAIL日志SHA和历史保护证据。

独立工作树 `/home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-20261102`，分支codex/gv2-cross-dataset；执行Commit `3277fa12935501ed22adad61ce3a3a44ef058318`。完整失败现场 `/home/lzf/ldx/outputs/tbps-framework-gv2-cross-dataset-20261102`；日志与控制 `/home/lzf/ldx/tmp/gv2_cross_dataset_20261102`。大型数据/权重/对抗像素/矩阵不进入Git，代码/配置及本报告/小型来源摘要发布。

Supervisor PID/PGID/SID为678709，PPID1，无TTY；失败后已退出。只结束过本次启动wrapper父shell以完成PPID1脱离，未停止监督器或任何外部任务；没有重复Supervisor。GPU资源观察模式，无最低空闲显存门槛、没有等待外部进程退出。

## CPU验证与限制

112个不同CPU用例PASS；主工作树111 passed、1个历史绝对路径测试deselected，该用例在原冻结工作树单独1 passed。它们未覆盖“禁止nn.Module神经前向但允许Resize CPU预处理”组合，未能提前捕获本次集成错误；不把CPU回归通过等同真实Smoke审计通过。

停止后只读复核123738份历史CUHK Full/Checkpoint文件stat保持，G与两目标IRRA/annotation SHA再次核验通过；执行源码未改。原始数据、历史Full、权重及其他C-PGC/SEP任务均未修改。没有在失败后重跑任何GPU或独立审计。
