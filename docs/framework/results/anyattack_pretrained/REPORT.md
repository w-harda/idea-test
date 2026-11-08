# AnyAttack官方预训练Decoder：CUHK-PEDES三模型Full结果

任务开始：2026-11-08 21:01:09；三组Full完成：2026-11-09 00:02:35（Asia/Shanghai），完整执行链约181.43分钟。

三组正式Full及各自独立CPU审计全部PASS；3074 Gallery / 6156原Query。官方coco_cos.pt + CLIP ViT-B/32固定权重，无TBPS训练或微调，无反向/Optimizer/JVP，无Victim反馈。三模型使用同一图像攻击集合。

| Victim | 输入 | R@1 (%) | R@5 (%) | R@10 (%) | mAP (%) |
|---|---|---:|---:|---:|---:|
| IRRA | Clean | 73.148148 | 89.879791 | 93.713448 | 66.097443 |
| IRRA | Image-only | 58.625732 | 78.947372 | 84.957764 | 50.404858 |
| RaSa | Clean | 76.218323 | 90.285896 | 94.087067 | 69.184067 |
| RaSa | Image-only | 53.784924 | 73.781677 | 80.279404 | 44.916290 |
| APTM | Clean | 74.610138 | 89.262512 | 93.177391 | 64.126839 |
| APTM | Image-only | 58.219624 | 78.183884 | 84.340485 | 47.629734 |

名义预算16/255，原始文本不变；参考图由固定异PID映射选定，Test PID仅用于该映射。保留官方CLIP原始特征及Decoder，通过CPU可重放的噪声尺寸桥接适配384×128图像。该信息条件和预算不同于Idea/SGA，不作等预算优劣结论。

## 审计与时间

两组生成（Smoke、Full）与六组模型评分分别在独立CPU进程审计。全部Manifest/文件SHA、来源、模型参数/buffers不变、完整原始文本、异PID参考一一映射、像素桥接/预算、完整分数重排名均核验。RaSa保留Top128融合；APTM保留全Query/Gallery归一化；没有以Smoke代Full。

| 阶段 | 范围 | 秒 | peak RSS MiB |
|---|---|---:|---:|
| generate generation | smoke | 7.307 | 2015.23 |
| audit-generation generation_cpu_audit | smoke | 0.625 | 660.29 |
| evaluate irra | smoke | 21.341 | 2369.84 |
| audit-evaluation irra_cpu_audit | smoke | 3.253 | 648.21 |
| evaluate rasa | smoke | 35.072 | 4143.17 |
| audit-evaluation rasa_cpu_audit | smoke | 4.427 | 647.98 |
| evaluate aptm | smoke | 19.452 | 2392.88 |
| audit-evaluation aptm_cpu_audit | smoke | 2.386 | 648.19 |
| generate generation | full | 425.580 | 2014.48 |
| audit-generation generation_cpu_audit | full | 34.653 | 668.79 |
| evaluate irra | full | 394.297 | 2374.33 |
| audit-evaluation irra_cpu_audit | full | 7.287 | 1498.37 |
| evaluate rasa | full | 5580.061 | 23135.23 |
| audit-evaluation rasa_cpu_audit | full | 66.188 | 1495.32 |
| evaluate aptm | full | 4159.596 | 5866.26 |
| audit-evaluation aptm_cpu_audit | full | 8.167 | 1497.42 |

Formal生成耗时425.580秒，实际double测量最大L∞=0.062745124102，名义16/255=0.062745098039，超出名义值的浮点端点差=2.60624231085e-08。未把FP32端点舍入伪称为严格实数零误差。

## 与R²TUA表1的关系

R²TUA公开表1的AnyAttack（CUHK）R@1/mAP为RaSa8.48/6.49、APTM10.15/6.92、IRRA33.56/26.26。其表5还记录AnyAttack训练16h、攻击10.8ms，故3090环境不能证明没有训练。本次遵守用户免训练要求，直接迁移官方COCO Decoder；R²TUA精确Decoder身份/微调/适配代码尚未在公开来源中核实，不能称本次是其表1精确复现。三数据集/模型分数不同本身不能证明分别重训。

[完整协议及论文/官方代码来源](../../ANYATTACK_PRETRAINED.md)。运行源码冻结Commit `536d54a77fec8ee2e7f60f57439e2c50c2652f72`；后续报告发布仅增加文档/独立报告辅助，未改数值源码。

控制目录：`/home/lzf/ldx/tmp/anyattack_pretrained_20261108`；`status.json=SUCCESS_ALL_THREE_FULL_CPU_AUDITED`。原G/H、SGA及其他历史资产保持。

收尾独立只读校验再次确认：官方Decoder/CLIP、三Victim源码/Checkpoint/Tokenizer、164份执行源码均保持；当前完整数据集身份与三组结果一致，同一攻击集合SHA保持。无模型加载、CUDA未初始化；源校验耗时12.924秒。校验回执已收入SOURCE_EVIDENCE.json。
