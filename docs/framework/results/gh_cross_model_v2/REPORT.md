# Cooperative、原Push-pull与图像增强Push-pull：IRRA训练、RaSa/APTM迁移评测

六组CUHK-PEDES完整3074 Gallery／6156 Query正式评测与独立CPU审计通过。固定各版Round60配对权重，复用原IRRA正式Test封存的FP32攻击RGB与Unicode文本。

图像预算相对384×128的原始IRRA输入严格L∞≤8/255；目标模型随后执行自己的Tensor预处理，不将插值后的数值另称相同像素空间预算。文本沿用原Natural-K，每个原词最多替换1个Unicode字符。未对目标模型训练、微调或获取攻击梯度。

RaSa保留Top-128原生ITM融合；APTM保留Top-128融合与完整Query/Gallery矩阵归一化。Clean与Joint均独立编码和评分。

| 方法 | 目标模型 | 模式 | R@1 | R@5 | R@10 | mAP |
|---|---|---|---:|---:|---:|---:|
| cooperative | rasa | Clean | 76.22 | 90.29 | 94.09 | 69.18 |
| cooperative | rasa | Joint | 17.45 | 27.57 | 31.87 | 12.34 |
| cooperative | aptm | Clean | 74.61 | 89.26 | 93.18 | 64.13 |
| cooperative | aptm | Joint | 34.52 | 52.88 | 60.12 | 25.61 |
| attraction | rasa | Clean | 76.22 | 90.29 | 94.09 | 69.18 |
| attraction | rasa | Joint | 12.36 | 21.09 | 25.02 | 8.70 |
| attraction | aptm | Clean | 74.61 | 89.26 | 93.18 | 64.13 |
| attraction | aptm | Joint | 26.32 | 44.46 | 52.19 | 20.02 |
| augmentation | rasa | Clean | 76.22 | 90.29 | 94.09 | 69.18 |
| augmentation | rasa | Joint | 12.95 | 21.51 | 25.99 | 9.01 |
| augmentation | aptm | Clean | 74.61 | 89.26 | 93.18 | 64.13 |
| augmentation | aptm | Joint | 22.22 | 37.77 | 45.22 | 16.37 |

数值单位%，Joint越低表示攻击越强。Round60在运行前固定，未按目标Test效果选Checkpoint。原Cooperative报告阶段FAIL保留，本迁移实验不重试该流水线。

原精度及来源见METRICS.csv与SOURCE_EVIDENCE.json。权重、原始攻击产物、缓存及日志只保留在服务器。
