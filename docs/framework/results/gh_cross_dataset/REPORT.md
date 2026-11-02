# CUHK G-v2 E60 + H-v1 E7跨数据集冻结迁移结果

FROZEN：所有Full和独立CPU审计均已PASS；无重新训练或微调。Clean/Image-only只读引用v2原结果；Text-only/Joint共用各目标新H文本和特征。

| Dataset | Victim | Mode | R@1 | R@5 | R@10 | mAP |
|---|---|---|---:|---:|---:|---:|
| RSTPReid | IRRA | Clean | 60.25 | 81.55 | 88.20 | 47.19 |
| RSTPReid | IRRA | Image-only | 28.10 | 53.15 | 64.10 | 18.83 |
| RSTPReid | IRRA | Text-only | 49.00 | 74.10 | 82.65 | 38.83 |
| RSTPReid | IRRA | Joint | 23.05 | 45.45 | 57.50 | 15.74 |
| ICFG-PEDES | IRRA | Clean | 63.43 | 80.35 | 85.77 | 38.04 |
| ICFG-PEDES | IRRA | Image-only | 25.78 | 43.71 | 52.61 | 9.29 |
| ICFG-PEDES | IRRA | Text-only | 58.24 | 75.62 | 82.01 | 34.03 |
| ICFG-PEDES | IRRA | Joint | 22.79 | 40.37 | 49.13 | 8.10 |

G SHA：`9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049`；H SHA：`487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df`。

原始完整精度CSV、来源与审计绑定：`/home/lzf/ldx/outputs/tbps-framework-gh-cross-dataset-20261102/final/`。

目标Test PID仅用于正式评分/独立CPU排名审计；H只接收原Caption，无目标反馈、配对图或标签输入。合法NOOP保留；未重跑G、图像IRRA或历史Clean/Image-only评分。
