# R²TUA Table1逐行引用表

完整33物理行×3数据集=99条，396个已发表指标；顺序沿论文RaSa→APTM→IRRA。第1/2/3类分别为近似同协议、带差异引用、不能放入同一公平比较表。分类范围是当前CUHK-PEDES对照；不是对其他数据集或方法的永久评价。

| Dataset | Target | Method | Type | 实际模态 | 实际epsilon | R@1 | R@5 | R@10 | mAP | G类 | Joint类 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CUHK-PEDES | RaSa | Clean (None*) | Clean | None | N/A | 76.51 | 90.29 | 94.25 | 69.38 | 2 | 2 |
| ICFG-PEDES | RaSa | Clean (None*) | Clean | None | N/A | 65.28 | 80.40 | 85.12 | 41.29 | 3 | 3 |
| RSTP-ReID | RaSa | Clean (None*) | Clean | None | N/A | 66.90 | 86.50 | 91.35 | 52.31 | 3 | 3 |
| CUHK-PEDES | RaSa | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 14.60 | 30.30 | 38.86 | 9.84 | 2 | 2 |
| ICFG-PEDES | RaSa | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 7.15 | 16.58 | 22.88 | 1.34 | 3 | 3 |
| RSTP-ReID | RaSa | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 12.45 | 29.45 | 41.50 | 7.71 | 3 | 3 |
| CUHK-PEDES | RaSa | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 10.20 | 23.68 | 32.28 | 7.60 | 2 | 2 |
| ICFG-PEDES | RaSa | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 4.19 | 11.06 | 16.07 | 0.92 | 3 | 3 |
| RSTP-ReID | RaSa | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 7.70 | 22.15 | 31.80 | 5.87 | 3 | 3 |
| CUHK-PEDES | RaSa | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 18.99 | 32.62 | 38.60 | 14.30 | 2 | 2 |
| ICFG-PEDES | RaSa | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 9.22 | 19.83 | 25.43 | 1.85 | 3 | 3 |
| RSTP-ReID | RaSa | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 8.82 | 17.87 | 23.80 | 6.02 | 3 | 3 |
| CUHK-PEDES | RaSa | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 8.48 | 18.21 | 24.90 | 6.49 | 2 | 2 |
| ICFG-PEDES | RaSa | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 11.58 | 23.57 | 30.47 | 7.68 | 3 | 3 |
| RSTP-ReID | RaSa | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 16.40 | 36.85 | 46.95 | 10.89 | 3 | 3 |
| CUHK-PEDES | RaSa | MisRanking | Untargeted | Image-only | NOT_REPORTED | 16.93 | 28.28 | 34.47 | 13.72 | 2 | 3 |
| ICFG-PEDES | RaSa | MisRanking | Untargeted | Image-only | NOT_REPORTED | 18.83 | 31.65 | 38.52 | 6.02 | 3 | 3 |
| RSTP-ReID | RaSa | MisRanking | Untargeted | Image-only | NOT_REPORTED | 16.50 | 30.40 | 39.00 | 11.18 | 3 | 3 |
| CUHK-PEDES | RaSa | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 8.61 | 13.92 | 16.96 | 5.68 | 2 | 3 |
| ICFG-PEDES | RaSa | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 8.61 | 14.91 | 18.38 | 1.44 | 3 | 3 |
| RSTP-ReID | RaSa | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 6.70 | 10.80 | 13.50 | 3.24 | 3 | 3 |
| CUHK-PEDES | RaSa | MTGA | Untargeted | Image-only | NOT_REPORTED | 18.60 | 32.60 | 40.55 | 15.06 | 2 | 3 |
| ICFG-PEDES | RaSa | MTGA | Untargeted | Image-only | NOT_REPORTED | 21.53 | 36.04 | 44.06 | 7.68 | 3 | 3 |
| RSTP-ReID | RaSa | MTGA | Untargeted | Image-only | NOT_REPORTED | 9.50 | 21.45 | 29.55 | 7.05 | 3 | 3 |
| CUHK-PEDES | RaSa | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.11 | 1.14 | 2.16 | 0.60 | 2 | 3 |
| ICFG-PEDES | RaSa | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.15 | 0.63 | 1.09 | 0.29 | 3 | 3 |
| RSTP-ReID | RaSa | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.65 | 6.20 | 11.05 | 1.74 | 3 | 3 |
| CUHK-PEDES | RaSa | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 31.26 | 58.01 | 66.49 | 14.35 | 3 | 3 |
| ICFG-PEDES | RaSa | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 34.09 | 68.03 | 80.49 | 9.19 | 3 | 3 |
| RSTP-ReID | RaSa | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 32.40 | 67.68 | 79.30 | 18.48 | 3 | 3 |
| CUHK-PEDES | RaSa | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 89.43 | 97.72 | 98.47 | 54.89 | 3 | 3 |
| ICFG-PEDES | RaSa | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 68.48 | 89.21 | 93.79 | 18.13 | 3 | 3 |
| RSTP-ReID | RaSa | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 84.60 | 96.30 | 98.30 | 45.86 | 3 | 3 |
| CUHK-PEDES | APTM | Clean (None*) | Clean | None | N/A | 76.53 | 90.04 | 94.15 | 66.91 | 2 | 2 |
| ICFG-PEDES | APTM | Clean (None*) | Clean | None | N/A | 68.51 | 82.99 | 87.56 | 41.20 | 3 | 3 |
| RSTP-ReID | APTM | Clean (None*) | Clean | None | N/A | 67.50 | 85.70 | 91.45 | 52.56 | 3 | 3 |
| CUHK-PEDES | APTM | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 8.33 | 19.36 | 25.68 | 5.42 | 2 | 2 |
| ICFG-PEDES | APTM | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 3.74 | 10.64 | 16.11 | 0.90 | 3 | 3 |
| RSTP-ReID | APTM | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 8.70 | 19.75 | 28.60 | 5.34 | 3 | 3 |
| CUHK-PEDES | APTM | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 8.53 | 18.89 | 25.71 | 5.54 | 2 | 2 |
| ICFG-PEDES | APTM | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 3.61 | 9.74 | 14.81 | 0.87 | 3 | 3 |
| RSTP-ReID | APTM | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 8.70 | 19.40 | 28.45 | 5.41 | 3 | 3 |
| CUHK-PEDES | APTM | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 32.12 | 48.75 | 55.12 | 19.12 | 2 | 2 |
| ICFG-PEDES | APTM | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 22.02 | 36.86 | 44.36 | 4.57 | 3 | 3 |
| RSTP-ReID | APTM | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 20.55 | 36.30 | 44.15 | 10.98 | 3 | 3 |
| CUHK-PEDES | APTM | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 10.15 | 20.45 | 26.22 | 6.92 | 2 | 2 |
| ICFG-PEDES | APTM | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 11.25 | 19.58 | 24.01 | 2.18 | 3 | 3 |
| RSTP-ReID | APTM | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 21.20 | 39.90 | 49.80 | 11.41 | 3 | 3 |
| CUHK-PEDES | APTM | MisRanking | Untargeted | Image-only | NOT_REPORTED | 7.17 | 12.75 | 16.08 | 4.58 | 2 | 3 |
| ICFG-PEDES | APTM | MisRanking | Untargeted | Image-only | NOT_REPORTED | 13.33 | 24.81 | 31.54 | 2.82 | 3 | 3 |
| RSTP-ReID | APTM | MisRanking | Untargeted | Image-only | NOT_REPORTED | 9.55 | 18.55 | 24.65 | 5.43 | 3 | 3 |
| CUHK-PEDES | APTM | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 8.54 | 15.63 | 20.27 | 5.89 | 2 | 3 |
| ICFG-PEDES | APTM | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 20.87 | 33.99 | 40.59 | 4.35 | 3 | 3 |
| RSTP-ReID | APTM | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 21.60 | 37.15 | 44.90 | 11.17 | 3 | 3 |
| CUHK-PEDES | APTM | MTGA | Untargeted | Image-only | NOT_REPORTED | 6.69 | 12.49 | 15.55 | 4.88 | 2 | 3 |
| ICFG-PEDES | APTM | MTGA | Untargeted | Image-only | NOT_REPORTED | 12.29 | 22.59 | 28.31 | 2.92 | 3 | 3 |
| RSTP-ReID | APTM | MTGA | Untargeted | Image-only | NOT_REPORTED | 8.05 | 17.95 | 24.00 | 5.15 | 3 | 3 |
| CUHK-PEDES | APTM | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.54 | 2.21 | 3.64 | 2.13 | 2 | 3 |
| ICFG-PEDES | APTM | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 2.51 | 6.51 | 9.68 | 0.62 | 3 | 3 |
| RSTP-ReID | APTM | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.35 | 1.45 | 4.35 | 1.19 | 3 | 3 |
| CUHK-PEDES | APTM | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 25.41 | 53.64 | 65.45 | 15.78 | 3 | 3 |
| ICFG-PEDES | APTM | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 15.31 | 40.93 | 54.72 | 3.26 | 3 | 3 |
| RSTP-ReID | APTM | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 25.00 | 57.00 | 69.30 | 14.07 | 3 | 3 |
| CUHK-PEDES | APTM | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 62.18 | 87.95 | 93.82 | 16.53 | 3 | 3 |
| ICFG-PEDES | APTM | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 45.56 | 75.10 | 83.58 | 9.46 | 3 | 3 |
| RSTP-ReID | APTM | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 42.00 | 71.90 | 81.10 | 24.70 | 3 | 3 |
| CUHK-PEDES | IRRA | Clean (None*) | Clean | None | N/A | 73.38 | 89.93 | 93.71 | 66.13 | 2 | 2 |
| ICFG-PEDES | IRRA | Clean (None*) | Clean | None | N/A | 63.46 | 80.25 | 85.82 | 38.06 | 3 | 3 |
| RSTP-ReID | IRRA | Clean (None*) | Clean | None | N/A | 60.20 | 81.30 | 88.20 | 47.17 | 3 | 3 |
| CUHK-PEDES | IRRA | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 15.55 | 35.40 | 46.22 | 12.25 | 2 | 2 |
| ICFG-PEDES | IRRA | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 5.46 | 15.98 | 23.56 | 1.68 | 3 | 3 |
| RSTP-ReID | IRRA | SGA | Untargeted | NOT_REPORTED | NOT_REPORTED | 11.45 | 30.45 | 43.40 | 8.21 | 3 | 3 |
| CUHK-PEDES | IRRA | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 7.52 | 20.78 | 28.62 | 6.78 | 2 | 2 |
| ICFG-PEDES | IRRA | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 3.78 | 11.67 | 17.66 | 1.00 | 3 | 3 |
| RSTP-ReID | IRRA | VLPTAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 9.10 | 23.25 | 33.90 | 6.22 | 3 | 3 |
| CUHK-PEDES | IRRA | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 20.53 | 38.76 | 48.16 | 15.69 | 2 | 2 |
| ICFG-PEDES | IRRA | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 9.22 | 20.11 | 26.98 | 2.57 | 3 | 3 |
| RSTP-ReID | IRRA | MFHA | Untargeted | NOT_REPORTED | NOT_REPORTED | 14.60 | 33.80 | 47.40 | 10.23 | 3 | 3 |
| CUHK-PEDES | IRRA | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 33.56 | 55.18 | 63.39 | 26.26 | 2 | 2 |
| ICFG-PEDES | IRRA | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 29.72 | 48.91 | 57.47 | 10.12 | 3 | 3 |
| RSTP-ReID | IRRA | AnyAttack | Untargeted | NOT_REPORTED | NOT_REPORTED | 24.25 | 47.15 | 60.00 | 15.83 | 3 | 3 |
| CUHK-PEDES | IRRA | MisRanking | Untargeted | Image-only | NOT_REPORTED | 15.62 | 25.68 | 30.23 | 11.60 | 2 | 3 |
| ICFG-PEDES | IRRA | MisRanking | Untargeted | Image-only | NOT_REPORTED | 14.89 | 27.37 | 33.48 | 25.24 | 3 | 3 |
| RSTP-ReID | IRRA | MisRanking | Untargeted | Image-only | NOT_REPORTED | 15.35 | 29.35 | 35.90 | 8.93 | 3 | 3 |
| CUHK-PEDES | IRRA | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 8.79 | 17.95 | 24.14 | 7.51 | 2 | 3 |
| ICFG-PEDES | IRRA | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 13.24 | 23.74 | 29.53 | 3.01 | 3 | 3 |
| RSTP-ReID | IRRA | MetaAttack | Untargeted | Image-only | NOT_REPORTED | 7.80 | 17.55 | 24.15 | 4.97 | 3 | 3 |
| CUHK-PEDES | IRRA | MTGA | Untargeted | Image-only | NOT_REPORTED | 7.49 | 13.66 | 16.93 | 5.44 | 2 | 3 |
| ICFG-PEDES | IRRA | MTGA | Untargeted | Image-only | NOT_REPORTED | 12.29 | 22.59 | 28.31 | 2.92 | 3 | 3 |
| RSTP-ReID | IRRA | MTGA | Untargeted | Image-only | NOT_REPORTED | 8.05 | 17.95 | 24.00 | 5.15 | 3 | 3 |
| CUHK-PEDES | IRRA | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 0.58 | 2.71 | 4.56 | 1.07 | 2 | 3 |
| ICFG-PEDES | IRRA | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 4.31 | 11.05 | 15.83 | 1.15 | 3 | 3 |
| RSTP-ReID | IRRA | R²TUA | Untargeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 2.55 | 8.10 | 12.65 | 2.24 | 3 | 3 |
| CUHK-PEDES | IRRA | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 12.39 | 33.34 | 46.39 | 9.00 | 3 | 3 |
| ICFG-PEDES | IRRA | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 8.26 | 26.40 | 41.09 | 2.65 | 3 | 3 |
| RSTP-ReID | IRRA | AnyAttack (Targeted) | Targeted | NOT_REPORTED | NOT_REPORTED | 21.80 | 52.90 | 68.40 | 13.85 | 3 | 3 |
| CUHK-PEDES | IRRA | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 84.10 | 98.00 | 99.10 | 52.16 | 3 | 3 |
| ICFG-PEDES | IRRA | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 48.16 | 80.42 | 88.63 | 13.15 | 3 | 3 |
| RSTP-ReID | IRRA | R²TUA (Targeted) | Targeted | Prompt-conditioned Image-only | 8/255 (R²TUA自身设置，正文p4-5) | 75.50 | 95.10 | 97.50 | 44.30 | 3 | 3 |
