# AnyAttack统一预算8/255：CUHK三模型正式Full

三组3074 Gallery / 6156原Query Full及各自CPU审计全部PASS。完成时间：2026-11-09T13:36:21.994688+08:00。

图像实际L∞≤8/255，FP32向内投影，审计容差0。文本上限每条Query1词；用户确认AnyAttack没有文本分支就不另加，本组实际0替换，Image-only。

| Victim | 输入 | R@1 (%) | R@5 (%) | R@10 (%) | mAP (%) |
|---|---|---:|---:|---:|---:|
| IRRA | Clean | 73.148148 | 89.879791 | 93.713448 | 66.097443 |
| IRRA | Image-only | 68.745941 | 86.744644 | 91.877838 | 61.169327 |
| RaSa | Clean | 76.218323 | 90.285896 | 94.087067 | 69.184067 |
| RaSa | Image-only | 68.518517 | 85.136452 | 90.042236 | 60.068695 |
| APTM | Clean | 74.610138 | 89.262512 | 93.177391 | 64.126839 |
| APTM | Image-only | 68.144897 | 85.233917 | 90.497078 | 57.634605 |

实际double测量最大L∞=0.031372549019608，上限8/255=0.031372549019608，审计容差0。

原官方coco_cos.pt、CLIP ViT-B/32、原始Decoder噪声和embedding保持；从旧封存raw_noise重新按8/255裁剪和严格投影。新增Decoder/CLIP加载与前向0，训练/微调/Backward/Optimizer/JVP均0。

原16/255数据完整保留为补充实验，不填入本表。三模型共用同一8/255攻击集合，固定异PID参考映射的Test PID访问条件仍需披露。原生RaSa Top128融合及APTM完整Query/Gallery归一化保持；所有模型权重/buffers前后exact。

[完整协议](../../ANYATTACK_EPS8.md)。执行源码Commit `acc93c536ab72acf4def747754a2601ba9a7aaf8`。状态：SUCCESS_ALL_THREE_FULL_CPU_AUDITED。
