# G-v2 + H-v1：IRRA、RaSa、APTM正式Full结果汇总

整理日期：2026-10-18（Asia/Shanghai）。只读提取三份正式Full CSV/summary、协议、执行报告与已完成CPU审计日志；没有使用Smoke数值，没有重新运行模型、攻击生成、GPU、排名或独立审计入口。

## 1. 实验范围与解释边界

- 数据：CUHK-PEDES官方Test，3074 Gallery、6156 Query；三模型Dataset Manifest、Gallery/Query完整顺序、PID/正确配对、四类输入内容SHA以及冻结G/H来源逐项一致。
- 模型排列固定为IRRA、RaSa、APTM；所有新表格和CSV的方法排列固定为Clean → H-v1 Text-only → G-v2 Image-only → G-v2 + H-v1 Joint。
- 图像攻击：冻结G-v2 Epoch10，RGB [0,1]空间L∞≤8/255；文本攻击：冻结H-v1 Epoch7/visit14336，Caption-only Unicode编辑，保留所有Query，包括40条NOOP。6116条有编辑，历史13537次字符替换均按封存来源保留。
- IRRA是攻击器训练来源模型上的**同模型评估**；RaSa、APTM是IRRA训练得到的冻结攻击产物的**跨模型迁移评估**。三者评估时均不重新运行G/H，不用目标模型反馈修改攻击，不更新参数。
- Clean：原始物化Gallery＋原Caption；H-v1 Text-only：原始物化Gallery＋冻结H文本；G-v2 Image-only：冻结G图像＋原Caption；G-v2 + H-v1 Joint：冻结G图像＋冻结H文本。
- 每个受害者采用各自原生预处理、Tokenizer和评分。Clean是各自模型在统一物化输入上的基线，不能直接当成论文原始输入设置下的Clean。固定RGB已在384×128工作空间，RaSa再做384×384 Tensor resize；不量化Adv回PIL，也不重新生成攻击。
- 当前仅固定CUHK-PEDES、固定G10/H7及三个受害者；没有多seed、置信区间或跨数据集结论。Joint优于单模态不等价于已证明SCST/属性模块的独立贡献，也不能仅据此建立因果性的协同机理。

## 2. 正式实验主表

单位：百分数（%），四位小数；越低表示受害者检索性能下降越大。原精度见同目录full_precision.csv，未用四舍五入后的数值计算DR。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP | mINP |
| --- | --- | --- | --- | --- | --- | --- |
| IRRA | Clean | 73.1481 | 89.8798 | 93.7134 | 66.0974 | 50.2777 |
| IRRA | H-v1 Text-only | 56.5789 | 75.9747 | 82.5211 | 51.4716 | 36.3625 |
| IRRA | G-v2 Image-only | 37.4431 | 60.7700 | 70.0780 | 32.2823 | 17.2166 |
| IRRA | G-v2 + H-v1 Joint | 26.5270 | 46.8324 | 57.1962 | 23.5532 | 11.8168 |
| RaSa | Clean | 76.2183 | 90.2859 | 94.0871 | 69.1841 | 54.6222 |
| RaSa | H-v1 Text-only | 39.9285 | 50.2599 | 53.4113 | 32.8586 | 21.4167 |
| RaSa | G-v2 Image-only | 42.3327 | 61.7122 | 69.5582 | 35.0470 | 19.6103 |
| RaSa | G-v2 + H-v1 Joint | 22.8720 | 34.6816 | 40.1235 | 17.2026 | 7.8045 |
| APTM | Clean | 74.6101 | 89.2625 | 93.1774 | 64.1268 | 45.6776 |
| APTM | H-v1 Text-only | 60.8025 | 78.1351 | 83.7719 | 51.8019 | 34.1501 |
| APTM | G-v2 Image-only | 50.1949 | 70.2079 | 77.1280 | 39.6620 | 20.9412 |
| APTM | G-v2 + H-v1 Joint | 38.8239 | 58.7394 | 66.6017 | 30.4903 | 14.7272 |

## 3. 相对下降率DR

DR=(Clean−Attack)/Clean×100%，各模型单独使用自己的Clean。Clean的DR统一显示“—”（原始完成文件中的Clean DR为0）；本节数字为相对下降率%，不是指标下降百分点。重新按原精度计算并核对原CSV/summary的45项非Clean DR，最大差1.4210854715202004e-14个百分点。

| Target Model | Attack Method | mAP DR (%) | R@1 DR (%) | R@5 DR (%) | R@10 DR (%) | mINP DR (%) |
| --- | --- | --- | --- | --- | --- | --- |
| IRRA | Clean | — | — | — | — | — |
| IRRA | H-v1 Text-only | 22.1278 | 22.6516 | 15.4708 | 11.9431 | 27.6766 |
| IRRA | G-v2 Image-only | 51.1596 | 48.8119 | 32.3875 | 25.2210 | 65.7570 |
| IRRA | G-v2 + H-v1 Joint | 64.3659 | 63.7353 | 47.8945 | 38.9669 | 76.4970 |
| RaSa | Clean | — | — | — | — | — |
| RaSa | H-v1 Text-only | 52.5055 | 47.6130 | 44.3325 | 43.2320 | 60.7913 |
| RaSa | G-v2 Image-only | 49.3425 | 44.4587 | 31.6481 | 26.0704 | 64.0983 |
| RaSa | G-v2 + H-v1 Joint | 75.1350 | 69.9915 | 61.5869 | 57.3550 | 85.7119 |
| APTM | Clean | — | — | — | — | — |
| APTM | H-v1 Text-only | 19.2196 | 18.5064 | 12.4659 | 10.0941 | 25.2366 |
| APTM | G-v2 Image-only | 38.1507 | 32.7237 | 21.3467 | 17.2245 | 54.1544 |
| APTM | G-v2 + H-v1 Joint | 52.4531 | 47.9643 | 34.1947 | 28.5216 | 67.7583 |

## 4. Joint相对单模态的额外下降

额外mAP下降定义为单模态mAP−Joint mAP，单位是**百分点**，与上一节相对DR区分。全部基于未舍入原值计算。

| Target Model | 相对H-v1 Text-only额外mAP下降（百分点） | 相对G-v2 Image-only额外mAP下降（百分点） |
| --- | --- | --- |
| IRRA | 27.9183 | 8.7291 |
| RaSa | 15.6560 | 17.8443 |
| APTM | 21.3116 | 9.1717 |

## 5. 结果分析

G-v2 Image-only在IRRA上的mAP从66.0974降到32.2823，DR为51.1596%；在RaSa/APTM上分别从69.1841/64.1268降到35.0470/39.6620，DR为49.3425%/38.1507%。因此在这两个真实跨模型目标上均观察到图像攻击迁移，APTM在本设置下的相对mAP损失较小。

H-v1 Text-only在IRRA/RaSa/APTM上的mAP DR分别为22.1278%/52.5055%/19.2196%。RaSa对该冻结Unicode文本攻击表现出更大的相对mAP和R@1下降。IRRA和APTM上Image-only的五项指标均低于Text-only；RaSa上Text-only的mAP与R@1更低，但Image-only的mINP更低，不能把单一指标的强弱推广到全部指标。

Joint在三个目标的五项指标上均低于两种单模态。Joint mAP为23.5532/17.2026/30.4903，mAP DR为64.3659%/75.1350%/52.4531%；R@1 DR为63.7353%/69.9915%/47.9643%。相较Image-only，Joint分别额外降低mAP 8.7291/17.8443/9.1717个百分点，满足本组正式测试中Joint优于Image-only的研究目标。

按各自Clean归一化后的Joint mAP DR和R@1 DR，本设置下RaSa下降最大、APTM最小，IRRA居中。但IRRA属于同模型，RaSa/APTM属于跨模型；不同原生预处理、Tokenizer、融合评分以及基线水平共同影响观测值。当前数据不能单独证明某一种架构或Tokenizer导致敏感性差异，也不把不同目标的绝对mAP直接解释为普适鲁棒性排序。

## 6. 原生协议与输入一致性

| Target Model | 图像/归一化 | 文本 | 原生评分 |
| --- | --- | --- | --- |
| IRRA | 384×128，CLIP均值/方差 | Byte-BPE，77 tokens | 单位化图文特征点积 |
| RaSa | 384×384 Tensor BICUBIC，CLIP均值/方差 | 官方BERT WordPiece，50 words/tokens及原pre_caption | 全局候选Top128，融合ITM正类logit；未选项−100 |
| APTM | 384×128，CUHK专用均值/方差 | 官方BERT WordPiece，56 words/tokens及原pre_caption | Top128 ITM；未选项填本行最小值；完整Query/Gallery矩阵minmax(ITM)+0.002×minmax(global) |

全部FP32/math/no_tf32、Query块128面对完整Gallery；APTM归一化依赖完整Query集合，没有逐块独立minmax。排名均保留同PID全部正例，无camera过滤。ModelSpace/Tokenizer/预处理/Score来源各自独立，没有把IRRA Embedding复用给其他目标。

共同有序Dataset Manifest digest：a338f5429b64393b8062e0a7127773d72adb5dd6832eeb89d24c2c63c2f4d9eb。
共同评测身份digest：03508e62b79feae042008f9a3c28506139addd104a3931de79c2c7f69deb680b。
共同Ranking协议digest：136a80737536827632a8c26e182b763410de3760a8f62dd5b44d16594ed17871。
共同物化Collection SHA：eafcae668dc708b4f92a8916ac7ddf68f27f9e27085f456e33584e07bbd0c4be。

## 7. Checkpoint及完成证据

本轮只读重新核验三个Victim与两个攻击器Checkpoint的实际文件SHA，不反序列化模型。三个Full的完成/执行回执、CSV/summary/result metrics一致，模型参数/buffers前后摘要一致；Full及CPU审计成功标志均存在。本轮没有重复执行CPU审计或逐个重读全部特征/Score Tensor，完整缓存审计通过状态引用原CPU审计记录。

| Target Model | Checkpoint SHA-256 |
| --- | --- |
| IRRA | 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009 |
| RaSa | bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2 |
| APTM | 521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c |

G-v2 Epoch10：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt。
SHA-256：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。

H-v1 Epoch7/visit14336：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/training/checkpoints/visit_00014336/checkpoint.pt。
SHA-256：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。

| Target Model | summary.json SHA-256 | completed Manifest SHA-256 |
| --- | --- | --- |
| IRRA | 4530f3f97156d6e9f65c07b0775588324c39679feb0f16ae66b356692b017eb0 | 10cd2ebb890225ec8bf38cb5e7bfb1505ca10a51148fb356e58b18a76eee650e |
| RaSa | 12b3a90ca761c632bdf7781e7d52b43b753ddb5e7bdf36ad53e65515d7fc37ac | 9e02d0c9ab18be84f1de381beec7182ab5561e795d307d2f9c8922e986ee4e06 |
| APTM | 8944ea0cd29e5c9f4410bf9ad7ccc65f67e3c26ad8a794e2a8af6915bca36d18 | 1c577586605588b4887c66ee56dbffaae6c1447bb68006c2b05794342679e242 |

### IRRA

- Full目录：/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/
- Full成功：/home/lzf/ldx/tmp/irra_materialized_full_gpu.log，精确标志 MATERIALIZED_IRRA_EVALUATION_PASS full。
- CPU审计：/home/lzf/ldx/tmp/irra_materialized_full_audit.log，精确标志 MATERIALIZED_IRRA_CPU_AUDIT_PASS full。
- CPU审计日志SHA-256：70d9630c0f5bc2d7d1565cba4a100d5345c97c057ea55028d1888198c6104798。
- 监督器成功：/home/lzf/ldx/tmp/irra_materialized_full_status.log，SUCCESS | IRRA_FULL_AND_CPU_AUDIT。
- 正式执行报告：/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/execution-000001/report.json，status=PASS，文件SHA-256 a620c28466d4ecdbee2c03fe60262cdb53b23d67a1156fbc02888b2bcdc77ec5。
- 模型Checkpoint：/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth。
### RaSa

- Full目录：/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/
- Full成功：/home/lzf/ldx/tmp/rasa_full_wait/full_gpu.log，精确标志 RASA_MATERIALIZED_EVALUATION_PASS full。
- CPU审计：/home/lzf/ldx/tmp/rasa_full_wait/cpu_audit.log，精确标志 RASA_MATERIALIZED_CPU_AUDIT_PASS full。
- CPU审计日志SHA-256：684d3bf2d4dc2e9937f50521608533db4fe9c930ce000c53ed4a5dd08bc83670。
- 监督器成功：/home/lzf/ldx/tmp/rasa_full_wait/launch.log，SUCCESS | RASA_FULL_AND_CPU_AUDIT_SUCCESS。
- 正式执行报告：/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/execution-000001/report.json，status=PASS，文件SHA-256 3cc96dd373be51dd54b2a2fcf65e50b421a250fe5ba33434883637e3c5e42fff。
- 模型Checkpoint：/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth。
### APTM

- Full目录：/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/
- Full成功：/home/lzf/ldx/tmp/aptm_full_wait/full.log，精确标志 APTM_MATERIALIZED_EVALUATION_PASS full。
- CPU审计：/home/lzf/ldx/tmp/aptm_full_wait/full_audit.log，精确标志 APTM_MATERIALIZED_CPU_AUDIT_PASS full。
- CPU审计日志SHA-256：03fdf0c6662c5b08d7f38862c721e84845151d911e03c33104e08ec6a4c6892c。
- 监督器成功：/home/lzf/ldx/tmp/aptm_full_wait/launch.log，SUCCESS | APTM_FULL_AND_CPU_AUDIT。
- 正式执行报告：/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/execution-000001/report.json，status=PASS，文件SHA-256 1311d1bd73da0939f0fa09265c1f641443183c59ec2f2546b9d3fb12ba7ad507。
- 模型Checkpoint：/home/lzf/ldx/checkpoints/APTM/aptm_cuhk_checkpoint_best.pth。

## 8. 正式执行统计

以下为三份Full execution-000001/report.json记录，不含Smoke，也不把GPU等待和独立CPU审计计入该elapsed_seconds。速度不作公平比较，各次资源占用环境不同。

| Target Model | 模型加载 | 图像样本前向 | 文本样本前向 | 融合pair | elapsed_seconds | GPU allocated MiB | GPU reserved MiB | Peak RSS GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IRRA | 1 | 6148 | 12312 | — | 219.2642 | 922.3789 | 1002.0000 | 2.3429 |
| RaSa | 1 | 6148 | 12312 | 3151872 | 9495.5647 | 1900.0586 | 2082.0000 | 24.4277 |
| APTM | 1 | 6148 | 12312 | 3151872 | 8560.5621 | 993.6265 | 1074.0000 | 7.4295 |

三者均单次正式Attempt、模型加载1次；图像6148=2×3074、文本12312=2×6156，因为原/攻击两类特征复用到四个组合。RaSa/APTM融合pair数均3151872=4×6156×128，与Hook计数一致。IRRA没有融合前向，“—”表示不适用。各Full中G/H加载/前向、Backward、Optimizer、JVP均0。模型eval、requires_grad=False、无参数梯度及前后摘要一致由执行记录确认。

RaSa/APTM审计日志仅有旧Tokenizer默认值FutureWarning，无异常堆栈或OOM标志，审计正常PASS。APTM执行结束快照显示Ollama已重新占用GPU；执行仍PASS，未根据资源改变模型精度、权重或方法。不能用本组elapsed直接比较模型效率。

## 9. 汇总文件与本轮核验

- metrics_4dp.csv：与主表同序，适合WPS/论文表格。
- dr_4dp.csv：与DR表同序，五项相对DR，Clean为“—”。
- full_precision.csv：原CSV完整精度指标文本和按未舍入值计算的五项DR；不覆盖原数据。
- evidence.json：原始路径、SHA、Checkpoint、成功标志、模型空间、输入一致性及原文件保护证据（本轮2682个已读取文件SHA/mtime/inode保持；完整保护清单只存服务器tmp，Git仅收录关键来源索引）。只含汇总元数据，不包含模型、特征Tensor或原始日志副本。
- 本轮核验12行/60项正式指标、45项攻击DR、3份完整Dataset/输入绑定和5份唯一Checkpoint；报告/CSV均机械按固定顺序导出，无数值插补或Smoke替代。
- 本轮只进行标准库CPU读取、SHA、标量统计和文档写入；没有模型推理、GPU查询/任务、攻击生成、排名重算或新独立审计。

正式结论：三个Full及其独立CPU审计均已完成；IRRA提供同模型攻击结果，RaSa/APTM提供固定攻击产物的跨模型迁移结果。后续研究/额外实验需另行授权。
