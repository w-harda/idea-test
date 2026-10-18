# SGA-TBPS White-box、AP-Attack B3与G-v2：三受害者正式Full对比

冻结SGA执行Commit：6f89b799922ac54f041dabd3cd8e9d444735b146。本报告在独立分支codex/tbps-sga-full-results整理真实正式Full文件；本轮仅标准库CPU读取、SHA/身份核验及表格整理，没有GPU、模型恢复、攻击生成、排名重算或重复独立缓存审计。

## 正式结果

固定CUHK-PEDES官方Test：3074 Gallery、6156 Query，多Caption、全部同PID正例、无camera过滤，Gallery/Query各自保持官方完整顺序。只攻击Gallery，全部Query保持原文；RGB像素空间L∞≤8/255。单位为百分数，两位小数，越低表示本设置下攻击越强。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| IRRA | Clean | 73.15 | 89.88 | 93.71 | 66.10 |
| IRRA | AP-Attack B3 | 46.35 | 68.18 | 76.51 | 40.43 |
| IRRA | G-v2 Image-only | 37.44 | 60.77 | 70.08 | 32.28 |
| IRRA | SGA-TBPS White-box | 0.49 | 1.56 | 2.34 | 0.54 |
| RaSa | Clean | 76.22 | 90.29 | 94.09 | 69.18 |
| RaSa | AP-Attack B3 | 46.86 | 66.60 | 74.43 | 39.99 |
| RaSa | G-v2 Image-only | 42.33 | 61.71 | 69.56 | 35.05 |
| RaSa | SGA-TBPS White-box | 1.61 | 4.16 | 5.93 | 1.46 |
| APTM | Clean | 74.61 | 89.26 | 93.18 | 64.13 |
| APTM | AP-Attack B3 | 52.44 | 73.02 | 80.26 | 42.39 |
| APTM | G-v2 Image-only | 50.19 | 70.21 | 77.13 | 39.66 |
| APTM | SGA-TBPS White-box | 1.71 | 3.95 | 6.17 | 1.39 |

[metrics_2dp.csv](metrics_2dp.csv)可直接用于论文/WPS/PPT；[full_precision.csv](full_precision.csv)保留原始CSV四项指标的完整精度。方法固定Clean → AP-Attack B3 → G-v2 Image-only → SGA-TBPS White-box，模型固定IRRA → RaSa → APTM；没有混入10+10 Smoke结果。

## 实验分析与权限区别

- **IRRA**：SGA使R@1降至0.49、mAP降至0.54；相较G-v2，分别进一步降低36.96、31.74个百分点；相较B3，分别降低45.86、39.89个百分点。
- **RaSa**：SGA使R@1降至1.61、mAP降至1.46；相较G-v2，分别进一步降低40.72、33.59个百分点；相较B3，分别降低45.26、38.54个百分点。
- **APTM**：SGA使R@1降至1.71、mAP降至1.39；相较G-v2，分别进一步降低48.49、38.27个百分点；相较B3，分别降低50.73、41.00个百分点。

在三个模型上，SGA的四项检索指标都低于B3和G-v2；不存在本表某一目标上SGA更弱的情况。两个冻结生成器之间，G-v2四项指标在三个模型上均低于B3。这些判断仅适用于本次固定权重、输入及协议。

从SGA攻击后剩余R@1和mAP看，IRRA最低，表现出更明显的检索破坏。RaSa与APTM无法得到一致的敏感性排序：APTM的R@1略高于RaSa，但mAP略低。不同目标的Clean、原生结构和白盒优化目标不同，这一观察不能直接推广为模型固有鲁棒性排名。

**SGA与两个生成器不是同权限、同推理成本或同优化预算下的严格公平对比。** SGA对每个目标单独生成3074张图，可以读取该目标参数、输入梯度，以及每张Test Gallery的原始配对Caption集合；三目标各自执行10轮、多尺度反传。G-v2/B3为冻结IRRA训练来源生成器：IRRA属于同模型评估，RaSa/APTM属于不使用目标反馈的跨模型迁移。不能把SGA的RaSa/APTM结果也标成IRRA→目标迁移。

B3与G-v2训练规模和监督也不相同：B3为34054张Train图像、60 Epoch、ReID/Semantic联合训练；G-v2为2048内部Train、128内部Validation、10 Epoch及文本/属性/JVP监督，其评分参考还涉及34054张clean train图像。训练访问范围及辅助来源差异见[已审计B3报告](../b3_three_model_full_20261018/REPORT.md)。本表展示相同Test与目标原生评估协议下的结果，不能单独据此宣称等成本算法优势。

## SGA算法和实验协议限制

- 官方SGA源码固定f49e6926cd13148e49bdbc2c3e537660e096230c；当前是**image-only TBPS适配**，保留官方集合引导、五尺度增强、噪声、随机起点、10轮及mean-abs/sign更新，未实现原论文完整图文交替扰动。不是普通单尺度PGD换名。
- 官方默认epsilon=2/255；统一实验改为8/255，step_size保持0.5/255。固定尺度0.5/0.75/1.25/1.5加原图尺度，Gaussian std0.05、BICUBIC缩放/逆缩放。FP32严格端点修正只处理舍入，不放宽预算。
- Caption集合为同一原图paired_image_id对应的全部原文，按官方顺序；不扩展成同PID其他图像Caption，不按攻击效果筛选，最终每张Gallery只产生一个版本。
- IRRA优化负的原生单位化图文dot集合和；RaSa/APTM优化各自原生ITM正类logit的负集合和。**APTM是局部ITM可微代理，而非最终Top128＋完整集合minmax＋0.002 global排名公式的全链导数**；RaSa也未对离散候选选择求导。
- 最终评估分别复用三个目标的原生预处理、Tokenizer、候选/融合评分和官方检索定义，没有把IRRA余弦评分套到融合目标。Clean是统一384×128 FP32物化工作空间下的目标基线，不能直接等同论文原始文件预处理报告的Clean。
- B3原像素256×128扰动按已冻结bilinear映射接入同一384×128工作空间；G-v2和SGA直接使用384×128。这是已有显式适配差异，未为结果比较重新生成B3或调整目标输入。
- 固定seed42派生每图随机种子；为共享GPU逐尺度计算梯度并经官方增强VJP累加。当前Torch2.5.1+cu124/Torchvision0.20.1/Transformers4.44.2与官方原requirements不同，Tensor Resize默认antialias等可能存在版本差异。现有同环境CPU算法对照不代表原论文环境bitwise等价。
- 单一固定seed/Checkpoint/数据集，没有多seed置信区间、同优化预算对齐或全原论文多模态复现；本轮不据Test结果调参。

## Full完成与独立CPU审计证据

| Target Model | Full执行 | 独立CPU审计 | 模型加载 | 视觉样本前向 | 融合pair | 耗时 (min) | 峰值allocated/reserved (MiB) | Peak RSS (GiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| IRRA | PASS | PASS | 1 | 156774 | 0 | 37.44 | 930.23 / 1064.00 | 2.33 |
| RaSa | PASS | PASS | 1 | 156774 | 1095768 | 123.85 | 2536.28 / 2570.00 | 18.50 |
| APTM | PASS | PASS | 1 | 156774 | 1095768 | 123.56 | 1131.82 / 1300.00 | 5.23 |

每目标视觉样本前向156774＝153700次攻击（3074×10×5）＋3074次最终SGA像素编码。文本神经前向0：攻击指导和评估复用严格同目标空间、官方身份和顺序的Clean文本表示。RaSa/APTM每目标融合pair1095768＝307800次攻击＋787968次原生最终候选评分。每目标输入autograd.grad调用184440；这些是白盒输入优化，受害者Optimizer为0、参数未更新。G/H加载、前向和属性JVP为0。

三个受害者参数/buffers前后摘要exact，eval、requires_grad=False、参数梯度为空。各目标3074个图像单元、3074个Claim、30740条轮记录完整；元数据中所有轮次梯度有限非零，最大累计L∞=0.03137254901960784≤8/255，无pending。真实像素范围、Tensor内容SHA及文件SHA完整校验由已完成的独立CPU审计证明；本轮只读核对审计日志和元数据，没有再次反序列化Tensor。

监督器状态/home/lzf/ldx/tmp/sga_whitebox_full_v2/phase=SUCCESS，status.log记录：
~~~text
2026-10-18T22:19:11+08:00 | SUCCESS
SUCCESS | SGA_THREE_MODEL_FULL_AND_CPU_AUDIT
~~~
从17:27:02启动至22:19:11完成审计，总墙钟时间4小时52分09秒。表中execution/report.json耗时不含各目标CPU来源预检及链外审计，不能与总墙钟直接混用。

正式SGA证据（model为irra/rasa/aptm）：
- /home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/full/<model>/completed/{metrics.csv,metrics_2dp.csv,summary.json,manifest.json,completed.json}
- 同一目标execution/report.json、protocol/protocol.json、pixels/protocol/protocol.json及pixels/collection/collection.json。
- /home/lzf/ldx/tmp/sga_whitebox_full_v2/<model>-full.log：SGA_TBPS_EVALUATION_PASS full <model>。
- 同目录<model>-full-audit.log：SGA_TBPS_CPU_AUDIT_PASS full <model>。

Clean/B3/G-v2均来自此前已审计正式Full，不重新编码控制组。B3来源/home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation/full/<model>/completed/，Full和CPU审计日志在/home/lzf/ldx/tmp/b3_shared_run/logs/。Clean/G-v2进一步对照/home/lzf/ldx/outputs/tbps-framework-<model>-materialized-evaluation/full/completed/summary.json，24项控制数值exact；对应原物化Full的CPU审计也已只读确认PASS。

## 数据、Checkpoint与冻结来源

官方annotation：/home/lzf/TBPS/Datasets/CUHK-PEDES/reid_raw.json，SHA-256 fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0。标准库独立读取annotation逐条核对原始Caption、caption_index、PID、配对和Gallery/Query顺序，并核验3074原图文件SHA。目录名train_query/test_query不用于推断official split，split以annotation为准。

同一目标SGA/B3/历史G-v2的Manifest和ModelSpace逐字段exact，涵盖Checkpoint、Tokenizer、预处理、精度及评分协议；不同目标保持独立空间。三目标SGA的原RGB内容SHA序列exact，各自白盒生成的Adv内容独立，不跨目标复用。

冻结配置：/home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/configs/framework/sga_whitebox.server.json，SHA-256 0ace2d29a339ea02f55e5152bfc7e5ef04870865903cf54beb648ede40c2f410。78项运行来源和18项官方文件SHA一致，原工作树HEAD保持6f89b799922ac54f041dabd3cd8e9d444735b146且干净。

- IRRA：/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth；SHA-256：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
- RaSa：/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth；SHA-256：bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2。
- APTM：/home/lzf/ldx/checkpoints/APTM/aptm_cuhk_checkpoint_best.pth；SHA-256：521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c。

- B3 E60：/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10/G_CUHK_IRRA_AP.pth.tar，SHA-256 c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb。
- G-v2 E10：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt，SHA-256 f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。

五份Checkpoint实际SHA以普通文件流核验，无torch.load或模型构造。原始SGA首次Smoke FAIL及全部旧正式结果保留，没有清理或改签。具体结果/执行/审计/配置来源及305项关键文件索引见[evidence.json](evidence.json)。

## 本轮CPU核验与交付边界

- 12行48项指标；原精度CSV、两位小数CSV、summary与SGA实际ranking结果JSON逐项一致，24项历史控制数值exact。
- 三模型完整官方Test身份/顺序、独立Checkpoint/ModelSpace、全部SGA图像/Claim元数据、Caption集合及10轮记录核对通过；Full与独立CPU审计成功证据一致。
- 本轮实际读取的58711份文件，前后SHA、size、mtime、inode保持；此范围是实际读取的源码、annotation/原图、Checkpoint、协议、元数据及日志，不冒充未重新读取的大型像素/特征Tensor全审计。
- 未启动GPU、模型前向、攻击、Full/Smoke/独立缓存审计或排名重算；没有计算相对下降率，也没有新增检索指标。
- 仅发布本报告、两份四指标CSV、必要来源索引以及统一状态文档；不发布模型、Tensor、缓存、原始日志或outputs。

三模型正式Full及独立CPU审计均PASS，本次结果可标记为正式完成。下一Baseline VLPTAttack（ECCV2024/DRA）需后续明确授权；本轮没有开始实现或执行。
