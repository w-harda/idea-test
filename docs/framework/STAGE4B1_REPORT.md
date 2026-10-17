# Stage 4B-1验收报告：历史Joint Full特征只读桥

## 结论与范围

**METADATA_VERIFIED、FEATURES_VERIFIED通过；METRICS_REPLAY_VERIFIED仍PENDING。** 已开发历史重放分型、只读Loader、无模型IRRA cached scorer及统一Engine接线；本轮真实分数矩阵/排名/指标计算均为0，无IRRA/G/H模型加载、神经前向、GPU、训练或新攻击生成。

开发基准32f55827c94c6512b81d32a9aa478b1d14c70b92，独立分支codex/tbps-framework-stage4b1，独立树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4b1-20261017。实现Commit在PROJECT_STATE由后续文档提交登记。原Stage 3B FAIL及其他历史证据保留。

## 新增与必要接线

- historical_contracts：ReplayIdentity/FeatureBinding/Representation独立分型；像素SHA强制UNKNOWN，不放宽InputBinding/BoundRepresentation。
- historical_sources：原IRRA/CUHK Test、完整3074/6156顺序、协议/Selection/Checkpoint/完成来源及文件声明核验，显式映射新ModelSpace。
- historical_loader：四组868批次/3472文件只读加载，FP32/维度/有限性/L2及原行身份核验，立即detach().clone，原生H动作审计，不可变新检查回执。
- irra_cached：原生归一化点积、Query切片clone；真实score默认拒绝，合成数据可通过同一Engine。
- evaluation：新增显式historical执行分支和单独score证据格式；原评分/排名循环及严格materialized分支保留。
- replay_acceptance及固定JSON：下阶段阈值锁定，仅校验计划。
- 两个测试文件、两份配置、API说明、本报告、ADR-0006和统一状态/架构/注册表索引。

engines/metrics.py、engines/provenance.py、models/base.py、contracts.py及全部历史prototypes未修改。没有复制一套Full指标算法。

## 来源与真实CPU核验

唯一特征来源是原Joint Full：RELEASE/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/；RELEASE=/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014。未读取有大Storage兼容负担的旧G-v2 Text/Clean Tensor。

固定Checkpoint SHA：

| 模型 | SHA-256 |
| --- | --- |
| G-v2 Epoch10 | f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e |
| H-v1 Epoch7 / visit14336 | 487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df |
| 官方IRRA | 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009 |

原Joint协议digest59ffb61a575cbeebf38e06bee5154b96f8967bb165b9b22305fad6c9cc967b48；Selection文件SHA584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757。Selection仍以独立Validation已保存结果选H7，未使用Test反馈重选。

最终真实检查：

| family | 批次 | 行 | 检查 |
| --- | ---: | ---: | --- |
| clean_gallery | 49 | 3074 | PASS |
| image_gallery | 49 | 3074 | PASS |
| clean_text | 385 | 6156 | PASS |
| adv_text | 385 | 6156 | PASS |
| 合计 | 868 | 18460 | 3472文件SHA/回执/结构通过 |

source_files_verified=3239；最终新验收Manifest绑定去重6720个FileRef，包括原来源、全部特征文件、75项冻结动作运行依赖和相关检查producer。历史缓存SHA/mtime前后不变，无pending、遗漏、声明外文件或损坏。每批Gallery64、文本16，默认不保留整表。

全部6156条H缓存动作经冻结原validate_action/qualify_caption与Canonical/Span/Unicode/Byte-BPE检查。原文本与缓存动作身份、编辑重放、合法候选/掩码、Tokenizer审计通过。**未运行H网络验证predicted_importance或贪心选择的神经数值**；这些记录的真实性由原文件SHA/协议和已完成历史证据绑定，本轮只验证其结构、合法性与文本一致性。

CPU Tensor读取限定features.pt白名单，weights_only=True/map_location=cpu；阻断Checkpoint反序列化、模型构造/forward、CUDA初始化、导数。真实报告forbidden_attempts为空，model_loads/neural_forwards/scores_computed均0。

## 资源与不可变证据

目录：/home/lzf/ldx/outputs/tbps-framework-stage4b1/，不提交Git。

- metadata_v1/result：元数据检查通过。
- features_v1/result：首次完整特征检查通过，流式检查段43.35702297091484秒；原证据保留。
- features_final/result：补全新验收回执的producer和全部feature FileRef绑定后，再次只读核验通过；未重写历史文件。
- 最终流式检查段43.30987264215946秒；CPU进程peak RSS512.78125MiB，Torch2.5.1+cu124、2线程。时间不含前置来源SHA等全部CLI开销；不是Full排名耗时或下阶段峰值承诺。
- 最终report SHA7d8dd8364df3537471658a0d3ce3c30f5e4609f700bb75a68e2f68e0fdc979bf。
- 最终Manifest SHA5cd8dcf0e8ccbc61b58adc7dae37e26e51f361369e15f6fefa77c9465bfc1370。
- cpu_framework_final.log、cpu_legacy.log、protection_result.json为测试/保护证据。

## CPU测试与历史保护

**412框架tests（383原有+29新增）+36历史基础=448项通过。**

合成12 Gallery/24 Query多正例Fixture通过历史路径与独立正常materialized路径，四模式score、五指标及DR exact。覆盖来源缺项/篡改、Selection/Checkpoint拒绝、顺序/模型空间隔离、UNKNOWN像素分型、动作原文、dtype/归一化/NaN、旧view载入后紧凑Storage、pending/中断、不可变score发布与读回、真实评分默认禁止和固定阈值不可放宽。另有一项小规模真实动作CPU测试，只读一条metadata，验证原生Unicode/Tokenizer规则及篡改拒绝，没有模型前向。

7991个原保护文件SHA/mtime不变；3472个Joint特征文件SHA/mtime不变。旧实验发布树、Stage4A/Stage3B-2树HEAD/clean不变；Stage3B FAIL、Stage3B-2 PASS及Stage4A回执/绑定来源只读核验继续有效。此保护范围不等于重新审计所有旧训练轨迹/AP/TTA缓存。

## 统一评测映射与边界

clean_gallery→clean_gallery；image_gallery→adv_gallery；clean_text/adv_text同名。四模式沿用Stage4A：Clean=(原文,原图)，Image-only=(原文,攻击图)，Text-only=(攻击文,原图)，Joint=(攻击文,攻击图)。

原生Scorer只接收表示，不含PID/正确配对/攻击策略回调；Engine按原Manifest顺序处理全部同PID正例，无camera过滤。IRRA CPU FP32归一化点积、官方默认argsort/tie、五指标百分数及DR=(Clean-Adv)/Clean×100保持。真实模式必须显式历史分型、来源一致及chunk128；当前CLI没有真实评分选项。

缺少对抗RGB SHA是历史证据限制，不能由feature反推。本桥只合法重放原IRRA特征，不是新RGB/新攻击结果，不能供RaSa/APTM重编码或特征复用。通用Adapter仍允许任意维度/融合评分，IRRA512约束只在专属桥内。

## Stage 4B-2固定验收与停止

固定JSONexecution_authorized=false：CPU FP32/Torch2.5.1+cu124/2线程、Query128/完整Gallery、RSS2GiB/1800秒；feature exact、score最大误差1e-6、官方/新Ranking索引exact、五指标及DR对原summary绝对误差≤1e-5个百分点。失败停止，不改排序/容差。完整受控运行入口和资源监视留给下阶段，经用户批准才开发/执行。

本阶段完成后停止等待源码审查；未运行真实Full评分/排名，不进入Stage4B-2、Stage4C、GPU或跨模型实验。

本轮独立CPU验收回执：/home/lzf/ldx/outputs/tbps-framework-stage4b1/cpu_verified_v1/result/；
Manifest SHA80c3446cd9f149d54e7d3a403bfb5121f95b39b01498c8bfb851379e624e44ba，
report SHA4dc8a6d87c94830a246b84e0d99edb7ba69f2eb73653ada82ba713ad2331b8a8。
绑定本轮代码/配置/测试、测试日志、保护记录和最终特征验证回执；不包含新的真实分数或指标。
