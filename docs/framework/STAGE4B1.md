# Stage 4B-1：历史IRRA特征只读接入

本阶段只验证历史来源和特征，不执行真实Full评分或排名。权威状态见[PROJECT_STATE](../PROJECT_STATE.md)，验收见[报告](STAGE4B1_REPORT.md)，身份决定见[ADR-0006](../decisions/0006-explicit-historical-feature-replay.md)。

## 职责与状态

| 模块 | 职责 |
| --- | --- |
| engines/historical_contracts.py | 独立HistoricalReplayIdentity、HistoricalFeatureBinding、HistoricalRepresentation；不是materialized输入的子类 |
| engines/historical_sources.py | 固定IRRA/CUHK官方Test、原协议/Selection/Checkpoint/完成回执与有序成员核验；映射新ModelSpace |
| engines/historical_loader.py | 只读批次SHA/回执/结构、紧凑clone、原生H动作合法性检查、不可变CPU检查回执 |
| models/irra_cached.py | 无模型的IRRA原生FP32归一化点积；真实评分默认拒绝 |
| engines/evaluation.py | 显式历史分型接入现有四模式和同一排名/指标循环 |
| engines/replay_acceptance.py | Stage 4B-2固定验收计划校验，不执行计划 |

三种状态不可互换：

- METADATA_VERIFIED：来源、原协议/完成回执、Selection、成员和特征文件声明已核验；未读取全部Tensor内容。
- FEATURES_VERIFIED：全部声明文件SHA和批次内容/身份/H文本动作检查通过；没有计算真实分数或指标。
- METRICS_REPLAY_VERIFIED：尚未取得，需另行授权Stage 4B-2后与原Full结果比较。

历史对抗RGB未保存逐图内容SHA。HistoricalReplayIdentity明确携带 `adversarial_pixel_sha256="UNKNOWN"` 和 `artifact_kind="historical_feature_replay"`。不能生成AttackArtifact/RGB证明，不能用feature SHA替代像素SHA，不能把本阶段标成新的攻击实验。

## 固定来源和缓存映射

配置：[stage4b1_replay.server.json](../../configs/framework/stage4b1_replay.server.json)。仅读取原Joint Full紧凑缓存：

`/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/`

| 原family | 新引擎输入名 | 数量 | 用途 |
| --- | --- | ---: | --- |
| clean_gallery | clean_gallery | 3074 | 原始Gallery表示 |
| image_gallery | adv_gallery | 3074 | 固定G10攻击Gallery表示 |
| clean_text | clean_text | 6156 | 原始Query表示 |
| adv_text | adv_text | 6156 | 固定H7攻击Query表示 |

Gallery与Query独立有序，不zip。四种模式分别组合原/攻击文本和原/攻击图库，保留全部同PID正例。只可用于同一IRRA原生评分；不能跨Tokenizer、Victim或重标为RaSa/APTM。

每批核对features.pt、metadata.json、intent.json、receipt.json；要求完整声明集、SHA、原行身份、FP32 [N,512]、有限性及原L2阈值。这里的512维是历史IRRA专属约束，不改通用Model Adapter。读取后立即detach().clone()并释放原Tensor；默认流式校验，不保留完整表。pending、缺项、冲突、错误顺序、损坏和意外文件明确拒绝。

H动作使用冻结原validate_action、CaptionFrontend、qualify_caption、Byte-BPE检查：原Caption/Canonical/Span、预算、Unicode编辑重放、tokenizer_audit和记录内合法动作掩码。只读已有importance/选择记录，不恢复H权重或重新预测其数值。Checkpoint只流式SHA，不torch.load。

## CPU命令

以下命令只核验，不提供score/run/GPU选项。已有输出拒绝覆盖；再次人工检查应使用新的输出目录，不删除旧回执。

元数据检查：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4b1-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.engines.historical_loader --config configs/framework/stage4b1_replay.server.json --output /home/lzf/ldx/outputs/tbps-framework-stage4b1/review_metadata_v1
```

显式流式特征检查（只读全部Tensor/SHA，不评分）：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4b1-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.engines.historical_loader --config configs/framework/stage4b1_replay.server.json --verify-features --output /home/lzf/ldx/outputs/tbps-framework-stage4b1/review_features_v1
```

成功标志分别为HISTORICAL_METADATA_VERIFIED、HISTORICAL_FEATURES_VERIFIED和退出码0；报告/Manifest/完成回执在指定目录的result/。失败保留FAILED报告及异常；SIGINT可中断，不自动续跑或删除pending。硬中断未发布结果不能视为通过。

CPU回归：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4b1-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

## 最小接口

```python
from tbps.engines.historical_sources import JointFullSources
from tbps.models.irra_cached import IRRACachedScorer
from tbps.engines.evaluation import EvaluationEngine

sources = JointFullSources("configs/framework/stage4b1_replay.server.json")
scorer = IRRACachedScorer(sources.space, sources.identity)  # 真实评分仍拒绝
engine = EvaluationEngine(sources.manifest, scorer, direction="higher_better",
    execution="historical_feature_replay", model_sources=tuple(sources.refs.values()))
# 本阶段不调用真实engine.run；合成Fixture覆盖同一入口。
```

不能将HistoricalRepresentation传入materialized路径或混入BoundRepresentation。Stage 4A严格InputBinding、像素SHA要求和正常ScoreRecord保持。历史Score采用单独schema，绑定原身份、特征文件、行特征SHA、实际原/对抗文本SHA和映射ModelSpace，并注明UNKNOWN像素身份与not_new_attack。

## Stage 4B-2计划，不执行

[固定JSON](../../configs/framework/stage4b2_acceptance.json)的execution_authorized=false。下一阶段需用户单独授权，并增加受控CPU入口及资源监视；本轮没有Full评分命令。

预先固定CPU FP32、Torch 2.5.1+cu124、2线程、Query分块128、完整3074 Gallery/6156 Query；RSS≤2GiB、总时长≤1800秒。要求同一特征内容exact、原生点积max_abs_error≤1e-6、独立官方Ranking与新Ranking索引exact；五指标及DR对原Joint Full对应summary绝对误差≤1e-5个百分点。使用官方默认argsort tie行为，不增ID排序、不变指标定义、不事后放宽阈值。

下一阶段需比较四组真实结果、验证指标产物和完成回执后，才可登记METRICS_REPLAY_VERIFIED。失败停止并保留证据，不能为了匹配结果改排序或换来源。GPU、新攻击生成、重新训练和跨模型评测仍不在此计划内。
