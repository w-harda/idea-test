# Stage 4A：统一TBPS Evaluation Engine（CPU-only）

基准 `0d90e79c08bf313b9085e75531a4226b3b266698`；分支 `codex/tbps-framework-stage4a`。
本轮不加载真实模型/特征、不重算Full、不生成攻击数据。

## 文件与API

| 文件 | 职责 |
| --- | --- |
| src/tbps/engines/metrics.py | ranking(scores, EvaluationIdentity, direction)；官方定义、百分数与drops |
| src/tbps/engines/provenance.py | AttackOrigin、BoundRepresentation、原生结构payload_digest；来源/内容绑定 |
| src/tbps/engines/evaluation.py | EvaluationEngine.run；原生评分、四模式、Query分块；publish_result/load_result |
| src/tbps/engines/legacy.py | LegacySpace、describe_joint_full；只读元数据索引，require_replay始终拒绝 |
| src/tbps/engines/__init__.py | 无副作用命名空间，不导入Torch/模型 |
| configs/framework/stage4a_legacy.server.json | 固定Joint/G-v2历史JSON及原生IRRA I/O来源，不是执行配置 |
| tests/framework/test_stage4a_engine.py | 原生评分/四模式/指标/隔离/存储/独立官方数值CPU对照 |
| tests/framework/test_stage4a_legacy.py | 历史桥合成元数据与拒绝行为 |
| src/tbps/config.py | 只修正执行拒绝提示：已有缓存引擎，但尚无统一真实执行driver；权限不放宽 |

## 四模式协议

| 模式 | Query | Gallery |
| --- | --- | --- |
| Clean | clean_text | clean_gallery |
| Image-only | clean_text | adv_gallery |
| Text-only | adv_text | clean_gallery |
| Joint | adv_text | adv_gallery |

每个family具有独立InputBinding：Manifest摘要、ModelSpace、有序ID、有序实际输入SHA。原图SHA域为original_image_file，攻击图为adversarial_rgb_tensor，文本为caption_utf8。Feature payload另有typed摘要和原缓存FileRef；禁止把Feature SHA填进像素SHA字段。

clean必须匹配Manifest原内容；adv必须携带原输入SHA、攻击Checkpoint/协议、预算和代理来源摘要。允许合法NOOP内容与原文相同，不能删Query。H/G策略对象、正确配对和目标分数不进入评分输入接口，更不进入攻击接口；引擎没有攻击生成回调。

ModelSpace绑定Victim/Checkpoint、native源码、Tokenizer、预处理、运行精度与评分协议。ScoreBatch须与同一space/score_id一致；未来融合模型可返回本地tokens/原生结构，Engine只要求最终[Nq,Ng]有限CPU FP32分数。readonly_cached还需显式纯缓存Adapter标志、source_verified和Victim Checkpoint FileRef。本阶段未提供该真实加载器，不会自动实例化IRRA。

## 最小使用示例

下面变量由调用方显式提供：小型合成DatasetManifest、Native Mock Adapter及四个BoundRepresentation。完整可执行构造见CPU测试Fixture；没有真实模型默认值。

```python
from tbps.engines.evaluation import EvaluationEngine, publish_result, load_result
from tbps.artifacts import ArtifactStore

engine = EvaluationEngine(
    manifest, native_mock,
    direction="higher_better", execution="cpu_fixture",
)
protocol, report, scores = engine.run(bound_inputs, query_chunk=3)
store = ArtifactStore(
    "/home/lzf/ldx/outputs/tbps-framework-stage4a-demo/results",
    new_base="/home/lzf/ldx/outputs/tbps-framework-stage4a-demo",
)
publish_result(store, "fixture-v1", protocol, report, scores)
saved = load_result(store, "fixture-v1",
                    expected_protocol_sha256=report["protocol_sha256"])
```

cpu_fixture明确限制已知Fixture Dataset、*_fixture模型及≤64 Gallery/Query，不允许把真实3074/6156放进Fixture入口。其他真实Model/Dataset组合默认拒绝；RaSa/APTM未实现。

Query分块要求Adapter提供slice_query(batch,start,end)并声明query_independent_scores=True。不能Gallery分块后各自取Top-K，否则会破坏完整排名及tie定义。源分数允许lower_better，但声明冲突拒绝。指标包含全部同PID正例，无camera过滤；Gallery<10明确拒绝R@10，不静默缩小rank。

## 新输出和来源

publish_result原子保存protocol.json、metrics.json、score_records.json及四组scores/*.pt，并生成原ArtifactStore Manifest/完成回执。ScoreRecord同时绑定Query/Gallery输入、有序成员、模型Checkpoint、评分协议、方向和实际score文件SHA。来源自动包含annotation、所有特征cache FileRef、G/H Checkpoint/协议、传入Victim来源及3个实际engine producer。

已发布版本不可覆盖；中断pending不自动恢复/删除。Safe-stop回调在每模式及Query分块边界执行，异常不返回伪完成结果。load_result先核验回执/来源，再读取clone后的Score，验证协议/顺序/内容，不重新排名。

原parse_config/require_executable仍不提供真实模型/攻击/Full调度。本轮API与配置解析不是GPU授权。

## CPU命令

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4a-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

本轮383框架+36历史基础=419项CPU tests通过；51新增，全部独立官方小矩阵对照exact。没有GPU/Full命令。详细映射、旧资产缺口和Stage 4B方案见[报告](STAGE4A_REPORT.md)。
