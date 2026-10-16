# Stage 2：CUHK-PEDES Dataset Adapter与IRRA Model Adapter

当前状态见[PROJECT_STATE](../PROJECT_STATE.md)，保护边界见根AGENTS。基准61471db976508411aafe58e97113322c14d6bbc7；独立分支codex/tbps-framework-stage2。本轮只执行CPU验证，未加载真实模型。

## 新增接口

| 文件 | 职责 |
| --- | --- |
| src/tbps/datasets/cuhk_pedes.py | 官方annotation、原始顺序/多Caption/全部正例、Manifest及公开输入投影 |
| src/tbps/datasets/historical_membership.py | annotation+内部split+G来源+H资格/Gallery/调度/Validation的实际成员绑定 |
| src/tbps/models/base.py | ModelAdapter评分协议、ModelSpace、输入身份与持久化Record；不要求双塔 |
| src/tbps/models/irra_source.py | 固定权重/配置/BPE/CLIP bootstrap/30份官方源码SHA；显式离线加载器 |
| src/tbps/models/irra_io.py | 官方Byte-BPE包装、77-token报告、RGB预处理和可微Normalize |
| src/tbps/models/irra.py | IRRA原生编码、原生相似度、Inference/Surrogate View及Artifact绑定 |
| src/tbps/models/irra_cpu_checks.py | 显式小规模CPU官方函数对照，不调用真实模型 |
| src/tbps/preflight.py | CPU身份/来源检查；没有GPU、模型加载或Full排名选项 |
| configs/framework/stage2_sources.server.json | 真实服务器路径与固定来源SHA，仅元数据 |
| tests/framework/stage2_fixtures.py、test_stage2_*.py | 合成来源、成员权限、编码器Mock、缓存隔离与负例测试 |
| src/tbps/.gitignore | 局部开放被旧根datasets/规则忽略的新Adapter源码，不修改根规则 |

src/tbps/config.py仅增强Joint组件权限及dry-run能力说明。原contracts/Store/legacy bridge及历史攻击程序不修改；项目文档同步更新，“8类资产”更正为“8个资产索引”。

## 数据读取与成员权限

官方annotation是唯一原文来源，保持列表顺序及Caption原字符串，包括空格、大小写、Unicode和全部Caption。native PID保留annotation原值，以PersonID("cuhk_pedes", value)隔离命名空间。官方训练代码的id-1只用于分类标签重编号；本Adapter不把它施加到历史攻击身份或Test检索PID。

- official train：34054 Gallery / 68126 Caption / 11003 PID。
- official val：3078 Gallery / 6158 Caption / 1000 PID。
- official test：3074 Gallery / 6156 Query / 1000 PID。
- 攻击内部Train：2048图及固定Caption / 1861 PID。
- 攻击内部Validation：128图及固定Caption / 121 PID，来自official train，不能冒充official val。

image_id使用cuhk:annotation_index:0，query_id使用cuhk:annotation_index:caption_index。DatasetManifest保存顺序/配对，全部同PID图库项都是正例。没有假设每图恰好两条Caption。

```python
from tbps.artifacts import read_json
from tbps.contracts import FileRef
from tbps.datasets.cuhk_pedes import CuhkPedes
from tbps.datasets.historical_membership import load_historical_partitions

c = read_json("configs/framework/stage2_sources.server.json")
dataset = CuhkPedes(FileRef(**c["annotation"]), c["image_root"])
test = dataset.manifest("test", expected_counts=(3074, 6156))
image_input = dataset.public_image(test, test.gallery[0].image_id)
text_input = dataset.public_text(test, test.queries[0].query_id)
parts = load_historical_partitions(
    dataset, {name: FileRef(**ref) for name, ref in c["membership"].items()}
)
```

构造Dataset读取annotation；manifest显式读取所选图片字节计算SHA，不解码图片或模型编码。公开输入只有FileRef/pixel_space或Caption/准确Span；没有PID、正确Gallery、配对标签和奖励。给public_text传入AttributeSpan时必须与原Caption准确一致，不重新搜索或归一化Span。

实际成员核验不能用调用方声明的role替代。load_historical_partitions校验：

1. G Manifest/Protocol/完成回执及2176份metadata的SHA与原始身份。
2. annotation、原internal split记录与PID成员资格；不是只看PID属于某个集合。
3. H Train/Validation资格记录与G完整身份一致，Caption/原Span准确。
4. H训练Gallery及独立Validation Gallery身份一致；H训练调度每Epoch仅访问全部2048 Train，资格账本与调度成员对应。
5. H Validation协议关联原训练协议及128条独立成员；Train/Validation PID、图像不重叠。

parts.require_member检查完整记录，拒绝相同role但伪造Caption/PID或未入选Caption。parts.reward_binding只接受实际Train Query与实际Train Gallery，要求配对正例及负例；Validation/Test不能进入。AttackPartitions禁止未经校验的直接构造。TrainRewardContext只是底层数据契约，不是成员权限凭证；未来训练入口必须通过已校验parts绑定奖励。

冻结Manifest按对象缓存身份索引，不缓存像素、特征Tensor或计算图。公开图片使用前仍校验其SHA。正式运行前应重新验证来源；此处没有训练调度或恢复功能。

## 原生模型接口与两种View

通用ModelAdapter只约定spec、space及score(query, gallery)，不强迫先生成512维向量或调用encode_images。Fusion模型可以直接接收公开输入/local tokens并返回ScoreBatch；当前只通过Fusion Mock验证结构，未实现RaSa/APTM。

IRRA专属接口：

- inference().encode_images：接收float32 RGB [B,3,384,128]、[0,1]。
- inference().encode_texts：接收Caption字符串或PublicTextInput序列，返回RepresentationBatch与TokenReport。
- encode_tokens：官方LongTensor[B,77]，检查SOT/EOT和词表范围。
- score(text_representation, image_representation)：Query×Gallery矩阵。
- surrogate()：参数仍eval/requires_grad=False，显式允许图像输入梯度；不宣称Unicode/Tokenizer可微。

pixels_from_pil是PIL RGB→官方Resize bilinear(384,128)→ToTensor；攻击/预算应在此后的原RGB [0,1]空间。encode_images内部仅执行一次可微mean/std Normalize及官方encode_image，不做PIL转换、重复Resize、裁剪或扰动投影。mean/std与官方build_transforms一致。

模型原生encode_image取CLS，encode_text按EOT/argmax位置取特征；Adapter委托官方方法，再执行float32 L2归一化。IRRA只有这个Adapter声明512维/77 tokens，通用接口没有该限制。

Inference View内部inference_mode且输出无梯度；返回前在inference_mode(False)内clone，使其可安全作为Surrogate常量。Surrogate View显式启用输入梯度，即便外部处于no_grad/inference_mode；仍拒绝模型参数解冻或train状态。CPU Mock覆盖上述嵌套场景、输入梯度非零及参数/buffers不变。

真实模型只允许IRRAAdapter.from_source(source, device=..., allow_model_load=True)显式加载；默认拒绝且无CLI加载入口。本轮未调用真实加载。加载器只读本地已锁定CLIP bootstrap，禁止下载，strict=True加载官方Checkpoint，最后fp32/eval/冻结；同名model包已占用时拒绝，建议独立进程。

fp32/math、禁用外部autocast、TF32关闭等运行开关与旧协议一致，编码结束恢复调用方开关。真实加载、GPU注意力路径与数值等价尚待授权Smoke；不能把CPU Mock通过解释为已完成真实编码验收。

## 原生检索评分和缓存隔离

IRRA评分是L2(text) @ L2(image).T。官方rank采用torch.argsort降序、默认非stable，不添加新tie规则；全部同PID正例、无camera过滤，保留全部Query。mAP为逐Query AP均值；R@k为首次命中CMC；mINP为正例数量/最后正例排名的均值，单位百分数。

ModelSpace绑定模型Checkpoint、官方来源、实际4个Adapter producer、预处理/运行版本、Tokenizer代码/BPE/ftfy/regex/Python Unicode、精度和排名协议。只锁相关文件，不枚举整个新框架目录。

持久化必须使用InputBinding，额外绑定Dataset Manifest SHA、有序成员ID、有序实际输入SHA及模态。原文与Unicode编辑文本必须使用各自真实内容SHA；对抗图必须使用其像素或已验证攻击产物来源绑定，不能冒用原图SHA。PID不进入该公开模型/策略输入。

representation_artifact(..., inputs=...)返回RepresentationRecord；score_artifact(..., query_inputs=..., gallery_inputs=...)返回ScoreRecord。模型空间相同也不能忽略数据、输入编辑或顺序。Record可转asdict后交给新ArtifactStore；本轮未生成任何真实特征缓存。不同Checkpoint/Tokenizer/预处理、同ID不同文本或不同顺序均有不同cache_id。

历史Full特征缺少新Record身份，不能直接改标签为新空间；需后续显式只读兼容核验。本轮没有复制、转换或重写旧缓存。

## 可复制CPU命令

仅检查来源/身份，无模型加载：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016
TMPDIR=/home/lzf/ldx/tmp CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.preflight --sources configs/framework/stage2_sources.server.json
```

成功标志STAGE2_CPU_PREFLIGHT_PASS。严格核验模型及选中图片字节SHA，有CPU磁盘I/O，但没有torch.load/模型构造/编码/排名。

官方Tokenizer与合成PIL/排名小Fixture：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016
TMPDIR=/home/lzf/ldx/tmp CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.models.irra_cpu_checks --sources configs/framework/stage2_sources.server.json
```

成功标志STAGE2_CPU_IO_EXACT_PASS。10种合成文本仅Byte-BPE tokenization；3组合成图像；4×12合成排名。没有IRRA.encode_text/encode_image或Full特征。

合成CPU tests及Stage 1回归：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016
TMPDIR=/home/lzf/ldx/tmp CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
```

183通过；另36项历史基础CPU回归通过。完整数据及未验证范围见[Stage 2报告](STAGE2_REPORT.md)。

## 下一项最小GPU授权建议（本轮未执行）

建议仅授权独立进程、冻结官方IRRA一次加载、2张已核验Train原图+2条原Caption：

1. Adapter推理与同一官方Backend直接调用对照：各2张图/2条文本，检查Token exact、预处理、512维范数及Query×Gallery分数。
2. 1张原图通过Surrogate做一次输入梯度检查，无Optimizer，无G/H调用；参数/buffers SHA前后不变。
3. 总计建议最多5个IRRA图像样本前向、4个文本样本前向；不生成攻击、不评测检索指标。
4. 预先固定fp32/max_abs_error<=1e-6；异常先停止，不临时放宽。记录实际次数/梯度/显存/耗时及冻结证明。

这只是下一步授权方案，不是当前运行许可；不包含Stage 3 Attack Plugin或Evaluation Engine。
