# Stage4C-1：模型无关攻击产物准备接口

本轮只CPU来源检查和Fixture测试；没有真实G/H/IRRA前向、GPU、全量生成或排名。当前状态见[PROJECT_STATE](../PROJECT_STATE.md)，验收见[报告](STAGE4C1_REPORT.md)，设计见[ADR-0007](../decisions/0007-model-independent-materialized-attacks.md)。

## 模块

| 路径（src/tbps/materialization/） | 职责 |
| --- | --- |
| contracts.py | 冻结MaterializationPlan、MaterializedUnit；Source/Manifest与目标模型分离 |
| store.py | 复用ArtifactStore，FP32 RGB/UTF-8保存、SHA/预算/动作读回检查 |
| pipeline.py | 公开输入投影、底层单元处理、独立Gallery/Query集合及目标InputBinding |
| generation.py | 可恢复生成的前置Claim/完成证明与组合索引 |
| legacy_text.py | 只读旧Joint Full文本/动作及精确记录指针，零模型/Feature Tensor读取 |
| audit_sources.py | 默认CPU来源检查入口；无生成、模型加载、GPU或评分选项 |

测试：tests/framework/test_stage4c_materialization.py。已有G/H插件、模型、投影、旧Storage和历史源码不修改。

## 来源CPU命令

以下命令只读取来源。输出名必须新建，已有目录拒绝覆盖；本轮正式CPU来源证据在source_audit_v2，不能重写。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4c1-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.audit_sources --output /home/lzf/ldx/outputs/tbps-framework-stage4c1/review_sources_v1
```

默认不传--output只打印报告。成功标志MATERIALIZATION_CPU_SOURCE_PASS/退出码0。检查3074原图SHA、G真实调用路径、训练缓存与旧G1条件身份，以及6156条H原动作/Unicode/Canonical/Tokenizer，均不调用模型。禁止Checkpoint反序列化、神经构造/前向、CUDA、Backward、JVP。G1条件只校验文件字节/元数据，不反序列化。

CPU回归：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4c1-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
```

没有真实物化或GPU CLI。本轮不提供可自动启动真实模型的命令。

## 计划与单元

MaterializationPlan绑定DatasetManifest、固定G10/H7 Checkpoint/训练协议/Selection、来源FileRef和RGB工作尺寸。真实模式必须验证LegacyAttackSources；Fixture来源与真实来源分开。原数据标签可留在Manifest身份层，但插件只收到公开请求；H后端仅Caption，不能传入PID或Gallery。

标准RGB工作空间是PIL RGB→bilinear Resize384×128→ToTensor [0,1]，不先做IRRA均值/方差归一化。L∞预算相对这个原RGB Tensor，而非原始JPEG文件字节或其他模型的归一化Tensor。

G单元保存pixels.pt：

```python
{"original": FloatTensor[1,3,384,128],
 "adversarial": FloatTensor[1,3,384,128],
 "effective_delta": FloatTensor[1,3,384,128]}
```

全部CPU FP32、有限、像素范围[0,1]；effective_delta逐值等于adv-original，FP64测量L∞严格≤8/255。保存/载入真正clone，保持紧凑独立Storage。metadata.json保存原图FileRef、三种Tensor内容SHA、L∞、完整trace、预算和G来源；manifest/completed绑定文件SHA和来源。当前不生成预览；未来预览须另标preview，不能替代pixels.pt。

H单元保存original.txt、adversarial.txt和metadata.json，保留原UTF-8字节/不做Unicode规范化；元数据保存完整action、Span、原/替换字符及码位、Dynamic-K、STOP/NOOP、Tokenizer审计、H来源和两个文本SHA。读回必须使用冻结Caption-only校验器。

Tensor内容SHA沿用attacks.base.tensor_sha：dtype/shape的规范化JSON加连续Tensor字节；不同于torch.save文件SHA。文本内容SHA就是实际UTF-8字节SHA。MaterializedUnit分开保存content_sha256与data FileRef，不使用UNKNOWN或Embedding SHA充当真实像素。

## 生成、恢复与组合

未来授权的Runner应使用：

```python
from tbps.materialization.generation import (
    generate_image_once, generate_text_once, seal_generated_collection)
from tbps.materialization.store import MaterializedStore

# plan/ArtifactStore/冻结插件由显式来源预检构造；这里不实例化真实模型。
store = MaterializedStore(artifact_store, plan)
store.publish_plan()
# 真实计划默认拒绝生成；后续获授权Runner才显式传allow_real_generation=True。
# 图像调用generate_image_once；在线H调用generate_text_once。
# 历史H复用则通过HistoricalTextSource验证后publish_text，不调用H。
# 完整两类单元准备后：
# seal_generated_collection(store, "joint", images=image_refs, texts=text_refs,
#                           validate=frozen_caption_validator)
```

本示例是接口约定，不是本轮GPU执行命令。pipeline底层函数供协调器复用；需要中断保护时使用generation前置Claim入口。

Claim在任何模型前向之前原子发布。若崩溃发生在前向后、pixels.pt发布前，保留Claim并拒绝自动再执行；已发布单元但生成证明缺失也拒绝直接恢复。完整单元/证明验证通过才跳过。没有Step级恢复、自动清理或自动rollback承诺；需要人工审查失败现场后另行决定。

Joint只生成collection/index引用，不再调用G/H；四模式可消费原/Adv图库与原/Adv文本。集合封存要求原Gallery/Query各自完整顺序，3074不与6156 zip配对，NOOP/不可编辑Query不丢弃。

## 历史H与目标模型

HistoricalTextSource逐批校验原Joint protocol/summary/completed/Selection/metadata/intent/receipt与固定H7；直接返回原Caption、攻击Caption及动作，不读取features.pt。物化时验证metadata中的原记录指针，不能用相同Feature推断文本或用另一条合法编辑冒充历史H输出。当前6156条来源已验证，但尚未批量发布新的真实文本单元。

target_input_binding(store, model_space, modality, unit_refs)将同一内容SHA和完整有序ID绑定到不同ModelSpace；缓存ID不同，样本本身不变。RaSa/APTM后续各自原生预处理/Tokenizer/融合评分并生成独立Representation/Score缓存；本轮没有真实Adapter实现或评分许可。目标Tokenizer可能改变长度/截断，应记录而不能反过来调整H字符。
