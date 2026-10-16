# Stage 2.5：IRRA真实GPU最小等价性验证

本阶段只有一次2张内部Train原图、2条对应原始Caption的数值对照及一次输入梯度检查。没有G/H、攻击Loss、Optimizer、排名或Test评测。实现与实测结果见[验收报告](STAGE25_REPORT.md)。

## 独立入口与固定协议

- `src/tbps/smoke_irra.py`：默认CPU来源/实际成员预检；显式`--allow-gpu-smoke`才加载模型。
- `configs/framework/stage25_irra_smoke.server.json`：锁定Stage 2来源配置SHA、2个row_id、成员摘要、FP32/math/no_tf32、1e-6阈值、最多5图像/4文本样本前向；不提供放宽规模/阈值的CLI。
- `tests/framework/test_stage25_smoke.py`：30项CPU护栏、数值比较、冻结/梯度、原子证据及无副作用测试。

使用Stage 2独立树的新分支`codex/tbps-framework-stage25`。不修改Stage 2 Adapter、历史prototype、官方模型或任何数据/结果。

CPU预检只读annotation、已核验的G/H实际成员来源、图像字节SHA及官方来源。不构造Test Manifest，不解码全量图像，不读取条件Tensor/Full特征。样本身份留在验证与审计层；Model View收到的只有像素或原始Caption。

## 官方独立对照路径

同一个冻结Backend只加载一次：

1. A：从已锁定官方文件提取原`build_transforms(is_train=False)`、`tokenize`函数，独立实例化官方SimpleTokenizer；直接调用Backend.encode_image/encode_text，再执行官方检索的FP32 L2 Normalize及Query×Gallery点积。
2. B：Stage 2 Adapter的pixels_from_pil、Tokenizer和Inference View；通过Adapter.score评分。
3. Surrogate View：第一张原图的归一化特征[0,0]做普通输入导数探针，不定义攻击目标。

A未调用Adapter预处理/Tokenizer/编码辅助函数。两路径只共用模型实例及同一执行设置。完整原函数经AST提取，避免导入官方datasets包触发无关依赖；来源SHA在读取前后检查。

原生Backend.encode_image/encode_text外围计数，只保留标量；先检查样本预算再调用。总计官方2+Adapter2+梯度1=5图像样本、官方2+Adapter2=4文本样本。超预算提前拒绝，不用会保留Tensor参数的Mock记录真实调用。

## 服务器命令

以下GPU命令是**本轮已经执行的一次性命令记录，不是再次运行授权**。本轮已完成；不要更换输出目录绕过一次性保护或重复执行GPU。

工作树：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage2-20261016
```

默认CPU预检（不加载模型、不初始化CUDA）：

```bash
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.smoke_irra --config configs/framework/stage25_irra_smoke.server.json
```

成功标志：`IRRA_SMOKE_CPU_PREFLIGHT_PASS`。固定配置内已绑定真实best.pth路径及SHA，不通过CLI换权重。

一次性GPU执行命令归档：

```bash
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.smoke_irra --config configs/framework/stage25_irra_smoke.server.json --output /home/lzf/ldx/outputs/tbps-framework-stage25/irra_equivalence_v1 --allow-gpu-smoke
```

成功标志：`IRRA_GPU_EQUIVALENCE_PASS`。只接受空的新输出目录。执行前及加载前均检查物理GPU0为RTX 4090、无已有compute进程、可用显存至少8192MiB、利用率不超过10%；不抢占/kill任何进程。资源不足拒绝，不自动等待或重试。

CPU回归：

```bash
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

新证据只读SHA/回执校验（无模型加载/排名）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -c 'from tbps.artifacts import ArtifactStore; s=ArtifactStore("/home/lzf/ldx/outputs/tbps-framework-stage25/irra_equivalence_v1",new_base="/home/lzf/ldx/outputs/tbps-framework-stage25"); s.verify("attempt"); s.verify("result"); print("STAGE25_RECEIPTS_SHA_PASS")'
```

## 输出、安全停止与限制

新目录`/home/lzf/ldx/outputs/tbps-framework-stage25/`，其中：

- `irra_equivalence_v1/attempt/`：开始前原子发布的输入/资源、2236个相关来源SHA、Manifest及完成回执。
- `irra_equivalence_v1/result/`：报告、Manifest和完成回执；报告绑定attempt Manifest SHA。
- `preflight.log`、`gpu_smoke.log`、`cpu_before.log`、`cpu_after.log`、`legacy_cpu.log`、`protection_result.json`、`receipt_audit.log`：本轮运行/CPU证据，不提交Git。

输出目录只接受新的tbps-framework命名空间，拒绝历史树。attempt或pending/失败证据存在时拒绝GPU重入，不覆盖也不自动恢复。SIGINT/SIGTERM、OOM、差异超限等保留失败报告及已计数调用；SIGKILL可能只留下attempt/pending，不承诺终止时一定完成报告。需要人工审查后另行授权，不自动修参或重试。

只写新框架输出；不保存或修改模型Checkpoint/特征缓存。报告数值属于小规模等价性验证，不是攻击效果或Full指标。尚未验证其他输入、batch/dtype/硬件、所有文本截断、Full排名、G/H插件、其他模型和数据集。历史特征仍不能直接标记为新ModelSpace。
