# Stage 3B：单次真实G/H Smoke——FAIL现场与安全边界

**本次唯一GPU尝试失败，未产生任何G/H神经前向或输出对照。不能标记VERIFIED。** 详情见[报告](STAGE3B_REPORT.md)。

## 新入口与文件

- src/tbps/smoke_attacks.py：保存本轮实际执行的脚本，包含固定样本、单次attempt/回执、G/H加载和前向预算、完整动作对照及三种Joint模式计划。为保持失败证据的源码SHA，执行后未修改此文件。
- configs/framework/stage3b_smoke.server.json：锁定两个Train样本、资格、G/H来源、4/4/0上限、FP32/seed42及1e-6阈值。
- src/tbps/check_attack_runtime.py：失败后新增的严格CPU实际运行闭包检查，无GPU/run选项。
- tests/framework/test_stage3b_smoke.py、test_stage3b_import_blocker.py：比较/计数/单次门禁，以及当前缺项必须拒绝的CPU回归。

独立工作树：
/home/lzf/ldx/projects/idea-test-tbps-framework-stage3b-20261016。
分支codex/tbps-framework-stage3b，基准ef835a60a85c30910ef4a3442230882f72a05c44。

## 当前阻断项

原GPU脚本的部分CPU预检通过，但漏查G像素读取函数的传递导入：
irra_image_adapter → cache_io → condition_schema。

G真实权重加载/转GPU后，严格导入闸门拒绝未列入runtime_sources的condition_schema.py，尚未恢复H权重。实际G/H/IRRA前向均0。

当前smoke_attacks默认CPU入口只证明其既有检查范围，不是完整运行闭包许可。**不能依据GH_SMOKE_CPU_PREFLIGHT_PASS启动GPU。** 新严格检查覆盖真实G像素读取链及H状态检查链，当前必须返回拒绝；源缺项未修复。

## 可安全执行的CPU检查

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage3b-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.check_attack_runtime --config configs/framework/stage3b_smoke.server.json
```

当前预期exit code=1：
GH_RUNTIME_PREFLIGHT_REJECTED unpinned legacy import: .../condition_schema.py。
这是已确认阻断，不要放宽检查或据此继续GPU。

CPU回归：

```bash
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m unittest discover -s tests/framework -v
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

已执行的GPU命令仅作归档，**本轮授权已用完，禁止再次执行、换目录重试或恢复**：

```bash
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.smoke_attacks --config configs/framework/stage3b_smoke.server.json --output /home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v1 --allow-gpu-smoke
```

脚本拒绝已有attempt/pending/结果目录重入；SIGINT/SIGTERM/异常保留证据，不自动恢复。当前失败标志GH_PLUGIN_EQUIVALENCE_FAIL，不能因result目录含completed回执就当作科学验证PASS：回执只证明FAIL报告完整发布。

## 证据与只读核验

输出仅/home/lzf/ldx/outputs/tbps-framework-stage3b/。主要为equivalence_v1/attempt与result、gpu_smoke.log、preflight.log、strict_preflight.log、cpu_final.log、legacy_cpu.log、receipt_protection_audit.log及protection_result.json。

新回执/来源只读校验，无模型或排名：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -c 'from tbps.artifacts import ArtifactStore; s=ArtifactStore("/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v1",new_base="/home/lzf/ldx/outputs/tbps-framework-stage3b"); s.verify("attempt"); s.verify("result"); print("FAIL_REPORT_RECEIPTS_SHA_VALID")'
```

后续最小修复方案见报告，需要另行审查/授权；本轮未改Stage 3A来源配置、Plugin算法或历史资产，未继续Stage 4。
