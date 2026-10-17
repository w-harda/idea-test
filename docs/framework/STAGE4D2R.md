# Stage4D-2R：GPU启动顺序CPU-only修复

开发基准883c4be59284b79f475412eb40ea6d96e8b1742a，独立分支codex/tbps-framework-stage4d2r。原工作树、原generation_v1 FAIL及其源码/配置不修改。本轮不授权GPU、新Attempt或真实图像生成。

## 修复与门禁

full_gpu.child保留CPUStartupGate的全部检查。完成CPU来源/Plan/文件检查并退出cpu_only后，checked/exit_checked验证Torch函数、导入设置和legacy锁恢复。资源、线程、seed和精度配置完成后，先initialize_cuda，再立即安装no_optimization和g_load_only，只有两个上下文均成功才进入G工厂。

资源guard仍覆盖CUDA初始化与后续执行。权重加载次数、实际G Hook、非G神经前向阻断、Claim先于前向、原子保存、读回、禁止Backward/Optimizer/JVP与冻结检查保持。Stage4D1公共库、G/H算法/权重、预算、预处理不改。

新入口固定generation_v2，原generation_v1不可重用。原stage4d2_generation.server.json原样保留；新增stage4d2r_generation.server.json与derivation，仅output及独立entry_sources路径/SHA变化。派生记录GPU_execution_authorized=false表示本轮没有执行许可，未来仍须用户明确授权。资源阈值、7200秒、3074/6156、原Plan及冻结Stage4D1 config不变。

## CPU回归

所有命令只在独立修复工作树执行：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q -p no:cacheprovider tests/framework tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

新增测试调用真实child、CPUStartupGate、安全包装、source_execution_audit、流式Runner、G Hook及生成Claim/Proof事务。仅使用合成CPU模型、Mock权重入口和Mock CUDA reset；真实CUDA init/lazy/native入口硬拒绝。两个Fixture图像不能冒充3074，实际child在完成小样本流式事务后仍由原固定3074总数检查拒绝，这是预期边界。Joint封存仅在临时合成Fixture中验证，没有发布正式Image/Joint。

错误顺序必须被原门禁拒绝；来源/Plan/导入路径/锁/RAM/资源异常、初始化或包装安装失败均不得进入G；包装生效时拒绝非G权重、非G前向及优化。异常后函数、导入环境和锁须恢复。

## 下一次GPU方案：仅设计，尚未授权或执行

1. 审查本修复及CPU证据，另行明确授权generation_v2唯一Attempt。
2. 保持冻结Stage4D1 Plan与6156文本；确认v1永久FAIL、v2尚不存在、正式Store仍无图像或未提交逐图Claim。
3. 检查无外部GPU计算进程、空闲≥8192MiB、利用率≤10%、RAM≥8GiB、磁盘≥10GiB及inode≥100000。未通过即停止。
4. 使用下列修复入口重新执行完整来源与资源预检；预检不创建Attempt。
5. 仅在另行授权且预检PASS后运行显式GPU命令。新attempt/control、entry_sources及日志绑定修复后Runner SHA，原Plan及H回执不改。
6. 一次恢复G，3074张batch1，旧两图像素SHA对照、读回及Image/Joint封存、退出后独立CPU全量验收沿用原代码。失败保留现场，不自动重试。

下列命令仅供未来获授权后使用，**本阶段未运行**：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017/src/tbps/materialization/full_gpu.py --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017/configs/framework/stage4d2r_generation.server.json
```

显式GPU入口（仅未来单独授权后）：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017/src/tbps/materialization/full_gpu.py --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017/configs/framework/stage4d2r_generation.server.json --allow-gpu-generation
```

独立CPU全量审计（只在未来全量成功、GPU进程退出后）：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017/src/tbps/materialization/full_gpu_audit.py --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r-20261017/configs/framework/stage4d2r_generation.server.json --output /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v2_cpu_audit_v1
```

正式输出仍在原物化Store追加图像；新控制回执为/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v2/。无需清理、移动或重签6156文本；v1 G前向/新增图像均0，没有生成进度需要恢复。当前Image/Joint全量及两图新输出对照仍PENDING。
