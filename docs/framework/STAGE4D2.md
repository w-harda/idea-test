# Stage4D-2：单次G全量GPU生成与Joint封存

本入口只物化攻击输入，不运行IRRA/RaSa/APTM、排名、训练或JVP。实际完成状态见STAGE4D2_REPORT.md。

## 固定库与Plan

生产脚本位于新Stage4D2工作树，但所有tbps公共库、G插件、stream_images及Store均从冻结Stage4D1工作树导入。运行时核验模块真实路径、原Git Commit/clean及Plan ID；禁止用新树的producer路径重构计划。

- 冻结库：/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src。
- Plan：7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。
- 仅追加Store：/home/lzf/ldx/outputs/tbps-framework-stage4d1/materialized_full_v1/artifacts/。
- 新控制与日志：/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v1/。
- 新入口full_gpu.py、gpu_safety.py、full_gpu_audit.py另由stage4d2_generation.server.json逐文件SHA绑定，不改原Plan/文本回执。

## 实际命令

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src
export CUDA_VISIBLE_DEVICES=""
```

默认仅CPU来源预检（外部nvidia-smi查询资源，不初始化CUDA）：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B \
  /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017/src/tbps/materialization/full_gpu.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017/configs/framework/stage4d2_generation.server.json
```

预期GH_RUNTIME_PREFLIGHT_PASS、G_FULL_CPU_PREFLIGHT_PASS。预检重新读取完整历史文本、动作/NOOP、原图SHA及严格导入闭包；G权重不反序列化。

显式一次GPU授权命令（本轮资源门禁阻断，尚未执行；资源就绪并获用户继续指示后使用，目录消费后禁止再次执行）：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B \
  /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017/src/tbps/materialization/full_gpu.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017/configs/framework/stage4d2_generation.server.json \
  --allow-gpu-generation
```

父进程保持CPU环境，子进程显式CUDA_VISIBLE_DEVICES=0；GPU子进程必须验证不可变CPU launch gate和单次Claim，再恢复G10一次。3074图按官方顺序batch1，复用原PIL RGB/bilinear384×128/ToTensor及原投影。真实模块Hook计数，逐图Claim→G→保存三FP32 Tensor→读回/证明。两张旧Test样本对比原/Adv/effective_delta内容SHA；不一致保存失败证据并停止。

GPU子进程完全退出后，父进程再执行CPU历史H完整校验与Joint封存。只引用原6156文本和新3074图，不修改原TEXT_FULL_MATERIALIZED回执中的历史PENDING字段；新joint-full回执表示当前完整集合。

查看进度与资源：

```bash
tail -n 3 /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v1/gpu.log
nvidia-smi -i 0
```

日志每图G_FULL_PROGRESS，含实际started/completed、单元耗时、累计秒数、内容/文件SHA、RSS、显存与磁盘/inode；完整模型冻结/计数在gpu-result/report.json。完成标志IMAGE_FULL_MATERIALIZED及JOINT_FULL_MATERIALIZED。

Ctrl-C/超时仅停止本入口启动的子进程；失败保留已完成单位、Claim/pending和日志，无自动重试/恢复。正式Attempt目录已存在时入口拒绝；新恢复Attempt需要现场审计及用户再次授权。不能通过删除Claim或换目录重复未提交前向。

独立CPU只读验收（无模型前向；只允许已声明pixels.pt的weights_only读取）：

```bash
/home/lzf/ldx/envs/tbps-image/bin/python -B \
  /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017/src/tbps/materialization/full_gpu_audit.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2-20261017/configs/framework/stage4d2_generation.server.json \
  --output /home/lzf/ldx/outputs/tbps-framework-stage4d2/user_cpu_audit_v1
```

独立Joint验收拟用cpu_audit_v1（本轮尚未执行）；新输出存在时拒绝重写。成功标志JOINT_FULL_CPU_AUDIT_PASS，核对G Hook、逐图日志、Claim/Proof、Tensor预算/SHA、Gallery/Query顺序、完整Unicode/action、旧30797文本文件、7991历史文件、3472缓存与70旧FAIL文件保护。

CPU测试在新树使用自身源码（Fixture可导入副本；生产路径校验仍只允许冻结库）：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/framework tests/test_dataset_adapter.py tests/test_dynamic_topk.py \
  tests/test_generator_framework.py tests/test_joint_baseline.py
```

## 资源协议与后续使用

起始空闲磁盘≥10GiB、inode≥100000、系统可用RAM≥8GiB、RTX4090空闲显存≥8192MiB且无其他计算进程。运行时系统RAM安全余量2GiB、磁盘安全余量2GiB，时间预算7200秒；记录RSS，不设置每进程RSS硬上限。监测其他GPU进程时只停止自身任务，不抢占或终止其他用户进程。

后续目标Adapter只通过MaterializedUnit/FileRef读取同一原/Adv FP32 RGB与Unicode，执行各自原生预处理/Tokenizer并创建独立ModelSpace表示及Score缓存。Gallery/Query分别有序读取，不能zip配对或把IRRA Embedding当跨模型输入。G/H已冻结，目标反馈不得改变产物。

本轮不执行受害者Full编码/排名。后续最小计划为单独授权目标Adapter的固定小样本输入/编码一致性，再决定完整评测；不得把JOINT_FULL_MATERIALIZED当作跨模型攻击效果通过。
