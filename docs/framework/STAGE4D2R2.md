# Stage4D-2 generation_v2失败后的CPU修复与generation_v3执行说明

> 本文保留原R2独占资源命令作为历史说明。generation_v3当前共享许可、独立工作目录及新配置SHA以[GENERATION_V3_GPU_SHARING.md](GENERATION_V3_GPU_SHARING.md)为准；不要使用本文旧树命令启动共享模式。

基准dd0a213f2ac1b787412b755372c378ad49f31f9c，独立codex/tbps-framework-stage4d2r2。当前仅完成代码和CPU验证；下列GPU与全量审计命令供用户后续手动执行，本轮未运行。

## 修复与恢复边界

- GObservation只在原_LegacyG.load_pixels的受限读取作用域中允许原torchvision Resize。原pixels_from_pil、backend及torchvision源码分别SHA绑定；输入必须是当前官方顺序中的已绑定图像。只允许RGB PIL、bilinear384×128、原antialias/max_size、无Hook和无参数/buffer的原Resize。ToTensor本身不是nn.Module，由固定原读取函数执行，不增加通用神经网络白名单。
- 作用域之外的Resize、其他非G神经网络、错误图像、未验证读取器/来源和异常设置继续拒绝。G不能在读取作用域内前向；G实际Hook、单次加载、优化禁令和原预处理数值均保留。
- 唯一恢复对象为原Plan首图cuhk:37132:0，原Claim unit_id=50b19770bb01c5c7e52237a4af73e4186342e77ccafe06204a8c3c91ecbfe3b7。配置固定generation_v2完整FAIL文件SHA/mtime、G恢复1次且实际前向0/0、唯一Claim的3文件、无Proof/图像/pending及原身份。
- 当前Full必须恰好等于generation_v2原30797文件快照加这3个Claim文件；任何其他新增/篡改或未完成Claim拒绝。generation_v1/v2整体FAIL不提升，旧Claim不删除、覆盖或重新签名。
- CPU预检及child的fresh_images显式核验恢复资格。新恢复调度只对首图一次原generate_image_once的_prepare调用安装单线程局部能力，原_finish照常保存正常Proof；其他Claim走原保护。
- 首图成功后，其余3073图复用冻结Stage4D1 stream_images和同一G工厂/模型会话。原Runner内部读回跳过刚完成首图1次；公开本次统计generated=3074、skipped=0、recovered_first=1，另记录legacy_stream_generated=3073、legacy_stream_skipped=1，不掩盖内部调度行为。
- 新Attempt额外发布claim-recovery-start及claim-recovery-proof，绑定旧FAIL、原Claim、新Attempt、正常Proof及新像素单元。失败保留现场，不自动重复恢复。

原Stage4D1源码、Plan、6156文本、G10/H7、8/255、PIL读取及全部攻击算法不修改；新入口只追加独立恢复证明与图像。CPU-only资格回执不等于首图已生成。

## 工作目录与冻结环境

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src
export CUDA_VISIBLE_DEVICES=""
```

必须使用冻结Stage4D1的PYTHONPATH，不能用新树src构造另一Plan。新配置固定输出generation_v3，v1/v2目录永不重用。资源仍为无外部GPU计算进程、空闲≥8192MiB、利用率≤10%、RAM≥8GiB、磁盘≥10GiB、inode≥100000；总预算7200秒，不设2GiB RSS硬上限。

## 默认CPU预检：用户后续手动执行

```bash
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B \
    /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017/src/tbps/materialization/full_gpu.py \
    --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017/configs/framework/stage4d2r2_generation.server.json \
    > /home/lzf/ldx/tmp/stage4d2r2_generation_v3_preflight.log 2>&1
)
```

退出0、GH_RUNTIME_PREFLIGHT_PASS及G_FULL_CPU_PREFLIGHT_PASS才算成功。含一次外部GPU资源查询，不初始化CUDA或创建Attempt；不满足即停止，不降门限、不轮询重试。完整来源、旧文本与唯一Claim资格在此重新核验。

## generation_v3唯一GPU执行：需另行明确授权

```bash
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B \
    /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017/src/tbps/materialization/full_gpu.py \
    --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017/configs/framework/stage4d2r2_generation.server.json \
    --allow-gpu-generation \
    > /home/lzf/ldx/tmp/stage4d2r2_generation_v3.log 2>&1
)
```

父入口再次完整预检后才创建generation_v3。成功要求子结果IMAGE_FULL_MATERIALIZED、父结果JOINT_FULL_MATERIALIZED且退出0。G10本Attempt恢复一次、实际前向3074；H/IRRA/Backward/Optimizer/JVP全0。旧两图像素SHA对照保持原标准。

输出/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3/，逐图日志gpu.log；gpu-result/report.json记录实际Hook、冻结、资源、恢复与原Runner计数。claim-recovery-start/recovery.json和claim-recovery-proof/recovery.json分别记录恢复消耗与完整证明。原Store仅追加图像与Image/Joint集合，旧6156文本和旧首图Claim保持。

## GPU成功并退出后的独立CPU全量审计

```bash
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B \
    /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017/src/tbps/materialization/full_gpu_audit.py \
    --config /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017/configs/framework/stage4d2r2_generation.server.json \
    --output /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3_cpu_audit_v1 \
    > /home/lzf/ldx/tmp/stage4d2r2_generation_v3_cpu_audit.log 2>&1
)
```

成功标志JOINT_FULL_CPU_AUDIT_PASS及退出0。审计新增旧v1/v2 FAIL、原Claim、恢复Start/Proof与正常生成Proof的一致性，核验恢复只发生1次。其余3074图/6156文、预算、顺序、来源及历史保护沿用原独立CPU验收。

## 手动查看一次进度

```bash
tail -n 20 /home/lzf/ldx/tmp/stage4d2r2_generation_v3_preflight.log
tail -n 20 /home/lzf/ldx/tmp/stage4d2r2_generation_v3.log
tail -n 5 /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3/gpu.log
tail -n 20 /home/lzf/ldx/tmp/stage4d2r2_generation_v3_cpu_audit.log
```

Ctrl-C或超时仅停止本入口创建的子进程。任何FAIL都保留目录、旧Claim、新恢复记录和已提交单位，禁止删除后重跑。generation_v3已存在时拒绝复用；下一次恢复必须重新审查并授权。无需人工清理任何旧文本或旧Claim。

## CPU测试命令

测试使用新树自身源码和临时合成Fixture；不运行真实GPU、模型权重或正式物化：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4d2r2-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q -p no:cacheprovider tests/framework tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```
