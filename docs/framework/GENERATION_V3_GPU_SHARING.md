# generation_v3：固定Ollama身份的最小GPU共享修改

基于Stage4D2R2 Commit `029a0817f764b8241ca1790bbf335fdd144ed9a9`，独立分支 `codex/tbps-generation-v3-gpu-sharing`。这只是当前Runner的资源许可修改，不新增攻击方法或工程阶段。本轮只进行CPU开发与测试，未执行正式预检入口、GPU生成或全量审计；generation_v3仍未创建。

## 修改边界

- 源码仅修改当前 `full_gpu.py` 和 `gpu_safety.py`。冻结Stage4D1、原G/H、Claim恢复、全量Store、图像预处理、8/255及所有模型权重保持原样。
- 仍使用原Plan `7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5`、原6156条文本及唯一首图Claim。原generation_v1/v2永久FAIL，generation_v3身份不变，不创建v4。
- 配置位于本独立工作树 `configs/framework/stage4d2r2_generation.server.json`，旧工作树同名配置未修改。新配置SHA：`ea66d70c140313abb7539e1f3eedf1ea879d43939efc44fe6626343d4d40cbfb`。
- 新配置绑定4项当前入口源码、原恢复配置及3项原预处理来源。恢复配置内容SHA仍为 `429209225c9a947b1dd254b51addbf41c9ef5a5e8dc0532ea7dfbc3396604bd6`。

## 唯一共享许可与停止条件

仅允许GPU0上原Ollama PID **2436556**。绑定机器boot ID、进程start_ticks、UID/GID、comm、完整argv的SHA、cgroup、父PID及父进程同等身份，并核验固定二进制SHA及nvidia-smi报告路径。没有按名称或UID放行其他进程的通配规则。

只读确认的身份：UID997/GID984，start_ticks=2238557017，父PID277892/start_ticks=1787729494，cgroup为ollama.service；固定二进制SHA为 `234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b`。完整身份保存于配置。跨UID的/proc/PID/exe链接不可读，未声称核验该链接；实际使用可读proc身份、父服务身份、argv0二进制及GPU进程报告的组合核验。

- 启动：空闲显存≥3584MiB、GPU利用率≤10%，只允许该Ollama计算进程。
- 运行：仅新增本次GPU子进程自身；至少每秒采样其他计算进程身份及空闲显存，最低安全余量固定1024MiB。GPU利用率可因本次G前向升高，不把启动利用率门限误用于运行中。
- 资源查询前后均核验固定身份；任何未知PID、PID复用、Ollama退出/重启、命令行或来源变化、无法读取身份、查询失败/超时、空闲低于安全余量均停止。后台资源监督把检查异常转换为安全停止信号，仅管理本任务创建的子进程。
- 不终止Ollama，不等待/轮询资源就绪，不自动重试。身份变化后需重新审查，不自动替换PID或放宽配置。
- RAM≥8GiB、启动磁盘≥10GiB、inode≥100000、运行RAM/磁盘余量及7200秒预算保持不变。资源采样存在最多约1秒间隔及外部查询延迟，不保证共享服务突发分配时不会OOM；OOM同样保留FAIL并停止。

## 用户手动执行：环境与只读确认

以下命令仅供用户手动运行，Codex没有执行。生产PYTHONPATH必须指向冻结Stage4D1，禁止用新工作树src重新构造Plan。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src

test ! -e /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3 && echo GENERATION_V3_NOT_CREATED
sha256sum /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/configs/framework/stage4d2r2_generation.server.json
sha256sum /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/configs/framework/stage4d2r2_recovery.server.json
```

应分别得到上文新配置SHA和原恢复配置SHA。任何不一致先停止。

## 用户手动执行：正式CPU预检

```bash
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
    /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/src/tbps/materialization/full_gpu.py \
    --config /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/configs/framework/stage4d2r2_generation.server.json \
    > /home/lzf/ldx/tmp/generation_v3_shared_preflight.log 2>&1
)
```

成功为退出0、`GH_RUNTIME_PREFLIGHT_PASS`及`G_FULL_CPU_PREFLIGHT_PASS`。CPU来源检查后包含一次共享资源检查，不初始化CUDA、不创建Attempt。资源不足或身份不符直接退出，不自动等待。

## 用户手动执行：nohup唯一GPU Attempt

以下完整命令包含环境与未创建检查；GPU入口内部再次执行完整CPU预检，全部通过后才创建Attempt。不要再套自动重试循环。

```bash
(
  set -o noclobber
  cd /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017 || exit 1
  test ! -e /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3 || { echo "generation_v3 already exists; STOP"; exit 1; }
  export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src
  export CUDA_VISIBLE_DEVICES=0
  nohup /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
    /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/src/tbps/materialization/full_gpu.py \
    --config /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/configs/framework/stage4d2r2_generation.server.json \
    --allow-gpu-generation \
    > /home/lzf/ldx/tmp/generation_v3_shared_gpu.log 2>&1 < /dev/null &
  echo "launcher PID=$!; inspect log for preflight/launch status"
)
```

日志使用noclobber；若同名日志已存在，不覆盖或自动换Attempt，先检查已有运行。成功要求子结果 `IMAGE_FULL_MATERIALIZED`、父结果 `JOINT_FULL_MATERIALIZED`。仍要求G恢复1次、实际图像前向3074；H/IRRA/RaSa/APTM、Backward/Optimizer/JVP均0。

输出保持 `/home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3/`，图像/文本Store及Plan不变。逐图日志 `gpu.log` 增加GPU共享资源快照；`gpu-result/report.json` 保存最低观测空闲显存与停止原因。原首图恢复仍单独保存Start/Proof，不删除旧Claim。

## 用户手动执行：GPU成功退出后的独立CPU审计

```bash
(
  set -o noclobber
  CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
    /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/src/tbps/materialization/full_gpu_audit.py \
    --config /home/lzf/ldx/projects/idea-test-tbps-generation-v3-sharing-20261017/configs/framework/stage4d2r2_generation.server.json \
    --output /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3_cpu_audit_v1 \
    > /home/lzf/ldx/tmp/generation_v3_shared_cpu_audit.log 2>&1
)
```

先运行本页环境设置。成功为退出0及 `JOINT_FULL_CPU_AUDIT_PASS`；此CPU审计不要求Ollama仍存活，也不调用GPU。

一次性查看日志：

```bash
tail -n 20 /home/lzf/ldx/tmp/generation_v3_shared_gpu.log
tail -n 5 /home/lzf/ldx/outputs/tbps-framework-stage4d2/generation_v3/gpu.log
tail -n 20 /home/lzf/ldx/tmp/generation_v3_shared_cpu_audit.log
```

## CPU验证及历史保护

完整CPU回归687项通过，88.78秒（原647＋新增40）；最后补充“拒绝后不再查询”及显存错误详情后，40项共享测试最终复核通过，2.54秒。最初定向组合57项通过，16.11秒。覆盖固定进程完整身份、启动与运行阈值、未知进程/PID复用/退出/二进制SHA拒绝、外部查询失败不重试、每秒监督及真实child的G加载前阻断；原预处理、首图恢复和3074顺序CPU Mock回归通过。

回归日志SHA：`d8771d04fdc0b0393775b7d0106de9abc10b04e04149d0ba3cbc88604080354b`；最终共享测试日志SHA：`5d8e9e02f2eef9e22996c699c87013a3e4714511d7aff0405fe2fbad65de2599`。日志仅存在新CPU证据目录 `/home/lzf/ldx/outputs/tbps-generation-v3-sharing-cpu/`，不提交。

保护检查：原R2工作树580文件SHA/mtime相同；generation_v1/v2及首图Claim共35文件SHA/mtime相同；原物化Store30800文件成员、size/mtime相同；冻结Stage4D1 HEAD/clean不变。本轮没有重算整个Store内容SHA或重复7991/3472全量历史审计，引用原R2独立CPU审计证据。保护报告SHA：`19c7b9b0506108e11868bc8159f58dad659fe96477b42d223876b149a96adaf8`。

实际GPU共享生成尚未验证；CPU通过不能提升Image/Joint完成状态。本轮停止，用户审查后自行启动。
