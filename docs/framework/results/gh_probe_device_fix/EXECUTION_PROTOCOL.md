# probe_v2执行协议（本轮未执行）

## 先决条件与边界

1. 用户新的明确GPU授权。本轮授权仅CPU，不能执行下面的GPU命令。
2. 正确独立工作树与已发布修复Commit；工作树干净，配置文件SHA与SOURCE_EVIDENCE一致。
3. 140项producer、496项legacy来源、G60/H10/IRRA、原数据和完整父状态核验PASS。
4. 原probe_v1/FAIL及旧Watcher证据保持；不得删除或复用旧Claim。
5. probe_v2及smoke_v2不存在任何Claim/pending/FAIL/completed/其他文件。
6. 无相同probe_v2实例；旧Watcher已经PROBE_FAIL退出、启动标记已消费，不重新启动它。
7. 即时GPU0空闲显存≥10240MiB，RAM≥32GiB，磁盘≥25GiB，inode≥80000。
   新配置把10GiB作为native_cli内部硬门槛；利用率只记录。
   来源/资源不足时在Attempt/CUDA前停止，不降低门槛、不抢占、不自动重试。

## 固定数值范围

- cuhk:26530:0：NOOP。
- cuhk:26747:0：原合法非NOOP候选，不更换。
- G图像≤4，IRRA图像≤4，IRRA文本≤6，JVP≤1，reader≤1，H策略≤1。
- Optimizer.step=0，完整奖励图库刷新=0，Validation/Test/Full=0。
- 原FP32/math/no_tf32、原8/255投影；不改变梯度/JVP/FD数学、K或Loss/Reward。
- 原始文本缓存对照≤1e-6，NOOP梯度≤1e-6，非NOOP梯度差>0，原FD阈值0.02。
- 本次CPU设备修复不能替代上述真实数值验收；CUDA RNG恢复仍必须以真实证据判断，不补写PASS。

输出固定为：
/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/probe_v2/。

保留新Smoke身份：
/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2/。

本阶段不授权Smoke。v2 Smoke的原门禁将只查本配置probe_v2的同来源回执，
不会引用probe_v1。默认GPU授权字段保持false，不能通过改JSON开门。

## CPU复核命令

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/home/lzf/ldx/tmp/gh_probe_device_fix_20261023
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023/src
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.alternating.native_cli \
  --config /home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023/configs/framework/gh_bidirectional_backend.probe_v2.server.json \
  --action preflight --check-parent-states
~~~

PASS标志：GH_NATIVE_CPU_PREFLIGHT_PASS、GH_NATIVE_PARENT_CPU_RESTORE_PASS、
GH_NATIVE_IMPORT_CLOSURE_PASS；startup_events不能含CUDA初始化。

## GPU命令（仅供未来新授权后执行；本轮禁止）

先按上方命令设置工作目录与环境，核验配置原始字节SHA：

~~~bash
printf '%s  %s\n' \
  '8b6fce14ce4236f76bd66233c061f2f06a32d975d6771cbad77f15a45ea5e443' \
  '/home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023/configs/framework/gh_bidirectional_backend.probe_v2.server.json' \
  | sha256sum -c -
~~~

经新的GPU授权后，单次前台执行入口如下；需要后台时可外层nohup setsid，仍只调用同一入口：

~~~bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.alternating.native_cli \
  --config /home/lzf/ldx/projects/idea-test-tbps-gh-probe-device-fix-20261023/configs/framework/gh_bidirectional_backend.probe_v2.server.json \
  --action probe --allow-gpu-probe \
  > /home/lzf/ldx/tmp/gh_probe_device_fix_20261023/probe_v2_execution.log 2>&1
~~~

本轮没有执行上述命令，也没有创建该日志或Attempt。
任何真实失败保留probe_v2/FAIL及已有Claim/pending，不能自动重试或另建v3。

成功时必须有GH_NATIVE_NUMERIC_PROBE_PASS及completed/manifest.json、completed/completed.json、
completed/report.json，全部按新source_key/prototype协议验证后才能报告PASS。
原失败入口的问题已最小修复，但不能据CPU通过保证下一次GPU数值必然通过。

## 关于未来轮询

本轮不需要自动轮询，未实现或启动v2 Watcher。
如果未来授权自动等待，必须用独立gh_native_probe_wait_v2控制目录、明确锁和一次性启动标记，
固定新配置/脚本SHA，仅消费probe_v2；启动前核验原Watcher终态、旧锁空闲和原probe_v1证据不变。
绝不把原固定v1 Watcher直接指向新输出，也不清理其已消费标记。
