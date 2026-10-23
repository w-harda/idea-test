# 独立RNG门槛与单次Smoke后台监督

本轮授权在008a1c8基础上仅调整operations层。生产树、配置及probe_v2不可变。

## 独立RNG

rng_resource_protocol只将独立CUDA随机数验证的最低空闲显存设为2048MiB，RAM/磁盘/inode沿用原值。
返回独立字典，不修改原Smoke的10240MiB。真实验证使用原G60/H10 RNG与原restore/capture及阶段切换。
只允许18次小CUDA随机Tensor调用，不进行神经前向、JVP或Optimizer。cuda_rng_v1如有任何Claim/FAIL/completed均拒绝重做。
验证成功后独立Manifest成为监督器启动门禁；不重签原probe_v2。

## 后台状态和单次执行

run_gh_rng_smoke.sh仅exec标准库监督程序gh_smoke_watch.py；nohup+setsid脱离SSH。
WATCHING：flock唯一实例，未创建smoke_started.json，也未消费原生smoke_v2。
每180秒查询GPU/RAM/磁盘/inode；利用率仅记录。
首次快照已经合格时允许立即完整检查；否则连续两次满足门槛才进入CPU Preflight。
CPU Preflight后核验来源、probe/RNG回执与历史保护，再做最终即时资源检查。资源回落继续WATCHING，不写消费标记。

READY：技术与资源门禁通过。
真正Popen前以O_EXCL+fsync持久化smoke_started.json，绑定来源、回执、脚本SHA及资源快照；崩溃后也不得重试。
SMOKE_RUNNING：原native_cli smoke进程已经创建，原生Claim/FAIL/completed语义不变。
SMOKE_PASS：生成四Round完成；随后独立CPU进程运行原audit-smoke。
SMOKE_FAIL：实际子进程、来源、审计或监督异常；保留现场，不自动重试。

监督器不导入Torch/CUDA，不实现任何G/H/JVP/Reward计算。监督日志分别记录控制流、CPU审计及feedback_effects，不能据运行成功宣称性能提升。
SIGTERM仅在等待阶段结束等待；Smoke消费后不通过“停止等待”信号中断实验。
原GPU/前向/JVP/更新预算、固定8样本、4×2048图库、A1/B两轮协议全部保持。

## 当前生命周期的CPU检查

旧准备态测试要求probe目录不存在，历史失败记录保留，不修改/删除旧断言。
新增CPU测试要求同来源probe PASS存在且Smoke目录不存在；任何Claim、FAIL、pending、completed或未知既有Smoke目录一律拒绝。
19项CPU测试通过，包含原5项RNG组合测试及资源、生命周期、互斥、单次消费和失败停止覆盖。

## 目录

独立RNG执行日志：/home/lzf/ldx/tmp/gh_rng_smoke_monitor_v1_20261023/。
后台监督控制：/home/lzf/ldx/tmp/gh_smoke_wait_v1_20261023/。
独立RNG产物：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/cuda_rng_v1/。
原Smoke产物：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/smoke_v2/。
旧资源阻塞控制目录保持不变，不复用其中rng_called.json或错误日志。
