# CUDA RNG与A1/B启动准备：RESOURCE_BUSY，未执行GPU

## 本轮结论

未运行真实CUDA RNG验证，未启动A1/B。状态RESOURCE_BUSY，不是CUDA数值FAIL。
独立脚本完成CPU来源、父模型/双Adam恢复、导入隔离退出及历史保护后，在初始化CUDA/发布Claim之前由资源检查拒绝。
没有cuda_rng_v1、smoke_v2、后台监督器或新GPU进程；没有Watcher、等待或自动重试。

## 资源记录

2026-10-23 13:02:05 +08:00诊断快照：
- RTX4090总24564MiB，空闲3865MiB，利用率0%。
- Ollama PID3352822占20178MiB，未干预。
- RAM可用114862227456 bytes，磁盘可用6178286866432 bytes，inode可用240213840。
- 原Smoke显存门槛10240MiB、RAM32GiB、磁盘25GiB、inode80000保持。RAM/磁盘/inode通过，GPU不通过。

新独立验证脚本采用同一原Smoke资源协议进行执行前保护，故在资源阶段停止；CUDA随机序列验证仍NOT_RUN/PENDING，不能标记CUDA_RNG_PASS或SMOKE_READY。
没有把旧probe_v2回执改写成CUDA RNG证明。

## 新增执行工具

在独立分支codex/gh-rng-smoke-launch和工作树/home/lzf/ldx/projects/idea-test-tbps-gh-rng-smoke-20261023新增：
- operations/gh_cuda_rng_validation.py：实际调用冻结restore_rng/capture_rng及NativeBackend阶段RNG方法；随机序列重放、G→H→G/H→G→H隔离、持久化读回；禁止神经前向、autograd及Optimizer。
- operations/gh_smoke_checks.py：同来源probe/RNG回执、唯一Smoke身份、历史保护与最终证据检查。
- operations/run_gh_rng_smoke.sh：单次后台锁/启动标记、原native_cli Smoke、完成后原audit-smoke，不包含攻击算法。
- tests/framework/test_gh_cuda_rng_validation.py：5项CPU组合测试。
- 本目录协议、CPU审查和证据。

上述操作工具实现Commit5018113ed989fcd01f187dccc440ae00ad99fcba；没有修改生产源码/configs/旧测试。
真实生产工作树仍为gh-probe-shared的aafc7715，数值生产代码仍a7597f9c；配置77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649，source_key不变。

## CPU测试应如实区分

新增5项RNG CPU组合测试PASS。首轮共74 passed/15 skipped，跳过项是尚未传专用配置的服务器测试。
随后有一次启动目录错误，pytest导入新操作树源码而配置指向冻结树，来源校验正确拒绝；原日志保留。
改用冻结工作树执行服务器测试后22 passed/1 failed：剩余失败是旧test_v2_identity_budget_and_complete_condition_sources最后要求probe_v2目录不存在。
该断言是设备修复阶段“准备执行探针”的历史生命周期约束，当前probe_v2已真实PASS并封存，不适用；未修改断言、未删除产物，也不将整套旧测试报告成全部PASS。
当前阶段检查的是既有probe SHA/PASS有效、smoke_v2不存在；真实CPU prepare_cpu_gate已完成。详见CPU_REVIEW.json及原始日志。

## 历史与数值边界

151378历史metadata、28份旧FAIL/Watcher SHA和metadata、G60/H10/IRRA关键SHA及probe_v2全部原artifact SHA/metadata在停止后再次核验不变。
原probe_v2 Manifest保持d6e0b960e20a1ba5128f9e0b4edf497e7ea4b098fbf5cd3125799d2ad1136764。
本轮CUDA初始化、神经前向、JVP、Optimizer更新、图库刷新和评测全部0。
脚本退出码1及trace保留在控制目录cuda_rng.log/cuda_rng.exit；错误发生在资源检查，不能写成真实CUDA RNG失败。
没有Smoke PID/PGID/SID和轮次进度，不能宣称已经后台启动。

控制目录：/home/lzf/ldx/tmp/gh_rng_smoke_v2_20261023/。
只读查看：cat /home/lzf/ldx/tmp/gh_rng_smoke_v2_20261023/phase；cat同目录resource_blocked.json；tail同目录cuda_rng.log。
完整日志/脚本SHA、资源及保护证据见SOURCE_EVIDENCE.json。

## 后续

本轮停止，不继续检查GPU、不启动任何后台任务。
CUDA RNG验证尚未执行，A1/B四轮也未开始；资源满足后仍需重新核验来源、唯一身份及即时资源，再按用户授权决定执行。
不得把本报告或CPU测试当作真实CUDA RNG/双向更新PASS。
