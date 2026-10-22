# CPU验收清单

最终源码：SOURCE_SHA256.json；协议摘要见SOURCE_EVIDENCE.json。
固定环境：Python3.10、Torch2.5.1+cu124、NumPy1.26.4，CUDA_VISIBLE_DEVICES为空。
真实GPU数值不在本轮验收范围。

| 范围 | 用例数 | 结果 |
| --- | ---: | --- |
| 原交替原型、属性审计、契约/配置/存储 | 162 | PASS，原断言保留 |
| 新门禁/隔离/梯度控制/状态与设备绑定 | 49 | PASS，合成CPU及meta设备 |
| 真实CPU父恢复、BPE/Span、组装与原bank读取 | 10 | PASS，无真实神经前向 |
| 完整2048刷新调度Mock、失败Claim与顺序拒绝 | 3 | PASS，不生成真实图库 |
| 合计 | 224 | PASS，0 skip，退出码0 |

另有3项真实CLI检查：probe无授权拒绝、smoke无授权拒绝、smoke缺少数值探针回执拒绝。
均在Attempt、CUDA及模型加载前停止，输出目录保持不存在。
真实CPU预检走同一prepare_cpu_gate控制流，只有cpu_sources_verified→cpu_isolation_exited两个事件。

服务器复核命令：

~~~bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/home/lzf/ldx/tmp/gh_bidirectional_backend_20261022
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/src
export TBPS_NATIVE_TEST_CONFIG=/home/lzf/ldx/projects/idea-test-tbps-gh-bidirectional-backend-20261022/configs/framework/gh_bidirectional_backend.server.json
/home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q -p no:cacheprovider \
  --basetemp=/home/lzf/ldx/tmp/gh_bidirectional_backend_20261022/pytest_manual_review \
  tests/framework/test_gh_alternating.py tests/framework/test_gh_attribute_audit.py \
  tests/framework/test_artifacts.py tests/framework/test_contracts.py tests/framework/test_config.py \
  tests/framework/test_alternating_native.py tests/framework/test_alternating_native_server.py \
  tests/framework/test_alternating_native_bank.py
~~~

真实CPU集成测试需显式TBPS_NATIVE_TEST_CONFIG；本轮实际设置并全部执行，没有通过skip冒充通过。
测试Fixture中的篡改、设备重定向、参数替换和新Adam更新均限定合成对象，不修改正式模型。
GPU峰值、CUDA RNG恢复、真实梯度和Reward差仍PENDING。

开发日志保留于/home/lzf/ldx/tmp/gh_bidirectional_backend_20261022/：
最终pytest为release_verified.log，实际CPU门禁为preflight_verified.log；
此前错误及其退出码未删除或改写，均未创建GPU Attempt。
资源快照来自较早preflight_gate_v1.log，保存在CPU_PREFLIGHT_RESOURCE_SNAPSHOT.json；
最终源码的门禁报告为CPU_PREFLIGHT.json，不把较早源码验证冒充当前数值验证。

151378历史文件及10918上一审计来源metadata零变化；来源/Checkpoint及139个当前producer、
496项runtime pin核验。真实导入执行56份旧源码。不是声称对全部151378文件重做了内容SHA扫描。
唯一已知warning为Kornia的torch.cuda.amp.custom_fwd弃用提醒，不代表CUDA初始化或模型执行。
