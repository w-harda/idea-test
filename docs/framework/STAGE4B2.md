# Stage 4B-2：单次CPU历史指标重放

本入口仅重放原Joint Full特征，不恢复模型、编码图像/文本或生成攻击。接口承接[Stage4B1](STAGE4B1.md)；验收与实际执行记录见[报告](STAGE4B2_REPORT.md)。原对抗像素SHA仍UNKNOWN，不能视为端到端攻击重新验证。

## 文件职责

- engines/replay_cpu.py：默认预检、独立显式授权、唯一Attempt、统一Engine及发布/读回调用。
- engines/replay_control.py：本进程RSS/耗时监视、信号中断、只读历史保护；临时CPU堆释放不改变数值。
- engines/replay_oracle.py：独立历史分块点积、官方rank加载、索引/指标/DR对照、二次原文件逐值检查。
- engines/evaluation.py：只补强历史Score读回类型、shape、FP32、有限性校验；Ranking/指标定义不改。
- tests/framework/test_stage4b2_replay.py：合成独立数值/非法Score/授权/单次/资源/身份/保护回归。

冻结stage4b2_acceptance.json原样保持，execution_authorized=false。配置表示验收条件，不授予运行权限；一次性命令行授权单独控制运行。

## CPU预检命令

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4b2-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.engines.replay_cpu
```

默认只读原FEATURES_VERIFIED回执及其全部来源、原协议/Selection/权重SHA、原有序成员和7991文件保护记录；固定Torch2.5.1+cu124、CPU2线程。成功标志CPU_REPLAY_PREFLIGHT_PASS，退出码0；不读取feature Tensor、不评分/排名、不创建正式Attempt。Checkpoint只SHA，不反序列化。

## 唯一一次正式命令（执行记录，不用于重跑）

本任务已授权该命令一次；是否完成见阶段报告。目录一旦存在，入口即拒绝再运行；不能换目录绕过门禁。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4b2-20261017
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.engines.replay_cpu --authorize-cpu-full-replay
```

固定输出：/home/lzf/ldx/outputs/tbps-framework-stage4b2/replay_v1/。

- attempt/：排他Claim后的预检/授权与完整来源Manifest。
- evaluation/：统一publish_result写出的四组Score、协议、指标和来源/完成回执。
- result/：全部验收通过才发布的METRICS_REPLAY_VERIFIED报告。
- failure/：发生异常或中断时保留原因、traceback、逐模式已取得的对照和资源状态。已发布evaluation不自动删除；evaluation存在本身不表示全部验收通过。

不能自动重试、恢复或覆盖。SIGINT/SIGTERM安全中断；资源监视每50ms取本进程RSS/HWM与单调时钟，达到2GiB或1800秒则发送本进程中断信号，由主线程退出并保留失败证据。Torch原生调用中的Python异常可能等到调用返回才交付；记录实测峰值，超过预算不能判PASS。监视线程仅持有标量，不持有Tensor/旧计算图；退出时回收线程并恢复信号处理器。

查看本次已授权任务进度：

```bash
tail -n 40 /home/lzf/ldx/outputs/tbps-framework-stage4b2/replay_v1.log
```

REPLAY_FEATURE_PROGRESS仅是载入；REPLAY_MODE_VERIFIED是独立官方数值对照完成；最终必须同时有METRICS_REPLAY_VERIFIED、退出码0和result完成回执。不自动运行任何CPU重放或后台任务。

## 独立比较与内存边界

先复用JointFullLoader加载四family并clone，再逐批独立torch.load/clone对照原文件内容exact。原始H动作/Span/Unicode/Tokenizer规则继续由Stage4B1验证。

EvaluationEngine.run(inputs,query_chunk=128)计算全部四组，新IRRACachedScorer评分。测试仪器只在原ranking调用处收集对照：同一输入独立执行原eval_metrics的128分块q@g.T表达式，调用SHA锁定官方metrics.py::rank；新Ranking仍调用原Stage4A函数。比较分数max_abs≤1e-6、完整排名索引torch.equal、五指标及DR误差≤1e-5个百分点。没有重新实现指标，未改变tie排序。

逐模式释放官方参考完整Score、索引及临时Tensor，保留新引擎最终四组Score供统一publish_result/load_result。读回校验实际shape对应原Manifest及报告计数、CPU FP32、有限性、payload摘要和逐值一致；不重新排名已发布Score。

原数据3074 Gallery/6156 Query全部参与，PID仅供排名身份层使用；没有攻击策略回调、正确配对筛选或camera过滤。完整来源绑定旧协议、Checkpoint、Selection、特征文件、Stage4B1回执、新代码和冻结验收配置。
