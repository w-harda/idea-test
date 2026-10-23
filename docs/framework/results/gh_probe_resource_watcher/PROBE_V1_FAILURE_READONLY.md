# probe_v1已自动执行并失败：只读状态核验

## 结论

**情况B：probe_v1已FAIL，禁止再次启动或覆盖。**

轮询器并非一直未执行：2026-10-23 00:08:32 +08:00启动，52次周期检查后，
02:38:36、02:41:36连续两次达到10240MiB门槛，02:41:45完成CPU预检及最终资源检查后调用原探针。
02:41:55原探针退出码1，轮询器记录PROBE_FAIL并正常按失败策略停止，没有自动重试。

| 项目 | 真实状态 |
| --- | --- |
| 原监督器 | PID/PGID/SID1349747，现已退出 |
| 原探针子进程 | PID1748064，02:41:45启动，02:41:55退出码1，现已退出 |
| 启动前资源 | 空闲24051MiB、利用率0%，RAM/磁盘/inode全部PASS |
| 本次只读GPU快照 | 空闲17787MiB、利用率90%；外部PID3042610占用6256MiB，未干预 |
| Attempt | claim、FAIL、.writer.lock存在；无completed、无pending |
| 当前相同探针进程 | 无 |
| 本轮新执行GPU实验 | 0；没有再次调用native_cli |

## 准确失败位置

原native_cli.py:100完成gate.initialize_cuda，随后在101进入loaded_backend。
native_backend.py:347调用：

~~~python
IRRAAdapter.from_source(source, device="cuda", allow_model_load=True)
~~~

已有irra_source.py:83–84要求明确的cpu或cuda:index，因此报错：

~~~text
tbps.artifacts.ArtifactError: explicit cpu or cuda:index required
~~~

这是此前新后端的设备字符串接线遗漏，CPU测试未覆盖真实加载入口的该约束。
并非显存不足、其他任务干扰、Tokenizer/JVP数值失败或G/H算法错误。
本次仅定位问题，没有修改设备字符串、加载器或任何算法。
最小后续修复方向是显式使用cuda:0，并增加不加载真实模型的组合入口契约测试；
必须另行批准CPU修复和新的Attempt，不能复用probe_v1或改写当前FAIL。

## 真实验证边界

- Claim保存了CPU来源/导入闭包/父恢复结果，G/H完整CPU恢复PASS。
- G Adam全部step484080，H逐参数20124–20370，CPU RNG exact；CUDA RNG原字节保留。
- 运行已初始化CUDA，但加载器在恢复IRRA权重之前拒绝，未进入NativeBackend构造。
- 因冻结调用顺序可确定：G/H/IRRA/reader神经前向、JVP、Optimizer更新均0；
  IRRA与reader真实权重恢复0，G模型尚未移到GPU。
- 上述计数依据异常栈及源码控制流，不冒称已有完整Hook计数回执。
- 新Unicode文本、新语境JVP/FD、NOOP及非NOOP梯度差均NOT_RUN。
- CUDA设备RNG恢复及GPU完整Adam迁移未验证。
- 原失败入口未保存GPU allocated/reserved/RSS/elapsed终态报告，不能填0或使用轮询显存冒充峰值。
- 控制日志从启动到失败约10秒，只是调用墙钟区间，不是有效数值探针耗时。
- 不满足后续A1/B完整Smoke的真实数值前提；不得标记GH_NATIVE_NUMERIC_PROBE_PASS。

## 来源及保护

执行源码基准a46c6980c4abf58df02ca1ee199da799892dc81e。
冻结工作树07a31edb4d1a4fc8e28ce8968a037b03b5681075，本次仍干净。
资源轮询器代码3b2ef7391e792a9c8599f350c2b2be5752aed763，部署记录2760c809。
新增状态报告不会改变原数值实验来源。

139项producer、496项运行来源SHA全部一致。
G60 SHA9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049。
H10 SHA4ae964d51ea94840de71d9597d90a2234a36104e5d649cfee96577f59f94bacd。
IRRA SHA9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
151378项历史metadata无变化；该检查不是全部历史文件重新SHA扫描。

## 原始路径

控制目录：/home/lzf/ldx/tmp/gh_native_probe_wait_v1/。
Attempt：/home/lzf/ldx/outputs/tbps-framework-gh-bidirectional-backend/probe_v1/。
重点文件：poll.log、cpu_preflight.log、probe.started.json、probe.process.json、probe.exit.json、probe.log、status.json、failure.json及Attempt/FAIL。

[完整SHA及证据索引](PROBE_V1_FAILURE_EVIDENCE.json)包含28份原控制/Attempt文件的SHA、大小与mtime。
所有原FAIL、启动标记、Claim及日志保持原样；只新增本报告及派生证据。
没有重新运行CPU审计、GPU探针、A1/B、训练、Validation、Test或Full。
