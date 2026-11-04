# SEP-TBPS Source-only Smoke

执行 Commit `167c91d3aacadb78781809c35ac9a274dae12f4f`。
协议 `SEP-TBPS-SRefPseudoPID-S1`。详见[固定执行协议](EXECUTION_PROTOCOL.md)。

CPU Preflight PASS，189 项 CPU 测试。
CUHK 官方 Train 34,054 图、11,003 PID。参考清单 SHA `7c21e500200753a722cda0f7c0246b87506b2b582fee8b5433b13698c8b6fde4`。
Landlock ABI 6，11 个真实拒绝探针。

服务器控制目录 `/home/lzf/ldx/tmp/sep_source_only_smoke_20261104`。
输出目录 `/home/lzf/ldx/outputs/tbps-framework-sep-source-only-smoke-20261104`。

## 实际运行阶段

| 阶段 | 状态 | 秒 |
| --- | --- | ---: |
| parity-gpu | PASS | 20.23 |
| parity-cpu | PASS | 1.24 |
| source | FAIL | 11.52 |

## A 类优化 Exact

同一份真实 FP32 原生矩阵通过两套生产 CPU classifier。PID 顺序和成员、聚合、softmax、原 margin、DPP、候选排序、缓存键与 RNG 一致。
候选排序 `[3, 2, 0, 1]`。冻结矩阵 SHA 如下。

- irra `6c6ee5856ef18c97f887c3ae16ad37fe4041d87ac6a4241d2deb59546627893d`，形状 [4, 128]。
- rasa `0f480b606e095565bcd10f7306a79fde0966cc10a3288fea944dda2d10e1947f`，形状 [4, 128]。

## 本次执行停止

失败阶段 `source`。错误 `CalledProcessError(1, ['/home/lzf/ldx/envs/tbps-image/bin/python', '-B', '-m', 'operations.sep_source_only.phases', 'source', '/home/lzf/ldx/outputs/tbps-framework-sep-source-only-smoke-20261104', '/home/lzf/ldx/tmp/sep_source_only_smoke_20261104', '1800'])`。
未完成阶段保持未运行。详细 traceback、阶段日志和已冻结产物在服务器目录。

## Full 可行性

真实 Smoke 仅用于验证链路与估算已测样本的开销。逐候选全 Train 检索、官方零方差循环和共享 GPU 资源均影响 Full 可行性。Full 尚未验证或授权。
