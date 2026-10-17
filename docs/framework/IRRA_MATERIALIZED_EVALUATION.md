# 已封存generation_v3的官方IRRA四组评估准备

本轮基于 `0f5b8838b9ef3d0032f656e5169c74054ddd78cf`，独立分支 `codex/tbps-irra-materialized-evaluation`。只开发和运行CPU测试/只读来源预检；未加载真实模型、初始化CUDA、生成扰动或计算真实Full排名。实现Commit：bc710e5baf807bb8347a51c6071d03b43b2c0245；本后续文档提交登记。

## 已确认的Stage4D输入

- 同一Plan：`7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5`。
- generation_v3/result Manifest SHA：`fb696a1c0c3235018202118a2d945fa728f6fc06be39738f04a26e6a39fe7634`，JOINT_FULL_MATERIALIZED。
- 独立CPU审计Manifest SHA：`5a693b3877a35c6778e79de7c3f9e555f7772191bbd2e3af5624fcef268bdce5`，JOINT_FULL_CPU_AUDIT_PASS。
- Image3074、Text6156，G历史实际前向3074/恢复1；最大L∞=0.0313725471496582。6116编辑、40 NOOP、13537替换。
- 物化Store仍为 `/home/lzf/ldx/outputs/tbps-framework-stage4d1/materialized_full_v1/artifacts/`；本入口只读，不复制、重签、修改或生成其单位。

## 复用与最小新增

| 职责 | 复用代码/来源 | 新增接线 |
|---|---|---|
| 官方身份/顺序/PID/Caption/配对 | datasets/cuhk_pedes.py；原Joint Full协议 | 重新比对完整官方Manifest，Gallery/Query分别迭代 |
| 原Plan与物化读取 | 冻结Stage4D1 full_prepare、StreamingMaterializedStore、BulkSession、MaterializedStore.load | 只读解析joint-full指针及完整集合 |
| Unicode动作 | native_action_validator、ValidatedActions | 编码前CPU核验6156条动作，退出隔离后不调用H |
| IRRA权重/Tokenizer/预处理 | models/irra_source.py、irra_io.py、IRRAAdapter.inference() | FP32保存RGB直接送入原normalize_pixels，不重新Resize/G生成 |
| 官方原生评分 | IRRA原生L2(text) @ L2(image).T、已核验irra_cached.normalized | 新的无模型缓存评分薄层CachedIRRA，独立ModelSpace |
| Ranking/五指标/DR/结果发布 | EvaluationEngine、engines/metrics.py、publish_result/load_result | 仅调用既有实现，不新增Ranking或指标算法 |
| 紧凑特征缓存 | ArtifactStore原子发布/SHA/完成回执 | FeatureCache按64行保存，保存/读取真正clone |
| 历史Clean一致性 | 原Phase5 Joint Full summary/protocol/completed；原G-v2 Clean门槛0.001pp | 只读参考五指标，不读取旧大Storage、不填入历史结果 |

新配置SHA为 `79d33968d343134b8f48329f81fb8282440dfb6e59ff9b2031b84bb13909e56c`。新增仅 `src/tbps/engines/materialized_irra.py`、`materialized_cache.py`、独立配置与CPU测试；没有修改Stage4D、G/H、IRRA、Engine或旧指标源码。生产运行继续使用冻结Stage4D1的PYTHONPATH，保持同一Plan及原生产者路径。

## 四组与执行协议

| Group | 图像输入 | 文本输入 |
|---|---|---|
| Clean | 保存的original FP32 RGB | 原始Caption |
| Image-only | 保存的adversarial FP32 RGB | 原始Caption |
| Text-only | 保存的original FP32 RGB | 历史H Unicode文本 |
| Joint | 保存的adversarial FP32 RGB | 同一历史H Unicode文本 |

官方Test3074 Gallery/6156 Query，全部同PID正例，无camera过滤，保留全部NOOP。图文独立有序，不能zip。IRRA官方权重SHA为 `9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`，原384×128/Byte-BPE77规则、FP32/math/no_tf32，参数eval/frozen。

四family只编码一次，四组共用；首次Full应6148个IRRA图像样本前向、12312个文本样本前向、IRRA恢复1次，G/H恢复/前向及Backward/Optimizer/JVP=0。缓存块64，模型batch1，Query评分分块128、每块面对完整Gallery；官方完整排名及FP32全Query均值由原Engine完成。指标单位百分数，五项DR均为(Clean-Group)/Clean×100，Clean为0返回null。

CPU预检校验原封存/审计、全部来源SHA、官方完整身份及Unicode，并读回首尾两张RGB；所有RGB在真实编码前继续逐张调用原load核验字节、形状、范围、effective_delta及预算。Cache绑定实际输入SHA、ModelSpace、Tokenizer/预处理、Plan及原集合；不能把IRRA特征标为RaSa/APTM。

Full结果发布前必须通过历史Clean五指标0.001个百分点门槛。其他组数值来自当前实际编码与评分，不强制等于历史攻击指标，不使用Test调参或选权重。

## CPU验证与明确边界

- 新增25项最终CPU测试通过；真实Runner控制流程由CPU Mock执行四组编码、实际缓存/Score读回、NOOP exact、计数、只读审计和已提交结果复用。
- 独立加载官方utils/metrics.py::rank，对10×10合成四组Score进行五指标逐项exact对照；未计算真实Full排名。
- 新工作树704项回归通过/123.14秒（当时含19项新测试；2项旧Runner绝对生产路径配置测试转到原生产树执行，未改签旧配置）；这2项在原树通过。最后25项新测试通过（含编码会话/Tokenizer来源绑定补强），28.71秒。
- 实际CPU Full预检退出0、MATERIALIZED_IRRA_CPU_PREFLIGHT_PASS：3074/6156顺序、PID/Caption/配对、6116编辑/40 NOOP/13537替换、两个真实RGB通过；CUDA初始化及G/H/IRRA加载0。
- 证据仅在 `/home/lzf/ldx/outputs/tbps-framework-irra-eval-development-cpu/`，不提交日志、数据、缓存或模型。
- 这证明CPU接线与来源就绪；本新入口尚未执行任何真实IRRA GPU编码或正式四组Full。不能将历史指标或CPU Mock指标当作本次正式结果。

首次合成CPU Fixture把同一文件重复当作config与生成证据，原ArtifactStore正确拒绝；已修正Fixture为独立来源后通过，原失败日志保留。没有放宽生产来源规则。

## 历史保护与计算量

generation_v3及独立审计26文件SHA/size/mtime保持；物化Store61555文件成员、size/mtime保持，原冻结Stage4D1和generation_v3生产工作树HEAD/clean不变。本轮不重复整个Store保护性内容SHA扫描；实际CPU预检另按原集合声明验证所有文件SHA。保护报告SHA为 `263cc155cddc0ea031e0bfcbc147065165ee05edcc002cfb2a9100e9910c0741`。

真实CPU预检日志SHA `d179bb4bb1ccfbe6dec8aee5d5db0321ea3679ef8db9eb653c0e60c084beaf43`；最终25项测试日志SHA `854ce88ec26f4985d88f69fc433a26b1eb67bf8f069ebb16b6cd284039f0cbd9`。

首次Full只做6148图像样本＋12312文本样本IRRA编码；新特征纯数据约36.05MiB，四组FP32 Score纯数据约288.75MiB，另有协议/回执与中间内存。不复制原约5GiB物化图像。GPU显存、加载RSS和正式耗时本入口尚未实测，不用CPU Mock耗时推断正式GPU性能。

## 用户手动命令

环境：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017/src
```

CPU预检（无GPU查询/模型加载/排名/正式输出写入）：

```bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
  /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/src/tbps/engines/materialized_irra.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/configs/framework/materialized_irra.server.json
```

正式Full（仅用户运行，Codex未执行）：

```bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
  /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/src/tbps/engines/materialized_irra.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/configs/framework/materialized_irra.server.json \
  --action run --scope full --allow-full-evaluation
```

建议先单独授权一次10+10新入口Smoke；不是本轮执行内容：

```bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
  /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/src/tbps/engines/materialized_irra.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/configs/framework/materialized_irra.server.json \
  --action run --scope smoke --allow-smoke
```

Smoke固定前10 Gallery及其按官方顺序的前10合法Query，不按攻击效果挑选；20图像+20文本IRRA样本前向、G/H=0。小样本指标仅验证接口。NOOP路径在本轮CPU Mock及原Stage4C3固定真实样本中已验证；当前前10子集不保证包含真实NOOP，正式Full保留全部40条。

GPU入口沿用原IRRA资源门禁：GPU0无其他计算进程、空闲≥8192MiB、启动利用率≤10%。本轮未查询GPU资源，未把G生成阶段对Ollama的专属共享许可自动扩展到IRRA。资源不足在创建评估Attempt之前拒绝，不等待或自动重试。

## 输出、停止、恢复与审计

正式输出独立于Stage4D：`/home/lzf/ldx/outputs/tbps-framework-irra-materialized-evaluation/full/`；Smoke在同根smoke目录。GPU进度为IRRA_FEATURE_PROGRESS，成功标志MATERIALIZED_IRRA_EVALUATION_PASS full。

- `completed/summary.json`：四组完整精度五指标、全部DR、Joint与Image-only的mAP差值。
- `completed/metrics.csv`：四组五指标和五项DR。
- `result/`：原Engine协议、四组Score、指标和来源。
- `execution-000001/report.json`：实际前向、冻结、显存峰值、RSS及运行耗时；后续恢复独立递增。
- 分块缓存已提交且SHA/协议通过才可复用；不覆盖旧块。普通失败有闭合execution记录时，原Full命令追加 `--resume`；重新恢复IRRA一次，只编码未提交部分。未提交部分可能重新编码，实际调用记录按Attempt保留，不能说总前向始终等于首次理论值。
- pending或未闭合Attempt（如SIGKILL、默认SIGTERM）拒绝自动恢复，需先人工审查；不删除旧缓存或伪造回执。Ctrl-C正常Python异常会尽力保存FAIL。已完成重入仅核验，不重载模型或排名。
- 缓存不得与历史Feature Replay或其他Victim混用。没有G/H生成入口，不会修改原Stage4D数据。

完成后的CPU只读审计（不重算排名、不编码）：

```bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u \
  /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/src/tbps/engines/materialized_irra.py \
  --config /home/lzf/ldx/projects/idea-test-tbps-irra-materialized-eval-20261017/configs/framework/materialized_irra.server.json \
  --action audit --scope full
```

成功为MATERIALIZED_IRRA_CPU_AUDIT_PASS full；重新核对Stage4D输入来源、全部特征/Score SHA、summary/CSV/结果内容和最终冻结执行证据。此时不调用官方rank或新Engine.run。

尚需用户实际执行小规模/Full编码及完成后审计。本轮不接入RaSa/APTM；其后只能消费同一物化RGB/Unicode，独立形成各自原生ModelSpace与评分缓存。
