# C-PGC / SEP v3吞吐量优化与正式恢复报告

截至2026-10-31T12:12:52.373621+08:00：已完成并审计5/12，第六组C-PGC × ICFG × RaSa已在新v3 Attempt真正运行，已提交2715/19848张图像物化产物。
此时处于CPU FP32 RGB物化阶段，RaSa已加载GPU；后续自动执行原生编码、完整Top128评分和独立CPU审计。
不将初始图像物化计作已完成的GPU排名；不提前宣布第六组或12组Full PASS。

## 验收结论

| 检查 | 实际结论 |
| --- | --- |
| 前五组Full及各CPU审计 | PASS，28个既有成功步骤的Manifest、payload SHA、CSV、summary及审计绑定核验 |
| 历史保护 | 535776个v2文件size/mtime/inode完全不变；原控制及Partial保留 |
| 新配置GPU Smoke | 六组全部PASS，128 Gallery / 2 Query，覆盖完整Top128融合 |
| 新独立CPU审计 | 六份全部PASS |
| 新旧Smoke数值对照 | 六组像素/特征payload exact；三模式评分逐元素exact；顺序、排名及四指标exact |
| 完整ICFG候选对照 | 固定前128 Query × 19848 Gallery，采用路径评分矩阵与旧v2逐值exact |
| CPU测试 | 58个不同用例PASS；六个真实Source复用CPU契约PASS；两个真实Python环境Preflight PASS |
| 正式Full | 第六组RUNNING，未完成；SEP后续自动串行执行 |

前五组真实原精度数据见VERIFIED_V2_METRICS.csv，来源见VERIFIED_V2_REUSE.json。
没有重复运行这五组Full或原CPU审计，也没有使用Smoke指标填充正式表格。

## 吞吐量与数值取舍

真实四条ICFG Query的Token读取，512个候选，旧实现两遍41.0875 / 20.1025秒，新实现8.3807 / 0.3040秒。
第二遍局部缓存读取加速约66.12倍，全部Tensor逐值exact。不是整条Full端到端66倍。
使用64GiB有界共享CPU Token缓存、stat失效SHA缓存，减少无效重复读盘、重复哈希与克隆，不人为填充GPU。

图像Batch8/16/32和文本Batch16/32/64均不能逐值复现原特征，全部不采用。
在完整ICFG Gallery对照中，融合32/64/128最大分数差7.5102e-6 / 6.6757e-6 / 7.9870e-6，虽未改变该128 Query子集四指标，却改变了排名，全部拒绝。
最终保留image_batch=1、text_batch=8、fusion_batch=16、query_chunk=32、完整Top128和既有精度。
Query chunk32/64/128在实测完整Gallery全局候选上完全一致；其瞬时计时包含首调用暖机，且非ITM瓶颈，未据此改动正式值。

PERFORMANCE.csv保留各Batch吞吐与显存；FULL_GALLERY_PERFORMANCE.csv区分冷读I/O和计算/主机开销。
不同轮次有页面缓存和暖机影响，不能直接把混合冷读总耗时当纯融合加速。
PyTorch CPU/CUDA算子及传输耗时见PROFILE_RASA16.txt。GPU利用率原始1秒采样见EXPLORATORY_BENCHMARK.json，
部分Batch不足1秒，不能精确分离其单独利用率，不虚构每Batch平均利用率。
正式后台每15秒记录GPU_monitor.jsonl。Full结束后才有真实端到端耗时及Full峰值；当前不作已验证宣称。

新六组Smoke最大allocated 1921.84MiB、reserved 2114.00MiB。
资源门槛仍8192MiB显存，额外要求96GiB可用RAM给64GiB缓存及系统余量；磁盘100GiB、inode80000。
无新的精度降低；IRRA既有原生混合权重、RaSa FP32路径不变。

## 后台与执行身份

监督器909831：PPID=1、PGID=909831、SID=909831、无TTY。实际Full子进程950195。
nohup + setsid独立会话、stdin=/dev/null、输出写本地文件，SSH/本机退出不影响运行。
持有原全局恢复锁和新process.lock。启动后实际已创建新Full started、protocol及物化产物，不只依据PID。
源码Commit：4c8eab66860d40b74a6f55116b510f53dc9d38de。
配置SHA：a52e31c7ed3e455e14d1b5c17a4262e3c212f4d784681d39685a19615f45f903。
原v2执行树保持36cf91f97ea3c483883eb6d93456c51ef7d662f0不变。

原监督器2733135及旧第六组2895196经身份核验后定向SIGTERM，未发送PGID广播。
原v2日志和Partial保持，不把原RUNNING_STEP历史快照改成成功；停止证据见SOURCE_EVIDENCE.json。
C-PGC官方UAP文件、数值、预算、文本攻击及六个Victim权重不变。
Source通过明确父配置SHA/Manifest SHA及语义字段核验只读复用，不重新生成UAP。

## 后续无人值守流程

第六组Full → CPU审计 → 六组SEP Source/Target Smoke与审计 → 六组SEP Full及各CPU审计 → 明确v2/v3身份汇总12组并发布。
SEP Source保留原图像逐张/文本逐条/融合16、DPP、代理与候选选择，缓存不引入浮点分数变化。
无可用的旧SEP真实GPU产物可作两版本逐样本对比；官方核心SHA与CPU原算法组合测试有效，真实Source Smoke仍为后续硬门禁。

资源不足只在尚未创建Attempt时每5分钟有界等待，累计12小时。OOM、算法/SHA/审计错误首错停止，保留全部现场，不自动重试。
最终发布使用另一工作树，不修改正在运行的冻结源码。
任何最终汇总都保留前五组v2、后七组v3来源，不假称同一个配置身份。

## 查看进度

~~~bash
cat /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031/status.json
tail -n 30 /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031/status.log
tail -n 30 /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031/0013_cpgc_full_icfg_rasa_target.log
find /home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v3/full/cpgc/icfg/rasa -mindepth 2 -maxdepth 2 -path '*/pixels-*/manifest.json' | wc -l
nvidia-smi
~~~

当下只能宣称5/12正式完成、第六组已真实启动。独立审计和12组最终成功标志仍需后台实际产生。
