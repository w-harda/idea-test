# 执行协议

工作树：/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-20261028
配置：configs/framework/cpgc_sep_tbps.server.json
输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v1
控制：/home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028

显式授权入口：python -B -m tbps.baselines.cpgc_sep.cli --config <绝对配置路径> --action <cpgc-source|sep-source|target|audit> --scope <smoke|full>。GPU Smoke需要--allow-gpu-smoke，Full需要--allow-full及对应方法六份同来源Smoke审计。CPU审计为独立进程，不加载模型。

一次性后台命令：

```bash
mkdir -p /home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-20261028/operations/run_cpgc_sep_full.sh >> /home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028/launcher.log 2>&1 < /dev/null &
```

查看阶段：cat /home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028/phase
查看日志：tail -n 30 /home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028/status.log
监督器PID/PGID/SID写入同目录，GPU实际进度在每阶段独立log。资源不足只停止当前启动，不部署未经请求的等待器。任何真实错误首错停止，禁止自动重试。停止后要先审查Claim和pending；不能直接重跑整个命令。

源代码producer SHA、官方Commit和资源SHA逐阶段核验；后台启动后不修改冻结执行代码/配置。报告更新放派生目录，不能改变执行身份。精确全量评分chunk128Query，RaSa每Query128 ITM；Score矩阵完整保存。预计ICFG局部Token磁盘量远大于CUHK，不把Smoke速度直接当Full保证。
