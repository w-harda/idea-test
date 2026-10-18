# SGA官方IRRA Source Full：v2启动器最小修复

基准：4363e3afebab4eadfeaa9652bfd9d5e50bfcb891。独立工作树/home/lzf/ldx/projects/idea-test-tbps-sga-full-launch-fix-20261019，分支codex/tbps-sga-full-launch-fix。

## v1失败及保护

2026-10-19 00:43:34 +08:00，v1在CPU Preflight PASS后，因Full目录尚不存在，df "$OUT/.."无法解析而退出1。Source尚未启动，正式Full输出不存在；与模型、算法和GPU资源无关。

/home/lzf/ldx/tmp/sga_official_irra_full_v1/全部13个文件保留，包括started、FAIL、日志、PID/PGID；SHA/mtime/inode保护清单留在/home/lzf/ldx/tmp/sga_official_full_v1_protection_before_fix.json。不改签、不删除、不覆盖。

## 最小修复与验证

仅改run_sga_irra_full.sh：
- 三处磁盘记录、inode记录和可用容量检查，统一使用"$(dirname "$OUT")"。
- ROOT指向新独立修复工作树。
- CONTROL指向/home/lzf/ldx/tmp/sga_official_irra_full_v2/。
- OUT及冻结Full配置保持原路径；参数/模型/攻击/评分均不改。

bash -n与git diff --check PASS。8项CPU测试PASS，新增真实df回归覆盖Full不存在时的三项检查，验证不创建Full目录；验证v1/v2隔离。所有其他tracked src/configs文件与基准逐字节exact，没有重复GPU Smoke或Batch8实验。

## v2执行

用户明确授权验证后直接启动Full。修复代码Commit：f8d3431b4f51d4fb117cf031f6d827bb89fab69c。一次即时资源检查通过后，2026-10-19 00:51:36 +08:00以nohup setsid启动；00:51:42 CPU Preflight PASS并进入RUNNING_SOURCE；00:51:54核验监督器存活。资源门限仍为20GiB空闲显存、48GiB可用RAM、35GiB磁盘；不轮询、不结束其他进程、不自动重试。

启动核验：PID=3001120、PGID=3001120、SID=3001120、PPID=1、TTY=?，已脱离SSH。Source日志已出现SGA_IRRA_CPU_PREFLIGHT_PASS full 3074 6156及IRRA原生初始化记录。没有再次启动、重复Smoke或容量测试。

即时资源：GPU空闲24051MiB、利用率0%、无计算进程；RAM119788152KiB（约114.2GiB）、磁盘6414560399360bytes（约5.83TiB）。资源快照/home/lzf/ldx/tmp/sga_official_full_v2_start_resources.json；正式启动证明/home/lzf/ldx/tmp/sga_official_irra_full_v2/startup_verified.json。v1全部13文件SHA/mtime/inode再次核验不变，原4363e3a工作树干净。首次启动尚未产生Source输出，因此本次没有重执行任何已完成Source批次。

这是启动时快照，不表示Full、Target或独立CPU审计已完成；后台独立运行，Codex结束后不持续轮询。

正式输出：/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/。
控制日志：/home/lzf/ldx/tmp/sga_official_irra_full_v2/。

```bash
cat /home/lzf/ldx/tmp/sga_official_irra_full_v2/phase
tail -n 30 /home/lzf/ldx/tmp/sga_official_irra_full_v2/status.log
tail -n 30 /home/lzf/ldx/tmp/sga_official_irra_full_v2/launcher.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/source.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/irra.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/rasa.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/aptm.log
tail -n 20 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/audit.log
```

同一IRRA源生成3074图/6156文，再三Target原生评估、独立CPU审计；原官方SGA三阶段、batch8/seed42/2/255/10轮/0.5/255/五尺度不变，保留FP32投影舍入语义。Matched Clean不直接拼接历史Joint Raw Clean，旧白盒结果不冒充官方SGA。

预计6–8小时。启动成功不等于实验完成；仅当所有阶段及独立CPU审计通过并出现SUCCESS | SGA_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT，才标记正式完成。
