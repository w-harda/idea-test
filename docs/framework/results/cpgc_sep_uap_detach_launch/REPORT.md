# C-PGC UAP修复后v2真实Full启动记录

## 当前结论

修复执行Commit 36cf91f97ea3c483883eb6d93456c51ef7d662f0。107项CPU回归和两个真实环境预检PASS。C-PGC三数据集×IRRA/RaSa全部六组真实GPU Smoke及六份独立CPU审计PASS；原UAP Tensor requires_grad错误已消除，Full来源及实测峰值余量门禁通过。

正式Full已进入C-PGC CUHK Source文本生成，随后自动串行执行IRRA→CPU审计→RaSa→CPU审计，然后RSTP/ICFG；SEP先完成其六组Smoke/CPU审计才允许六组Full。SEP尚未真实验证，不能称其GPU PASS。Full完成数与文本进度见本报告快照SOURCE_EVIDENCE.json，尚未结束的检索指标不填写。

## 唯一后台进程

监督器PID/PGID/SID=4125880，已从启动SSH进程重归PPID=1，TTY=?，持续存活。nohup+setsid启动，输入/dev/null、输出独立日志，不依赖Codex、Windows电脑或SSH终端在线。不部署额外Agent或Watcher，不自动修复、重试；任何真实FAIL保留现场并停止。

执行树：/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-uap-detach-20261029
输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/
控制：/home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/

## 原始资源与历史保护

生产修复仅Target/Audit加载后detach，未修改provenance拒绝契约。作者UAP文件SHA与数值、shape、dtype不变，未训练或重生UAP；旧v1的45份FAIL、Source PASS及控制文件SHA/stat保持。没有干预Ollama，启动时空闲9067MiB。资源门槛仍为8192MiB，RaSa融合batch16但全部Top128、Query Chunk32、原精度与评分不变。

GPU Smoke的峰值reserved（MiB）：

| Dataset | IRRA | RaSa |
|---|---:|---:|
| cuhk | 458.0 | 2114.0 |
| rstp | 446.0 | 2114.0 |
| icfg | 446.0 | 2114.0 |

以上是Smoke资源证据，不是Full检索指标。全部实测峰值+2048MiB余量均满足8GiB门槛；每阶段仍即时检查，资源不足/OOM不会降低参数或抢占外部进程。

## 查看进度

```bash
cat /home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/phase
tail -n 50 /home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/status.log
tail -n 30 /home/lzf/ldx/tmp/cpgc_sep_full_v2_20261029/cpgc_full_cuhk_SOURCE.log
nvidia-smi
```

完成12组真实Full与审计后，监督器自动生成正式RESULTS.csv、RESULTS_2DP.csv、CLEAN_BASELINES.csv及报告，并在独立codex/cpgc-sep-full-v2-results分支发布；不为文档发布修改正在运行的执行树。Git发布若失败只保留publish.log，不重跑Full。
