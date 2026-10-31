# C-PGC / SEP v3吞吐量优化与隔离恢复协议

用户授权安全终止v2第六组Partial、进行性能/数值验证并启动v3；不得重算前五组成功Full。

## 历史与停止
原执行36cf91f97ea3c483883eb6d93456c51ef7d662f0，配置4b56eb60a3f2c928ef084fad64f1648bee17129e450b707a4d166169bc8a370a。
仅核验并SIGTERM原监督器2733135及RaSa ICFG子进程2895196，不向整个PGID广播。两者已退出。
旧控制状态是停止前快照，不篡改为成功。原日志、Partial、Claim、Manifest全部保留。

## 优化边界
图像Batch支持真实拼接，但8/16/32不能逐值复现旧特征，正式保留1；文本16/32/64亦不exact，保留8。
保留完整Top128、原两级候选排序、原精度、全部Gallery排名和精确mAP。
融合Batch只能通过完整ICFG Gallery、真实Top128、逐元素评分及排名/指标对照后选择。
Query Chunk32/64/128测量候选序列，正式保留32，避免改变外层矩阵乘形状。
RaSa各Bank共用64GiB有界CPU Token块缓存，不人为占用GPU。
VerifiedArtifactStore完整保留receipt/manifest/payload/source/provenance/path校验；仅SHA计算按dev/inode/size/mtime_ns/ctime_ns缓存，变更强制重新校验。
SEP Source保留原图像逐张、文本逐条、融合Batch16和官方DPP/候选流程，仅改善读取缓存，不传播Target浮点分数变化。

## 来源及新Attempt
新树/home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-throughput-v3-20261031。
配置configs/framework/cpgc_sep_tbps.throughput_v3.server.json。
新输出/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v3。
新控制/home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031。
C-PGC Source显式绑定旧配置SHA、完成Manifest SHA及算法/数据/权重语义字段。
只读引用v2 Source，不改旧Manifest，不冒用旧Target Smoke PASS。
新配置先执行六个C-PGC Target Smoke及各独立CPU审计，再重做第六组完整ICFG Test。
SEP完成六组新Source/Target Smoke及审计后串行执行六组Full。
前五组只读引用v2成功Full及审计；最终汇总明确v2/v3身份。

## 资源与保护
GPU8192MiB、RAM96GiB、磁盘100GiB、inode80000；利用率仅记录。
启动前不足时每300秒等待，累计最长12h，不初始化CUDA。
持有原全局cpgc_sep_v2_resume.lock、新控制process.lock，started一次性。
GPU/OOM/算法/SHA/审计/数值错误首错停止，保留全部现场，不重试。
仅原生runtime.resources在尚无Attempt时的RESOURCE_BUSY可等待。
运行时源码树不修改；最终发布使用另一工作树。

## 启动与查看
~~~bash
mkdir -p /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-cpgc-sep-throughput-v3-20261031/operations/run_cpgc_sep_throughput_v3.sh >> /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031/launcher.log 2>&1 < /dev/null &
cat /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031/status.json
tail -n 30 /home/lzf/ldx/tmp/cpgc_sep_throughput_v3_20261031/status.log
~~~
必须核验PPID=1、无TTY、PID/PGID/SID及真实进度。最终12组Full及各CPU审计PASS才算全量完成。

补充：旧535776项产物的只读深度重哈希与新配置Smoke可并行进行；新Full必须等待五组完整payload/Manifest/SHA及审计证明全部PASS。若只读扫描异常退出或两小时内不完成，停止Full放行，不跳过校验。
