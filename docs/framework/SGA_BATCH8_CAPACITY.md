# SGA IRRA Source：官方Batch=8容量验证与Full启动准备

基准Commit：9e15291a6d6601a72cdb3deff247b0e2066d26a6。独立分支codex/tbps-sga-batch8-capacity；工作树/home/lzf/ldx/projects/idea-test-tbps-sga-batch8-capacity-20261019。

**结论：一次真实Batch=8 Source-only测试PASS，无OOM；本轮未启动正式Full。** 代码及实际验证Commit：28677244c98057c7ae93460b7364af0c0c8341f0。实际报告及证据SHA见[JSON](SGA_BATCH8_CAPACITY_EVIDENCE.json)。

## 1. 固定输入与复用方式

固定CUHK官方Test顺序前8张Gallery：cuhk:37132:0至cuhk:37139:0；对应全部16条Caption（各caption0/1）。GPU前锁定成员及顺序，不按攻击效果选样。

新增sga_batch8_capacity仅将冻结source_run的Dataset读取限定为这8图/16句，并改为独立输出；实际调用原SGAttacker.attack。原Full runner、官方attacker.py、模型Wrapper、所有攻击算法文件及其producer SHA不改。仍是原native IRRA/PIL RGB/bilinear384×128、官方pre_caption30、BERT MLM/Tokenizer→IRRA Byte-BPE桥。

IRRA源Checkpoint、MLM、FP32、batch8、eps2/255、steps10、step0.5/255、五尺度、全局seed42和完整文本→图像→文本均与Full相同。沿用Full自带的一次额外输入梯度检查，未替换攻击Loss，不运行任何Target检索评分。

## 2. 实测结果

| 项目 | 真实记录 |
|---|---|
| GPU | RTX 4090，启动空闲24051MiB，无其他计算进程 |
| 图像/Caption | 8 / 16，单个完整batch |
| GPU peak allocated | 6,841,224,192 bytes = 6.3714 GiB |
| GPU peak reserved | 11,410,604,032 bytes = 10.6270 GiB |
| CPU进程Peak RSS | 2,448,379,904 bytes = 2.2802 GiB |
| 攻击及预算核验 | 9.022531121969223秒 |
| Source会话（含加载/检查/封存） | 19.157618556171656秒 |
| 系统可用RAM（启动前） | 119801620KiB，约114.25GiB |
| OOM / 自动重试 | 无 / 0 |
| IRRA源加载 / MLM加载 | 1 / 1 |
| 实际IRRA图像样本前向 | 417：额外检查1＋clean8＋10轮×5尺度×8＋adv8 |
| 实际IRRA文本样本前向 | 1026，包含全部候选、遮词、T1等，不是1026条独立Query |
| MLM调用 / 样本 | 4 / 80，含多WordPiece候选内部MLM调用 |
| G/H及目标模型加载/评分 | 全0 |
| Optimizer | 0 |

RSS是本任务进程峰值，不冒充整台服务器所有任务的总RAM峰值。GPU数值为本次进程Torch allocator统计，不包含其他进程或全部驱动占用。

IRRA/MLM各自参数与buffers前后摘要exact，eval、requires_grad=False、parameter_grads_none=True。官方三阶段flow为[text1,image,text2]；T1确实被中间图像攻击使用，两次文本攻击均从原Matched Caption开始。

1026行真实BERT→IRRA桥均WordPiece往返exact、IRRA新增截断0。完整像素FP32 [8,3,384,128]，范围[0,1]，effective_delta读回bitwise exact，来源及文件SHA通过CPU读回检查。

官方FP32投影上下界exact PASS。实际最大实数L∞=0.007843166589736938；数学2/255=0.00784313725490196，差2.9334834977801094e-08。按用户批准公开该舍入，不声称严格数学上界PASS，未修改投影。

实际成功标志：
- SGA_IRRA_CPU_PREFLIGHT_PASS batch8_v1 8 16
- SGA_OFFICIAL_IRRA_SOURCE_PASS batch8_v1
- SGA_BATCH8_CAPACITY_PASS
- BATCH8_SOURCE_CPU_READBACK_PASS

证据：
- /home/lzf/ldx/tmp/sga_batch8_capacity_v1.log
- /home/lzf/ldx/outputs/tbps-framework-sga-irra-capacity/batch8_v1/source/execution/report.json
- /home/lzf/ldx/outputs/tbps-framework-sga-irra-capacity/batch8_v1/source/completed/
- /home/lzf/ldx/outputs/tbps-framework-sga-irra-capacity/batch8_v1/source/batch-000000/metadata.json
- /home/lzf/ldx/tmp/sga_batch8_cpu_readback.log

该Attempt已消费，不可再次执行容量命令或覆盖输出。

## 3. 必要的Full执行管理修正

旧Full CLI可由nohup/setsid独立运行，但未单独持久化PID/PGID/当前phase，IRRA编码也缺少中途进度。新增两项薄管理组件：

- run_sga_irra_full.sh：一次性flock/started，记录监督器PID/PGID、当前子进程PID及phase；顺序调用已有Source、三个Target、最后CPU audit。退出码和精确PASS标志双重检查；任何失败停止、不重试、不覆盖。
- sga_full_execution.py：CPU启动前核验真实Batch8 PASS、完整来源和原Full preflight；Target端只包装已有encode函数记录标量完成数，原函数只调用一次，返回值不变，不改参数、原生融合评分或Ranking。

Full仍在IRRA Source上生成一次3074图/6156文，IRRA/RaSa/APTM分别只读取同一套冻结产物。没有采用旧SGA-TBPS White-box、重新训练、目标反馈或追加攻击优化。

Full使用原20GiB最低空闲显存门禁，未因本次峰值较低而降低；共享策略只观察外部进程，不结束Ollama、不轮询等待。另记录RAM/磁盘/inode快照，要求可用RAM≥48GiB、磁盘≥35GiB。运行中真实OOM按原错误路径保留并停止。

新脚本/进度入口/冻结Full配置的SHA写到控制目录launch_sources.sha256；既有Full源配置和历史实验不变。

## 4. CPU测试与保护

- 8项新增容量入口测试＋18项既有官方算法/文本桥CPU测试：26 PASS。
- 6项Full执行管理CPU测试：6 PASS，包括真实Bash控制流的Fixture成功链、RaSa阶段失败即停、禁止重入、默认不执行；进度包装逐值不变/退出恢复、启动预检禁止模型加载/CUDA。
- 共32项CPU测试PASS；bash -n及git diff --check通过。
- 164份旧Smoke文件逐字节SHA/size/mtime/inode不变。
- 官方及历史145来源SHA/stat、67380旧结果/FAIL stat、4旧工作树保护结果记录到证据JSON；冻结原Source工作树HEAD和状态保持。
- 正式Full输出及新控制目录均尚不存在，未部署后台任务。

只做必要回归，没有重复旧GPU Smoke或全量排名。

## 5. 成本更新与剩余风险

实测纯攻击每8图约9.02秒，线性外推3074图约57.8分钟；加文本长度波动、SHA/fsync及封存，Source预计约1–1.5小时。三目标历史同类原生评估参考：IRRA约4分钟、RaSa约2.64小时、APTM约2.38小时；含审计建议总计6–8小时，非运行时保证。

本次证明官方batch8可在该24GiB GPU环境执行，不保证全部后续batch的峰值相同。10-caption图所在batch、长WordPiece候选、外部任务再占显存、首次遇到的Tokenizer边界及磁盘I/O均可能影响资源/耗时。失败保留证据，不改batch、精度、Loss、轮数或候选策略来继续。

原始3份FP32图像Tensor约5.066GiB，另有三模型独立特征/Score及文本审计；保持35GiB磁盘余量。RaSa Full历史CPU RSS约24.4GiB，故保留48GiB可用RAM条件。

正式SGA比较使用最终Joint图文攻击；主方法为G-v2 + H-v1 Joint。SGA Matched Clean采用pre_caption30和源RGB桥，不能与历史Raw Clean不加说明拼成完全公平对比。IRRA是同源评估；RaSa/APTM是同一攻击产物的黑盒迁移。

## 6. 供下一次明确授权后启动Full的命令

**本轮没有执行以下启动命令。** 容量Attempt不写入正式Full，也不将其8图缓存混入Full；正式Full按官方完整batch序列和全局seed重新生成一次。

```bash
mkdir -p /home/lzf/ldx/tmp/sga_official_irra_full_v1
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-sga-batch8-capacity-20261019/src/tbps/baselines/run_sga_irra_full.sh --allow-full >> /home/lzf/ldx/tmp/sga_official_irra_full_v1/launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
```

正式输出：/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/
控制目录：/home/lzf/ldx/tmp/sga_official_irra_full_v1/

```bash
cat /home/lzf/ldx/tmp/sga_official_irra_full_v1/supervisor.pid
cat /home/lzf/ldx/tmp/sga_official_irra_full_v1/supervisor.pgid
cat /home/lzf/ldx/tmp/sga_official_irra_full_v1/phase
tail -n 30 /home/lzf/ldx/tmp/sga_official_irra_full_v1/status.log
tail -n 30 /home/lzf/ldx/tmp/sga_official_irra_full_v1/launcher.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/source.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/irra.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/rasa.log
tail -n 10 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/aptm.log
tail -n 20 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/audit.log
```

Source日志逐8图显示SGA_OFFICIAL_SOURCE_PROGRESS。Target显示SGA_TARGET_ENCODING_PROGRESS（图像6148、文本12312总编码），RaSa/APTM另有原生融合评分块进度。phase依次CPU_PREFLIGHT/RUNNING_SOURCE/RUNNING_IRRA/RUNNING_RASA/RUNNING_APTM/RUNNING_CPU_AUDIT/SUCCESS，失败为FAIL，原阶段保留在status.log。

成功需全部Target PASS及SGA_OFFICIAL_IRRA_CPU_AUDIT_PASS full，最后：
SUCCESS | SGA_OFFICIAL_IRRA_SOURCE_THREE_TARGET_FULL_AND_CPU_AUDIT

必要时仅停止本次独立会话（先核对PID/PGID；以下命令本轮未执行）：

```bash
ps -o pid,ppid,pgid,sid,etime,args -p "$(cat /home/lzf/ldx/tmp/sga_official_irra_full_v1/supervisor.pid)"
kill -TERM -- "-$(cat /home/lzf/ldx/tmp/sga_official_irra_full_v1/supervisor.pgid)"
```

不因SSH断开终止；以supervisor.pid/pgid确认真实进程，不能只凭Shell返回PID判断成功。失败目录、Claim和started保留；禁止直接重新执行同一脚本。等待用户正式Full启动授权。
