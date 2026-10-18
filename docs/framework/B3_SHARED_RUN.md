# B3共享GPU一次性后台实验链

2026-10-18，基准097c96d9deec68c1bfc71b74a2fba5f98e3e81c1。用户授权：自己手动启动一次后台任务，三模型Smoke及独立CPU审计全部PASS后自动继续三模型Full及审计；Codex只准备代码/CPU测试，不部署、不启动GPU、不保持会话。

## 当前入口与最小修改

独立分支codex/tbps-b3-shared-run，工作树/home/lzf/ldx/projects/idea-test-tbps-b3-shared-run-20261018。只改B3 Runner和原run_b3_full.sh，不改Model Adapter、Ranking、攻击算法或原正式结果。

- 删除瞬时利用率≤10%要求；Ollama和其他非B3计算进程的存在不禁止启动。不等待GPU、不轮询、不干预其他进程。
- 保留启动空闲显存≥3072MiB检查，按此前三原生模型实测峰值预留空间。不足立即明确失败。之后资源变化导致的实际CUDA OOM/错误保留原堆栈与现场，停止整条链，不调精度/batch、不重试。
- 全链启动只取得一次GPU0资源及其他B3进程快照，保存startup_gpu.json。各评估子进程复用这份**启动观察**，不冒充实时状态、不重复执行nvidia-smi。显存峰值来自各阶段真实CUDA统计。
- flock、不可重入started标记、现有B3进程及Smoke/Full目录检查共同防止重复启动和覆盖。任何失败不删除现场或消费记录；再次尝试必须先人工审查，脚本不自动清理或重试。
- nohup+setsid创建独立会话，真实监督器PID/PGID写入文件。终止独立进程组会同时停止监督器及当前Python，不触碰Ollama。
- 新配置baseline_b3_shared.server.json绑定75个Python来源与Shell SHA；模型、预算、种子、FP32、原生评分和原Checkpoint不变。旧097阶段源码树、FAIL、ready十图CPU产物和协议保持只读；新输出隔离以避免修改来源SHA后改签旧产物。

使用已有B3历史3074图/97批FP32缓存。READ_HISTORICAL_B3_RGB阶段只复用既有native像素并执行已经验证的CPU尺寸映射/原子保存，不加载Generator、不重新计算G/H攻击。Full的Clean与G-v2 Image-only对照和Clean文本表示仍使用已审计同模型来源。

## 固定执行顺序

一次资源检查 → 完整CPU来源预检 → IRRA Smoke/CPU审计 → RaSa Smoke/CPU审计 → APTM Smoke/CPU审计 → 读取映射历史B3 RGB → IRRA Full/CPU审计 → RaSa Full/CPU审计 → APTM Full/CPU审计 → SUCCESS。

每个Smoke固定10 Gallery/10 Query；Full固定3074 Gallery/6156 Query。退出码和精确PASS标志同时满足才进入下一步，Smoke失败不能进入Full。方法顺序Clean → AP-Attack B3 → G-v2 Image-only；对外四项R@1、R@5、R@10、mAP，百分数两位小数；保留这四项原始精度，不输出额外指标表。

## 用户启动：Xshell可直接复制的一条命令

```bash
mkdir -p /home/lzf/ldx/tmp/b3_shared_run && { nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-b3-shared-run-20261018/src/tbps/baselines/run_b3_full.sh --allow-full >> /home/lzf/ldx/tmp/b3_shared_run/launcher.log 2>&1 < /dev/null & echo "启动器PID=$!；真实监督器PID见 /home/lzf/ldx/tmp/b3_shared_run/supervisor.pid"; }
```

首次启动后可关闭Xshell/SSH，不依赖Codex会话。勿在运行期间改源码/config、pull/checkout或清理目录。重复执行只会被拒绝，launcher.log采用追加，不覆盖原启动日志。

环境由脚本固定：/home/lzf/ldx/envs/tbps-image/bin/python，CUDA_VISIBLE_DEVICES=0，PYTHONPATH指向新树src，FP32/math/no_tf32，线程2，TMPDIR=/home/lzf/ldx/tmp，Tokenizer离线。

## 查看阶段、日志及GPU

```bash
cat /home/lzf/ldx/tmp/b3_shared_run/phase
cat /home/lzf/ldx/tmp/b3_shared_run/supervisor.pid /home/lzf/ldx/tmp/b3_shared_run/child.pid
tail -n 30 /home/lzf/ldx/tmp/b3_shared_run/status.log
tail -n 30 "$(cat /home/lzf/ldx/tmp/b3_shared_run/current_log)"
nvidia-smi
```

持续看阶段：tail -f /home/lzf/ldx/tmp/b3_shared_run/status.log。看当前子阶段详细进度：tail -f "$(cat /home/lzf/ldx/tmp/b3_shared_run/current_log)"；阶段改变后重新执行。GPU显示命令由用户按需调用，不是自动等待器。

phase可能为RESOURCE_CHECK、CPU_PREFLIGHT、RUNNING_SMOKE_irra/rasa/aptm、RUNNING_SMOKE_AUDIT_*、READ_HISTORICAL_B3_RGB、RUNNING_FULL_*、RUNNING_FULL_AUDIT_*、SUCCESS、FAIL、CANCELLED。所有阶段日志位于/home/lzf/ldx/tmp/b3_shared_run/logs/；资源拒绝详见startup_gpu.json及resource_errors.log。最终成功标志为SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT。

## 查看最终四列结果

```bash
for model in irra rasa aptm; do
  echo "===== $model ====="
  cat "/home/lzf/ldx/outputs/tbps-framework-b3-shared-evaluation/full/$model/completed/metrics_2dp.csv"
done
grep -F "SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT" /home/lzf/ldx/tmp/b3_shared_run/status.log
```

同目录metrics.csv/summary.json保留四项原精度。execution/report.json记录真实加载/前向、模型冻结摘要、耗时、峰值显存及错误。只有最终SUCCESS和各CPU审计PASS同时存在，才能标记本轮完整完成。没有结果时不得用旧B3、Smoke或其他模型数字补齐。

## 停止本次任务

```bash
C=/home/lzf/ldx/tmp/b3_shared_run
p=$(cat "$C/supervisor.pid") && g=$(cat "$C/supervisor.pgid") &&
if [ "$p" = "$g" ] && ps -p "$p" -o args= | grep -Fq '/home/lzf/ldx/projects/idea-test-tbps-b3-shared-run-20261018/src/tbps/baselines/run_b3_full.sh'; then
  kill -TERM -- "-$g"
else
  echo "未找到本任务的活动监督器，不发送信号"
fi
```

此命令核对本任务实际监督器后，仅终止其独立进程组（含当前Python）。TERM后保留全部已发布/未提交资产及日志，phase记录CANCELLED；不要删除started再直接重跑。

## CPU验证与执行边界

- 新增17项共享资源/真实Bash链CPU测试：高利用率99%和Ollama放行、3072MiB边界、其他B3进程拒绝、快照仅一次、来源冲突拒绝、真实脚本顺序、缺少PASS拒绝、Smoke/Full/审计失败即停、锁与重复启动、独立进程组取消。
- 原32项B3测试全部通过；合计49项B3 CPU测试。相关模型/Engine回归149项通过、6项旧配置路径测试按既定筛选未执行。Fixture实际执行Bash，模型CLI是CPU文件桩，不调用GPU。
- Full CPU来源预检PASS，3074/6156、97历史批、CUDA=false；日志/home/lzf/ldx/tmp/b3_shared_full_cpu_preflight.log。
- CPU日志/home/lzf/ldx/tmp/b3_shared_cpu_tests.log、b3_shared_regression.log。
- 本轮不执行resources真实查询、Smoke、Full、全量RGB映射或后台部署。正式控制目录与新评估输出均尚不存在；仍待用户手动启动。

历史保护复核：101份旧B3证据的SHA/inode/size/mtime全部一致；12203份既有资产的inode/size/mtime全部一致。原B3工作树仍为097c96d9deec68c1bfc71b74a2fba5f98e3e81c1且干净。bash -n与git diff --check通过。

实现及CPU验证Commit：d229e6ea5e9c2f5a764d4fbb297f8397c48e9fed；当前发布只补充Git索引，不改变已绑定Runner/Shell/配置。
