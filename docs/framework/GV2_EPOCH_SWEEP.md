# G-v2六节点三模型消融评估准备

本轮基准6187eed96ed9eff63b8da37f5d98b5adddf75300；独立分支codex/gv2-six-epoch-evaluation。
原准备阶段仅完成CPU来源核验；后续用户已明确批准共享GPU并授权唯一真实Smoke。
共享准入/实际执行结果见[共享策略与Smoke报告](GV2_EPOCH_SWEEP_SHARED_SMOKE.md)。
本说明中的原准备状态为历史快照。2026-10-20获最终授权后已使用原冻结入口启动Full，
2026-10-21 19:07:27整链及15份独立CPU审计已完成；
[正式结果/曲线/只读核验](results/gv2_epoch_sweep_full_analysis/REPORT.md)，
[启动记录](GV2_EPOCH_SWEEP_FULL_STARTUP.md)保持历史快照，不再是最新完成状态。
详见[准备报告](GV2_EPOCH_SWEEP_PREPARATION_REPORT.md)和[证据](results/gv2_epoch_sweep_preparation/evidence.json)。

## 复用与边界

- 原ImageOnlyGenerator(width32/blocks4)、inference_once.generate_once及budget_projection逐字节复用。
  不改G(I)、Global/Local/Retain、K+1训练、权重或预算，不使用Optimizer。
- 原六点Checkpoint容器schema不同，新增只读加载接线验证Epoch/完整contract/父SHA，然后严格装入原G；
  PyTorch容器会在CPU读入其它状态，但只有generator字段用于推理，不构造或恢复Adam。
- 直接读取generation_v3原始FP32 RGB；不从PIL重新量化，不重新生成H。
- 原MaterializationPlan只接受G10。新Epoch采用独立gv2-epoch-rgb-variant-1身份，
  引用旧Plan/原RGB单元和同一H7文本，保持旧Plan及MaterializedUnit不变。
- ArtifactStore、原image_checks、FullReference、原生Adapter/FeatureCache/NativeFeatureCache、
  CachedIRRA/ChunkCachedScorer/APTMSetScorer、EvaluationEngine及publish_result/load_result均复用。
  不复制生成器、投影、Tokenizer、Ranking或评分公式。
- IRRA dot；RaSa原ITM Top128；APTM原ITM/global及完整Query/Gallery集合归一化。
  Smoke原生Top10，不能用其结果宣称Full Top128已验证。
- H7历史6156条文本/完整动作已核对：6116编辑、40 NOOP、13537字符替换。
  各Victim只复用自身ModelSpace的Raw/H文本特征，不跨模型借用Embedding。
- FullReference的adv_text分支不传image origin；新接线按原Query UTF-8 SHA建立文本AttackOrigin，
  避免将Gallery的SHA误当作文本原始身份；旧桥文件不改。
- 对外CSV仅R@1/R@5/R@10/mAP，百分数；保留原精度和两位小数。
  底层Engine保存原有schema，没有改动其指标定义。

## 运行环境与CPU预检

以下命令在Ubuntu服务器执行。CPU预检不加载Victim、初始化CUDA、生成图像或重算Full排名。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/src
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.gv2_epoch_eval \
  --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/configs/framework/gv2_epoch_sweep.server.json \
  --action preflight --scope full
```

成功标志：GV2_SWEEP_CPU_PREFLIGHT_PASS。核验官方3074/6156、有序PID/配对关系、
六个权重SHA、扩展训练来源、原架构strict_state_dict_load、H7、历史Full来源和已有审计。
CPU解析权重时禁止神经forward/CUDA；旧current.json、训练恢复入口不调用。

## 单次最小Smoke（已完成；以下为执行索引，不得重复运行）

预先固定Gallery原序0–8及15，Query原序前9条及cuhk:37147:1，10+10均有正例。
后者是已知历史NOOP；选择在生成前冻结，不根据攻击效果选样本。
G10和G60各生成10图；G10与原generation_v3像素对照max_abs≤1e-6。
G60同一Gallery供三个原生Target；文本特征只读复用。各Target编码10图/0文。
总G加载2次/前向20次，H0，三Victim各加载1次；无Backward/Optimizer/JVP。
随后各Target独立CPU来源/缓存/结果一致性审计，不重新计算指标。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/src
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.gv2_epoch_eval \
  --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/configs/framework/gv2_epoch_sweep.server.json \
  --action smoke --scope smoke --allow-smoke
```

唯一输出：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/smoke_v1。
默认不执行GPU，显式allow-smoke才可进入。空闲显存不足6144MiB或RAM/磁盘不足时停止；利用率/其他进程仅记录，不等待、不干预其他任务。
最初两次资源检查未创建Attempt；随后共享策略修订后已完成唯一smoke_v1及三份CPU审计。
当前不得重复执行本命令、覆盖旧Attempt或自行创建新版本。实际结果见共享策略与Smoke报告。
运行之后若任何阶段失败，保留Claim/pending/FAIL/日志，禁止自动重复该Attempt。

## 正式Full拟执行流程（本阶段未获授权，勿直接启动）

入口已准备，必须先有上述真实Smoke PASS及三份CPU审计完成回执，
再由用户明确授权；仅CPU预检PASS不足以运行Full。
新产物根：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full/。
每个epoch_0020/0030/0040/0050/0060：
images保存3074份original/adversarial/effective_delta紧凑FP32，三个Target子目录保存原生表示和两模式Score/结果。
Epoch10只导出已通过审计的原Full引用，不重新生成/评分；Clean/H Text-only保留历史对应原精度控制值。

未来授权后的后台命令如下，本轮没有运行/部署：

```bash
nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/src/tbps/ablations/run_gv2_sweep_full.sh --allow-full \
  >> /home/lzf/ldx/tmp/gv2_epoch_sweep_full_launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
```

控制目录/home/lzf/ldx/tmp/gv2_epoch_sweep_full_v1必须不存在。脚本按：
CPU_PREFLIGHT → E10三模型只读引用/审计引用 → 五Epoch依次G生成/IRRA/RaSa/APTM/各CPU审计。
不部署轮询，不自动重试/恢复。成功标志：
GV2_FIVE_NEW_EPOCHS_THREE_TARGET_FULL_AND_CPU_AUDIT。

单阶段命令（只在对应正式授权下使用）：

```bash
# E20生成一次；其它节点仅将20改为30/40/50/60
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.gv2_epoch_eval \
 --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/configs/framework/gv2_epoch_sweep.server.json \
 --scope full --action generate --epoch 20 --allow-full-materialization
# 同一份E20像素的IRRA两模式；model可为rasa/aptm
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.gv2_epoch_eval \
 --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/configs/framework/gv2_epoch_sweep.server.json \
 --scope full --action target --epoch 20 --model irra --allow-full-evaluation
# 独立CPU审计；只校验产物/来源/CSV与既有Engine结果，不重新排名
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.gv2_epoch_eval \
 --config /home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020/configs/framework/gv2_epoch_sweep.server.json \
 --scope full --action audit --epoch 20 --model irra
```

每次G前先Claim；已提交单元不可变并立即读回。缺Proof的Claim、pending、failure或完成态拒绝重入。
只在另行审查授权的明确恢复场景下，--resume可复用完整已提交单元；未提交前向不能自动重做。
目标缓存允许相同来源下复用已提交chunk；存在失败execution时拒绝自动恢复。
本次不提供绕过失败保护的清理/重命名命令。

未来查看阶段和日志：

```bash
cat /home/lzf/ldx/tmp/gv2_epoch_sweep_full_v1/phase
tail -n 30 /home/lzf/ldx/tmp/gv2_epoch_sweep_full_v1/status.log
tail -n 30 /home/lzf/ldx/tmp/gv2_epoch_sweep_full_launcher.log
```

## 成本与限制

五节点新增15370次G；每个Target共15370张Adv图编码，文本编码0；
RaSa/APTM每Epoch各约1575936个ITM pair（6156×128×2模式）。
历史完整四模式三Target约219/9496/8561秒；按两模式、五节点估算评分主耗时约12.5小时，
加生成、SHA、CPU预检及读回，整链暂估15–20小时。没有本轮GPU实测支撑，不能作为运行保证。
新像素每Epoch约5.066GiB、合计25.33GiB；RaSa图像token缓存约25.4GiB、
APTM约2.9GiB，加Score/Manifest/日志，建议新增65–80GiB并保留至少100GiB空闲。
原RaSa Full RSS约24.43GiB，配置可用RAM≥32GiB；启动可用显存≥6144MiB，利用率只监控；显存快照不能保证后续无OOM。
旧三Target实测reserved约1.0/2.03/1.05GiB，不能据此保证新GPU环境不会OOM。

本次不生成五节点Full Gallery，不启动训练或Full，不根据Test指标选点或调整算法。
G10新加载接线的像素SHA exact、G60生成和三Target Smoke/独立CPU审计已在固定样本通过。
完整Top128、五节点正式生成/评分及六节点趋势尚未执行；小样本PASS不代替正式Full。
