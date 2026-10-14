# Step7.5.3：Aggregated Objective Priority Projection

日期：2026-10-04。本轮用户明确将 aggregated objective 作为方法定义，停止追求和逐属性 slow backward 的逐像素等价。**实现、33项CPU tests和五句短GPU smoke通过；正式100句未启动。**

## Prototype 和冻结边界

服务器：

`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step753_aggregated_objective_20261004/`

源码、tests、原始outputs只在服务器，Windows只更新本摘要、AGENTS和项目状态。旧Step7.5/7.5.1/7.5.2文件、失败证据与正式源码保持，无commit/push。

复用 frozen AttributeInstance、ontology、reader、basis、S_raw、原100句Natural Dynamic-K manifest及排序、JVP、IRRA和官方TTA。每轮10steps、8/255、step2/255、momentum .9、原IG、5 scales（原图及.5/.75/1.25/1.5）、每scale6 transforms均不改。无caption loss、score重算、训练、特征重提或generator。

真实环境 `/home/lzf/ldx/envs/tbps-image/bin/python` 为torch2.5.1+cu124，4090；没有更换软件环境。

## 三模式定义与真实接线

- **PREFIX_BASELINE**：直接使用原PREFIX的未归一化JVP向量均值、原官方单一目标IG路径。不做逐属性backward诊断，不替换baseline梯度。
- **SCORE_WEIGHTED_PREFIX**：每个真实IG节点显式构造 `terms=[L(a_i)]`，`weighted_loss=sum((S_i/sum S)*terms_i)`，一次image backward。不是先独立backward再手工相加。
- **SCORE_WEIGHTED_PRIORITY_PROJECTION**：每节点分别对historical weighted objective和当前新增属性目标求导；先按原scaled/transformed路径因子累积完整历史IG `g_H` 和新增IG `g_N`，之后判断dot<0，仅修正 `g_N`，再按历史score总和/当前score合成。

投影数值沿用既有定义：fp64 dot/norm、machine epsilon、结果回到fp32；历史方向不修改。投影位于完整IG之后、原mean-abs normalization/momentum/sign之前，既有AST hook检查正式更新消费的tensor就是组合器返回tensor。没有在未变换round输入点做投影，也没有把投影挪到IG节点内部。

weighted每真实IG节点1次image backward，priority2次；原正式每step含5个scaled节点、30个transformed节点，所以分别35/70次。不随当前prefix属性数增加独立IRRA backward；原adaptive变换概率梯度与后置观察诊断保持。

三模式仅共享同一次完整round1的image/momentum/observer/RNG/trace checkpoint；后续分叉状态各自求导。Baseline作为categorical trace anchor，其它模式回放变换ID且保持同样RNG消耗，各自原adaptive概率更新不改。

`aggregated_ig.py` 是本阶段方法实现。原slow只作为历史证据，不作为正式gate，不导入SLOW_REFERENCE、不跑slow-vs-new攻击对照。之前FAST的fp32轨迹分叉不再构成拒绝本阶段方法的理由。

## CPU tests

`outputs/cpu_tests.log`：**33 passed**。

覆盖：k1/3/5/6/7的目标求导与固定backward次数；冲突/无冲突/历史不变/新方向身份检查；投影真实进入原momentum；AST只替换目标梯度获取及观察点，原变换概率、normalization、momentum/sign/projection保持；原PREFIX在toy k1/2/3/5/7逐值回归；共享round1/noise/trace/RNG；多scale/transforms；短观察器仍传steps=10；固定100句358轮manifest与148份旧来源SHA；显式授权的用户诊断入口与无slow-equivalence gate。

## 五句短GPU smoke

固定原100句的34:0(k1)、27701:0(k3)、17359:1(k5)、25488:1(k6)、24308:1(k7)。没有按结果重新选样本。

所有官方调用参数均为10steps/原IG/5scales/6transforms：
- k1完成原10步，三mode共享完整实际结果，final image/momentum/metrics exact；
- k>1每轮在第1个完整优化步和投影完成后由只读observer截停；走完该句所有属性轮次，累计image/momentum正常传递；
- 每mode22逻辑轮，共享round1后56物理轮、**65个实际优化步**；
- 此smoke不是每轮10步的完整攻击评测，不据它宣布解决历史recovery或三mode性能优势。

输出 `outputs/smoke/`，日志 `outputs/smoke.log`，已COMPLETE。共享round1、noise/trace/RNG、zero、IRRA/实例/S/TopK/JVP冻结全部通过。JVP最大FD relative error .00103507（.1035%），沿原门禁。score backward=0、caption loss=0。148份冻结上游SHA前后不变；protocol共158项来源/本轮源码锁定。正式服务器tracked/staged diff为空。

最大L∞ .03137257695198059，相对数学8/255仅多2.79324e−8，属于既有fp32舍入；没有更改预算或projection。GPU已停止。

## Projection 行为

Priority共17个后续轮次：
- 冲突7、无冲突10；
- 冲突后最大 `abs(cos(g_N',g_H))=2.54758470e−10`；
- 所有无冲突 `projected_new_unchanged_exact=true`，修改max abs=0；
- 历史方向不变、只修新增属性、正式momentum输入身份核验全通过。

25488:1：
- round3/4/5触发；
- cosine_before −.40736051/−.03388987/−.36442626；
- cosine_after −6.89535e−11/+2.17830e−10/−9.03171e−11；
- round2/6无冲突，新增方向保持。

24308:1：
- round2–7全部无冲突，修改量0；
- 此smoke中weighted与priority逐轮image/metrics完全相同；
- F轨迹（含clean）：1.4874→1.0347→2.5405→38.5359→233.6053→738.5567→1310.6321→2184.5610。round1有恢复，round2–7继续增强；
- 原baseline最后F2316.9783，不能宣称本smoke证明weighting更强；这里证明projection没有无故干预该case，完整10步行为仍待正式100句。

25488:1的weighted F轨迹：2.3027→10.3009→101.1770→729.0984→1763.0573→2762.2742→3909.9653；
priority：2.3027→10.3009→101.1770→778.0548→1820.7417→2759.2617→3856.0806。这是短轨迹机制观察，不作为正式forgetting结论。

## 计时与预计100句耗时

GPU空闲。下表为对应caption末轮的**每个完整原配置优化步**，单位秒；不含JVP准备、retrieval评价和JSON写盘：

| k | PREFIX | weighted | priority |
|---|---:|---:|---:|
| 3 | .755 | .744 | 1.283 |
| 5 | .835 | .823 | 1.418 |
| 6 | .780 | .798 | 1.327 |
| 7 | .759 | .786 | 1.301 |

k增长时weighted/priority每step耗时基本稳定；全句步数随K线性增加，已恢复到可接受的近似O(K)成本。loss项构造仍有轻微随K增加的标量工作，不声称总FLOPs与K完全无关。此处没有slow timing对照或speedup gate。

五句总计含末尾SHA回检为77.45s；实际case_time及每轮timing保留于JSON。

正式原100句：每mode358逻辑轮，共享100个round1后874物理轮、8740实际优化步，其中PREFIX3580步、weighted后续2580步、priority后续2580步。
按短测平均每step .773/.785/1.346s，纯攻击步估计约**2.30小时**；含准备、评估、SHA、输出与波动，空闲4090暂按**2.5–3小时**预算，非实测完整运行承诺。有并发时会更长。估计明细在服务器 `outputs/timing_estimate.json`。

## 手动正式100句命令

以下命令**未执行**：

```bash
tmux new-session -d -s tbps-step753 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u prototypes/attribute_tta_step753_aggregated_objective_20261004/run.py --mode diagnostic --allow-diagnostic --manifest prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/clean_scores/selection_manifest.json --output prototypes/attribute_tta_step753_aggregated_objective_20261004/outputs/diagnostic > prototypes/attribute_tta_step753_aggregated_objective_20261004/outputs/diagnostic.log 2>&1'
```

- log：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step753_aggregated_objective_20261004/outputs/diagnostic.log`
- outputs：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step753_aggregated_objective_20261004/outputs/diagnostic/`
- 完成标志：日志末尾COMPLETE和 `completed.json`。
- 首先检查：case_count100、steps_per_round10、logical_rounds_per_mode358、physical_rounds874、actual_optimization_steps8740、method_audit_passed=true、slow_equivalence_gate=false、共享round1/随机控制/source SHA/frozen检查通过、score backward0/caption loss0。
- 输入固定原Step7.4 manifest和clean score缓存，只读使用，不重评分/重排/重新TopK。
- diagnostic必须显式 `--allow-diagnostic`、不得覆盖已完成输出；入口先验证本阶段smoke与其冻结来源/SHA，再加载模型。CPU只读preflight已核对100句/358轮；没有调用正式main或启动GPU。

输出：`summary.json`、`per_caption_results.json`、`per_round_results.json`、`conflict_statistics.json`、`projection_statistics.json`、`method_audit.json`、`protocol.json`、`audit.json`、`completed.json`。这些仍是固定clean gallery下的paired replacement机制诊断，未构造共享adversarial gallery，也不是正式TBPS总体性能。

本轮仅确认新方法实现和接线成立，是否减少recovery或提升最终攻击留待用户100句。没有自动扩大实验、修改冻结前置组件、commit或push。
