# Step7.4：CAPPED-4 vs NATURAL-DYNAMIC-K（2026-10-04）

## 当前事实

本轮仅在服务器新建独立 prototype：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step74_dynamic_k_20261004/`。
源码为 `selection_audit.py`、`deps.py`、`prepare_scores.py`、`paired_prefix.py`、`run.py`、`test_dynamic_k.py`；正式 tracked/staged 源码未改，没有训练、commit/push。

继承 Step7.3.2 用户完整25句结果：PREFIX三项final指标均13胜10负1平（24非空）；23多轮final=best从7增到11，F best-final mean从4820.55降到1672.27；round3恢复11/19→4/19，round4恢复8/9→5/9。收益主要在k4（7胜2负），k3为4胜6负。本轮将PREFIX作为主实验scheduler，不宣称已经冻结其有效性，不继续设计scheduler。

本轮只有两个逻辑模式：

- `CAPPED_4`：k=min(ceil(N_eff),m,4)。
- `NATURAL_DYNAMIC_K`：k=min(ceil(N_eff),m)。

固定 S_raw=||QᵀwᵀG||₂、gamma=3、beta=1、原AttributeInstance/source spans/region/value/provenance、reader/basis、原JVP、PREFIX算术平均loss、TTA变换/IG/动量、10steps/round及累计8/255预算。不引入caption额外loss，不修改冻结接口。

## Clean score preparation

原Step5.1只保存25句score，无法直接生成100句validation自然k分层。用户已明确授权新增独立准备阶段：仅用户手动运行；对新增caption各做一次clean retrieval backward，攻击阶段完全只读缓存。

`prepare_scores.py --mode prepare --allow-score-preparation` 消费原13700条PID-held-out validation（CUHK原train内部固定PID划分，不是改split），直接调用冻结Step5.1的 `baseline_gradient`、`FrozenLocalization.attach`、`score_instances`。原25句中属于validation的16句复用已核验score；其余各一次backward。保留原w与缓存上的semantic representation；不训练reader、不重建basis、不重写H/Z缓存或提取34054图库。准备阶段必要的单caption原IRRA forward/backward不等于改用缓存猜测G。

主缓存：`outputs/clean_scores/score_cache.jsonl`；含实例、S_raw、w、状态与source/paired metadata。另有protocol/source SHA、cache SHA、completed/audit。任何旧来源不一致即拒绝；rank0/unresolved/无表示/非正S不进候选，不新增fallback。

攻击入口只读取缓存与已生成manifest；门禁仅核验metadata、score、排名及selected prefix，不重算entropy/K，不调用scorer。Score来源检查锁定真正参与准备的代码，攻击runner/tests不参与score数值来源；当前攻击代码另锁SHA。最终门禁改动没有改变smoke时的攻击main或paired-prefix执行代码，原source快照保留于服务器smoke_source_snapshot。

## 100句抽样规则（攻击结果读取前固定）

1. 仅原validation、有至少一个通过冻结门禁且S_raw>0的实例。
2. 使用S_raw降序，tie保留原instance输入顺序；p=softmax(3 log S)、N_eff=exp(entropy)，自然k不截断。
3. 分bin：1、2、3、4、5、6、7+。
4. 每bin按SHA256("42|caption_id")升序，caption_id破hash同分；按上述固定bin顺序轮流取1条，直到100。
5. 稀缺bin全取，剩余名额分给尚有样本的bin。不存在的bin如实报告，不制造样本；m、精确natural k、capped k、是否截断均逐句输出。
6. 记录bin_available/bin_selected、自然k/m分布、distinct PID与每PID最大复用量。分层100句的均值不是validation真实分布加权总体均值。

100句manifest须在用户完整准备后才能得到；当前只有3句smoke，不能宣称已得到100句自然k分布。正式攻击入口要求完整准备及100个冻结rows，避免把不足样本冒充100句。

## 共同前缀执行

直接调用旧Step7.3.2 `cached_round_guidance("PREFIX_NEW",...)` 与 `execute`，不复制/改写TTA优化循环。

每句只执行一条真实PREFIX轨迹，CAPPED保留min(k,4)处的实际checkpoint，NATURAL从同一image/momentum/RNG/observer状态继续第5轮以后。共同前缀的step/image/metric/noise/transform trace与RNG精确一致；k<=4两模式完整终态相同。不存在重新执行导致的CUDA非确定性对照混杂，也不以F筛选best作为实际final。

旧Step72准备函数有单次<=4项门禁；新adapter对连续chunk复用同一门禁，然后恢复原全局selection_rank，不修改旧函数或实例。JVP对自然selected序列预计算一次/实例，前4轮guidance仍是原前4项的均值；剩余向量不进入这些轮的loss。

两模式每轮同为10steps，但NATURAL有额外轮次，这是本轮明确改变的因素。实际计算共用前缀，completed同时报告两模式逻辑轮数与物理轮数。结果只能直接说明该TTA设置中的额外属性轮次价值，不能单凭它证明未来G训练的收益。

## 固定诊断规则及输出

round4后累计差：F_natural_final−F_round4。

- >=100：实质提升。
- <=−100：明显变差。
- −100<差<100：近似不变。

这是事前固定的绝对100个soft-rank单位，仅为描述性分组；无阈值的F/cosine/rank连续差值、严格wins/losses/ties同时完整输出，不根据结果调阈值。

`summary.json`：全部100句与真正被截断的k>4子集，两模式final/clean→final均值、中位数、范围；总体及每自然k bin的win/loss/tie；best-prefix retention、F/rank best-final损失；上述round4后分类。best按F最大、earliest tie，rank取同一F-best状态，不另按rank择优。

`round_marginal_effects.json`：每round、round5、6、7+、5+的ΔF/Δcosine/Δrank、恢复比例、实际更新/clipping/update cosine与aggregate input-gradient norm。7+为所有对应轮的event加权汇总，同caption多轮不视为独立样本。

`per_caption_results.json`：完整source metadata、排名/S_raw、JVP报告、两个模式所有states/steps/rounds、累计预算与共同前缀checks。

## Tests / smoke

- 最终42项CPU tests通过（37项本prototype、5项旧PREFIX回归），覆盖自然k0–9、稀缺bin/确定性100抽样、source-span与same-value/different-span、rank0/unresolved排除、旧门禁分chunk、每caption一次clean backward、共同前缀以及旧PREFIX原TTA独立逐值回归、CLI full门禁、SHA变更拒绝、attack不重新Top-K。
- 固定3句：34:0（m1/k1）、26:0（m5/k5）、9781:0（m12/k7）。前两句复用旧score；新caption由原v5.5门禁候选数最多、numeric row_id破同分预先选定，未按新score或attack结果选样本。
- Score smoke：1次新增clean backward、2条旧score复用，18 scored/1 rank0；zero、cache表示回归、IRRA/reader冻结通过。
- Attack smoke：2steps/round、1transform、scales=None；逻辑CAPPED9轮/NATURAL13轮，实际13轮26steps。
- k1完整结果exact；k5/k7共同前4轮exact，只有NATURAL第5+轮继续。score backward=0，JVP13次且不在step内重构；FD最大相对误差0.00118984（0.119%）。
- 最大L∞=0.03137257695，比数学8/255多2.79e-8，仅原fp32舍入。无caption loss/新objective/训练。
- 119份旧冻结来源SHA保持；正式tracked/staged diff为空，GPU停止。完整clean_scores与diagnostic目录均未创建。
- 初次tests的fixture缺instance_id已修复，失败log保留；smoke后只收紧缓存门禁，未改攻击main/执行数学，最终CPU tests验证该门禁仅读已有plan。

产物在服务器 `outputs/score_smoke/`、`outputs/smoke/`、对应logs及`outputs/tests_final.log`。不从减配smoke判断是否去掉kmax。

## 用户手动运行（Codex未执行）

先完整准备clean分数及100句manifest：

```bash
tmux new-session -d -s tbps-step74-scores 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u prototypes/attribute_tta_step74_dynamic_k_20261004/prepare_scores.py --mode prepare --allow-score-preparation --output prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/clean_scores > prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/clean_scores.log 2>&1'
```

完成看 `SCORE_PREPARATION_COMPLETE` 和 `clean_scores/completed.json`，核对caption_count=13700、new_clean_retrieval_backwards=13684、reused_verified_scores=16；manifest case_count=100及bin分布。已有输出拒绝覆盖，失败后不要直接覆盖证据。

随后手动运行100句：

```bash
tmux new-session -d -s tbps-step74-dynamic-k 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u prototypes/attribute_tta_step74_dynamic_k_20261004/run.py --mode diagnostic --allow-diagnostic --manifest prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/clean_scores/selection_manifest.json --output prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/diagnostic > prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/diagnostic.log 2>&1'
```

完成看log末尾 `COMPLETE` 与 `diagnostic/completed.json`：case_count=100、steps_per_round=10、common_prefix_exact=true、k_le4_final_exact=true、score_backwards_during_attack=0、SHA/frozen通过、caption_loss_terms=0。

输出根：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/`。

正式耗时取决于prepare阶段新增scoring及最终natural k总轮数；不以短smoke时间假装正式速度。参考Step7.3.2完整轨迹约8秒/实际属性轮，100句attack可据manifest总natural k粗估，另加loading/JVP/来源核验；机器负载影响明显。准备全13700句可能比attack更耗时，Codex未实测全量时间。

## 回传模板

1. 完成标志、100句自然k/m分布、bin_available/bin_selected、被截断数、distinct PID。
2. CAPPED vs NATURAL三项final与clean→final mean/median/range；全部样本及k>4子集wins/losses/ties；每bin结果。
3. round5/6/7+ ΔF mean/median/range、恢复比例及样本/轮数。
4. 两模式final=best比例、F_best−F_final、rank_best−rank_final（尤其k>4子集）。
5. round4之后实质提升/近似不变/明显变差数量（固定±100F）及更新/clipping/预算。
6. 基于结果判断是否建议去掉k_max=4，不修改Attribute/S/reader/JVP或继续设计scheduler。

回传文件位置：`diagnostic/{summary.json,per_caption_results.json,round_marginal_effects.json,selection_manifest.json,completed.json,audit.json,protocol.json}`、`outputs/diagnostic.log`及`clean_scores/{audit.json,completed.json}`。
