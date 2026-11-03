# SEP性能优化与TBPS迁移协议审查（CPU Only，2026-11-03）

**本轮完成CPU审查，未部署优化、未启动CUDA/SEP或重新执行历史Query。** 固定Manifest的PID分组预计算已在给定相同CPU分数矩阵下验证bitwise exact；仍使用原高级索引、成员顺序、max与stack算子。原47项+新增97项共144项CPU测试通过，其中80个含位模式断言的测试80/80通过。CPU微基准中CUHK形状单条聚合由430.783ms降至8.229ms，实测52.35倍；这不是完整SEP攻击加速。

基准为停止诊断Commit `b4a62f31e2bf8cab31fbd9beb1b3bb1e8bd53ba0`。原执行树`idea-test-tbps-cpgc-sep-no-gpu-floor-20261101`保持冻结，本次独立`codex/sep-cpu-optimization-review`。所有原Source/Claim、C-PGC、G-v2、G+H资产只读；SEP仍为0/6 Full、44条发布、第45条未完成Claim。

## A. 性能瓶颈与等价性边界

记G为Gallery大小，P为PID数，B为一次给定分数矩阵的行数，U为一个候选调用中未缓存的文本数。

| 代码位置 | 问题 | 原复杂度/工作量 | 建议方案 | 当前等价性 |
|---|---|---|---|---|
| sep_bridge.py:12–18 | 每次聚合对每个PID重新遍历全部Gallery | Python比较O(PG)，再做O(BG)归约与P个小算子 | 固定Manifest构建一次PIDPlan，按原索引list执行原max | **A：已测bitwise exact**；构建O(G+P)，每次索引物化O(G)+原归约 |
| sep_source.py:50及95 | 反复构造gallery_pids、PID顺序 | 每次O(G)元数据构造 | 同一个不可变Manifest内复用相同tuple/order | **A限定**：只复用本次已验证Plan中的身份/分组；不新建跨Manifest全局缓存 |
| sep_source.py:40–53 | 官方批候选在包装层逐text编码与评分 | U次IRRA文本、U次RaSa文本+128U融合pair | 合批候选，按原顺序散回 | **B**：数学等价不保证浮点或排名等价 |
| cache.py:108–126、victims.py:70–93 | 全图库Top128后，NativeRasa又对这128个做相似度/TopK与分块融合 | O(UGd)+128U融合pair；16一批为8U融合调用 | 复用TopK及其处理顺序、避免重复元数据 | **B**：全局Top128必需；不能直接删第二次排序，pair分组/处理顺序可能变 |
| cache.py:65–80 | Token块按所选索引拼接，索引/数据频繁转CPU/GPU | 与128U token收集/传输相关 | 复用验证过的Token块；减少冗余拷贝 | 既有CPU缓存回归PASS；新增搬运/流/驻留方案仍为**B** |
| sep_source.py:45–54 | 分数到CPU聚合，聚合结果再转GPU | 每未缓存文本两代理多次同步 | 最小化无用复制；保持现有CPU归约 | 仅CPU固定数据的分组复用为A；把max移到GPU等为**B** |
| sep_source.py:115–126 | 整个attack启用Python全局trace，深层Python调用也产生call事件 | 随深层Python调用次数增长 | 只在不影响观测的深层作用域暂停trace并finally恢复 | **B**：合成观察器输出相同，但未验证完整真实栈/并发调试钩子 |
| 官方SEP_Attack.py:116–117 | 零重要性向量乘10不改变值，无退出条件 | 全零时无界 | 拒绝、超时或定义回退 | **C**：边界行为改变，未修官方算法 |

### PIDPlan为何可以保持结果

独立实现位于[sep_cpu_review_support.py](../../../../tests/framework/sep_cpu_review_support.py)，生产src不引用它。

1. `order=tuple(dict.fromkeys(gallery_pids))`与原实现一致，不排序PID；PersonID保留dataset命名空间。
2. 对正式Manifest的不可变PersonID/str，在构建时按Gallery原顺序追加位置。每组索引与原`enumerate + p==pid`逐项相同；测试也检查所有成员与索引顺序。特殊非正式类型走原相等扫描fallback，保留NaN PID产生空组等行为。
3. 聚合继续执行`scores[:, list(indices)].max(dim=1).values`，最后`torch.stack(..., dim=1)`。没有index_select/scatter/向量化归约，没有改变Tensor值、dtype、维度或归约/并列成员顺序。
4. CPU FP32/FP64、非连续/切片/expand、零行、负数、极端有限值、signed zero、NaN payload和Inf均与原位模式一致。空Gallery、空组、宽度/维度错误和不可hash PID保留原异常类型与消息；**没有新增finite检查或“修正”原边界行为**。非有限传播Exact不表示这种输入对攻击是合法的。
5. 在原Classifier的合成编码/评分替身中对比缓存键、调用序列、8192容量/淘汰、输出与RNG；固定原Ensemble代码的softmax/margin与稳定候选排序一致。3个固定Seed的官方attack合成夹具最终文本、全部捕获候选、oracle调用序列和Python/NumPy/Torch RNG一致。
6. **固定Manifest是前提**。测试包含“长度相同但PID成员改变，错误复用旧Plan会出错”的反例。未来必须每Classifier/固定bank绑定Plan，禁止按G或P简单跨数据集/图库复用；不改变原Caption字符串缓存键。

## B. CPU测试与实测性能

[CPU_TESTS.json](CPU_TESTS.json)包含逐项结果和原始日志/JUnit的路径、SHA；[BENCHMARK.json](BENCHMARK.json)保留各次计时及环境，[BENCHMARK.csv](BENCHMARK.csv)便于比较。

| 测试范围 | 通过 |
|---|---:|
| 原test_cpgc_sep.py | 28 |
| 原test_cpgc_sep_throughput.py | 19 |
| 新test_sep_cpu_optimization_review.py | 97 |
| 合计 | **144** |
| 新测试中包含bitwise断言 | **80/80，100%** |
| 原异常类型/消息对照 | 5/5 |
| 原Ensemble固定下游softmax/margin/排名场景 | 17/17 |
| 官方原候选流程合成端到端场景 | 3/3（Seed 0/1/7） |

含多PID、不均匀成员、单PID/唯一PID、多个正例、并列、邻接FP32值、命名空间、异常输入、原缓存命中/淘汰、RNG及Trace恢复等。还在CPU双线程、3074 Gallery/1000 PID/8行FP32下验证原/新结果exact，与原Source的CPU线程数对照。**所有被比较的Exact用例无字节差异，未改变候选排名或最终文本。** 运行官方候选函数的原有/新增CPU夹具均使用人工词表/人工分数，不是44条真实攻击的重放。原throughput回归另只读对照已有小型Smoke分数及排名，没有编码、攻击或读取大型Full Tensor。

微基准：Intel Core i9-14900K，Torch2.5.1+cu124、Python3.10.21；单线程、进程仅绑CPU0，固定Seed20261103、FP32。预热后交替计时，7次（ICFG 5次，trace 3次）取中位数。B只是合成矩阵行数，不改生产Batch。

| 固定合成形状 G/P/B | 原实现 ms | 预计算后 ms | 含每次构建 ms | 单独聚合实测比值 |
|---|---:|---:|---:|---:|
| RSTP形状1000/200/1 | 29.764 | 1.693 | 2.236 | 17.58× |
| CUHK形状3074/1000/1 | 430.783 | 8.229 | 10.006 | 52.35× |
| CUHK多行3074/1000/8 | 430.368 | 8.597 | 10.328 | 50.06× |
| ICFG形状19848/1000/1 | 2855.459 | 9.712 | 21.475 | 294.02× |
| CUHK形状+原式global trace模式 | 1340.312 | 8.436 | 13.751 | 158.88× |

CUHK构建Plan中位1.702ms、ICFG11.543ms；热复用比值不隐藏该一次性成本，冷构建数据同时列出。全部基准返回bitwise exact。trace模式使用原式全局call回调形态、作用于合成调用栈，不是运行SEP Source。

**不能据此预测104分钟Query变成多少秒或Full的加速倍数。** 真实候选编码、RaSa融合、cache命中率、I/O与GPU同步均未在本轮执行。CPU已有相同分数矩阵是Exact结论的条件；没有验证新模型批次或GPU算子。

另有两个风险实测：

- 在4×512分数、128 PID的合成观察器中，深层暂停trace前后捕获候选相同；中位66.670→20.244ms，回调数133135→17。这个特定夹具的收益不能把生产Trace修改升级为A类。
- 固定8×512 query与3074×512近并列Gallery，单行GEMV与批量GEMM均为FP32，却有21990个分数值不同，最大差3.9100647e-5，8行排名均有变化。**这是CPU合成反例，不是实际IRRA/RaSa编码结果**；它证明“批量计算数学一样”不足以保证bitwise/排名一样。

官方零向量循环的CPU边界测试保留原函数体，只注入人工重要性分数，并由测试tracer在第16轮主动中断。零向量每轮仍全零，印证无退出的条件性缺陷；常数非零也持续零跨度，后续还可能溢出。**未在历史第45条或任何真实Query上复现，不将其认定为原慢速根因。**

### 优化分级

| 方案 | 类别 | 当前结论/限制 |
|---|---|---|
| 固定Manifest预计算PID组；Python重复扫描换原顺序索引访问 | **A** | 本轮80项位模式测试及5组基准通过；生产尚未替换 |
| 复用固定gallery_pids/order等不可变元数据 | **A限定** | 仅本Plan内已验证的固定身份；不复用带游标/候选状态的NativeScorer对象 |
| 既有同Query分数/Token缓存复用 | **A现有能力** | 原47项回归含缓存验证通过；新增跨Query/跨模型缓存为B，不能盲目沿用字符串键 |
| 候选文本批量编码、改变fusion batch、移除重复TopK | **B** | 需真实模型分数/Top128/pair顺序/排名exact验证；没有GPU证据 |
| 减少CPU/GPU拷贝、移动PID聚合到GPU、异步流 | **B** | 纯数据复制与算子迁移必须区分；设备/同步/归约内核变化未验证 |
| 缩小Trace范围 | **B** | 合成结果/异常恢复一致；真实递归/线程/外部调试钩子需单独核验 |
| 降低T或patience、改变代理数/DPP/Top128/预算、改变FP32精度 | **C** | 改攻击/数值/预算协议，不是普通工程提速 |
| 改Batch Size | **B，且需新执行协议** | 不能因数学等价视为无损；冻结参数本轮未改 |
| 给零向量循环加上限、错误/NOOP/均匀回退 | **C** | 改原算法边界行为；本轮不修 |

## C. 三种SEP协议必须分开

### 作者论文协议

[论文§4.1.2/§4.1.5](https://arxiv.org/html/2605.24958v1#S4.SS1)的模型池为BERT、ALBERT、RoBERTa、LSTM、TextCNN；攻击BERT时用其余四个作为代理，Victim排除。实验N=4、E=4，每权重T=10。原分类数据为MR/AG/Yelp/IMDB，不是TBPS三数据集。[固定官方main](https://github.com/ZevineXu/SEP-Attack/blob/3a6d848a48097aee79be0299da8959e78db543c7/SEP_Attack.py#L549-L560)也明确从代理池pop目标模型。

需要区分论文的“目标不指导代理候选优化”和实现的评测查询：原attack仍有Clean正确性门禁及末端成功统计oracle。不能据此声称整段评测从未调用Victim；关键是独立Victim不进入代理优化/迁移评分。若要求严格无Target反馈的冻结生成，未来应把Target Clean正确性筛选放在统一后验评测中，而不是让它控制生成和RNG。

### 当前TBPS适配协议（已停止，未修改）

- `sep_source.py:81–100`对每个dataset同时加载**该域IRRA和RaSa**，它们都进入Ensemble。`victim`为IRRA或RaSa时，对应同一权重仍是其中一个代理，不是独立未见Target。
- `classifier(...manifest)`按**该Test Gallery真实PID**聚合，`label=pid_order.index(q.pid)`使用该Query的真实PID；Victim还参与Clean错误早退。即使固定gamma分支在末端成功查询前捕获，也不能消除前述Target参与。
- 从访问权限看是**Victim权重可访问的白盒设置**，优化形式是score-based、无梯度，不应写成“梯度白盒攻击”。若外部只给分数API才是另一种score-based黑盒/灰盒限制；当前本地完整模型加载并非该限制。相对于当前IRRA/RaSa任一Victim，都不能称独立Target跨模型黑盒迁移。
- N/E为2/2，保留作者`keep=len(models)`规则，但不等于论文4/4。它是官方候选算法的分类→TBPS检索适配，原输入/置信度构造和任务协议均有明确变化。
- 当前CUHK/RSTP/ICFG三域分别用本域代理、图库和标签重新优化，是**三个同域Test-time攻击**，并不是“CUHK训练一次冻结攻击器→其他域”。未训练模型不等于没有Test-time目标反馈。只有CPU/缓存审计PASS不能认证威胁模型公平。

### 与Idea IRRA Source→RaSa/APTM的公平比较建议（未实施）

1. 若对齐本轮指定的**IRRA-only Source**，SEP生成不能读取RaSa/APTM权重、分数或预测，两个Target都只在攻击文本冻结后独立评测。同一份源文本必须复用于所有Target，不按Target重新生成或挑选。
2. 仅从当前集合删RaSa会把代理/DPP从2/2改为1/1，属于C类，且不再等于论文4代理SEP；必须另命名并单独登记。若用多个IRRA Source变体组成4代理，可接近论文N/E，但源模型数量和训练信息又多于Idea单Source，也不能当作访问预算相同。**本轮未擅自选择新代理协议。**
3. **标签/配对权限还需先解决。** 当前SEP使用真实Test PID和Gallery PID分组，Idea冻结H(T)不接收这些信息。即使排除RaSa/APTM，仍不能自动宣称公平。若保留GT标签须单列强信息设置；若改成仅Source预测的伪标签/其他caption-only目标，会改变适配目标，属于另行授权的C类方法选择。本轮只列OPEN，不改算法。
4. 要覆盖Idea的RaSa/APTM跨模型比较，建议把**APTM补为独立Target**，只增加评测，不把APTM放进代理集合。当前SEP执行器仅接IRRA/RaSa，APTM需独立官方Adapter与完整Gallery评测接线验证；IRRA特征绝不可冒充RaSa/APTM特征。
5. 当前SEP是Text-only，应优先与H-v1 Text-only比较。G-v2 Image-only、G+H Joint需明确模态与预算差异；不能仅因同一mAP表就叫等信息/等预算。字符替换与SEP同义词替换也不是天然同一文本预算，需报告两者实际编辑/Token/语义约束与生成开销。

| 协议轴 | 跨模型对齐 | CUHK→RSTP/ICFG跨数据集对齐 |
|---|---|---|
| 攻击/生成器训练域 | 明确记录Idea的CUHK训练；SEP是否有训练/Source权重训练域另记 | CUHK训练的G/H权重冻结；SEP Source权重也应锁定CUHK来源 |
| 生成时可访问模型 | 仅预注册Source，排除RaSa/APTM | 不按目标域切换到该域IRRA/RaSa代理权重 |
| 生成输入域 | 同一预注册评测域Caption/公开输入 | 目标域原Caption重新生成，不复用CUHK既有文本 |
| Test PID/配对与Target反馈 | 当前SEP有，公平新协议需先统一/解决 | 禁用目标Test标签/配对/模型反馈；现有SEP接口不能直接满足 |
| Target评测 | RaSa/APTM各自原生权重和特征 | 目标域原生Target Checkpoint仅用于评测 |
| 输出绑定 | 一个源adv_text，多个独立Target | 每目标数据集一份源adv_text，保持官方Query/Gallery顺序 |

后续Manifest应分别登记：attack训练域、Source模型及权重训练域/SHA、生成输入域、生成时Gallery/标签权限、Target模型及其权重域/SHA、评测域/split/模态/预算、是否Target反馈。不要用单个dataset或victim字段混合这些轴。

## D. 最小后续方案与本轮交付

**仅A类工程落地候选**：未来如授权，最小修改`sep_bridge.py`（受固定身份约束的分组Plan）和`sep_source.py`（Classifier构建时一次预计算并复用）。保留原pid_logits兼容入口、缓存键/容量/清空时机、候选顺序、RNG、CPU max路径和原观察器。官方SEP_Attack.py/SEP_utils.py、RaSa/IRRA Adapter、DPP、预算和配置均不改。

即便该局部CPU变换已exact，生产源码SHA仍会变化。旧44条和第45条Claim不能在原Attempt无条件重播；需要新版本/明确兼容来源方案，并在另行授权后做限定真实数值/候选链验证。不能拿本轮微基准替代新的Smoke审计。**先解决公平协议，再决定正式Full；不自动执行。**

B类每项分别要求原/新native分数逐元素、Top128成员及顺序、softmax/margin、候选选择、RNG和异常恢复对照；C类先写新研究协议。Trace与批量数值风险不与PID A类混成一个补丁。

重现本轮CPU工作（独立Worktree，需人工执行，不在历史执行目录）：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -p no:cacheprovider -q tests/framework/test_sep_cpu_optimization_review.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -B operations/sep_cpu_review_benchmark.py --output /home/lzf/ldx/tmp/sep_cpu_optimization_review_20261103/benchmark_manual_new.json
```

原型仅在tests，微基准仅CPU；没有修改生产src/configs或官方定义。运行证据目录`/home/lzf/ldx/tmp/sep_cpu_optimization_review_20261103`，保护核验检查75个小型完成/Partial回执及必要源码/Checkpoint stat；原有回归另只读核对小型Smoke缓存分数，未再次扫描大型Full Tensor缓存。停止状态、原Claim和旧源码保持，详见[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)。
