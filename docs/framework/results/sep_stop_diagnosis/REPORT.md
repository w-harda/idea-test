# SEP Official Full停止与性能异常诊断（2026-11-03）

**STOPPED_BY_USER。** 2026-11-03 14:43:39（Asia/Shanghai）已停止CUHK-PEDES × IRRA的SEP Full Source。计算子进程3041291、Supervisor2960820及其后代均退出，SEP原3008 MiB显存释放。未重启SEP、重跑Smoke/Full、调整算法/Batch/精度，未停止Ollama或其他实验。

## 停止方式与真实现场

停止前通过完整cmdline、cwd、PPID、PGID/SID、/proc启动tick复核身份：Supervisor启动tick2361740670、SEP子进程2361911378，父子关系3041291→2960820，组/会话2960820；无其他后代。源码没有专用停止文件、信号处理或迭代级Checkpoint。原unit机制仅在每条Query完全结束后发布文本和RNG；Claim在计算开始前保留。

通过Linux pidfd仅向确认身份的3041291发送SIGTERM，未向进程组发信号；Supervisor自行收到returncode=-15，写原生FAIL和exit=1后退出，无需终止Supervisor。Python运行时未暴露os.pidfd_open，准备阶段未发信号，随后使用同内核pidfd syscall完成。原控制目录的FAIL是用户中断导致的调度记录，不是已发现攻击数值错误。保留其原文，不改成PASS，也不伪造停止单元完成回执。

原输出：`/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-no-gpu-floor-v4`；原控制：`/home/lzf/ldx/tmp/cpgc_sep_no_gpu_floor_v4_20261101`。停止/诊断原始证据全部另存`/home/lzf/ldx/tmp/sep_stop_diagnosis_20261103`，含停止前后进程/GPU、文件stat、信号回执、原日志尾、内存/I/O、耗时表、只读完整性核验及SHA索引。

## 完成与中断的边界

日志每4条Query打印一次，因此最后的41/6156不是精确已发布计数。停止时真实发布**44/6156**，索引0–43，均有独立Manifest、SHA、文本和RNG；第45条（index44，`cuhk:37154:0`）仅有`text-000044.Claim.json`，未发布文本或完成回执。该单元中断前约3066.28秒；不恢复、不重播、不丢弃Claim。两个代理各3074张Gallery特征已完整构造，分别385个8张块。

当前Full Source整体未完成，无Source completed，无本组Target Full评分或Full独立CPU审计；其他5个SEP Full也未运行。**SEP正式完成组数0/6**。全链原completed_Full=6只表示此前的**C-PGC六组**，不能算作SEP完成。没有发布SEP正式Full指标。

历史C-PGC六组及SEP六组Smoke的本次完整性复核结论与核验范围见下方“完整性回执”；未重新计算旧模型/攻击/指标。

## 性能异常：证据与判断

| 现场观测 | 数值 | 说明 |
|---|---:|---|
| Full Source墙钟时间 | 61.76小时 | 包含已发布44条及第45条中断段 |
| 已完成实际攻击Query | 30 | 不是每条都执行攻击 |
| 实际攻击耗时中位数/均值/最大 | 103.84 / 121.69 / 426.64分钟 | 根据Claim与原子发布文件mtime计算 |
| Clean错误提前返回NOOP | 14条，中位2.44秒 | 原算法初始门禁跳过，不是加速攻击 |
| Source事务开始至首Query Claim | 189.72秒 | 包括模型/两代理Gallery准备，未包含事务前CLI预检 |
| SEP进程累计CPU时间 | 65.70小时 | /proc utime+stime；约占墙钟106.38% |
| GPU遥测 | 14802点，中位0%，70.79%采样为0 | 设备全局数据，不能直接归因于单一进程 |
| 低总显存占用(<8GiB)时 | 7345点，平均GPU利用率2.13% | 即使没有大量显存占用仍长期低利用率 |
| 停止前SEP RSS / Swap | 8.12GiB / 0 | 主机MemAvailable约107.54GiB，memory PSI近期0 |
| 进程实际块设备读取 | 约13.79MiB | rchar约41.92GiB，主要命中页缓存；无I/O阻塞主因证据 |

### 已确认的实现瓶颈

1. **PID聚合每个候选重新做P×G Python扫描。** `sep_bridge.py:12–18`对1000个PID分别遍历全部3074 Gallery，逐个比较PersonID并做小Tensor max。每个模型每个未缓存候选约3074000次Python比较；两个代理约6148000次。PID与Gallery的分组其实固定，当前却在每次候选评分重建。`sep_source.py:50`在每次cache miss调用它。这是明确的重复计算和CPU性能实现问题，不是官方分类器原本必需的运算。
2. **候选批处理在桥接层退化成逐文本执行。** 官方一次传入整批遮词/同义词候选，`sep_source.py:40–53`却逐个text调用encode_texts([text])、全Gallery全局评分与PID聚合。RaSa每个未缓存候选还完整做Top128融合，固定fusion_batch16意味着8次融合BERT + 8次ITM head调用。IRRA每候选一次文本编码；RaSa每候选一次文本编码及上述融合。Gallery图像没有反复前向，但候选检索和融合很贵。
3. **频繁CPU/GPU往返与小操作。** IRRA全局分数转CPU；RaSa TopK索引转CPU并从CPU Token缓存收集、传GPU，融合结果又转CPU；CPU PID聚合结果再传GPU进入原Ensemble softmax/margin。原optimize用torch.tensor(wi_score)、argmin和.cpu等读取设备标量，也引入同步。不能把这些路径称为全模型CPU回退；模型明确在cuda:0，只是中间聚合/数据搬运在CPU。
4. **Python trace观测放大调用成本。** `sep_source.py:115–126`在整个attack期间启用sys.settrace。虽然只为官方attack帧返回逐行tracer，其他Python函数进入时仍触发全局trace回调；大量PersonID相等比较因此额外经过Python回调。未保存Profiler，不能准确拆分各项耗时或承诺某个优化倍数。

上述结构与CPU约106%、GPU长期空闲、NOOP秒级而攻击小时级相互一致。**主诊断为检索适配桥接的CPU重复聚合、候选串行和同步开销；不是显存占用高就证明GPU计算繁忙。** 不将尚未测量的每项占比表述为定量因果。

### 未发现的重复工作，以及尚不能认定的原因

- 两个代理仅在Query循环外各加载一次；日志仅一次IRRA构造记录。Gallery每代理完整编码一次，存储块覆盖恰好3074张，未见重复重建图库或重新训练。
- v4的sep_source/sep_bridge/cache/victims/official五文件与v3逐字节相同，SEP参数和evaluation配置也相同。无证据显示取消显存门槛导致迭代规模扩大或新增数值性能回退；瓶颈是继承的接线实现。固定DPP最终2组权重，不是把100个随机候选权重都执行攻击。
- 停止前GPU仅列出SEP，停止后无计算进程；未停止Ollama。历史显存总占用≥20GiB时与<8GiB时都存在长期GPU空闲。共享GPU可能影响部分区间，但遥测是全局统计，没有逐进程kernel时间，不能归因或量化Ollama阻塞；它不能解释所有慢速时段。
- 无CUDA OOM、异常反复加载、训练、主机内存不足或SEP Swap证据。进程停止前为R、wchan=0；长期实际块设备读取很小。CPU Token缓存和SHA校验会有成本，未见磁盘是主要瓶颈。

## 优化步骤、Forward与Backward的实际口径

原作者`SEP_Attack.py:249–286`：每条进入攻击的Query生成100组随机权重，经DPP保留len(surrogates)=2组；每组T=10，patience=5可提前停止，故最多20次optimize，后面还有10档候选迁移排序/成功统计。每次optimize重新做词重要性、最多50同义词/所选位置的搜索，以及回退较不重要编辑。单条Query会评估许多候选，绝不能将20次optimize等同20次神经Forward。

令W为可扰动词数，M为该轮选词预算，每轮至少有W个遮词候选、最多约50M个同义词候选，再加编辑回退与最终候选评分；原始Query还先调用Victim Clean门禁。两个分类包装各有8192条候选分数缓存，在同Query中复用相同文本，Query结束清空。真实神经Forward只发生在cache miss，并非所有Ensemble调用都重新编码。

**Backward、Optimizer step为0**：原Ensemble在no_grad下调用冻结模型，SEP此处是离散文本搜索，不是梯度优化。Partial未写出最终源报告；逐Query的实际optimize次数、候选cache miss、IRRA/RaSa累计Forward数没有持久化，无法只凭44条文本/成功统计准确重建。本次没有为补这些计数重跑攻击或模型；以上是代码执行结构和上界，不伪造实际计数。

## 官方代码与论文协议核对

固定作者仓库Commit `3a6d848a48097aee79be0299da8959e78db543c7`，工作树干净，原定义AST直接加载、文件SHA保持；无修改算法。对照[原论文§4.1.5](https://arxiv.org/html/2605.24958v1#S4.SS1.SSS5)与[固定官方实现](https://github.com/ZevineXu/SEP-Attack/blob/3a6d848a48097aee79be0299da8959e78db543c7/SEP_Attack.py)：

| 项目 | 当前实际 | 核对结论 |
|---|---|---|
| T / 初始patience | 10 / 5 | 原代码一致；T与论文一致 |
| DPP随机池 | 100 | SEP_utils.py:186–233一致 |
| 最终代理数 / DPP保留数 | 2 / 2 | 原keep=len(models)规则；论文实验为4 / 4 |
| η / max_change_rate | .2 | 原公式有至少3词的边界，不是所有短文本严格20% |
| ε / 最终回退编辑数 | .15 | 原floor并至少1词；不是图像扰动预算 |
| γ | .02 | 原10档搜索中ms_idx=1、floor且至少1，匹配当前显式选取分支 |
| 同义词 / top_k_words | 50 / 1000000 | 官方词表/邻居与默认值一致 |
| k_margin | .9 | Python默认一致；README命令示例为1 |
| Victim与代理隔离 | 当前Victim包含于IRRA+RaSa集合 | 原分类main会pop目标模型，不能称为论文独立Victim纯迁移同等复现 |
| 数据/评分 | TBPS完整图库→PID logits | 明确的分类→检索适配，非作者原四分类数据集配置 |

因此只能称“固定作者候选算法的TBPS检索适配”。官方函数未改不代表完整复现论文设置或公平的独立Victim黑盒威胁模型；这一口径差异需要单独审查，不能用Smoke/缓存审计PASS替代。

还发现一个**原上游条件性缺陷**：`SEP_Attack.py:116–117`在重要性向量max−min<1时循环乘10。若向量全为0，永远无退出；其他常数向量可能溢出导致非法分布。现场没有保存Python栈或该向量，本次也未执行官方optimize去触发它；**不能把这次慢速确诊为死循环**。44条持续发布的时间线也不支持“从开始就卡死”。

## 完整性回执

PASS。此前C-PGC六组Full及各自原始CPU审计回执均完整；CUHK/RSTP/ICFG × IRRA/RaSa的SEP六组Smoke各自Source、Target、Audit共18个完成范围的全部已发布payload/Manifest/SHA深核验通过。发现的全部SEP历史目录为18个已完成Smoke范围加当前1个Full Partial，无其他已完成SEP Full。

核验了冻结全量711130个文件的size/mtime/inode，覆盖六组C-PGC Target与Audit的425310个文件；逐组完成Manifest与先前verified_reuse SHA一致，原CPU audit结果SHA、scope、成员数量、CSV与summary指标绑定一致。**这属于全量stat + 完成/审计SHA和来源核验，不声称重新SHA全部数百GiB C-PGC Tensor缓存，更未重算旧指标。**

本次只读SHA核验共89945个不同文件、60675537619字节，约56.51GiB；SEP六组Smoke和当前385×2个Gallery块、44条文本/RNG及协议的已发布payload全部深SHA通过。44条Query与原Caption/ID顺序、原config/Claim的raw_sha逐条exact；第45条Claim身份exact且无输出目录，无pending、不伪造source completed。

停止前已有原输出文件0变更、0缺失；原控制目录仅Supervisor自然更新的日志/status/phase和新增退出/FAIL记录。G60、H60、H7 Checkpoint、六个Victim Checkpoint、三个annotation、原producer/作者源码SHA保持；G+H正式报告既有Manifest校验通过。没有向任何其他实验发信号或修改它们的资产。

完整逐组回执、原SHA与静态保护范围见[SOURCE_EVIDENCE.json](SOURCE_EVIDENCE.json)；逐Query原Claim→原子发布耗时见[QUERY_TIMING.csv](QUERY_TIMING.csv)。原始完整停止证据位于`/home/lzf/ldx/tmp/sep_stop_diagnosis_20261103`，索引SHA `c6ebc5dfb1c80c41989dc51f83291cd6d9cc2054bcd3db32c6c6c5740ef88234`。首次只读全payload扫描范围过宽，已主动中止本任务自己的CPU校验进程后，使用上述明确核验范围完成；两份校验日志均保留，不属于SEP运行/重试。


## 后续建议（未执行）

1. 先冻结本Partial为用户停止现场，不清Claim、不自动恢复或在原Attempt重播第45条。另行授权后才能决定新Attempt/明确恢复兼容协议。
2. 首先只做CPU接线优化方案审查：为固定Manifest预计算PID→索引；复用固定分组、减少Python扫描/trace和CPU/GPU往返。保持原每PID max的成员与顺序，先做逐元素exact测试，再谈性能。改变浮点归约顺序或scatter实现可能改变结果，不能默认等价。
3. 保留同义词顺序、RNG与原候选决策；若未来合批编码、改变fusion_batch/精度，已有历史实测提示可能改变排名，必须另立协议并验证，不能作为无损提速直接上线。减少T或改代理数量/TopK均是协议变化。
4. 对原零向量循环加拒绝/超时/回退属于算法边界修改，须另行授权并单独命名，不能悄悄修作者函数。可先增加只读阶段计数/采样栈和候选cache命中统计；实际Profiling/GPU运行需新授权。
5. 对论文4代理、Victim排除与当前TBPS 2代理/含Victim的差异先定实验口径，再设计新的复现。当前0个SEP Full不得填正式对比表。

本次仅停止、读取、文件/SHA核验与写诊断文档；没有修改或运行攻击算法，没有创建新GPU实验。
