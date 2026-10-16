# H-v1 + G-v2 联合攻击消融评测 v1

本模块独立于已冻结的Phase4训练、Phase5正式Full评测和Validation Selection。只开发和运行CPU测试；所有真实文本GPU编码、Full消融执行及恢复均由用户手动启动。默认入口为CPU预检，正式运行必须显式传入`--user-run-ablations`。不训练任何模型，不加载G作推理，不调用IRRA图像前向。

## 冻结实验协议

源码基准为`d501f115f8877c9a3a1880e7500eb3e737738cbf`。`config.json`锁定70项历史源码/配置SHA，以及随机种子、搜索节点上限和统计定义。实际运行时另绑定本模块6项生产文件SHA、原Phase5 Full协议、Selection、各模型权重、图像身份、四类特征和全部原回执SHA。校验失败直接停止，不修改旧证据。后续添加无关文件不改变锁定集合。

固定3074 Gallery、6156 Query，全体Query保留。PID及正例关系仅用于官方指标计算，随机决策及H推理不接收PID、图像、匹配分数或奖励。每个Caption的随机数由`SHA256(seed|kind|Caption)`独立生成，不使用Query身份；相同Caption同种子同策略产生相同结果，不消费全局Python RNG。

| 消融组 | 属性及字符决策 | 编辑数量 |
| --- | --- | --- |
| Learned H-v1 | 直接读取已审计Phase5 Full原结果、动作记录和原特征 | 已有真实编辑数，不重新运行H |
| Random Attribute + Random Unicode | 在Caption-only合法属性中随机选择，随机合法位置及Unicode字符 | 每Query目标为Learned实际编辑数m，沿用Learned预算K |
| H-selected Attribute + Random Unicode | 固定Learned**实际编辑的属性ID及顺序**，随机合法位置及Unicode字符 | 同样目标m，保留Learned完整selected IDs，包括选中后NOOP的属性 |
| Untrained H | 相同网络结构，seed42从头初始化，原贪心策略及Natural Dynamic-K/STOP/NOOP | 自身自然输出，不强制匹配Learned编辑数 |

两类随机组固定seed=`42,43,44,45,46`，共10次；Untrained仅seed42一次。全部报告，禁止根据Test选seed、调搜索上限或选择H。Learned和Untrained只有单次结果，标准差记为null/空单元格。多seed使用样本标准差（ddof=1），不能视为显著性检验。

随机组以随机打乱的属性、位置和字符候选进行深度优先搜索，采用首个可行完整序列；不是全部合法编辑序列上的均匀抽样。每Query最多256次候选边扩展，受共享词、Span、Tokenizer变化和截断约束。每个属性最多编辑一个字符，共享原词最多一次，全局不超过原K。m=0时保持NOOP；不得为了提高攻击效果额外编辑。若节点耗尽仍未完成m次，保留合法部分序列，记录target/actual、失败原因`search_node_limit_no_complete_match`和比例，不使用Learned字符作为回退。该失败仅表示有限搜索未找到，不表示不存在合法解。若真正穷尽仍失败，则与原Learned合法见证冲突，直接报错。

这一定义让H-selected对照保留“实际生效的属性支持集”，避免把Learned的STOP后属性重新编辑。它检验固定支持集下位置/字符策略的贡献，不单独分离位置Head与Replacement Head。Untrained编辑数量可不同，解读时须同时看编辑分布与NOOP，不能把差异全部归因于SCST。

## 复用与输出

原输入只读：`/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/`。
复用Clean/Adv Gallery和原文文本特征，不复制图库缓存，不重复计算原Clean、Image-only或Learned指标。读取原Learned对抗文本用于动作身份核验；Learned结果直接使用原Full完整精度数据，不采用手写四舍五入值。

新输出仅允许：`/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full/`。禁止将输出改到旧Phase4/Phase5目录。

- `protocol.json`：配置、源码、Selection/模型/数据/输入特征SHA、固定种子、运行环境。
- `variants/<kind>_seed<seed>/features/adv_text/`：紧凑CPU文本特征、逐Query编辑/Unicode码位/Span、Tokenizer审计、原子回执。
- 各variant的`summary.json`、`execution.json`、`completed.json`：四组指标、DR_mAP、Joint差值、行为统计、调用计数及SHA。
- 根目录`summary.json`、`metrics.csv`、`aggregate.csv`：全部12组策略（含Learned）各自四条件，以及多seed均值/标准差。
- `attempts/`：本次执行/失败/中断证据；`abandoned/`：显式恢复时保留的未提交批次。历史证据不删除。

四条件为Clean、Image-only、Text-only、Joint；指标为官方IRRA相同排名规则下的mAP、R@1/5/10、mINP。DR_mAP = (mAP_Clean − mAP_group) / mAP_Clean × 100%。`Joint_minus_ImageOnly_mAP_pp`和`Joint_minus_LearnedJoint_mAP_pp`是百分点差，负值表示更低mAP。重点同时检查每Query目标/实际编辑数、匹配率、K分布、NOOP、属性/字符统计；不得筛掉失败或NOOP Query。

## 用户手动命令

下列命令按实际CLI编写。固定线程、text-batch和运行环境会进入协议，恢复时应保持一致。以下CPU预检也会读取并核验原Full特征/SHA及动作，可能有较多磁盘I/O；不执行模型前向或检索指标重算。Codex本轮没有执行真实Full预检。

### CPU预检

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action preflight --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full
```

成功标志：`ABLATION_CPU_PREFLIGHT PASS`。只生成内存中的协议，正式运行开始时才不可变发布新输出协议。

### 正式Full消融，仅用户运行

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action run --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full --user-run-ablations
```

成功进度标志：`ABLATION_TEXT_COMPLETE <trial> n / 6156`、`ABLATION_METRIC_COMPLETE`、`ABLATION_TRIAL_COMPLETE`。全体完成：`ABLATION_FULL_COMPLETE`。不得把完成单个trial误认为所有seed已完成。

如需保存终端日志，可由用户自行加`2>&1 | tee <独立日志路径>`；使用管道时先`set -o pipefail`以保留Python失败退出码。程序不会自动后台启动或自动重试。

### 中断与恢复，仅用户运行

前台Ctrl+C或向该Python PID发送SIGTERM，程序在原子批次/指标组边界停止，记录`ABLATION_SAFE_STOP`及中断文件。正在进行的CUDA调用或单个指标计算不会立刻中止。不要删除原协议、回执或已完成特征。

正常中断后，重新执行上面的**同一正式命令**：已提交且SHA有效的文本批次/已完成trial只读复用。随机数按Caption固定，Untrained无训练状态，恢复不依赖历史RNG。若在trial结果写出后、完成回执前中断，会核对已保存结果digest及冻结策略，保留原execution时间和计数后补齐回执。

若SIGKILL/OOM留下未提交批次，先保留并检查本次failure/pending证据；普通命令会拒绝继续。确认需恢复时，用户显式执行：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action run --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full --user-run-ablations --recover-pending
```

此参数仅允许把相同协议的未提交批次完整移动到新输出的abandoned，保留SHA，然后重算该未提交文本批次；不覆盖或重算有效已提交批次。未知协议、损坏已提交文件均拒绝，不能用该参数绕过来源校验。异常退出的尝试可能已实际执行文本编码但没有成功发布；硬件调用总量和有效提交样本数必须区分，不宣称失败尝试从未计算。

### 完成后独立CPU只读审计，仅用户运行

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action audit --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full
```

成功标志：`ABLATION_CPU_AUDIT PASS`。审计验证模型/来源/原输入SHA、全部新回执、动作约束、NOOP特征、CSV和多seed统计，并从已保存特征在CPU重算新trial的Text-only/Joint指标。没有模型前向或GPU推理，不重算Learned原结果。审计不修改缓存和完成证据。

用户可自行查看终端进度，或`tail -n 40 <自己的日志路径>`；退出后用`pgrep -af 'joint_ablation_v1.evaluate'`及`nvidia-smi`确认自己的进程已退出。Codex不会持续监控。

## 计算与存储预算

有效完整运行新增G前向=0，IRRA图像前向=0，训练/优化/JVP=0。11个新文本策略最多编码67716条（11×6156）；NOOP直接复用原文特征，实际通常更少。batch16时最多4235次批次文本编码。Learned不重新编码，Untrained最多6156个CPU H样本前向，随机组不调用H。不跨seed按相同字符串去重，以保持每trial独立回执；恢复会复用有效已有批次。

新float32[6156,512]文本特征每trial约12MiB，11次约132MiB；含动作/Span/Tokenizer metadata及回执建议预留2–4GiB，失败归档另计。估计CPU内存6–10GiB、GPU约1–3GiB、总耗时约30–120分钟；这些**没有真实GPU实测**，CPU合法候选搜索、22次完整排名、重复SHA核验和存储速度可能显著增加耗时。显存沿用官方IRRA文本路径，仍以用户实际运行记录为准。

## CPU验证

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest joint_ablation_v1.test_ablation phase5_joint.test_phase5 test_policy test_scst test_tokenizer_real -v
```

覆盖随机数量/K/NOOP、H支持集、重复词、Span/Unicode/BPE掩码、有限搜索失败、Caption-only隔离、初始H seed42 SHA、官方指标/seed统计、紧凑Storage、SHA篡改拒绝、pending及完成回执中断恢复、CPU完成态重入。测试仅用合成特征、CPU Mock编码器和少量合成Caption，不产生正式消融结果。
