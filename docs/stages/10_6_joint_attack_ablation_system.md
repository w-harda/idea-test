# 联合攻击消融 v1：代码、CPU验证与冻结执行协议

日期：2026-10-16。源码基准：`d501f115f8877c9a3a1880e7500eb3e737738cbf`；分支`codex/gv2-per-attribute-cache`。本报告随实现同一Commit提交，发布SHA以Git历史及最终交付为准。

## 阶段结论

已新增隔离的`prototypes/attribute_text_policy_h_v1_20261015/joint_ablation_v1/`，实现三种新消融策略、冻结来源核验、文本特征原子缓存/恢复、四条件官方指标及多seed统计。177项CPU测试通过（新增52 + Phase5回归57 + H核心68），耗时81.897秒。

**没有运行任何真实GPU实验、Full消融、原Full预检/指标重算、训练、缓存生成或持续监控。** 本轮没有新攻击效果结果；用户提供的已完成Joint Full结果仅为研究背景，程序使用服务器原Full完整精度证据，不采用报告中的四舍五入数值。真实规模的协议预检、运行、恢复、审计留待用户审查后手动执行。

## 新文件

全部实现新增，不改Phase4/Phase5源码、配置、Selection、G/H/IRRA、Checkpoint或历史产物：

| 文件 | 职责 |
| --- | --- |
| joint_ablation_v1/__init__.py | 独立包 |
| joint_ablation_v1/config.json | 70项既有来源SHA、seed及搜索/统计定义 |
| joint_ablation_v1/common.py | 原Full特征/动作/证据核验、固定协议和隔离输出 |
| joint_ablation_v1/strategies.py | 两类随机策略、初始H、数量匹配与动作审计 |
| joint_ablation_v1/backend.py | 仅IRRA文本编码、NOOP特征复用、冻结及资源统计 |
| joint_ablation_v1/storage.py | 复用Phase5原子Store，仅发布adv_text，禁止图库复制 |
| joint_ablation_v1/evaluate.py | 预检/用户运行/审计、11trial及Learned参考、CSV和恢复 |
| joint_ablation_v1/test_ablation.py | 52项新增CPU测试 |
| joint_ablation_v1/README.md | 完整冻结协议、命令、恢复和成本说明 |
| docs/stages/10_6_joint_attack_ablation_system.md | 本中文报告 |

## 对照的精确定义

- **Learned H-v1**：直接复用已完成且用户独立审计通过的Phase5 Full四组指标、逐Query动作与原特征，不重新选权重、不调用H。
- **Random Attribute + Random Unicode**：仅由Caption合法Canonical/Span与Unicode/BPE约束产生随机编辑，目标为该Query的Learned实际编辑数m，保留原K上限。
- **H-selected Attribute + Random Unicode**：固定Learned实际编辑属性ID及顺序，随机位置与字符；完整selected IDs（含选中后NOOP）保留在记录。这样不把STOP后的未编辑属性重新纳入编辑。
- **Untrained H**：同架构，seed42初始化，原贪心输出、Natural Dynamic-K和STOP/NOOP；不强行配平数量，报告与Learned数量差异以提示混杂因素。

两类随机组各seed42/43/44/45/46，共10trial；初始H单trial，总计11个新文本策略。随机序列依赖seed、策略名和Caption，不含PID/配对/奖励；同Caption同seed输出相同。所有6156Query保留，指标层才读取PID。编辑记录保留原字符、新字符、码位、位置和原Span。

随机匹配采用打乱候选后的首个可行DFS序列，每Query最多256次候选边扩展；不是完整合法序列上的均匀采样。每属性/共享词最多1次编辑，总数≤K；m=0保留NOOP。节点耗尽时保留合法部分，报告`search_node_limit_no_complete_match`、target/actual和比例，不扩大预算，不改用Learned字符作兜底。此失败不代表数学上不可行。搜索穷尽却找不到原Learned合法见证时直接停止。

H-selected检验固定实际属性支持集下位置/字符策略的贡献，没有单独分离两个Head。两种随机组的m和K来自Caption-only的已有H行为，不利用Test匹配反馈。Untrained使用完整自然策略，不能忽略编辑数量/NOOP差异后将所有指标差异归因于SCST。

## 保护与完整性

只读复用`outputs/phase5_joint/full/`的Clean/Adv Gallery、原文文本特征、Learned动作与指标。新输出固定为`outputs/joint_ablation_v1/full/`，CLI不能指向历史目录。旧协议、metadata、回执、Selection、模型和结果均不覆盖，不关闭任何来源SHA检查。

预检核验原Full完成证据、所有缓存SHA、图像预算与3074/6156覆盖、Canonical/Span动作及70项历史来源；绑定模型、Selection、原图和正式Full全部输入SHA。本模块另锁6个生产文件，新增无关文件不导致锁定集合变化。原模型架构来自正式Selection协议，初始H参数SHA在CPU测试中与原seed42初始化一致：
`9d607ba8b6b348a8783b3dad9b79d0206dd23147edec5fa794b7e932c5a2853f`。

G-v2/IRRA权重SHA、H Selection及Checkpoint SHA均从原Full协议及其来源证据读取并核验，不硬编码新H选优结果。原Learned四组不重新排名；新trial仅Text-only/Joint调用已验证的官方IRRA指标实现，Clean/Image-only复用原精度数据。指标定义、Gallery、Query和预处理不改变。

新缓存紧凑独立Storage，原子保存feature/metadata/回执；正常恢复复用已提交单元。SIGINT/TERM在原子批次或指标组边界记录停止；SIGKILL/OOM的未提交单元仅在用户显式`--recover-pending`时保留至abandoned后重算。未知协议、损坏已提交数据拒绝。结果发布到完成回执之间中断，校验已保存执行记录绑定同一结果/策略后补齐完成回执，不改旧执行计数或时间。全部新尝试及恢复归档的SHA进入完成证据。

## CPU验证

最终命令：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest joint_ablation_v1.test_ablation phase5_joint.test_phase5 test_policy test_scst test_tokenizer_real -v
```

177 tests / 81.897s / OK。测试日志仅位于`/home/lzf/ldx/tmp/hablation_final_tests.log`，不提交。

覆盖数量匹配/预算/NOOP、合法属性随机性、实际H支持集、共享词/Span/Unicode及Tokenizer约束、有限搜索失败记账、Caption-only隔离、固定种子/RNG恢复、Untrained权重初值、冻结模型、合成特征官方排名、seed均值与样本标准差、紧凑Storage、SHA篡改拒绝、pending和完成回执边界恢复、完成态CPU重入。首次测试发现JSON排序造成CSV行顺序不一致，已仅在新序列化代码固定输出顺序并添加回归测试，未改任何指标。

已复核原有402个tracked文件SHA，全部保持不变；本阶段只新增文件。未读取或生成任何真实新消融缓存，未对历史实验目录进行写入。正式数据级SHA/真实动作覆盖核验留给用户CPU preflight，不把合成测试当作真实规模验收。

## 输出与统计

每个trial均输出Clean/Image-only/Text-only/Joint的mAP、R@1/5/10、mINP及DR_mAP；记录Joint−ImageOnly、Joint−LearnedJoint的mAP百分点差。DR_mAP=(Clean mAP−group mAP)/Clean mAP×100%。

根目录`metrics.csv`保存每seed完整精度明细，`aggregate.csv`保存策略族指标均值及样本标准差ddof1；单次Learned/Untrained标准差为空（JSON null）。同时输出各Query实际/目标编辑分布、NOOP、精确匹配率、不足/超额及原因、K/属性/字符统计。不只报告最佳seed，不作5seed显著性宣称。

## 用户手动执行命令

以下命令不得由Codex自动执行。本阶段结束后等待源码审查。

CPU预检（不做模型前向/指标重算）：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action preflight --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full
```

成功：`ABLATION_CPU_PREFLIGHT PASS`。

正式Full消融：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action run --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full --user-run-ablations
```

进度：`ABLATION_TEXT_COMPLETE <trial> n / 6156`、`ABLATION_METRIC_COMPLETE`、`ABLATION_TRIAL_COMPLETE`；全体完成：`ABLATION_FULL_COMPLETE`。

正常中断后原命令恢复；有未提交pending时，用户先检查失败证据，再显式：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action run --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full --user-run-ablations --recover-pending
```

完成后独立CPU只读审计（会重算新trial已缓存Text-only/Joint指标，无模型/GPU）：

```bash
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m joint_ablation_v1.evaluate --action audit --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/joint_ablation_v1/full
```

成功：`ABLATION_CPU_AUDIT PASS`。详细查看日志、Ctrl+C/SIGTERM停止、pending恢复、退出确认及结果路径见新README。运行环境和batch16必须保持一致，不自动后台启动、监控或重试。

## 预计成本与待验收事项

新增G前向0、IRRA图像前向0；最多67716条新IRRA文本样本（11×6156，NOOP复用所以通常更少），batch16最多4235个文本批次；Untrained最多6156个CPU H样本前向。原Learned无新编码，无模型训练或JVP。

原始新文本feature约132MiB，含可追踪编辑metadata建议预留2–4GiB，失败归档另计。CPU内存约6–10GiB、GPU约1–3GiB、耗时约30–120分钟仅为无GPU实测的粗估，实际可能因候选搜索、22次排名及输入SHA I/O更长，不能视为性能承诺。

等待用户审查并手动执行真实CPU preflight、正式文本GPU消融及独立CPU审计后，才能报告实际数量匹配率、NOOP、运行耗时/显存和攻击效果。当前没有这些真实结果，不判断属性选择或字符替换策略是否已获得贡献证据。不启动后续实验。
