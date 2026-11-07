# GH-Alternating-v2 从零60轮与六节点Joint Test

当前：短GPU训练Smoke、Joint执行Smoke及两项独立CPU审计全部PASS；正式从零60轮已启动，当前第1轮进行中（0/60轮完成）。**尚无本次正式Test指标，不以Smoke或训练诊断填表。**

执行源码Commit `f2aeb14ba24c50f70d124d3be6da76895dc4904b`，分支 `codex/gh-alternating-v2`。配置独立冻结。

- 原151份v1源码与496项运行来源保持，新版本当前158个producer严格绑定。
- 44项不同CPU用例PASS：17项v2新增、27项已有直接相关回归。
- 真实初始化证明G/H均seed42新建、Adam为空、历史G/H Checkpoint加载0、旧动态Bank未加载、WordBudget1未安装。
- 原H策略已验证K=2、两个不同原词位各替换1字符；不是新整句1词预算。
- 固定2048 Train、每轮G/H各一遍；原学习率2e-4/1e-4、原Adam、原脚本无Scheduler，状态None。
- 真实训练Smoke：G更新27、H访问8/更新7、新JVP19、完整图库2048，独立CPU审计PASS。G/H参数均发生变化：True/True。最大L∞=0.0313725471496582≤8/255；SCST动作最多观察到3个词位编辑。
- 10 Gallery/10 Query Joint执行Smoke和原生独立CPU排名审计PASS，仅链路验证；不把它作为正式攻击效果。
- 正式训练是另一套seed42新G/H、空Adam/0计数，未继承Smoke状态。正式第1轮已完成2048条新语境条件重建，进入G训练；本次只读快照G累计更新2415次、H尚未更新，0/60轮完成。Supervisor将在60轮/训练审计完成后自动六组正式Test。
- 第10/20/30/40/50/60轮导出独立配对G/H；全部训练和审计结束后自动进行六组CUHK-PEDES × IRRA Joint Test。
- 训练恢复只保留最新完整边界及Bank，旧新产物显式退休，六组评估权重和小型账本永久保留。真实失败停止保留，无自动重试。

正式结果0行，R@1/mAP曲线尚未生成。六组正式Test完成并独立审计后，本报告会自动更新为六行表格及两张曲线并推送GitHub。

实时控制：`/home/lzf/ldx/tmp/gh_alternating_v2_supervisor_20261107/status.json`。
训练进度：`/home/lzf/ldx/tmp/gh_alternating_v2_train_20261107/status.json`。
输出：`/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107`。

[完整运行、参数与恢复说明](../../GH_ALTERNATING_V2.md)，[来源证据](SOURCE_EVIDENCE.json)。
