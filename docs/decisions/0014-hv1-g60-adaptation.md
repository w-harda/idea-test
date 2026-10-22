# ADR-0014：H10父状态与G60奖励条件的独立适配续训

日期：2026-10-21。决定：按用户本轮授权实施。

## 背景

原H1–H10使用G10图库。新任务延长至H60，并切换为G60奖励图库，不能改旧epochs=10协议或伪称从零G60训练。

## 决定

保留完整原H10协议、Checkpoint及Adam/Python/NumPy/Torch/H采样Generator/CUDA RNG；
独立Extension绑定父SHA、原配置、新G60奖励图库、原seed及绝对Epoch顺序和新输出。
原H10不重发成新初始Checkpoint。新链从父cursor20480/updates20370开始，
调用原Checkpoint事务并按新父边界审计，不调用旧audit/rollback写历史入口。
训练复用原SCST单步和RewardBank/FreshReward，不重新实现奖励数学。
G60仅生成2048 Train一次；原Clean和已审计128 G60 Validation特征经来源绑定只读复用。
正常停止在Epoch边界提交；pending拒绝自动回放，恢复只能显式核验最后完整提交状态。

## 影响

这是相同H算法/Optimizer状态下改变奖励条件的适配轨迹，不是原G10轨迹不变的延长。
预设H10/20/30/40/50/60，在重复内部Validation上描述选优；旧H7 Selection与正式Test均不改。
没有G10延长训练控制，不能将改善完全归因于G60。CPU恢复exact不证明CUDA未来轨迹bitwise一致。

## 2026-10-22：CPU审计修复版本隔离

保留原GPU执行树、113 producer配置与失败证据；新树仅给审计器局部stat字典改名。
独立版本配置核验精确允许差异，旧配置仍由旧运行包完整验证。
修复版审计使用新独立版本回执绑定旧产物，不冒充原producer。
正式训练复用旧算法入口，最终报告仅输出到新修复分支。
