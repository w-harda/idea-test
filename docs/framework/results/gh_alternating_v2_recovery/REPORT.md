# GH-Alternating-v2：训练后CPU审计恢复（准备快照）

原60轮训练计算已完成，独立CPU语义审计待执行，正式Joint Test 0组完成。

- 原训练：60/60轮，G 484080更新，H 122880访问/122220更新，79478.77秒。不会重新训练。
- 原失败：`06_train_audit`调用全局`read_json`读取第2轮68178636字节账本，超过67108864字节限制。旧Supervisor保持`FAIL_STOPPED_NO_RETRY`。
- 60份账本完整严格解析和Manifest/SHA核验通过；总3890547998字节，范围63102008–68178636字节，超限轮2/3/4/5/7/9/10。结构核验不能代替完整数值/预算审计。
- 第2轮字段占比：G 16.038%、H 28.657%、conditions 55.170%，其余约0.135%。逐轮精确大小/字段字节见DIAGNOSIS.json。
- 同一第2轮账本的独立纯解析进程：整份加载峰值421588992字节（约402.06MiB），分项读取20951040字节（约19.98MiB），见MEMORY_COMPARISON.json。正式语义审计含Torch及最终Bank，内存峰值另行实测。
- 已封存43907个历史文件的SHA及inode/size/mtime基线，覆盖原输出和原Supervisor；158份生产源码、496份运行依赖及原配置SHA保持。

修复仅新增operations独立工具。全局read_json、原训练/评测生产源码和冻结配置不改。原Manifest/完成回执/文件SHA先验核验，大数组逐条解析；严格拒绝重复键、非有限数值（包括溢出）、非法UTF-8、截断、尾随及结构错误。64Ki字符读取块，单项最大4Mi字符，无整份大账本常驻。

审计完整覆盖60×2048 G/H/conditions、K+1与累计更新、全部H sampled/greedy Unicode回放、原候选合法性和Natural-K、状态/父态声明链、当前RewardBank、原JVP回执/FD与冻结隔离，以及六组G/H状态身份。第60轮保留的条件/rollout和完整Bank独立读取Tensor。第1–59轮原始Bank/完整状态/rollout已按原滚动策略退休，只能用封存账本/退休Manifest验证，不能宣称重新读取已退休Tensor，也不会重建。

9项新增CPU测试（含17类非法JSON、6种块边界及真实超64MiB合成输入）与27项相关回归PASS。正式独立CPU审计尚未PASS，不填正式指标。

新控制目录：`/home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108`。新审计输出：原输出下`posttraining_recovery_20261108/train_cpu_audit`。原始审计失败回执和所有历史文件保留。

新Supervisor只执行：CPU审计 → G10/H10至G60/H60六组Joint-only完整Test → 各组独立CPU审计 → 表格/曲线/报告 → Git发布。任何真实错误停止，无重试；不调用原Supervisor，不训练，不重新计算训练JVP或历史Bank。六组固定CUHK-PEDES×IRRA，3074 Gallery/6156 Query。
