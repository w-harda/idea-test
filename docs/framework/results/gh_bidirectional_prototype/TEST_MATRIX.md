# CPU测试与验收清单

本轮测试以合成CPU状态/特征执行真实新控制流；不是旧GPU Smoke重跑。

| 类别 | 覆盖 |
| --- | --- |
| H→G锚点 | 六字段严格匹配、不同G/H ID、重复歧义拒绝、原Span/新UTF-8 Span、完整原生replay先行 |
| 新条件 | 非NOOP调用新编码/定位/JVP；旧JVP改标签/缺项/错误空间/FD失败拒绝；A1及NOOP不触发重算 |
| 原算法薄接线 | G函数参数顺序、原Loss梯度变化/NOOP exact、原RewardEvaluator在新bank下Reward/Advantage变化 |
| 两轮A1/B | 完整8条更新顺序、相同父状态、4次2048全库刷新、G阶段H不变、H阶段G不变、异常无round提交 |
| 信息隔离 | H公开输入仅caption/attributes；PID输入拒绝、错误IRRA空间和Clean绑定拒绝 |
| 恢复 | 双模型/双Adam、Python/NumPy/Torch CPU/H generator、下一步连续/恢复exact；CUDA RNG不得隐式初始化 |
| 存储 | clone脱离Storage、原子round读回、重复Claim/不可变版本、篡改/中断pending/更晚Claim拒绝 |
| 资源 | 高利用率且显存足够允许观察；显存/RAM/磁盘不足拒绝；GPU状态异常停止；无终止其他进程操作 |
| 无副作用 | 新模块在独立Python进程导入不加载torch；CPU测试硬拒绝真实CUDA初始化 |

最终61项新增、162项相关回归PASS；命令、日志SHA及退出码记录在SOURCE_EVIDENCE.json。
早期完整回归为155项，后续补充完整native动作转换与空间隔离测试，不把多次运行相加成独立用例数。

CPU标准库预检核验：

- G60/H10/IRRA真实Checkpoint文件字节SHA、父元数据和原完整contract索引；
- 原113 producer、G prepare 7 producer、4450 upstream、H训练76来源及本版本代码；
- 2048/128资格与annotation实际成员、Caption、PID、原图SHA、固定8条严格锚点；
- 原Clean图库完成回执/源文件、有序成员；不反序列化Tensor或重跑排名；
- 历史151378份文件size/mtime/inode及关键来源读取前后保持；
- 新实验输出目录未创建，GPU前向/JVP/训练为0。

真实GPU待验收项：

- 新文本的实际BPE/EOT/block8/reader定位与JVP数值；
- 两臂真实G梯度/Adam更新差，真实新bank对H梯度/更新的影响；
- 双Adam/CUDA RNG完整读回、模型参数与buffers冻结及实际预算/资源；
- 完整四轮真实库刷新和最终独立CPU审计。

只有CPU PASS不能将上述GPU项标VERIFIED。
