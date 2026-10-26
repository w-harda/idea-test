# 可比性独立审计

结论判定针对明确场景，不给方法名永久贴标签。MATCHED表示已核实的关键条件一致；PARTIAL表示部分一致且有关键已知差异/未核实项；NOT_MATCHED表示已知关键条件冲突；UNKNOWN表示证据不足。

| 编号 | 比较 | 判定 | 证据与解释 |
| --- | --- | --- | --- |
| A | G60→IRRA vs R²TUA Table1 IRRA untargeted | PARTIAL | 同为IRRA同源、Image-only、最终8/255；但对方权重/图像桥/实际Train规模、Test prompt构造不公开。不能把21.05/17.51与0.58/1.07解释为等条件纯算法差。算法架构/Loss不同本身不是不公平理由。 |
| B | G+H Joint vs R²TUA image-only untargeted | NOT_MATCHED | 我们修改Gallery与Query，R²TUA图文条件生成图像、未证明修改Query；Unicode预算与prompt权限不同。可做方法级结果展示，不能称等攻击面比较。Untargeted不是Image-only同义词。 |
| C | IRRA→RaSa/APTM vs R²TUA Table3 | NOT_MATCHED | 对方RaSa来源APTM+MARS，IRRA来源Ham+MLLM4Reid，均为集合元学习；没有APTM unseen-target结果。不是固定IRRA训练后迁移。 |
| D | 实际2048图/1861PID vs 对方训练规模 | UNKNOWN | 我们的实际规模已核实；R²TUA仅说采用官方划分，未给实际攻击训练图片/PID数。不能把34054官方Train总数当作对方实际用量。 |
| E | Clean差异的具体原因 | UNKNOWN | 数值不同已确认；原因可能涉及权重、输入桥、tokenization、排名等，但论文无SHA/代码，不能从Clean差异直接锁定某一原因。直接分数比较仅PARTIAL。 |
| F | 本项目8/255迁移 vs Table3未加星16/255 | NOT_MATCHED | 最终图像预算相差2倍。星号8/255只对齐epsilon，仍未对齐源集合/训练权限及输入协议。 |
| G | SGA/VLPT/MFHA与冻结G/H的测试权限 | PARTIAL | 原算法需Test图文配对和源梯度，我们无Test优化；R²TUA具体适配Source/文本输出协议未知，不能自动宣称它们使用我们相同权限。 |
| H | 本项目官方SGA/DRA移植 vs R²TUA同名表格行 | UNKNOWN | 本项目官方字节SHA和IRRA Source明确；对方作者适配diff/Source/预算等未公开。算法名相同不能证明实验身份相同。 |
| I | G10 vs G60 Image-only 本项目内部 | MATCHED | 同Raw输入、同Manifest、同三Victim权重/评分、同8/255与生成方式；预设轮数消融可比较。训练轮数是有意变化量。 |
| J | G60+H60官方Test vs 任一论文数值 | UNKNOWN | 未找到指定G60+H60官方Test Full；128内部Validation不能混入。 |
| K | Table2 Market1501 vs 本项目CUHK文本检索 | NOT_MATCHED | 数据集、查询模态、受害者及评价集合不同；不能抽取其R@1/mAP替代TBPS。 |
| L | Targeted↑ vs 我们Untargeted↓ | NOT_MATCHED | 成功语义及指标方向不同，不能比较数值大小判断攻击强弱。 |

## 方法差异不等于无效比较

不同方法本来就允许有不同架构、Loss、训练策略及原生计算成本。这些是被比较的算法组成，不应为了“公平”把SGA/DRA改成我们的Loss，或把G改成R²TUA。
方法级Benchmark可以保留原生预算/权限并在同表明确披露。只有声称“等预算、等权限、同评估器下更优”或据跨论文分数归因单个机制时，才需要相应控制。
当前证据支持的是有协议标签的描述性对照，不支持把跨论文差距全部归为G算法弱，亦不支持宣称论文造假或我们实现被优化。

## 各Baseline的R²TUA适配证据

SGA、VLPTAttack、MFHA：Table5显示无离线训练；原算法的源梯度及配对文本需求可核实，但Table1具体Source、文本输出是否全部进入评分、每行epsilon/步数未知。
AnyAttack：Table1含目标/非目标，Table3纳入元学习；不能用其原始16/255 CLIP/LAION设定填补R²TUA实际适配参数。
MisRanking、MetaAttack、MTGA：正文明确在TI-ReID中只用图像；确切Loss、被攻击图像范围、标签使用细节、Checkpoint和训练量未公开。
R²TUA：图像+prompt生成器；IRRA只用PPLs，RaSa/APTM添加Matching Loss。Table3源池和epsilon可直接核实。
