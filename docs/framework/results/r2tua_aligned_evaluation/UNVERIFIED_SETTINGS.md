# 尚不能核实的R²TUA运行设置

UNKNOWN=事实身份未知；NOT_REPORTED=已读正文/补充未报告；NOT_VERIFIED=存在公开参考但不能证明作者实际使用。未找到公开实现不等于断言全球绝对不存在。

| ID | 设置 | 证据缺口 | 目前能确认到哪里 | 引用影响 |
| --- | --- | --- | --- | --- |
| U01 | R²TUA精确Test成员、Gallery/Query数/PID清单 | 正文仅官方split，无Manifest/运行日志 | 可确认标准读取器及本机3074/6156/1000；对方具体集合UNKNOWN | 不宣称逐成员完全一致 |
| U02 | R²TUA IRRA/RaSa/APTM Checkpoint SHA及代码版本 | 正文引用模型论文，未提供二进制来源身份 | 官方发布入口存在；对方使用哪份UNKNOWN | 禁止用Clean接近反推SHA一致 |
| U03 | R²TUA实际图像管线与精度 | 缺少公开实现/库版本/运行配置 | 公开IRRA实现可确认；作者实际dtype/Resize/量化UNKNOWN | 未来公开参考Profile不能冒称作者Profile |
| U04 | R²TUA实际文本预处理/Tokenizer边界 | 正文未给实际代码/词表摘要 | 本机Raw6156 BPE逐值一致；作者是否另做pre_caption未知 | 不得合并Raw和Matched Clean |
| U05 | 各Table1 Baseline的实际epsilon | R²TUA自身8/255设置，不足以证明其他方法都8 | SGA/VLPT等原论文参数不能代替作者适配参数 | 引用表epsilon列NOT_REPORTED，不写全体8/255 |
| U06 | R²TUA8/255的准确像素空间/投影与归一化位置 | 仅L∞/255数学表述，无源码 | 原RGB[0,1]是否确为运行空间NOT_VERIFIED | 保留本项目原RGB8/255；不得把二者写成已严格同空间 |
| U07 | SGA/VLPT/MFHA/AnyAttack在Table1实际修改Query吗 | 只称可适配VLP攻击，未公开最终输入组合 | 原算法模态作为背景单列；作者实际输出UNKNOWN | 不把论文行直接命名为Joint或Image-only |
| U08 | Table1各Baseline Source/每Target是否重新生成 | 无Source权重及路由配置 | 不能推测统一IRRA，也不能仅凭Target列断定每方法逐Target白盒 | 本项目IRRA迁移不与Table1当作相同权限 |
| U09 | Test正确图文配对、PID标签、prompt构造 | 原方法/训练章节描述不等于作者测试实现 | SGA等原算法Test配对/源梯度需求可知；R²TUA实际保留情况未知 | 权限差异/未知必须脚注 |
| U10 | 统一评分器/图像预处理/预算是否适用于Table1全部方法 | 共享Clean和指标名不证明全部输入及调用相同 | 无作者适配代码和逐行配置，统一性NOT_VERIFIED | 不能标注统一复现或等条件受控实验 |
| U11 | 作者实际攻击训练图片/PID/样本次数 | 官方split声明，无训练成员账本 | 不能把官方Train34054当作实际攻击训练量 | 方法级引用可保留，训练预算优势不得推断 |
| U12 | 实际全排名、ties、multi-positive实现一致 | 正文仅R@K/mAP标准指标 | 公开IRRA与本项目数学可核实；作者实际脚本未知 | 按来源报告，不计算虚假精确差或显著性 |

## 公共资料搜索边界

本轮复核CVF正式论文页面、正文与补充材料，并按R2TUA、Reconstruction-residual、Targeted and Untargeted person检索GitHub仓库。
R2TUA仓库搜索为0；词组检索只有无关工程；CVF页面未给GitHub入口。没有找到可核实的作者适配代码。
本报告不使用AI论文解读、聚合榜单或搜索引擎生成摘要作为运行事实。
IRRA官方Git对象已可获得，RaSa/APTM官方源与发布链接也可核实；这些是“公开参考”，不能自动升级为“R²TUA实际使用”。
任何未来新增作者资料应单独版本更新本清单，不回写已发表结果或本轮原始证据。
