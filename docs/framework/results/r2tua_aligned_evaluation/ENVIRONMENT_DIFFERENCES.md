# Clean差异与环境核查

## 数字差异

差值=本项目未舍入值减论文已发表两位数，单位百分点；它只描述报告差距，不表示知道论文未舍入值。
| Target | Ours R@1 | Paper R@1 | 差值pp | Ours mAP | Paper mAP | 差值pp |
| --- | --- | --- | --- | --- | --- | --- |
| IRRA | 73.148148 | 73.38 | -0.231852 | 66.097443 | 66.13 | -0.032557 |
| RaSa | 76.218323 | 76.51 | -0.291677 | 69.184067 | 69.38 | -0.195933 |
| APTM | 74.610138 | 76.53 | -1.919862 | 64.126839 | 66.91 | -2.783161 |

R²TUA的IRRA Clean四项数字与IRRA官方README报告一致；RaSa四项与官方README一致；APTM与其原论文CUHK结果一致。
这确认了公开参考数字的对应关系，**不能证明R²TUA直接复制了Clean、没有运行Clean，或必然用了同一权重文件**。

## 三层证据：已确认、公开源码可核实、作者实际环境未知

| 项目 | 本项目与公开参考的证据 | 对R²TUA作者实际运行能否确认 |
| --- | --- | --- |
| 官方Test意图 | R²TUA正文p5说官方split；我们实际3074/6156/1000PID | 逐成员Manifest、PID具体集合未公开 |
| IRRA架构源码 | 本项目实际加载副本11份文件与官方固定Git对象byte-exact | 作者具体源码版本/修改未公开 |
| IRRA权重 | 本地9ff620...只读重hash不变；官方README提供Model & log入口 | 作者Checkpoint SHA未公开；未确认同一二进制 |
| IRRA图像输入 | 384×128、PIL bilinear、CLIP mean/std；本轮3张Clean像素差0 | 作者Resize库版本/额外输入桥未知 |
| IRRA文本 | SimpleTokenizer/BPE77；6156条官方函数与本项目Token ID一致，15条截断 | 作者是否附加pre_caption及Token版本未知 |
| IRRA编码精度 | 官方build_model调用convert_weights，本项目Loader最后model.float | 作者dtype、AMP、TF32、batch未知 |
| IRRA指标 | 官方L2点积；全部同PID正例R@K/AP；与本项目定义一致 | 作者实际评分脚本、排序设备/tie行为未知 |
| RaSa图像路径 | 官方原PIL直接bicubic384²；本项目先共同bilinear384×128，再Tensor bicubic384² | 作者是否按原JPEG、是否使用共享物化桥未知 |
| APTM图像路径 | 官方原PIL直接bicubic384×128；本项目共同bilinear384×128后原生Tensor接口 | 作者具体图像管线未知 |
| RaSa/APTM文本与评分 | 本项目保留WordPiece50/56及各自Top128/融合公式，来源已冻结 | 作者文件版本、实际原生调用未公开 |
| 软件与硬件 | IRRA发布README torch1.9/vision0.10；本项目原生环境torch2.5.1/vision0.20.1；R²TUA正文torch2.1/3090 | R²TUA vision/Pillow/Transformers/CUDA及具体开关未公开 |
| 统一epsilon | R²TUA自身最终8/255明确；本项目G原RGB8/255 | Table1其他方法是否全部8、归一化位置未确认 |

不能把“R²TUA实际未知”写成“已证实不同”。同时，已确认的官方参考与我们差异也不能因为Clean接近就忽略。

## 新的CPU只读证据

仅固定官方Test前3张原Clean图像，比较归一化前的两条处理链；没有攻击、神经前向或评分。
| Gallery index | IRRA max_abs | RaSa max_abs | APTM max_abs |
| --- | --- | --- | --- |
| 0 | 0.000000000000 | 0.094117671251 | 0.058823540807 |
| 1 | 0.000000000000 | 0.114015512168 | 0.047058835626 |
| 2 | 0.000000000000 | 0.056353718042 | 0.050980418921 |

RaSa差异来自原图直接缩放与中间物化后再缩放/插值实现；APTM在相同目标尺寸下仍有bilinear与bicubic差异。
数值是**两条Clean预处理链之差，不是攻击扰动**，不能用它判定8/255违规。Tensor bicubic还可能出现轻微[0,1]外插值值，不能因此修改历史管线/加clamp后重签结果。
本轮只保存标量及源图SHA，不发布图像Tensor或数据集。

IRRA未发现当前这两条输入链的Token ID差异，3张像素对照也相同。因此对这个已测配置，不应继续把它的Clean差距直接归因为bilinear/bicubic。
但本轮没有真实特征/排名对照，尚未证明FP32与官方混合编码到底改变了多少Query、多少mAP。
RaSa/APTM的输入路径差异是已验证的候选影响因素，仍不是检索差距的已确定根因。

## 能做和不能做的归因

已确认：源码dtype路径不同，RaSa/APTM像素链不同，三组Clean报告值不同。
可利用公开源码进一步核实：固定官方权重/输入/精度后做一次独立Clean，记录feature/score级差异；此项本轮未执行，需要GPU新授权。
因作者未公开而无法确认：R²TUA实际权重、实际预处理/精度、完整测试清单、Baseline源路由/预算。CPU阅读不能替代这些证据。

不通过改Caption、删Query、挑Checkpoint、修改排序规则、调epsilon来“追平”73.38/66.13。
当前结果全部保持原归属，Raw Clean与SGA/DRA Matched Clean不互换。架构/Loss差异是方法设计，不是本报告拒绝比较的原因。
