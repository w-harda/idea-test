# Push-pull＋局部/全局图像增强：独立从零60轮实验

## 目标与方法身份

用户授权参考AoA、SGA和GLEAM扩展图像训练输入，并联合原push-pull训练G/H。独立分支codex/gh-pushpull-augmentation-v1，以62e575913894fff6a10556b37c9226a8f045c3a8为基准；原attraction、cooperative、alternating、prototypes、配置和历史产物保持。

这是TBPS三视图增强适配，不是AoA全算法复现。AoA官方仓库在冻结Commit a98dd4b496c5a7a719d8f395d04a3678342449e5仅有README和许可证，未公开MSA实现。局部/全局结构依论文III-D第5页实现；SGA固定f49e6926cd13148e49bdbc2c3e537660e096230c，GLEAM固定2ec57f387c0ae53d74e69d2cbaecc820589e933a。文件、许可证、SHA和适配差异见configs/framework/gh_pushpull_aug_v1.upstreams.json。第三方源码只读，不导入执行。

## 图像视图

每条原Train图像生成三视图：原始图、局部图、全局图。每轮2048原样本对应6144视图；没有增加原始Train成员或PID，也不使用Test反馈生成增强。

局部图：按4×2分块，每块随机一种变换，在同一水平身体分区内随机排列左右块，保留上下身体分区。全局图：随机串联3个变换。操作池为Gaussian blur、水平翻转、轻度旋转（±5度）、平移（±4%）和保持长宽比的resize-pad；不进行垂直翻转或跨上下身体分区重排。resize-pad先等比缩放，再随机填充，最后恢复384×128，参考SGA尺度多样性和GLEAM全局扩展；没有额外加高斯噪声。

recipe由seed42、Round、原row_id确定，使用独立Python Random，不消耗G/H全局RNG；同一Round内G训练和当前Bank使用相同clean视图。增强在CPU确定性生成再传GPU，可在独立CPU审计中逐像素exact回放。新Round更换recipe。所有view_names、操作及参数完整记账。

## G如何使用三视图

令x_v=T_v(x)，u_v=f_I(x_v)，u_av=f_I(G生成并投影后的x_v)，v_a为本轮冻结H生成文本的表示。沿用原候选Train集合C、温度0.07、pull/push权重均1及原KL方向。

每个视图的收益仍为：

J_v = KL(P(v_a) || P(u_v)) - KL(P(v_a) || P(u_av)) + KL(P(u_v) || P(u_av))。

G的Global Loss = −mean_v(J_v)。增强视图使用各自干净图像表示作push参考；原始视图继续绑定原clean候选特征。梯度共同进入同一G，每个原样本仍K个属性步＋1个原图联合步，每属性步立即一次Adam，原逐步detach机制保持。属性Local、紧邻Retain及对应JVP仅在原始视图上计算，不把空间变换后的图像当作原位置属性监督。

联合步从三个clean视图重新开始。所有累计扰动分别相对该视图的clean输入约束，实际FP32 RGB经向内舍入投影后严格L∞≤8/255，容差0。几何变化本身属于训练输入变化，不作为最终Test可用的扰动。

## H如何使用三视图

每轮G更新后，当前G对全部2048 Train的三个视图分别调用原generate_once并编码，得到6144条当前G图像特征。保留原2048×512原始视图库身份，同时独立封存2048×3×512视图特征和每行recipe/输入/特征SHA。

对sampled和greedy文本分别计算原push-pull奖励在三个当前G配对图像上的均值，然后沿用原SCST贪心baseline：A=R(sampled)−R(greedy)，loss=−detach(A)×sequence_log_prob。H每条合格原文本只更新一次，仍只编码原文/sample/greedy三条文本。H的属性、字符位置、替换字符三个Head、Natural-K、STOP/NOOP及每原词最多改1字符保持；不换成AoA MLM词替换。

推理仍为G(I)、H(T)。图像/PID只进入训练奖励绑定，H推理不额外接图像或PID。下一轮G条件来自更新后的H，并按原规则重新计算真实语境/JVP。

## 预算与审计

每正式Round仍8068次G Adam、2048次H访问、2037次H Adam；60轮总484080/122880/122220。三视图增加真实G/IRRA图像样本数，不增加Adam步数。每Round G输入30348个视图样本（含完整Bank），IRRA图像编码36492个视图样本（含clean参考），不含原有文本/JVP计数。Smoke仍8原Train、27 G更新、7 H更新，完整Bank仍2048×3。

独立CPU审计覆盖全部Round与原Train成员、三路KL及均值、实际SCST优势/loss、原JVP/FD、状态连续性、对方冻结、双Adam/RNG和Checkpoint身份、Unicode规则及全部view预算记录。数值重算使用独立FP64公式，atol=5e-5、rtol=2e-4；像素预算和SHA没有容差。最新完整Bank额外逐像素回放clean增强，并核验三路raw像素/扰动/特征。已退休历史像素依封存账本、原Manifest和预算记录核对，不声称重读已退休Tensor。

大型JSON继续≤4MiB分片，不修改全局64MiB限制。每轮发布完整新边界后，只退休本次输出中上一轮大Tensor；保留全部Ledger、三视图库特征、每行输入/recipe/feature绑定及每十轮配对权重。

## 唯一后台执行计划

新真实训练Smoke→独立CPU审计→10图/10原Query Joint Smoke→CPU审计；然后从零60轮，每10轮保存配对G/H并在训练前缀审计PASS后执行CUHK-PEDES × IRRA完整3074 Gallery/6156 Query Joint-only Test及独立CPU审计，再继续下一10轮。评测输入为原图/原Query，不做Test-time增强，图像仍严格8/255。六个节点预先固定，不根据Test选择模型或调整增强。

这六组IRRA测试用于比较同模型效果；跨模型泛化是否改善需要后续独立迁移评测，不能由增强设计直接推断。最终自动发布表格和R@1/mAP曲线。纯绘图使用已验证的apattack解释器/Matplotlib，不安装共享训练依赖。真实失败停止，保留FAIL，不自动重试。原监督器不重启。

输出：/home/lzf/ldx/outputs/tbps-framework-gh-pushpull-aug-v1-20261111。

控制：/home/lzf/ldx/tmp/gh_pushpull_aug_v1_20261111。

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-pushpull-augmentation-v1-20261111/operations/gh_pushpull_aug_v1_status.py"
```

成功状态SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED；失败状态FAIL_STOPPED_NO_RETRY。助手核验后台启动成功后停止监督，由用户按需检查。实现测试和启动身份另见证据与启动回执，不把CPU fixture当攻击效果。
