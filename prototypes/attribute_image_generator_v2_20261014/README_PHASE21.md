# G-v2 Phase 2.1：独立图像生成器、投影与损失

基准分支codex/gv2-per-attribute-cache，Commit 0a2afc93ab8f36849ac294712dab853a49d792d1。
本阶段只实现模块与CPU验证，无K轮训练循环、optimizer step、GPU、正式训练或Phase 2.2。
G-v1及Phase 1原源码/配置/缓存保持；新模块不导入G-v1或离线integration。

## 文件与接口

| 文件 | 接口 |
|---|---|
| image_only_generator.py | ImageOnlyGenerator(width=32, blocks=4, epsilon=8/255).forward(image) |
| budget_projection.py | project_image(original, proposed)、add_projected_delta(original, current, delta) |
| losses.py | attribute_losses、joint_losses、retain_loss、retain_reference |
| test_phase21.py | 66项新增CPU tests |
| README_PHASE21.md | 本说明 |

实现位于服务器发布工作树：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/
原实验工作树及5条Phase 1缓存保持，不自动镜像两工作树。

## 生成器

输入仅为float32/float64 RGB [B,3,H,W]，范围[0,1]，H/W至少8。
内部2I-1；7x7卷积Encoder、两次stride2、4个ResNet Blocks、两次双线性上采样Decoder。
默认width32、1,377,411参数；InstanceNorm带affine且无running statistics，SiLU激活。
Decoder显式恢复目标尺寸，支持奇数宽高；输出epsilon*tanh(logits)，与输入同尺寸。
无FiLM、文本分支、z_t、属性或JVP输入；只从预算模块导入常数及校验工具。
未加载G-v1/AP权重，未训练，未做性能选优。

## 预算与梯度

proposed = current + delta；adversarial = project_image(original, proposed)。
投影区间为[0,1]与[original-epsilon, original+epsilon]交集，epsilon不允许超过8/255。
原图仅作detach固定参考；current不隐式detach，未来轮间detach由后续训练循环负责。
单轮delta和输入current累计预算严格检查，不接受额外容差。
float32/float64端点向内舍入到可表示像素，避免原图加减epsilon的舍入造成越界。
nextafter只用于无梯度边界计算，proposed保留真实torch.clamp计算图。
单轮输出幅度常数也向内舍入，没有增加预算。

- 内部导数1，严格越界饱和分量导数0。
- 恰好等于可表示端点，PyTorch采用导数1的边界约定；数学上该点不可微。
- 混合测试导数[0,1,0]；全部越界饱和会完全阻断本次G梯度。
- 不采用STE、soft clamp或额外loss绕过饱和。

网格/随机/连续纯投影算子测试、float64测试和内部gradcheck通过。
实测fp32最大L∞为0.03137254901230335，不超过8/255=0.03137254901960784。
只有CPU模拟验证，尚未评估真实IRRA训练轨迹的饱和比例。

## 损失

image_features及监督为[B,512]且dtype/device匹配；监督张量须detached。
不重新编码文本、评分或构造JVP，不重加权/归一化JVP。
各分项按整个batch取mean，逐属性三个系数均为1：

Global = mean(f_current dot z_t)
Local = mean(f_current dot current_jvp)
Retain = mean(ReLU(f_current dot previous_jvp - reference_score.detach()))
Attribute total = Global + Local + Retain

正dot最小化。Retain仅针对紧邻上一轮属性，不是所有历史属性。
参考应在上一轮完成的对抗图像上取分数；retain_reference返回stop-gradient结果。
第一轮previous_jvp/reference_score均不提供，Retain=0；只提供一项则拒绝。
参考分数即便requires_grad也会在loss内detach，不接收Retain梯度。
分数升高才惩罚；继续下降及ReLU零点导数为0。

Joint total = mean(f_current dot z_t) + mean(has_local ? f_current dot v_local : 0)

直接消费Phase 1原S_raw加权v_local；K0监督须为零，Local严格为0。
只计算loss，不执行原图重新前向、optimizer或训练调度。无扰动惩罚Loss和TV Loss。

## CPU验证

111项通过：新增66项+Phase 1原45项，原测试文件不改。
覆盖标准384x128/奇数尺寸、幅度/累计预算/像素边界/饱和与端点导数、
loss数值和方向、K0/mixed mask、Retain范围/stop-gradient和冻结模拟编码器回传。
CPU读取原实验目录5条真实Phase 1缓存验证joint_losses接口，无新JVP或评分。
AST白名单及禁止前端导入的独立子进程验证新模块无G-v1/integration依赖。

模拟attribute/joint的G梯度L2分别0.0330260/0.0323426，编码器参数梯度均0。
仅loss backward，无optimizer step，不能据此评价实际攻击效果。
42份保护文件SHA保持，含G-v1源码/config/epoch10权重/Test摘要、原Phase 1缓存、
本发布目录8个Phase 1文件。全程未初始化CUDA。

~~~bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase21 test_conditions -v
~~~

证据仅在本目录outputs/phase21/：
cpu_tests.log、summary.json、protected_sources.json。
cpu_tests_initial.log保留首次历史文件标记冲突的失败证据；新模型改名后111项通过。
未放宽门禁或改Phase 1测试；当前无generator_model.py/train.py/generate.py。

本阶段不commit/push，等待用户确认；未准备2176条缓存，未进入Phase 2.2。
