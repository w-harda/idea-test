# G-v2 Phase 2.1：独立图像生成器、预算投影与损失

2026-10-14：源码与CPU验证完成。新增66项测试加Phase 1原45项，共111项全部通过。实现与验证只做CPU模拟编码器及loss backward，无optimizer step、GPU初始化、正式训练、完整缓存准备或Phase 2.2。初步验收后用户授权将本阶段5个新增文件和本中文报告提交到当前远端分支，供独立源码审查；发布阶段不重跑GPU、不进入Phase 2.2。

## 工作树与新增文件

本阶段在已发布Phase 1工作树开发：

`/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/`

分支`codex/gv2-per-attribute-cache`，本阶段实现基准为`0a2afc93ab8f36849ac294712dab853a49d792d1`。所有新增实现位于其中`prototypes/attribute_image_generator_v2_20261014/`，未同步至旧实验工作树，也未改G-v1。原实验目录仍保存Phase 1真实5条缓存。

| 新增文件 | 职责 |
|---|---|
| image_only_generator.py | 独立ImageOnlyGenerator，仅G(I) |
| budget_projection.py | 相对原图的累计L∞预算、像素裁剪、单轮扰动合成 |
| losses.py | Global、当前Local、上一轮Retain与联合目标 |
| test_phase21.py | 66项新增CPU测试与冻结模拟编码器 |
| README_PHASE21.md | 本阶段接口、公式、梯度约定与命令 |

模型、投影、loss均不import G-v1或Phase 1 integration。实现模块的依赖只包含torch、标准库和本阶段budget_projection。AST白名单及禁止导入integration/common/runtime/conditioning/deps的独立子进程验证通过。Phase 1 schema/cache_io只在读取监督的测试中使用。

## 模型

`ImageOnlyGenerator(width=32, blocks=4, epsilon=8/255).forward(image)`

默认1,377,411参数。7×7卷积Encoder、两次stride2、4个ResNet Blocks、两次双线性上采样Decoder；SiLU、affine InstanceNorm且无running statistics。内部归一化为2I−1，输出epsilon×tanh(logits)。没有FiLM、文本/属性分支或预训练权重；架构未做实验选优。

输入float32/float64 RGB [B,3,H,W]、[0,1]且H/W至少8；输出同尺寸扰动。384×128、奇数尺寸和float64均通过CPU测试。z_t/JVP/属性不进入G接口。

## 预算投影与梯度

`add_projected_delta(original, current, delta)`只做一次合成：

`proposed = current + delta`

`adversarial = clamp(proposed, max(0, original−epsilon), min(1, original+epsilon))`

original为detach固定预算参考；current不被隐式detach，未来轮间detach由Phase 2.2训练调度负责，本阶段没有实现该循环。输入current的累计预算与delta单轮幅度均严格检查，epsilon不允许超过8/255。

边界向内舍入到可表示像素，nextafter仅在无梯度的边界计算中使用；单轮幅度常数同样向内舍入。避免原图加减epsilon的fp32舍入产生预算外像素，不放宽容差、不增加预算。

真实torch.clamp导数：内部为1，严格越界的饱和分量为0；恰好等于可表示端点时PyTorch采用1的边界导数约定（数学上不可微）。混合测试导数为[0,1,0]，内部float64 gradcheck通过；像素0/1边界也通过。

**全分量越界饱和时，本次前向会完全阻断G梯度。** 这是硬投影的预期行为，本阶段没有使用STE、soft clamp或额外loss绕过。尚未评估真实IRRA训练中的饱和比例。

网格/随机、正负方向、float64与多次纯投影算子验证均通过。fp32最大实际L∞=0.03137254901230335，严格不超过8/255=0.03137254901960784；不沿用G-v1的fp32额外预算容差。

## 损失契约

输入feature和监督为匹配dtype/device的[B,512]；z_t/JVP须detached。各分项按全batch取mean，逐属性三个分项权重均为1。保留原JVP尺度，不重新编码文本、评分、JVP、归一化或调整S权重。

`Global = mean(f_current · z_t)`

`Local = mean(f_current · current_jvp)`

`Retain = mean(ReLU(f_current · previous_jvp − reference_score.detach()))`

`Attribute total = Global + Local + Retain`

正dot最小化。Retain仅约束紧邻上一轮的属性；参考分数应由上一轮完成后的对抗图像取得，retain_reference返回detach结果。参考分数即使requires_grad也在loss内detach；测试确认参考图/参考分数不接收Retain梯度。恢复使dot升高才惩罚，继续下降不惩罚，ReLU零点导数为0。

第一轮不提供previous_jvp和reference_score，Retain=0；只提供其中一项则拒绝。此处不实现获取上一轮图像或参数更新的调度。

`Joint total = mean(f_current · z_t) + mean(has_local ? f_current · v_local : 0)`

直接使用Phase 1保存的S_raw归一权重聚合v_local；不重复加权。K=0时监督必须为零，Local严格为0；混合K batch以全batch均值计算。z_t只用于Global监督，始终不作为G输入。没有扰动惩罚Loss和TV Loss。

## 验证结果与证据

- 新增66项CPU tests通过；Phase 1原45项回归通过，共111项。CUDA未初始化。
- Global/Local/Retain解析数值、最小化方向、JVP尺度、K0、mixed mask与stop-gradient通过。
- 两个目标经冻结模拟编码器回传到G，G梯度L2分别0.0330260（attribute）与0.0323426（joint），扰动梯度均非零且有限；编码器参数梯度均0、参数状态保持。
- 全分量饱和阻断G梯度测试通过；内部/混合/预算与像素饱和分别验证，未用替代梯度。
- CPU只读原5条真实Phase 1缓存验证联合loss接口，没有新评分或JVP。
- 42份保护文件SHA保持，涵盖G-v1源码/config/epoch10权重/完整Test摘要、原Phase 1缓存与发布目录8个Phase 1文件。没有改Phase 1源码或测试。
- 首次联合测试发现Phase 1以generator_model.py存在作为阶段边界标记，随后仅将新模型改名image_only_generator.py并更新新增测试；失败日志保留，未改原测试或放宽数值门禁。

输出仅在本阶段独立目录的`outputs/phase21/`：`cpu_tests.log`、`cpu_tests_initial.log`、`summary.json`、`protected_sources.json`。通过日志SHA256为`1820ca74cdff455490dbbc75c3a8466e93816aad6cbdd124f1ee1f74a3722e3b`。

手动CPU复核，在本阶段服务器目录执行：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase21 test_conditions -v
```

只有CPU接线与数值验证，不是正式IRRA训练或攻击性能结果。未实现完整K轮loop、原图联合步调度、checkpoint恢复、完整缓存或评测；未进入Phase 2.2。发布范围仅为5个新增文件和本报告；数据、权重、缓存、outputs、临时文件及Windows状态管理文档不纳入Git。README_PHASE21.md中的“不commit/push”记录实现验收时边界，本次后续用户授权允许发布，源码与已测文件内容保持一致。发布完成后等待独立审查。
