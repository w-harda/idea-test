# 图像侧一次前向条件生成器最小 prototype

2026-10-14：服务器独立实现、37项CPU tests、极短GPU smoke及独立CPU生成入口通过。仅2 train+2 validation条件、2次G optimizer update；长时间训练未启动，无commit/push。

## 实现位置与模块

服务器：/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/

| 文件 | 职责 |
|---|---|
| generator_model.py | 轻量条件ResNet、clip合成、最小loss、generate_once |
| conditioning.py | 冻结Top-K的S_raw归一权重、context聚合和JVP聚合 |
| integration.py | 只读复用冻结Step5.1、v5.5、IRRA、reader、JVP |
| prepare.py | 独立离线条件准备；默认preflight |
| train.py | G训练、内部validation诊断、fixed-final checkpoint、epoch级resume |
| generate.py | 独立在线入口：一次G forward，不导入victim/scorer/JVP/TTA |
| smoke.py / test_generator.py | 2-step GPU smoke / 37 CPU tests |
| config.json | 固定第一版配置；未搜索超参数 |

未修改正式Stage01/Stage03/Top-K、generator_framework.py或任何旧prototype。旧Stage06只接收image的抽象接口没有被直接改成conditional consumer；本轮使用独立train.py。Windows只保存本研究文档。

## 数据流与一次前向的准确边界

离线：

caption + 已有Stage01 provenance
→ 冻结AttributeInstance adapter
→ 原IRRA block8 / frozen v4 reader / Euclidean basis
→ 冻结S_raw与Natural Dynamic-K
→ 原JVP
→ 固定(z_t, c_attr, v_local)缓存及source/span/score/SHA记录。

训练：

(image, z_t, c_attr) → G → δ → I_adv
→ frozen IRRA image encoder
→ global/local/perturbation/TV loss
→ optimizer只更新G。

在线：

(image, 已准备text condition中的z_t和c_attr)
→ 一次G forward → bounded δ → clip(image+δ)。

**“一次前向”指给定冻结条件的一次G调用，不是任意raw image+caption端到端只做一次神经网络调用。** 冻结S_raw需要retrieval backward，未知caption的准确S不能靠现有前向接口获得。本prototype选择离线准备并锁定条件；在线不偷偷反传、重评分或迭代攻击。未知row/caption/image不匹配时明确拒绝，不临时发明S predictor/fallback。

JVP仅用于离线构造训练local target；G在线不接收v_local，不需要victim checkpoint。generate.py没有导入integration/runtime/attribute_tta；可以在CPU上只加载G和缓存条件生成。

当前prepare只提供原train内部train/validation条件，在线示例为held-out validation。正式test G条件准备和共享图库性能评测未在本轮实现。一次G前向也不意味着与无caption的AP生成器具有相同信息访问条件。

## 条件与公式

z_t=f_T(T)，为冻结IRRA最终L2归一retrieval caption feature，512D。

对于冻结Top-K实例a_i，保持S_raw、source span、region/value/provenance和原projected_context不变：

\[
\omega_i=S_i/\sum_{j\in A^*}S_j,\quad
\hat e_i=e_i/\|e_i\|_2,\quad
c_{\rm attr}=\sum_{i\in A^*}\omega_i\hat e_i .
\]

e_i是已有block8 Euclidean projected_context；L2 normalization只在G条件读取端进行，沿用旧attribute_delta使用单位方向的习惯，不改原AttributeInstance。排序不变，权重是S_raw，不是S_raw^gamma。

G内部对z_t、c_attr分别LayerNorm后concat。这是不同表示类型的拼接与可学习条件调制，不直接将block8向量当作retrieval feature做点积。

局部目标继续使用原映射：

\[
B_i=w_i^{loc}\hat e_i^\top,\qquad
v_i=D_H f_T(T)[B_i],\qquad
\bar v=\sum_i\omega_i v_i .
\]

w_i^{loc}是原单位L2 region/location token weights。每个v_i都来自原block8之后的真实IRRA forward JVP，使用原zero/FD gate。不把block8 e_i与image retrieval feature直接点积。v_i保持原未归一化尺度，聚合后也不重新归一化。

## G结构与预算

AP-style encoder-decoder：ReflectionPad/7×7 Conv → 两次stride2 downsample → 6个residual blocks → 两次bilinear upsample+Conv → 3通道输出。基础width32，bottleneck128通道。

条件MLP为1024→256→256，产生bottleneck的gamma/beta：

h' = h * (1 + 0.1*tanh(gamma)) + beta。

条件MLP属于新G，不是重新训练reader、query或semantic basis。使用无running-stat的InstanceNorm；不使用AP spectral-state更新或原生随机style模块，也不声称逐值复现AP/MTGA实现。

\[
\delta=\epsilon\tanh(u_\theta(I,z_t,c_{\rm attr})),\quad
I_{\rm adv}=\operatorname{clip}_{[0,1]}(I+\delta),\quad
\epsilon=8/255.
\]

image为官方resize后的[0,1] RGB 384×128；G内部用2I-1，IRRA内部仍使用官方mean/std。网络直接在384×128生成扰动，不进行AP的256→384映射。

接口：

ConditionalGenerator.forward(image[B,3,H,W], z_t[B,512], c_attr[B,512]) -> delta[B,3,H,W]

generate_once(...) -> {delta, adversarial, effective_delta}

约2,295,299参数。在线无momentum/sign/TTA/IG/optimization loop。

## 最小训练损失

Adam最小化：

\[
L_g=f_I(I_{\rm adv})^\top z_t,\quad
L_l=f_I(I_{\rm adv})^\top\bar v,
\]
\[
L_\delta=\operatorname{mean}[(\delta_{\rm eff}/\epsilon)^2],\quad
L_{TV}=\operatorname{mean}|\nabla_y(\delta_{\rm eff}/\epsilon)|
+\operatorname{mean}|\nabla_x(\delta_{\rm eff}/\epsilon)|,
\]
\[
L=L_g+L_l+10^{-3}L_\delta+10^{-3}L_{TV}.
\]

delta_eff=I_adv-I；上述项按batch平均。正dot被最小化，降低匹配；global/local共同使用一次IRRA image forward。IRRA参数requires_grad=false，但训练时保留对输入的梯度。optimizer只有G参数。

K=0时c_attr=0、local target=0，仍使用global分支训练/生成；不是临时给rank0定义score。unresolved、rank0、不可用representation仍由原scorer排除。此行为与旧attribute-only TTA的K0 clean passthrough不同，明确作为本次有global loss的G原型定义。

本用户新授权包括whole-caption global loss，旧TTA的“无caption loss”合同不被修改，也不应错误套用到本G prototype。

## 数据与来源

- 沿用v4固定PID split：原train54246 captions/8779 PID，validation13700 captions/2195 PID。
- 每张image取split中首条caption，再按固定SHA256排序；不按score或攻击效果筛样本。
- 默认手动准备2048 train images/1861 PID，128 validation images/121 PID；两组PID交集0，与正式test PID交集0。
- 新train条件各执行一次冻结Step5.1 clean backward；validation直接复用已锁SHA的Step7.4旧S_raw/AttributeInstance/w，不重新评分。
- 新train representation走原FrozenLocalization.attach(cache_index=None)已有fresh规则；validation继续使用旧缓存e。未重提全量block8 cache、未训练reader/basis。
- S参照图库仍为已核验的34054 clean train gallery。该图库包含内部validation images，因此是共享clean reference下的PID-disjoint G优化数据，而不是完全inductive gallery-disjoint设置。正式test不参与优化或条件准备。
- prepare保存每条condition tensor、全部原score/实例metadata、原source image SHA、选择序列、聚合权重和JVP FD检查。
- train和validation只读条件，不调用scorer/Top-K/JVP；source/cache/config修改会拒绝续跑。

## 验证结果与边界

37 CPU tests通过：形状/奇数尺寸、tanh与clip预算、两路条件生效、无batch依赖/无eval状态更新、单forward无autograd、聚合权重与JVP尺度、K0、原metadata不变、loss方向、TV/扰动单位、冻结toy-surrogate下仅G更新、默认不启GPU/训练flag、未知caption拒绝、安全路径/SHA等。

极短真实GPU smoke：
- 2 train + 2 validation，16 scored/1 rank0 excluded；新scoring backward2次，validation旧score复用2条。
- 仅2次G optimizer update，每次一次G forward、一次IRRA image forward；global/local诊断各额外对adv求一次梯度，不更新victim。
- 两步global输入gradient norm .2800/.2845，local .01290/.01235；G gradient norm .5307/.9399，condition MLP gradient norm .0015035。
- IRRA/reader冻结，106项source/cache/config SHA保持。
- JVP FD最大相对误差.00215747（.215747%）。
- train两步L∞ .005312/.013981，validation最大.029459，生成检查.027509，均小于8/255。
- 改caption/attribute condition后delta最大变化分别.00043169/.00032388；证明接线生效，不证明语义解耦。
- batch2 G前向短测平均2.316ms；峰值allocated约1343.5MiB；warm第二次更新.0483s，首步冷启动1.5968s。非正式训练性能benchmark。
- 独立CPU generate.py成功：一次G forward、victim_initialized=false、score backward/JVP/TTA/optimization均0；可重复推理exact。
- 首次smoke在加载模型前发生model.py与官方model包重名，已将新模块改名generator_model.py，失败日志/目录保留；该次无optimizer update。

smoke只证明可训练与单前向生成闭环。两步checkpoint不能用来评价G攻击效果，更不能继承TTA的98%mDR。默认lambda未校准；raw JVP local项尺度可能较小，当前梯度已接通但未证明有效权衡。聚合丢失逐实例顺序及显式空间控制，现有ontology/binding/leakage限制保持。本轮不搜索lambda/架构，也不宣称Top-K优于Random。

GPU已停止。outputs/conditions和outputs/train均不存在，未启动手动训练。正式服务器tracked/staged diff为空。

## 手动命令（Codex未执行）

在服务器终端执行以下一条后台命令，依次完成离线条件准备和10 epoch小型G训练：

~~~bash
tmux new-session -d -s tbps-image-g-v1 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 && /home/lzf/ldx/envs/tbps-image/bin/python -B -u prototypes/attribute_image_generator_20261014/prepare.py --allow-preparation >> prototypes/attribute_image_generator_20261014/outputs/prepare.log 2>&1 && /home/lzf/ldx/envs/tbps-image/bin/python -B -u prototypes/attribute_image_generator_20261014/train.py --allow-training >> prototypes/attribute_image_generator_20261014/outputs/train.log 2>&1'
~~~

也可分别运行prepare.py --allow-preparation与train.py --allow-training。默认配置2048/128、10 epochs、batch4、Adam lr2e-4/betas(.5,.999)，无augmentation、无AMP、无best-validation挑权重。正式训练耗时未实测，短smoke不能保证吞吐。

准备支持按逐条已完成条件恢复；训练支持epoch边界恢复（中断epoch未保存的更新需重跑），命令：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -B -u prototypes/attribute_image_generator_20261014/train.py --allow-training --resume >> prototypes/attribute_image_generator_20261014/outputs/train.log 2>&1
~~~

完成后，在固定第一条held-out validation条件上测试一次前向生成：

~~~bash
cd /home/lzf/ldx/projects/idea-TBPS-test1
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -B -u prototypes/attribute_image_generator_20261014/generate.py --checkpoint prototypes/attribute_image_generator_20261014/outputs/train/last.pt --cache prototypes/attribute_image_generator_20261014/outputs/conditions --row-id cuhk:8836:0 --output prototypes/attribute_image_generator_20261014/outputs/validation_example.pt --device cuda
~~~

generate默认读取该record对应image，也可--image提供字节SHA匹配的原图，--caption验证原caption一致。输出fp32 pt，不做PNG/JPEG量化；拒绝覆盖已有结果。

## 路径和完成标志

- 条件：outputs/conditions/{protocol.json,manifest.json,completed.json,records/,tensors/}
- 训练：outputs/train/{protocol.json,epoch_*.json,last.pt,last.pt.json,completed.json}
- 日志：outputs/prepare.log、outputs/train.log
- 已完成smoke：outputs/smoke/completed.json、outputs/smoke_conditions/completed.json、outputs/cpu_tests.log、outputs/standalone_inference.log
- 条件准备完成：CONDITIONS_COMPLETE + method_audit_passed=true
- 手动训练完成：TRAINING_COMPLETE + only_G_trained=true / IRRA_frozen=true / test_used=false
- 在线生成完成：GENERATION_COMPLETE + generator_forward_calls=1 / victim_initialized=false


## 2026-10-14 最小修复：checkpoint 事务与锁

本次只修改 train.py，新增 checkpoint_io.py 和 test_checkpoint.py；G、loss、条件、属性模块和正式源码不变。64项CPU tests通过（原37+新增27），包含4次真实子进程os._exit中断、写入故障、旧格式迁移、并发锁及Adam恢复后下一步逐值一致。本轮没有GPU计算或正式训练。

每个epoch先在 .last.pt.checkpoints/.pending-* 内写原格式 checkpoint.pt 与 checkpoint.pt.json，执行文件/目录fsync并核验SHA，再通过同文件系统目录rename提交为不可变 epoch_* 版本。所有完整旧版本保留；不完整pending版本不参与恢复。last.pt和last.pt.json为固定别名，通过单次原子切换 .last.pt.current 同时指向同一已完成版本。不要手工修改或删除隐藏的版本目录/指针。

--resume 扫描最近完整有效epoch，核验原payload/sidecar及来源，必要时修复last入口；写入失败不会覆盖旧epoch。修复前普通.pt+.json可以读取，首次新保存时无损导入legacy目录。SHA损坏允许明确警告后退回有效旧版本；来源SHA/配置/cache不匹配仍拒绝，不能靠回退绕过来源检查。

训练锁提前到protocol.json或任何训练产物的首次写入前；仅创建输出目录和锁文件是取得锁的必要先行操作。仍是epoch边界恢复，中断epoch尚未完整提交的更新需要重跑；首次epoch尚未提交时不存在可恢复权重，不声称支持step级恢复。

保留原GPU smoke与所有历史产物，不修改旧SHA记录。outputs/checkpoint_repair/completed.json精确锁定唯一train.py来源迁移、新IO/tests及CPU验证，并验证step/validate/epoch loop AST及模型/损失/条件源码保持，供训练smoke gate衔接。未知来源变化仍拒绝，无需重提条件或运行新的GPU训练。

原prepare、train --allow-training、train --allow-training --resume及generate --checkpoint .../last.pt命令均不变。
验证记录：outputs/checkpoint_repair/cpu_tests_verified.log、completed.json。


## 第一版G官方Test评测准备

10 epochs已由用户训练完成；新增g_test_*.py，94项CPU tests与两图/四Query smoke通过，完整Test未运行。正式协议、参考图库说明、输出路径与用户手动命令见README_TEST.md。原模型、损失、训练及generate.py保持。
