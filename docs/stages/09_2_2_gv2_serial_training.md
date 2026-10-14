# G-v2 Phase 2.2：K轮逐属性串行训练机制

2026-10-15：方案A的单样本训练闭环与CPU验证完成。新增36项加原111项回归，共147项全部通过；当前阶段按用户授权提交并push，完成后等待独立源码审查。不进入Phase 2.3，不启动正式训练、GPU、2176条完整缓存或Gallery评测。

## 位置与文件

工作树：`/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/`。

分支`codex/gv2-per-attribute-cache`，本阶段基准`13d073ef90eda9e34be9a018b5a6af23e990f6f6`。所有新增实现位于`prototypes/attribute_image_generator_v2_20261014/`：

| 新增文件 | 职责 |
|---|---|
| serial_training.py | LossWeights与train_single_sample方案A闭环 |
| test_phase22.py | 36项CPU合成样本测试 |
| README_PHASE22.md | 接口、算法、日志与运行边界 |

本中文报告从Windows同步进入发布工作树，纳入同一提交。G-v1、Phase 1数据协议、Phase 2.1源码及原实验目录不修改，不自动镜像两工作树。新训练模块只依赖标准库、torch及G-v2自身schema/预算/loss，不导入G-v1或integration，不调用评分、Top-K或JVP。

## 输入与可配置权重

`train_single_sample(generator, encoder, optimizer, image, condition, weights=None, observer=None)`

image为单张[1,3,H,W]图像；condition为已读取的Phase 1 detached CPU监督dict，按原schema验证。函数直接读取原selected_instance_ids、attribute_jvps、z_t及v_local，保留原S_raw降序，不重新选属性。调用方负责真实图像与缓存身份配对，本阶段没有数据加载器、官方IRRA接入或正式运行CLI。

使用Phase 2.1 ImageOnlyGenerator与原budget_projection、losses。encoder由调用方提供可微nn.Module，输出[1,512]；函数设置eval、冻结参数并清理已有参数梯度。optimizer必须且只能持有G的全部可训练参数。G/encoder/image的dtype/device须匹配。

`LossWeights(lambda_global=1, lambda_local=1, lambda_retain=1)`支持有限非负配置。在训练模块中组合Phase 2.1返回的原始分项，不修改旧loss函数，也不加入扰动/TV正则。

## 方案A实际行为

原图先detach并clone，作为所有轮次固定预算参考。每个属性轮独立执行：

1. optimizer.zero_grad(set_to_none=True)。
2. 用当前G对上一轮detach图像做一次forward，生成新增delta。
3. 累加后使用相对原图的8/255累计投影。
4. 冻结encoder计算当前feature，保留对输入和G的梯度。
5. 计算加权Global+当前Local+上一轮Retain，独立backward。
6. 检查梯度有限、encoder冻结后立即optimizer.step，再检查G参数有限。
7. 返回本轮实际图像detach，以及由本轮feature与本轮JVP得到的detach参考分数。

下一轮明确使用更新后的G参数和上一轮图像；没有retain_graph/create_graph，也不跨轮传递旧feature/loss。Retain只使用紧邻上一轮的JVP及上一轮实际对抗图像匹配分数，不使用clean分数、当前属性参考或历史属性集合。参考复用同一次encoder forward，不在step后重新生成上一轮图像。

K轮结束后，重新把原图输入已更新G，执行一次Global+缓存聚合v_local联合更新。K=0跳过属性循环，仅执行Global联合步。支持全部K=0–7，不设cap4；原schema允许的更大K也没有新增人为截断。

每次调用的G forward、encoder forward、backward和optimizer.step均严格K+1。返回serial_adversarial、joint_adversarial、records和调用计数；两个图像均detach，分别为对应更新前真实前向所得图像，不在最后step后再运行G。

## 日志与保护

每次更新记录phase/index、当前/上一轮属性ID、Global/Local/Retain与加权total、上一轮参考与本轮参考分数、G梯度L2、累计L∞和有限梯度状态，并区分：

- budget_saturation_ratio：输出位于有效预算端点的比例，排除只由像素边界激活的端点。
- pixel_saturation_ratio：输出为0或1的比例。
- projection_endpoint_ratio：输出位于任一投影端点的比例。
- strict_projection_clipping_ratio：proposed严格越界的比例，对应硬clamp零导数分量。

精确端点使用PyTorch导数1约定，因此端点比例不能直接解释为零梯度比例。投影沿用Phase 2.1向内舍入，不放宽预算。全饱和轮允许有限零梯度，仍执行该轮step，不加入STE或跳步规则。

observer仅在更新后获得detach clone快照，不保留训练图。非有限图像/loss/梯度会拒绝；encoder参数版本、参数梯度和buffer状态均核验。optimizer若产生NaN，在更新后报错，不承诺回滚或轮内恢复；本阶段没有checkpoint/epoch/batch调度。

## CPU验证结果

| K | 属性更新次数 | 联合更新次数 | 实测optimizer.step | 常规探针G参数变化次数 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 |
| 1 | 1 | 1 | 2 | 2 |
| 3 | 3 | 1 | 4 | 4 |
| 7 | 7 | 1 | 8 | 8 |

新增36项测试另覆盖全部K0–7、SGD/Adam、真实zero→G→encoder→backward→step调用顺序、detach输入、下一轮参数等于上一step后参数、正确Retain JVP与分数、联合原图重启、可配置权重、K0 Global-only及异常拒绝。

弱引用测试确认旧G/encoder前向张量在下一轮开始前释放；观察快照与返回值无计算图。encoder参数/buffer不变、参数梯度为0，G各正常轮可以更新；正常案例的图像、loss及梯度均有限。原111项回归通过，Phase 1和Phase 2.1文件不改。

K7饱和压力用例中，第2–7轮预算/严格裁剪比例均1，G梯度为0，仍逐轮调用step；该用例最后联合步从原图开始，梯度恢复非零。压力与常规探针最大累计L∞=0.0313725471496582≤8/255=0.03137254901960784，无预算越界。这只证明CPU机制行为，不是正式IRRA训练效果或饱和比例结论。

新增score backward、Top-K、JVP均0，未加载官方IRRA，CUDA未初始化。47份保护文件SHA保持，涵盖G-v1源码/config/权重/Test摘要、原Phase 1缓存和Phase 1/2.1实现。

## 证据、命令与发布范围

服务器本阶段目录`outputs/phase22/`保留：cpu_tests.log、summary.json、protected_sources.json、cpu_tests_initial.log。首轮失败为测试辅助optimizer方法名与PyTorch state字段冲突，修正辅助命名后147项通过，生产机制未因此修改。通过日志SHA256：

`fd65e93f99a2b16b0f871a74984bead4eec814559b9b57bfe11d6bec70b3445c`

在服务器G-v2目录运行：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase22 test_phase21 test_conditions -v
```

提交仅包含3个新增文件与本中文报告；不包含数据集、权重、条件缓存、outputs、临时文件或Windows状态管理文档。阶段结束后停止，等待独立审查；未准备完整缓存、未运行正式G训练/Gallery评测、未进入Phase 2.3。
