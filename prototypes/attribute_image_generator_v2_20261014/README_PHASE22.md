# G-v2 Phase 2.2：方案A单样本K轮串行训练机制

2026-10-15。基准Commit：13d073ef90eda9e34be9a018b5a6af23e990f6f6。
本阶段只有机制实现与CPU合成验证，无正式训练、GPU、完整缓存、Gallery评测或Phase 2.3。

## 新增文件

| 文件 | 职责 |
|---|---|
| serial_training.py | LossWeights配置与train_single_sample闭环 |
| test_phase22.py | 36项新增CPU测试 |
| README_PHASE22.md | 接口、算法、日志与验证说明 |

不修改Phase 1 schema/cache_io或Phase 2.1模型/预算/loss。
依赖仅为标准库、torch及G-v2自身condition_schema、budget_projection、losses。
不导入G-v1/integration/selection_audit/attribute_tta，不计算score、Top-K或JVP。

## 接口

train_single_sample(generator, encoder, optimizer, image, condition, weights=None, observer=None)

- image为单样本[1,3,H,W]、[0,1]；原图detach并clone作为固定预算参考。
- condition为Phase 1已读取的detached CPU监督dict，使用原schema校验，不改协议。
- generator使用ImageOnlyGenerator；optimizer必须且只能包含G的全部可训练参数。
- encoder由调用方提供可微nn.Module，输入图像输出[1,512]；本函数设置eval、冻结参数并清理旧参数梯度。
- 模型/image须dtype/device匹配；本阶段只在CPU验证。
- 调用方负责真实图像与缓存样本身份配对。本阶段不实现数据加载器或官方IRRA接入。

LossWeights(lambda_global=1, lambda_local=1, lambda_retain=1)允许有限非负配置。
复用Phase 2.1原始分项，按配置组合，不引入扰动/TV正则。

## 方案A顺序

从缓存的selected_instance_ids/attribute_jvps按原S_raw降序依次取属性，不重选。
每轮：

1. optimizer.zero_grad(set_to_none=True)。
2. G(current)生成新增delta，add_projected_delta(original,current,delta)投影。
3. 冻结encoder保持输入梯度，取得当前feature。
4. 原Global+当前Local+紧邻上一轮Retain，按LossWeights组合。
5. total.backward()，检查有限梯度和冻结状态。
6. optimizer.step()立即更新G，再检查参数有限。
7. 把本轮实际图像detach交下一轮；参考分数由同一次feature与本轮JVP取得并detach。

下一轮使用已更新G参数。没有retain_graph/create_graph，不把旧feature/loss带入下一轮。
Retain只用j-1的JVP及j-1实际对抗图像分数，不使用clean参考、当前属性参考或历史属性集合。
取得参考不增加G/encoder forward，也不在更新后重新生成上一轮图像。

完成K轮后，明确重新以original作为G输入，执行一次Global+缓存v_local联合更新。
K0跳过属性循环，联合步Local为0；总G/encoder forward、backward及optimizer.step都是K+1。
支持实际变长K=0–7，不设置cap4；只受原缓存schema约束。

返回serial_adversarial、joint_adversarial、records、K与调用计数。
返回图像均detach，分别是串行末轮/联合步更新前真实前向所得，不在最后optimizer.step后再推理。
当前无epoch/batch调度、checkpoint恢复或正式运行CLI。

## 日志与预算

每次独立更新记录phase/index/当前ID/上一轮ID、Global/Local/Retain、加权total、
上一轮参考分数、本轮留给下一轮的分数、G梯度L2、累计L∞及有限梯度状态。
预算始终相对original，复用Phase 2.1向内舍入投影，不增加容差。

- budget_saturation_ratio：输出落到有效预算端点的比例，排除只由像素限制激活的端点。
- pixel_saturation_ratio：输出为0或1的比例。
- projection_endpoint_ratio：输出位于投影区间任一端点的比例。
- strict_projection_clipping_ratio：proposed严格越界的比例，对应clamp零导数分量。
- 精确端点采用PyTorch导数1约定，不能把所有端点比例都称为零梯度比例。

observer只在更新完成后提供detach clone快照，不保存训练图。
非有限loss/梯度/图像会拒绝；optimizer若已写入NaN则更新后报错，不承诺回滚。
冻结编码器参数版本、参数梯度与buffer状态均检查；未接入resume或事务恢复。

## CPU验证结果

147项通过：新增36项+原111项回归。
K=0/1/3/7分别严格1/2/4/8次更新；另覆盖全部K0–7及Adam。
核验真实调用顺序zero→G→encoder→backward→step、输入detach、下一轮参数等于上一step后参数、
Retain引用/stop-gradient、联合重新输入原图，以及无旧前向张量进入下一轮。
模拟编码器参数与buffer保持；G各步均可正常更新，所有正常案例loss/梯度/图像有限。

标准合成探针K0/1/3/7的G参数变化次数也分别1/2/4/8。
饱和压力用例K7：第2–7轮预算/严格裁剪比例均1且G梯度0，仍各执行一次step；
最后联合从原图开始，梯度恢复非零。未引入STE、跳步或替代梯度。
压力与常规探针最大L∞=0.0313725471496582≤8/255，无预算越界。
新增score/Top-K/JVP=0，官方IRRA加载0，CUDA未初始化。
47份基线文件SHA保持，包含G-v1、原Phase 1缓存及Phase 1/2.1实现文件。

~~~bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase22 test_phase21 test_conditions -v
~~~

源码在服务器发布工作树的G-v2目录；证据仅在本目录outputs/phase22/：
cpu_tests.log、summary.json、protected_sources.json及cpu_tests_initial.log。
首轮失败仅为测试辅助optimizer方法名与PyTorch state字段冲突，已改辅助方法名，生产机制不变。
测试不是实际攻击效果或正式训练稳定性证明。当前阶段可按用户授权commit/push，完成后等待独立审查。
