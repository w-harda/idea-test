# 双向靠近KL版本：从零60轮与六组IRRA Full执行协议

用户已明确授权启动Full。沿用此前从零60轮、每10轮配对G/H及CUHK-PEDES × IRRA Joint-only的实验范围。科学目标继续为48ff8ab实现的吸引对方对抗表示、排斥自身干净表示；温度0.07、双权重1，G原Local/Retain/K+1、H三头SCST、原Natural-K和严格8/255保持。

## 单次执行链

1. CPU来源/初始化预检与资源准入；共享GPU只观察，无最低空闲VRAM门槛。
2. 八样本真实训练Smoke，独立CPU审计。
3. 十图/十Query Joint链路Smoke，独立CPU审计。
4. 正式训练至10/20/30/40/50/60轮，各节点完整边界保存G/H、双Adam、双阶段RNG和当前Bank。
5. 对应轮次CPU前缀审计PASS后，使用配对Checkpoint独立生成完整3074 Gallery/6156 Query攻击输入，原生IRRA Joint评分并CPU独立审计；随后恢复下一十轮。
6. 汇总六组R@1/R@5/R@10/mAP、R@1与mAP曲线及来源证据，自动发布GitHub。

退出码3仅在预设10/20/30/40/50轮完整边界接受。任何真实错误停止且不重试。旧GH-Cooperative-v1和其他实验不修改。

## 独立CPU审计

在原每轮连续状态、G/H更新数、当前Bank绑定、JVP/FD、Unicode编辑、冻结隔离、Checkpoint身份和最新图库像素预算核验上，新增对全部G属性/联合步及合格H步的KL独立重算。

GPU执行额外封存每个G步已计算的512维图像特征及对应H文本特征、H原文/sample/greedy特征，以及每轮2048×512当前G Bank特征。复制detach值不增加模型前向、Backward、JVP或Optimizer更新。

CPU使用FP64、共同候选矩阵及手写KL求和，与实际优化记录逐项核对pull/push、干净基线、Global、SCST奖励与loss，并核验H配对图像特征与当轮Bank逐值一致。数值重算使用预先固定atol=5e-5/rtol=2e-4覆盖CPU/GPU FP32舍入差；图像L∞预算容差仍为0，SHA和来源核验仍为exact。

所有大JSON数组按≤4MiB分片，继续使用原严格JSON读取器。全局64MiB上限不修改。每轮旧模型/图像Tensor在下一完整边界发布后按本次目录退休；全部账本分片、Bank特征、退休Manifest、十轮配对评测权重保留。

## 路径及用户检查

输出：`/home/lzf/ldx/outputs/tbps-framework-gh-attraction-v1-20261109`。

监督控制：`/home/lzf/ldx/tmp/gh_attraction_v1_20261109`。

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attraction-v1-20261109/operations/gh_attraction_v1_status.py"
```

该命令只读查看阶段、轮次/游标、当前日志末尾及已审计正式节点，执行一次退出。最终成功状态为SUCCESS_60_ROUNDS_SIX_JOINT_TESTS_PUBLISHED，真实失败为FAIL_STOPPED_NO_RETRY。

本文件为启动前协议；实际启动身份由后续启动回执记录。助手核验启动成功后停止监控，由用户检查进度。

## 启动前验证与报告依赖

30项CPU测试PASS，覆盖生产G的K0/1/3/7记录独立重算、H三头SCST真实CPU更新与当前G绑定、篡改拒绝、分片及边界退休/恢复、原生Joint评分和六节点报告。Smoke与Train两个scope来源预检PASS；这些不代替真实GPU Smoke或正式攻击结果。

共享训练环境未安装新依赖。报告阶段单独调用`/home/lzf/ldx/envs/apattack/bin/python`的既有Matplotlib 3.7.5，配置锁定解释器、Matplotlib入口及绘图脚本SHA。该子进程只读取已审计METRICS.csv并绘图，不进行模型/训练/指标计算。SVG仅去掉行末空白，检查路径指令及XML内容保持一致。完整报告测试已实际验证PNG/SVG生成。
