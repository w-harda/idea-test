# GH-Adversarial-Attraction-v1：G1→H1→G2双向吸引与自身排斥

用户本轮要求先实现简单代码：G1攻击图像指导H1，使攻击文本靠近攻击图像、远离原文；H1攻击文本再指导G2，使攻击图像靠近攻击文本、远离原图。本版本独立于已启动的GH-Cooperative-v1，旧源码/配置/产物与控制进程保持。

## 目标定义

继续使用共同Train图像候选分布。C为原固定2048张干净Train图像的归一化IRRA特征，候选身份/顺序由原训练catalog和clean receipt核验。

`P(z)=softmax(normalize(z) @ normalize(C).T / temperature)`。

`D(a,b)=KL(stopgrad(P(b)) || P(a))`，a为当前待优化表示，b为冻结参考。默认temperature=0.07、pull_weight=push_weight=1，属于本次实现的初始配置，尚无真实效果调参。

记u0、ua为干净/攻击图像特征，v0、va为原文/攻击文本特征。

G最小化的Global目标：

`L_G_global = pull_weight * [D(ua, va) - D(u0, va)] - push_weight * D(ua, u0)`。

H最大化的SCST奖励：

`R_H = pull_weight * [D(v0, ua) - D(va, ua)] + push_weight * D(va, v0)`。

第一项衡量相对干净基线与当前对方攻击表示靠近的收益；第二项奖励远离自身干净表示。双方为NOOP时对应奖励/Global均为0。G使用奖励的负值作loss；H通过原SCST的采样−贪心优势更新。

当前变体使用新Global及新SCST奖励。原R_sim、R_rank仅保留诊断，不加入新奖励，以免其降低配对相似度的方向与本次吸引目标相互覆盖。上一版四模式synergy、逐编辑额外loss、残余权重及多描述项不叠加到这个简单变体。属性Local和前一步Retain仍沿用原值，G仍为原K+1次更新。H的属性/位置/替换三个头及SCST单次更新保持。

## 双向调度与身份

1. 初始G1使用当前H0条件训练。
2. 完成G1后，以G1的状态SHA刷新全部2048张Train攻击图库。
3. H1的吸引目标是该图库中自己的配对攻击图像，排斥目标是自己原文的新编码。旧Bank身份或G参数已经变化时拒绝更新H。
4. 下一轮按更新后的H1重新生成文本及所需语境/JVP，然后训练G2。

继续复用`tbps.alternating.configured_training.run_rounds`，整轮先G后H。`tbps.attraction.backend.AttractionBackend`提供原后端同名接口，`loaded`提供显式能力授权后的原生模型装载接口。它们供后续真实执行入口接入；当前operations只提供CPU预检，不提供新Supervisor或可绕过审计的60轮训练入口。

G日志记录teacher_H_state、condition_identity、固定候选anchor_sha256；H日志记录teacher_G_state和bank_sha256。全局项分别记录pull_kl、push_kl、基线与改善、当前表示对干净/对方表示的余弦相似度。

## 预算、冻结与推理

图像沿用原相对原图的FP32向内舍入投影，实际L∞≤8/255。文本沿用原Natural-K、Unicode合法候选、每属性最多一次、每个原词最多1字符、共享词最多一次、STOP/NOOP。H仍为原字符策略，当前实现没有加入逐词候选枚举的新训练监督。

G推理只读图像，H推理只读Caption及Caption-only属性。干净图像、当前对方攻击特征、图库与配对信息只进入训练损失/奖励绑定。IRRA/Reader及对方模型在相应阶段冻结；原prototypes、alternating、cooperative源码不修改。

## 验证范围与研究含义

CPU验证覆盖吸引/排斥的梯度符号、KL方向、NOOP精确0、共同候选旋转不变性、目标与锚点stop-gradient、实际原G K0/1/3/7接口、8/255投影、原三头H单次SCST和Unicode预算，以及两轮G1→当前Bank→H1→H1新语境→G2控制流。编码器在测试中采用明确CPU模拟特征，H和原训练机制使用冻结原实现；测试数值不是正式攻击结果。

两侧对抗表示靠近可能保留正确图文配对，远离干净表示本身不意味着Joint检索被破坏。后续真实实验需要同时检验对齐变化和完整Gallery排名，比较Clean、单模态与Joint实际R@1/R@5/R@10/mAP。

本次仅实现及CPU验证，没有启动新GPU实验、60轮训练或Test，没有检查旧后台实验进度。

CPU回归最终30项PASS（17项新增、13项既有机制回归）；4项绑定旧配置的测试未运行，新变体由两套真实CPU预检完整核验固定2048成员、Canonical及来源SHA。Smoke/Train预检均为GH_ATTRACTION_CPU_PREFLIGHT_PASS。详细证据见GH_ATTRACTION_V1_CPU_EVIDENCE.json。

## CPU检查命令

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-gh-attraction-v1-20261109
/home/lzf/ldx/envs/tbps-image/bin/python -B operations/gh_attraction_v1.py --scope smoke
/home/lzf/ldx/envs/tbps-image/bin/python -B operations/gh_attraction_v1.py --scope train
```

两条命令只读核验全部固定来源和2048成员，建立原G/H CPU初始化对象，禁止CPU预检进入CUDA；不创建实验Attempt。
