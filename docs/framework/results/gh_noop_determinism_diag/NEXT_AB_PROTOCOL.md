# 下一版严格A1/B数值对照候选协议（未执行）

本文件是待审查研究设计，不修改原smoke_v2。当前NOOP GPU诊断因资源不足未运行，不声称下列容差已获实测验证。

## 一、先完成原环境诊断

保持独立noop_determinism_diag_v1四次更新上限。记录G/Adam/输入/条件/RNG exact控制；对G输出→投影→IRRA→Loss→IRRA输入梯度→G输出梯度→各参数grad→Adam逐阶段比较。
G叶模块输入/输出反向信号用于缩小bilinear、InstanceNorm、Conv等区域；仅定位模块范围不能自动等同于具体CUDA kernel。
有/无Hook时CPU数学exact不等于GPU时序完全相同；Hook会同步与复制数据，应在结论中说明它可能影响并行调度。
若4次均exact，只能说明记录条件下可复现，不能断言原长流程不存在噪声或状态问题。

## 二、独立确定性环境（必须另行授权）

新的独立Attempt、独立配置和显式调用预算，在CUDA初始化前设置适用CUBLAS_WORKSPACE_CONFIG=:4096:8；
设置torch.use_deterministic_algorithms(True)、cudnn.deterministic=True、benchmark=False、关闭TF32，其他FP32/math、模型、Loss、Adam保持。
PyTorch 2.5.1的bilinear确定性路径采用decomposition，虽数学操作意图不变，浮点规约与速度可能改变，必须如实列为运行环境差异。
任何不支持算子报错都停止；不得warn_only忽略、替换插值、删除反向、修正梯度或修改Loss。
本轮未执行这套环境，不增加当前四次之外的GPU调用。

## 三、相同父态的因果干预

共同初始完整状态S含G/H参数与buffers、双Adam、双阶段RNG及sample cursor。
NOOP负控独立执行/比较，不能使其微小分叉污染后续正控：
公共NOOP更新只提交一次得到S0；A1/B均从同一个S0复制恢复。首非NOOP的G_before及Adam/RNG必须exact。
H rollout使用同一H状态和sampler，固定相同动作；A1使用原条件，B只替换已核验的新语境z/JVP。
不能为了让条件不同而选择样本或改动作。若固定样本产生NOOP，如实停止正控验证。
既要保存首次反事实梯度/更新，也要保存持续训练轨迹；之后的自然分化不能冒称始终相同父态。

## 四、预先冻结的候选数值合同

以下只是结合既有probe 1e-6合同的起始提案，尚未由当前四次诊断验证，不可用于重审旧Smoke。
正式新实验前应单独批准、冻结；若负控超界则停止，不可事后放宽。

| 层级 | 候选条件 |
| --- | --- |
| 输入/条件/父G/父Adam/RNG/step/身份/shape/dtype | exact，SHA和Tensor逐值一致 |
| NOOP每参数grad | 每元素abs差≤1e-6+1e-5×abs(reference)，并记录整体relative L2；整体候选上限1e-5 |
| NOOP更新后参数 | 每元素abs差≤1e-7+1e-6×abs(reference)，整体relative L2候选上限1e-6 |
| Adam exp_avg | 每元素abs差≤1e-7+1e-5×abs(reference) |
| Adam exp_avg_sq | 每元素abs差≤1e-10+1e-5×abs(reference) |
| Adam step、参数组及非Tensor状态 | exact |

零参考范数时使用绝对误差判据并显式标记，不用除0或丢弃零Tensor。必须保存每Tensor误差及分母。
正控除条件/来源真实不同外，梯度max_abs应超过max(1e-5,10×预先测得并冻结的同臂/NOOP最大噪声)，并报告L2/relative L2。
这是区分数值噪声的工程门槛，不是统计显著性检验；单次不能给出置信区间。
若原/新条件效果低于此门槛，标记未验证，不增加K、Loss或重选样本。

## 五、独立G→H归因

固定同一H/Adam/RNG和同一sample/greedy文本；只切换旧G与更新G生成的合法RewardBank。
图库必须绑定各自G SHA、完整2048 Train顺序、原IRRA特征空间；不可混合旧/新行。
先比较同文本Reward和Advantage，再比较SCST梯度，最后比较完整Adam更新。
零梯度且参数移动须分离为Adam历史动量证据，不能全部计为新Reward学习。
冻结IRRA/reader、K、原Reward公式，不添加Joint奖励。允许缓存的仅为身份完整且未失效的文本特征。
未来完整Smoke仍需单独授权，禁止本文件触发任何训练或评估。
