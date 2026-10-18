# ADR-0008：SGA-TBPS Image-only白盒适配与冻结迁移实验分型

状态：本轮用户批准实现和10图Smoke；Full尚未执行。

## 背景与决定

此前G-v2/H-v1/B3是不使用目标反馈的冻结攻击产物；本轮SGA获准使用每个受害者的参数、输入梯度及图像对应原始Caption。为避免混淆，SGA产物按whitebox_target分别隔离，不能把IRRA扰动用于RaSa/APTM后称为白盒。

固定官方SGA f49e6926cd13148e49bdbc2c3e537660e096230c。直接复用官方ImageAttacker.get_scaled_imgs；保留随机起点、多尺度0.5/0.75/1.25/1.5、Gaussian std0.05、BICUBIC resize/reverse、原图尺度、集合梯度sum、10轮、0.5/255步长和原图相对投影。只把epsilon从原2/255调整为统一8/255，并对FP32端点内舍入满足严格预算；没有动量或新的正则。

每图Caption集合固定为原annotation中paired_image_id相同的全部Caption，按官方Query顺序。不是全同PID文本，不根据匹配分数挑选，不修改任何Query。Image-only不执行原SGAttacker的两阶段BERT文本扰动，因此明确命名SGA-TBPS White-box适配，不称原论文完整多模态SGA复现。

攻击目标：
- IRRA：负的原生单位化图像/原文表示点积之和，与官方图像集合对齐目标同型。
- RaSa：负的原生ITM正匹配logit之和，梯度穿过视觉与融合层。
- APTM：负的原生ITM正匹配logit之和；这是可微局部代理目标，**不等于**完整Test的Top-128+全矩阵minmax+0.002 global最终排名分数。不把IRRA cosine冒充融合白盒目标。

RaSa/APTM的离散候选选择与APTM的完整集合归一化不进入逐图优化；这是显式适配差异。最终排名完全复用原生Adapter/Scorer/Engine，不修改候选、归一化或tie规则。若未来研究要更换攻击目标，应另行冻结配置，不能据Test攻击结果临时调目标或权重。

## 计算与资产边界

逐尺度microbatch先求目标对增强图像的输入梯度，再经官方增强一次VJP汇总到原扰动；保留相同数学集合sum，不同时保留5份大模型图。CPU独立官方循环对照检验；真实GPU与原批次5路实现的bitwise等价不在本轮声明范围。

新SGA缓存使用ArtifactStore和已有image_checks；原MaterializedStore及Plan只支持固定G/H，因此不放宽原类型，SGA另存受害者专属像素协议、Claim、实际FP32 RGB、delta与来源证明。只从generation_v3读取original RGB，不使用其攻击像素；只读历史Clean文本表示，不使用H文本。

先生成全部固定Gallery，再运行目标原生评分；Engine不持有攻击回调。白盒权限只属于新SGASurrogate，旧G/H/B3能力及历史结果完全不变。失败保留Claim/pending/错误，不自动重试；Full须用户手动显式启动。
