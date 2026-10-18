# ADR-0009：官方Baseline复现与额外白盒适配分开验收

状态：用户明确纠正SGA复现目标；本轮完成静态审查，官方算法尚未实际复现。

## 背景

此前SGA-TBPS White-box保留部分增强但修改了权限、完整图文流程、预算及融合损失。Full/缓存审计PASS只证明已实现实验的执行和结果完整，不能证明与原论文算法一致。原ADR-0008及旧结果保留为当时白盒适配设计快照。

## 决定

- SGA-TBPS White-box归类额外白盒适配实验，保留全部源码、结果、审计及失败证据；不作为官方SGA Baseline，不覆盖其原文档/CSV。
- 官方SGA固定f49e6926cd13148e49bdbc2c3e537660e096230c，完整调用SGAttacker的文本→图像→文本、原损失/候选/更新/参数；官方源模型生成一次，黑盒迁移三个目标。
- 主方法为G-v2 + H-v1 Joint；正式比较须区分source、Test配对访问、2/255与8/255、词/字符预算及Matched Clean，不以攻击强弱替代忠实性。
- Baseline默认复现原算法；需要算法改动时先报告障碍，另行授权并单独命名适配，不借接口接线新增Loss或目标反馈。
- 方法忠实性先于单元测试/Smoke/Full；通过存储SHA和排名审计不自动升级原论文复现状态。

## 影响

当前只写审查/分类文档，不修改旧算法或新建训练。源ALBEF权重本地未就绪、原requirements环境及输入桥/Matched Clean尚待冻结；CLIP权重本地字节可用，但有crop与官方长度调用差异。具体证据和最小方案见../framework/SGA_OFFICIAL_REPRODUCTION_REVIEW.md。

本ADR修正后续Baseline的使用和验收口径，不撤销历史白盒实验的数据完整性，也不把它重签为官方SGA结果。
