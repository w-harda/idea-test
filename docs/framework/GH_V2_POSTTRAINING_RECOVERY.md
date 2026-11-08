# GH-v2原60轮的后处理恢复（当前工具第二版）

本任务只复用原训练Attempt的产物。没有重跑60轮、重新训练G/H、重算训练JVP或重建历史RewardBank。

当前入口为`operations/gh_v2_recovery_v2_supervisor.py`，操作入口`operations/gh_v2_recovery_v2.py`，共用严格分项读取器`operations/gh_v2_recovery_stream.py`。恢复版本配置为`configs/framework/gh_v2_posttraining_recovery_20261108_v2.json`，执行源码Commit为`176571dedc0268d1856c4c1ed727c9db07215ff9`。

原158份生产源码、496份运行来源及冻结配置SHA保持；新增operations工具通过独立版本Manifest逐阶段核验。没有静默关闭SHA、来源或Manifest检查。原全局read_json的64MiB限制保持。

## 执行身份与状态

第二版控制目录：`/home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108_v2`。Supervisor PID/PGID/SID2487647、PPID1、无TTY。started.json为不可变单次身份；**本入口已经启动消费，不应再次启动或删除身份文件。**

```bash
cat /home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108_v2/status.json
cat /home/lzf/ldx/tmp/gh_alternating_v2_posttraining_recovery_20261108_v2/child.json
```

原工作树：`/home/lzf/ldx/projects/idea-test-tbps-gh-alternating-v2-20261107`。
原输出：`/home/lzf/ldx/outputs/tbps-framework-gh-alternating-v2-20261107`。
新训练CPU审计：原输出下`posttraining_recovery_20261108/train_cpu_audit_v2`。
六组正式Test：原输出下`test/round_10`至`test/round_60`，各自独立生成完整3074 Gallery/6156 Query。

阶段只包含：独立CPU训练审计 → 六组Joint Test及逐组CPU审计 → 六行指标/两张曲线/报告 → 历史保护复核 → Git发布并核对远端SHA。没有训练入口，没有重试循环。真实错误停止保留；资源只观察，共享GPU无VRAM floor，不操作外部任务。

最终阶段为`SUCCESS_POSTTRAINING_SIX_JOINT_PUBLISHED`；它只属于新后处理控制，不能改写原Supervisor的`FAIL_STOPPED_NO_RETRY`。实时状态及当前最终结果见[独立恢复报告](results/gh_alternating_v2_recovery/REPORT.md)。

## 大账本读取与CPU审计

原60份账本合计3890547998字节，第2轮首次超64MiB，超限轮为2/3/4/5/7/9/10。读取器在原ArtifactStore完成回执、Manifest和文件SHA验证后，按项完整解析G/H/conditions，保留其余有界字段；64Ki字符读取块、单项最多4Mi字符，完整消费到EOF。重复键、NaN/Infinity/浮点溢出、非法UTF-8、截断、尾随、非法分隔、字段结构错误全部拒绝。原文件不拆写、不重存。

CPU审计已PASS：60×2048完整覆盖，G484080更新、H122880访问/122220更新；原选中ID/S_raw/K+1、状态和父态声明链、所有H动作原Unicode回放/原候选合法性/Natural-K、当前Bank绑定、冻结隔离、原JVP/FD回执和六对Checkpoint身份核验。耗时157.251830秒，实际峰值RSS586027008字节（558.879MiB）。模型前向、反向、Optimizer更新、新JVP及CUDA均0。

43907个原输出/原Supervisor文件SHA及inode/size/mtime基线在审计前后及最终收尾复核。第1–59轮原始Bank、完整状态、G-rollout按原滚动策略已退休，使用封存账本/退休Manifest核验，不能宣称重读了不存在的Tensor。全部历史H sampled/greedy仍在账本，已逐条回放；第60轮完整Bank、条件和rollout直接读回原资产。

## H逐步与轮边界摘要域

第一版恢复工具Commit e2abc63在新增H逐步检查处13秒停止，未产生PASS或Test；其原源码、配置、7个控制文件及FAIL全部保持。错误来自工具把两种不同摘要混比：原生feedback_backend.state_sha使用(shape,dtype)，框架tree_sha使用(dtype,shape)。同一原始初始H的tree_sha为02dfb3a...，原生state_sha为9d607ba8...，均与各自原记录精确对应。

第二版使用冻结原state_sha验证整条H逐步/跨轮链，tree_sha验证轮边界链，并在初始、六对现有Checkpoint及最新完整状态重算双摘要桥接。未跳过连续性检查。第一版7个控制文件额外SHA/stat保护，每阶段核验。

第二版10项不同新增CPU测试和27项相关回归PASS；首版9项同名用例不重复计数。包含真实初始H双摘要校验，以及改变一个权重值必须改变原生摘要的回归。

## 评测与发布

评测复用原fresh_evaluation的完整Joint-only生成/评分数学。独立preflight仅构建架构并严格加载对应已训练权重，不创建训练初始状态、Adam或阶段RNG。内存中只将训练审计证明路径显式指向新审计；原配置文件及SHA不改。

每组用自己的G/H权重、图像和文本，原生IRRA FP32完整Gallery全部同PID正例检索。仅Joint，图像L∞≤8/255；文本沿用原Natural Dynamic-K、每原词位最多1字符，STOP/NOOP合法，无WordBudget1。原CPU Test审计独立重读RGB预算、Unicode和分数，并用原生ranking复算四项指标。没有新增Clean/单模态/跨模型/跨数据集实验，Test不反馈训练或选点。

六组审计全部PASS后，独立`docs/framework/results/gh_alternating_v2_recovery`发布METRICS.csv、R@1/mAP曲线PNG/SVG、报告和来源证明，自动commit/push并核对远端SHA。权重、Ledger、原始数据、Tensor和大型日志不发布Git。

## 六组全部PASS后的独立绘图发布

第二版后处理已完成六组正式Test与六组CPU审计。其final_report阶段因tbps-image环境未安装matplotlib而停止，原FAIL/status/日志保持；没有重试任何评测。收尾使用独立仅发布入口operations/gh_v2_posttraining_publish.py，版本configs/framework/gh_v2_publication_20261108.json，控制/home/lzf/ldx/tmp/gh_v2_publication_20261108。

发布器复用原六结果校验和文档写作逻辑，仅将绘图交给既有deepmisranking Python3.9.25/Matplotlib3.7.5/NumPy1.23.5解释器。该子进程只读取SHA锁定的六行CSV并画图，不导入TBPS或模型；缓存全部放在新发布控制目录，不安装依赖，不修改任何科学环境。原158源码/496来源/训练配置及后处理工具版本保持。第二版后处理整个控制目录另有逐文件SHA/stat保护。

绘图CSV新增4项门禁测试PASS，验证原值不缩放、SHA不匹配、端点错序/重复、非有限/越界数值和多余列拒绝。发布成功状态为SUCCESS_SIX_JOINT_RESULTS_PUBLISHED。它与原训练Supervisor和第二版后处理收尾FAIL分开记录，不把旧FAIL改成成功。

仅发布器生成曲线/报告后，Git门禁发现Matplotlib SVG行末空格，原发布FAIL仍保留。最终入口operations/gh_v2_publication_finalize.py只核对现有六结果、归档原SVG、清除派生SVG行末空白并验证XML语义等价，PNG/指标SHA保持，不重绘或运行科学计算。控制/home/lzf/ldx/tmp/gh_v2_publication_finalize_20261108；最终成功状态SUCCESS_ALL_SIX_JOINT_PUBLISHED。
