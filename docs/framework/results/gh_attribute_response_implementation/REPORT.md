# GH属性更新收益引导：代码与CPU验收

日期：2026-11-16。状态：实现及CPU验收完成；GPU/Smoke/正式训练/Test均未运行。正式指标0行。

## 已实现

- G：保持原Top-K和K+1正式更新，新增有限分支前瞻收益蒸馏及均值为1的属性Local权重。
- H：保留三头SCST，新增完整合法位置/字符/NOOP条件优势和属性头机会监督。
- 双向评价器：共享语义编码、历史GRU、G→H及H→G收益头、独立CPU Adam及有界Replay。
- 语义属性记忆：跨样本共享，不以PID/row_id/图像路径作为键；记录模型轮次并拒绝逆版本更新。
- 前瞻隔离：克隆模型及完整Adam，异常退出也验证正式模型/梯度/模式不变，并恢复Python/NumPy/CPU/CUDA及采样器RNG。
- 主线：原Push-pull、三视图增强、完整当前G奖励Bank、原Unicode预算及严格8/255。
- 可选同G梯度协调默认关闭。
- 完整训练/测试入口：从零60轮，每10轮配对G/H及完整教师Checkpoint、CPU前缀审计、原生Joint Full/CPU审计、表格及曲线。默认GPU授权关闭。

## 验证

36项新增CPU测试PASS，包含：

- 配置、语义共享记忆、容量/版本与反馈隔离、双向回归及精确恢复；
- 虚拟Adam/RNG/梯度/模式隔离及异常退出；
- 原K=0/1/3/7的K+1更新，三视图梯度/严格预算，梯度协调开关；
- 真实H三头、BPE/Unicode接口及完整合法局部候选，条件概率/奖励/优势/损失FP64复核；
- 真实CPU H与合成图像编码器的双向虚拟Adam前瞻、实际生产G/H接线及扩展数值审计；
- 10个完整边界、旧IO退休后的反事实特征保留、G/H与教师导出、损坏拒绝；
- 原生Joint评分fixture、六节点报告及实际PNG/SVG曲线生成，缺失审计结果拒绝发布。

最终测试耗时见SOURCE_EVIDENCE.json指向的原始日志；一次kornia弃用警告，不影响通过。
Smoke/Train双scope的CPU来源预检PASS：完整2048 Train身份、213份框架producer、真实CPU导入闭包与seed42空Adam初始化；CUDA均false。
默认Supervisor拒绝GPU执行的负向检查PASS；没有创建实验输出或启动控制目录。

最终CPU日志：/home/lzf/ldx/tmp/gh_attribute_response_cpu_20261116/pytest_1794794895914362629.log。
双scope报告及拒绝检查在同一CPU证据目录；失败的早期测试日志保留，不覆盖。
Python语法及git diff --check通过。历史Python源码、实验权重与产物未修改。

## 必须保留的近似边界

G→H前瞻只替换冻结完整Train Bank的support/probe行，不重生成克隆G的整个图库。
H→G标签来自1个Global-only虚拟G步，不是完整K+1链，不重标记旧JVP。
探测使用固定合法轨迹支持集上的条件期望；H信用分配是固定属性/其余编辑的局部条件策略适配。
这些属于本项目的研究实现，不宣称Meta-Weight-Net、COMA或L2T-DLN原算法复现。
CPU图像/文本数值fixture为合成数据，不能代表真实IRRA攻击效果、GPU显存需求或正式Full已通过。

[方法、引用与入口](../../GH_ATTRIBUTE_RESPONSE.md)。源码分支codex/gh-attribute-response，工作树/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116。Windows仅同步Markdown。
