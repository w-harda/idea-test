# 跨论文Baseline引用指南

## 可以不重跑所有算法吗？

**可以。** 直接、准确引用R²TUA Table1的已发表值，是合法且可解释的文献对照；不需要把每一种算法重训/重跑一遍才能引用。
必须标注“R²TUA作者报告值（author-reported）”，与“本项目实测”分栏/分区，并保留数据集、Target、攻击类型和可核实预算。
它不自动成为统一环境下重新评测的受控Benchmark，也不能把未知设置补成相同。

## 三类的具体含义

1. 可直接作为近似同协议：关键数据/任务/攻击面/名义预算及评价环境有足够证据支持；仍需常规文献来源标记，不要求GPU逐位一致，也不要求Loss/架构相同。
2. 可以引用但说明差异：作为方法级文献结果或独立文献参考区使用，标注已知差异/未知参数，不据此宣称等权限、等预算的单因素优势。
3. 不适合同一公平比较表：如Targeted↑对Untargeted↓、不同数据集、已确认Image-only对Joint的“同攻击面公平比较”、不同Source迁移设置混入相同Transfer列。仍可放背景/扩展文献表，不能丢弃其来源标签。

当前没有证据充分、无需实质性差异说明的第1类攻击行。原因是Baseline实际epsilon/Source等未报告，以及已确认的参考数值路径差异；不是因为算法Loss或架构不同。
这不阻止使用第2类，也不要求先重跑全部方法。
BASELINE_TABLE.csv分别给G和G+H判类：同一论文行可以对G为2类、对Joint公平表为3类。

## 推荐论文表结构

- 面板A：本项目统一协议实测，列出本机Clean、G Image-only、H Text-only、Joint及已完成同协议对照。
- 面板B：Published reference results†，仅CUHK-PEDES/Untargeted，列Victim、Method、实际报告epsilon/模态、R@K/mAP。未知明确NR。
- RaSa/APTM注明本项目固定IRRA Source迁移；Table1相应行未公开Source路由，不标成同一IRRA→Target迁移。
- Joint单独面板，不把修改图文的结果当成击败Image-only的等攻击面证据；可以讨论不同方法的实际最终指标与权限成本。
- 不把Targeted高R@1误写成Untargeted弱攻击；Targeted只在另表按其成功语义展示。

### 可直接使用的中文表注

“†结果引自Wang等人的R²TUA（CVPR2026）Table1，为原作者报告值，并非本项目复现实测。本文采用官方CUHK-PEDES Test及IRRA公开参考实现；不同运行环境的Clean单独报告。原文未逐行公开部分Baseline的Source、epsilon及预处理配置，未知项标为NR；因此该部分为跨论文方法级参考，不作为等预算、等权限的受控结论。”

### 可直接使用的英文表注

“† Results are quoted from Table 1 of Wang et al., R²TUA (CVPR 2026), rather than re-evaluated by us. Our measured clean baseline is reported separately. Per-baseline source models, perturbation budgets, and preprocessing settings are not fully disclosed in that paper; unavailable settings are marked NR. These entries provide cross-paper references, not a controlled comparison under verified identical threat models.”

### 正文表述示例

可写：“在本项目固定协议下，G60对IRRA的R@1/mAP为21.05/17.51。R²TUA Table1在CUHK/IRRA/Untargeted中报告0.58/1.07；该数值作为文献参考，相关环境和Prompt权限差异见表注。”
不可写：“我们已在完全相同设置复现全部Baseline”“所有方法均8/255”“所有RaSa/APTM行均为IRRA黑盒迁移”“Clean相近证明同一权重”“因不同Loss所以不能比较”。
不应从不同Clean或原文两位数反推出精确ASR/DR/逐Query差异。

## 引用原论文还是R²TUA？

方法名称/机制应引用各方法原论文；**使用R²TUA表内的TBPS适配数字时，同时引用R²TUA并明确数值来源是其Table1**。
不能引用SGA原论文，却把R²TUA在CUHK上的数字写成SGA作者的原始CUHK实验。
VLPTAttack保留论文表名，必要时括注DRA；MetaAttack注明R²TUA引用的TPAMI扩展版，不偷换为AAAI21代码版本。
R²TUA自身使用图像+adversarial prompt，不因有文本编码器就标成Query文本Joint攻击。
