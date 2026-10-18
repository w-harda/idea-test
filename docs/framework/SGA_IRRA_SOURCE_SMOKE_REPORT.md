# 官方SGA / IRRA Source 最小复现实施报告

日期：2026-10-19。开发基准 `04bddc407a59763118cdd3b1200231ec6c2d8e8e`，独立分支 `codex/tbps-sga-irra-source`。发布代码Commit由PROJECT_STATE后续文档提交登记。

**结论：官方攻击完整流程在IRRA Source上实际跑通；固定两图四文的三模型原生Score和独立CPU审计通过。Full未执行。** 初始pipeline在IRRA编码后因两图不满足R@10门禁退出1；原FAIL保留，CPU复用已编码IRRA特征恢复小样本评分，未重跑攻击或IRRA。该事实不能简写成“原pipeline一次全PASS”。

执行命令见[README](SGA_IRRA_SOURCE.md)，完整数值、像素SHA、文本和证据SHA见[JSON索引](SGA_IRRA_SOURCE_SMOKE_EVIDENCE.json)，[CSV](SGA_IRRA_SOURCE_SMOKE.csv)仅供链路验证，不是Full攻击效果。

## 1. 保留的算法与必要接口差异

直接只读加载官方SGA `f49e6926cd13148e49bdbc2c3e537660e096230c`；attacker.py SHA：
`79384bb914824c59c2a7c65c9b3b565cebac9c3b469a7a15d8f21b5ebf110a30`。

实际调用：
`SGAttacker.attack → TextAttacker.img_guided_attack(T_original,I_clean) → ImageAttacker.txt_guided_attack(T1) → TextAttacker.img_guided_attack(T_original,I_adv)`。
观测包装均委托原方法、返回原Tensor，不重写攻击循环。图像阶段收到的T1字符串与刚编码的监督特征逐值核对一致。

保留BERT MLM top10候选、原0.3 logits阈值、原过滤词/WordPiece候选逻辑、KL重要性、图像引导负匹配候选选择、每段至多1词、第二段从原句重启；图像负匹配Loss、NumPy随机起点/高斯增强、平均绝对梯度归一化及sign更新、原投影均保留。eps2/255、steps10、step0.5/255、原图＋四尺度、batch8、全局seed42不变。未使用旧白盒自定义循环、ITM、Push-Pull、属性Loss、8/255或目标反馈。

必要差异明确如下：

| 项目 | 本次IRRA源移植 |
|---|---|
| Source | 官方CUHK IRRA，非原论文ALBEF/TCL/CLIP实验配置 |
| 图像 | IRRA原生PIL RGB/bilinear384×128，再官方Normalize与增强 |
| 表示 | 原生IRRA单位化512维global image/text feature，cls=False |
| 文本桥 | 官方定制BERT Tokenizer/MLM不改；可见WordPiece解码成文本，再IRRA原生Byte-BPE77重新编码 |
| UNK | BERT重要性mask经文本桥保留为字面[UNK]，IRRA以自身Byte-BPE编码；并非给IRRA增加BERT的UNK embedding |
| 输入窗口 | 原BERT30窗口、IRRA77窗口分开；原BERT截断记录，新增IRRA截断拒绝 |
| Matched Clean | 官方pre_caption(raw,30)与同一源RGB输入桥；不拼接历史Raw Clean指标 |
| Target | IRRA同源；RaSa/APTM黑盒迁移，分别原生预处理/Tokenizer/融合评分 |

这是原攻击算法在IRRA源上的移植，不宣称ALBEF论文实验完全复现，也不宣称R²TUA的SGA适配协议相同。源表示改变会改变实际重要性/候选优选结果，未声称与ALBEF的文本得分数值等价。

## 2. Tokenizer/MLM兼容性

官方定制BertTokenizer对单句仅添加CLS、没有结尾SEP。桥只去掉真实CLS/attention padding；内部UNK及替换候选保留，不误删最后一个词。每行记录原输入、可见BERT IDs、解码文本、IRRA IDs及两侧截断状态，要求BERT decode→encode IDs exact。

旧Tokenizer在TBPS的Transformers4.44初始化失败。单独建立Source环境：Torch2.0.0+cu118、torchvision0.15.0+cu118、Transformers4.28.1、tokenizers0.13.3、timm0.6.13、ftfy6.1.1。目标环境未变。官方旧版Tensor Resize默认antialias行为保留。

真实265行源文本桥均WordPiece往返exact，没有新增IRRA77截断；其中146行存在官方BERT30窗口截断、196行含重要性mask UNK。它们包含候选/遮词/中间文本，不能当成265条独立Query。

BERT MLM实际加载1次、调用2次/8个文本样本；没有缺失MLM权重新初始化警告。未使用的seq_relationship两项是NSP Head，不属于MaskedLM任务。MLM SHA：
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`。

## 3. 固定真实输入、扰动与文本动作

Gallery `cuhk:37132:0`、`cuhk:37147:0`；Query依次为两图各caption0/1，2个PID，全部合法正例且有负例。未根据攻击效果选样。

两图Tensor均FP32 [1,3,384,128]，像素在[0,1]，effective_delta读回bitwise exact；各自实数L∞均 `0.007843166589736938`。

数学2/255 = `0.00784313725490196`；两图高出 `2.9334834977801094e-08`，来自原官方FP32 `I±eps` 边界舍入。原FP32投影上下界exact通过；**严格实数≤2/255不应标为通过**。没有额外容差掩盖、二次投影或改epsilon。Full前需确认继续沿用官方浮点语义。

| Gallery | 对抗像素内容SHA-256 |
|---|---|
| cuhk:37132:0 | 1410caf6608b624809667f6381e05ea1ff3b4ba225f245aa6e14e157bb06cc61 |
| cuhk:37147:0 | cc2cf8507db26ca4c1c34f2a6be9ef168eb2e58ff388ba81ed5ff9eee54db80a |

| Query | T1替换 | 最终替换（重新从Matched Clean开始） |
|---|---|---|
| cuhk:37132:0 | tank → each | tank → each |
| cuhk:37132:1 | tank → striped | tank → colored |
| cuhk:37147:0 | away → going | away → going |
| cuhk:37147:1 | visible → has | visible → has |

本轮最终4/4句各1词替换，NOOP=0。实际候选选择和原/最终整句在official_flow.jsonl与证据JSON中。部分替换影响可读性/语义，原官方过滤未保证语义保持；未为提高可读性或攻击效果另加筛选。

## 4. 三目标真实Smoke评分

源生成完成后，目标会话仅消费同一批已提交像素/文本。RaSa/APTM均未参与生成；G/H加载、训练及重新生成均0。

原Engine拒绝2 Gallery的R@10是正确保护，保留不改。独立两图Smoke桥调用原生Scorer并用官方 `rank(max_rank=2)` 计算R@1/mAP；不复制Gallery、不把R@2标为R@5/R@10。

| Target | Mode | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---|---|---:|
| IRRA | Clean | 100.00 | N/A | N/A | 100.00 |
| IRRA | Image-only | 100.00 | N/A | N/A | 100.00 |
| IRRA | Text-only | 100.00 | N/A | N/A | 100.00 |
| IRRA | Joint | 50.00 | N/A | N/A | 75.00 |
| RaSa | Clean | 100.00 | N/A | N/A | 100.00 |
| RaSa | Image-only | 100.00 | N/A | N/A | 100.00 |
| RaSa | Text-only | 100.00 | N/A | N/A | 100.00 |
| RaSa | Joint | 100.00 | N/A | N/A | 100.00 |
| APTM | Clean | 100.00 | N/A | N/A | 100.00 |
| APTM | Image-only | 100.00 | N/A | N/A | 100.00 |
| APTM | Text-only | 100.00 | N/A | N/A | 100.00 |
| APTM | Joint | 100.00 | N/A | N/A | 100.00 |

这些指标不证明迁移有效或无效，不用于挑样/调参。四组2×2以外实际上为**4 Query×2 Gallery**，不能误写成2×2。

## 5. 调用、冻结、资源与审计

Source IRRA加载1次，MLM加载1次。源图像样本前向105＝额外梯度检查1＋原图2＋10轮×5尺度×2图＋最终图2；源文本265包含梯度检查1和大量遮词/候选。仅一次固定输入梯度检查，norm `0.22080878913402557`，有限非零。攻击执行原10轮backward；未安装独立backward计数hook，不能把固定配置数包装成独立hook实测。

Source IRRA及MLM参数/buffers前后摘要exact、eval、requires_grad=False、参数grad均None。IRRA源权重SHA：
`9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009`。

目标IRRA另加载1次，完成4个图像/8个文本编码产物；异常发生在Ranking，原执行报告未写出实际forward计数和最终冻结摘要。因此这两项为**未完整记录**，不能冒充独立计数/冻结后验已通过。CPU评分恢复的模型加载及前向均0，没有为补日志再运行GPU。

RaSa/APTM各加载1次，实际Hook均4图像、8文本、32 fusion pairs；各自参数/buffers前后摘要exact，参数grad None，source_regeneration0。

| 会话 | 耗时秒（含加载） | GPU allocated峰值GiB | reserved峰值GiB | RSS峰值GiB |
|---|---:|---:|---:|---:|
| IRRA Source + MLM | 12.5121 | 6.0494 | 7.1914 | 2.2786 |
| IRRA Target（排名前FAIL） | 18.1146 | 0.8926 | 0.9785 | 2.0676 |
| RaSa Target | 24.8547 | 1.8555 | 2.0352 | 3.8865 |
| APTM Target | 16.6316 | 0.9703 | 1.0488 | 2.1600 |

源攻击本身2.41734秒，加载/来源核验与评分分开；不是Full吞吐实测。启动GPU无计算进程、空闲24051MiB，没有终止其他用户任务。

CPU：18项源接口/原算法测试＋9项小样本评分桥测试＋119项适用回归＝146通过。初始回归中2项因锁定其他实验工作树绝对路径失败；后续显式排除，未修改旧配置或测试来伪造全量通过。官方旧版Resize、TypedStorage及目标Tokenizer默认值告警均记录，不掩盖错误。

独立CPU审计：
- `SGA_OFFICIAL_IRRA_TWO_GALLERY_CPU_AUDIT_PASS`：来源/缓存SHA、四组成员及Score与指标一致。
- `SGA_SOURCE_PIXELS_TEXT_CPU_AUDIT_PASS`：真实FP32像素、原FP32预算边界、effective_delta bitwise、4次最终词编辑。
- Full CPU preflight `SGA_IRRA_CPU_PREFLIGHT_PASS full 3074 6156`，未做Full排名或编码。

145份官方/历史来源SHA+stat、67380份旧Full/FAIL stat及4个旧工作树的HEAD/status保护，运行前后核验均PASS（0变化），SHA登记到证据JSON。本证据不等于重新审计全部旧Tensor内容。

## 6. 文件职责、Full前待办与停止

新增：
- `models/sga_irra_source.py`：只做IRRA源接口、Token转换及审计。
- `baselines/sga_official_io.py`：固定官方模块、CUHK映射、原生RGB读取、官方投影检查。
- `baselines/sga_irra.py`：Source/Target分进程薄入口、不可变产物、原生评估、CPU审计，默认preflight。
- `baselines/sga_irra_smoke_scores.py`：严格2/4限定的Score/官方rank2桥，以及旧IRRA特征CPU复用；无重新攻击。
- 三份Source/Full/Smoke配置、两份CPU tests、中文README/报告、证据JSON/CSV、ADR-0010及统一文档更新。

Full前：确认官方FP32预算舍入口径；batch8资源容量尚未测量（不得擅自改batch）；未见候选的桥兼容性尚未验证，若出现WordPiece往返不一致或额外77截断必须停止；承认Matched Clean与旧Joint Raw Clean不同。Full派生配置仅资源门禁提高至20GiB，其余算法不变。预计总6–9小时，Source batch8显存可能接近24GiB，需RAM至少48GiB、磁盘预留35GiB；非实测保证。详见README的独立授权命令。

本轮没有Full、训练、目标反馈优化、其他Baseline或后台任务。旧白盒实验永久保留并单列，不回填为官方SGA。完成后停止，等待用户审查与Full授权。
