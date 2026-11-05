# SEP-TBPS-TextOnly-WordBudget1

本轮将所有搜索、评分候选和最终文本限制为相对原始空白分词序列最多替换1个单词位置。检查使用Python str.split，保持一对一单词位置映射。最终文本按原始非空白词跨度替换，其他词、大小写、标点和空白均保持。

固定Source-only接线复用CUHK IRRA、完整34054图Train参考库和11003个Train PID。每条原始Caption固定Source Top1伪PID，Target Test PID、配对、评分和Clean正确性不进入攻击生成。IRRA/RaSa只在冻结文本后独立评分。同一域只生成一次文本。CUHK IRRA也是Source，该行属于Source可访问模型评测。

## 官方代码的必要预算适配

- 固定官方文件SHA，复用原函数AST，仅将max_change_num与att_succ_num的比例公式改为min(1,len_text)，后续原合法位置数量上界保留。max_change_rate=.2保持兼容形参，但不参与本版预算决定，不将1误当作1词。
- 整句预处理和256词截断会改变多个原始单词，本版搜索保留全部原词，只在词表查找时对单个词使用大小写/边缘标点别名。非修改词不清洗。
- 原重要性评估使用当前文本删词。为使所有Source评分输入都满足1词替换，本版改为基于原文的单位置[UNK]探针。探针评估的上下文与原方法存在明确差异，不能宣称与原目标数值等价。
- optimize、top50同义词、POS顺序、随机选择、DPP、T=10、patience=5及tau gamma=.02沿用原代码。单代理N/E=1/1退化为权重[1]，已显式检查。
- 单一可选位置的softmax恒为1，必要地跳过无法终止的单元素跨度缩放循环；多位置常量或非有限重要性仍报错停止，未改成NOOP。
- 没有合法同义词时保留原始Caption，记录NO_LEGAL_SYNONYM。合法搜索没有产生不同tau候选时如实记录NO_DISTINCT_OFFICIAL_TAU_CANDIDATE。
- 每次Source前向、搜索中间词列表和最终输出均检查原始位置预算。超预算抛错，保存Claim和日志，不截断或自动修复。

本版是固定单词预算SEP-inspired TBPS适配。已验证176项历史CPU回归和真实分数Exact直接复用，本轮只新增单词预算定向测试。

## 执行

三个数据集各固定8条原始Query做Text-only Smoke。六组原生评分和CPU预算核对全部通过后自动进入6156/2000/19848 Query完整Full。没有Raw Clean、Matched Clean、Image-only或Joint评分。RaSa完整Gallery Top128和ITM fusion_batch16、query_chunk32、FP32原生路径保持。

原比例预算执行链559548/627211已按新预算指令核实身份后停止，完成的CUHK/RSTP Smoke和ICFG中断Claim保留。原比例预算文本不复用于本版。仅复用与Query及预算无关且已发布的Source Train图像特征和只读数据身份。

新工作树为 `/home/lzf/ldx/projects/idea-test-tbps-sep-word-budget1-20261105`。
输出为 `/home/lzf/ldx/outputs/tbps-framework-sep-word-budget1-20261105`。
控制为 `/home/lzf/ldx/tmp/sep_word_budget1_20261105`。

状态以控制回执为准。真实错误停止保留，不自动重试。共享GPU无固定空闲显存门槛。正常长耗时不触发停止。最终报告由Supervisor在真实Full和独立CPU核对后发布。
