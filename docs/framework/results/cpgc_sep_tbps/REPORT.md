# C-PGC与SEP-Attack三数据集正式评测

本报告的执行准备版本不包含任何正式Full指标。12组结果仅在原生评分完成且独立CPU审计PASS后填入RESULTS.csv，缺失值不使用Smoke或论文数据替代。

## 实验范围

CUHK-PEDES3074 Gallery/6156 Query、RSTP-ReID1000/2000、ICFG-PEDES19848/19848；仅IRRA、RaSa及各自对应数据集官方权重。三个Test集合全部图像引用存在；ICFG标注的Train中存在同一路径跨PID重复，但Test图库无重复，不修改原始数据。

C-PGC仅使用作者ALBEF/Flickr30K epoch39发布UAP，图文联合；SEP仅修改文本，代理固定IRRA+RaSa，Target属于代理集合。两者不是本项目G/H的新训练。

## 执行与状态

代码、来源、适配边界见同目录审计文档。唯一后台监督器operations/run_cpgc_sep_full.sh先执行真实Smoke与独立CPU审计，再允许对应方法Full；数据顺序CUHK→RSTP→ICFG、Victim顺序IRRA→RaSa，串行执行。Smoke固定原顺序前128 Gallery/前2 Query，覆盖真实RaSa Top128路径，不作为攻击效果结论。

正式输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v1/
控制日志：/home/lzf/ldx/tmp/cpgc_sep_full_v1_20261028/
每个Source、Target、CPU审计独立Attempt；每个成员/特征块/评分块不可变原子提交。存在FAIL、未提交Claim或pending时不自动重跑；显式resume仅复用来源一致的已完整提交单元，不能承诺崩溃于单元内部时自动恢复。

## 评价与限制

完整Gallery分数和多正例PID排序，R@1/R@5/R@10/mAP为百分数。RaSa使用全图库Top128候选及原生ITM正类logit，候选外-100；完整排名mAP并非把Top128截断后的AP冒充全量AP。ICFG局部Token分块落盘，不近似缩小图库。

分别记录Raw Clean及Matched Clean，后者只执行官方攻击Caption前处理，避免将小写、标点处理或截断当成攻击收益。IRRA为官方convert_weights精度路径，RaSa原生FP32；与历史公共物化输入Full的桥和精度并非必然相同，不直接拼接Clean。

共享GPU只观测，不因利用率高单独拒绝；每阶段空闲显存至少10240MiB、RAM32GiB、磁盘100GiB、inode80000。真实OOM和其他异常保存FAIL，绝不降精度、缩TopK或改算法重试。SEP大量原生融合调用可能很慢，最终耗时以实际日志为准。

## 本轮真实完成情况（2026-10-28）

96项CPU测试PASS，真实sga-irra-source与tbps-image两环境v2 CPU预检PASS。最终即时GPU快照：空闲1302MiB/24564MiB、利用率84%；外部Python PID2411159占7760MiB，Ollama PID3763832占14976MiB，均未干预。资源不足时停止，没有创建输出根目录、GPU Attempt或后台控制目录；没有启动Smoke/Full/Watcher。

正式完成0/12，12组均RESOURCE_BUSY_NOT_STARTED；CSV指标全部空白。CPGC原生Joint和SEP检索适配代码已经CPU验证，但尚无真实GPU验证，不能声称算法执行或Full PASS。资源释放后须先通过原生Smoke与审计，再正式Full。

资源及成本：仅ICFG RaSa的原/Adv/delta FP32像素落盘约98.13GiB，两个Gallery局部Token约65GiB，加文本、评分及SHA元数据会更多；全链需预留数百GiB。当前服务器磁盘充足，运行脚本仍逐阶段核验。SEP每次候选需RaSa全图库Top128融合，可能显著慢于普通分类；无真实Smoke前不虚构Full耗时或显存峰值。

保护核验：旧审计工作树保持干净，六套Victim权重、G60/H60关键SHA和官方攻击源码通过真实CPU预检；只在独立新工作树/资源/临时目录写入，不声称重新扫描历史所有大体积文件。
