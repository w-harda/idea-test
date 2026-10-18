# ADR-0010：官方SGA算法固定为IRRA Source的I/O移植

状态：用户明确指定IRRA源；固定两图四文攻击及三目标Score Smoke已执行，Full未授权。

## 决定与边界

保留ADR-0009的算法忠实性原则，旧白盒实验及旧审查ALBEF推荐作为历史快照保留。本次按用户最新授权，唯一Source改为既有官方IRRA，而不是ALBEF，也不分别针对RaSa/APTM白盒优化。原官方SGAttacker/ImageAttacker/TextAttacker文件保持固定Commit和SHA。

IRRA Wrapper只提供inference_image/inference_text：原生IRRA全局单位特征、384×128输入、BERT WordPiece解码后原生Byte-BPE重新编码。BERT MLM及词候选/重要性/负匹配评分/更新仍由原攻击器执行；不新增ITM、属性或其他Loss。将BERT Token ID直接当作IRRA Token ID禁止。

官方定制Tokenizer单句只加CLS、不追加SEP。只移除真实CLS与attention padding，内部UNK/候选保留；逐条验证WordPiece IDs decode→encode exact，额外77上下文截断拒绝。官方BERT窗口30带来的截断独立记录，不声称源模型看到了完整原句。

匹配Clean与Adv共同使用原生IRRA像素桥及官方pre_caption30。目标各用原生Tokenizer、预处理和评分；不得复用旧Raw Clean数值来隐藏输入桥变化。

## Smoke与预算口径

两图无法报告R@5/R@10。保持通用Engine对至少10 Gallery的要求；独立Smoke桥验证原生Score及官方max_rank2的R@1/mAP，绝不填充/复制Gallery或伪造召回率。原IRRA排名FAIL及完整缓存保留，CPU恢复评分，不重新生成攻击/编码IRRA。

官方FP32投影可能出现约3e-8实数预算舍入超量。本次保留算法并公开实际数值，不擅自添加二次投影或声称严格实数上界已验证。Full前需用户确认此口径及batch8资源方案。

本结果是SGA原始算法在IRRA源模型上的移植，不能称原论文ALBEF实验完全复现或R²TUA的同协议复现。主方法仍为G-v2+H-v1 Joint；比较需声明Test配对访问、2/255与8/255、词与Unicode编辑、训练来源和推理成本差异。
