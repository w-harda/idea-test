# C-PGC作者预训练UAP审计

官方仓库https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks ，Commit248fbdaa02a7d34c9f1f68772c3a82c8e19dbfed。ICCV2025论文及补充材料已读取，默认ALBEF/Flickr30K发布目录、eval epoch39作为唯一预注册选择，未按TBPS效果筛选。

作者目录：https://drive.google.com/drive/folders/1XwSiBpdOgbtI0lyjEO6coq_fiObg-9P6 。资源SHA见CONFIGS.json和SOURCE_EVIDENCE.json。原图像UAP为FP32 [3,384,384]，max_abs=0.0470588244497776；这是官方12/255的FP32端点，不宣称严格实数上界。原噪声文件和embedding均只读，不加载或训练Generator，不优化UAP。

文本最近词遵照eval.py的torch.dist/topk/filter_words顺序，得到pauline；不因语义或效果更改此词。随后直接执行作者TextAttacker.get_adv_text的完整原函数：pre_caption(...,30)、Source KL重要性、至多替换1词。这里是通用嵌入解码词，不应误称评测时重新用MLM生成候选。ref_net仅被读device，复用实际Source BERT设备，未加载未使用的MLM。

官方attacker.py顶层会导入本实验无关的BLIP/fairscale。新official.definitions逐字节SHA核验后，编译原始AST定义节点，未改函数体或候选/损失。与直接执行整个eval.py的区别：只加载所需ALBEF Source，不加载其他无关Target和训练器；最近词循环移入输入接线，记录源码来源，不声称整个官方eval.py原封执行。

ALBEF官方Flickr30K Checkpoint来自Salesforce发布地址https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/flickr30k.pth 。官方strict=False保留；实际missing只有不用的动量encoder/queue，全部有效推理权重完整，出现任何有效层缺失即拒绝。CPU真实加载已PASS且未初始化CUDA。

输入适配：RaSa保留384方形噪声；IRRA只做确定性bilinear align_corners=False至384×128，不放大幅度；每图保存原图/Adv/delta及实际L∞。原始图片先走各模型原生PIL Resize/ToTensor，随后加固定噪声并clamp[0,1]，不将Adv重新PIL/JPEG量化。两个Victim及三个数据集共享完全相同作者UAP来源SHA。最终主模式始终Joint，未省略文本侧。
