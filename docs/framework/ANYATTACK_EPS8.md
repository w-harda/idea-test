# AnyAttack图像预算修正版：严格8/255、原文文本

用户2026-11-09最新澄清：Idea/GH图像实际L∞≤8/255，文本为每个词最多替换1个字符，可编辑多个词，数量仍由原Dynamic Top-K/Natural-K控制；并非每条Query最多1词。AnyAttack没有文本攻击分支，用户确认无需另加，本实验为Image-only，原文实际0编辑。

本次AnyAttack冻结配置中1词的元数据沿用了此前误记；该任务没有启用文本分支，实际0编辑。此次更正文档及后续Idea设计口径，保留既有冻结源码、配置和运行证据。

此前16/255三模型Full为预算未对齐的补充实验，不能列入8/255正式对照，所有原始记录和分数保留。修正版固定同一官方coco_cos.pt及CLIP ViT-B/32、同一预先固定异PID参考映射；参考映射使用Test PID的访问条件仍须披露。保留官方Decoder权重及图像噪声算法，仅将预算按用户要求改为8/255。

## 实现与来源

旧16/255已保存逐图预算裁剪前的raw_noise和CLIP embedding，且已通过独立CPU审计。本轮按原Manifest/SHA/来源代码/权重严格核验后，只读复用这些Tensor：从原始Decoder输出重新clamp到8/255，再CPU bilinear到384×128，并对最终RGB候选执行FP32向内端点投影。不是将旧16/255最终图片或已裁剪噪声简单减半。实际double差值必须≤8/255，审计容差0；0/1边界同样核验。

新图像每条绑定原始image artifact Manifest SHA，原图/原始噪声/embedding逐值保持；所有新产物独立保存。新CLIP/Decoder加载和前向均0，训练/微调/Backward/Optimizer/JVP均0。独立CPU审计逐图重放全部预算与几何过程；不覆盖旧账本、图像、权重或回执。

新工作树/home/lzf/ldx/projects/idea-test-tbps-anyattack-eps8-20261109，分支codex/anyattack-tbps-eps8；配置configs/framework/anyattack_eps8.server.json；输出/home/lzf/ldx/outputs/tbps-framework-anyattack-eps8-20261109；控制/home/lzf/ldx/tmp/anyattack_eps8_20261109。

## 执行边界

沿用CUHK-PEDES × IRRA/RaSa/APTM三模型，全部真实10 Gallery/10 Query Smoke与CPU审计通过后，自动三组3074 Gallery/6156 Query Full与独立CPU审计。各模型保留原生编码、Tokenizer与评分，RaSa Top128融合，APTM完整Query/Gallery归一化；三模型共享同一严格8/255攻击集合。所有结果为Clean/Image-only；未完成前不得把16/255结果填入8/255报告。

共享GPU仅观察，无VRAM floor，不干预其他进程。真实失败停止、不自动重试、不删失败证据。按用户最新要求，助手仅完成必要CPU验证、启动、交付PID及检查命令，之后由用户检查进度，不持续轮询或等待实验结束。监督器仍自动串联所有门禁和评分。独立报告发布器只在三Full及CPU审计PASS后生成正式结果并推送，不启动、重试或恢复实验。

## 启动前验证

70项适用CPU用例PASS（含严格实数预算、256种字节级端点和向内舍入），保留1项历史APTM冻结目录身份用例的既有排除说明。真实原始噪声首/末Smoke样本CPU预算与完整来源预检PASS，CUDA未初始化。原Decoder/CLIP/模型权重/旧16/255产物未修改。启动后实验进展及最终结果由用户自行检查，当前8/255正式指标PENDING。
