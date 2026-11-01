# C-PGC CUHK-PEDES单模态消融正式结果

两Victim新增Image-only、Text-only完整评测及各自独立CPU审计均PASS；3074 Gallery、6156 Query。
Matched Clean与Joint为历史Full只读引用，已从历史完整评分矩阵重新计算指标并与原CSV、summary、审计逐值核对。

| Victim | Mode | R@1 | R@5 | R@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| IRRA | Matched Clean | 69.44 | 87.22 | 92.02 | 63.02 |
| IRRA | Image-only | 63.65 | 83.02 | 88.95 | 56.47 |
| IRRA | Text-only | 54.09 | 75.52 | 83.17 | 49.40 |
| IRRA | Joint | 47.27 | 70.99 | 79.47 | 42.76 |
| RaSa | Matched Clean | 74.59 | 89.05 | 93.42 | 67.51 |
| RaSa | Image-only | 4.16 | 8.02 | 10.79 | 3.03 |
| RaSa | Text-only | 56.64 | 76.09 | 83.11 | 50.84 |
| RaSa | Joint | 3.64 | 6.86 | 9.02 | 2.58 |


## 实验解释

RaSa单模态中mAP更低的是Image-only。Joint在R@1与mAP上均低于两种单模态：True。
这里只比较冻结攻击产物的模态组合；不把该观察解释为未经验证的训练或机制因果。

## 输入和评分协议

- Matched Clean：clean_gallery + matched_text。
- Image-only：adv_gallery + matched_text，明确不使用Raw Caption的clean_text。
- Text-only：clean_gallery + adv_text。
- Joint：adv_gallery + adv_text。
- C-PGC官方ALBEF/Flickr30K Epoch39 UAP、FP32 12/255原始语义、原单词替换产物保持。
- 原Victim Checkpoint、Tokenizer、完整Gallery、Query顺序和PID保持；新攻击生成、图文编码、训练均0。
- IRRA原生评分，RaSa FP32、全3074 Gallery全局Top128、全部128个候选原生ITM融合，fusion_batch16、query_chunk32。
- 原Matched Clean与Joint各前32条Query额外重放评分，逐元素exact后才执行新增两模式；不重跑原Full。
- 新两模式完整矩阵和逐块分数均保存。独立CPU审计验证SHA、Manifest、身份、完整排名、指标和历史保护。

## 来源

执行Commit：7973128ec1fc73ab8e2428dfbdea600a565421a9；配置SHA：8d6d63e90fa79828de8e5fd45f6bcf76872bc4ae32f48b386aec0973f31fe8e7。
原Full根目录：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/full/cpgc/cuhk/。
新结果：/home/lzf/ldx/outputs/tbps-framework-cpgc-cuhk-modal-ablation-v1。
来源及资源详见SOURCE_EVIDENCE.json。旧Full与SEP执行链未修改。
