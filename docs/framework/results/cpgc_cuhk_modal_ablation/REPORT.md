# C-PGC CUHK-PEDES单模态消融：执行准备与历史结果

本任务只补充IRRA、RaSa的Image-only和Text-only。原6组C-PGC Full、Source、UAP、图文特征及审计均只读；不重新生成或编码图文攻击产物。

## 严格输入组合

| Mode | 图像特征 | 文本特征 |
| --- | --- | --- |
| Matched Clean | clean_gallery | matched_text |
| Image-only | adv_gallery | matched_text |
| Text-only | clean_gallery | adv_text |
| Joint | adv_gallery | adv_text |

通过独立Matched Dataset Manifest与显式来源绑定，将Engine的clean_text指向matched_text缓存；没有改Engine默认映射，没有把Raw Caption混入Image-only。

## 已有真实Full数值

单位均为%。横线表示尚未执行新增评分，不能当作0或已有结果。

| Victim | Mode | R@1 | R@5 | R@10 | mAP |
| --- | --- | ---: | ---: | ---: | ---: |
| IRRA | Matched Clean | 69.44 | 87.22 | 92.02 | 63.02 |
| IRRA | Image-only | — | — | — | — |
| IRRA | Text-only | — | — | — | — |
| IRRA | Joint | 47.27 | 70.99 | 79.47 | 42.76 |
| RaSa | Matched Clean | 74.59 | 89.05 | 93.42 | 67.51 |
| RaSa | Image-only | — | — | — | — |
| RaSa | Text-only | — | — | — | — |
| RaSa | Joint | 3.64 | 6.86 | 9.02 | 2.58 |

新增评分完成并通过独立审计前，不判定RaSa主要受哪个模态影响。

## 实现范围

- 新模块modal_ablation.py：读取旧配置、Artifact、特征和Dataset来源；核验全部Manifest/SHA与旧审计；新两模式评分及独立CPU审计。
- 复用原NativeScorer、NativeRasa、TBPSEvaluationEngine及ranking，未修改其数学。
- 旧配置SHA保持4b56eb60a3f2c928ef084fad64f1648bee17129e450b707a4d166169bc8a370a；新配置单独绑定所有producer SHA。
- IRRA保留原生mixed权重/float32检索特征，RaSa FP32、Top128完整融合、fusion_batch16、query_chunk32。
- 新结果包含原精度/两位小数CSV、完整分数矩阵、逐Query Chunk、RaSa全局Top128候选索引及冻结模型摘要。
- 先各重放历史Matched Clean、Joint前32 Query，要求score逐元素exact。历史Full不重新计算。
- C-PGC官方12/255 UAP仅只读引用，文本保持官方原产物；新图像生成、文本攻击、图文编码、Backward、Optimizer均0。

## 与SEP隔离

2026-11-01检查时SEP监督器2960820与Source进程3041291仍运行，进度6/12。未向两者或Ollama发送信号，未修改其工作树、配置或输出。

默认一次性监督器等待SEP持有的同一全局flock释放，并等待GPU其他计算进程退出，再执行本消融。等待期间不初始化CUDA；没有固定空闲显存数值门槛。保留RAM/磁盘/inode检查及真实OOM首错停止。

该默认方案优先保证不与SEP争用GPU，不能把后台等待误称为消融已完成。

## 执行与审计

控制目录：/home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_v1_20261101/。
新输出：/home/lzf/ldx/outputs/tbps-framework-cpgc-cuhk-modal-ablation-v1/。
旧结果：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/full/cpgc/cuhk/{irra,rasa}/。

43项CPU回归PASS。测试期间一次命令引用不存在的测试文件、一次新增测试字段误写epsilon（实际ImageBudget.value）；均保留原日志，修正命令/测试拼写后完整受控运行通过，没有改预算或断言语义。

真实来源Preflight PASS；两份历史Full及原CPU审计、Source和所有特征单位核验通过，保护文件90869份。后台PID与最新状态以控制目录回执为准。不存在新两模式GPU/CPU审计PASS时不发布最终成功。

每Victim评分后独立进程运行CPU audit，核验完整3074×6156分数、块拼接、原生Top128、排序与四项指标、Matched文本身份、CSV、SHA、历史保护。

两Victim均PASS后，自动在独立结果分支codex/cpgc-cuhk-modal-results提交REPORT.md、METRICS.csv、METRICS_2DP.csv及SOURCE_EVIDENCE.json。不会修改正在运行的执行工作树；发布失败单独保留，不重跑评分。

## 查看命令

```bash
cat /home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_v1_20261101/status.json
tail -f /home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_v1_20261101/status.log
tail -n 40 /home/lzf/ldx/tmp/cpgc_cuhk_modal_ablation_v1_20261101/evaluate-rasa.log
```

## 局限

原RaSa三组Full总耗时约19.2小时；新增两组同量级Top128融合评分可能较长。缓存可减少读取成本，未据此声称已测得新Full耗时。最终只依真实日志报告资源与时长。

本实验只研究冻结产物的模态组合，不使用新结果调参或再生成攻击。
