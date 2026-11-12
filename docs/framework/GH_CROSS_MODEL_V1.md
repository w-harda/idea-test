# Cooperative与原Push-pull：IRRA→RaSa/APTM迁移评测

本任务固定两版各自Round60配对G/H，数据集CUHK-PEDES，目标RaSa与APTM，共四组Joint Full。两版原IRRA正式Test已保存完整3074张攻击图像、6156条攻击文本及CPU审计回执，本任务直接只读复用，不重新训练、不重新生成G/H攻击，不加载IRRA或计算JVP。

## 输入与来源

新分支`codex/gh-cross-model-v1`，独立服务器工作树`/home/lzf/ldx/projects/idea-test-tbps-gh-cross-model-v1-20261112`。配置`configs/framework/gh_cross_model_v1.server.json`固定：全部当前Python生产者/执行入口、官方目标模型来源与权重、两版原训练配置、Round60 G/H SHA，以及原训练审计/IRRA Full完成和审计/攻击完成清单的SHA。历史配置按其原工作树核验，不改其源码身份、不重签历史清单。

图像预算：封存的384×128、FP32 RGB输入相对原图严格L∞≤8/255，容差0。逐图SHA校验并从原始图重新执行原IRRA的PIL Resize与ToTensor，核对干净像素exact；随后目标模型执行其自己的Tensor预处理。RaSa为384×384 BICUBIC，APTM为384×128；不将插值后的空间另称原8/255约束空间。

文本沿用Natural-K，可修改多个词，每个原词最多替换1个Unicode字符。独立导入原SHA锁定contracts/constraints，逐条原文、原词位置、动作与Unicode replay核验。不根据目标模型Tokenizer结果删除、更换或补做编辑；统计目标BERT Token实际改变的Query数量。

CPU输入核验保存只读collection索引与原文件SHA/size/mtime/inode保护快照。两版各有独立索引，两模型共享该版同一批像素和文本；不同方法不能混用Checkpoint、图像或文本。原Cooperative报告缺失matplotlib的FAIL保持。

## 原生评分

沿用已存在的RaSa/APTM Adapter，不修改官方模型或公共评分源码。目标ModelSpace绑定各自Checkpoint、Tokenizer、预处理及评分协议。RaSa保留完整Gallery全局候选选择Top-128和原生ITM正类logit。APTM保留Top-128、原生未选中候选填充以及整个6156×3074矩阵的ITM/global归一化与0.002全局项。禁止把IRRA特征当成目标特征、改成cosine或缩小正式Gallery。

每组保存Clean与Joint，完整3074 Gallery／6156原Query、全部同PID正例、R@1/R@5/R@10/mAP（另保留mINP）。图像、文本、融合均microbatch1；缓存块16，Query块128。Clean与Joint各需6156×128融合对；每组实际神经样本计数应为image6148/text12312/fusion1575936，运行计数与真实前向Hook核对。Smoke仅10图10Query，每组fusion200，不作为正式结果。

## 执行顺序

1. 两版完整独立CPU输入核验（不进入CUDA）。
2. 等待启动前已存在的增强版Supervisor退出；使用PID及`/proc`启动标识避免PID复用误判，不干预其生命周期。
3. 两版×两目标共四组真实Smoke，每组后立即启动独立CPU审计；全部通过才进入Full。
4. 同样四组Full，每组后独立CPU审计。真实异常立即FAIL并保留，不自动重试、恢复或覆盖。
5. 四组Full均PASS后生成中文REPORT、原精度METRICS.csv和SOURCE_EVIDENCE.json，核对原历史文件完整SHA/stat，自动仅提交报告并push本分支、核对远端SHA。

独立CPU审计禁止NN前向、导数、Optimizer及CUDA，只允许白名单已保存Tensor反序列化；逐块核对原生特征及分数、对应ModelSpace/完整输入顺序/配对Checkpoint。RaSa逐块拼接与最终分数exact；APTM从原始两类分块重建整个矩阵、独立核对全矩阵归一化，再对Clean和Joint重算完整排名，要求五指标exact。记录审计实际耗时与RSS峰值。

新输出`/home/lzf/ldx/outputs/tbps-framework-gh-cross-model-v1-20261112`；控制和逐阶段日志`/home/lzf/ldx/tmp/gh_cross_model_v1_20261112`。GPU仅单模型顺序执行；共享GPU不设固定VRAM floor，不修改或终止外部进程。Full阶段仅消耗已有攻击结果，两个训练流程及当前增强版算法/来源不改。

## 检查命令

```powershell
ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-cross-model-v1-20261112/operations/gh_cross_model_v1_status.py"
```

`inputs_cooperative`／`inputs_attraction`为CPU样本核验；`WAITING_FOR_EXISTING_AUGMENTATION`表示已启动但GPU阶段排队；`*_smoke`和`*_full`日志记录编码/原生评分进度；对应`*_audit`为独立CPU复核。最终成功状态`SUCCESS_FOUR_FULL_TRANSFER_TESTS_PUBLISHED`；失败状态`FAIL_STOPPED_NO_RETRY`。仅确认脱离终端的启动身份后助手退出，不持续监控。

开发CPU证据见同目录`GH_CROSS_MODEL_V1_CPU_EVIDENCE.json`；开发期失败日志保留在服务器development目录，不是正式Attempt。真实Smoke/Full与迁移效果以新的审计回执为准。
