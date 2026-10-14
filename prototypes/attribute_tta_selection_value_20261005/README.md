# 冻结 S_raw 的属性选择价值诊断

2026-10-10：实现、31项CPU tests、固定5句短GPU smoke及CPU-only正式入口preflight通过。GPU已停止；没有运行正式100句，没有commit/push。服务器目录沿用2026-10-05中断前已创建的路径。

## 研究边界

仅比较在相同候选集、相同Natural K和相同TTA预算下，冻结S_raw是否具有攻击选择价值。Step7.5.3 full此前审计为D，没有冻结priority主线；本轮使用原等权PREFIX，不优化scheduler。

三组固定为S_TOPK、RANDOM_K、S_BOTTOMK。Top取原降序前K，Bottom取末K，Random无放回均匀抽K；三组内部均按原S_raw降序，平分保留原实例顺序。S_raw只决定选择和顺序，不作为loss权重。AttributeInstance、source span、region/value、reader/basis、JVP、IRRA、epsilon、IG、transform、momentum/sign保持，无caption loss、新训练或重评分。

## 服务器位置

`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_selection_value_20261005/`

代码：selection.py、prepare_manifest.py、engines.py、report.py、run.py、test_selection.py；deps.py只读接入既有实现。源码、tests、原始outputs不复制到Windows。

## 固定manifest

只读复用Step7.4的13700句score_cache.jsonl；原100句仅78句满足m>K，所以重建确定性100句，不依据攻击结果筛选。

先按K=1/2/3/4/5/6/7+轮流分配；各K内部按(m,候选semantic-axis集合)分层轮流取样，stratum和caption使用固定SHA256排序。原selector仅在CPU manifest准备时从已存S_raw计算原gamma3/beta1自然K，写入后锁SHA；攻击阶段不重算K或score。

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 句数 | 0 | 20 | 20 | 19 | 19 | 19 | 3 |

m分布：3:2、4:8、5:9、6:14、7:35、8:19、9:11、10:1、12:1。95个PID；原缓存12544句符合m>K，1156句排除。没有符合条件的K1，不人为修改K补样本。

Random的seed为SHA256('random-k|42|caption_id')前8字节转换的整数，使用本地random.Random.sample；不消耗攻击RNG，不重抽，不搜索seed。按AttributeInstance ID抽样，不合并同value或相同语义实例。

Top/Random有8句集合完全相同，Random/Bottom也有8句；Top/Bottom没有完全相同。平均Jaccard分别.4840/.4457/.2402（最后一项为Top/Bottom）。高K下重叠较大，正式输出保留逐句intersection/Jaccard及按K结果。该分层诊断不代表validation总体分布，K7仅3句。

## 攻击控制

直接复用原PREFIX的JVP算术平均与官方IG/TTA，不进入weighted/priority梯度实现。每组逻辑执行K轮、每轮10steps、5个scale路径、每scale6个transform、每step35个IG节点；8/255全局L∞预算、2/255 step、momentum=.9、paired soft-rank tau=.02、34054 clean train gallery均保持。

S_TOPK记录真实categorical transform trace，其余模式回放同一trace并保持原adaptive参数更新及随机数消费。各模式seed42，噪声/trace/RNG严格核对。不同属性不能共享round1图像状态。完全相同的完整属性序列复用同一次实际轨迹，避免给相同处理引入独立CUDA漂移；记录execution_reuse_of，逻辑K/steps预算保持。

每个union-selected实例的原JVP在优化前构造一次，后续只读；不跨不同image/momentum状态复用梯度。文本caption只用于原forward上下文和检索评价，无caption guidance loss。

## 验证

31 CPU tests通过：集合和顺序、随机可复现且不消费攻击RNG、实例metadata/source SHA门禁、unresolved/unsupported排除、same-value不同source span、manifest对cache迭代顺序不敏感、原PREFIX toy k1/2/3/5/7逐值回归、噪声/trace/RNG、预算、observer及手动full门禁。

固定GPU smoke：

| caption | K | m |
| --- | --- | --- |
| cuhk:7358:1 | 2 | 6 |
| cuhk:22504:0 | 3 | 7 |
| cuhk:12666:1 | 5 | 9 |
| cuhk:23923:0 | 6 | 9 |
| cuhk:17560:0 | 7 | 8 |

仍传入10steps/原IG/scales/transforms，仅每轮完成第1个完整更新后observer截停，共67个实际轮次/优化步。不是完整每轮10步攻击评测，不据smoke判断选择价值。

method_audit_passed=true；同K、同源候选、选择序列、随机回放、zero和冻结检查通过。最大L∞ .03137257695198059，仅数学8/255之外2.79324e-8原fp32舍入。JVP最大FD相对误差.00449228（.4493%），沿用既有2%门禁。167项来源/本轮源码锁保持；score backward0、caption loss0、无训练。

构造IRRA时日志中的“Training Model with ... tasks”是既有模型构造信息；实际参数requires_grad/grad/version检查保持frozen。第三方既有kernel cache/affine_grid警告未改变算法或环境。

## 计时

原真实step平均.77647s、中位.76143s。k3/5/6/7各组平均每步约.756–.793s。完整100句每mode406逻辑轮；相同完整序列复用后预计1153物理轮/11530实际步，纯攻击估计2.49小时；含准备、JVP、评价与输出，空闲4090暂按2.5–3小时预算，非完整实测。有并发会更长。

## 用户手动100句命令

下面命令未执行：

```bash
tmux new-session -d -s tbps-selection-value 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u prototypes/attribute_tta_selection_value_20261005/run.py --mode diagnostic --allow-diagnostic > prototypes/attribute_tta_selection_value_20261005/outputs/diagnostic.log 2>&1'
```

- manifest：上述目录下outputs/manifest/selection_manifest.json。
- 日志：outputs/diagnostic.log。
- 正式结果：outputs/diagnostic/。
- 完成标志：日志末尾COMPLETE与completed.json，method_audit_passed=true、case_count=100、same_K=true、noise_transform_RNG_exact=true。
- 正式入口需显式allow-diagnostic、成功smoke及冻结SHA，不覆盖已有输出。CPU-only preflight已通过，没有调用正式main；outputs/diagnostic不存在。
- 输出：summary.json、per_caption_results.json、round_marginal_effects.json、selection_manifest.json、protocol.json、audit.json、completed.json。包含Top vs Random/Bottom最终三指标、win/loss/tie、mean/median/range、按K统计以及逐轮和clean-to-prefix比较。
- 这是固定clean gallery下paired replacement诊断，不是共享adversarial gallery正式TBPS评测。

不修改正式源码、score、Top-K或generator；不自动扩实验。后续只审阅100句结果来回答选择价值。
