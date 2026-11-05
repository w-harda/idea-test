# SEP Text-only 三数据集 × 两 Victim

协议 SEP-TBPS-SRefPseudoPID-S1。Source 为冻结 CUHK IRRA，参考库为完整 CUHK Train。生成阶段只使用 Caption 和 Source 伪 PID，IRRA/RaSa 共用同一份冻结文本。

执行 Commit `a22b1a412df66886c73b9d14524a56551a8a1a96`。实际状态 `FAIL`。
控制目录 `/home/lzf/ldx/tmp/sep_text_only_full_v2_20261105`。输出目录 `/home/lzf/ldx/outputs/tbps-framework-sep-text-only-full-v2-20261105`。

继承176项CPU回归和已有真实分数Exact。本轮只执行受影响接线的定向CPU测试。原Landlock CUDA初始化失败改为Caption-only输入投影及文件审计钩子。S1保留原100次随机权重/DPP调用，N/E=1/1，权重退化为[1]。CUHK IRRA也是Source，因此该行是Source可访问评测。

原同义词/POS/搜索预算/T=10/patience=5/gamma=.02保持。每Caption使用不透明键派生的固定Seed，Smoke产物直接复用于Full。常量/非有限重要性导致官方缩放循环无法结束时显式FAIL，保留原Query，未添加NOOP回退。

## 实际阶段

|阶段|状态|秒|
|---|---|---:|
|reference|PASS|187.78|
|smoke-source-cuhk|PASS|619.59|
|smoke-target-cuhk-irra|PASS|10.30|
|smoke-audit-cuhk-irra|PASS|4.26|
|smoke-target-cuhk-rasa|PASS|20.41|
|smoke-audit-cuhk-rasa|PASS|4.36|
|smoke-source-rstp|PASS|507.69|
|smoke-target-rstp-irra|PASS|10.03|
|smoke-audit-rstp-irra|PASS|3.98|
|smoke-target-rstp-rasa|PASS|21.51|
|smoke-audit-rstp-rasa|PASS|3.99|
|smoke-source-icfg|FAIL|542.99|

## Full结果

|Dataset|Victim|Attack|R@1|R@5|R@10|mAP|Query|
|---|---|---|---:|---:|---:|---:|---:|
|CUHK-PEDES|irra|SEP Text-only|—|—|—|—|未完成|
|CUHK-PEDES|rasa|SEP Text-only|—|—|—|—|未完成|
|RSTPReid|irra|SEP Text-only|—|—|—|—|未完成|
|RSTPReid|rasa|SEP Text-only|—|—|—|—|未完成|
|ICFG-PEDES|irra|SEP Text-only|—|—|—|—|未完成|
|ICFG-PEDES|rasa|SEP Text-only|—|—|—|—|未完成|
