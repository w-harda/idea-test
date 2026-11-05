# SEP Text-only 三数据集 × 两 Victim

协议 SEP-TBPS-SRefPseudoPID-S1。Source 为冻结 CUHK IRRA，参考库为完整 CUHK Train。生成阶段只使用 Caption 和 Source 伪 PID，IRRA/RaSa 共用同一份冻结文本。

执行 Commit `749bc1ca65ac1a44e33cc8cd2eb472b793b96ce1`。实际状态 `FAIL`。
控制目录 `/home/lzf/ldx/tmp/sep_text_only_full_v2_20261105`。输出目录 `/home/lzf/ldx/outputs/tbps-framework-sep-text-only-full-v2-20261105`。

继承176项CPU回归和已有真实分数Exact。本轮只执行受影响接线的定向CPU测试。原Landlock CUDA初始化失败改为Caption-only输入投影及文件审计钩子。S1保留原100次随机权重/DPP调用，N/E=1/1，权重退化为[1]。CUHK IRRA也是Source，因此该行是Source可访问评测。

原同义词/POS/搜索预算/T=10/patience=5/gamma=.02保持。每Caption使用不透明键派生的固定Seed，Smoke产物直接复用于Full。常量/非有限重要性导致官方缩放循环无法结束时显式FAIL，保留原Query，未添加NOOP回退。

## 实际阶段

|阶段|状态|秒|
|---|---|---:|
|reference|FAIL|7.70|

## Full结果

|Dataset|Victim|Attack|R@1|R@5|R@10|mAP|Query|
|---|---|---|---:|---:|---:|---:|---:|
|CUHK-PEDES|irra|SEP Text-only|—|—|—|—|未完成|
|CUHK-PEDES|rasa|SEP Text-only|—|—|—|—|未完成|
|RSTPReid|irra|SEP Text-only|—|—|—|—|未完成|
|RSTPReid|rasa|SEP Text-only|—|—|—|—|未完成|
|ICFG-PEDES|irra|SEP Text-only|—|—|—|—|未完成|
|ICFG-PEDES|rasa|SEP Text-only|—|—|—|—|未完成|

## v2 修复

首个新Attempt在首次图像前向前，torchvision依赖导入需要临时文件而被Source写入门禁拒绝。旧输出与FAIL报告保留于原路径。v2仅为生成进程提供新输出内专用TMPDIR和CUDA cache，Target标注和权重读取限制保持，未修改攻击定义。
