# 完整 test 的 Top-K / Random-K / Bottom-K 共享图库评测

2026-10-11：独立 prototype 已实现；35项新增 CPU tests、31项冻结选择/PREFIX回归通过；固定两图短GPU smoke与CPU续跑检查通过。GPU停止。Codex未启动 full prepare、全量攻击或正式 test 评测，无commit/push。

## 位置与冻结边界

服务器：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_selection_full_test_20261011/`。
实现：core.py、runtime.py、run.py、test_full_test.py。源码、缓存、fp32图像及原始结果均留在服务器；Windows只保存本文摘要。

三组 S_TOPK / RANDOM_K / S_BOTTOMK；同图同候选、同Natural K；集合内部按冻结S_raw降序，使用原等权PREFIX_BASELINE、10steps/round、8/255、原IG/scale/transform/momentum/sign。无weighting、priority projection、caption loss、训练、G或文本攻击。原正式Stage01/Stage03/Top-K及旧prototype均未修改。

## 正式 test 协议

- 官方 CUHK-PEDES test：3074张唯一图像，6156条原始caption，全部保留，PID relevance、text-to-image、无camera排除。
- 每张图像固定使用annotation中caption_index=0作为攻击条件；不按score/效果选择caption。所有6156条caption检索同一份3074张对抗图像图库，每种mode各有一套。
- 该攻击仍依赖已知caption条件；图库固定后不随query变化。这是完整test共享图库评测，不等于caption-independent生成器或跨模型迁移评测。
- m=K时相同集合保留；K=0时三组均保留clean图像，不筛掉难例。相同完整序列复用同一次实际轨迹，其余模式独立从clean开始并严格回放TOP anchor noise/transform/RNG。
- 不复用旧13700 train-validation S cache。准备阶段对3074个固定条件caption各做一次冻结Step5.1 clean backward；公式/reader/basis不变，竞争图库明确由旧34054 clean train切换为3074 clean test。prepare写逐图S/w/e/固定K/选择序列及SHA；attack只读缓存、不重评分、不重新计算K。
- 每张最终对抗图保留fp32的模型输入像素张量及其真实IRRA编码feature，不PNG/JPEG量化，不回填paired cosine来伪造共享图库。
- 对全部query与全部adv gallery features计算完整相似度矩阵（分batch处理）并复用正式PID指标实现。
- 当前单victim IRRA fp32，aAP=mAP，mDR=(aAP_clean-aAP_adv)/aAP_clean*100；复用既有多victim聚合函数，未接通其它victim。
- 原AP-ATTACK结果不由本入口重跑；横向比较仍需核对同一受害者精度/预处理及caption-conditioned threat model。

## 已验证

- 全量CPU preflight：3074图/6156query，全部Stage01 test provenance覆盖；不重写提取器。
- 新35CPU tests：完整覆盖、首caption固定、同PID多图、无放回随机与攻击RNG隔离、同K/排序、m=K、K0、异常/缺失拒绝、安全路径/SHA、完整共享图库多正例AP、aAP/mDR聚合和fp32归档。
- 原31CPU回归：冻结选择、原PREFIX toy k1/2/3/5/7和随机控制通过。
- 固定annotation前两图37132:0/37133:0，K=3/2，m=4/4；smoke仅两图/四query，没有完整test feature extraction。保留10step配置，observer每round首次完整更新后截停，共15真实更新；不是10step正式攻击性能。
- 第一轮smoke遇到历史probe模块namespace查找缺失，已在新runtime只读扩展已锁SHA快照路径修复；原源码不变，失败日志保留。
- 最终图capture SHA等于原TTA实际post-projection endpoint；fp32 archive roundtrip exact。
- 最大L∞=.03137257695198059，超过数学8/255仅2.79324e-8既有fp32舍入；JVP FD最大相对误差.00272689（.272689%）；zero/frozen、同K、noise/transform/RNG及93项源码/来源SHA通过。
- smoke两张图属于同PID，因此四条query在两图图库中的AP必为100%；这个数值只验证评测链路，不评价攻击性能。
- CPU重入同一smoke output显示PREPARE_ALREADY_COMPLETE / ATTACK_ALREADY_COMPLETE，无新backward/攻击；读取已存图库重新汇总成功。
- GPU已空闲；正式tracked/staged diff为空。outputs/full不存在。

## 用户手动后台命令

在服务器终端执行（Codex未执行）：

```bash
tmux new-session -d -s tbps-selection-full-test 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -B -u prototypes/attribute_selection_full_test_20261011/run.py --stage all --scope full --allow-full >> prototypes/attribute_selection_full_test_20261011/outputs/full.log 2>&1'
```

此命令依次执行prepare → attack → evaluate。首次full需要已通过smoke，SHA不一致时拒绝；文件锁防止同目录并发。支持按已完成逐图score/attack文件恢复，读取前核验来源SHA，异常数据拒绝，不覆盖其它实验。中断后同一命令可续跑；如tmux同名session仍存在，先查看而不是重复启动。

也可将--stage all分别改为prepare、attack、evaluate依次手动执行；其它参数保持。已有完整stage会核验并跳过，不重跑攻击。full evaluation只读取保存的query/adv feature caches及fp32图像归档。

日志：
`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_selection_full_test_20261011/outputs/full.log`

结果根：
`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_selection_full_test_20261011/outputs/full/`

主要产物：
- protocol.json、selection_manifest.json（K/m、集合重叠、预计时间）、clean_features.pt
- scores/*.json（每图冻结一次score）
- gallery/{S_TOPK,RANDOM_K,S_BOTTOMK}/*.pt（完整3074图/组的fp32像素和真实IRRA feature）
- attacks/*.json、gallery_index.json（来源/预算/每轮轨迹与SHA）
- metrics.csv、summary.md、summary.json、victim_summary.json、per_query_metrics.jsonl
- prepare_completed.json、attack_completed.json、evaluate_completed.json、completed.json

最终完成：日志COMPLETE，completed.json中scope=full、gallery_count=3074、query_count=6156、method_audit_passed=true、shared_adversarial_gallery=true、formal_test_evaluation=true。

## 时间与空间

短smoke每真实step平均.8062s，中位.7576s，仅估计依据。完整耗时取决于full prepare生成的实际K和相同序列复用数；selection_manifest给出physical_rounds与按.8s/step的纯攻击小时数。不承诺短smoke速度代表多日负载。

参考旧100句每图约11.53物理round，直接线性外推3074图约79小时纯攻击；新test总体K/重叠可能不同，可先按约2–4天安排，prepare后查看实际估算。不要将旧100句2.5–3h当成全test耗时。

三组fp32像素约5.4GB，另加score/轨迹/cache，建议预留10GB以上。服务器当前剩余空间充足。没有自动启动full。
