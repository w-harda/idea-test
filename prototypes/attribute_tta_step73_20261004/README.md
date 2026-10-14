# Step7.3：受控 Attribute-Guided IRRA-TTA（独立 prototype）

真实目录：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step73_20261004/`。所有源码、tests及原始outputs仅服务器；正式Stage03/Top-K/旧TTA/generator未改，不commit/push。

## 模式与公平性

- NEW：冻结AttributeInstance的block8方向，经原剩余text forward JVP得到v，原TTA loss仅为`-image_feature·v`，不加caption term、不重新归一化v。
- RAW：仅编码选中实例已有raw_text/source span，经同一frozen IRRA encode_texts真实路径得到最终retrieval text feature；不加caption term，不编码canonical模板。
- NONE：沿用原TTA的caption-only图像目标`-image_feature·full_caption_feature`，去掉属性项；不是零梯度/no-op。仍严格执行同一个clean Top-K plan的k轮，不能解释为原vanilla协议的固定两轮。
- 可选LEGACY：caption+raw两个负相似度项之和；比NEW/RAW多caption监督，不是严格controlled baseline。默认不运行，需手动加`--include-legacy`。

全部使用Step7.2已完成固定25句、存储gamma3/beta1/kmax4的clean plan；不重算S或Top-K。初始化噪声、steps、step_size、scales、transforms、momentum=.9、累计epsilon=8/255投影相同。原TTA每轮噪声初始化仅clamp[0,1]，内部首点可能越过累计epsilon；每个完成step/round均检查预算，保留原行为。

原TTA的aug/block采样概率依赖loss而动态变化，单纯相同seed不能保证相同实际变换。NEW作为预先固定的reference记录真实categorical block/op选择，RAW/NONE/可选LEGACY回放其选择；各模式仍保留原概率权重及其梯度更新。回放仍调用原采样器以保持RNG消耗，并强制核验所有模式的选择记录、每轮噪声SHA与最终RNG指纹一致。NEW的采样及数学更新不变；anchor没有按结果选择。

## 记录定义

每句包含原caption、candidate m、固定selected metadata/S_raw/source span/representation、JVP核验；每mode保存：
- `states[0..k]`：paired cosine、paired soft-rank F、paired-image hard rank及tie数。
- `rounds`：instance ID/region/axis/value/S_raw；round input上的image gradient norm；pre-projection net proposal、post-projection actual net change L2/L∞；累计L∞；与上一round实际净更新cosine。
- `rounds.steps`：每个优化step的pre/post更新、被投影移除量、clipping coordinate fraction、IG gradient与momentum norm。
- `metric_delta`和`clean_to_final`：逐轮及总变化，F增大/cosine下降表示对应mechanism指标恶化。
- `actual_round_update_pairs`：全部round净更新对的cosine，包含same-axis/different-region。round处于不同累计状态，不能据此证明纯语义分离。
- `random_trace`/`noise_sha256`/`rng_end_sha256`及冻结checks。

round pre-update定义为最后一步未投影proposal减round输入；post为完成round图像减round输入，包含原TTA随机初始化与全部步。每个step另有完整局部记录。clipping ratio是proposal与projected image不同的坐标比例。removed L2/requested L2在原noisy内部点越界时可大于1，不当作概率。`actual_change_linf<=1e-8`只作numerical almost-zero诊断，另报exact zero；不影响scheduler。

原34054 clean train gallery复用并核验文件SHA、annotation/checkpoint/config/官方源码/torch版本、paths及normalized features。F继续`1+sum(j!=paired)sigmoid((s_j-s_plus)/.02)`；固定其它clean图分数，只动态评价当前paired image。hard rank为1+严格高于paired score的竞争图数，ties另报。不是PID first-hit rank、不是正式共享adversarial gallery性能，不生成任何adversarial dataset。

## Tests与短smoke（已完成）

29项CPU tests通过，包含旧Step7.2回归；显式UTF-8读取并有进程locale回归测试。最终日志`outputs/tests_final.log`。

仅固定35:1/0:1，分别k2/k3；三mode每属性2steps、1transform、scales=None，15个属性轮、30个优化step。最终产物`outputs/smoke_observer_verified/`，日志同根`outputs/smoke_observer_verified.log`；完成耗时7.45秒，两个case阶段约2.44秒。smoke不是默认25句攻击效果比较，不能据它冻结优势结论。

51项旧source/checkpoint/cache SHA前后保持；IRRA/metadata/basis冻结、zero exact、noise/trace/RNG控制及输出格式通过。NEW每轮图像与原Step7.2逐值一致，最终momentum最大差3.814697e-6，通过Step7.2已有atol=2e-6/rtol=1e-6；不能说momentum bitwise exact。最大JVP FD相对误差.094374%，最大L∞=.03137257695198059，比实数8/255多2.79e-8，保留fp32舍入事实。

早期smoke/Debug/Final/Verified快照保留：曾因严格momentum回归、默认字符编码和观测梯度工作区而停止。最终将额外诊断梯度置于原更新之后，未改loss/TTA/样本/超参，并完成两句每轮图像回归；以smoke_observer_verified/completed.json为成功产物。

核心source SHA：
- controlled_tta.py：cace0619fd91a75869a99b3d3db5ef2fb1ddffba7559e1cf11b612698a5d71ed
- run.py：1ace9ab7d2cafa3fb0966dfecaba1fd42c0e7b262c725e0ca75df3c9f4bdc075
- test_controlled_tta.py：180e075c471e5be2739f2fd7ff4ca70c821e89555fe53cfe6e532420db363175

## 用户手动固定25句（Codex未执行）

```bash
tmux new-session -d -s tbps-step73 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step73_20261004/run.py --mode diagnostic --allow-diagnostic > prototypes/attribute_tta_step73_20261004/outputs/diagnostic.log 2>&1'
```

完整设置沿Step7.2：每属性10steps、6transforms/scaled image、scales=(.5,.75,1.25,1.5)。默认三mode，每mode75个属性轮。4090上Step7.2实测约10.2分钟，三mode估计30–50分钟；不是由smoke减配速度直接外推，实际依首个CASE_DONE再估。可选LEGACY约再增加一份攻击计算。

log：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step73_20261004/outputs/diagnostic.log`。
output：同prototype `outputs/diagnostic/`，包括protocol/results/summary.json/summary.md/audit/completed.json。拒绝覆盖已有output；如需重新执行，显式另给新`--output`和log路径，不删除旧证据。

完成标志：log末尾`COMPLETE`且`completed.json`的case_count=25、modes=NEW/RAW/NONE、attribute_rounds_per_mode=75、SHA/trace/zero/frozen checks=true。
优先看：fixed_sequence、NEW/RAW caption_loss_terms=0、random trace/noise/RNG一致、states长度k+1、每轮metric_delta、actual net change、steps clipping、summary的逐句wins/ties/mean/median及轴/轮次统计。两个历史重复representation实例不去重、不改Top-K。

## 正式共享图库准备（只审计）

既有`scripts/run_ap_duke_irra_baseline.py::evaluate`/`src/attributes/tbps_retrieval_metrics.py`已采用全3074 test images、6156 captions、PID positives及与官方IRRA rank一致的R@1/5/10/mAP/mINP，可以复用指标/encoder/查询协议。

当前CLI与AP generator spec/progress/native tensor cache及“generation不使用victim”的metadata绑定，不能直接指向本prototype的每caption状态摘要当作新TTA gallery。NEW尚无一图一固定条件的gallery导出/manifest adapter，本轮不实现或自定多caption聚合，因此没有可直接执行的NEW full-gallery命令。后续仅在用户25句结果支持时准备该adapter；不使用历史paired-gallery replacement做正式结果，不启动full gallery。

实际runtime的src/attributes/irra_victim.py另核验与未修改正式HEAD完全一致，SHA=9489ad4007ec684ac0cab838bfe60a12e2fa82d554c6519b6a1a9442985d5c29。手动diagnostic guard因此为52份；成功smoke原51份协议不改写，补充审计见outputs/source_audit.json。gallery旧encoder wrapper沿原spec保留来源，未假装与当前wrapper源码相同；不重提gallery。
