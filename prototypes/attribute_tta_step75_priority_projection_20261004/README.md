# Step7.5：Score-Aware Priority Conflict Projection

本目录是独立prototype；不修改正式源码、AttributeInstance、ontology、S_raw、Top-K、JVP或官方TTA。不要commit/push。

固定读取Step7.4的100句selection_manifest.json及冻结clean score缓存。实验全部使用原Natural K，gamma=3、beta=1、10steps/round、累计8/255预算，无caption loss、无训练。此前审计“暂不正式去掉cap=4”仍成立；本轮uncapped只用于受控prototype。

## 三模式

1. PREFIX_BASELINE：原PREFIX均值guidance及官方IG聚合原样执行；记录各属性贡献，但实际返回原ig_grad张量。
2. SCORE_WEIGHTED_PREFIX：每个optimization step在同一scaled/transformed IG graph上计算各属性真实IG贡献，按S_i/sum(prefix S)合成。
3. SCORE_WEIGHTED_PRIORITY_PROJECTION：同上。历史g_H为前r-1项的S_raw加权均值；仅当新属性g_N与g_H的dot<0时，删除g_N沿历史的反向分量，随后按原S_raw权重合成。

不修改历史梯度，不对称PCGrad，不使用S^gamma。投影dot/norm在fp64运算，分母仅加fp64 machine epsilon（2.220446049250313e-16），结果回到原fp32更新路径；没有可调稳定超参数。

## 真实IG路径

real_ig.py对SHA冻结的官方ig、transform_ig_batch及img_attack AST添加捕获调用，保留所有原语句。每个属性梯度来自该step现有image forward graph，不新增image forward、IG路径或transform，不在未变换round-input诊断点代替真实IG。

真实聚合点：scaled IG + transformed IG → 每属性完整IG贡献 → weighting/projection → 原mean-abs normalization → 原momentum → 原sign → 原L∞ projection/clamp。

原loss为负image/JVP similarity，官方更新是加号，因此g_i为原loss的ascent方向。aug/block概率仍按各mode原聚合属性loss更新；image冲突投影不修改这些概率参数的算法。

无冲突时priority mode直接返回同一未投影weighted tensor。日志的“局部一致”是同状态、同组件的counterfactual；不能声称发生轨迹分叉后的两个mode在后续step仍有相同梯度。

## 随机控制与回归

每case只执行一次原单属性round1，image/momentum/observer/metrics/tape/RNG完整checkpoint由三模式共享。PREFIX_BASELINE记录完整categorical trace，其余模式回放，仍调用原multinomial维持RNG消耗；其余随机初始化与变换参数的RNG序列一致。

CPU回归检查原PREFIX与新baseline各轮image SHA、metrics、step observer、noise、trace、RNG。AST剥除新增capture/observer调用后必须与官方原方法完全相同。

正式输出记录与Step7.4历史结果的逐轮metric/image SHA差异，作为只读历史审计；不凭SHA不同宣称行为变化，不发明或放宽全局CUDA tolerance。共享round1及当前三模式控制必须exact。

## 执行边界

Codex仅允许CPU tests、2–4句短GPU smoke和少量debug。默认smoke：固定manifest中首个k=1，以及25488:1、24308:1，1step/round、1transform、scales=None。两句长K实际经过round5/6/7。四个历史关键case均保留在正式100句manifest中，不筛选。

正式100句只能由用户手动运行：

```bash
tmux new-session -d -s tbps-step75-priority 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -u prototypes/attribute_tta_step75_priority_projection_20261004/run.py --mode diagnostic --allow-diagnostic --manifest prototypes/attribute_tta_step74_dynamic_k_20261004/outputs/clean_scores/selection_manifest.json --output prototypes/attribute_tta_step75_priority_projection_20261004/outputs/diagnostic > prototypes/attribute_tta_step75_priority_projection_20261004/outputs/diagnostic.log 2>&1'
```

拒绝覆盖已有输出。完成标志：log末尾COMPLETE及diagnostic/completed.json。

## 输出

summary.json：三模式clean→final及两组逐层比较，全100句/k>4/各natural k；best-prefix retention。
per_caption_results.json：固定属性metadata/S/JVP、所有state/round/step、真实IG逐属性贡献、历史baseline差异。
per_round_results.json：逐轮ΔF/rank/cosine、recovery比例/幅度、实际update与clipping。
conflict_statistics.json：all/round2+/逐round/逐natural k/逐新属性axis的冲突计数、rate、cosine。
projection_statistics.json：修改比例/移除分量、no-conflict exact、历史保持、真实momentum路径及IG重构误差。
protocol.json / audit.json / completed.json：配置、来源SHA、冻结状态、随机控制、预算与完成标志。

正式每mode358个逻辑属性轮；共享100个round1，实际共874个optimization rounds、每轮10steps。各IG节点另计算属性贡献，成本高于Step7.4，不能据单步smoke声称正式速度。

smoke只验证机制与数值链路，不判断A/B/C/D或冻结方法。

## 已完成验证（2026-10-04）

36项CPU tests通过，显式UTF-8修复后34项相关回归再次通过。成功GPU smoke在outputs/smoke_utf8_verified/，固定34:0(k1)/24308:1(k7)/25488:1(k6)，每round1step，共36实际round。三模式round1完整状态/随机条件exact；priority round2+3冲突、8无冲突，真实momentum路径核验。冲突后cos最大abs3.54e-10，无冲突修改0。129旧来源/源码/cache SHA、IRRA冻结与预算通过。GPU已停止，正式100句未运行，outputs/diagnostic不存在。
