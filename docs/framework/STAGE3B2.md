# Stage 3B-2：严格CPU门禁后的单次真实G/H等价验证

基准 `a2988e0ae2a77f41f5416c02e354f4651ed49646`；分支 `codex/tbps-framework-stage3b2`。本次授权的一次真实尝试已完成，结果PASS，不能再次执行GPU命令。

## 最小新增入口

`src/tbps/smoke_attacks_gated.py` 强制调用Stage 3B-R的check_derived，再把其实际preflight返回的同一组请求/来源交给原smoke_attacks.run_once。严格CPU报告、检查器/派生证明和新入口SHA加入attempt来源锁定。

新入口默认只做CPU检查；GPU选项仍必须经过相同门禁。固定唯一新输出 `/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2`，已有目录拒绝，无恢复/重试分支。原equivalence_v1、原Smoke比较逻辑、工厂及历史攻击源码全部不改。沿用原75项派生来源配置，无新增/放宽pin。

`tests/framework/test_stage3b2_gate.py` 新增9项Mock CPU测试：默认无GPU、CPU失败阻断、CPU先于单次GPU调用、失败不重试、旧输出/其他输出拒绝、已用attempt拒绝、派生参数必需、实际预检必须执行一次、错误状态/来源数量拒绝。

## CPU命令（可重复，只读来源）

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage3b2-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.smoke_attacks_gated --config configs/framework/stage3br_preflight.server.json --derivation configs/framework/stage3br_derivation.server.json
```

预期退出码0、GH_RUNTIME_PREFLIGHT_PASS；JSON含strict_runtime_preflight.status=PASS、runtime_source_count=75。没有GPU参数时不调用run_once，不创建GPU attempt，也不恢复模型。

## 已执行的唯一GPU命令（审计记录，不可重跑）

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage3b2-20261016
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.smoke_attacks_gated --config configs/framework/stage3br_preflight.server.json --derivation configs/framework/stage3br_derivation.server.json --output /home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2 --allow-gpu-smoke
```

本次退出码0、GH_RUNTIME_PREFLIGHT_PASS、GH_PLUGIN_EQUIVALENCE_PASS。GPU阶段前强制再次进行严格CPU检查，沿用原run_once的资源门禁、单次模型恢复及计数/数值对照。原GPU脚本没有改动。

日志：`/home/lzf/ldx/outputs/tbps-framework-stage3b2/gpu_equivalence_v2.log`。
不可变结果：`/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2/{attempt,result}/`。
后续CPU只读核验：`/home/lzf/ldx/outputs/tbps-framework-stage3b2/readonly_audit_v1/result/`。

## 实际调用路径

- A图像：原pixels_from_pil读取 → 原inference_once.generate_once(同一冻结G, 原图)。
- B图像：JointPlugin.generate → GV2Plugin.generate → _LegacyG.load_pixels → 同一原generate_once；重新执行G，不复用A输出。
- A文本：原TextAttacker.attack(原Caption)，可编辑时调用同一冻结H。
- B文本：JointPlugin.generate → HV1Plugin.generate → _LegacyH.attack → 原TextAttacker.attack；独立生成记录，不复用A文本输出。
- Joint复用同一G/H插件实例，故意组合Gallery0与Query1的独立身份；image_only/text_only只调用对应组件。

A/B共用模型权重是固定设计；输入像素与结果来自两次独立路径。G hook分别采集真实delta，H完整记录逐字段exact，非比较单个返回值与自身。

## 测试与停止范围

332框架CPU（323原有+9新增）和36历史基础回归通过，共368项；GPU完成后框架回归再次通过。没有运行Full、训练、真实IRRA、跨模型或Stage 4。

对应固定两样本/FP32/math/no_tf32/RTX4090/CPU-H的输出等价可标记VERIFIED；插件execution_kind仍保留保守的legacy_unverified类型，不把本轮证据扩成所有输入/设备/运行模式的执行承诺。完整误差、Unicode动作、冻结与SHA见[报告](STAGE3B2_REPORT.md)。
