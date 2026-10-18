# 官方SGA算法 / IRRA Source 最小移植与执行说明

本入口直接调用固定官方Commit `f49e6926cd13148e49bdbc2c3e537660e096230c` 的 `SGAttacker.attack()`、`ImageAttacker`、`TextAttacker`，官方文件只读。IRRA是唯一攻击源；RaSa/APTM只消费冻结产物。不是此前SGA-TBPS White-box，也不是原论文ALBEF实验的完全复现。

状态、实际结果及限制见[Smoke报告](SGA_IRRA_SOURCE_SMOKE_REPORT.md)、[证据索引](SGA_IRRA_SOURCE_SMOKE_EVIDENCE.json)、[ADR-0010](../decisions/0010-sga-official-irra-source.md)。

## 已固定输入和算法

- CUHK官方Test顺序3074 Gallery / 6156 Query，保留每图全部原配对Caption（3073图各2句、1图10句）。不按同PID扩充攻击集合。
- Source像素：PIL RGB → bilinear Resize384×128 → ToTensor；官方攻击在此FP32空间投影。
- Source/Target共用Matched Clean：上述原始像素、官方 `pre_caption(raw,30)`；不能使用历史Raw Clean指标作为这条桥的Clean。
- 官方默认batch8、全局seed42；Smoke两图自然形成尾批2。不逐图重设随机种子。
- eps=2/255、10轮、step_size=0.5/255；原图及0.5/0.75/1.25/1.5尺度、原高斯噪声和BICUBIC增强。
- 原始BERT MLM、KL重要性、原过滤集合、图像引导负匹配候选选择；每段文本最多1词。第二段从原始Matched Caption重新开始。
- 使用global feature，所以 `cls=False`。BERT窗口30；IRRA仍使用原生77 Byte-BPE。候选转换逐条审计，额外IRRA截断拒绝，不静默删候选。
- 原FP32投影保持原样；实数差分可能比数学epsilon高约3e-8，单独报告，未添加二次投影。严格实数上界不应写成已通过。

## 环境与隔离

工作树：`/home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019`。

Source专用环境 `/home/lzf/ldx/envs/sga-irra-source/bin/python`：Torch2.0.0+cu118、torchvision0.15.0+cu118、Transformers4.28.1、tokenizers0.13.3、timm0.6.13、ftfy6.1.1；其余依赖继承原tbps-image。旧官方BertTokenizer在Transformers4.44初始化不兼容，因此新建隔离环境，未改官方Tokenizer或旧TBPS环境。原版本Tensor Resize的antialias默认行为保留。

Target环境 `/home/lzf/ldx/envs/tbps-image/bin/python` 保持原状；Source和各Target分别运行子进程，避免原生模型模块冲突。共享GPU仅观察，不结束其他进程；资源不足/OOM报错停止，不自动重试。

## CPU预检与测试

以下命令不加载神经模型，不初始化CUDA：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/src
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/sga-irra-source/bin/python -B -m tbps.baselines.sga_irra --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_source.full.server.json --scope full --action preflight
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/sga-irra-source/bin/python -B -m pytest -q tests/framework/test_sga_irra_source.py
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q tests/framework/test_sga_irra_smoke_scores.py
```

成功标志：`SGA_IRRA_CPU_PREFLIGHT_PASS full 3074 6156`。

## 本轮Smoke实际路径和原失败保留

固定Gallery `cuhk:37132:0`、`cuhk:37147:0`；各自两条Query，共2/4，未按效果挑样本。

原始命令已执行，**不可再次运行**：

```bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.sga_irra --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_source.server.json --scope smoke --action pipeline --allow-gpu-smoke
```

Source PASS，IRRA编码完成后通用Ranking因Gallery不足10而FAIL。原日志、execution FAIL、特征和Source产物全部保留。随后仅用CPU读回IRRA完整缓存；没有重新运行攻击或IRRA。独立两图评分桥使用相同官方 `rank(max_rank=2)` 报告R@1/mAP，R@5/R@10不适用；不放宽Full Engine门禁。RaSa/APTM各完成一次编码及原生融合评分。

已执行的后续动作（只为记录，不应重复发布已有输出）：

```bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.sga_irra_smoke_scores --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_smoke_scores.server.json --action recover-irra
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.sga_irra_smoke_scores --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_smoke_scores.server.json --action evaluate --model rasa --allow-gpu-smoke
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.sga_irra_smoke_scores --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_smoke_scores.server.json --action evaluate --model aptm --allow-gpu-smoke
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.sga_irra_smoke_scores --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_smoke_scores.server.json --action audit
```

当前完成证明：
- `/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/source/completed/`
- `/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/irra_smoke_scores/completed/`
- `/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/rasa/completed/`
- `/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/aptm/completed/`
- `/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/smoke_audit/completed/`
- `/home/lzf/ldx/outputs/tbps-framework-sga-irra-source/smoke/source_cpu_audit/completed/`

审计成功标志 `SGA_OFFICIAL_IRRA_TWO_GALLERY_CPU_AUDIT_PASS`。原pipeline的exit1没有被改签为0；组件完成和独立审计证据另存。

## 后续Full方案：需另行授权，本轮没有执行

Full派生配置保留攻击参数及原Source来源，只把最低空闲显存设为20480MiB，避免把两图所需资源当作batch8容量证明。batch8峰值尚未测量，20GiB只是保守启动门禁，不保证不OOM。需确认允许官方FP32预算舍入语义，不能擅自二次投影。

预计Source约1–2小时，三目标评分参考历史IRRA约4分钟、RaSa约2.64小时、APTM约2.38小时；总计约6–9小时，非保证。建议RAM至少48GiB、磁盘预留35GiB。原始3份RGB Tensor约5.066GiB；另有三模型独立特征/Score和审计日志。不得使用历史目标Embedding冒充本次Matched Clean/Adv特征。

**仅在用户另行批准后复制执行**：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019
mkdir -p /home/lzf/ldx/tmp/sga_irra_source_full
nohup setsid env PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/src PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /home/lzf/ldx/envs/tbps-image/bin/python -B -u -m tbps.baselines.sga_irra --config /home/lzf/ldx/projects/idea-test-tbps-sga-irra-source-20261019/configs/framework/sga_irra_source.full.server.json --scope full --action pipeline --allow-full >> /home/lzf/ldx/tmp/sga_irra_source_full/launcher.log 2>&1 < /dev/null & echo "launcher PID=$!"
```

该命令先一次IRRA Source全量生成，再分别IRRA/RaSa/APTM评估四组，最后CPU审计；目标不反传攻击。Full正常使用原EvaluationEngine及R@1/R@5/R@10/mAP，不使用两图Smoke评分桥。不输出mINP或DR表。

```bash
tail -n 30 /home/lzf/ldx/tmp/sga_irra_source_full/launcher.log
tail -n 30 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/source.log
tail -n 30 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/rasa.log
tail -n 30 /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/aptm.log
cat /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/audit.log
cat /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/irra/completed/metrics_2dp.csv
cat /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/rasa/completed/metrics_2dp.csv
cat /home/lzf/ldx/outputs/tbps-framework-sga-irra-source/full/aptm/completed/metrics_2dp.csv
```

最终成功标志 `SGA_OFFICIAL_IRRA_PIPELINE_PASS full` 及 `SGA_OFFICIAL_IRRA_CPU_AUDIT_PASS full`。已有目录拒绝覆盖；任一阶段失败即停止，不自动恢复、重新生成或删除Claim。Full尚未验证中断恢复；失败后先人工审查。
