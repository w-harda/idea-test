# APTM物化跨模型评估：CPU就绪，GPU尚未执行

本轮仅开发/CPU验证。**不得在RaSa Full运行或等待器可能触发期间启动APTM GPU任务。** 未启动APTM Smoke、Full、后台任务或新的G/H生成。

## 官方资源与复用

- 官方源码：https://github.com/Shuyu-XJTU/APTM ，固定Commit `b20025445471eb4b743164a6faec29a9eccfb293`。
- 服务器源码：`/home/lzf/ldx/external/APTM`。服务器直连GitHub克隆未完成，保留其临时目录；通过本机获取完整Git bundle后传入，Commit一致、官方checkout clean，未修改第三方代码。
- 官方README权重入口：https://drive.google.com/drive/folders/1N1Lumvb4epP0awHLcJ3RzQmv5zwrAFBh 。
- 官方CUHK包：https://drive.google.com/file/d/1dokRhjYL71VbzFfLfyyDmh7MXUOyBySf/view ，`ft_cuhk.zip` 793576008字节，下载成功。
- 解压权重：`/home/lzf/ldx/checkpoints/APTM/aptm_cuhk_checkpoint_best.pth`，859801179字节。
- Checkpoint SHA-256：`521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c`。
- CPU结构验证：官方APTM_Retrieval、629项状态strict全部匹配、214456629参数，CUDA未初始化。模型GPU恢复及真实编码仍PENDING。
- BERT使用已存在官方bert-base-uncased本地snapshot；33项APTM源码/配置/BERT文件和运行版本单独锁定。源配置记录受害者训练来源，与G/H攻击器训练来源分开。
- 工作树：`/home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018`；分支：`codex/tbps-aptm-materialized-evaluation`。

复用CuhkPedes、MaterializedStore/BulkSession、封存输入校验、NativeFeatureCache、EvaluationEngine、原指标及结果发布/读取；物化encode/bound_inputs复用RaSa阶段已有的模型无关函数，不修改其实现。APTMAdapter不接收G/H、PID、奖励或攻击反馈。当前生产输入桥仍限定generation_v3；未来其他攻击需接入其物化输入桥，不必修改APTM编码/评分。

新增：
- `models/aptm.py`：官方来源、加载、预处理/分词、原生图文/融合接口。
- `engines/aptm_score_cache.py`：原始分块ITM/全局Score缓存，完整Query集合归一化及结果读取。
- `engines/materialized_aptm.py`、`aptm_eval.py`：独立CLI与四组接线。
- 两份APTM来源/执行配置、CPU测试及本文。

## 原生协议及与IRRA/RaSa差异

| 项目 | IRRA | RaSa | APTM |
|---|---|---|---|
| 图像尺寸 | 384×128 | 384×384 | 384×128 |
| 图像编码 | CLIP ViT | ViT局部Token | 官方Swin-B，49×1024 Token |
| RGB归一化 | CLIP均值/方差 | CLIP均值/方差 | CUHK均值 .38901278/.3651612/.34836376，标准差 .24344306/.23738699/.23368555 |
| 文本 | Byte-BPE，77 | BERT，50 | 官方BERT WordPiece，56 words/56 tokens |
| Caption处理 | 原IRRA tokenize | 原pre_caption | CUHK pre_caption仅截56 words；不套用RaSa标点清洗，不启用EDA |
| 原生检索分数 | 单位特征点积 | Top128融合ITM正类logit，其他-100 | Top128融合；未选项填本行sentinel矩阵最小值；完整矩阵minmax(ITM)+0.002×minmax(global) |

APTM评分来自官方 `reTools.py::evaluation`；编码直接调用官方 `get_vision_embeds/get_text_embeds/get_cross_embeds`。Full固定Top128，Smoke仅10候选，是明确小样本例外。单进程、FP32/math/no_tf32、microbatch1；没有改变模型、权重或打分公式。

**APTM原始评分不具有Query块独立性。** 原始Score可按128 Query缓存，但必须汇合完整Query/Gallery之后再归一化，不能逐块minmax。APTMAdapter明确query_independent_scores=False；APTMSetScorer只在已绑定的完整集合内读取最终Score切片，Engine复用原排名逻辑。常量矩阵会令官方分母为零，明确拒绝，不加epsilon悄悄改变公式。

四组共享generation_v3同一3074 Gallery/6156 Query，全部同PID正例，无camera过滤。GPU只运行APTM，不恢复G/H/IRRA/RaSa。Clean是已物化384×128 FP32原RGB上的目标基线；APTM BICUBIC Tensor处理保留FP32，不从Adv转回PIL量化，也不宣称等同于原始高分辨率文件直接PIL Resize的论文基线。

五项指标与官方 `reTools.py::mAP` 在独立合成Fixture逐项一致；单位为百分数。DR=(Clean−Adv)/Clean×100，Clean=0返回null；报告Joint−Image-only mAP及全部五项DR。

## CPU验证与边界

- 36项新增APTM CPU测试；相关必要回归合计140项通过：新树139项＋原IRRA工作树1项绝对路径来源测试。未修改旧测试来适配新路径。
- 独立执行官方完整evaluation函数配合CPU Mock，对照新Adapter Score，固定误差≤1e-6；覆盖Top128填充和全矩阵归一化。
- 129 Query跨128块测试，验证一次完整集合归一化、缓存复用及篡改拒绝。
- 覆盖原生Token/FP32/归一化、像素非量化、Unicode、非法来源/精度/batch/跨ModelSpace拒绝、namespace/RNG恢复、默认CPU与显式GPU许可。
- Full CPU预检PASS：3074/6156、Plan `7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5`；完整保留6116编辑、40 NOOP、13537替换。首尾两个RGB仅CPU读回，其余编码前逐单元校验；原generation_v3独立审计作为全量内容来源。
- 10+10 CPU预检PASS；固定Gallery cuhk:37132:0至cuhk:37141:0，Query为37132至37136各caption_index 0/1，均有正例且包含负例。不是6156张独立图像。
- CPU证据：`/home/lzf/ldx/tmp/aptm_development/{full_cpu_preflight,smoke_cpu_preflight,cpu_tests_final}.log`。
- 13项当前执行源码SHA、33项官方/BERT来源锁定；原RaSa11项producer SHA不变，RaSa/IRRA/Stage4D1工作树clean。没有改动旧实验源码或正式输出。
- 旧timm导入、Tokenizer默认值及meshgrid的兼容警告保留，不修改官方代码来消除警告。
- 未执行APTM真实GPU、真实排名或Full；不把CPU Mock等价当成真实模型数值等价/迁移攻击效果。

## 用户执行：环境与CPU预检

以下均为真实路径/CLI。所有GPU命令仅准备，**本轮Codex未执行**。先确认RaSa任务已结束、等待器不再可能与APTM冲突，再运行Smoke。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/src
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/src/tbps/aptm_eval.py --config /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/configs/framework/materialized_aptm.server.json --action preflight --scope full
```

成功标志 `APTM_CPU_PREFLIGHT_PASS`。预检不创建GPU Attempt，不加载模型，不初始化CUDA；Checkpoint仅SHA检查。权重由配置绑定，无需在命令重复指定。

## 用户执行：固定10 Gallery＋10 Query真实GPU Smoke

沿用上面的工作目录及环境；明确只选10+10后编码，绝不先编码Full再切片。

```bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/src/tbps/aptm_eval.py --config /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/configs/framework/materialized_aptm.server.json --action run --scope smoke --allow-smoke > /home/lzf/ldx/tmp/aptm_materialized_smoke.log 2>&1
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/src/tbps/aptm_eval.py --config /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/configs/framework/materialized_aptm.server.json --action audit --scope smoke > /home/lzf/ldx/tmp/aptm_materialized_smoke_audit.log 2>&1
```

预期APTM恢复1次、20图像样本、20文本样本、400融合pair；Hook核对计数与模型参数/buffers前后一致。G/H/IRRA/RaSa恢复/前向、Backward/Optimizer/JVP为0。Tokenizer原/攻击ID变化记录在execution报告。Smoke指标只用于链路诊断。

成功标志：`APTM_MATERIALIZED_EVALUATION_PASS smoke`、`APTM_MATERIALIZED_CPU_AUDIT_PASS smoke`。

## 用户执行：Full（Smoke验收通过后）

沿用环境。下列命令只在用户明确决定执行时使用，不加入自动测试或RaSa等待器。

```bash
nohup env CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/src/tbps/aptm_eval.py --config /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/configs/framework/materialized_aptm.server.json --action run --scope full --allow-full-evaluation > /home/lzf/ldx/tmp/aptm_materialized_full.log 2>&1 < /dev/null &
echo $!
```

成功后由用户CPU只读审计，不再编码、前向或排名：

```bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/src/tbps/aptm_eval.py --config /home/lzf/ldx/projects/idea-test-tbps-aptm-materialized-eval-20261018/configs/framework/materialized_aptm.server.json --action audit --scope full > /home/lzf/ldx/tmp/aptm_materialized_full_audit.log 2>&1
```

Full成功标志：`APTM_MATERIALIZED_EVALUATION_PASS full`；审计成功：`APTM_MATERIALIZED_CPU_AUDIT_PASS full`。结果：
`/home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/{smoke,full}/completed/{metrics.csv,summary.json}`。
来源/运行设置在protocol，加载/Hook计数、冻结、RSS、显存、耗时及完整错误在execution-000001/report.json。

进度与结果查看：

```bash
tail -f /home/lzf/ldx/tmp/aptm_materialized_full.log
cat /home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/completed/metrics.csv
cat /home/lzf/ldx/outputs/tbps-framework-aptm-materialized-evaluation/full/execution-000001/report.json
sha256sum /home/lzf/ldx/checkpoints/APTM/aptm_cuhk_checkpoint_best.pth
```

停止用户启动的任务：先用ps核实本任务PID，再kill -TERM该PID。正常捕获异常发布FAIL；SIGKILL/OOM等导致未关闭Attempt或pending时必须先人工审查，不能删除现场。只有已完整关闭Attempt、无pending且原协议一致时，原run命令增加`--resume`才能复用已核验特征/原始Score批次；没有自动重试。G/H不会重复生成。audit必须在评估成功后执行。

## 成本预案

Full预期6148图像编码、12312文本编码和3151872融合pair；实际数以Hook/完成报告为准。模型参数FP32约0.799GiB，推理激活/加载峰值尚未实测，不能将RaSa峰值当作APTM保证。建议先在GPU空闲时执行Smoke，确认峰值再决定Full。

结构化图文特征纯Tensor约3.14GiB，四组原始ITM/global及最终/result Score约1.13GiB；还需绑定JSON、SHA、不可变回执及加载临时空间，建议至少10GiB独立可用磁盘、16GiB可用系统RAM。流式编码不缓存所有RGB；原生Token在CPU保存并按pair送GPU。最终Score只保留一个当前模式的额外缓存副本。

耗时主要由约315万融合pair决定；尚无真实APTM GPU吞吐，不给出伪精确Full时长，待10+10实测更新。6156 Query不等于6156 Gallery。无新训练或其他攻击方法接入。
