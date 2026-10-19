# VLPTAttack（DRA）官方算法 / IRRA Source

本阶段固定官方Test前8 Gallery及全部16 Caption，Source、三个Target和CPU审计已PASS。没有Full执行入口或后台任务。
保真结论见DRA_OFFICIAL_FIDELITY_AUDIT.md；结果、资源和完整失败记录见DRA_SMOKE_REPORT.md；原精度/来源SHA见DRA_SMOKE_EVIDENCE.json。

## 目录和复用

- src/tbps/baselines/dra_official.py：冻结来源、多Caption/顺序、预算、只读sys.settrace。
- src/tbps/models/dra_irra_source.py：复用IRRASGASource原生I/O，仅管理官方候选backward所需参数梯度。
- src/tbps/baselines/dra_irra.py：默认CPU预检、固定单批官方Attacker.attack、Claim/紧凑像素存储。
- src/tbps/baselines/dra_target.py：三个原生Adapter、FeatureCache、EvaluationEngine校验及官方rank(max_rank8)；无自写排名或攻击Loss。
- tests/framework/test_dra*.py：CPU官方原函数执行/梯度、Tokenizer、映射、Storage和原生评分契约。
- configs/framework/dra_irra_source.server.json / smoke_v2：失败配置保留。
- configs/framework/dra_irra_source.smoke_v3.server.json：成功Source的独立来源。
- configs/framework/dra_irra_targets.server.json：成功Target会话协议。
- configs/framework/dra_irra_audit.server.json：最终独立CPU审计来源。

官方核心未复制、修改、继承覆盖或Monkey Patch。三阶段、8/255、10轮、2/255步长、5候选、原整数采样、五尺度、MLM/KL、1原词槽位替换、.6/.2/.2原样。
官方首轮无历史区域搜索，后9轮各5候选。不能因为“1词”而过滤原代码产生的多词短语候选。

## 当前CPU预检命令

Source环境/home/lzf/ldx/envs/dra-irra-source为独立Torch2.1.0+cu121、torchvision0.16.0+cu121、Transformers4.28.1、safetensors0.4.5。
Target环境/home/lzf/ldx/envs/tbps-image未改。原SGA环境、Checkpoint、缓存、Full均不变。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-vlpt-official-complete-20261019
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-vlpt-official-complete-20261019/src
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' /home/lzf/ldx/envs/dra-irra-source/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-official-complete-20261019/configs/framework/dra_irra_audit.server.json --action preflight
```

默认不初始化CUDA；完整3074/6156身份及10-Caption样本只做CPU来源验证。没有Full参数，不能直接启动全量生成。

## 实际执行命令索引（已消费，不可重跑）

以下是已执行版本的精确命令，供审查来源；不可删除目录/Claim重做。三套工作树因保护已锁定运行源码而分别保留：

```bash
# 成功Source：8图/16文，唯一已封存攻击批次。
PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-vlpt-smoke-publication-20261019/src CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/dra-irra-source/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-smoke-publication-20261019/configs/framework/dra_irra_source.smoke_v3.server.json --action source --allow-gpu-smoke
# 成功Target入口；--model依次irra、rasa、aptm。没有重新生成Source。
PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-vlpt-target-smoke-20261019/src CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-target-smoke-20261019/configs/framework/dra_irra_targets.server.json --action evaluate --model irra --allow-gpu-smoke
# 最终CPU审计已发布；再次执行会拒绝覆盖完成回执。
PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-vlpt-official-complete-20261019/src CUDA_VISIBLE_DEVICES='' /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-official-complete-20261019/configs/framework/dra_irra_audit.server.json --action audit
```

输出/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v3/smoke/：
source、irra-session-v2、rasa、aptm、audit完成；旧irra保持FAIL和完整特征。
更早Source环境/保存审计FAIL分别在tbps-framework-vlpt-irra-source和tbps-framework-vlpt-irra-source-v2，不覆盖。

成功标志DRA_OFFICIAL_IRRA_SOURCE_PASS smoke、三个DRA_NATIVE_TARGET_PASS、DRA_INDEPENDENT_CPU_AUDIT_PASS smoke。
8图的R@10不适用，不补图；Smoke指标不得当正式Full结论。

## 下一步授权范围建议

仅增加Full范围/调度入口，按官方Batch8/全局seed42/3074图及6156原配对Caption直接调用同一官方算法，385批。
Source一次固定图文产物，三个Target串行独立进程原生评分，Full复用EvaluationEngine.run，保留全部同PID正例和Matched Clean，只输出R@1/5/10/mAP两位小数。
保持FP32、原Loss/候选/预算，不为速度近似或删步骤。当前详细被动追踪单批122.22秒，Source线性约13小时，连三个Target暂预留18–22小时；未来实际长Caption/候选/RNG批次会变化。
建议20GiB空闲GPU、48GiB RAM、35GiB磁盘。10-Caption batch GPU峰值、全量Token边界和长期稳定性尚未覆盖；不声称Full已PASS。
本阶段结束，不启动Full或新的算法开发。
