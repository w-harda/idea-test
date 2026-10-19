# VLPTAttack（DRA）官方算法 / IRRA Source

本阶段仅固定官方Test前8张Gallery及全部16条Caption的Smoke。没有正式Full执行入口。
官方仓库SensenGao/VLPTransferAttack，Commit 91f78325e7cf0c497c9bbb5be2215be652d8cd9d。
核心来源与保真结论见DRA_OFFICIAL_FIDELITY_AUDIT.md；不得修改或运行时替换官方攻击方法。

## 最小接线

- baselines/dra_official.py：SHA、完整顺序/多Caption、预算读回和只读sys.settrace，不含攻击更新或候选决策。
- models/dra_irra_source.py：继承已验证IRRASGASource图文I/O，仅开启原官方候选评分backward所需参数梯度；无Optimizer。
- baselines/dra_irra.py：CPU预检、单批官方Attacker.attack、Claim/紧凑FP32像素存储。
- baselines/dra_target.py：复用三个原生Adapter、FeatureCache、EvaluationEngine身份验证和官方rank(max_rank8)。R@10不适用，不补图、不放宽Full门禁。
- configs/framework/dra_irra_source.server.json：首次缺少safetensors的环境失败配置，永久保留。
- configs/framework/dra_irra_source.smoke_v2.server.json：补齐依赖后的新输出与配置。
- tests/framework/test_dra*.py：合成CPU官方完整循环、真实Tokenizer、来源/顺序/存储/评分。

官方参数8/255、10轮、2/255步长、5候选、原整数三角区域采样、五尺度、MLM topk10/threshold .3、最终1词预算、.6/.2/.2。
实际首轮没有历史三角搜索，后9轮各5候选，共45次rand3Num和100次图像backward，保留所有原步骤。

Source固定IRRA与BERT MLM，eval、权重不更新；参数梯度不等于参数更新。仅在整个官方攻击结束后统一清理梯度并关闭开关，再核验权重/buffers。Target全冻结，无反馈。
BERT77、IRRA77与pre_caption30分别绑定；候选WordPiece往返必须exact，额外IRRA截断拒绝。

## 环境与实际命令

工作树/home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019。
Source环境/home/lzf/ldx/envs/dra-irra-source：Torch2.1.0+cu121、torchvision0.16.0+cu121、Transformers4.28.1、safetensors0.4.5。
Target环境/home/lzf/ldx/envs/tbps-image。未改SGA和旧Target环境。
首次pip旧版依赖解析失败后，仅修新环境包安装；首次GPU在MLM加载前因缺少safetensors退出。旧FAIL保留，新配置不覆盖旧来源。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/src
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES='' /home/lzf/ldx/envs/dra-irra-source/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/configs/framework/dra_irra_source.smoke_v2.server.json --action preflight
```

以下为本轮真实调用格式，已消费的Attempt不得重复执行或删除目录重试：

```bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/dra-irra-source/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/configs/framework/dra_irra_source.smoke_v2.server.json --action source --allow-gpu-smoke
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/configs/framework/dra_irra_source.smoke_v2.server.json --action evaluate --model irra --allow-gpu-smoke
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/configs/framework/dra_irra_source.smoke_v2.server.json --action evaluate --model rasa --allow-gpu-smoke
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/configs/framework/dra_irra_source.smoke_v2.server.json --action evaluate --model aptm --allow-gpu-smoke
CUDA_VISIBLE_DEVICES='' /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.dra_irra --config /home/lzf/ldx/projects/idea-test-tbps-vlpt-irra-source-20261019/configs/framework/dra_irra_source.smoke_v2.server.json --action audit
```

新输出/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source-v2/smoke/；旧失败输出/home/lzf/ldx/outputs/tbps-framework-vlpt-irra-source/smoke/source/不变。
CPU预检DRA_CPU_PREFLIGHT_PASS；Source DRA_OFFICIAL_IRRA_SOURCE_PASS；目标DRA_NATIVE_TARGET_PASS；独立审计DRA_INDEPENDENT_CPU_AUDIT_PASS smoke。

## Full下一步

另行授权后只增加范围/调度入口，官方顺序/Batch8/全部Caption，385批；仍直接调用同一官方攻击器。
固定一次3074像素/6156文本后送三个Target。正式Full走原EvaluationEngine.run，保留全部同PID正例，仅R@1/5/10/mAP两位小数；Matched Clean独立保存。
本阶段不部署后台、不开始Full，不调整算法或根据Smoke收益选择样本。8图Smoke不覆盖10-Caption batch GPU峰值、全量上下文边界与长期内存趋势。
