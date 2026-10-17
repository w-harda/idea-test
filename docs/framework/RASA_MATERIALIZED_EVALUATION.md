# RaSa跨模型迁移评估：固定物化输入与原生融合评分

本入口只消费generation_v3封存的FP32 RGB/Unicode，不加载或优化G/H，不复用IRRA Embedding。官方源码固定 `bd16aa1a15f149548a90d196fcf10a27d7ab7c66`，位置 `/home/lzf/ldx/external/RaSa`；用户上传CUHK-PEDES权重 `/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth`，SHA `bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2`。严格恢复849项状态；使用本地BERT缓存，不在线下载模型。

## 复用与边界

- Dataset Adapter、官方3074 Gallery/6156 Query顺序、全部同PID正例和多Caption、EvaluationEngine、Ranking/五指标/DR及ArtifactStore继续复用。
- 只读输入桥复用原collection/select/bindings、BulkSession和原Store.load；生产入口只将attacks/materialization两个包指向冻结Stage4D1，保持原Plan和生产者路径。当前models/engines仍来自新工作树。默认CPU，GPU必须显式授权。
- RaSa Adapter不含G/H/PID或攻击奖励。输入桥中的G/H字段仅描述现有集合来源；换攻击样本时新增输入来源映射，Adapter及融合评分不需要修改。
- 新增materialized_native执行类型要求真实Adapter能力、source_verified、官方Test身份及Checkpoint来源；不把配置中的未实现模型自动变成可执行插件。
- AttackOrigin从固定G/H名称扩展为显式组件ID与单模态预算，Engine核验预算与模态。保留已知G/H名称的原有预算约束。仅描述产物来源，不提供新算法训练/生成入口。
- NativeFeatureCache保存原生Tensor字段；ChunkCachedScorer按原生Query块保存最终Score，可复用给其他已实现Adapter。模型原生表示/验证仍由各Adapter定义，不共享RaSa/IRRA特征空间。

## RaSa原生协议

官方 `configs/PS_cuhk_pedes.yaml` 与 `Retrieval.py::evaluation/itm_eval`：
- 384×384 BICUBIC，官方mean/std。
- 原官方pre_caption（小写、标点处理、最多50词）、官方WordPiece及最大50 Token。
- 图像局部Token [N,577,768]，文本 [Q,50,768]，Attention [Q,50]；投影CLS为256维、L2归一化。
- 全局相似度选择Top-128，最终仅使用融合输出 `itm_head(CLS)[:,1]`；其余候选分数-100，不把global相似度加回。
- 官方全PID正例、无camera过滤、降序argsort。mINP由既有Engine补充；官方RaSa itm_eval原有R@1/5/10和mAP，未提供mINP。
- FP32/math/no_tf32；编码batch1、融合microbatch1。Full保持128个候选；10+10 Smoke显式使用10个候选覆盖其完整小Gallery，不能当成Full的Top-128实验。

原物化像素为384×128，送RaSa时用torchvision Tensor BICUBIC/antialias调整至384×384，再做官方归一化。不转为uint8/PIL、不压缩、不改变已保存攻击像素。**这是固定物化输入上的Clean基线；不能宣称与直接从原始文件执行PIL Resize的论文Clean基线逐值相等。**相比官方直接读取高分辨率图片，多了一次既有384×128工作空间重采样；四组共享该输入约束，Full应报告自身Clean及DR。不得为接近论文指标更换输入或重生成攻击。

现有Transformers4.44与旧源码有两处初始化API差异，Adapter只在隔离加载边界兼容：丢弃废弃的文档装饰器tokenizer_class参数；在官方Tokenizer父类初始化前读取同一词表。官方外部源码、模型forward、词表及WordPiece规则不改。命名空间退出后恢复，原权重strict=True载入。依赖弃用/FutureWarning保留日志，不隐藏。

## 四组和成本

Clean=原RGB+原文；Image-only=Adv RGB+原文；Text-only=原RGB+历史H文；Joint=Adv RGB+历史H文。全部Query保留，目标分数/Tokenizer结果不反馈给攻击器。不同目标自行编码同一产物。

首次Full预计6148个图像样本编码、12312个文本样本编码、3151872个融合pair（4×6156×128）。图像局部Token纯数据约10.149GiB、文本约1.761GiB，另有global、mask、Score及协议/回执。建议预留至少20GiB新输出磁盘和约30GiB可用RAM；与仅缓存512维IRRA特征的资源量不同。Full需要大量逐pair融合，真实长程耗时由运行日志确定，不能按10+10总耗时简单放大。

## 服务器执行命令

以下环境固定，工作树、原冻结库和输出不要移动。长程Full由用户亲自启动；Codex本轮不执行Full。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src
```

CPU Full预检（无模型前向、CUDA、排名或正式Attempt创建）：

```bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u   /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src/tbps/rasa_eval.py   --config /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/configs/framework/materialized_rasa.server.json   --scope full
```

用户后台Full及成功后独立CPU审计（完整复制；运行前确认full目录尚不存在）：

```bash
nohup bash -c '
set -euo pipefail
cd /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017
export TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src
test ! -e /home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u  /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src/tbps/rasa_eval.py  --config /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/configs/framework/materialized_rasa.server.json  --action run --scope full --allow-full-evaluation  > /home/lzf/ldx/tmp/rasa_materialized_full_gpu.log 2>&1
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u  /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src/tbps/rasa_eval.py  --config /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/configs/framework/materialized_rasa.server.json  --action audit --scope full  > /home/lzf/ldx/tmp/rasa_materialized_full_audit.log 2>&1
echo RASA_FULL_AND_CPU_AUDIT_SUCCESS
' > /home/lzf/ldx/tmp/rasa_materialized_full_supervisor.log 2>&1 < /dev/null &
echo $!
```

独立CPU审计也可单独执行：

```bash
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -u  /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src/tbps/rasa_eval.py  --config /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/configs/framework/materialized_rasa.server.json  --action audit --scope full
```

成功标志：GPU `RASA_MATERIALIZED_EVALUATION_PASS full`；CPU `RASA_MATERIALIZED_CPU_AUDIT_PASS full`；外层 `RASA_FULL_AND_CPU_AUDIT_SUCCESS`。GPU状态仅记录，允许共享，不干预Ollama；CUDA OOM保留完整堆栈并退出，无自动重试。

## 查看结果、停止与恢复

```bash
tail -n 30 /home/lzf/ldx/tmp/rasa_materialized_full_gpu.log
tail -n 20 /home/lzf/ldx/tmp/rasa_materialized_full_audit.log
cat /home/lzf/ldx/tmp/rasa_materialized_full_supervisor.log
cat /home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/completed/metrics.csv
cat /home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/full/execution-000001/report.json
ps -eo pid,etime,args | grep '[r]asa_eval.py'
nvidia-smi
```

Feature进度为RASA_FEATURE_PROGRESS，融合/Score进度为NATIVE_SCORE_PROGRESS（每128 Query一个已提交块）。最终summary保存五指标、五DR及Joint−Image-only mAP。结果、缓存与执行报告只写 `/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/{smoke,full}/`，与IRRA/Stage4D隔离。

需要停止时可向实际Python PID发送SIGINT，保存KeyboardInterrupt/FAIL回执；SIGTERM/SIGKILL可能留下未闭合Attempt，须先检查，不能删除证据自动恢复。只有无pending、已有Attempt均具execution记录且来源/配置不变时，手动使用原Full命令追加 `--resume`，输出新日志并保留旧日志。已验证的特征块和融合Score块复用；未提交部分可能重新计算RaSa，G/H始终不运行。禁止自动恢复或修改失败证据。已完成时重入只执行审计，不重新编码/排名。

```bash
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -u  /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/src/tbps/rasa_eval.py  --config /home/lzf/ldx/projects/idea-test-tbps-rasa-materialized-eval-20261017/configs/framework/materialized_rasa.server.json  --action run --scope full --allow-full-evaluation --resume
```

Smoke实际结果、CPU测试和保护证据见RASA_SMOKE_REPORT.md。其指标仅用于检查链路，不宣称迁移攻击有效；正式结论等待用户Full结果。本轮不接入APTM或新攻击算法。
