# RaSa接入与真实10+10 GPU Smoke报告

状态：PASS。独立分支 `codex/tbps-rasa-materialized-evaluation`，基准 `b51d9e55190e96b6422f0c87952471ad6ac12bed`。只执行一次短时真实Smoke，未启动RaSa Full、APTM、新攻击生成或训练。

## 来源与实现

官方仓库 `https://github.com/Flame-Chasers/RaSa` 固定Commit `bd16aa1a15f149548a90d196fcf10a27d7ab7c66`，只读部署于 `/home/lzf/ldx/external/RaSa`。用户上传 `/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth`，大小1816575847字节，SHA `bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2`。与官方CUHK下载文件名称相符，内容仅model，严格恢复849项State；未含Epoch/Optimizer元数据，不推断具体训练Epoch。

本地BERT snapshot 86b5e0934494bd15c9632b12f734a8a67f723594，词表/配置/模型文件与官方40项代码及资源共同绑定。环境Torch2.5.1+cu124、torchvision0.20.1+cu124、Transformers4.44.2、timm0.9.8；离线加载，不修改原环境包或官方源码。

新增文件：
- models/rasa.py：来源校验、官方加载隔离、BERT兼容边界、原生编码/Top-128 ITM。
- engines/materialized_inputs.py：封存输入只读桥，复用原collection/select/bindings、CPU门禁与Store读取。
- engines/native_cache.py、native_score_cache.py：局部Token/原生Tensor缓存与分块Score恢复，不绑定攻击算法。
- engines/materialized_rasa.py、rasa_eval.py：默认CPU入口、受控GPU/审计、生产者路径桥。
- 两份独立配置、两个CPU测试文件、执行README与报告。
- Engine只补materialized_native执行权限；AttackOrigin/Engine按显式组件与预算隔离，保留G/H预算规则，旧IRRA路径不变。原Stage4A对应测试更新为拒绝空身份，另加AP来源CPU Fixture验证；这不实现AP攻击。

数据使用同一generation_v3 Plan `7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5`，四组独立有序，不zip。所有3074 Gallery/6156 Query来源经CPU核验，6116编辑/40 NOOP/13537字符替换保持。本次仅固定官方前10 Gallery及前10正例覆盖Query进入模型，未按效果选样。

## CPU验证

- 125项必要CPU回归通过，46.20秒；另1项旧IRRA绝对路径来源测试在原工作树通过，0.92秒。
- 独立抽取官方RaSa itm_eval、在合成多正例Score上对照共享R@1/5/10及mAP exact，1项通过，0.95秒。合计127项通过。
- CPU测试覆盖原生融合独立表达式、维度/归一化/NaN拒绝、FP32 resize、IRRA空间拒绝、结构化Storage clone、来源缺项/篡改/环境拒绝、命名空间恢复、冻结生产者入口、四模式与Score块恢复、未知模型权限和攻击侧隔离。
- 一次真实CPU加载检查严格恢复849项，eval/requires_grad=False、CUDA未初始化，无神经前向。其后GPU Attempt只加载一次。
- 真实CPU预检88.818724秒通过；未加载模型。最终GPU入口再次以最终SHA配置完成CPU预检后才初始化CUDA。

早期CPU检查暴露并修复了旧Transformers文档装饰器、Tokenizer初始化顺序及跨工作树Plan fingerprint问题。采用原生算法不变的兼容边界及冻结攻击/物化包生产路径；最初CPU日志保留，未创建GPU Attempt。最初测试的Fixture目录/PID命名空间问题修正；原IRRA来源测试回原树执行，不修改历史配置。没有GPU失败或重试。

## 真实GPU结果

标志 `RASA_MATERIALIZED_EVALUATION_PASS smoke`，退出码0，唯一attempt-000001。

| Group | R@1 | R@5 | R@10 | mAP | mINP | DR_mAP(%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 |
| Image-only | 100.0 | 100.0 | 100.0 | 95.88888549804688 | 91.0 | 4.111114501953125 |
| Text-only | 100.0 | 100.0 | 100.0 | 99.16666412353516 | 97.5 | 0.8333358764648438 |
| Joint | 100.0 | 100.0 | 100.0 | 91.44444274902344 | 84.33333587646484 | 8.555557250976562 |

Joint−Image-only mAP=-4.4444427490234375个百分点。小样本指标仅验证真实链路，不作为正式迁移效果或成功率。

RaSa GPU加载尝试/成功1/1；实际Neural Hook图像样本20、文本样本20、融合pair400，与Adapter记录一致；原生Score调用4次，新提交Score块4、缓存命中0。G/H/IRRA加载/前向及Backward/Optimizer/JVP全部0。10条原文/攻击文本的BERT Token ID均发生变化，完整ID保存execution报告；目标Tokenizer没有修改历史H动作。

四类原生表示通过CPU FP32形状、有限性与global L2检查；输入FP32 [0,1]经原Store预算验证。839个参数Tensor和6个buffers运行前后摘要完全一致，eval=True、requires_grad=False、parameter_grads_none=True。
- 参数SHA：371998d47a18ffa8a0a2540d1ec12e35a469eec1c20d41cde1fc8410ae95f493
- buffers SHA：ce5f3986cd19bee31cacddc61b11f353c608f37e423185c3ed6ff2145befc59a

结果发布后独立CPU只读load_result读回四组10×10 Score及summary成功，没有调用模型或重算排名。无pending，未启动Full。

## 资源与限制

运行前GPU空闲3865MiB，Ollama PID2436556占用20178MiB；未干预外部进程。峰值allocated1992355840字节=1900.05859375MiB，reserved2183135232字节=2082MiB；进程内报告GPU剩余1320MiB，退出后恢复3865MiB、仅Ollama存活。Peak RSS4368650240字节，约4.069GiB。

execution段28.28327091410756秒；完整进程121.41659281402826秒，包含前置CPU来源检查。没有OOM、NaN或异常；保留原依赖FutureWarning/弃用提示，未隐藏错误。

Full保持原Top-128；Smoke因Gallery只有10张明确使用10候选。融合microbatch1控制显存，同一数学ITM表达式，不提供原128-pair batch在所有输入上的bitwise等价承诺。正式Full约315万融合pair，显著重于IRRA；磁盘建议20GiB、RAM约30GiB余量。实际服务器当时可用RAM115057205248字节、磁盘约6.48TB，inode约2.4亿。

RaSa官方原始PIL输入为384×384，现有物化样本已经固定384×128。Adapter使用FP32 Tensor BICUBIC/antialias到384×384并官方归一化，保留原像素而不量化为PIL；不宣称与从原始文件直接PIL Resize逐值一致，也不要求本次Clean与论文mAP一致。四组使用完全相同输入约束，正式报告必须注明。细节与完整命令见RASA_MATERIALIZED_EVALUATION.md。

## 证据及历史保护

输出：`/home/lzf/ldx/outputs/tbps-framework-rasa-materialized-evaluation/smoke/`
- completed/summary.json SHA：036ca6df1ffc793dfb39e36cc7b38294b12655f0f23975dc005822829c04aa61
- completed/metrics.csv SHA：6fd7e418cdda6e4b8a61690b5f43e5c4c1c03513a8a71f5743d841f7ac3bc6f4
- execution-000001/report.json SHA：8a9b7bcc0b73017f3cc85a8809b530c4a21e55ca071ab0242291f39d7c989995
- 完整日志/home/lzf/ldx/tmp/rasa_materialized_smoke_gpu.log SHA：75a4f976bbbb19b22ad78fc604f0c1e3583a2b0586c27ad19c5c96bd6c069277
- CPU日志/home/lzf/ldx/tmp/rasa_adapter_cpu_tests_final.log；预检日志/rasa_materialized_preflight_fixed.log位于同一tmp目录。

原生成/审计26文件SHA/mtime、61555物化文件成员/size/mtime保持。原IRRA Full summary、CSV、execution SHA与之前只读审计记录一致；冻结Stage4D1、原IRRA评估工作树和官方RaSa源码树clean。不重复重跑7991历史保护或旧Full排名。

本轮不提交权重、BERT文件、外部官方源码、缓存、结果或日志；只提交新接线、必要通用兼容、配置/tests/文档。RaSa Full仍PENDING，由用户手动启动；不自动重试、后台监控、恢复或进入APTM。
