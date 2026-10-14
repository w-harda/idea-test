# 第一版 G：官方 Test 共享对抗图库评测准备

2026-10-14：用户完成10 epochs训练；本轮仅准备正式评测，94项CPU tests与固定两图/四Query GPU smoke通过。未运行完整Test，未重新训练，无commit/push。

## 最终权重与入口

最终checkpoint为epoch10、scope=training，SHA：
`3c7d34f40f0303e9e0b1670f951f4dd449facc6804921d8bccd1b6430710c0a5`。

训练完成记录、sidecar、不可变版本、训练cache manifest、配置和权重有限值核对通过。原独立generate.py在held-out validation条件上通过CPU调用，恰好一次G forward，victim未初始化，无score/JVP/优化。

源码仅服务器：
`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/`

新增：
- g_test_core.py：官方协议、最终权重与缓存审计、原PID指标、condition/归档校验。
- g_test_evaluate.py：preflight → prepare → generate → evaluate，默认只preflight。
- g_test_generate.py：独立Test条件一次G前向，不加载victim。
- g_test_tests.py：30项新增CPU tests。

没有修改已训练G结构、参数、损失、原generate.py、训练入口、AttributeInstance/S/reader/basis/JVP或正式源码。另设Test入口是因为原generate.py明确绑定训练cache；没有放宽该入口的来源检查或伪造Test为validation。

## Test条件为何可构造

训练时S参考图库为34054张clean train；其中不含Test图，无法给Test图使用该图库中的paired positive index。

沿用用户此前已完成的官方Test协议：3074张clean Test图作为评分参考，每图固定annotation caption_index=0，并使用对应原图的准确index作为paired positive。冻结paired soft-rank、tau=.02、S_raw、gamma3/beta1/Natural K、reader/basis/JVP不变。

本轮读取并完整核验此前full Test的3074条score文件、selection manifest、feature cache和源码/协议SHA，复用其精确S_raw/Top-K/w/e；不重评分，不复用train/validation条件，不自行加入Test图到train图库。

93项共同来源SHA一致，包括IRRA、reader、basis。3074个image/首caption/PID/paired_index与原始annotation一致。共有14868个scored实例、323 unresolved、321 unsupported、4 representation/source unavailable，沿原策略排除，不设计fallback。

K0共14图，遵循已训练G的global-only行为：c_attr=0仍生成扰动；不沿用旧attribute-only TTA的clean passthrough。K分布0/1/2/3/4/5/6/7为14/57/963/1336/647/49/6/2。

prepare仅通过原TextTransport JVP与aggregate_condition构造新的Test条件缓存。保留独立JVP finite-difference/zero gate、metadata/source绑定、S归一权重；JVP属于离线导数计算。正式生成阶段不计算JVP、S或image梯度。

该协议仍为known-caption、offline surrogate-gradient-conditioned G，不宣称raw image+caption端到端零梯度条件准备，也不自动等同AP的image-only威胁模型。

## 共享图库与指标

每图用首caption对应条件，一次G forward生成同一张fp32对抗图，并真实编码IRRA feature。全部6156 Query使用同一套3074张G-Adv图库。所有同PID图为正例，无camera过滤。

复用旧official Test的exact SHA-pinned clean image/query feature cache，以及src/attributes/tbps_retrieval_metrics.py和pid_metrics.drop_rate。正式执行重新汇总clean，并逐项核对旧summary的mAP/R@1/R@5/R@10/mINP，误差超过1e-8百分点写clean_mismatch.json并停止，不静默接受差异。

期望旧Clean为：mAP66.0974694979%、R@1 73.1481481481%、mINP50.2776649092%。本轮未重新计算完整Clean或G-Adv Test指标。

当前仅IRRA单victim，aAP=mAP；mDR=(clean_mAP-adv_mAP)/clean_mAP*100。输出不称跨模型结果。

## Tests与smoke

原64+新增30，共94项CPU tests通过。覆盖Test PID/source/caption/score缓存与Top-K绑定、K0、同PID多正例AP/INP、图库顺序、clean mismatch拒绝、full授权门禁、共享图库及单forward/无backward/权重保持。

固定annotation顺序中前两个不同PID：
- cuhk:37132:0，official index0，K3；
- cuhk:37135:0，official index3，K4。

两图/四Query smoke完整通过：
- new score backward=0；7个属性JVP、14次离线autograd.grad；
- zero exact；JVP FD最大相对误差0.00109285（0.109285%）；
- 每图恰好一次G forward，生成backward/JVP/optimization均0；
- 最大L∞=.03137257695198059，超数学8/255仅2.79324e-8 fp32舍入；
- fresh clean image feature与原缓存max abs差0；
- G/IRRA/所有来源SHA保持；
- 独立Test生成CLI在CPU成功，victim_initialized=false；
- 对已完成smoke禁用CUDA/backward/integration后重入成功，不重新准备条件或生成。

两图clean/adv AP均100%，仅验证链路，不能评判G攻击性能。GPU smoke进程已退出，无完整Test输出。

## 用户手动命令（未执行）

在服务器终端执行：

```bash
tmux new-session -d -s tbps-image-g-test-v1 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python -B -u /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/g_test_evaluate.py --stage all --scope full --allow-full >> /home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/outputs/test_full.log 2>&1'
```

依次准备新的Test条件 → 一次前向生成3074图 → CPU汇总完整检索指标。可以将--stage all分别改为prepare/generate/evaluate；显式--allow-full且必须通过同源码/权重的smoke gate。已完成逐图结果可核验后复用。先查看同名tmux是否仍运行，避免并发启动。

日志：
`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/outputs/test_full.log`

结果：
`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_20261014/outputs/test_full/`

产物：protocol.json、conditions/manifest.json与records/tensors、prepare_completed.json、gallery/G_ADV/*.pt、generated/*.json、gallery_index.json、generate_completed.json、summary.json、metrics.csv、per_query_metrics.jsonl、completed.json。

完成标志：G_TEST_COMPLETE；completed.json中method_audit_passed=true、formal_test_evaluation=true、gallery_count3074、query_count6156、generation backward0。

主要耗时为离线JVP与IO，不是TTA多轮优化；完整耗时尚未实测。fp32图像/扰动归档加条件metadata需预留约6GB以上空间。

证据：outputs/test_readiness/{checkpoint_check.log,cpu_tests.log,test_source_audit.log,standalone_validation.log,standalone_test.log,completed_resume.log,final_preflight.log,completed.json}；outputs/test_smoke/与outputs/test_smoke.log。
