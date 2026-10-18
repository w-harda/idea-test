# AP-Attack B3三模型对比接线

本阶段只使用B3。冻结CUHK-PEDES/IRRA ReID+Semantic Generator，不重新训练，不运行B1/B2，不使用目标RaSa/APTM反馈调整攻击。正式Full尚未执行，等待用户确认。

## 来源与复用

- B3权重：/home/lzf/ldx/outputs/AP-Attack/cuhk_irra_semantic_10_10/G_CUHK_IRRA_AP.pth.tar。
- 实际SHA-256：c1c637e2e5346a3011f23006bb77bad60e4bd4bc28eb746d5f6bfcd881ba4ebb。
- config、Checkpoint sidecar及progress一致：60 Epoch、seed1234、Adam2e-4、ReID/Semantic权重10/10、固定最后Epoch、不使用caption/IRRA文本。
- 34054张CUHK train/11003 PID；语义骨干来源DukeMTMC-reID。此训练规模及额外语义来源与G-v2的2048图训练不同，不能声称训练条件完全匹配。
- 历史完整像素缓存：/home/lzf/ldx/outputs/idea-TBPS-test1/baseline-ap-cuhk-irra-surrogate/adversarial-gallery/。3074图、97个批次、FP32真实RGB，逐批SHA及完整路径顺序已核验。对应冻结B3权重、原图SHA和“无victim/query信息”来源声明。
- 旧源模块src/attributes/ap_gallery_baseline.py的SHA为88041946242332523f337bb03f353342dd649529c7c455cb79c4e9d654e1adca，原生AP GD/spectral/gumbel来源SHA按旧Gallery协议检查。只调用native_pixels/transport_delta，**G权重反序列化和G前向均0**。
- 历史生成采用官方CUDA AMP、batch32，SpectralNorm内部u/v更新被原缓存记录。通过复用原像素保留该真实生成序列，不擅自重设状态。
- 训练配置记载PIL bicubic 256×128；历史**评测生成缓存**及其读取器明确为PIL bilinear 256×128。本轮复用评测缓存，保留这一训练/评测差别，不更改历史输入。
- 新CPU物化沿用旧transport_delta：bilinear(adv256-clean256, align_corners=False)叠加generation_v3同一clean384×128；保存原RGB/Adv/effective_delta并做严格8/255及内容SHA检查，不JPEG重编码、不引入新的预算裁剪算法。

generation_v3原Plan及G-v2像素Store保持不变。旧MaterializationPlan/MaterializedStore只接收固定gv2/hv1组件；因此新增B3Pixels薄桥复用ArtifactStore原子发布和MaterializedStore.image_checks，不将B3伪装成gv2，不放宽旧Plan。

## 评测契约

固定官方3074 Gallery/6156 Query，多Caption、全部同PID正例、无camera过滤。独立Gallery/Query顺序，文本始终Clean。

- IRRA：同训练来源模型；官方Byte-BPE/77 tokens，384×128/CLIP归一化、L2特征点积。
- RaSa：跨模型迁移；原生384×384、BERT50、Top128候选的ITM正类logit及-100非候选值。
- APTM：跨模型迁移；原生384×128、BERT56/Swin局部表示；ITM与global矩阵在**完整所绑定Query/Gallery集合**中各自min-max，最终ITM+.002×global。绝不逐Query块独立归一化。
- FP32/math/no_tf32、图像及融合microbatch1，Query chunk128；三个模型的Checkpoint、预处理、Tokenizer与ModelSpace必须与已完成原Full完全一致。
- Full直接引用已独立审计的同模型Clean与G-v2 Image-only指标及原Clean Query特征；新编码仅B3 Gallery，文本神经前向0。引用完整有序Manifest、原图与攻击内容SHA、原协议/完成/执行证据，禁止跨ModelSpace引用。
- Smoke从Full特征中先选固定前10张Gallery及其合法前10条Query，重新计算Clean/B3/G-v2三个10×10 Score；只新编码B3的10张图，Clean/G-v2图像与Clean文本特征复用，**不声称三组均重新编码**。RaSa/APTM Smoke Top-K为10；小指标只检查接线。
- 复用EvaluationEngine的原生评分和Ranking，不写新指标算法。对外只发布R@1/R@5/R@10/mAP，百分数、两位小数；另存这四项原始精度CSV/JSON。顺序Clean → AP-Attack B3 → G-v2 Image-only，不发布额外指标或下降率。

旧B3曾有IRRA Full，其mAP40.4273941577433只作为历史证据，不填入本轮尚未执行的Full。旧流程的Clean mAP66.09696256982703与统一物化Full66.09744262695312存在小差异，旧批量/混合精度及指标实现不同；本轮用当前固定FP32原生协议重新评估B3，不混用旧指标。新Full的Clean直接引用本模型原已审计结果，因此不存在本轮重新编码Clean造成的数值变化。

## 模块职责

- src/tbps/baselines/ap_b3.py：B3历史来源/97批SHA/单行像素读取，复用原扰动传递与严格RGB契约，独立紧凑不可变像素Store。
- src/tbps/baselines/comparison.py：正式Full只读表示桥、三模型原生加载/评分、四列结果及CPU审计。
- src/tbps/baselines/launch.py：在CPU启动快照前完成torchvision首次导入；不绕过CPUStartupGate。
- src/tbps/b3_eval.py：固定旧攻击/物化生产命名空间。
- src/tbps/baselines/run_b3_full.sh：用户显式确认后的顺序Full脚本，flock防重入、无自动重试。
- configs/framework/baseline_b3_ready.server.json：**当前唯一执行配置**，75个明确源码SHA及全部历史来源。
- configs/framework/baseline_b3.server.json：首轮CPU门禁失败的原配置快照，保留来源证据，**不要用它启动实验**。

## 启动顺序修复与CPU验证

第一次新入口在CPU门禁前失败，victim加载/CUDA初始化/真实前向均0，失败证据保留于/home/lzf/ldx/outputs/tbps-framework-b3-comparison/smoke/irra/execution/report.json。旧源模块首次导入torchvision会向sys.path增加setuptools/_vendor及临时目录，原CPUStartupGate正确拒绝。新launch在建快照前完成该合法CPU依赖导入，不修改旧门禁、原攻击代码或失败回执；后续使用独立ready输出。

tests/framework/test_b3_startup.py在独立CPU进程中执行真实旧模块导入和Resize/ToTensor，核验CPU读取→隔离退出→Mock CUDA→Mock模型入口，真实CUDA/模型均0。新增32项测试通过；修复后相关Adapter/Engine/缓存回归132项通过、6项因指定筛选未执行。首次广泛框架回归798通过、4失败；四项均是旧配置绝对工作树路径门禁，在未改的基准7bff515工作树中同样4失败，不能计为本轮新增回归，也未为让其通过改冻结配置。日志在/home/lzf/ldx/tmp/b3_cpu_tests.log、b3_base_regression.log、b3_final_cpu_tests.log。

## 可复制的服务器命令

以下命令使用当前实际新工作树，入口默认CPU。Full脚本尚未启动。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018
export PYTHONPATH=/home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018/src
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=/home/lzf/ldx/tmp
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# CPU Full来源预检：不编码、不加载模型、不排名。
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.launch --config /home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018/configs/framework/baseline_b3_ready.server.json --action preflight --scope full

# 本轮Smoke被Ollama高利用率拦截，尚无真实指标；资源空闲后可手动逐模型执行。
# 成功后不能重复运行，失败现场不得删除或自动重试。
for model in irra rasa aptm; do
  /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.launch --config /home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018/configs/framework/baseline_b3_ready.server.json --action run --scope smoke --model "$model" --allow-smoke || break
  CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.launch --config /home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018/configs/framework/baseline_b3_ready.server.json --action audit --scope smoke --model "$model" || break
done

# 三模型Smoke及CPU审计成功且用户确认后，一键后台执行CPU像素传递 → IRRA → RaSa → APTM → 各自CPU审计。
nohup bash /home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018/src/tbps/baselines/run_b3_full.sh --allow-full > /home/lzf/ldx/tmp/b3_three_model_full_background.log 2>&1 < /dev/null & echo $!

# 查看监督器及单模型进度。
tail -n 50 /home/lzf/ldx/tmp/b3_three_model_full_background.log
tail -n 30 /home/lzf/ldx/outputs/tbps-framework-b3-comparison-ready/logs/rasa_full.log

# 独立CPU只读审计，Full成功后可逐模型执行；不重新排名。
for model in irra rasa aptm; do
  CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.baselines.launch --config /home/lzf/ldx/projects/idea-test-tbps-baseline-b3-20261018/configs/framework/baseline_b3_ready.server.json --scope full --action audit --model "$model"
done
```

Full脚本已含完整具体CLI参数与CPU物化授权参数，缺少任一成功Smoke或其CPU审计时拒绝进入Full。不带--allow-full立即退出；若Full目录已经存在或持有flock，拒绝覆盖和重复启动。进程异常/OOM/未提交批次保留，不自动重试；--resume不能用来重跑失败Attempt，恢复前需人工审查。停止使用kill -TERM对应本轮监督器/子Python PID，不操作Ollama或其他任务；有子进程时仅终止监督器不能保证子Python停止。用ps和nvidia-smi确认本任务计算进程退出。

输出：/home/lzf/ldx/outputs/tbps-framework-b3-comparison-ready/{smoke,full}/{irra,rasa,aptm}/completed/metrics_2dp.csv及metrics.csv、summary.json；pixels目录保存目标无关RGB。运行报告execution/report.json含计数、冻结摘要、耗时和显存。成功标志为B3_CPU_PREFLIGHT_PASS、B3_PIXELS_PASS、B3_EVALUATION_PASS、B3_CPU_AUDIT_PASS；后台最终SUCCESS | B3_THREE_MODEL_FULL_AND_CPU_AUDIT。

## 资源估算与边界

本轮Full预检仅CPU，正式3074图CPU像素传递/Full GPU评分均未启动。Full新增FP32三Tensor约5.066GiB，RaSa局部表示约5.1GiB，APTM约0.6GiB，IRRA很小，另有Score与来源元数据；建议至少20GiB可用磁盘和32GiB可用RAM。当前文件系统可用约5.87TiB、inode约2.4亿，系统可用RAM约105GiB。

原四组Full执行耗时IRRA219.26s、RaSa9495.56s、APTM8560.56s；本轮仅新增B3图像编码及一组评分，两个融合模型各788032个Top128融合对。估算两者各约40–70分钟，加CPU来源/像素物化和IRRA，整体约2–3小时，属于粗估而非本轮Full实测。真实Smoke资源见阶段结果，不保证Ollama负载变化时性能/显存不变。入口记录共享GPU状态，空闲显存不足3072MiB或利用率超过10%时停止，不干预其他进程，不自动等待/重试；实际OOM原样报告。

本阶段不进行七种新Baseline训练/攻击。其官方来源、权限与适配优先级见[可行性报告](BASELINE_FEASIBILITY.md)。

## 本轮实际执行状态

- B3_CPU_PREFLIGHT_PASS full，exit0；3074/6156、97历史批SHA及三受害者Full协议核对通过，CUDA为false。
- 10张历史B3图完成CPU像素传递、原子保存及读回；最大L∞见阶段报告。
- 真实GPU Smoke **PENDING**：修复启动顺序后，GPU资源检查时Ollama PID726931占用20178MiB，空闲3865MiB，利用率22%，后续一次只读状态为92%。在资源门禁处停止，ready/smoke/irra/attempt尚不存在；RaSa/APTM也未启动。
- G/H及三个受害者真实模型加载/前向均0；无Smoke指标可报告，不以旧Full或旧B3指标替代。
- 无Full像素传递、Full模型评估、后台任务或GPU轮询部署。本轮停止时保留初次启动门禁FAIL、资源拒绝日志和已发布10图CPU产物。
