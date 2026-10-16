# H-v1 Phase 5：官方 CUHK-PEDES / IRRA Joint Full Test

本目录新增独立评测系统。G-v2、H-v1网络/训练/奖励、Phase4 formal_training、运行中的Validation/Selection、历史Checkpoint与Full结果均不修改。Codex只开发和执行合成CPU测试，**没有运行真实模型前向、GPU Smoke、Test缓存生成、Full或真实Full CPU指标计算，也没有查看/监控当前Validation进度或假定最佳H**。

## 固定协议与Selection

- 官方Test：3074个唯一Gallery图像、6156条原始Query；**不按Query重复生成图库**。
- 四组：Clean（原图原文）、Image-only（G-v2图原文）、Text-only（原图H文）、Joint（G-v2图H文）。
- 全部Query保留，含STOP/NOOP、无属性、无合法替换、原文截断和H输入超长；不合格Query固定原文NOOP。原文截断沿用官方77-token编码；H不新增截断。
- H为当前Validation selection指定权重的贪心策略。H只拿原始Caption，由冻结Caption-only前端提取Canonical/type/span；不拿Gallery、PID、配对index、S_raw、JVP或Test反馈。PID/index只在身份校验与最终官方检索指标层使用。
- 使用既有官方IRRA rank、FP32单位化余弦、全部同PID正例、无camera过滤。输出五指标完整精度、各组DR_mAP及Joint-minus-Image-only百分点差，允许负收益，不据Test反馈重新选模型或调整方法。
- 固定G-v2 E10：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt，SHA f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
- 官方IRRA：/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth，SHA 9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。

正式执行时读取Phase4完成证据，校验全部不可变Checkpoint链/来源、2048/128隔离、每个训练Epoch独立Validation完成记录、selection内容摘要与文件名、最低Joint mAP/最早visit并列规则、选中H Checkpoint SHA及架构。**不提前硬编码H Epoch或SHA**。

resolve-selection仅在整个训练与所有Epoch Validation完成且存在一个可验证最终selection时输出路径。旧的未覆盖全部Epoch的中间selection不作为最终结果；找不到/歧义/来源不符则显式失败，不等待、不轮询、不自动选另一模型。--selection也可由用户明确传入原不可变selection文件。

## 已验证G-v2特征复用与边界

本版专门复用：
- 旧G-v2 Full相同官方布局与已完成summary/completed/clean_verified/Storage修复兼容记录。
- 原SHA锁定Clean/Text bulk参考文件，避免读取旧单行视图占用的约90GiB底层Storage文件。
- 旧G-v2 Full的3074条Epoch10图像特征，每批核对metadata、特征SHA、原summary绑定、计算回执、严格8/255预算和producer兼容。

**本阶段真实执行的新G forward=0、IRRA image forward=0**。Image-only与Joint共享同一套已验证G-v2对抗Gallery；Clean与Text-only共享Clean Gallery。历史E10 G及Adv image样本各3074，Clean bulk参考3074、原文text参考6156，和当前新前向数量分开记录。

只有H实际改变且通过合法Unicode/Tokenizer检查的文本才新调用冻结IRRA text encoder，最多6156个样本，按--text-batch分批。NOOP直接复制原文feature，逐值exact。H保持CPU eval/inference，不加载optimizer；IRRA仅按需GPU加载，eval/冻结，参数/buffers前后SHA一致；禁止图像前向。调用计数不使用保存历史Tensor的MagicMock。首批H前向前就固定与Phase4 Validation相同的FP32、TF32关闭、MHA/flash/memory-efficient fastpath关闭、math SDP开启；不等IRRA懒加载后才改变全局路径。Torch/Python/CPU线程数绑定运行协议。

在任何H/IRRA新前向前，重新计算并核对固定Clean/Image-only五项指标。历史mAP参考66.09744262695312 / 32.28228759765625，沿用G-v2原0.001百分点校验阈值，不匹配立即停止，不放宽。正式Text-only/Joint数值须用户运行后取得；CPU合成测试不代表攻击效果。

## 可复制的服务器命令（全部由用户执行）

以下在服务器Bash执行。当前正在运行的Phase4无需git pull、切换工作树或改动文件；本阶段仅新增phase5_joint。完成独立代码审查、Phase4 Validation与选优后，再手动执行下面命令。

### 1. 可选：仅源码CPU预检，Selection尚未生成也可执行

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m phase5_joint.evaluate --action preflight
~~~

成功：PHASE5_CPU_SOURCE_PREFLIGHT PASS; SELECTION_REQUIRED; GPU_BLOCKED。该入口不读取运行中的Selection，也不生成Test缓存。

### 2. 解析真实最终Selection并做Full CPU Preflight

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
SELECTION=$(TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m phase5_joint.evaluate --action resolve-selection) &&
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m phase5_joint.evaluate --action preflight --selection "$SELECTION" --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full
~~~

成功：JOINT_FULL_CPU_PREFLIGHT PASS 3074 6156，打印G/IRRA/H真实SHA。严格检查来源、全部身份/图像SHA、官方reader、旧Full完成证据与实际selection；不构造模型，不编码、不计算完整Test指标、不写输出。

Selection尚未完成时会失败退出，**由用户完成Phase4后再次运行**，不是后台等待。

### 3. Full正式评测（仅用户GPU执行）

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
SELECTION=$(TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m phase5_joint.evaluate --action resolve-selection) &&
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m phase5_joint.evaluate --action run --user-run-full --selection "$SELECTION" --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full
~~~

CLI没有Smoke、100或任意子集模式，3074/6156固定校验，不能传6156为Gallery数量。--user-run-full必须显式提供，默认绝不会启动GPU。

阶段标志：
- JOINT_FULL_CPU_PREFLIGHT PASS：全部前置检查通过。
- JOINT_METRIC_COMPLETE Clean / Image-only：初始基线校验；GPU尚未加载。
- JOINT_REFERENCE_COMPLETE：紧凑参考缓存已发布。
- JOINT_TEXT_COMPLETE n / 6156：H文本动作与相应新IRRA文本特征批次已提交。
- JOINT_METRIC_COMPLETE Clean / Image-only / Text-only / Joint：CPU正式指标计算；这时GPU利用率0%可以正常。
- JOINT_FULL_EVALUATION_COMPLETE：四组完整结果、CSV及关联执行SHA已发布。

需要文件日志时，用户可先创建outputs/phase5_joint/logs，set -o pipefail，然后在命令末尾加 | tee -a <新的日志文件路径>；不要覆盖旧日志。本程序前台运行，没有后台任务或自动监控。

### 4. 安全停止和恢复（仅用户）

Ctrl+C/SIGTERM在当前原子批次、指标组或哈希检查边界处理；完成当前批次后出现JOINT_FULL_SAFE_STOP。不在模型forward中强制抛异常；官方CPU排序一次操作结束前信号可能延迟。恢复直接重新执行第3节**相同Selection、batch、输出**命令，已验证的完成批次不再调用H/IRRA。

硬中断/OOM可能留下*.pending。默认保留现场并拒绝自动重复。检查failure/interruption证据，确认允许重新执行未提交的无状态文本/参考拷贝后，由用户显式执行：

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
SELECTION=$(TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m phase5_joint.evaluate --action resolve-selection) &&
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -u -m phase5_joint.evaluate --action run --user-run-full --recover-pending --selection "$SELECTION" --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full
~~~

仅处理同协议/同身份的未提交Phase5批次：整目录原子移入abandoned，保存文件SHA与恢复回执，原证据永久保留，然后重算未提交文本/参考拷贝；不覆盖任何已发布单元。此处允许重算是因为H/IRRA文本推理无状态、G/image新前向始终0。损坏的已提交批次、未知pending、Selection/源码/参数变化仍拒绝，不能通过此选项绕过SHA。

完成后再次执行第3节命令只读查SHA，不构造模型或重算完整指标。真正独立指标复核使用下一节CPU audit。

### 5. 完成后CPU只读审计

~~~
cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015
TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m phase5_joint.evaluate --action audit --text-batch 16 --output /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full
~~~

成功：JOINT_FULL_CPU_AUDIT PASS。audit自动读取已锁定selection路径重新核验，不重新选择H；核对模型/来源/所有单元/回执/attempt SHA、动作Span/Unicode/Tokenizer、参考逐值一致、原G-v2 E10特征逐值一致、CSV/summary；**在CPU用已缓存特征重算四组官方检索指标**。不做任何H、G或IRRA前向、不写文件。

CPU audit完整检索计算由用户执行，Codex本阶段未执行。审计命令不需要--user-run-full，因为没有GPU前向。若实际运行时改变text-batch，preflight/run/resume/audit必须使用同一冻结值。

## 输出与完整性

固定独立目录：
~~~
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase5_joint/full/
├── protocol.json             # 动态选中的H/selection、G/IRRA、身份、来源、官方协议
├── baseline_verified.json    # Clean/Image-only五项校验
├── image_reuse.json          # 原E10单元SHA/回执/预算及历史前向计数
├── features/
│   ├── clean_gallery/
│   ├── image_gallery/
│   ├── clean_text/
│   └── adv_text/              # 每批包括全部原Query身份、H动作、Tokenizer及编码计数
├── attempts/*/execution.json  # 或failure/interruption；有GPU峰值/实际耗时/冻结SHA
├── abandoned/                # 显式确认恢复留下的旧未提交证据
├── metrics.csv               # 四组五指标及DR_mAP，Python float完整精度
├── summary.json              # Joint与Image-only比较、K/属性/Unicode/NOOP统计
└── completed.json            # summary、CSV、基线、完整attempt文件SHA
~~~

DR_mAP=(mAP_Clean−mAP_group)/mAP_Clean×100%。Joint-minus-Image-only<0表示此次Joint mAP更低；不把负synergy或训练奖励当作正式成功标准，也不把这次Test用于选优。

每批features显式clone生成独立紧凑Storage，加载后也clone；feature/metadata/receipt同目录fsync后原子发布。全部Query动作可回放至原Caption，保留完整Canonical源Span、位置、旧/新字符、Unicode码位、字节偏移、选择IDs及Tokenizer变化。所有回执都有原子完成证据与SHA，异常不静默跳过。

summary中前向数量是**已提交有效特征**统计；execution/interruption/failure记录各尝试请求/实际返回的编码样本，硬失败尝试不能冒充有效提交。--recover-pending可能重新编码未提交文本，废弃尝试保留，因此累计硬件尝试次数可能高于最终有效编码数。正常恢复不重复已提交批次。

## 成本预估与限制

没有本阶段真实GPU测量。由于重用图库，新的G/IRRA image计算为0；IRRA text最多6156个，实际取决于H编辑/NOOP，H CPU最多6156次greedy。粗略建议预留10–30分钟（含CPU H/资格/源码/单元SHA及官方排名），并非吞吐承诺；GPU显存按batch16暂估1–3GiB，CPU RAM建议4–8GiB。

四套原始feature约36MiB，动作/元数据与来源证据另计，建议预留0.5–1GiB磁盘。历史约90GiB旧大Storage文件完全保留，但不逐条重读Clean/Text旧view；本系统不保存像素、不重新生成Gallery。

目前仅CPU合成测试通过，真实Selection解析、真实Full及其CPU审计必须等用户执行后确认。本系统不支持改变算法或根据Test结果重选Checkpoint。完成代码后停止，不监控用户实验。
