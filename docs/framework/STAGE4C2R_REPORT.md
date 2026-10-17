# Stage4C-2R：IRRA启动顺序CPU修复与旧产物只读审计

## 结论

启动顺序已在独立工作树修复，**554 CPU tests通过**，真实CPU来源/产物准备通过；本轮没有GPU、真实模型反序列化/前向或新的正式Attempt。

旧两图G10和两条H7文本可严格只读引用，原Plan ID和整体FAIL保持。**当前不具备申请GPU的资源依据**：现有IRRA完整Checkpoint eager加载的保守估算约2.3548GiB，超过固定2GiB；这是风险预检，不是实测OOM或严格RSS下界。Stage2.5没有CPU RSS记录，不能证明预算适用。特征及四组2×2 Score等价继续PENDING。

基准fcf097d3ab31785381947289a680e5f5ccbf9d42；分支codex/tbps-framework-stage4c2r；工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4c2r-20261017。

## 1. 具体修改及真实控制流程

原失败树smoke.py:160在child中调用torch.cuda.reset_peak_memory_stats，原stage_irra在:90才读取历史特征。原树和回执保持原样。

新树最小diff：

- child仍检查GPU资源；只有G阶段保留原位置的峰值统计初始化。IRRA阶段不提前初始化CUDA。
- stage_irra先调用新irra_startup.prepare_inputs，完成历史六个紧凑批次读取、两张物化RGB读取、两条原生H动作校验、原图/内容/文件SHA及source.verify、plan核验。
- 整体CPU读取受原feature_only_cpu白名单保护；内层原cpu_only及native_action_validator未删除、未放宽。
- 退出后显式比较Torch读取/神经构造/前向/CUDA/导数函数、SourceFileLoader.get_code、sys.path/meta_path、pyc标志均已恢复，且legacy_imports锁释放；CUDA仍未初始化。
- 之后才允许gate.initialize_cuda→reset_peak_memory_stats→IRRAAdapter.from_source。状态事件保留于阶段报告。
- G/H阶段函数AST与基准完全一致；模型、Tokenizer、权重、输入、预算、配置和1e-6阈值不变。

## 2. CPU测试证据

新增17项测试，最终518框架+36历史基础=**554项全部通过**，最终带时序日志复核37.90秒（此前回归36.87秒亦通过）。

启动时序测试执行实际child→stage_irra→prepare_inputs；真实CPU Fixture写入/读取Tensor、验证文件SHA、动作记录和source，使用原feature_only_cpu/cpu_only。只将设备、IRRA Backend及正式来源接口替换为CPU Mock，没有将整个准备函数替换成两个Mock步骤：

`cpu_read_started → cpu_sources_verified → cpu_isolation_exited → cuda_initialization_call → irra_model_load_entry`

- 正确顺序进入Mock IRRA加载及8个图文样本编码，完成四组2×2合成Score路径；没有真实CUDA或神经模型。
- CPU期间检查CUDA未初始化；隔离未退出即申请初始化被拒绝。
- 来源SHA、Unicode动作或特征NaN异常都在CUDA及模型加载前拒绝。
- 不恢复Torch函数或旧模块锁未释放时不允许进入CUDA。
- CPU门禁仍拒绝模型反序列化，不通过删除门禁使测试通过。
- 旧G/H插件与物化测试全部回归。
- 原Plan只读引用、SHA损坏、缺证明、pending、整体FAIL不得提升、Storage紧凑clone和Checkpoint词法扫描均覆盖。

真实小样本CPU准备也执行了同一prepare_inputs：四family仅取固定两行，输出均[2,512] FP32有限；CPU上下文显式恢复后停止，**没有调用gate.initialize_cuda**。

G/H固定75项runtime来源、33个实际旧执行文件及5个官方IRRA执行文件闭包通过。Torch模型load/forward/CUDA在严格CPU预检期间禁止。

## 3. 旧G/H产物严格复用

只读来源：

`/home/lzf/ldx/outputs/tbps-framework-stage4c2/materialized_smoke_v1/`

原stage-g、stage-text PASS，stage-irra及result FAIL；stage报告与整体报告逐项一致。两图像素/有效delta与raw delta实际文件读取、内容SHA、预算和原生成Claim/证明均有效；两条UTF-8原/Adv内容和完整Unicode动作经原生校验器核验。

- 两张L∞均0.0313725471496582≤8/255；真实FP32 RGB，effective_delta=Adv−Original，内容SHA与原报告一致。
- cuhk:37132:0保留原2编辑：position3 a U+0061→а U+0430，position44 o U+006F→о U+043E。
- cuhk:37147:1保留NOOP/K0及no_legal_token_changing_edit，原文/Adv UTF-8 SHA相同。
- G/H冻结Checkpoint、训练协议、Selection、原runtime/公共producer来源均验证。没有加载G/H。
- 原Attempt全部**70个文件SHA/mtime exact**；原result Manifest c1922bdf32935623dc9e85b1aeffe8a817cef2ab42b3ba767f62eda90e45155b不变。

新RetainedInputs使用明确的FrozenPlanReference分型，重算原protocol的canonical body digest并核对原producer路径/SHA与冻结来源，保留原单位FileRef、原Plan、原生成证明。复用原MaterializedStore.load/image_checks/Unicode核验规则；没有publish/generate接口，没有复制、改签或更改旧文件。

- 原Plan ID：7c0c00daf022bfcba94d6e3583c6de26c89b58c0e9c6696449cdbadb1ab7072d。
- 新树普通计划ID：e94013ed0986b3aacb780b611df247081af40e4c381cc7d4f7610dca033e6b78。
- 二者明确不同；旧单元不进入新Plan冒充新生成。
- 新引用schema=tbps-retained-smoke-input-reference-1，input_kind=readonly_materialized_smoke_reference，original_attempt_status=FAIL，republished=false，new_G/H_forwards=0。
- 复用范围仅这两张RGB与两条文本的小样本IRRA验证，不是3074/6156完整集合或新的攻击结果。

## 4. 固定2GiB资源预检

未反序列化IRRA，未加载Storage内容。通过ZIP目录与pickletools.genops词法扫描约191928 bytes描述；GLOBAL/REDUCE/BUILD只检查标记，不执行pickle，不创建Tensor或模型。

| 项目 | 原始bytes | MiB |
| --- | ---: | ---: |
| Checkpoint文件 | 1320992076 | 1259.796215057373 |
| 完整Checkpoint Storage，1101项 | 1320591330 | 1259.4140338897705 |
| model部分Storage，367项 | 440197110 | 419.80467796325684 |
| 其他Storage（含optimizer） | 880394220 | 839.6093559265137 |
| 已构造模型＋完整Checkpoint重叠估计 | 1760788440 | 1679.2187118530273 |
| 固定RSS预算剩余运行时空间 | 386695208 | 368.78128814697266 |
| 本轮CPU准备进程峰值RSS | 767696896 | 732.1328125 |
| 保守叠加估算 | 2528485336 | 2411.3515243530273 |

实际load_official_model先build模型，再torch.load整个包含optimizer的Checkpoint，load_state_dict后才del state；因此模型与完整payload有重叠。上表使用旧存储规模估算已构造模型占用，没有把Checkpoint文件长度直接当作模型大小。

保守估算2.3548354730010033GiB>2GiB。但CPU峰值包含可释放堆/分配器缓存，dtype转换与bootstrap瞬态也未知，所以**不是严格实测RSS或必然超限证明**。当前不能证明剩余368.78MiB足以容纳Torch/规则/元数据开销，应暂停GPU申请，不用一次GPU尝试“试试看”。

Stage2.5已有PASS：12.00397365167737秒，GPU allocated930.5205078125MiB/reserved1060MiB；report与gpu_smoke.log没有CPU RSS统计，不可从显存推导CPU峰值。本轮仅只读既有记录。

资源状态：**GPU_APPLICATION_BLOCKED_RESOURCE_BUDGET_UNPROVEN**。本轮未修改IRRA Adapter加载器，未提高2GiB或300秒。建议后续单独批准研究保持数值/Checkpoint不变的低内存读取（例如评估mmap是否避免读入无用Adam Storage）或明确修订资源预算；任何实现和GPU验证都需新授权及来源兼容审查。

## 5. 后续IRRA-only最小方案（仅设计，未创建Attempt）

资源问题解决并源码审查后，另行批准一次新的IRRA-only入口：

1. 新独立输出，如tbps-framework-stage4c2-irra-only/verification_v1；不恢复、覆盖或提升旧materialized_smoke_v1。
2. 严格CPU导入闭包＋RetainedInputs只读核验原FAIL中的G/H单元、原Plan及文件SHA；固定Gallery index0/15、Query index0/31。
3. 仅发布新的引用清单，引用原单位FileRef/生成证明，显式区分旧生成计数G2/H0和本次新生成G0/H0；不发布新MaterializedUnit、不重签旧Plan。
4. 直接使用本轮prepare_inputs的只读store_factory/selected_features路径，待所有CPU上下文退出后才初始化CUDA。
5. IRRA只恢复1次，图像4、文本4样本前向；G/H恢复/前向0，Backward/Optimizer/JVP0；FP32/math/no_tf32、原1e-6阈值。
6. 用真实保存RGB/Unicode编码，与独立加载的历史对应行比较特征、四组2×2原生Score；不排名、不Full。
7. 预算重新审查前保持2GiB/300秒，现阶段入口应拒绝正式执行；不得根据结果放宽阈值。
8. 独立新Claim/结果回执证明仅IRRA验证状态，旧Attempt永远保留FAIL。

**本轮没有实现可启动新Attempt的IRRA-only GPU CLI。** 当前旧Smoke入口仍固定已用输出并拒绝重试；不能自行通过改路径执行。下一步首先是资源/加载方案CPU-only审查。

## 6. 来源证据与历史保护

- CPU审计：/home/lzf/ldx/outputs/tbps-framework-stage4c2r/cpu_audit_v1/result/。
- report SHA7b2b6cdf71fed7b203e896867b15f3c87462112ab9bb98ea3e9ad4e524ba8a33。
- Manifest SHAdf0a94bea0bb8b454ba44d9a1e4b02ead859137327a0a6cce5e73250b9f78bf5。
- 历史保护：protection_v1/result/，Manifest SHAda6626c11b36a4b53d929d7ac5b94abb912f2c60e536e5818c9f3da252ca4423。
- CPU日志：/home/lzf/ldx/tmp/stage4c2r_cpu_audit_v1.log、stage4c2r_all_cpu_tests.log、stage4c2r_final_cpu_tests.log（含真实控制流程Mock事件顺序）。
- 最终CPU验收：同命名空间cpu_verified_v1/result/，Manifest SHA8fbf97ccfe373f80f64c1f5c3bcc9be7ad5ee6744c1857def242ac60eba06e70，绑定修复源码、测试、CPU日志与以上证据。

7991项历史保护文件及3472个Joint缓存SHA/mtime保持；Stage4C-2、C1、B2、Stage3B FAIL、Stage3B2 PASS、原发布树HEAD/clean均保持。没有覆盖旧Stage4C2报告或任何失败/完成回执。

## 7. 文件与CPU命令

修改src/tbps/materialization/smoke.py；新增irra_startup.py、retained_inputs.py、checkpoint_layout.py、audit_retained.py及两个CPU测试文件。同步更新PROJECT_STATE/ARCHITECTURE/EXPERIMENT_REGISTRY/ADR-0007，未修改原Stage4C2配置。

CPU测试：

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-framework-stage4c2r-20261017
CUDA_VISIBLE_DEVICES="" TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m pytest -q -p no:cacheprovider tests/framework tests/test_dataset_adapter.py tests/test_dynamic_topk.py tests/test_generator_framework.py tests/test_joint_baseline.py
```

CPU来源/旧产物审计入口（已完成cpu_audit_v1；如用户复核须使用独立未使用的输出，不能覆盖）：

```bash
CUDA_VISIBLE_DEVICES="" TMPDIR=/home/lzf/ldx/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.materialization.audit_retained --config configs/framework/stage4c2_smoke.server.json --output /home/lzf/ldx/outputs/tbps-framework-stage4c2r/cpu_audit_user_review_v1
```

成功标志STAGE4C2R_CPU_AUDIT_PASS；资源输出同时明确GPU_APPLICATION_BLOCKED_RESOURCE_BUDGET_UNPROVEN。CPU通过不能被当作GPU执行授权。特征及四组Score等价保持PENDING；提交推送后停止等待审查。
