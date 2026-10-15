# G-v2 Phase 2.5：2176条完整条件缓存准备与审计

2026-10-15：原2048 train + 128内部validation完整缓存准备并通过独立审计。6385个属性JVP全部有效，其中复用原Phase 1的15个，本阶段新增6370个。没有重新评分、重新选择Top-K、构造/训练G或运行Gallery评测。停止等待独立审查，不进入Phase 2.6。

## 工作位置与交付

仓库：w-harda/idea-test；分支：codex/gv2-per-attribute-cache。
本阶段基准：5a9f19e30862da8c488ae2b81cc3f2de554dc2b1。
服务器工作树：/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014。

所有新增源码位于 prototypes/attribute_image_generator_v2_20261014/：

| 文件 | 用途 |
|---|---|
| phase25_prepare.py | 原prepare_full完整入口的观测外壳，记录资源/耗时/异常，禁止G/optimizer/Epoch训练 |
| phase25_audit.py | CPU preflight、独立逐样本来源/监督/统计审计 |
| test_phase25.py | 21项新增CPU审计与计数器回归测试 |
| README_PHASE25.md | 入口、结果与边界 |

本中文报告在Windows docs/stages/09_2_5_gv2_full_condition_cache_audit.md编写并同步到服务器同一提交。AGENTS.md与docs/PROJECT_STATUS.md仅更新Windows状态，不提交。缓存、输出、日志及失败现场全部留服务器，不进入Git。

7个已锁producer文件、训练协议、G/Loss/Retain/串行机制、G-v1和官方IRRA未修改。64份保护文件SHA保持，包含前阶段源码/文档及G-v1最终权重和Test结果。

## Preflight与实际入口

先核对HEAD/分支、干净工作树、Tailscale服务器和GPU占用，再运行原prepare_full默认CPU preflight及独立phase25_audit --preflight。2176条原样本、4450项上游SHA、raw annotation、源图SHA、caption/PID/row_id、原评分/选择/权重与内部角色均通过。CUDA未初始化。

Train包含1861 PID，validation包含121 PID，交集0；原5条独立Smoke缓存完整核验。开始时outputs/conditions_full不存在。

实际GPU准备始终调用原 prepare_full.main(["--allow-full-preparation"])，默认输出为：

    /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/conditions_full/

未使用small模式，未使用Phase 2.4补算的K1替代原Phase 1缓存。原5条Smoke的全部独立JVP、z_t、权重和v_local逐元素复用。

观测外壳只计数、记录资源和设置禁止训练的断言，不改变输入、图、数值、精度或JVP公式。原producer的逐条原子发布、SHA验证、已完成样本恢复和完成Manifest规则保持。

## 异常与恢复记录

首轮发生一次CUDA OOM，原因是本阶段新增观测外壳使用全程MagicMock(wraps=torch.autograd.grad)，保存了所有调用的张量参数，导致计算图累计留存。属于本轮监测代码问题，不是属性、数据或原producer检查失败。

首轮在发布79条后安全退出，GPU释放。失败日志、完整traceback、资源采样、invocation SHA及修复前wrapper源码快照保留在：

    outputs/phase25/preparation_attempts/40e5acf474544158abdcd0bedcc947ae/

只修改新增的phase25_prepare.py，将全程记录替换为CallCounter：保存函数和整数计数，不保存参数、返回值或图。CPU弱引用测试验证张量不被计数器保留；原autograd.functional.jvp值与2次内部grad调用保持。未修改producer或放宽FD/zero/聚合/身份/SHA门槛，未改变CUDA allocator或其他进程。

恢复前逐条核验79条缓存：全部通过、聚合重建误差0，无pending目录。快照记录158个tensor/metadata文件的SHA、inode和mtime。随后利用原入口恢复剩余2097条；最终这些158个文件全部保持，未覆盖或重算任何已完成样本。

| 运行 | 发布前→后 | 新增已保存JVP | 实际autograd.grad | 入口墙钟秒 | peak allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|
| 首轮，OOM退出 | 0→79 | 240 | 480 | 45.5748 | 3189.4434 / 3392 |
| 修复观测器后恢复 | 79→2176 | 6130 | 12260 | 406.1039 | 1225.1001 / 1312 |
| 累计 | 0→2176 | 6370 | 12740 | 451.6786 | 含失败峰值，分列见上 |

墙钟包含原prepare_full内部CPU来源复核、模型加载、JVP、写盘与完成检查；不包含wrapper前置CPU preflight、诊断和修复等待。不能将其称为纯GPU kernel耗时。两轮官方模型加载分别3.7763/4.0982秒。

首轮的240个已计算JVP全部落在已保存样本内；累计12740次grad正好等于2×6370，没有额外重算JVP。原缓存另复用15个JVP，总计6385个。

恢复运行无异常，官方IRRA终态参数/buffers SHA exact，参数冻结、无参数梯度。失败运行未完成终态模型SHA检查，其metrics的false表示未通过完成态分支，不表示已观察到模型变更；官方文件与权重的保护SHA始终保持。

资源每5秒采样。既有PID3507659占用20178 MiB始终保持；恢复进程显存稳定约1.4–1.8 GiB，没有停止或修改其他进程。准备GPU进程已经退出。

## 完成数量与K分布

2176个唯一row_id、2176张唯一图像，精确覆盖原Manifest，无遗漏、重复或多余发布目录。Train/validation图像、PID交集均0，与官方Test PID交集0。全部样本仍属于官方train，内部role分别train/validation。

上述隔离指生成器样本集合。原S_raw仍继承34054张clean train评分参考图库，其中包含内部validation图；保持既有协议，不称为gallery-disjoint评分设置。

| K | Train | Validation | 合计 |
|---:|---:|---:|---:|
| 0 | 5 | 1 | 6 |
| 1 | 34 | 6 | 40 |
| 2 | 639 | 36 | 675 |
| 3 | 842 | 53 | 895 |
| 4 | 463 | 32 | 495 |
| 5 | 61 | 0 | 61 |
| 6 | 3 | 0 | 3 |
| 7 | 1 | 0 | 1 |
| 总数 | 2048 | 128 | 2176 |
| 平均K | 2.939453125 | 2.8515625 | 2.9342830882 |
| JVP总数 | 6020 | 365 | 6385 |

这里只统计原冻结K，没有重新计算或设置cap。若后续启动现有调度，单个正式train Epoch应为6020个属性更新+2048个联合更新=8068次step；validation仍不得更新。本阶段未执行这些更新。

## 逐样本监督与SHA审计

独立审计在禁CUDA环境运行，逐条核对：

- gv2-training-cache-2 Manifest及原gv2-attribute-supervision-1张量协议、全部tensor/metadata SHA、源协议/指纹、7个producer与4450项上游SHA。
- 原raw annotation、完整唯一标识、caption_index、caption、PID、image_rel_path、图像SHA及内部role。
- selected_instance_ids、原降序S_raw、原float64权重逐元素exact；[K,512] JVP维度、检查记录ID和顺序一致。
- z_t与原完整文本监督逐元素exact，仅作Global Loss监督，不传入G。
- 所有v_local由各v_i和score_weights重建，最大绝对/L2重建误差均0。
- 与原G-v1 GPU聚合对照最大绝对差9.313225746154785e-10，沿既有atol=1e-9/rtol=1e-5检查通过，属于原CPU/GPU求和差异；未重新归一化或改权重。
- 6个K=0全部固定空ID/分数/权重、[0,512] JVP、零v_local、has_local=false。
- 无失败样本、无失败JVP、无零向量替代、无静默跳过，pending目录0。

最终Cache SHA：

| 文件 | SHA256 |
|---|---|
| manifest.json | 506caf30b8f82b8a29fc35e06df7bb04a5b2bcddc58a1f159b524bb107e7511e |
| protocol.json | 9ad4903a6de6dc221df1a15af48d9e63e6332d7fcd3a9a62ecd3f11a2ed91c9d |
| completed.json | 3cdcbc3e38509dd190b6e1507c878feb2a1aff3c2ea380de6beebd0121c8ee99 |
| summary.json | 20275832440500164708417e4a317dbc71596faffb5ffaacde9eaf1519f7f01a |

独立审计per_sample.jsonl SHA：7701583476f460aa2c5e08d8695b4e7ded6a085447ce68465f37f44a6a2452f2。

## JVP检查与已知方向冗余

6385个JVP检查全部zero_exact=true，全部非零有限。FD误差采用原中央差分/JVP相对L2定义，最大0.3131863%，小于原2%门槛：

| FD统计 | 相对误差百分数 |
|---|---:|
| min | 0.0164440% |
| mean | 0.0604426% |
| median | 0.0566419% |
| p95 | 0.0992884% |
| max | 0.3131863% |

发现158条样本中不同instance_id具有逐元素相同的JVP，共170个额外重复方向。逐对CPU只读诊断确认170对的冻结w和单位projected_context也完全相同，符合原冻结前端方向共享，非缓存错配或零向量替代。

这是后续解释属性监督贡献时应保留的限制：不同属性实例未必提供不同局部方向。保持原实例、S_raw、K、权重和JVP，不合并、不删除、不重加权；不据此修改方法或宣称属性模块贡献。逐对证据见outputs/phase25/equal_JVP_diagnostic.json。

## 只读重入与CPU测试

完成后再次调用原完整入口，同时禁止CUDA初始化、官方模型/前端加载、Tensor.backward和autograd.grad。顺利返回：

- CUDA_initialized=false，官方模型加载0、JVP grad调用0、新评分0。
- 全部4356个缓存/协议/完成文件的SHA、inode和mtime保持。
- 最初79条缓存及64份保护文件保持。
- 原summary继续保留恢复那一轮的6130个新JVP/12260次grad和3条Smoke复用计数；这是this_run语义。累计6370个新JVP与5条Smoke复用由两轮metrics、79条快照和独立审计核实，未覆盖原summary伪装单次运行。

21项新增+45项原缓存CPU tests，共66项通过。覆盖K0/7、S/权重/ID顺序、身份/源SHA、z_t、FD/zero/零JVP拒绝、方向冗余统计、无图引用计数器与原autograd JVP数值。

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_phase25 test_conditions -v

本阶段CPU测试不执行G、optimizer或Epoch调度。未修改前阶段测试/训练实现。

## 复现入口与证据

GPU完整入口（本阶段已执行并完成，后续重入只读核验）：

    cd /home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python phase25_prepare.py --allow-full-preparation

只读审计：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python phase25_audit.py --output outputs/phase25/audit_new

审计输出不可覆盖，重审须选新目录。原prepare_full.py --allow-full-preparation同样可直接使用；wrapper只是观测外壳，新增无关文件不改变原producer缓存锁。

证据位于outputs/phase25/：preflight/、cpu_preflight.log、cpu_tests_counter_fix.log、prepare_full.log、prepare_full_resumed.log、两个preparation_attempts/、partial_verified_before_resume.json、audit/、reentry_verified.json、equal_JVP_diagnostic.json。失败现场不删除。outputs/conditions_full/本体不发布。

## 正式训练前待确认

1. 独立审查本次完整缓存审计及OOM观测器修复。
2. 确认正式Epoch数量、lambda/Adam/seed、输出目录与Epoch边界恢复计划，冻结配置后再授权启动训练。
3. 保持validation仅诊断，正式参数更新只使用2048 train；保留方向冗余限制。
4. Gallery评测和后续阶段另行授权。本阶段G forward、optimizer.step、正式Epoch训练和Gallery评测均0。

仅发布本阶段新增3个Python文件、README与中文报告；发布前执行diff --check、文件白名单和远端SHA核对。停止等待审查，不进入Phase 2.6。
