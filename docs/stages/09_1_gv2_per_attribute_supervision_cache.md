# G-v2 Phase 1：逐属性训练监督缓存

2026-10-14：独立服务器实现完成，45项CPU tests与固定5条真实JVP smoke通过；仅生成5条新监督缓存。完整2176条准备、G-v2模型/训练循环/Retain Loss/生成/评测未实现或启动。用户随后授权将Phase 1源码、配置、测试和本阶段文档发布到独立审查分支`codex/gv2-per-attribute-cache`，基于G-v1发布基线`be7fc25e8c782f7859441601e45c1d59d8c80e41`。不进入Phase 2，等待下一阶段确认。

## 位置与基线保护

服务器实现目录：

`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_image_generator_v2_20261014/`

| 文件 | 职责 |
|---|---|
| condition_schema.py | 完整标识、变长K监督结构、聚合重建与严格校验 |
| cache_io.py | 输出边界、SHA、逐样本原子发布、读取和恢复 |
| integration.py | 隔离加载G-v1依赖，只读复用原plan/JVP |
| prepare.py | 默认CPU preflight及显式固定5条smoke |
| smoke_conditions.py | smoke入口 |
| test_conditions.py | 45项CPU tests |
| config.json | 原前端参数、来源缓存、固定smoke K和聚合容差 |
| README.md | 数据契约、边界和手动命令 |

G-v1整个目录未修改；没有向其添加Python文件。原reader/basis/AttributeInstance/S_raw/Dynamic Top-K/JVP、历史prototype与官方IRRA保持。Phase 1实现验收时，实验工作树tracked/staged diff为空，prototype仍未跟踪，发布工作树干净。后续发布使用独立分支，不在实验工作树切换分支。

G-v1已完成完整Test：mAP 66.09746949792628% → 36.10120901440987%，mDR 45.38185911104737%。旧文档“full未启动”仅为历史快照。本阶段未重算完整指标。G-v1 epoch10权重SHA再次核对为：

`3c7d34f40f0303e9e0b1670f951f4dd449facc6804921d8bccd1b6430710c0a5`

## 数据来源与复用

只读来源：G-v1 `outputs/conditions/`，2048 train和128内部validation。2176条tensor/metadata SHA、标识、原选择序列与K全部审计；保留原样本顺序、每图既有caption及PID划分。原评分图库仍为34054张clean train，不使用Test条件或Test评分。

G-v1只保存聚合JVP，无法从其可靠恢复单属性JVP。本阶段直接复用原score_record、selection、S_raw、K、属性顺序和z_t；按原prepare_uncapped恢复plan，用原TextTransport.read重新计算独立JVP。原selection实现只用于逐值复核，正式plan消费保存的payload，不重新选择或排序。原gamma=3、beta=1、kmax=null、tau=0.02不变。

split明确表示官方annotation划分，本阶段均为train；role表示内部train/validation。即便路径含test_query，也不据此判为官方Test。caption_index必须与row_id末段一致，并与原始annotation核对。

## 持久化数据契约

独立schema：`gv2-attribute-supervision-1`。

| 字段 | 类型与含义 |
|---|---|
| row_id / image_rel_path / caption_index / split | 完整样本定位标识 |
| role / pid / caption | 内部划分、PID与原始完整caption |
| K / selected_instance_ids | 自然K与原S_raw降序ID列表，tie保持原顺序 |
| S_raw | float64 [K]，保持原JSON分数精度 |
| score_weights | float64 [K]，S_i / sum_selected(S)，不使用S^gamma |
| attribute_jvps | detached CPU float32 [K,512]，原retrieval JVP尺度 |
| z_t | detached CPU float32 [512]，原完整caption单位化retrieval feature |
| v_local | detached CPU float32 [512]，S加权聚合，不再次单位化 |
| has_local | bool，与K>0一致 |

**z_t仅作为Global Loss监督缓存，不作为G-v2输入。G-v2输入契约为image-only G(I)。** 本阶段未实现G，未沿用G-v1的文本/FiLM条件输入。

K=0协议写死为空ID、空分数、空权重、attribute_jvps形状[0,512]、v_local为零向量、has_local=false；z_t仍保留。K=0不制造属性fallback，也不提前定义后续训练调度。

metadata保存原score_record/selection、来源entry与SHA、逐属性zero/FD检查、三类聚合误差和监督用途。读取会核对ID与JVP行顺序、原S_raw、权重、K及完整身份。

聚合采用原fp32乘法与sum表达式：

`v_local = sum_i score_weights.float()[i] * attribute_jvps[i]`

构建、写入和每次读取都重新验证。缓存重建与原设备/G-v1对照分开记录；数值门禁atol=1e-9、rtol=1e-5只用于缓存聚合检查。原JVP仍要求zero exact与central FD relative error不超过0.02，未改数学定义或其门禁。

## CPU与真实smoke结果

45项CPU tests通过，包括K=0/1/7、完整标识、非法路径/分数/权重/JVP、聚合损坏、SHA损坏、跨样本错配、禁止覆盖、进程锁、符号链接逃逸，以及一次真实子进程退出后pending不参与恢复。默认CLI只执行CPU preflight；本阶段没有full入口。

固定按G-v1 manifest顺序取各预定K的首条，不按新结果挑选：

| role | row_id | K | 聚合重建最大绝对误差 | 与G-v1聚合最大绝对误差 |
|---|---|---:|---:|---:|
| train | cuhk:26530:0 | 0 | 0 | 0 |
| train | cuhk:26747:0 | 1 | 0 | 0 |
| train | cuhk:9955:0 | 3 | 0 | 0 |
| train | cuhk:26196:0 | 7 | 0 | 4.656612873077393e-10 |
| validation | cuhk:17663:0 | 4 | 0 | 0 |

K=7的CPU/GPU聚合求和舍入差relative L2为6.911324222744376e-8，通过既定门禁；不能把全部G-v1对照宣称bitwise exact。每条新缓存自身聚合重建均exact。

- 新单属性JVP：15个；原zero与FD检查通过，FD最大relative error为0.0007834365491016138（0.0783436549%）。
- `new_scoring_backwards_this_run=0`；原评分API被失败断言保护。
- JVP内部`autograd.grad`调用30次，单独报告；不能称为完全无梯度计算。
- 训练Tensor.backward、optimizer step、G forward均0；没有G-v2训练循环。
- 新计算类型：原IRRA text-forward JVP及zero/central FD检查、CPU聚合重建与缓存SHA检查。
- 完成态重入在禁CUDA初始化、禁autograd.grad、禁Tensor.backward、不可导入integration的条件下通过；未重新计算JVP。
- 4479项来源/源码/config SHA在smoke前后保持，IRRA/reader冻结。
- smoke总计49.44秒，包含来源审计/加载/IO，非纯GPU计算计时，也不作全量耗时承诺。

本阶段GPU进程已结束；服务器其他既有进程未停止或修改。加载官方victim时打印的“Training Model with ... tasks”为其初始化日志，本阶段未训练victim或G。

## 输出与恢复

仅在独立目录outputs中写入：

- `outputs/smoke/protocol.json`：原前端协议、固定样本与完整来源SHA。
- `outputs/smoke/samples/<row_id>/condition.pt`及`metadata.json`：逐样本监督与审计。
- `outputs/smoke/manifest.json`、`summary.json`、`completed.json`：完整覆盖、逐样本误差、计算计数与完成门禁。
- `outputs/smoke.log`：真实smoke日志。
- `outputs/verification.json`：45项CPU tests结果及完成态禁CUDA/梯度重入验证记录。

summary SHA256：`8250b301604bd02bed65c58eaa838e4644a6261367748d1bc2053c68e1ce003d`。

逐样本目录先pending写入/fsync，原子发布后才参与恢复；锁在protocol写入前取得。完成样本不覆盖，同协议且来源/身份/SHA/聚合校验通过才恢复。完成缓存再次运行仅CPU核验。此处仅承诺缓存恢复，不承诺G-v2训练轮内或step恢复。

## 手动检查命令与后续边界

服务器进入本阶段目录后：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_conditions -v
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python prepare.py
PYTHONDONTWRITEBYTECODE=1 /home/lzf/ldx/envs/tbps-image/bin/python smoke_conditions.py --allow-smoke
```

最后一条在实验工作树的当前完成输出上只核验，来源变化则拒绝。完整2176条缓存尚未准备，当前只有5条接线验证，不能据此评价攻击效果。后续完整缓存入口、模型、逐属性/联合训练、Retain定义、多轮预算和正式评测需新阶段确认。

Phase 1发布范围仅为独立目录8个源码/配置/测试/README文件，以及从Windows同步的本中文阶段文档。数据集、条件缓存、模型权重、outputs、临时文件、Windows状态管理文档和其他历史未提交改动均不纳入Git。发布阶段只执行CPU检查，不重新运行GPU实验。GitHub副本不包含实验资产；源码中的历史绝对路径仍指向原服务器，只克隆仓库不能执行真实JVP smoke。
