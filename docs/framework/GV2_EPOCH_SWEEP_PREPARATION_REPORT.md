# G-v2 Epoch10–60六节点三模型消融：评估准备报告

日期：2026-10-20。基准Commit：6187eed96ed9eff63b8da37f5d98b5adddf75300。
独立分支：codex/gv2-six-epoch-evaluation。
工作树：/home/lzf/ldx/projects/idea-test-tbps-gv2-epoch-sweep-20261020。

**结论：CPU准备及来源审计PASS；真实GPU Smoke因资源繁忙未执行，Full尚不具备启动条件。**
没有新训练、五节点Full图像生成、完整排名或后台部署。没有改动历史模型、算法和实验产物。

## 1. 六节点来源

| Epoch | Checkpoint绝对路径 | SHA-256 | 原架构严格装载 |
| --- | --- | --- | --- |
| 10 | `/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt` | `f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e` | PASS |
| 20 | `/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0020/checkpoint.pt` | `67523b3f10b0145d2ba802b1f468f3bbb7554e36d8e1196376f8380d84a51d09` | PASS |
| 30 | `/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0030/checkpoint.pt` | `a55e1ec2181cd10ef1436c993231f333a445644b5378ca90a10a99c50143a87d` | PASS |
| 40 | `/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0040/checkpoint.pt` | `0b70d802ce867910b10dc70a9581c8a5c31907bf5ab4ace7206411c02f239e82` | PASS |
| 50 | `/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0050/checkpoint.pt` | `8357d8cca55d3b26190e85a53a48034985a6cec37274e2e56e894fe808752b86` | PASS |
| 60 | `/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train/checkpoints/epoch_0060/checkpoint.pt` | `9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049` | PASS |

六点均直接取自已发布训练报告evidence.json。全部CPU严格load_state_dict进入原ImageOnlyGenerator(width32/blocks4/epsilon8/255)。
E10 state SHA为d5681683475cb292199c203022358ecf9172c7f1cffa8127d17197a03cd723c6；
E60为5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47，与训练完成报告一致。
完整各点G-state SHA见[证据](results/gv2_epoch_sweep_preparation/evidence.json)。

E20–60验证gv2-epoch-extension-checkpoint-1、绝对Epoch、next_epoch、完整原contract、唯一父E10 SHA，
以及effective_training_configuration只将epochs改为60。没有恢复Adam/RNG来训练，也没有写旧current.json。
容器在CPU反序列化后只将generator字段用于推理，其余训练状态丢弃；没有修改旧Checkpoint格式或原load_generator。
不添加FiLM、TV、扰动Loss、Scheduler或其他算法。

## 2. Epoch10与历史控制组能否复用

**历史完整结果的来源/输入/原生评分协议严格复用成立；新入口重新生成E10像素的GPU逐值对照仍PENDING。**
这两个结论不能混为一次已完成的新GPU等价验证。

三份原Full的DatasetManifest、官方3074 Gallery/6156 Query顺序、PID、多Caption/正例、
原图/Raw UTF-8输入SHA、generation_v3 Joint Collection、G10/H7来源和原生ModelSpace均匹配。
核对protocol/completed/result/execution及已有CPU审计日志，历史CSV与summary的48个公开字段逐值一致。
没有重算任何历史Full分数/排名，也没有重跑旧CPU audit。

| Target | 原Raw Clean mAP | 原H7 Text-only mAP | 原G10 Image-only mAP | 原G10+H7 Joint mAP |
| --- | ---: | ---: | ---: | ---: |
| IRRA | 66.09744262695312 | 51.471561431884766 | 32.28228759765625 | 23.553220748901367 |
| RaSa | 69.18406677246094 | 32.85859298706055 | 35.04695129394531 | 17.202625274658203 |
| APTM | 64.12683868408203 | 51.801902770996094 | 39.662025451660156 | 30.490341186523438 |

此表全部为历史正式Full引用，**不是本轮Smoke结果**。
[原精度E10两模式CSV](results/gv2_epoch_sweep_preparation/epoch10_reused_raw.csv)；
[历史来源和完整四指标](results/gv2_epoch_sweep_preparation/historical_reuse.json)。

H7/visit14336 SHA：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。
历史6156条文本全部保留（6116编辑/40 NOOP/13537字符替换）；不运行H、不重新选择字符。
每个Victim只复用自身已核验的原文及H文本特征。旧Clean图像特征可作为原控制组来源，但新Epoch图像必须重新进行各Victim原生编码。
新Epoch不能复用旧Adv图像特征、全局相似度Top-K候选、ITM或最终Score，因为像素发生变化。
不把IRRA特征借给RaSa/APTM，不采用SGA/DRA的Matched Clean。

## 3. 新增接线与算法保护

- gv2_epoch_eval.py：只固定旧attacks/materialization的真实生产者命名空间。
- ablations/gv2_sweep.py：六点配置/来源CPU门禁、新Checkpoint只读G装载、逐图单次原generate_once调用。
- ablations/gv2_sweep_target.py：历史原文/H文本特征桥、新图像原生编码、两模式既有Engine评分/发布/只读审计。
- ablations/run_gv2_sweep_full.sh：未来授权后的串行后台入口；缺真实Smoke完成回执直接拒绝。
- configs/framework/gv2_epoch_sweep.server.json：固定六点、同一Raw输入桥、三Target来源、107项Python运行来源、独立输出及资源条件。
- tests/framework/test_gv2_sweep.py：36项新增CPU测试。
- 文档与来源索引：本报告、运行说明、统一项目文档、ADR-0013及results中的JSON/CSV。

没有修改任何既有源码。G网络、单步生成和预算检查均调用原实现。
旧MaterializationPlan只接受G10，新Epoch采用明确分型的独立RGB variant，引用旧Plan/原RGB而不改签旧MaterializedUnit。
真实FP32 original/adversarial/effective_delta按内容SHA、文件SHA、Checkpoint、官方成员顺序绑定，先Claim后前向，
保存后立即读回、budget检查与bitwise内容比较。输入不重新量化到PIL。
G生成与目标评分进程分离；目标模型不参与生成，也没有反馈给G的通道。

原生评分仍为IRRA原dot、RaSa ITM Top128、APTM ITM/global与完整集合归一化；query_chunk128、FP32/math/no_tf32。
公开结果仅R@1/R@5/R@10/mAP、百分数，原精度与两位小数CSV；底层Engine旧schema不改。

## 4. CPU测试和保护证据

- 新增36项PASS，覆盖六点身份/来源、配置/SHA/架构、只取G、顺序、Claim、单次实际CPU Mock forward Hook、
  预算/FP32/shape/有限性、Storage clone、原子中断、篡改拒绝、Full授权门禁、文本原始SHA、双模式原生Mock及只读结果审计。
- 存储/Engine/扩展训练既有99项PASS；原生缓存/Adapter/通用桥适用92项PASS；合计227项适用CPU测试PASS。
- 旧test_final_APTM_config_dry_no_model_or_cuda因旧绝对producer路径集合在新工作树不匹配而失败；
  在未修改的训练基准工作树复现同样失败，适用回归显式排除1项。没有修改旧APTM配置或放宽其来源检查。
- 三Target真实来源/Tokenizer依赖检查已在新入口CPU预检执行，通过；没有Victim权重反序列化或CUDA初始化。
- bash -n、默认无allow-full拒绝执行通过；git diff --check在提交前执行。
- 74184个相关历史文件size/mtime/inode未变，289项来源/Checkpoint/历史prototype SHA未变。
  保护清单包括原物化Store、三个原Full及正式扩展训练目录；不宣称对未读取Baseline重新做了完整审计。

## 5. 本轮GPU状态

即时快照：RTX4090，空闲17803MiB、已用6288MiB，GPU利用率76%，
另一个Python PID1249638占6240MiB；系统可用RAM约115GiB。
没有终止或修改该进程，没有等待/轮询资源。

已停止本任务仍在CPU预检阶段的Smoke父进程PID1319183（exit -15）；记录在
/home/lzf/ldx/tmp/gv2_epoch_sweep_preparation/gpu_smoke.log与resource_blocked.json。
**尚未创建smoke_v1 Attempt，没有G/Victim GPU加载或前向，没有产生新攻击图像/分数/指标。**
CPU权重结构装载检查与实际GPU模型恢复需区分。
本轮不能报告GPU显存峰值、耗时、预算实测或Smoke指标，均为NOT_RUN。

预先固定的Smoke为10 Gallery+10 Query，包含既有NOOP；G10/G60各10次原单步前向，
G10对历史像素误差≤1e-6，G60供三Target，各编码10张新图、0条文本，H加载/前向0。
三份新CPU审计均PASS后才发布GV2_SIX_NODE_SMOKE_PASS。详细命令见[运行说明](GV2_EPOCH_SWEEP.md)。
不因本次资源阻塞创建新版本，不自动重新启动Smoke。

## 6. 正式Full方案与成本

输出拟定为/home/lzf/ldx/outputs/tbps-framework-gv2-epoch-sweep/full/epoch_00XX/。
五节点各生成3074图，共15370次G；每套Gallery共享给IRRA/RaSa/APTM的Image-only与Joint，
文本全部复用历史H7；E10及Raw Clean/H Text-only通过已核验来源只读引用。
独立不可变图像/特征/Score/结果、逐阶段日志和CPU审计；出错即停，不自动重试/恢复。
具体输入、命令、完成标志和恢复拒绝规则见运行说明。

旧三Target四模式耗时约219/9496/8561秒。按两个模式、五节点估算主要评分约12.5小时；
加G生成、来源SHA、CPU预检、发布和审计，暂估15–20小时，缺本轮GPU实测修正。
FP32三Tensor约5.066GiB/Epoch，五节点25.33GiB；加RaSa约25.4GiB、APTM约2.9GiB图像缓存和Score等，
预估新增65–80GiB，配置保留至少100GiB磁盘。原RaSa Full RSS约24.43GiB，建议可用RAM≥32GiB。
GPU启动最低8192MiB空闲、利用率≤10%，仅一次快照，不抢占/等待。
实际磁盘快照约5828GiB可用、inode240488590，未创建正式Full目录。

## 7. 尚未满足的启动条件

1. 当前固定配置下，一次真实G10/G60→三Target Smoke及三份CPU审计PASS。
2. G10新装载路径对历史RGB误差、G60真实预算/冻结/调用次数与资源记录。
3. 用户对五个新增Epoch正式Full的最终明确授权。

因此本阶段状态为CPU_PREPARED_GPU_SMOKE_PENDING，**不是Full已就绪或已运行**。
本轮没有重新训练G/H、生成正式Gallery、运行完整排名或重跑原训练CPU审计。
源码/文档提交推送后停止，等待审查和后续授权。
