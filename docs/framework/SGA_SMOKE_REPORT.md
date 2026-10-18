# SGA三模型白盒TBPS适配与真实Smoke报告

实现及证据Commit：d6e719a2cccc5a2d3e01f2465600464358a013fa；分支codex/tbps-sga-whitebox-fixed。

本阶段完成三个受害者的10 Gallery / 10 Query真实GPU Smoke和独立CPU审计。121项相关CPU测试通过；正式3074 Gallery / 6156 Query Full未运行，也未部署后台任务。本文的小图库指标仅验证攻击、梯度和评分链路，不能用作正式Full对比或攻击成功率。

## 来源、算法与适配边界

官方源码：[Zoky-2020/SGA](https://github.com/Zoky-2020/SGA)，固定Commit f49e6926cd13148e49bdbc2c3e537660e096230c；[论文](https://arxiv.org/abs/2307.14061)。官方树/home/lzf/ldx/projects/SGA-official保持只读，18个官方文件SHA锁定，MIT许可证保留于SGA_LICENSE.txt。

原SGA包含文本攻击→图像攻击→文本攻击；这里明确命名SGA-TBPS White-box，优先复现其image-only部分，Query始终Clean。保留官方get_scaled_imgs、原尺寸及0.5/0.75/1.25/1.5四个增强尺度、Gaussian std0.05、双向BICUBIC Resize、随机起点、集合匹配下降目标、10轮、梯度mean-abs归一化及sign更新。官方默认epsilon=2/255；本实验统一8/255，step_size仍为0.5/255。每图从自己paired_image_id对应的全部原始Caption构成集合，按官方Query顺序，不使用其他同PID图像Caption或攻击效果筛选。Full为3074张独立对抗图，服务6156条Clean Query。

每目标独立访问该受害者输入梯度并生成像素，不能解释为IRRA扰动迁移。IRRA目标为负的归一化图文dot集合和；RaSa/APTM目标为各自原生融合正类ITM logit的负集合和。APTM这一路是原生ITM局部可微代理，未声称求导最终Top128和全矩阵minmax＋0.002 global公式。最终评估仍使用三个目标原生Adapter/评分，不用余弦代替融合排名。没有更新受害者参数。

为共享GPU限制，五尺度逐项计算梯度并经官方增强图反传累加，不同时保留五尺度神经图；CPU独立官方原始循环对照通过。每图使用seed42派生的固定随机种子，并恢复外层NumPy随机状态；这是确定性调度变化。当前Torch2.5.1+cu124、Torchvision0.20.1+cu124、Transformers4.44.2，与官方requirements环境不同；官方Resize未显式指定antialias，其默认行为可能不同。当前环境CPU对照不构成原论文环境bitwise等价保证。

## 固定输入与实际Smoke结果

每目标使用相同的官方Test前10张Gallery（cuhk:37132:0至cuhk:37141:0），按既有确定性选择取得10条有合法正例的Query。攻击集合覆盖这10张图像对应的全部20条原Caption；生成10张固定对抗图后再对10条Query排名。Clean特征来自相同目标、顺序和输入协议的已审计Full缓存，不新增Clean编码，也不修改任何旧Full结果。

| Target Model | Attack Method | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| IRRA | Clean | 100.00 | 100.00 | 100.00 | 100.00 |
| IRRA | SGA-TBPS White-box | 10.00 | 100.00 | 100.00 | 39.98 |
| RaSa | Clean | 100.00 | 100.00 | 100.00 | 100.00 |
| RaSa | SGA-TBPS White-box | 40.00 | 100.00 | 100.00 | 46.60 |
| APTM | Clean | 100.00 | 100.00 | 100.00 | 100.00 |
| APTM | SGA-TBPS White-box | 70.00 | 100.00 | 100.00 | 67.94 |

单位为百分数，保留两位小数；没有新增mINP或DR输出。指标只描述10+10 Smoke，不能据此判断三个Full白盒攻击的相对强弱。

每目标加载受害者一次，500个攻击视觉样本前向＋10个最终像素编码，共510；神经文本前向0（只读严格核验的原Clean文本缓存）。IRRA融合pair=0；RaSa/APTM各1200，其中1000攻击pair和200最终Clean/SGA排名pair。各目标600次输入autograd.grad调用＝500次目标梯度＋100次增强VJP。G/H加载与前向、Optimizer、属性JVP均0。

所有100轮输入梯度有限且非零；RGB及扰动有限，像素范围[0,1]。每目标最大实测L∞=0.03137254901960784，以FP64差分检查真实FP32像素，严格≤8/255，无容差。模型eval、requires_grad=False、参数无梯度，参数/buffers前后摘要exact。

## 资源与Full成本

| Target | GPU allocated / reserved (MiB) | Peak RSS (GiB) | 执行耗时 (s) | 梯度L2范围 | Full估计 (min) |
|---|---:|---:|---:|---:|---:|
| IRRA | 930.229 / 1064 | 2.317 | 40.458 | 0.977422–3.941547 | 63.86 |
| RaSa | 2236.807 / 2348 | 4.076 | 50.223 | 19.172163–191.926285 | 129.93 |
| APTM | 1025.789 / 1188 | 2.416 | 40.894 | 7.919187–96.305740 | 142.36 |

执行耗时不含CPU来源预检；包括启动来源预检的外部耗时分别为IRRA128.482、RaSa145.400、APTM133.938秒。启动时三次快照均空闲3865MiB，Ollama PID726931占用20178MiB；利用率分别0%/39%/10%。未终止或修改外部进程。本阶段成功Smoke无OOM。

Full估计＝本模型Smoke平均每图生成时间×3074＋此前同模型B3 Full编码/评分耗时，仅复用计算成本，不借用B3指标。三模型合计约5.6小时，建议共享GPU预留6–10小时；实际耗时受负载、I/O和融合评分影响，不是已测Full耗时。每目标攻击视觉样本153700＋最终编码3074＝156774；RaSa/APTM攻击融合pair各307800，最终SGA评分pair各787968。

三Tensor FP32纯像素约5.066GiB/目标，三目标约15.2GiB；还需原生特征、Score、证明和fsync开销，建议至少35GiB可用磁盘。RaSa Full局部Token缓存可能占用数GiB，建议系统可用RAM32–48GiB以上；这不是进程RSS人为硬上限。实测Smoke reserved上限2348MiB，不能保证Full碎片化和评分峰值完全相同。入口采用shared_observe_only、单次资源快照、最低空闲3584MiB；不足或真实OOM停止，不降低精度/batch/算法，不轮询或自动重试。

## 初始失败保留

最初IRRA Smoke因FP32低端预算边界舍入FAIL，Claim1、成功图像0、受害者加载1。异常路径未保存Hook终值，实际视觉样本数只能记UNKNOWN（5..50），不能伪报精确次数。原代码Commit80361ec1fdb7c4c96e4ec68fa256742cc8188566、原树及18个FAIL文件保持，见[初始FAIL](SGA_INITIAL_FAIL.md)。

派生实现用FP64计算clean±epsilon边界再转FP32并向内舍入，模型、输出仍FP32，攻击预算未改变；增加全部256个像素等级/随机端点测试和失败Hook计数持久化。新来源和输出采用独立v2，不覆盖原Claim或失败证明。

## 证据、CPU测试与历史保护

最终CPU结果121 passed、5 deselected、4已有第三方弃用warnings；42项SGA新增＋79项Engine/RaSa/APTM相关回归。5项旧绝对配置路径检查未运行，不计入通过。覆盖独立官方循环、真实Resize/多Caption、三目标Mock原生可微路径、预算端点、随机状态、Claim顺序、Storage clone、来源拒绝、冻结及Bash Full链一次性/失败退出。最终日志/home/lzf/ldx/tmp/sga_final_cpu_tests.log，Bash语法检查通过。

三组成功证据位于/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/smoke/{irra,rasa,aptm}/completed/及execution/report.json；各对应SGA_TBPS_EVALUATION_PASS smoke <model>。独立只读CPU审计日志/home/lzf/ldx/tmp/sga_fixed_smoke_<model>_audit.log均含SGA_TBPS_CPU_AUDIT_PASS smoke <model>，不重新计算排名。完整来源索引：[派生证据](SGA_SMOKE_EVIDENCE.json)，[Smoke CSV](SGA_SMOKE_METRICS.csv)。

- irra Checkpoint：/home/lzf/Attack/IRRA/logs/CUHK-PEDES/best.pth；SHA-256：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
  summary SHA-256：8c2bd91b4122537d59a2a14c6acea81136705e9dea1d244f9eb395a8c1402898；execution/report.json SHA-256：c35a92a9388d9bebee4f89bcba71a984410d2ecc3c3f7ed8c45c0c6bbb353d9e。
- rasa Checkpoint：/home/lzf/ldx/checkpoints/RaSa/rasa_cuhk_checkpoint.pth；SHA-256：bc85da09c2991d5de503c2c2ac4f032e20cc536fec5094f0acff45e93d0104d2。
  summary SHA-256：817635bd3eb71c7cc779cb376f83c26cc89e2f1104f307c5ff607af037ebcc46；execution/report.json SHA-256：ee0754ba50401c93a7d8d13ffb95df4d7a13ae66d04900702f1f1f17b3d9f14f。
- aptm Checkpoint：/home/lzf/ldx/checkpoints/APTM/aptm_cuhk_checkpoint_best.pth；SHA-256：521c9bf784e7e821ade471deaafa2f25e2501950591e574dc60ca177e9be5a2c。
  summary SHA-256：4368a5c059e6a6a4a78ebb704f873276858713e562dcd012a7aa438834f19983；execution/report.json SHA-256：5f5bb60cdeb5fa992dc999ff7c26c14c04cb032743e36c76493121871b0c8a35。

保护复核：2491份历史源码/正式结果元数据SHA及stat不变、4个旧工作树HEAD/status不变、18份初始FAIL文件SHA/stat不变；另外12203历史文件与既有保护清单stat一致（后者不是新增全文件内容SHA审计）。冻结初始SGA树保持80361ec且干净。G/H/B3、Stage4D、三个官方模型及已完成Full不修改。正式Full输出和后台控制目录在本次交付时均不存在。

## 用户手动正式Full命令

以下命令仅供用户审查后复制执行；本阶段没有执行或部署：

~~~bash
mkdir -p /home/lzf/ldx/tmp/sga_whitebox_full_v2 && { nohup setsid bash /home/lzf/ldx/projects/idea-test-tbps-sga-whitebox-fixed-20261018/src/tbps/baselines/run_sga_full.sh --allow-full >> /home/lzf/ldx/tmp/sga_whitebox_full_v2/launcher.log 2>&1 < /dev/null & echo "启动器PID=$!"; }
~~~

先验证三个既有Smoke CPU审计，再依次IRRA Full→CPU审计、RaSa Full→CPU审计、APTM Full→CPU审计。每目标生成自己的SGA像素后用原生模型排名。任何失败保留现场并停止，禁止覆盖、自动重试或复用未提交Claim。SSH断开不影响独立会话。

~~~bash
cat /home/lzf/ldx/tmp/sga_whitebox_full_v2/phase
tail -n 30 /home/lzf/ldx/tmp/sga_whitebox_full_v2/status.log
tail -n 30 /home/lzf/ldx/tmp/sga_whitebox_full_v2/irra-full.log
for m in irra rasa aptm; do cat "/home/lzf/ldx/outputs/tbps-framework-sga-whitebox-v2/full/$m/completed/metrics_2dp.csv"; done
~~~

最终成功标志SUCCESS | SGA_THREE_MODEL_FULL_AND_CPU_AUDIT。每目标正式输出仅Clean/SGA、R@1/R@5/R@10/mAP百分数两位；Clean复用已审计Full控制组，SGA指标来自本轮新生成像素及实际原生评分。详细CLI、资源及安全停止见[执行说明](SGA_WHITEBOX.md)。

仍未验证：真实3074/6156 Full、原论文多模态SGA全流程、原requirements环境等价、APTM最终排名公式全链可微、其他数据集/硬件/seed泛化；本阶段不自动扩展实验。
