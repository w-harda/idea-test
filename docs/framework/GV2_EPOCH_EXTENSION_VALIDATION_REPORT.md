# G-v2 Epoch11→60：续训验证阶段报告

日期：2026-10-20；独立分支codex/gv2-epoch60-ablation；开发基准9ea4467620ddb755d230beedc05fd53e8e1068cf。

实现与验证Commit：1256dced8399370d492d92ea6149531d5d4d4b7b。来源代码及冻结配置均在该Commit内。

结论：158项CPU回归、真实CPU来源预检、唯一一次GPU恢复Smoke和独立CPU审计全部PASS。
本轮未启动正式Epoch11→60、未发布正式Epoch11、未运行任何三模型Full或后台任务。
正式启动仍需用户下一次明确授权及新的即时资源检查。完整Epoch11/50轮稳定性尚未实测。

## 1. 算法和来源

直接调用冻结epoch_training.train_epoch / validate_epoch和serial_training.train_single_sample。
ImageOnlyGenerator(width32,blocks4)、Global/Local/Retain均为原定义、权重1；Adam lr2e-4、
betas(.5,.999)、batch1、8/255投影，逐属性立即更新、前轮detach、最后原图Joint均不变。
没有FiLM、TV、扰动惩罚、Scheduler、新评分、Top-K或JVP。模型训练仍只使用2048 Train；
原128 Validation隔离，Smoke仅其中1条诊断且零更新，不将此诊断冒充全量Validation。

父Checkpoint：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_image_generator_v2_20261014/outputs/train_v2/checkpoints/epoch_0010/checkpoint.pt

SHA-256：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
原配置SHA：7e87bd272abfee440afc13109d439507a34b140506a19a132a8f740afe093bc4；原contract digest：
1ada0c31ee4fd4bf35eaf02fcc5bcebc7a8276c2714f34aaa6d9a88e3501f50b。
11项原训练源码、两项原入口SHA及完整2176条件缓存均核验；父Epoch10关联样本/补充日志由原verify_version核验。
原current.json、原epochs=10及Checkpoint完全不改，不调用会修复旧指针的load_latest。

新协议包含完整原contract、父SHA、原epochs10、新目标60、起点11、唯一允许变化epochs和独立输出。
新配置SHA：ffc3cabb4fe607157aa0c30966242fb699d8ba62ebf0697b9f800eb4ad40136b；扩展contract digest：1b4c446f7759824a54988f6ab631fc2e00e6ed886b84b455d0b4f5c9c668d622。
新协议运行副本：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/resume_smoke_v1/protocol/protocol.json。

原训练Train Total从E1的0.4084998047978737降至E10的0.2636317811724832；
固定Validation Global从0.3991928893374279降至0.26946560155192856，E5曾回升，
不能据此证明已收敛，也不能证明增加Epoch会改善TBPS迁移指标。

## 2. CPU验证

- 新增33项：只改epochs、非法父SHA/配置/来源拒绝、绝对Epoch顺序、G/Adam/RNG恢复、
  原子发布、关联日志与配置SHA、损坏/缺项/pending/重复启动拒绝及边界恢复。
- 原Phase2.1/2.2共102项；ArtifactStore23项；总158通过，CUDA未初始化。
- 合成CPU连续执行与Epoch边界恢复后执行，G/Adam/RNG/样本顺序/样本日志逐值一致。
- CPU Fixture还执行实际execute(smoke=True)完整控制流；仅用CPU设备和ToyEncoder，
  不将它算为真实GPU验证。真实GPU证据仅下面一次独立Attempt。
- 真实tbps-image环境：Python3.10.21、Torch2.5.1+cu124、NumPy1.26.4；
  preflight及独立audit均exit0，GPU Smoke exit0。

## 3. 唯一真实GPU Smoke

预先按原epoch_order(seed42,absolute_epoch11)选择首条cuhk:211:0（K4）。
完整Epoch11顺序digest：8f7924061283c42d62f56a48e90853a97a2c946e6e0761d983f3bb91fd433aeb。
训练首样本4次属性步+1次原图Joint，共5次更新；另1条Validation只诊断。
真实G/IRRA图像前向各6次，backward/Adam.step各5次；新JVP/评分0。
训练G恢复父状态1次，IRRA加载1次；另构造独立G/Adam诊断副本恢复探针1次，副本不做前向。
不是整轮Epoch11，也不是新的正式训练轨迹起点。

| 轮次 | 阶段 | Global | Local | Retain | G梯度L2 | 预算饱和率 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | attribute | 0.107448824 | 0.002518372 | 0.000000000 | 0.592245153 | 67.8508% |
| 2 | attribute | 0.106809318 | 0.007473242 | 0.000238634 | 0.421099424 | 79.1328% |
| 3 | attribute | 0.093011007 | 0.000479555 | 0.000000000 | 0.633382639 | 86.6577% |
| 4 | attribute | 0.080855891 | 0.001764082 | 0.000000000 | 0.426171352 | 87.4919% |
| 5 | joint | 0.042879656 | 0.002886208 | 0.000000000 | 0.516767953 | 67.1278% |

Global/Local为同一真实样本上的训练量，不是正式TBPS检索指标，不据此判断方法收益。
全部梯度有限且非零，梯度范围0.421099424～0.633382639；
IRRA输入梯度范围0.709838540～0.766735228。
累计L∞最大0.03137254901960784，严格≤8/255；无STE或新裁剪规则。
G输入SHA证明前轮detach传递及最后原图重启；参数版本证明每步更新后再进入下一轮。

IRRA在eval且requires_grad=False，原参数/buffers前后tree_hash相等、参数梯度为空，运行断言通过。
原始报告保存了冻结断言结果，未另输出IRRA state tree_hash字符串；不能将其描述成已发布独立参数摘要文件。
Source权重原SHA保持，真实模型未更新。GPU进程已退出，无重试。

## 4. G / Adam / RNG恢复证据

| 项目 | 父恢复 | 探针读回恢复 |
| --- | --- | --- |
| G逐值一致 | True | True |
| 完整Adam逐值一致 | True | True |
| Python/NumPy/CPU/CUDA RNG一致 | True | True |
| 所有Adam累计step | 80680 | 80685 |

父G状态digest：d5681683475cb292199c203022358ecf9172c7f1cffa8127d17197a03cd723c6。
父Adam状态digest：17547545d494139e58417c8bbd31be3b04ba7aa315434722a01a5ed4054e23f5。
父RNG摘要：303b37ca627e6378484f34c5ca9fb296deb2fbd3c12dc93257c3630ae877f7b9。
新G状态digest：420fe14a473f9178e8241215e4bcc94a2ce2ee8278a53aa79a6cefa37fdf4c2e。
四类RNG各自摘要、Optimizer完整超参数见initial_restore/report.json和证据索引。

探针为probe_state/state.pt，独立schema，completed_epoch=null，resume_as_formal_allowed=false。
不创建train/checkpoints/epoch_0011，不将Smoke输出当正式续训父状态。
本轮证明状态完整恢复、同一算法和绝对Epoch顺序；不声称历史CUDA未中断轨迹bitwise等价。

## 5. 资源、时间与保护

启动RTX4090空闲24051MiB、利用率0%、无计算进程；可用RAM约113.91GiB，磁盘约5.74TiB。
GPU peak allocated 1032.008MiB，
reserved 1058.000MiB；进程Peak RSS 2.317GiB。
受控段45.946秒（不含事前CPU预检，包含IRRA恢复、全缓存复核、保存和读回）。
Claim发布至训练样本日志写完约2.736秒，含取样、图像读取和5次更新，不是纯CUDA kernel计时。
不以单样本线性外推50轮；参考原E4–10约84分钟，正式50轮预估10–14小时，受GPU共享与I/O影响。
此估算不含后续三模型Full，不是时限承诺。50轮新增403400步，E60预期Adam484080。
完整Checkpoint每轮保存50份，本轮探针状态约15.84MiB/份（完整Epoch还包含更多progress数据），另含样本日志和来源；建议继续保留≥10GiB余量，
后续全量攻击产物和Full存储需另估算。

历史保护：84237份文件dev/inode/size/mtime保持，
46份关键来源/权重/摘要SHA保持；两者范围不同，不声称全部文件重算SHA。
范围包含G10训练及缓存、H7正式训练、三模型Joint Full、B3、官方SGA和DRA。
原prototypes及其他历史源码Git差异为空，本轮只在独立新框架目录修改。
独立CPU审计PASS：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/resume_smoke_v1/cpu_audit/report.json，无pending或failure，正式train目录仍不存在。

## 6. 文件、命令与下一步

新增src/tbps/ablations/__init__.py、gv2_epoch_state.py、gv2_epochs.py；
configs/framework/gv2_epoch60.server.json；tests/framework/test_gv2_epoch_extension.py；
运行说明GV2_EPOCH_EXTENSION.md、本报告、GV2_EPOCH_EXTENSION_EVIDENCE.json与ADR-0012。
更新PROJECT_STATE、ARCHITECTURE和EXPERIMENT_REGISTRY。代码/测试/配置/文档提交，权重和outputs不提交。

正式预留输出：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/train（未创建）。
唯一Smoke：/home/lzf/ldx/outputs/tbps-framework-gv2-epoch60-ablation/resume_smoke_v1。
原始日志：/home/lzf/ldx/tmp/gv2_epoch60_validation/。
具体CLI见[执行说明](GV2_EPOCH_EXTENSION.md)，文件SHA和完整恢复报告见[证据索引](GV2_EPOCH_EXTENSION_EVIDENCE.json)。

已满足本轮续训恢复验证要求，正式启动仍待用户授权和即时资源预检。
只支持完整Epoch边界；最高边界损坏不回退，未提交Epoch Claim/pending不自动重跑。
尚未验证完整Epoch11、后续长期训练或六点三模型消融；本轮没有正式PID、后台训练或Full任务。
