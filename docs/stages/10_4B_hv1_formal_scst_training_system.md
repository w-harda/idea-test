# H-v1 Phase 4 正式训练扩展：2048 Train / 独立 Validation / 周期恢复

本阶段基于 3b26ddf62fd330db3e84fe08a993c678a19cd11b，在 codex/gv2-per-attribute-cache 新增独立 formal_training 包。完成代码、CPU验证及全体 Caption 资格检查，**没有启动任何真实 GPU缓存、H训练、Validation GPU评估、Smoke、100+100、Full，也没有持续监控、自动恢复或重试实验**。正式运行全部留给用户手动执行。

## 1. 本阶段交付

全部新增源码位于 prototypes/attribute_text_policy_h_v1_20261015/formal_training/，原 Phase1–4、G-v2、IRRA、模型和历史 outputs 不修改：

| 文件 | 用途 |
|---|---|
| __init__.py | 独立包 |
| common.py | 明确来源清单、不可变JSON、输出边界、排他锁与安全停止信号 |
| data.py | 原2048/128身份/原图/划分SHA核验，caption-only资格账本 |
| storage.py | Train/Validation隔离、紧凑Storage、特征/metadata/receipt原子一起发布、复用/审计 |
| runtime.py | 冻结G-v2/IRRA接口、只记标量的长程前向计数、原RewardEvaluator真实反馈 |
| checkpoint.py | 周期H/Adam/RNG/顺序/游标/日志SHA、不可变版本链、显式尾段回滚 |
| training.py | seed42重新初始化H、确定性Epoch顺序、既有train_step调用、分项标量汇总 |
| validation.py | 独立固定128图库/128Query，官方IRRA四组检索、Joint mAP选优 |
| cli.py | 默认CPU；GPU缓存/训练/评测均需用户显式确认参数 |
| test_formal.py | 57项新增CPU测试 |
| README.md | 全部真实服务器命令、成功标志、输出、停止/恢复、成本说明 |

新增中文阶段报告为本文件；Windows与服务器同步。数据、资格账本、特征、Checkpoint、outputs、日志不纳入Git。

完整可复制命令见[正式训练README](https://github.com/w-harda/idea-test/blob/codex/gv2-per-attribute-cache/prototypes/attribute_text_policy_h_v1_20261015/formal_training/README.md)。包括 CPU preflight、资格检查/只读资格审计、Train缓存、Validation缓存、正式训练、正常恢复、显式未提交尾段回滚、Validation、CPU选优及各项只读审计。README命令通过实际CLI参数核对；本阶段仅CPU命令被执行。

## 2. 数据资格实测

锁定原2048 Train（1861 PID）及128内部Validation（121 PID），PID交集0。每张图用原G-v2 Manifest的固定Caption、caption_index、row_id和split/role，保留原图SHA/metadata SHA/condition SHA；不读取条件Tensor作为H输入，不重算S_raw、Top-K或JVP。官方Test不参与奖励或选择。

| 项目 | Train | Validation |
|---|---:|---:|
| Caption/图像 | 2048 | 128 |
| Canonical属性覆盖 | 2043（99.755859375%） | 127（99.21875%） |
| 可合法编辑且符合长度条件 | 2037（99.462890625%） | 127（99.21875%） |
| 无Canonical属性，H自然K=0 | 5 | 1 |
| 固定NOOP，不合格 | 11 | 1 |
| 原始BPE截断 | 5 | 0 |
| 超1024 UTF-8 bytes | 0 | 0 |
| 提取Span逐项exact | 全部 | 全部 |

所有11条Train异常资格均永久记录：
- 无Canonical属性：cuhk:26530:0、cuhk:31873:0、cuhk:28228:0、cuhk:11364:0、cuhk:31942:0。
- 原文BPE截断：cuhk:10822:0、cuhk:13924:0、cuhk:4611:0、cuhk:4532:0、cuhk:31553:0。
- 有属性但无合法、改变Tokenizer结果的字符替换：cuhk:7586:0。
- Validation无属性：cuhk:16423:0。

固定协议：不合格训练Caption保留每Epoch访问和原因，固定NOOP、无奖励、无optimizer更新；对应图像仍保留在2048 Gallery。Validation全部128 Query都保留，不合格文本固定NOOP参与四组指标。H自己在合格样本选择STOP/NOOP仍执行原SCST更新，不和资格跳过混淆。

因此正式每Epoch为 **2048 visits / 2037 updates / 11 skipped visits**，10 Epoch候选为20480/20370/110。数据没有被静默删除，未按正确图像挑选属性。资格的无合法编辑数量不等于学习后的Dynamic-K，实际K仍由当前H每Step预测。

Canonical属性数分布（不是G-v2原S_raw Top-K）：
- Train，属性数0..11：5、31、86、178、362、652、413、225、73、16、5、2。
- Validation，属性数0..8：1、6、4、16、21、35、27、15、3。

CPU资格证据目录：
/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/formal_v1/qualification/

文件SHA：
- train.json：44a6d39aa00153acb6d8880e9271370f7ee664b6f79b2b7cacf31c163c216ccb
- validation.json：b3f8be039b18607ffcb4774375e9e459db1569975c8adeb901e246cfb8a5facd
- completed.json：d20e746a56662f3c873cfb806a8bb475e4391aa42e85592b7e66b518e5023195

FORMAL_QUALIFICATION_COMPLETE 与 FORMAL_CPU_AUDIT qualification PASS 已实际出现。qualification是CPU文本规则/Tokenizer资格检查，不是IRRA文本前向。正式输出根目录目前只有qualification；两套正式图库与训练目录尚未生成。

## 3. 方法与正式缓存

H保持原Byte Transformer（width64、2层、4heads、max_bytes1024）、三个Head、gamma3 Dynamic-K与Unicode编辑预算。SCST仍使用当前Sampled与Greedy，RewardEvaluator保持R_sim+R_rank、lambda1/1，best-positive-minus-max-negative Margin不变，记录synergy_increment不另加目标。Adam lr1e-4、betas0.9/0.999、eps1e-8、weight_decay0、batch1、seed42。

新H从seed42初始化，绝不继承10 Query Step32权重。配置默认候选10 Epoch，但CLI必须显式--epochs；不自动授权10轮。正式运行配置冻结后，恢复不得改动轮数、参数、来源或缓存。

正式Gallery分别新建gallery_train/和gallery_validation/协议；一图一次G，Clean/Adv各一次IRRA image，512维FP32归一化特征。Validation原文特征单独缓存一次并跨Checkpoint复用。已有且通过完整协议/身份/SHA的正式单元直接读取，不重新生成。不同role、Mock、Phase3/4 Smoke、官方Test缓存不能混用。图像不合格Caption也不能从Gallery删除。

固定模型：
- G-v2 Epoch10 checkpoint SHA：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
- 官方IRRA weight SHA：9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009。
- 保持原384×128、RGB [0,1]、官方mean/std、fp32归一化和严格8/255预算。训练阶段缓存完成后G与IRRA image forward预算固定0，只新编码三条当前文本。
- G/IRRA eval、requires_grad=False、inference_mode；H训练保留CPU梯度，优化器只包含H。

发现并提前处理一个长程监测风险：原Smoke InferenceRuntime的MagicMock会保留所有Tensor调用参数。正式FormalRuntime使用普通闭包计数，保持原冻结/预算接口，明确不保存历史图像或token；旧模块不改。弱引用测试确认图像forward参数已释放、方法无call_args_list、forward计数正确、退出时移除hooks。未用GPU复现风险，不把静态风险当作已发生的正式故障。

## 4. Checkpoint和中断契约

默认每256次H更新及每Epoch末提交完整H/Adam/RNG/样本游标/固定全体顺序。CLI不允许小于32的正式周期间隔；不是每Step保存完整模型。逐样本JSONL记录当前动作、Canonical/Unicode trace、Sampled/Greedy各奖励分量、Advantage、K/STOP/NOOP、Loss、梯度、参数变化、耗时、文本前向次数和GPU显存标量。

同一事务目录内fsync日志、checkpoint和回执，Checkpoint payload直接绑定段日志SHA与父版本SHA，原子目录发布即为commit边界。原Checkpoint与历史段不可覆盖；恢复逐一核对源协议、数据内容SHA、全体顺序、资格/更新计数和Adam状态。不同Head可能未在所有步骤获得梯度，因此各参数Adam step可不同；严格恢复全部已有状态，不伪造相等次数。

Ctrl+C/SIGTERM：当前样本结束后强制完整提交，可从该边界恢复。SIGKILL/OOM/异常留下pending：默认拒绝恢复。用户明确--rollback-uncommitted才把同来源未提交段永久移入abandoned，记录文件SHA、游标、已提交更新数及说明，再从最新发布版本恢复。尾段可能含已执行但尚未落日志的内存更新，未计为committed；显式回滚后重执行尾段不会重复计入已提交更新。未知来源尾段依然拒绝。

不声称任意崩溃瞬间的Step级无损恢复；CPU测试证明发布边界H/Adam/RNG恢复及后续轨迹exact，CUDA未来长程轨迹尚待用户实测，跨硬件/环境不承诺bitwise一致。Gallery未发布G调用意图不适用训练尾段回滚选项：为了不重复G，保留现场并拒绝自动重算。

## 5. Validation与模型选择

独立固定128 Gallery / 128首Caption Query，包含同PID多个正例，全部合法Query保留。指标直接调用官方IRRA utils/metrics.py::rank，输出mAP、R@1/5/10及mINP，特征单位化余弦。四组是Clean（原图原文）、Image-only（G图原文）、Text-only（原图H文）、Joint（G图H文）。

所有H推理只接TextExample，无图像/PID/配对S/JVP输入。PID仅在奖励绑定层或检索指标层。Validation不能创建TrainGallery，也不用于SCST梯度。每个Checkpoint独立绑定SHA的Query缓存/summary/completed，原文文本只编码一次；NOOP直接复用原文feature。

Validation由用户单独手动启动，可--all-epoch-checkpoints评估已发布的各Epoch。CPU select对已审计结果按Joint mAP最低选优，并列取更早visit，记录与Image-only的百分点差。选优回执不可变。训练不自动触发Validation，正式Test绝不参与选优。128图库小，不能和3074图库Test绝对指标直接比较。

本阶段没有真实Validation指标，也没有正式训练效果、mAP提升或联合协同有效性结论。

## 6. CPU验证与保护

最终CPU命令运行 formal_training.test_formal + test_phase4/test_phase3/test_phase2/test_policy/test_scst/test_tokenizer_real：
**261项全部通过（57新增 + 204历史），157.594秒。**

覆盖：
- 真实Canonical/重复Span、Unicode合法替换、截断/超长与固定NOOP。
- Train/Validation身份隔离、错误Caption/PID、来源SHA、单位特征与紧凑Storage。
- 原子回执、损坏/缺失/未知pending拒绝、已完成单元无额外前向。
- 周期保存、非每步全量、确定性顺序与H/完整Adam/RNG发布边界恢复exact。
- CPU模拟optimizer.step后尚未落日志的硬中断；默认拒绝、显式保留尾段、回滚重执行后最终H/Adam/RNG与连续轨迹exact。
- 日志/Checkpoint篡改拒绝、资格跳过计数、H-only更新、G与编码器模拟参数/buffers不变。
- Validation官方多正例AP数值、四组一致性、全部NOOP Query保留、H冻结、选优与并列规则。
- 默认CPU及所有GPU确认门、输出范围、排他锁、只读审计文件SHA/mtime不变。
- 普通hook不持有历史Tensor、计数与移除正确。

开发初轮49项中两个失败来自测试预期错误（把初始化说明字符串当作读取权重、以及带相似度并列的AP断言）；已改为禁止torch.load的初始化测试和无并列的人工多正例AP验证。未因此更改官方指标或研究方法。之后253、258及最终261回归通过。

674项既有tracked源码/文档及历史H outputs SHA、大小、mtime逐项保持，无历史实验覆盖。临时CPU日志位于/home/lzf/ldx/tmp/hformal_verified_tests.log，资格只读审计日志/home/lzf/ldx/tmp/hformal_qualification_audit.log，均不提交。没有对G-v2 Full做无关重复审计。

## 7. 用户执行前待办与成本

请先独立审查新增代码，确认11条训练资格NOOP及2037更新/epoch的固定处理方式、首次正式Epoch总数与存储目录。由用户按README手动准备两套缓存、审计，然后决定训练轮数；不继承旧Step32模型，不修改来源门禁。

预计计算：2176G、4352IRRA image、128原文text；训练每Epoch最多2037个真实三文本batch、6111文本样本，10 Epoch61110；Validation每Checkpoint最多127条修改文本新编码，原文特征复用。STOP/NOOP会降低Validation实际文本编码量。

依据旧10图缓存和32 Step记录仅粗估：图库准备4–10分钟；每Epoch3–15分钟，候选10轮30–150分钟；GPU显存约1–2GiB。旧32 Step平均update约0.059478秒但仅10图库，不承诺正式吞吐。本阶段没有GPU测量，用户自行检查RTX4090空闲资源与实际日志，不自动改规模。

原始图像feature8.5MiB，Validation原文0.25MiB；元数据/序列化另计。候选10轮约81完整Checkpoint（安全停止会增多），估计0.3–0.5GiB，轻量JSONL约0.3GiB；总建议预留1–2GiB。明确不保存对抗像素。未提交的资格JSON、全部正式缓存/权重/日志都属于服务器资产。

本阶段完成代码/CPU验证后停止。等待用户独立审查与亲自执行，不启动或持续监控正式实验。
