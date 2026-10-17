# Stage4D-1阶段报告：全量H历史文本物化与G流式Runner

开发基准a5a6f6ae8c9200482ae0c70d228e1873d3ca4dd2；独立分支codex/tbps-framework-stage4d1，工作树/home/lzf/ldx/projects/idea-test-tbps-framework-stage4d1-20261017。实现Commit：7f50cd1922d2610703170d3a7e10d8c69ab5255a；本后续文档提交登记。

本轮只执行历史文本CPU复制与审计。G/H/IRRA真实加载、神经前向、GPU、JVP、Backward、Optimizer和检索排名均为0。不修改历史prototypes、模型、旧Store、投影、Unicode规则、Checkpoint或正式结果。原Stage4C2整体FAIL及Stage4C3小样本PASS保持。

## 交付文件与职责

| 新文件 | 职责 |
| --- | --- |
| src/tbps/materialization/bulk_session.py | 新Store子类成员索引、明确来源的作用域SHA/stat缓存；原Store不改 |
| src/tbps/materialization/full_runner.py | H历史全文逐条发布/读回/集合，G单次工厂流式Claim/生成/证明 |
| src/tbps/materialization/full_prepare.py | 默认CPU预检、显式16条Pilot/6156条全量入口；没有GPU选项 |
| src/tbps/materialization/full_audit.py | 独立只读全文、来源、顺序及历史保护审计 |
| src/tbps/materialization/throughput.py | 合成CPU存储吞吐对照，非真实攻击效果 |
| configs/framework/stage4d1_materialization.server.json | 固定3074/6156、预算、统计、C1/C3来源回执，GPU授权false |
| tests/framework/test_stage4d1_full_materialization.py | 19项会话/预算来源/顺序/事务/G工厂Fixture测试 |
| tests/framework/test_stage4d1_readonly_audit.py | 4项独立读回、缺失、篡改、统计拒绝测试 |
| docs/framework/STAGE4D1.md | 实际CPU命令、产物、失败策略及下一阶段受控方案 |

另更新PROJECT_STATE、ARCHITECTURE、EXPERIMENT_REGISTRY与ADR-0007；没有修改旧实验文档。

## 来源与协议

统一完整MaterializationPlan：7ac4fefd0abaa10df085be3ec3ed56c4b38c782a21bfc44473f40b90965447d5。

- CUHK-PEDES official Test有序Gallery3074、Query6156，保留多Caption/全部NOOP。annotation SHA fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0。
- G10 SHA f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e；H7 SHA 487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。
- IRRA训练来源SHA9ff6209551ea8d6ee80396214a2534301fba5d58fa2c5a1875a338f61b295009仅来源描述/文件SHA，不恢复模型。
- Stage4C1来源审计Manifest SHA484605d0501d94dbb4a21315ba0f61c76499e17d283fd2a51c6eb7a1d96c9ce7；Stage4C3 PASS Manifest SHAe6aa5f89e9fce861bfd900c5160629b6f54bc196a858b43ee341116539062f70。
- 原Joint Full H源385批/1155份metadata、intent、receipt；不加载旧features.pt，不重新运行H，不重新选择字符。

G2推理为G(I)，不需要3074 Test条件缓存；本轮不补算S_raw/Top-K/JVP。H旧Span、属性、Dynamic-K、STOP/NOOP、Tokenizer审计与完整action逐条原样保存。目标受害者的Tokenizer/Score/Gradient不参与物化。

## 性能检查与必要优化

相同代码、131来源、1024 Query Manifest的Fixture对照，仅发布24文本+8图像：

| 指标 | 原Store路径 | 新流式会话 |
| --- | ---: | ---: |
| 总秒数 | 4.013664692640305 | 4.279980655759573 |
| Peak RSS bytes | 411230208 | 412356608 |
| 产物bytes | 14477930 | 14478308 |
| 产物文件数 | 156 | 156 |
| 被仪器捕获的ArtifactStore.checked调用 | 665 | 274 |
| 这些调用声明的累计文件bytes | 1143064927 | 34029498 |

仪器只计ArtifactStore.checked处；新会话首尾另有138/138来源SHA，未计入上述累计bytes。因此不能将该比值当作磁盘总I/O或速度提升。对照没有证明墙钟加速。CPU profiling显示fsync占约1.64秒/134次，仍为主要耗时，全部保留。

确认的重复开销：每单元重复读取固定Checkpoint/来源SHA、完整Manifest线性查询、完整来源stat及重复完整动作验证。局部优化仅新增成员索引、固定FileRef作用域缓存和64条完整action验证结果缓存；首/强制集合边界/尾完整SHA，直接来源每次stat，全部来源最多1秒完整stat扫描，输出/未知文件始终完整SHA。任何来源变化拒绝最终完成语义，异常恢复模块与锁。不关闭fsync、原子发布、pending、Claim或内容SHA。

证据位于/home/lzf/ldx/outputs/tbps-framework-stage4d1/throughput_{baseline,streaming}_final/benchmark/，Manifest SHA依次161e46682800623d137e7283667c0ca51c8513ce2423ea53ec58ff01d27d5787、dfd852610372a4107097659f955814cd654f0aed1f8a5a924d0f6a0d6f73d0a8。开发期Fixture基路径和导入作用域错误的独立输出保留；修正基准调用路径后通过，无真实模型、无Full重试。

## 真实H执行

16条Pilot PASS：16编辑/0 NOOP/38次替换；发布/读回10.393729396164417秒，记录总段10.746773470193148秒，Peak RSS607338496 bytes，88文件/4332205 bytes。Pilot没有全量Collection回执。Pilot result Manifest SHA0e6ccb33618ca4ada747071d9ba9df1a1d620c9c5114902044c76db979923a60。

唯一全量CPU Attempt退出码0，TEXT_FULL_MATERIALIZED：

| 项目 | 实际值 |
| --- | ---: |
| Query单位 | 6156 |
| 编辑 / NOOP | 6116 / 40 |
| 历史字符替换 | 13537 |
| 发布/读回/完整集合秒数 | 878.6321567669511 |
| 记录总段秒数 | 879.7607615217566 |
| Peak RSS bytes | 607948800（579.78515625MiB） |
| 最终文件数 | 30797 |
| 最终文件内容bytes | 89401019 |
| 文件已分配blocks bytes | 194134016 |
| H/G/IRRA真实加载、前向及GPU Attempt | 全0 |

逐条原文、对抗UTF8、完整action、Span及原历史来源立即读回；6156/6156最后进度后另用约69秒完整集合读回及来源SHA检查，才发布语义完成。输出无pending，无新图像单元。原Joint Full和原FAIL单位没有复制、改签或覆盖。

来源会话开始4445项SHA；集合强制边界采样已完成8890次尾SHA、220505次命中、797次完整stat扫描。该session字段在退出前取样，不包含退出时额外4445次完整尾SHA；退出后又强制核验Plan，再发布完成语义。它不是完整运行的总SHA调用计数。

H全量根目录/home/lzf/ldx/outputs/tbps-framework-stage4d1/materialized_full_v1/：

| 证据 | SHA-256 |
| --- | --- |
| result/manifest.json | a149a57fc7e8c87a389ede80c7e2eaedafec9d3ffc81dff9a7275915040c615f |
| result/report.json | 83bdc9e2bd16b45ca43eb5b42b6244967bd5cfca67440ad43f30d2664c9ec50d |
| artifacts/protocol/protocol.json | ac77b8d0f8abb3918f7daa8194c8929ee3e504a29bd502b312ad812deb25a9f6 |
| artifacts/text-full/manifest.json | 19aa71a104d5682dc26fc5aba86c52ac9c73f91dbea240232c805d84e8dcebc6 |
| artifacts/text-full/text_collection.json | 9ab8881dab9f82bd31cc31c0a7cc478e3beca9b2f9b940443603efa54d21069b |

独立CPU只读审计退出码0、TEXT_FULL_CPU_AUDIT_PASS：全部6156原文/Adv UTF8、完整action/trace及官方Query顺序exact；有序动作摘要afe9b2a7302a4c135984768aa07a33d175ed18d3200dc08f970c137ae555f868与Stage4C1一致。30797新文件SHA/mtime未变，7991历史文件、3472Joint缓存、70旧FAIL文件SHA/mtime不变，8个保护工作树HEAD/clean保持。没有缺失、篡改或pending；Image/Joint全量PENDING。

审计110.86877850443125秒，Peak RSS607436800 bytes。证据/home/lzf/ldx/outputs/tbps-framework-stage4d1/full_audit_v1/result/：Manifest SHA913b87ecf0620730e537a33f96858e4dff41fbb6de8fa233b085da077fbbc9a2，report SHA114857db1ca8c29b7a44cf592ab11693feeab0c823dc3f87bac6a3543143866f。原Full目录只读快照digest ed2c18d9e64715419d488ea0ac764586486e719aa9e39b4c38e743e5740dd163。会话统计为内部采样，Pilot的source_SHA_final=0采样发生在退出前；真正退出仍执行完整尾SHA，后续强制Plan核验也完整读取SHA。

## G Runner与成本

16张合成CPU G Fixture单次工厂，16次Mock生成、0真实模型加载/0真实G前向；事务、读回、生成证明及完整集合共7.866312887519598秒，约0.4916445554699749秒/图，28531234 bytes/208文件。证据G_stream_fixture_v3/artifacts/fixture-stream-report/。该Fixture Manifest SHA7cd1ac0383cd06aed40808fc4922d6404aa194736a42986c37f243f011dcaabd。报告中G_loads=1表示Mock会话工厂，不是恢复真实G10。

按这一小规模事务吞吐线性外推3074图约1511秒/25.2分钟，仅供存储调度量级估计；不是GPU计算实测，也未包含全量来源/最终完整证明额外成本。真实G延迟、PIL读取、同步及文件系统负载需Stage4D-2记录；建议先冻结宽裕的1小时执行预算作为待审方案，而非本轮已批准阈值。

纯FP32三Tensor数据为5439356928 bytes=5.0657958984375GiB；另加序列化、每图metadata、Claim、Proof、Manifest/完成回执及完整集合。建议至少10GiB空闲空间及100000空闲inode；实际预检空闲6487733911552 bytes、240954143 inode，RAM可用123708166144 bytes。H与G新单位、目录及证明均消耗inode，不能仅按6156+3074估算。

新G路径只保留当前图及小型FileRef索引，不缓存全部图像Tensor；载入后detach().clone，三Tensor约1.69MiB/图。当前H/Fixture RAM数据见实测，不设置2GiB进程RSS上限。旧Stage4C2两图真实G曾记录GPU allocated42.3857421875MiB/reserved52MiB、会话RSS1015422976 bytes；这不是全量预测保证。Stage4D-2须基于实际空闲GPU/RAM冻结资源策略并核查任务冲突，本轮未查询或使用GPU。

## 测试、保护与下一阶段

严格运行导入闭包CPU预检GH_RUNTIME_PREFLIGHT_PASS、退出码0；保持75项固定来源和33份旧实际执行文件，禁止真实模型反序列化、CUDA初始化和神经前向。日志/home/lzf/ldx/tmp/stage4d1_runtime_closure.log。

591 CPU tests通过（23新增，568既有），69.90秒；补充审计代码修改后4项针对性回归2.45秒通过。覆盖来源缓存边界/篡改/恢复、Manifest索引、完整文本计数/顺序、Pilot不能冒充Full、未完成Claim/pending在工厂前拒绝、单次工厂与已完成项跳过、独立读回与UTF8损坏拒绝。

历史保护及新产物SHA见上述已完成独立审计。Stage4D-2只能在新授权后，使用本轮固定Plan/producer路径及H完成集合，复用原G插件工厂单次恢复G10、batch1顺序生成3074图，先Claim后前向，严格预算/读回/生成证明；未提交前向不得自动重做。完整执行方案与命令见STAGE4D1.md。

当前不授权Stage4D-2、GPU、IRRA Full、RaSa/APTM或训练；H文本完成也不能称Joint物化或跨模型攻击成功。
