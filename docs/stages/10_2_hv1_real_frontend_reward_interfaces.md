# H-v1 Phase 2：真实属性前端与训练奖励接口接线报告

基准Commit：58657901c0cd148470a1475a3667dcce17b2c6cc。分支：codex/gv2-per-attribute-cache。代码在服务器发布工作树；本阶段只做CPU接线与验证。发布Commit以Git记录为准，避免报告写入自身Commit造成循环。

## 实现结果与保护边界

新增文件均位于 prototypes/attribute_text_policy_h_v1_20261015/：

| 文件 | 职责 |
|---|---|
| phase2_sources.py | 固定来源SHA、路径边界与只新增文件的IO |
| text_frontend.py | Caption-only Canonical提取与原文Span适配 |
| training_inputs.py | G-v2原2048/128样本身份与来源验证 |
| feedback_backend.py | CPU Mock冻结后端、Unicode反馈校验及训练奖励绑定 |
| feature_cache.py | 独立train Gallery特征缓存、SHA、紧凑Storage与发布回执 |
| inspect_phase2.py | 最多16条真实train Caption的CPU只读检查 |
| test_phase2.py | 56项新增CPU测试 |
| config_phase2.json | 冻结前端来源、检查范围及原奖励权重声明 |
| README_PHASE2.md | 接口、CPU命令、缓存边界与下一阶段方案 |

另新增本中文阶段报告 docs/stages/10_2_hv1_real_frontend_reward_interfaces.md，并同步到Windows同名文档。不提交Windows状态管理文档、任何缓存、数据、权重、outputs或日志。

未修改H第一阶段源码/配置/证据、G-v1、G-v2、IRRA、Checkpoint、训练/Full Test结果。100份保护文件前后SHA一致；包括已跟踪G-v1/G-v2/H-v1代码、H第一阶段证据、G-v2 Epoch10、IRRA权重、Full summary/completed及conditions_full Manifest/Protocol。未读取或重算Test指标。

## 训练来源与推理隔离

直接核验G-v2完整缓存：

- Train 2048条、1861 PID；内部Validation 128条、121 PID，PID交集0。
- Manifest SHA：506caf30b8f82b8a29fc35e06df7bb04a5b2bcddc58a1f159b524bb107e7511e。
- Protocol文件SHA：9ad4903a6de6dc221df1a15af48d9e63e6332d7fcd3a9a62ecd3f11a2ed91c9d。
- Raw annotation SHA：fc3d15542f91dd70e23028c2cb40171a8f41734a39afddb9505e352e396cabf0。
- 内部split SHA：33ec828fdf45de86d77a3fcaa23918953085fd2257d825150e4600d430687891。
- 全部2176条metadata与condition文件SHA、row_id/image_rel_path/caption_index/split/role/PID/Caption及原计划对应关系通过；只哈希condition文件，不加载其中Tensor。
- 训练目录指纹：a4736eacd7bbf70e91e4a09504e24a009dc9828e0c04e45cf9859ff4682fb8dd。
- 仅对所选16张train原图核验图像SHA；没有声称重新哈希全部2176张原图。后续每次图像使用前verify_image。
- 原annotation包含官方各划分，但绑定只允许其train行；没有用Test PID、图文对、标签或属性构建策略/奖励。
- Validation身份可以只读检查，require_train及缓存/奖励接口禁止其参与反馈。

文本前端只接受Caption字符串。它复用冻结规则提取器及v5.5文本适配器，不加载reader、basis或配对评分；准确Span从原provenance传递，不通过字符串搜索重定位。两个重复词分别保留位置和属性ID。H输入仅为TextExample的Caption与六个文本属性字段；PID、图像、旧K、S_raw及JVP不进入H。

## 小规模真实Caption检查

固定按SHA256("42|row_id")排序取前16条train。没有全量文本编码或检索评测。

| 项目 | 实测 |
|---|---:|
| 至少一个可解析属性的Caption | 16/16（100%） |
| resolved属性总数 | 80 |
| 精确匹配原文的Span | 80/80（100%） |
| 存在合法字符替换的属性 | 80/80（100%） |
| 存在合法编辑的Caption | 16/16（100%） |
| 单字符合法Unicode替换候选 | 288 |
| 原Caption截断 | 0/16 |
| 合法候选新截断 | 0/288 |
| 原Byte-BPE总长度（含SOT/EOT） | 20–66 |
| unresolved候选 | 2，保留原原因并排除 |

属性候选数量分布：2→2条、3→1条、4→2条、5→5条、6→4条、8→2条。均匀分数的容量K与该候选数一致，仅用于上限诊断，绝非学习到的重要性或原S_raw。

固定seed42、未训练H的CPU预测Dynamic-K：K2→12条、K3→3条、K4→1条。贪心实际编辑0–3次，允许STOP/NOOP；没有强迫用满预算。这些数值只能说明初始化接线，不证明属性选择质量、泛化或攻击效果。H结构及Dynamic-K公式未变。

80/80反映已解析属性的Span/候选覆盖率，不代表所有人类属性都已提取，也不等于语义标注准确率。只有16条确定性样本，不能外推2048条整体分布。

## 奖励与图库接口

本阶段实际运行的是合成CPU Mock，缓存协议明确execution_backend=cpu_mock、feature_space为mock而非官方IRRA。没有真实权重加载器，无法从本阶段CLI启动真实模型。

- 训练图库只含catalog.train；query对应自身图像的索引必须存在；所有同PID图片共同组成正例集合，其他PID为负例，要求至少一个负例。
- 每个Mock图库图像一次G前向、两次图像编码（Clean/Adv），通过原G-v2投影保证相对原图8/255；完整参数/buffers状态及梯度检查保持冻结。
- 真实Unicode文本使用官方Byte-BPE（77位置）生成token，交给TextMock；不调用真实IRRA文本网络。
- GalleryFeatureCache按完整样本身份、目录指纹、producer/source SHA和模型状态绑定；特征保存/加载clone为紧凑独立Storage。已发布有效单位不会再次调用G。
- 先独占pending并留下计算意图；失败或缺少发布回执时拒绝静默重算。保留失败证据，不自动删除或修补。
- 未改第一阶段RewardEvaluator、SCST、奖励权重或模型。R=R_sim+R_rank；synergy_increment=adv_sim_drop−clean_sim_drop仅诊断，不进入目标。SCST参考奖励保持stop-gradient。

best-positive Margin存在与多正例mAP的差异：正例0.9/0.8、负例0.7变成正例0.9/0.6、负例0.7时，最佳正例Margin保持0.2，而AP由1降为5/6。因此不能仅凭Margin变化宣称最终mAP提升或无提升。可选后续讨论平均/最低正例Margin、正负pairwise目标、列表式soft-rank；均未实现或替换当前奖励。

## CPU测试与过程记录

| 范围 | 通过项数 |
|---|---:|
| Phase 2新增测试 | 56 |
| H-v1第一阶段回归 | 68 |
| G-v2历史CPU回归 | 289 |
| G-v2评测及缓存修复CPU回归 | 68 |
| 合计 | 481 |

覆盖Caption-only隔离、重复Span、Unicode字符索引、非法动作/预算/截断、来源/身份/SHA错误、训练与验证/Test隔离、同PID多正例、paired绑定、负例缺失、缓存损坏/回执丢失/pending拒绝、恢复不重复G、紧凑Storage、Mock与真实缓存隔离、冻结参数/buffers、无G/IRRA梯度、SCST奖励stop-gradient和synergy恒等式。

初轮有三项测试断言问题：误把scst_loss返回的二元组当Tensor、AP浮点严格相等、Mock NOOP分数要求逐值精确零。仅修正新测试，保留原奖励数值路径；NOOP接受1e-6以内浮点误差。首次来源检查使用了非紧凑JSON计算protocol_hash，被原metadata门禁拒绝；新入口改为精确使用原cache_io.digest序列化，未改旧Protocol或任何证据。失败日志保留，最终通过。

真实G前向、真实IRRA前向、GPU操作、新评分backward、新JVP及真实训练更新均为0。CPU回归日志中的EPOCH/EVALUATION输出为合成fixture，不是正式训练/评测。少量SCST梯度测试仅使用合成CPU标量/原合成接口。

## 证据与下一阶段待办

服务器证据根目录：

/home/lzf/ldx/projects/idea-TBPS-test1-github-sync-20261014/prototypes/attribute_text_policy_h_v1_20261015/outputs/phase2/

关键文件：inspection.json、inspection_final.log、cpu_h_verified.log、gv2_regression.log、evaluation_regression.log、protected_before.json及completed.json。数据/日志仅服务器，不发布GitHub。

下一阶段需要独立实现并审查真实冻结G-v2 Epoch10/IRRA加载、原图预处理、真实512维图文特征空间对齐、real Gallery缓存生产者及首次受控GPU奖励链路。保持每张train Gallery图像一次G；128 Validation和官方Test不参与SCST奖励。真实H训练调度、Checkpoint和任何检索评测尚未实现或运行。

CPU命令和详细接口方案见同目录README_PHASE2.md。本阶段完成后停止，等待独立审查。
