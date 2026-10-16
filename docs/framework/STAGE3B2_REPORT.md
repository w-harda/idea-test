# Stage 3B-2真实G/H/Joint输出等价报告

**最终PASS，仅限固定环境和两条内部Train样本。** 一次新GPU尝试完成；原Stage 3B FAIL不覆盖、不恢复。无重试，无IRRA神经模型/前向，无训练、排名或Stage 4。

基准：`a2988e0ae2a77f41f5416c02e354f4651ed49646`。
独立分支：`codex/tbps-framework-stage3b2`。
工作树：`/home/lzf/ldx/projects/idea-test-tbps-framework-stage3b2-20261016`。
实现Commit由PROJECT_STATE后续文档提交登记。

## CPU前置与唯一新Attempt

新gated入口强制执行完整check_derived，再复用原run_once；原Smoke数值比较、H策略、预算、Tokenizer和加载工厂没有修改。沿用Stage 3B-R派生配置全部75项，parent/derived SHA和唯一增量证明仍有效。新增入口SHA：
`a600141540b5f86b704decc7287bd395918dc6113442ea780e6ca08784ee672e`。

单独CPU命令及实际GPU命令内的CPU门禁均退出成功，日志含GH_RUNTIME_PREFLIGHT_PASS；33份实际导入源码、11个符号及全部7个工厂入口检查通过。CPU过程中权重恢复/神经构造/前向/CUDA初始化尝试为空。

资源检查为RTX4090、free24051MiB、used40MiB、util0%、计算进程为空；实际加载前再次相同检查通过。唯一新输出 `/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2/`，结束GH_PLUGIN_EQUIVALENCE_PASS，退出码0。

## 样本与模型来源

两样本均由原annotation及内部Manifest验证为official train/attack_train，caption_index=0；不根据本轮攻击效果选择。

| row_id | 图像相对路径 | PID | 资格 |
| --- | --- | --- | --- |
| cuhk:9955:0 | train_query/p6841_s9083.jpg | 3193 | 可编辑，6属性 |
| cuhk:26530:0 | test_query/p9613_s16106.jpg | 8578 | 固定NOOP/K0，无Canonical合法候选 |

第二条目录名test_query不代表官方Test，其实际annotation split已核验为train。

- 图像9955 SHA：622ce009e5b1927f4a65171f9cdbb5a5e36e1070c84a0d66b3f09feb4f7e4fa9。
- 图像26530 SHA：8032ef99b7c280aa2588bcdef40233675e7eee65d71e267dc74c3e034d96a3b4。
- Caption9955 SHA：0c85e03c4508cdb222a9bc0480402c835cdcb2b93aada3bd09cb7acd3c73ae29。
- Caption26530 SHA：ab884c4bd6827f38bd7095f5b235c68366a7e86b1a87884cb2f8b06ba1709555。
- G-v2 Epoch10：f59bd3b5f9cbbaf8f6acc10c4bdc113ea83cacb2979ea0821b952c8f9a1c658e。
- H-v1 Epoch7/visit14336：487f3061b6cba2ddabccdd49745a7b99c8c549f4541781b8814b299ef5f279df。
- H原Validation Selection：584e4ba2afc7b17fea3bdfb8501716498b84731fa216558daa2e8e4bb353f757。

G/H真实权重各尝试恢复1次、成功1次；G在cuda:0，H在CPU。IRRA加载0。

## G数值对照

A使用历史原生pixels_from_pil/generate_once，B通过G/Joint插件重新读取并调用同一冻结G；分别记录真实前向输出，不复用A结果。

| row_id | 原图/工作像素误差 | 原始delta误差 | adversarial误差 | effective_delta误差 | 原生/插件L∞ |
| --- | ---: | ---: | ---: | ---: | ---: |
| cuhk:9955:0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0313725471496582 |
| cuhk:26530:0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0313725471496582 |

全部shape=[1,3,384,128]、dtype=torch.float32、finite=true、bitwise exact=true。原始delta与hook捕获也exact。对抗RGB范围[0,1]，两条均严格≤8/255，未使用额外容差；阈值1e-6未放宽。

## H完整动作对照

A调用原TextAttacker.attack，B通过HV1/Joint/_LegacyH.attack独立执行。12个顶层字段及全部嵌套Span、预测重要性、动作顺序、掩码、Unicode与Tokenizer记录规范化后逐字段exact。

可编辑样本：
- budget_k=2，原6个Canonical/Span保留；selected顺序为gender/male的He[0,2)，hair_length/short的short[8,13)。
- 选择ID：5143b71b1ce115bc122e3746；13d375cc76b287c91dd3bd36。
- predicted_importance完整值：[3.6704251766204834, 0.7163031697273254, 0.16556403040885925, 0.04613571986556053, 0.014918110333383083, 0.04356799274682999]。
- 实际2编辑；gender词内位置1（原文索引1）e U+0065 → е U+0435；hair_length词内位置2（原文索引10）o U+006F → о U+043E。
- 属性Head索引顺序0→1；Position索引1→2；Replacement索引21→22；所有legal_indices完全一致。本条未选择STOP或位置NOOP。
- BPE总长度21→25，model_tokens改变，无新截断；H_forward=1/路径。
- 完整动作digest：09858222f47893db24585031f8e9cd6a9a61e961dc9f18dd4e139dc4624d9fb9。

固定NOOP样本：
- eligible=false；reasons=[no_canonical_attributes,no_legal_token_changing_edit]。
- budget_k=0；原/对抗Caption字符串exact，空重要性/选择/编辑/decisions。
- Tokenizer总长度27→27，无变化/截断；H_forward=0/路径。
- 完整动作digest：0cdb3f2971152bb037addc50692bd4d1cfaee337c1609493dc7d14c1f3d47ecb。

Python、NumPy、Torch CPU及CUDA RNG摘要在native前/后/plugin后完全一致。未调seed、改候选、改策略或以最终字符串相同替代完整动作比较。原Caption、对抗Caption、全部Token ID和动作保存在result/report.json。

## Joint、冻结与计数

joint/image_only/text_only三模式均通过；共享同一G/H插件及权重，单模态缺席输出为None，额外前向0。Joint请求故意使用图像9955与文本26530独立身份，不要求正确图文配对。

| 项目 | 实际数量 |
| --- | ---: |
| G加载/完成 | 1/1 |
| H加载/完成 | 1/1 |
| G样本前向 native + plugin | 2+2=4 |
| H策略样本前向 native + plugin | 1+1=2（K0不调用策略） |
| IRRA加载/神经前向 | 0/0 |
| backward/Optimizer/JVP/新条件准备 | 0/0/0/0 |

G/H始终eval、requires_grad=false、参数grad=None；所有实际前向inference_mode=true、grad_enabled=false。前后参数和buffers摘要exact：
- G参数54个Tensor，SHA e247df82150c4b6757e4eaffef421226fa6ed3ed8129ac5e51c289ea777bbb33。
- H参数34个Tensor，SHA 1d262f0be46de22b725a0bbb4b71703496b289ee784c252786ea644ff3f3eff6。
- 两模型buffers均0个，空摘要e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855。

受控run_once段耗时1.8626863285899162秒（不含CPU预检和attempt发布等外围开销）。GPU peak allocated45.1982421875MiB、reserved56.0MiB。进程已退出；结束检查free24051MiB、计算进程为空。

## CPU回归与历史保护

332框架CPU tests（323原有+9门禁测试）及36历史基础通过，共368项；GPU后再次332项框架回归通过。没有无关Full审计/排名重算。

7991历史保护文件SHA/mtime不变；原Stage 3B的17份输出SHA/mtime不变。旧FAIL回执及2358来源、新attempt/result回执及2362来源、Stage 3B-R CPU回执均通过只读核验，无pending。原实验/Stage2.5/3A/3B/3BR树HEAD和clean状态保持。

本轮只新增门禁入口/tests及文档。原75项来源配置、smoke_attacks.py、legacy_runtime、G/H算法、Checkpoint/Tokenizer/缓存/协议/历史结果均无修改。

## 不可变证据与验收边界

- GPU日志：/home/lzf/ldx/outputs/tbps-framework-stage3b2/gpu_equivalence_v2.log。
- strict CPU日志：同目录strict_preflight.log；回归cpu_before_final.log、cpu_final.log、legacy_cpu.log。
- 新attempt/result：/home/lzf/ldx/outputs/tbps-framework-stage3b/equivalence_v2/。
- result/report SHA：50afe401bf57ad244d74a014221825b58a1feb24f07201a42c501404dffecbb6。
- attempt Manifest SHA：731e99aa2b8f19cf30ec1d8cdfad9ee78c1bca12d649d45e204269188bc8ec85。
- result Manifest SHA：26185c4cad3f964c121a4d25d189d8f12dd8db3bfb5bd76dd734bde5c99d1f43。
- CPU只读保护/回执审计：/home/lzf/ldx/outputs/tbps-framework-stage3b2/readonly_audit_v1/result/。
- 原FAIL report SHA仍为e72a65118cb0c216e2fbb0877152d11524b0254f73cc9f2c8547e91900c7a79c。

仅把固定两样本、固定G10/H7、FP32/math/no_tf32、RTX4090-G/CPU-H的等价状态标记VERIFIED。不验证全Train/Test、批量/其他尺寸、其他硬件/精度、采样策略、真实主动STOP/位置NOOP、Tokenizer迁移或RaSa/APTM；不宣称新的攻击效果/Full指标或跨模型迁移。插件保守execution_kind标签保持原状，不据本轮证据授权通用执行。

已用完本轮唯一GPU许可；提交推送后停止，等待审查，不进入Stage 4、Full或新实验。[入口/命令](STAGE3B2.md)。
