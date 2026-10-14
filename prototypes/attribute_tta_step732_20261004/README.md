# Step7.3.2：Prefix-Preserving Multi-Attribute IRRA-TTA

独立prototype目录：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step732_20261004/`。
不修改正式源码及旧prototype；不训练、不重算S/Top-K、不commit/push。

## 唯一方法变化

- SEQUENTIAL_NEW：round i仅使用原NEW JVP loss L(ai)。
- PREFIX_NEW：round i使用mean(L(a1),...,L(ai))。
- 原loss对JVP vector是线性的，因此预计算未归一化的vector算术平均，与平均原loss等价；单项直接复用原tensor。
- 不对aggregate做L2归一化，不使用S权重/新loss/caption loss。原selected顺序及实例/Q/w/e不变。
- 原TextTransport一次/实例构造原forward JVP，保留zero及FD核验；optimization step内没有text encoder或JVP构造。完整caption只定义既有JVP线性化context及detached retrieval metrics，不能成为额外image loss。
- 直接复用Step7.3 ControlledTTA及官方TTA observer，steps、IG、transforms、原adaptive概率更新、momentum、初始化、投影/clamp均不改。
- SEQUENTIAL_NEW为预先固定的transform anchor；PREFIX_NEW回放真实categorical选择，保留自身adaptive概率权重更新。噪声/trace/RNG必须一致。
- RAW仅引用旧Step7.3结果，没有重新运行RAW/NONE。
- PREFIX并不保证retrieval F数学单调；是否减少forgetting只由用户正式25句结果判断。

## 日志

每轮保存active prefix length及IDs；SEQUENTIAL active length为1，PREFIX为round i。
`selected_prefix_length`是已引入的属性数，两mode都是i。
`loss_gradient_decomposition`保存各原属性loss、未加权梯度norm、除prefix length后的贡献norm、prefix总/平均loss及aggregate梯度norm。
这些诊断在两mode全部更新之后，对保存的未变换round输入点补算；不是实际IG每个节点的梯度归因。
保存mean-gradient重构误差。没有根据诊断值改权重或进行gradient surgery。

保留旧逐step/round的pre/post projection、clipping、累计预算、净更新cosine及逐state cosine/F/paired positive rank。
best prefix按F选，rank差取F-best时的rank，不另用rank选择状态。
这是固定clean竞争图库上的paired-image机制诊断，不是正式PID/shared adversarial gallery评测。

## 已完成验证

- 31项CPU tests通过（18项新tests + 13项旧Step7.3回归）。
- 仅34:0(k1)、0:1(k3)真实IRRA短smoke，每轮2steps/1transform/scales=None；两mode各4轮，共16个optimization steps。
- k2/k4由CPU toy image encoder + 原官方TTA tests覆盖，未额外运行真实caption。
- k1和round1图像/metrics逐值一致；0:1的SEQUENTIAL所有round图像SHA/metrics/noise/trace/RNG与旧Step7.3同配置保存smoke逐值一致。
- 34:0没有旧Step7.3同2-step配置的保存结果：直接复用旧ControlledTTA，并核验PREFIX与SEQUENTIAL；不冒充已与10-step保存diagnostic比较。
- 正式25句入口逐句强制SEQUENTIAL与Step7.3保存diagnostic的图像SHA/metrics/noise/trace/RNG精确回归，失败停止；Codex未执行该正式回归。
- 60个冻结来源路径SHA前后保持，IRRA冻结/无parameter gradient、实例与JVP保持、budget通过。正式tracked/staged diff为空。
- smoke文件：outputs/smoke/；日志outputs/smoke.log；tests日志outputs/tests.log。
- 不从减配smoke判断prefix优劣或冻结scheduler。

## 用户手动正式25句命令

```bash
tmux new-session -d -s tbps-step732 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step732_20261004/run.py --mode diagnostic --allow-diagnostic > prototypes/attribute_tta_step732_20261004/outputs/diagnostic.log 2>&1'
```

默认正式25句、每轮10steps、6transforms/scaled image、scales=.5/.75/1.25/1.5；两个mode各75属性轮。
预计约20–35分钟：依据旧Step7.3三mode实测约30.8分钟粗估，不从短smoke外推，不保证独占GPU。
新output目录必须不存在；旧产物不覆盖。重跑显式指定新的--output及log。

- log：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step732_20261004/outputs/diagnostic.log`
- output：`/home/lzf/ldx/projects/idea-TBPS-test1/prototypes/attribute_tta_step732_20261004/outputs/diagnostic/`
- 完成标志：log末尾COMPLETE、completed.json；case_count=25、attribute_rounds_per_mode=75、steps_per_round=10、diagnostic_run=true。

首先检查：
1. 每句checks.sequential_saved_regression及round1_exact；noise/trace/RNG一致；caption_loss_terms=0。
2. PREFIX round i的prefix_attribute_instance_ids恰为原选序前i个，未重选。
3. loss_gradient_decomposition中之前属性贡献及mean重构误差；注意诊断点定义。
4. summary.modes.*.final_equals_best_count / final_weaker_than_best_count / F_best_minus_final / rank_at_F_best_minus_final。
5. summary.modes.*.rounds.3/4.F_recovery_count；历史SEQUENTIAL为11/19、8/9。
6. summary.comparisons的逐指标wins/ties及per_caption；预算、step数、projection同时查看。

完整25句没有由Codex运行。等待用户结果后再决定是否冻结prefix scheduler；本轮不自动进入其它scheduler/JVP修复。

## 2026-10-04：回归gate排查与修复

首个失败caption为cuhk:21:0，前四句已完成。原失败运行在gate前没有保存该case现场，原日志不能直接恢复具体字段；本轮只用21:0做单case调试，没有运行完整25句。

复现的首次图像/state差异为round4，round1–3输出SHA一致。sequence、每轮noise、categorical transform trace、RNG以及integer rank一致；不是读取错配或随机回放错误。未修改旧Step7.3 NEW循环的直接重复运行也产生该差异，因此不是PREFIX或新的loss变化。

具体数值：
- 图像仅[0,0,109,64]一个分量不同：0.6666666865 vs 0.6745098233，max absolute=L2=0.0078431368，relative L2=5.81662e-5。
- 对应末轮momentum为-3.5762787e-7与+3.5762787e-7，极小fp32差异经sign更新放大。整体momentum relative L2约7.68e-8至8.71e-8；不能将不同运行的momentum宣称bitwise或逐元素全部通过旧allclose。
- paired cosine：0.2280781269 vs 0.2280730754，absolute5.0514936e-6、relative2.2148575e-5。
- F：550.0396729 vs 550.0902100，absolute0.0505371094、relative9.1870585e-5；positive rank均456。
- 历史Step7.3仅有momentum/gradient norm和image SHA，没有完整tensor。本轮live旧代码/新代码tensor对照及来源保存在outputs/regression_live_old_21/，不假装持有原历史momentum tensor。

修复仅涉及checker及失败现场保存：
1. cosine/F使用项目已有atol=2e-6、rtol=1e-6；image SHA、sequence、noise/trace/RNG、integer ranks仍精确检查。
2. 图像差异不是一般fp32 allclose误差，未提高全局image或metric容差。
3. regression_reference_variants.json只登记旧代码直接复现的21:0 round4变体，锁source/report/tensor/history SHA，保留原历史快照。未知图像变化继续失败，不按F择优或替换本次metrics。
4. 接受变体时，checks.sequential_saved_regression显式记录accepted_verified_reference_variant、historical_reference_comparison、accepted_reference_comparison及来源，不冒充历史SHA一致。
5. 其它失败写regression_failure.json，包含caption、实际/旧结果及具体字段、绝对/相对误差，并打印REGRESSION_FAILURE。
6. PREFIX及所有scheduler执行函数源码hash与修复前相同；没有修改TTA/JVP、预算、Top-K或score。

44项CPU tests通过，修复后21:0正式配置单case SEQUENTIAL回归通过（本次恰与历史四轮图像/metrics全部一致）；旧代码参考变体另有实际数据tests确认通过且未知image/RNG拒绝。GPU已停止，正式25句未重新启动。tests日志outputs/regression_tests_verified.log，最终单caseoutputs/regression_policy_verified_21/。

最新用户手动命令（新session/new output，保留原失败证据）：

```bash
tmux new-session -d -s tbps-step732-fixed 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step732_20261004/run.py --mode diagnostic --allow-diagnostic --output prototypes/attribute_tta_step732_20261004/outputs/diagnostic_regression_fixed > prototypes/attribute_tta_step732_20261004/outputs/diagnostic_regression_fixed.log 2>&1'
```

上述命令未执行。新log为outputs/diagnostic_regression_fixed.log，新output为outputs/diagnostic_regression_fixed/；完成仍看COMPLETE与completed.json的25句/75轮每mode/10steps。其它未知参考偏差会继续停止并保存现场；不绕过gate或自动跑更多case。

## 最新：公共 round1 状态分支修复（2026-10-04）

用户重跑在 cuhk:35:1（k=2）报 round1/k1 not bitwise exact。该 gate 合并检查 round1 image SHA 与 metrics dict；旧失败现场未保存 PREFIX 实际 tensors/metrics，无法恢复原运行中究竟哪个 gate 字段不同。不能将新的单case复验冒充原失败快照。

只用该case复验，输入image、相同AttributeInstance、JVP、noise/transform IDs/RNG全部相同。round1直接使用原JVP tensor，无stack/mean、无JVP重算。详细observer记录350次loss全部相同，第一次可复现差异位于round1 step1 IG gradient：
- 首位置 [0,0,0,0]：PREFIX -2.0159453924861737e-5，SEQUENTIAL -2.0159452105872333e-5。
- gradient max absolute 2.3283064365386963e-10，max pointwise relative .03144654（近零分量放大；不作为整体误差）。
- step1 momentum max absolute 7.62939453125e-6；整轮最大1.52587890625e-5。部分近零分量不通过既有逐元素allclose，不能宣称所有momentum都在旧容差内。
- 初始化、每步sign update、pre-step与post-projection image、round输入/输出image、paired cosine/F/rank均exact；当前复验未重现原gate图像或metric不一致。

结论：独立执行两次同一个round1，虽然代码/随机条件/forward loss相同，CUDA反向出现细微数值漂移，不能保证完整状态bitwise相同。本轮未提高容差，也未新增参考变体。

修复：原 ControlledTTA 单属性路径只执行一次公共round1，保留其image、momentum、previous actual change/raw gradient、loss/update事件、metrics，以及完整Python/NumPy/Torch/CUDA RNG状态和categorical tape位置。SEQUENTIAL原样继续；PREFIX从该真实共同checkpoint续跑round2。k=1直接使用同一完整终态。恢复的是已发生共同round1后的随机状态，不改seed、采样规则、变换、noise或后续消费序列。两mode逻辑轨迹每round仍10steps；公共round1物理计算只做一次。

prefix mean、cached JVP、S/TopK、原TTA IG/momentum/sign/step/projection/epsilon不变；只修改execute状态分支及入口传参，增加失败现场保存。其它原有数学函数source SHA保持。旧saved SEQUENTIAL严格gate及已登记21:0变体政策保持，不接受未知image变化，不按retrieval结果择优。

验证：
- 52 CPU tests通过，含k1–4完整分支、单属性更新仅执行一次、RNG延续、非法guidance/trace/settings拒绝及旧回归tests。
- GPU仅35:1 k2原10steps/6transforms/scales配置（SEQ两轮、PREFIX只新增round2）；34:0 k1原10steps配置；0:1 k3、18:0 k4仅两步round1。后两条不是完整k3/k4正式回归，完整prefix逻辑由CPU tests覆盖。
- 35:1/34:0 SEQUENTIAL的历史image SHA/metrics/noise/trace/RNG exact；公共round1 event、image、momentum、metrics差为0。
- 61 frozen SHA、参数无梯度/冻结、JVP/metadata保持、zero exact通过。最大累计L∞=.03137257695198059，超数学8/255约2.79e-8，与既有舍入相同。
- 未重跑25句，不训练，不修改正式源码，不commit/push；GPU停止。

服务器证据：outputs/round1_debug_35/、round1_detailed_35/、round1_common_verified/，tests日志outputs/round1_tests.log，函数源审计outputs/round1_source_audit.json。原失败outputs/diagnostic_regression_fixed/保留。新正式运行output用outputs/diagnostic_round1_fixed/，log为outputs/diagnostic_round1_fixed.log。手动命令未执行：

```bash
tmux new-session -d -s tbps-step732-round1-fixed 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step732_20261004/run.py --mode diagnostic --allow-diagnostic --output prototypes/attribute_tta_step732_20261004/outputs/diagnostic_round1_fixed > prototypes/attribute_tta_step732_20261004/outputs/diagnostic_round1_fixed.log 2>&1'
```

下方上次gate修复和首次smoke为历史记录，不覆盖本节。


## 最新：35:1 SEQUENTIAL 重复性与条件数值 gate（2026-10-04）

本轮只做 cuhk:35:1 SEQUENTIAL_NEW：5次独立engine/reset seed42的正常重复，每次2round×10steps、6transforms、原scales/epsilon/momentum/JVP/TopK/checkpoint/gallery；无PREFIX运行、无25句。每次都是同一record-mode代码路径，未用不同采样路径代替重复。清图、JVP和来源固定，5次sequence/noise/完整categorical trace/RNG exact；74份冻结来源含原61份及新校准证据。完整读数保留，不筛选好看结果。

### Current vs current

5次中的第3次完整命中Step7.3历史图像SHA/metrics；第5次完整命中用户最新失败的round2图像SHA/metrics。因此最新历史差异由当前同一实现自身复现，不支持“只有历史不一致、当前bitwise稳定”的解释。

| 比较范围 | round | image max abs | cosine max abs | F max abs | rank max差 |
|---|---:|---:|---:|---:|---:|
| 全部5次 | 1 | .0627451241 | .0047131169 | 554.412109375 | 643 |
| 全部5次 | 2 | .0627451241 | .0040972829 | 26.5048828125 | 19 |
| 历史round1 exact的run3/4/5 | 2 | .0156862736 | 1.129508e-5 | .0705871582 | 0 |
| 最新失败 vs 历史（run5 vs run3重建） | 2 | .0156862736 | 5.424023e-6 | .0338745117 | 0 |

全5次中，run1/2在round1 step1已发生近零sign分叉，随后扩散到大量像素，不能将其说成纯舍入级image差，也不能为放行它们设置宽全局容差。对应all5详细逐step/round绝对、相对、L2、cosine、首不同坐标都保存在report.json。高相对误差的近零gradient分量不代表整体gradient变化同样大。

run3/4/5的round1图像和metrics均与历史exact，但momentum仍有微小漂移。其round2 rank始终276；最后图像最多9个分量不同、L2差.03327561125，扰动方向cosine最低.999994725034。最新失败的F relative差.000100522，低于该条件实测最大.000209466；不是历史误差大一个数量级。

### 最早来源与放大链

第一处可见差异已经在round1 step1的原sole IG输出，max absolute约1.16415e-10，relative L2约6e-8；不是projection或调度产生。相同forward loss后，CUDA backward出现少量ULP漂移。近零gradient/momentum经sign更新放大：
- 全5次较大分支，首个图像变化为round1 step1坐标[0,2,382,20]，单分量初始差2/255，随后传播。
- 历史round1一致的分支，首个图像变化为round2 step9坐标[0,0,299,45]；momentum跨正/零/负，step proposal差可达到4/255，随后最后一步扩散到少量分量。run3/4/5该位置momentum分别为+2.9802322e-7 / 0 / −3.5762787e-7；较早的round1关键分量IG为0 vs约3.64e-12，触发sign零/正分支。

额外deterministic诊断仅在独立进程设置CUBLAS_WORKSPACE_CONFIG=:4096:8，并开启torch deterministic。在原IG的autograd.grad处，upsample_bicubic2d_aa_backward_out_cuda明确报没有deterministic implementation。配置没有写入正式方法、CLI或环境；没有改resize/IG/TTA算法。早先无workspace的诊断只在CuBLAS前置条件处失败，二者分别保留，不能把前一个错误当作resize证据。

结论：已证实CUDA/TTA数值非确定性，经过sign可放大；没有当前SEQUENTIAL数学行为被修改的证据。不是所有自然数值分叉都可以视为无实质影响。

### Gate最小修复

只对35:1新增经SHA锁定的条件数值reference，不更改其它caption的旧gate、21:0已登记政策、PREFIX scheduler或攻击公式。

条件先固定为“round1图像及clean/round1 metrics与历史exact”，与本次失败条件一致。按该image/state条件取得run3/4/5，不按F/攻击效果筛选；run1/2始终列入完整结果并继续拒绝。第3次同时匹配所有历史round图像SHA，作为历史图像的字节核验重建；旧历史没有原tensor，不能凭SHA不同推算像素差。

新gate同时核验：
- sequence/noise/RNG/transform trace、属性metadata/labels、每轮10steps、epsilon/step/momentum/seed/scales exact；rank/tie保持严格。
- 实际image tensor必须存在；clean与round1 image exact。
- round2逐坐标值必须在实测min/max内；未变化坐标仍必须exact。额外约束max abs≤.0156862735748291、L2≤.033275611248927014、changed count≤9、扰动方向cosine不低于实测.9999947250336625。
- paired cosine范围[.29642167687416077,.2964329719543457]、F范围[336.98590087890625,337.0564880371094]，没有添加任意倍数安全裕量。
- 历史比较原始误差/失败字段保留；接受标记为accepted_empirical_repeatability，不冒充bitwise、旧fp32容差或新的历史快照。
- 数值reference之外的任何改变继续拒绝。其它caption未经本轮校准，不允许套用35:1 tolerance。

execute仅在全部更新完成后提供只读final image引用供gate检查；剥离该输出接口后数学AST与本轮重复时逐值相同，PREFIX scheduler数学AST保持。没有重写loss、TopK、RNG、epsilon、TTA或projection。

### Tests与单case回放

73项CPU tests通过，含原52项及真实5次artifact回放。run3/4/5通过；run1/2较大早期分叉拒绝。更改sequence/noise/RNG/trace/steps/epsilon/metadata/rank、metrics越界、未观测坐标或缺少actual tensor均拒绝；即使image SHA恰等于历史，也不能绕过settings与metadata检查。

最新失败JSON的图像SHA逐轮与run5 tensor核验，再经过修复后的单case数值gate通过。没有启动第6次GPU攻击，仅CPU回放；没有PREFIX实验、25句或训练。GPU停止，正式tracked/staged diff为空，无commit/push。该条件gate不保证所有caption或较大数值分叉能完成25句，也不据本轮宣布PREFIX有性能优势。

服务器证据：
- outputs/repeatability_35/report.json、summary.json、run1–5.json与逐step tensors；
- deterministic_configured_probe.json、source_semantic_proof.json、gate_verified_audit.json；
- outputs/repeatability_gate_tests_final.log；
- repeatability_gate.py、repeatability_reference_35.json/.pt，来源锁provenance.lock.json。
原失败outputs/diagnostic_round1_fixed/与log保持。

新用户手动命令（未执行）：

```bash
tmux new-session -d -s tbps-step732-repeatability 'cd /home/lzf/ldx/projects/idea-TBPS-test1 && PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 /home/lzf/ldx/envs/tbps-image/bin/python prototypes/attribute_tta_step732_20261004/run.py --mode diagnostic --allow-diagnostic --output prototypes/attribute_tta_step732_20261004/outputs/diagnostic_repeatability_gate > prototypes/attribute_tta_step732_20261004/outputs/diagnostic_repeatability_gate.log 2>&1'
```

新output为outputs/diagnostic_repeatability_gate/，log同名.log，仍以COMPLETE/completed.json判断完成。首先检查accepted_empirical_repeatability、repeatability_comparison、round1_exact、noise/trace/RNG与实际步骤/预算；未知偏差仍停止。

以下均为历史阶段记录，不覆盖本节。

