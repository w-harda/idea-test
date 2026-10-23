# A1/B首条NOOP分叉：静态溯源与独立诊断准备

| 核心问题 | 本轮结论 | 证据等级 |
| --- | --- | --- |
| 原smoke_v2最早已保存的差异 | 首NOOP梯度SHA/梯度L2记录不同，随后G_after不同；实际最早算子未知 | 已核实历史事实 |
| 同臂是否bitwise可重复 | NOT_RUN：本次GPU资源不足 | 不得推断 |
| A1/B是否存在Tensor级微小噪声 | probe_v2有max_abs 4.1443854570388794e-8的既有证据；不能代替本次Adam诊断 | 已核实，范围有限 |
| 原协议是否保证bitwise CUDA | 没有建立这样的保证：原fp32_runtime不启用deterministic algorithms | 源码确认 |
| 是否存在具体非确定性算子候选 | G含两次CUDA bilinear上采样；PyTorch 2.5.1对应反向实现明确atomicAdd/非确定性 | 一手源码确认风险 |
| 是否确认该算子造成原分叉 | NOT_VERIFIED：无本次GPU边界Tensor，不能把风险写成实测归因 | 待实测 |
| 是否确认跨臂状态污染 | 未发现明确代码证据，但未完成GPU逐Tensor排除 | NOT_VERIFIED |
| 是否修复原训练算法 | 没有；尚无证据支持修改训练数学 | 原源码不变 |
| 原CPU审计FAIL是否改变 | 不改变；保留严格NOOP与首非NOOP同父态失败 | 历史保护 |
| 本次GPU诊断状态 | RESOURCE_BUSY，未创建Attempt、未初始化CUDA、G/Backward/Adam均0 | 执行前资源证据 |

## 1. 执行范围与来源

生产执行Commit：a7597f9c964dda0c8c48973e4678f0b9d3fa095a。
冻结生产工作树HEAD：aafc7715ddafa78f9a30410b6a6ed8852e81780a。
只读审查基准：f2925aef84154ae2a19ba0ecab9afa252ac20e09。
本次独立诊断代码Commit：cc0521b9520258aecc4947a1aae02f1b77d9249a。
分支codex/gh-noop-determinism，工作树/home/lzf/ldx/projects/idea-test-tbps-gh-noop-determinism-20261023。

原配置SHA77ae41af56189ece991ee7cf21fd0ea0f3be5645202bd1cb7cf7e0b2ae735649，
source_key06b5c6a15061e635e49a9e13deb14022956fdbc272ad0ed378676e57ca5f826b保持。
本次没有改动src/tbps/alternating、IRRA、原G/H训练器、Checkpoint或旧Manifest。
代码、函数位置与逐文件SHA见STATIC_SOURCE_INDEX.json；日志与Artifact来源见SOURCE_EVIDENCE.json。

## 2. CPU控制流审查

### 父状态和Adam

冻结state.py:57–69 restore对G/H采用strict state_dict，双Adam先clone再load，并校验内容SHA。
parents.py恢复G60所有Adam step484080，H10为20480 visits/20370 updates，各H参数历史step允许不同。
bind_g_device先移动G再重建绑定对象、完整读回Adam，避免参数对象迁移造成悬空引用。

本次真实CPU读回：
G state=5d48caab6a891a1afd3aaaa6f3594eb5f860ccd7b6090708d289224e6c819e47；
G Adam=84ffa5fdd8132fb60ecbb4caf40d3242973168666841264c098d2aa36575ae14；
H state=1b1e3888ee8dc0e4dff39af3438ec48987cd472746cbfde71ad4d10f562ad24d；
H Adam=dfe930621066f140b5dd37d77d5a4fc4226cdef3f74ff05bc181de86423b6a4d。
这里只读CPU恢复，不宣称本轮已再次恢复CUDA RNG。

### 阶段RNG和遗留梯度

runner.py:30–45两臂从父state克隆开始，每轮restore_complete，先H rollout、再条件接线、再enter_phase_rng("G")。
native_backend.py:157–166恢复双模型/Adam/独立阶段RNG；进入阶段时重新设置requires_grad，G/H均zero_grad(set_to_none=True)，再恢复对应Python/NumPy/CPU/CUDA/sampler状态。
原serial_training.py:73每步再次清空G梯度。
因此“继承A1剩余G梯度”没有直接源码支持；仍需真实Tensor记录排除未覆盖的运行状态。

restore_complete不恢复模块training flag本身，也不清空所有后端Python日志；但原train_single_sample:132强制G.train，encoder.eval().requires_grad_(False)，并清空encoder梯度。
原G只有Conv、InstanceNorm(track_running_stats=False)、SiLU和上采样，无Dropout和持续运行均值。
IRRA/reader维持eval/冻结，并有原始版本/buffer/SHA断言。公开日志、计数和已准备条件不是G前向输入。

原B开始前已经执行完整A1训练及图库刷新，且B的8条条件在首G更新前一次性准备。
新诊断不会执行这些前置神经步骤，故不能排除原长执行历史影响的所有可能性；不能以4次最小诊断替代完整原运行历史。

### NOOP输入和训练数学

conditions.py:16–20在A1或NOOP时直接引用同一original.payload，不进行新Tokenizer/reader/JVP。
旧产物只读复核已确认首NOOP payload内容相同、G_before相同。
本次拟通过根G Hook保存实际image Tensor、shape、dtype和layout，避免只比较文件名或条件SHA。

原serial_training.py:69–114直接执行G→原投影→冻结IRRA→原Loss→total.backward→原Adam.step。
K=0只运行最后一次原图Joint更新，Local和Retain为0，不重新实现该目标。
新工具只观察NativeBackend.g_update，不复制Loss、投影、梯度或Adam数学。

### 数值环境

models/irra.py:14–28 fp32_runtime关闭TF32、cuDNN benchmark、MHA fastpath、flash及memory-efficient SDP，启用math SDP。
它没有调用torch.use_deterministic_algorithms(True)，没有强制cudnn.deterministic，也没有设置CUBLAS_WORKSPACE_CONFIG。
FP32与固定RNG并不是bitwise计算保证；原GPU进程完整这些开关未全部保存，不能从当前CPU默认值反推旧进程实际值。

image_only_generator.py:45–46使用bilinear、align_corners=False上采样。
[PyTorch v2.5.1 CUDA实现](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu)将upsample_bilinear2d_backward_out_cuda明确标记为因atomicAdd而非确定性。
服务器安装的torch/nn/functional.py:4569–4581显示deterministic algorithms启用时走decomposition，否则调用默认bilinear算子。
这是与版本对应的一手源码证据，尚不能证明原分叉一定发生在该算子，更不能未经测量归因于cuDNN。

## 3. probe与Smoke验收标准不一致

probe_v2：
native_numeric.original_gradient只计算同一固定G状态的第一步目标，以autograd.grad取得梯度，不做Adam；
native_cli.py:70允许NOOP max_abs≤1e-6，实际4.1443854570388794e-8。

smoke_v2：
原train_single_sample执行total.backward及Adam.step；
native_cli.py:178–180要求首NOOP的G_before、G_after及完整gradient_observations逐项exact，再要求首非NOOP的G_before相同。
gradient_observations还包含梯度L2等浮点记录，故数值噪声也会触发exact拒绝。

两者不仅阈值不同，还覆盖不同的反向调用和更新范围。probe PASS不能证明Smoke的bitwise前提。
原审计忠实执行其冻结协议，本次不能通过改断言或扩大旧阈值把FAIL改为PASS。
需要修正的是下一版严格实验的控制设计和数值合同；不能事后改写本次结果。

旧首NOOP记录：

| 项目 | A1-r1 | B-r1 |
| --- | ---: | ---: |
| Total Loss | 0.3706851899623871 | 0.3706851899623871 |
| 梯度L2范数 | 0.5205852405058475 | 0.5205852190515232 |
| 梯度向量差max_abs/L2 | 未保存 | 未保存 |
| 首步参数差max_abs/L2 | 未保存 | 未保存 |

两范数差约2.1454324e-8，不是梯度差向量L2。最终Round状态差不能替代首步参数差。
原产物没有首步完整梯度Tensor，因此CPU分析不能补造该表缺失值。

## 4. 独立诊断实现、测试和资源阻塞

operations/gh_noop_determinism_diag.py保存四次完整Tensor，固定A1重复两次和独立A1/B各一次，每次从同一完整父状态恢复。
通过原固定资格NOOP接线，H、IRRA文本、reader及JVP调用均为0。
每次只做原K0 G首步；外部Hook只detach/clone，不返回替代输出/梯度。CPU测试证实记录器不改变合成计算的输出、梯度、参数或Adam。
所有模型/Adam/RNG/模式、逐层前向及输入/输出反向信号均单独保存；比较采用FP64 CPU统计真实差向量。

CPU证据：

- 既有相关121项CPU回归通过。
- 新诊断12项CPU测试通过。
- 在冻结真实源码目录执行4项来源/身份篡改拒绝及资源合同测试通过。
- 共137项不同测试通过，重复执行不重复计数。
- 旧准备态“probe和Smoke均未存在”的测试与当前生命周期不符，未删除、未放宽；不宣称该旧断言当前PASS。

保留三类准备期失败日志：新测试错误消息匹配失败（补齐消息）；旧服务器测试从新目录执行时先触发producer路径拒绝（改为冻结目录正确执行，未改断言）；原prepare_cpu_gate因旧Smoke已存在而拒绝（改为外部独立诊断生命周期）。
独立预检仅把原preflight.output_base派生到新诊断目录，其他来源和算法字段完全不变；原verify_config、全部SHA、CPUStartupGate、父状态恢复和历史保护继续真实执行。
Claim将分别绑定父/派生协议摘要，不冒用旧SHA。该生命周期修复只存在operations工具，不改变生产门禁。

真实CPU Preflight PASS：
140项producer、496项运行来源按原verify_config核验；151378历史metadata、28旧FAIL/Watcher、父权重及probe回执保护通过。
原Smoke57637文件、控制目录24文件保持只读。

即时资源：RTX4090空闲1346MiB，利用率88%；外部进程4059826占5680MiB，4065316 Ollama占17012MiB，均未干预。
RAM约105.36GiB、磁盘约5739.36GiB、可用inode240129469，GPU显存不足。
独立诊断沿用小探针6144MiB显存准入，未修改原Smoke10240MiB要求。
因此本次未启动CUDA、未建Claim/Attempt、未执行前向/反向/Adam，也没有Watcher或等待轮询。

| 要求的实测结论 | 当前状态 |
| --- | --- |
| SAME_ARM_REPEAT_EXACT | NOT_RUN |
| SAME_NOOP_GRADIENT_EXACT | NOT_RUN |
| SAME_NOOP_GRADIENT_NUMERIC | NOT_RECORDED_THIS_TASK |
| SAME_NOOP_UPDATE_EXACT | NOT_RUN |
| SAME_NOOP_UPDATE_NUMERIC | NOT_RECORDED_THIS_TASK |
| FIRST_DIVERGENCE_STAGE | 原日志最早在梯度记录；实际算子NOT_VERIFIED |
| NONDETERMINISTIC_KERNEL_CONFIRMED | 未实测定位；已确认存在已知非确定性算子路径风险 |
| STATE_LEAK_CONFIRMED | 未发现明确证据；尚未完成GPU排除 |

诊断输出预留：/home/lzf/ldx/outputs/tbps-framework-gh-noop-determinism/noop_determinism_diag_v1/，尚不存在。
资源和CPU日志：/home/lzf/ldx/tmp/gh_noop_determinism_diag_v1/。
不得将RESOURCE_BUSY视为数值FAIL或消耗过的Attempt，也不能在本报告中虚构逐Tensor误差表。
待资源满足并获得继续执行指令后，可使用本独立入口，不需要重跑probe或原Smoke。

## 5. 下一版A/B的严格控制设计

完整候选协议见NEXT_AB_PROTOCOL.md，本轮未执行：

1. 所有输入、源状态、双Adam、完整RNG和条件身份继续exact。
2. 先以完整4次NOOP诊断确定实际分叉边界；确定性环境另建独立身份，不静默启用。
3. 相同NOOP公共前缀只执行一次并封存，然后从同一完整状态fork A1/B；首非NOOP也直接从同一父状态执行成对干预，不继承两个已分叉的NOOP结果。
4. 保存真实梯度/Adam/更新Tensor；负控采用预先冻结数值合同，正控必须显著超过独立负控噪声尺度。
5. G→H保持同一H/Adam/RNG、同一sample/greedy文本及文本特征，只切换绑定相应G版本的图库，分开核验Reward、Advantage、梯度与Adam动量。
6. K、Global/Local/Retain、SCST Reward和预算不变；不引入共享属性或新Joint目标。

不能据此宣称已经修复GPU可复现性，不能将原Smoke转PASS，不能自动启动新完整Smoke。
本轮交付是可审查的最小诊断与证据边界；真正的首分叉定位仍待GPU资源允许时完成。
