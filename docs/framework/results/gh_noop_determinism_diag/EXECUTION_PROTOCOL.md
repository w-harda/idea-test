# 首NOOP独立数值诊断协议

本轮仅noop_determinism_diag_v1，不启动原smoke_v2，不执行完整A1/B，不刷新图库，不训练H。
冻结生产执行a7597f9c964dda0c8c48973e4678f0b9d3fa095a，冻结工作树HEAD aafc7715ddafa78f9a30410b6a6ed8852e81780a。
外部operations只读观察原NativeBackend.g_update→LegacySteps→train_single_sample。原Loss、投影、FP32及Adam不变。

## 身份与上限

固定cuhk:26530:0，K=0；A1_repeat_1、A1_repeat_2、A1_cross、B_cross依次执行。
每次独立恢复完整G60/H10、双Adam和双阶段RNG，随后H固定资格NOOP（不调用H神经网络）、原条件接线、G阶段RNG恢复和原首个G更新。
共G forward/backward/Adam≤4，IRRA image≤4；H forward/update、IRRA text、reader、JVP、图库刷新均0。
不执行原8条H rollout及其余7条条件/JVP；因此不能完全模拟B在原Smoke前置全部条件重算后的执行历史，限制必须报告。

完整Tensor只存服务器，不提交Checkpoint。每次保存G参数/buffers前后、Adam前后、实际输入/条件、G输出、投影RGB、IRRA feature、原损失记录、每参数grad、G叶模块的前向/反向边界、完整RNG及模式。
观察Hook只detach/clone，不返回替代输出或梯度；CPU测试核验有/无Hook的梯度/Adam逐值一致。

## 独立预检与资源

原preflight.prototype要求smoke_v2/A1、B尚不存在，现为已完成生命周期，不能直接调用prepare_cpu_gate。
独立工具先verify_config原配置和全部producer，再仅派生preflight.output_base至新诊断目录。
原preflight其余所有数据、SHA、协议、样本、配置断言不变；父状态仍从原配置完整恢复。
Claim记录原/派生protocol摘要及唯一允许变化字段，不把派生协议冒充旧来源。
CPUStartupGate仍真实确认上下文退出，之后才初始化CUDA。原probe回执、原Smoke生成回执/FAIL日志只读核验。
诊断继承原RAM/磁盘/inode门槛；采用已获小探针授权的6144MiB显存准入，利用率仅记录，非新Smoke资源门槛。
执行前即时检查一次。资源不足不建Attempt、不启动CUDA、不轮询；已有诊断目录或锁冲突拒绝。
只允许一次执行；失败保留Claim/FAIL，不自动重试。

## 数学环境与判定

本次只观察原fp32_runtime：不启用新deterministic设置，不改变cuBLAS环境。
按同臂重复及跨臂分别输出exact和max_abs/L2/relative_L2/per-Tensor数据。
梯度范数之差不等于梯度向量之差。本诊断不改写旧审计FAIL，不用容差使旧结果转PASS。
若要做第二套确定性环境，必须独立新Attempt、另行授权和明确预算，不占用本次4次之外的计算。

## 执行

使用tbps-image Python、冻结src import路径和显式--allow-gpu-diagnostic。
输出：/home/lzf/ldx/outputs/tbps-framework-gh-noop-determinism/noop_determinism_diag_v1/
日志：/home/lzf/ldx/tmp/gh_noop_determinism_diag_v1/
后台使用nohup+setsid，保存PID/PGID/SID；不调用原native_cli的probe或smoke入口。
