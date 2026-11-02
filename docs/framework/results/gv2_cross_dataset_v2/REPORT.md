# G-v2跨数据集v2：CPU审计边界修复与执行准备

状态：精确修复及CPU验证完成；新Attempt的CPU Preflight、真实两目标Smoke及独立审计、Full尚待执行，不能宣称PASS或SUCCESS。

基准1ed89e4e857747358b79df0d003c1490385264f3，独立codex/gv2-cross-dataset-v2。原失败3277fa1及264份原输出/控制文件SHA、stat均保持；旧FAIL和旧报告完整保留，不修改、覆盖、清理或重启。

## 精准修复

CPU审计不再把纯CPU torchvision Resize误当成神经模型。仅精确Resize类型、没有参数/缓冲区/子模块/Hook、原forward函数、CPU图像且无需梯度时允许调用。ToTensor保持原纯CPU实现。其余nn.Module前向仍拒绝；CUDA初始化、Tensor/Autograd Backward、Adam/SGD等真实Optimizer覆写step、functional/func JVP和forward AD make_dual均拒绝。ExitStack在正常/异常退出及嵌套场景恢复原方法。

G-v2、generate_once、budget_projection、原生IRRA Adapter/Tokenizer/打分/排名源码字节保持；FP32/8/255、G网络、Batch Size和攻击参数不变。

## 完整CPU回归

原112项与新增29项，共141个不同CPU用例PASS。主工作树140 passed、1 deselected；依赖原绝对路径的missing-runtime-producer历史测试在其原冻结工作树单独1 passed。测试日志保存在新控制目录。

新增验证真实Resize/ToTensor、原FP32像素逐值一致、真实G及Linear/Conv/Sequential前向拒绝、严格Resize边界、CUDA/Backward/Optimizer/JVP拒绝及异常恢复。真实v1 RSTP Artifact只读复制至pytest隔离目录，完整audit路径包括像素对应和完整PID排名重算通过；故意打乱像素的隔离副本被拒绝。该CPU测试不是新真实Smoke，不改旧FAIL。

## 新Attempt

工作树/home/lzf/ldx/projects/idea-test-tbps-gv2-cross-dataset-v2-20261102；配置为该树configs/framework/gv2_cross_dataset.server.json，attempt_version=2。
输出/home/lzf/ldx/outputs/tbps-framework-gv2-cross-dataset-v2-20261102；控制/home/lzf/ldx/tmp/gv2_cross_dataset_v2_20261102。

固定CUHK G E60 SHA9148a6ac06178b2eb474194ef6892c50182c5623b4e88bdcf88faa358aa46049。仅RSTP/ICFG各自官方IRRA的Clean/Image-only；完整1000×2000、19848×19848，所有原Caption与同PID正例。无Target训练/微调、Backward/Optimizer/JVP或H。资源shared_observe_only，无任何VRAM最低门槛，不干预其他任务。两真实Smoke/CPU审计PASS后直接自动串行Full；真实错误停止保留，不自动重试。
