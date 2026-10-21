# G60 × H1–H10 内部Validation适配性诊断

本实验从be30206f3d6170c06cb34716e3b111f52945c0af派生，仅复用原128内部Validation。
它是探索性适配诊断，不覆盖H7 Selection，不用于官方Test自动选模。

## 实现与预算

- g60_h_bridge：逐文件SHA/完成回执/实际annotation与缓存成员核验；只读复用10组H文本特征及旧G10指标。
- g60_h_validation：复用原ImageOnlyGenerator、generator_state与generate_once、IRRAAdapter和官方rank。
- 前2张作为Smoke，读回CPU审计并检查模型冻结后继续126张。总G/IRRA图像样本前向各128，H和新文本编码为0。
- 原始PIL RGB Resize384×128/ToTensor仅用于读取原图；G输出FP32直接编码，禁止重新量化。
- 每图Claim后前向、不可变ArtifactStore提交、SHA及Tensor exact读回；不支持自动恢复/重试。
- 同一Gallery供所有10个H候选；全部同PID正例，无camera过滤。CPU FP32原生点积，官方rank百分数指标。
- 最终独立CPU进程核验本次产物与11组128×128分数/指标，不重跑历史Full或旧审计入口。

## 运行命令

以下为本阶段受控执行记录，不可重复启动已使用的validation_v1。

```bash
cd /home/lzf/ldx/projects/idea-test-tbps-g60-h-validation-20261021
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src"
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.ablations.g60_h_validation --config "$PWD/configs/framework/g60_h_validation.server.json" --action preflight
CUDA_VISIBLE_DEVICES=0 /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.ablations.g60_h_validation --config "$PWD/configs/framework/g60_h_validation.server.json" --action run --allow-validation-gpu
CUDA_VISIBLE_DEVICES="" /home/lzf/ldx/envs/tbps-image/bin/python -B -m tbps.ablations.g60_h_validation --config "$PWD/configs/framework/g60_h_validation.server.json" --action audit
```

共享GPU准入：空闲显存≥6144MiB、RAM≥32GiB、磁盘≥10GiB；利用率仅记录，不干预其他任务。
资源不足时不创建Attempt、不等待、不重试。正式输出位于/home/lzf/ldx/outputs/tbps-framework-g60-h-validation/；
开发日志位于/home/lzf/ldx/tmp/g60_h_validation_20261021/。

## CPU回归恢复说明

中断前tests_regression.log为158通过/1失败；失败为旧test_gv2_sweep.py:199试图从绑定旧工作树绝对路径的冻结配置中移除新树路径，导致KeyError，未进入被测函数。
保留该日志，不修改旧测试/配置。该用例在原绑定工作树只执行一次并PASS；其他129项公共回归有完整通过记录。
新模块32项CPU回归完整PASS/exit0；包含严格来源缺项/错误SHA拒绝、128顺序、前2张审计失败停止、Storage clone、预算、冻结及优化禁止。
合计162个不同用例验证通过，分日志记录，不声称最初单条测试命令全部通过。
真实GPU和最终CPU审计状态见本轮结果报告，不依据代码准备状态宣称实验成功。
