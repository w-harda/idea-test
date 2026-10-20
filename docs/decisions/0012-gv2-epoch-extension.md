# ADR-0012：G-v2 Epoch10父状态与独立扩展训练身份

日期：2026-10-20。决定：采用；正式训练仍待用户另行授权。

## 背景
原正式训练的epochs=10是完整来源契约的一部分。修改旧配置或跳过SHA会破坏历史证据。
G-v2实际为Image-only G(I)，没有FiLM、TV或扰动惩罚；本任务不改变算法。

## 决定
保留原Checkpoint、完整contract、配置、current.json和关联日志；新扩展contract嵌入完整原contract、
父Checkpoint SHA与独立输出路径。唯一有效训练配置差异是epochs 10→60，起点绝对Epoch11。
原Adam、Python/NumPy/CPU/CUDA RNG完整恢复；样本顺序调用原seed+绝对Epoch算法。
训练调用冻结train_epoch/train_single_sample，不复制K+1、Loss、Retain或投影实现。

扩展Checkpoint使用新的schema和ArtifactStore；每个完整Epoch不可变发布、SHA绑定样本日志。
仅允许显式边界恢复；缺项、损坏、pending或未完成下一Epoch Claim均拒绝自动重跑，
尤其不回退到较旧Checkpoint掩盖损坏。SIGINT/TERM请求在完整Epoch发布后停止；
进程崩溃的未提交Epoch保留现场，后续需人工审查和单独授权，不能声称Step级恢复。

单样本真实Smoke保存probe_state而非epoch_0011，completed_epoch=null、
resume_as_formal_allowed=false；正式训练目录不创建。它证明真实父状态可恢复及一次K+1更新，
不证明完整Epoch11或后续50轮训练完成。

## 影响与边界
续训具有相同算法、状态和绝对Epoch顺序；CPU合成连续/恢复轨迹可逐值比较。
GPU上验证G/Adam/RNG恢复exact，不宣称与2026-10-15未中断CUDA轨迹bitwise等价。
扩展协议不授权改变学习率、模型、Loss、缓存、数据、Source或训练机制。
本阶段不实现三Target Full消融流水线；六个预设评估点保留，后续另行授权。
