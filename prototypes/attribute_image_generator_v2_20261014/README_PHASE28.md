# G-v2 Phase 2.8：正式Epoch 4–10与收尾审计

本阶段仅新增入口、分析、测试和文档；既有G、Loss、Retain、投影、IRRA适配器、Epoch调度、Checkpoint及config_phase26.json均不改。复用原outputs/train_v2/和完整2176条条件缓存，不重新评分/Top-K/JVP，不运行Gallery评测。

## 新增文件

- phase28_resume.py：默认CPU preflight；固定从最高完整Epoch恢复，首次必须从Epoch3开始；复用原run_epochs(resume=True, stop_after_epochs=10-last_completed)，首次上限7，总目标仍10。
- phase28_analysis.py：逐Epoch只读验证Checkpoint、关联日志、完整缓存与原契约；复用已验证的audit_epoch和统计，生成Epoch1–10比较。每次读取后释放权重，不保留历史Tensor。
- test_phase28.py：18项新增CPU测试；历史271+新增18=289。
- 本README及docs/stages/09_2_8_gv2_formal_epochs4_10_final_audit.md中文阶段报告。

## 受控命令

工作目录为本目录，环境仍为/home/lzf/ldx/envs/tbps-image。

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python -m unittest test_conditions test_phase21 test_phase22 test_phase23 test_phase24 test_phase25 test_phase26 test_phase27 test_phase28 -v

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python phase28_resume.py

首次正式执行须显式授权参数：

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_CACHE_PATH=/home/lzf/ldx/cache/cuda XDG_CACHE_HOME=/home/lzf/ldx/cache/xdg TORCH_HOME=/home/lzf/ldx/cache/torch /home/lzf/ldx/envs/tbps-image/bin/python -u phase28_resume.py --allow-epochs4-10

中断后显式增加--resume-phase28，仅从最高完整已发布且全部校验通过的Epoch继续，剩余上限为10-last_completed。不使用会无约束运行的原CLI，不重新执行完整已发布Epoch。最高版本损坏直接拒绝，不静默回退。OOM/NaN/预算/恢复/来源错误保留证据并停止，不自动修改方法或重试。

完成后的独立CPU审计：

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/lzf/ldx/envs/tbps-image/bin/python phase28_analysis.py --audit

Epoch10完成后GPU入口拒绝再次运行。默认preflight仍可验证完成状态，next_epoch11只是恢复字段，不表示执行授权。

## 观测、日志与恢复

每Epoch2048 Train、6020属性+2048原图联合=8068 G/IRRA/backward/zero/step，128 Validation no_grad、不更新G/Adam/RNG。逐轮监控仅保存标量、字符串SHA和参数版本整数，不累计图像/特征Tensor或计算图；临时Tensor只用于当前观测，立即释放。

outputs/phase28/attempts/<UUID>/保存本次独立日志、执行上限、恢复记录及失败证据。各Epoch日志独立封存并将SHA绑定到对应不可变Checkpoint；恢复使用新attempt，不覆盖失败日志。正式Checkpoint继续位于原outputs/train_v2/checkpoints/epoch_0004至epoch_0010。

每次保存后新建独立G/Adam，通过原load_latest恢复，G/完整Adam/RNG逐值比较；恢复保存RNG后继续原训练状态。Checkpoint最终Adam80680，完整训练总80680次更新。Epoch1–3及原来源SHA保护。

最终审计单独CPU加载epoch_0010并调用既有image-only推理接口；只有一次G前向，输入为模拟像素图，不读取文本/属性/JVP/IRRA，不进行Gallery攻击或检索。

仅支持Epoch边界恢复，不承诺Step级无损恢复或CUDA未来轨迹bitwise一致。固定内部Validation是paired诊断，不是官方TBPS指标。后续轮投影饱和仍采用原真实clamp梯度，没有STE、扰动/TV Loss或超参数调整。

完整结果和各Checkpoint SHA见中文阶段报告。权重、Checkpoint、数据、缓存、outputs、日志及临时文件均不提交Git。完成Epoch10后停止等待独立审查。
