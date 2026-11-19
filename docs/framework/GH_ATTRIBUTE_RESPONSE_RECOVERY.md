# 属性更新收益引导版：第30轮之后的恢复

日期：2026-11-19。用户已授权恢复停止的程序并完成原实验剩余流程。

## 准备阶段结论（15:13历史快照）

原实验已提交并审计通过30轮，第10/20/30轮完整Joint Test与独立CPU审计PASS。
第31轮日志到H阶段2048/2048，尚未完成Ledger原子发布与完整状态提交，因此恢复必须从第30轮完整边界开始，重做未提交第31轮。
系统最近启动时间为2026-11-18 00:40:47；没有原程序退出回执，不能仅凭这一时间认定历史停止原因。

目前服务器能识别RTX 4090，但NVIDIA内核模块未加载，/proc/driver/nvidia/version与/dev/nvidia0均不存在。
nvidia-smi退出码9：NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
因此本次GPU续训尚未启动；不是空闲显存阈值拦截。没有修改驱动或系统配置，没有启动等待驱动的轮询进程。

## 已有正式结果

CUHK-PEDES × IRRA，Joint-only，3074 Gallery / 6156 Query；三组独立CPU审计全部PASS。

| 轮次 | R@1 | R@5 | R@10 | mAP |
|---|---:|---:|---:|---:|
| 10 | 13.43 | 24.89 | 31.74 | 11.02 |
| 20 | 11.37 | 21.39 | 27.29 | 8.98 |
| 30 | 9.94 | 19.17 | 24.74 | 7.92 |

第30轮累计正式更新：G242040、H61110；H访问61440。
第31轮已实际执行但未提交的计算另存中断证据，不计入已提交30轮；恢复后的正式训练账本仍以完整边界为准。

## 恢复范围及保护

恢复工具位于独立工作树：

    /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-recovery-20261119

分支codex/gh-attribute-response-recovery。继续调用原冻结工作树的训练、评测和审计入口：

    /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116

原来源HEAD720c607e70075b9c2625cf1377f142cbe8792906；
科学执行42a11a307bd28edbbd5eb5baff0f4b6e835c3a9b；
配置SHA256 bd7af6e51f56514c79831d8a76cbdbca919c1adf52289e5a2d3d1aec80bfa975保持。
没有改G/H、Loss、属性、增强、预算、原Manifest/SHA验证或科学配置。
图像严格8/255，文本Natural-K、每个原词最多修改1个Unicode字符。

新的控制目录：

    /home/lzf/ldx/tmp/gh_attribute_response_recovery_20261119

启动前拒绝驱动不可用或真实CUDA分配失败，不设最小空闲显存门槛。
GPU恢复后先核对来源、原30轮状态、六份G/H权重及三份收益评价器Checkpoint和已有审计。
只有精确符合本次中断身份的以下产物才可封存：

- boundaries/claim_0031，必须绑定原第30轮state SHA。
- round_io/round_0031。
- B/bank_31。
- ledgers中唯一intent为round_0031、配置SHA一致的.pending目录。

封存到原输出下recovery_evidence/20261119，逐文件记录并验证SHA、大小和mtime。
复制原控制日志与身份，另存第30轮完整恢复状态及对应Bank，避免原滚动退休策略清掉唯一回退边界。
不删除中断证据，不覆盖历史Ledger或Checkpoint；封存后原生严格恢复器的后续claim/IO护栏仍原样生效。
GPU不可用时不会移动这些产物，也不会消费started.json。归档或实际运行失败后停止，不自动重试。

恢复后的顺序：

    完整恢复G/H、双Adam、阶段RNG、收益评价器/记忆/Replay
    → 第31—40轮 → 40轮训练CPU审计 → G40/H40 Full → CPU审计
    → 第41—50轮 → 50轮训练CPU审计 → G50/H50 Full → CPU审计
    → 第51—60轮 → 60轮训练CPU审计 → G60/H60 Full → CPU审计
    → 六节点指标、R@1/mAP曲线与GitHub报告

已有10/20/30轮结果不重新生成。训练最初仍为从零初始化，恢复属于同一条训练轨迹的继续，不是第二次从零训练。
本次没有扩展新的模型或数据集实验。每个科学阶段使用独立子进程和新阶段日志，原Supervisor不会重启。

## 验证

15项CPU恢复用例PASS，覆盖精确中断身份、错误父状态、已发布31轮拒绝、未知待发布目录拒绝、
归档后内容/时间保持、30轮备份、驱动失败保护、无显存门槛和剩余三个节点审计门禁。
实际恢复CPU预检RECOVERY_CPU_PREFLIGHT_PASS；原冻结训练来源预检GH_AR_CPU_PREFLIGHT_PASS。
以上均为CPU检查，不代表GPU续训已经启动。

## 服务器指令

由用户或管理员先恢复服务器NVIDIA驱动；确认nvidia-smi正常后，执行一次后台启动：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-recovery-20261119/operations/gh_attribute_response_recovery.py --action launch --allow-recovery

查看恢复进度和六节点结果：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-recovery-20261119/operations/gh_attribute_response_recovery.py --action status

查看恢复Supervisor末尾：

    tail -n 40 /home/lzf/ldx/tmp/gh_attribute_response_recovery_20261119/supervisor.log

驱动检查未通过时supervisor.log尚不存在。状态BLOCKED_NVIDIA_DRIVER_OR_CUDA表示GPU条件未满足，尚未消费恢复身份；
FAIL_STOPPED_NO_RETRY表示已发生执行错误，保留现场，不重复执行launch。

不持续监控；启动成功后的实验进度由用户按以上命令检查。

## 19:33恢复启动更新

用户已修复GPU并重新授权继续原项目。RTX4090/595.99.02与原PyTorch CUDA实际计算PASS。恢复Supervisor2738134已后台启动。19:41确认39387份中断文件封存校验PASS，训练子进程2755266已恢复第30轮完整游标并进入第31轮H_ROLLOUT，GPU实际占用1874MiB。实时状态与命令见[启动交接](GH_ATTRIBUTE_RESPONSE_RECOVERY_LAUNCH.md)。本节取代上文驱动阻塞的准备期状态。
