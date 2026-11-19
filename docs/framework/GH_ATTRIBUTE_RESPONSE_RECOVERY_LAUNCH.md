# 属性更新收益引导版：GPU恢复后继续原60轮实验

日期：2026-11-19。用户明确要求GPU恢复后继续原项目，授权继续原训练及剩余评测。

## 真实启动身份

- 恢复工具执行Commit：b601c3af29fb82ad489bd7250b51a100e075dcc0。
- 原冻结工作树HEAD：720c607e70075b9c2625cf1377f142cbe8792906。
- 原科学执行Commit：42a11a307bd28edbbd5eb5baff0f4b6e835c3a9b。
- 配置SHA256：bd7af6e51f56514c79831d8a76cbdbca919c1adf52289e5a2d3d1aec80bfa975。
- Supervisor PID/PGID/SID：2738134，PPID1，脱离SSH、无TTY。
- 当前驱动595.99.02，内核7.0.0-43-generic。
- PyTorch2.5.1+cu124、CUDA12.4，识别一张RTX4090；实际GPU矩阵计算与CPU参考逐值一致。

驱动由用户恢复，本轮没有修改系统驱动/内核或Python环境。
原训练使用的驱动与恢复阶段驱动不同，这一运行环境变化已记录；不承诺跨驱动CUDA轨迹逐位一致。
G/H算法、模型、学习率、属性、Push-pull和图像增强、8/255及Natural-K每词最多1字符预算保持。

## 恢复范围

使用原第30轮完整边界恢复G/H、双Adam、RNG、收益评价器、历史记忆与Replay。
原第31轮未提交，先备份原控制文件和第30轮完整状态/Bank，并封存第31轮claim、IO、Bank和pending Ledger。
所有移动的中断文件都校验SHA、大小及mtime。不是重新初始化G/H或重跑已提交30轮。

继续至40轮 → 训练CPU审计 → G40/H40完整Joint Test → CPU审计；
继续至50轮和60轮，重复同一门禁；最后生成六节点指标/曲线与GitHub报告。
CUHK-PEDES × IRRA，Joint-only，3074 Gallery / 6156 Query。
已有10/20/30节点均审计PASS，保持原结果，不重复运行。
每个科学阶段独立子进程、独立新日志；真实失败保留停止，不自动重试。
启动核验结束后由用户检查进度，不持续监控。

## 证据和检查命令

新控制目录：

    /home/lzf/ldx/tmp/gh_attribute_response_recovery_20261119

原输出：

    /home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116

中断证据及第30轮备份：

    /home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116/recovery_evidence/20261119

服务器检查：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-recovery-20261119/operations/gh_attribute_response_recovery.py --action status

Windows检查：

    ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-recovery-20261119/operations/gh_attribute_response_recovery.py --action status"

训练阶段日志：

    tail -n 30 /home/lzf/ldx/tmp/gh_attribute_response_recovery_20261119/train_to_40.log

状态以新控制回执为准；启动不代表后续训练和剩余Full已经完成。

## 最终启动核验快照

核验时间：2026-11-19T19:41:35.796261。
Supervisor2738134与训练子进程2755266均存活，Supervisor PPID1、PGID/SID2738134，无TTY。
训练子进程实际GPU占用1874MiB；本次状态PID为2755266，已进入第31轮H_ROLLOUT。
恢复游标为第30轮：G242040、H更新61110、H访问61440，未重新初始化。

39387份未提交31轮文件已封存，逐文件SHA/大小/mtime核对PASS；
第30轮完整状态和Bank备份核对PASS。原始控制身份、日志、旧正式结果均保留。
本轮没有重新生成10/20/30节点的正式评测，没有把启动核验当作剩余训练或Full完成。

启动来源见新控制目录gpu_ready_verified_20261119.json、started.json、startup_verified.json；
封存依据见输出recovery_evidence/20261119/completed.json及两个文件清单。
助手在本快照后停止进度检查，后续由用户执行上述status命令。
