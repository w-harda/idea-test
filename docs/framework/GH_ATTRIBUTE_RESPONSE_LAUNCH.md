# 属性更新收益引导版：Full启动与用户检查入口

日期：2026-11-16。用户已明确授权启动最新版本。G/H从零开始交替60轮，数据CUHK-PEDES，训练源/评测目标IRRA；每10轮保存配对G/H及完整收益评价器状态，执行完整3074 Gallery/6156 Query Joint-only评测和独立CPU审计。

## 冻结范围

实现代码b62415e7319aca720d29e201f2db7c7054f5544e保持。本次只把新未消费配置切换为显式GPU/训练/Test授权，保留所有旧CPU配置修订和证据。
图像严格8/255，文本原Natural-K、每个原词最多修改1个Unicode字符，STOP/NOOP保留。
G内部前瞻属性重加权，H内部合法局部反事实优势与属性头监督，双向收益评价器及GRU历史；全局Push-pull和三视图增强保持。
前瞻是有限分支近似，G→H使用K+1虚拟G步及1次H步；H→G使用1次Global-only虚拟G步。正式H始终等待完整当前G三视图Train Bank。

## 执行链

真实训练Smoke → 独立CPU审计 → Joint执行Smoke → 独立CPU审计 →
从零训练至10/20/30/40/50/60轮 → 每节点前缀CPU审计 → 本节点Joint Full及独立CPU审计 →
六组表格与R@1/mAP曲线、GitHub报告。

每个科学阶段使用独立子进程。只启动本实验自己的Supervisor，不停止任何外部任务。
GPU先实际加载观察，不设固定空闲显存门槛、不轮询等待显存。
任何真实OOM/数值/来源/审计失败保留现场并停止，无自动重试。
助手只验证后台启动身份，不持续监控60轮。

## 输出与检查

工作树：/home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116

输出：/home/lzf/ldx/outputs/tbps-framework-gh-attribute-response-v1-20261116

控制及日志：/home/lzf/ldx/tmp/gh_attribute_response_20261116

服务器直接执行：

    /home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116/operations/gh_attribute_response_status.py

从Windows执行：

    ssh tbps-server "/home/lzf/ldx/envs/tbps-image/bin/python -B /home/lzf/ldx/projects/idea-test-tbps-gh-attribute-response-20261116/operations/gh_attribute_response_status.py"

状态PENDING不代表失败；FAIL_STOPPED_NO_RETRY表示真实失败已停止。Smoke只验证链路，正式结果必须有对应节点Full及CPU审计。
本文件的准备版本尚不宣称GPU Smoke或正式训练通过；实际启动身份由新started.json及后续交接记录提供。

## 启动核验快照

核验时间：2026-11-16T13:38:44.284338+08:00。

执行Commit：42a11a307bd28edbbd5eb5baff0f4b6e835c3a9b。
配置SHA256：bd7af6e51f56514c79831d8a76cbdbca919c1adf52289e5a2d3d1aec80bfa975。

Supervisor PID/PGID/SID：2887061，PPID1，无TTY。
首个真实Smoke子进程PID：2887064。
已实际加载GPU，核验时GPU占用2108MiB，日志进入本次Smoke的完整2048行三视图Train Bank刷新。
这不是正式Round1完成，也不代表Smoke/CPU审计或后续Full已经通过。

启动身份及完整快照保存在控制目录started.json、startup_verified.json；本次只核验启动，不持续监控。
