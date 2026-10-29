# C-PGC/SEP v2资源阻塞后的只读复用与恢复

## 事实与恢复起点

原v2于2026-10-29 01:40:43在RaSa Target的runtime.resources() CPU准入阶段停止，未创建RaSa Target目录，非GPU OOM或攻击错误。旧代码四项门槛合并成一个RESOURCE_BUSY且未保存失败快照，无法确证当时具体哪一项不足，应标NOT_RECORDED，不推测为显存或RAM。

当前资源快照见SOURCE_EVIDENCE.json：GPU空闲24051MiB，RAM、磁盘与inode均满足冻结门槛。原生执行仍为36cf91f97ea3c483883eb6d93456c51ef7d662f0及配置4b56eb60a3f2c928ef084fad64f1648bee17129e450b707a4d166169bc8a370a，算法、UAP、模型精度、8192MiB门槛、Top128和全部评分规则完全不变。

18个已有Source/Target/Audit步骤的全部Manifest、回执、实际payload和来源SHA只读验证PASS；包括C-PGC六组Target Smoke、对应六份独立审计、三个Source Smoke，以及CUHK Full Source、IRRA Full、独立CPU审计。没有重新调用成功GPU或CPU审计入口，没有排名重算。正式完成1/12。下一步骤为CUHK RaSa Target。

## 独立调度身份

调度代码单独位于operations/cpgc_sep_resume.py，薄Shell为operations/run_cpgc_sep_resume.sh。冻结原生工作树与配置不改，所有新GPU步骤仍调用原native CLI，沿用原输出v2；RaSa原本没有Attempt，因此是合法首次执行。原v2控制目录的started、FAIL阶段日志和已成功产物原样保留。

新控制：/home/lzf/ldx/tmp/cpgc_sep_resume_v1_20261029/
共享恢复互斥锁：/home/lzf/ldx/tmp/cpgc_sep_v2_resume.lock
输出仍为：/home/lzf/ldx/outputs/tbps-framework-cpgc-sep-official-v2/

## 有界资源等待

启动每个GPU步骤前逐项记录显存、RAM、磁盘、inode、阈值及缺口；只监测资源时不import torch或加载神经模型，不占用CUDA计算显存，不向外部进程发信号。GPU利用率只记录。每300秒检查，所有步骤累计资源等待最多43200秒，不含正常实验耗时。超时保留证据并停止。

若即时资源检查通过、原CLI再次检查时资源回落，仅当Traceback确认为冻结runtime.resources()的RESOURCE_BUSY且该步骤输出目录仍不存在时，才允许等待后再次进入CPU准入。这不是重试GPU Attempt。出现任何FAIL、pending、部分Attempt、SHA不一致、OOM、审计失败或其他异常立即停止，不删除、不修复、不重跑。

原生CLI每次仍执行来源及资源门禁。成功步骤逐项验证后跳过；新GPU结束后按原计划独立CPU审计，完成后进入下一组；SEP仍保留其六组真实Smoke门禁。全部12组审计完成才生成正式结果并调用原独立发布工具。

## CPU验证

23项新CPU测试PASS，覆盖原生命令和顺序、显存/RAM/磁盘/inode定位、保留8192MiB门槛、利用率不否决、状态读取异常拒绝、总等待预算、只允许启动前资源回落重入、已存在FAIL/Claim/pending拒绝、篡改SHA拒绝、配置不符拒绝、成功产物只读复用和部分Attempt拒绝。真实服务器inspect只读复核18个既有完成步骤PASS。没有重跑旧审计、模型推理、Smoke或UAP生成。

## IRRA已完成正式结果

| Mode | R@1 | R@5 | R@10 | mAP |
|---|---:|---:|---:|---:|
| Raw Clean | 73.15 | 89.83 | 93.73 | 66.09 |
| Matched Clean | 69.44 | 87.22 | 92.02 | 63.02 |
| C-PGC Joint | 47.27 | 70.99 | 79.47 | 42.76 |

百分数；原精度CSV与审计/summary严格一致。不可把Source、Smoke或其他历史G/H指标混入Full。后台是否真正运行以新控制status.json、PID和阶段日志为准，准备报告不预先宣布恢复成功。
