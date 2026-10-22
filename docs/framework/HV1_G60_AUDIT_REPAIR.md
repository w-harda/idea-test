# H60独立CPU审计修复与既有产物复用

日期：2026-10-22（Asia/Shanghai）。本阶段为审计门禁修复，不改变攻击或训练算法。

## 修复与来源隔离

原GPU执行Commit：`9765a536a11a9a0e9111f2f834230c53d41bf2d8`。
原冻结工作树HEAD：`1b16d653d17573d3dafaa7a26bd84bf8d31949ac`。
原工作树：`/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-adaptation-20261021`。
修复工作树：`/home/lzf/ldx/projects/idea-test-tbps-hv1-g60-audit-fix-20261022`，
分支`codex/hv1-g60-audit-fix`。

修复后的hv1_g60_audit.py相对原件仅三行局部字典stat改名epoch_stat；
保留完整历史stat(path)保护循环。未发现其他局部stat赋值。
旧树的审计源码、配置、113项producer与旧FAIL日志全部保留。

独立入口hv1_g60_audit_repair.py显式使用旧树PYTHONPATH和原配置；
仅通过独立模块名载入修复后的审计函数。它核验修复文件与原文件的精确允许差异，
旧执行Commit、旧树HEAD、113项完整producer、修复入口/测试/监督器/报告入口SHA。
新配置hv1_g60_audit_repair.server.json另行绑定新旧版本，未修改旧来源或SHA校验。
旧协议不会把新代码冒充原producer。

原始审计失败日志及exit1永久保留在：
`/home/lzf/ldx/tmp/hv1_g60_adaptation_development_20261021/smoke_cpu_audit.log`及同目录退出码记录。
新审计日志放在`/home/lzf/ldx/tmp/hv1_g60_audit_repair_20261022/`。

修复审计先发布原设计中的新smoke_audit/training_audit回执，再发布独立
`audit_repair_v1/smoke`或`audit_repair_v1/training`版本绑定回执。
后者同时绑定新代码、原执行来源、原产物和原始FAIL证据；
正式监督器要求版本化Smoke审计通过，不能仅凭旧裸标志继续。

## CPU验证

本轮73个不同用例PASS：17项新增组合审计测试，24项H60原回归，32项Validation相关回归。
初次合并运行71通过、1失败；失败为旧Validation用例从继承工作树读取绝对路径配置时的KeyError。
该用例在原g60-h-validation工作树和原配置下单独执行PASS；
确认回归72通过、该原路径用例1通过。没有修改旧断言/配置或隐藏首次失败日志。

完整新审计流程Fixture使用真实Checkpoint publish/verify/restore和历史stat循环；
只以合成Linear替代最终CPU模型构造，不调用神经forward或CUDA。
覆盖历史循环末项篡改、文件状态函数调用、Checkpoint/日志/协议/producer冲突、
pending拒绝、Smoke/正式目录隔离、冻结证明异常与修复来源缺项拒绝。

测试日志、退出码在上述新临时目录。真实Smoke审计尚待独立执行；
测试通过不替代真实产物审计。

## 训练与结果发布

2048张G60图库和已有4-step Smoke只读复用，不重新G生成、图像编码或Smoke更新。
正式训练仍使用原冻结hv1_g60_adaptation入口、原配置和113项producer，
从原H10恢复20480 visits/20370 updates，不继承Smoke20484状态。
目标122880 visits/122220 updates；新增102400/101850；保持H-only、原SCST与Adam/RNG。
新监督器只替换独立CPU审计入口，原训练/Validation控制流未修改。

监督器：`src/tbps/ablations/run_hv1_g60_adaptation_repaired.sh`。
控制目录：`/home/lzf/ldx/tmp/hv1_g60_adaptation_epoch60_v1/`。
正式训练目录：`/home/lzf/ldx/outputs/tbps-framework-hv1-g60-adaptation-epoch60/training/`。
开始前必须新Smoke审计、原图库审计、父状态、来源、历史保护和资源全部PASS。
shared_observe_only门限仍为6144MiB、RAM32GiB、磁盘35GiB；不干预其他进程。

结束后调用原冻结报告生成器，报告专用包装器仅将文档ROOT指向修复树，
不改变任何指标或训练数学。派生来源JSON附新版本审计回执。
最终报告、六节点CSV/曲线提交到修复分支；旧Selection不覆盖。
正式训练/Validation/最终审计完成前，六节点新结果及最终PASS均为PENDING。

没有官方Test、RaSa/APTM、G10控制续训、新Loss或训练自动重试。
