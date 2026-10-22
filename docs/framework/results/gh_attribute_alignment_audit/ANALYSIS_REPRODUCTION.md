# 分析执行索引

本轮仅CPU；实际日志/退出码在/home/lzf/ldx/tmp/gh_attribute_alignment_audit_20261022/。
主统计为tbps-image Python环境、CUDA_VISIBLE_DEVICES为空、OMP/MKL线程2。
绘图使用已有apattack的Matplotlib环境，不安装/修改环境。

1. gh_attribute_audit.py：既有G/H来源、三个完整Train Epoch动作和Validation、严格匹配/K/回放。
2. gh_attribute_supplement.py：原候选覆盖、共同候选标量K分解、含2属性Spearman、冻结规则重提取。
3. plot_gh_attribute_audit.py：8图PNG/SVG。
4. report_gh_attribute_audit.py：中文报告、Schema与方案草案。
5. validate_gh_attribute_audit.py：输入SHA/历史保护/账本及真实trace示例检查。

这些程序只写新派生分析目录，已有输出拒绝覆盖；不可对原失败报告目录运行。
分析解析器首次输出的Validation theoretical K+1列已澄清为实际Optimizer更新0，并保留单独theoretical_Kplus1。
只改新派生CSV与分析器字段，原缓存/日志/Checkpoint从未修改；初次执行脚本SHA在来源索引另记。
不调用旧训练/评估/audit，不提供GPU或训练开关。

类别层级统计另由gh_attribute_category_overlap.py从既有CSV生成。先完整保留成员（含空属性）再聚合；最终26项测试。
