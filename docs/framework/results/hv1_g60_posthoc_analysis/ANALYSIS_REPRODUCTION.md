# 本轮分析代码与复现顺序

仅在新的独立工作树/输出目录执行CPU派生分析。不要指向原失败报告目录。
所有命令都要求CUDA_VISIBLE_DEVICES为空；没有GPU、训练或模型编码选项。

1. hv1_g60_posthoc.py：完整流式训练/Checkpoint来源核对、已有128分数逐Query统计。
2. hv1_g60_posthoc_diagnostics.py：只读资源与奖励/排名补充诊断。
3. plot_hv1_g60_posthoc.py：使用已有apattack CPU Matplotlib环境绘图。
4. report_hv1_g60_posthoc.py：根据派生数值生成中文报告。
5. validate_hv1_g60_posthoc.py：交付/历史保护核验及来源定稿。

主统计Python为tbps-image的3.10/Torch2.5.1+cu124，CPU线程2。
第3步使用已有apattack Python，不安装或修改训练环境。
源SHA及已关闭日志见SOURCE_EVIDENCE.json；图像清单见DELIVERABLE_MANIFEST.json。

本轮原始格式读取失败日志及修复后的退出码均在服务器
/home/lzf/ldx/tmp/hv1_g60_posthoc_analysis_20261022/保留。
不把派生分析器错误当成原实验错误，不修改旧训练/审计/模型代码。
